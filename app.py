# app.py
import streamlit as st
import json
import os
from datetime import datetime
import subprocess
import re

from faster_whisper import WhisperModel
from diarization import diarize_interview
from nonverbal import analyze_audio_nonverbal
from eye_tracking import analyze_eye
from penilaian import penilaian_interview, rubric1, rubric2, rubric3, rubric4, rubric5, question1, question2, question3, question4, question5
import google.generativeai as genai

import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY belum diset!")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AI Powered Interview Assessment System",
    layout="wide",
    initial_sidebar_state="collapsed"
)
@st.cache_data(show_spinner=False)
def cached_penilaian(question, rubrik, answer):
    return penilaian_interview(
        question=question,
        rubrik=rubrik,
        answer=answer,
        model_ai=model
    )
# ==============================
# SESSION STATE
# ==============================
if "run_assessment" not in st.session_state:
    st.session_state.run_assessment = False

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "review_json" not in st.session_state:
    st.session_state.review_json = None



# ==============================
# GLOBAL STYLE
# ==============================
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# HERO (ALWAYS VISIBLE)
# ==============================
st.markdown('<div class="title">AI Powered Interview Assessment System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Automated analysis of speech, behavior, and visual attention</div>',
    unsafe_allow_html=True
)

# ==============================
# LOAD WHISPER (ONCE)
# ==============================
@st.cache_resource
def load_whisper():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

whisper = load_whisper()
VIDEO_DIR = "temp_media/videos"
AUDIO_DIR = "temp_media/audio"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)


# ======================================================
# PAGE 1 — UPLOAD (ONLY BEFORE ASSESSMENT)
# ======================================================
if not st.session_state.run_assessment:

    with st.container(border=True):
        uploaded_files = st.file_uploader(
            "Upload interview video files (.mp4 / .webm)",
            type=["mp4", "webm"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

            col_spacer, col_btn = st.columns([5, 1])

            with col_btn:
                if st.button("Nilai Interview", use_container_width=True):
                    st.session_state.run_assessment = True
                    st.rerun()
        else:
            st.info("Upload one or more interview video files to begin assessment.")

    st.stop()



# ======================================================
# PAGE 2 — RESULT
# ======================================================
if st.session_state.analysis_done:
    review_json = st.session_state.review_json
else:
    all_results = []
    all_rubrik =[rubric1, rubric2, rubric3, rubric4, rubric5]
    all_question =[question1, question2, question3, question4, question5]


    def extract_audio(video_path):
        base = os.path.splitext(os.path.basename(video_path))[0]
        wav_path = os.path.join(AUDIO_DIR, f"{base}.wav")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-map", "a:0?",
            "-ac", "1",
            "-ar", "16000",
            wav_path
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(wav_path):
            return None

        return wav_path



    review_json = {
        "assessorProfile": {
            "id": 47,
            "name": "AI Interview Assessor",
            "photoUrl": "xxx"
        },
        "decision": "Need Human",
        "reviewedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scoresOverview": {
            "project": 100,
            "interview": 0,
            "total": 0
        },
        "reviewChecklistResult": {
            "project": [],
            "interviews": {
                "minScore": 0,
                "maxScore": 4,
                "scores": []
            }
        },
        "notes": ""
    }

    def create_interview_score(
        question_id,
        score,
        transcript,
        justification,
        diar_result,
        eye_result,
        nonverbal_result
    ):
        return {
            "id": question_id,
            "score": score,
            "transcript": transcript.strip(),
            "aiAssessment": {
                "justification": justification,
                "cheatingSignals": {
                    "numSpeakers": diar_result["num_speakers"],
                    "diarizationConfidence": diar_result["confidence_score"],
                    "eyeTracking": {
                        "efsPercent": eye_result["efs_percent"],
                        "edpPercent": eye_result["edp_percent"],
                        "confidence": eye_result["final_confidence"],
                        "status": eye_result["status"]
                    }
                },
                "speechBehavior": {
                    "wordsPerMinute": nonverbal_result["wpm"],
                    "tempo": nonverbal_result["tempo_label"],
                    "totalSilenceSec": nonverbal_result["total_silence_sec"],
                    "fillerWordCount": nonverbal_result["filler_count"]
                }
            }
        }

    for idx, uploaded_file in enumerate(st.session_state.uploaded_files, start=0):
        # with st.container(border=True):
            match = re.search(r"question_(\d+)", uploaded_file.name, re.IGNORECASE)
            if match:
                question_number = int(match.group(1))
            else:
                question_number = idx + 1  
            with st.expander(f"Interview Question {question_number}", expanded=True):
                video_path = os.path.join(VIDEO_DIR, uploaded_file.name)

                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())

                # =========================
                # EXTRACT AUDIO
                # =========================
                audio_path = extract_audio(video_path)
                has_audio = audio_path is not None
                if not has_audio:
                    st.warning("No audio track detected — audio analysis skipped")
                    transcript = ""
                    segments_clean = []
                    info = None

                    diar_result = {
                        "num_speakers": 1,
                        "confidence_score": 0
                    }

                    nonverbal_result = {
                        "wpm": 0,
                        "tempo_label": "N/A",
                        "total_silence_sec": 0,
                        "filler_count": 0
                    }
                    # ==============================
                    # TRANSCRIPTION
                    # ==============================
                start = time.time()
                with st.spinner("Transcribing speech..."):
                        segments, info = whisper.transcribe(
                            audio_path,
                            beam_size=5,
                            word_timestamps=True,
                            language="en"
                        )

                        transcript = ""
                        segments_clean = []

                        for seg in segments:
                            transcript += seg.text.strip() + " "
                            segments_clean.append({
                                "start": float(seg.start),
                                "end": float(seg.end),
                                "text": seg.text.strip()
                            })

                with st.container(border=True):
                        st.markdown("#### Transcript")
                        with st.container(border=True):
                            st.markdown(transcript)


                    # ==============================
                    # DIARIZATION & EYE TRACKING
                    # ==============================
                with st.spinner("Running cheating detection..."):
                    diar_result = diarize_interview(
                            audio_path=audio_path,
                            segments=segments_clean
                        )

                    eye_result = analyze_eye(
                            video_path=video_path,
                            frame_skip=2,
                            max_frames=1000
                        )
                    # ==============================
                    # CHEATING DETECTION FRAME
                    # ==============================
                with st.container(border=True):

                    st.markdown("#### Cheating Detection")

                    col1, col2 = st.columns(2)

                        # ---------- DIARIZATION ----------
                    with col1:
                        with st.container(border=True):
                                st.markdown("##### Speaker Diarization")

                                st.markdown(f"- Number of Speakers: {diar_result['num_speakers']}")
                                st.markdown(f"- Diarization Confidence: {diar_result['confidence_score']}%")
                                
                    
                        # ---------- EYE TRACKING ----------
                    with col2:
                        with st.container(border=True):
                                st.markdown("##### Eye Tracking Analysis")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"- EFS percent: {eye_result['efs_percent']:.0f}%")
                                    st.markdown(f"- EDP percent: {eye_result['edp_percent']:.0f}%")
                                
                                with col2:
                                    st.markdown(f"- Confidence score: {eye_result['final_confidence']:.0f}%")
                                    st.markdown(f"- Status: {eye_result['status']}")
                            
                    # ==============================
                    # NON-VERBAL AUDIO
                    # ==============================
                    with st.spinner("Analyzing non-verbal audio..."):
                        nonverbal_result = analyze_audio_nonverbal(
                            wav_path=audio_path,
                            transcript=transcript,
                            whisper_info=info
                        )

                    with st.container(border=True):
                        st.markdown("#### Audio Non-Verbal Analysis")

                        col1, col2,col3,col4 = st.columns(4)

                        with col1:
                            st.markdown(f"- Words Per Minute: {nonverbal_result['wpm']}")
                        with col2:
                            st.markdown(f"- Speech Tempo: {nonverbal_result['tempo_label']}")
                        with col3:
                            st.markdown(f"- Total Silence: {nonverbal_result['total_silence_sec']}s")
                        with col4:
                            st.markdown(f"- Filler Word Count: {nonverbal_result['filler_count']}")
                    
                    with st.spinner("Evaluation ..."):      
                        nilai_result = cached_penilaian(answer=transcript, question=all_question[idx],rubrik=all_rubrik[idx])
                    with st.container(border=True):
                        st.markdown('#### Evaluation')
                        with st.container(border=True):
                            st.markdown(f"- Score: {nilai_result['score']}")
                            st.markdown(f"- reason: {nilai_result['justification']}")
                    
                    with st.container(border=True):
                            st.markdown('Time Statictic ')
                            st.markdown(f"- runtime: {(time.time()-start):.2f} s")
                            st.markdown(f"- video duration: {(info.duration):.2f} s")
                        
                # ==============================
                # SAVE RESULT
                # ==============================
                score_item = create_interview_score(
                    question_id=idx + 1,
                    score=nilai_result["score"],
                    transcript=transcript,
                    justification=nilai_result["justification"],
                    diar_result=diar_result,
                    eye_result=eye_result,
                    nonverbal_result=nonverbal_result
                )
                review_json["reviewChecklistResult"]["interviews"]["scores"].append(score_item)

    scores = review_json["reviewChecklistResult"]["interviews"]["scores"]
    total_interview_score = sum(item["score"]*5 for item in scores)
    review_json["scoresOverview"]["interview"] = total_interview_score

    project_score = review_json["scoresOverview"]["project"]
    review_json["scoresOverview"]["total"] = round((project_score + total_interview_score) / 2, 2)

    if any(
        s["aiAssessment"]["cheatingSignals"]["eyeTracking"]["status"] == "Need review"
        for s in scores
    ):
        review_json["decision"] = "Need Human"
        review_json["notes"] = "Eye tracking or behavioral signal requires manual review"
    else:
        review_json["decision"] = "PASSED"
        review_json["notes"] = "Lancar tidak mencurigakan"

    with st.container(border=True):
        interview_score = review_json["scoresOverview"]["interview"]
        decision = review_json["decision"]
        notes = review_json["notes"]
        st.markdown('#### Summary Result')
        with st.container(border=True):
            st.metric(label= "Interview score",
                    value=f"{interview_score}")
            st.markdown(f"- Decision: {decision}")
            st.markdown(f"- Notes: {notes}")
        
        # ==============================
        # SIMPAN HASIL (INI KUNCI UTAMA)
        # ==============================
        st.session_state.review_json = review_json
        st.session_state.analysis_done = True


# ==============================
# DOWNLOAD
# ==============================
st.divider()

# Tombol download (ATAS)
st.download_button(
    "Download Review Result (JSON)",
    data=json.dumps(st.session_state.review_json, indent=2),
    file_name="review_result.json",
    mime="application/json"
)

# Jarak kecil biar rapi
st.write("")

# Tombol upload ulang (BAWAH)
if st.button("Back"):
    st.session_state.run_assessment = False
    st.session_state.uploaded_files = None
    st.session_state.analysis_done = False
    st.session_state.review_json = None
    st.rerun()

