# nonverbal.py
import librosa
import numpy as np
import re

def analyze_audio_nonverbal(wav_path: str, transcript: str, whisper_info):
    y, sr = librosa.load(wav_path, sr=16000)
    results = {}

    # ======================
    # WPM
    # ======================
    words = transcript.split()
    word_count = len(words)
    duration_minutes = whisper_info.duration / 60 if getattr(whisper_info, "duration", 0) > 0 else 1
    wpm = word_count / duration_minutes

    tempo_label = "Normal"
    if wpm < 80:
        tempo_label = "Slow"
    elif wpm > 160:
        tempo_label = "Fast"

    results["wpm"] = round(wpm, 2)
    results["tempo_label"] = tempo_label

    # ======================
    # SILENCE
    # ======================
    intervals = librosa.effects.split(y, top_db=35)
    non_silent = sum((e - s) / sr for s, e in intervals)
    total = getattr(whisper_info, "duration", 0.0)
    silence = max(0.0, total - non_silent)

    pause_label = "Normal"
    if total > 0 and silence > total * 0.4:
        pause_label = "Many pauses"

    results["total_silence_sec"] = round(silence, 2)
    results["pause_label"] = pause_label

    # ======================
    # VOLUME VARIANCE
    # ======================
    rms = librosa.feature.rms(y=y)[0] if y.size > 0 else np.array([0.0])
    rms_std = float(np.std(rms))

    volume_label = "Stable"
    if rms_std > 0.08:
        volume_label = "Unstable"

    results["volume_variance"] = rms_std
    results["volume_label"] = volume_label

    # ======================
    # FILLER WORDS
    # ======================
    fillers = r"\b(uh|um|umm|like|you know|actually)\b"
    count = len(re.findall(fillers, transcript.lower()))

    filler_label = "Controlled"
    if count > 5:
        filler_label = "Many fillers"

    results["filler_count"] = count
    results["filler_label"] = filler_label

    return results
