# diarization.py
import numpy as np
import librosa
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================
# EMBEDDING (STABLE FOR INTERVIEW)
# =============================================================
def extract_embedding(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=40,
        n_fft=1024,
        hop_length=256
    )
    logmel = librosa.power_to_db(mel)

    return np.concatenate([
        np.mean(logmel, axis=1),
        np.std(logmel, axis=1)
    ])


def frame_embeddings(audio, sr, frame_sec=1.2):
    frame_len = int(frame_sec * sr)
    embeddings, frames = [], []

    for start in range(0, len(audio), frame_len):
        end = min(start + frame_len, len(audio))
        y_frame = audio[start:end]

        if len(y_frame) < sr * 0.5:
            continue

        embeddings.append(extract_embedding(y_frame, sr))
        frames.append((start / sr, end / sr))

    return np.array(embeddings), frames


# =============================================================
# INTERVIEW-AWARE DIARIZATION
# =============================================================
def diarize_interview(audio_path: str, segments: list):
    # start_time = time.time()

    y, sr = librosa.load(audio_path, sr=16000)
    embeddings, frames = frame_embeddings(y, sr)

    if len(embeddings) == 0:
        return {
            "segments": segments,
            "num_speakers": 1,
            "avg_similarity": 1.0,
            "dominant_ratio": 1.0,
            "confidence_score": 100,
            "runtime_sec": 0.0
        }

    sim = cosine_similarity(embeddings)
    avg_sim = float(np.mean(sim))

    # ================= DECISION LOGIC =================
    if avg_sim > 0.92:
        labels = np.zeros(len(embeddings), dtype=int)
        n_speaker = 1

    elif avg_sim > 0.80:
        clusterer = HDBSCAN(min_cluster_size=6, min_samples=3)
        labels = clusterer.fit_predict(embeddings)
        n_speaker = min(len(set(labels) - {-1}), 2)

        counts = [np.sum(labels == c) for c in set(labels) if c != -1]
        if counts and max(counts) / sum(counts) > 0.85:
            labels[:] = 0
            n_speaker = 1
    else:
        clusterer = HDBSCAN(min_cluster_size=4, min_samples=2)
        labels = clusterer.fit_predict(embeddings)
        n_speaker = min(len(set(labels) - {-1}), 2)

    # ================= ALIGN TO TRANSCRIPT =================
    for seg in segments:
        s = int(seg["start"] / 1.2)
        e = int(seg["end"] / 1.2) + 1
        seg_labels = labels[s:e] if len(labels) else []

        valid = seg_labels[seg_labels >= 0] if len(seg_labels) else []
        seg["speaker"] = int(np.bincount(valid).argmax()) if len(valid) else 0

    # ================= CONFIDENCE SCORE =================
    if n_speaker == 1:
        dominant_ratio = 1.0
    else:
        counts = [np.sum(labels == c) for c in set(labels) if c != -1]
        dominant_ratio = max(counts) / sum(counts) if counts else 1.0

    sim_score = np.clip((avg_sim - 0.6) / 0.4, 0, 1)
    speaker_penalty = 0.15 if n_speaker > 1 else 0.0

    confidence = int(np.clip(
        (0.5 * sim_score + 0.5 * dominant_ratio - speaker_penalty) * 100,
        0, 100
    ))

    return {
        "segments": segments,
        "num_speakers": n_speaker,
        "avg_similarity": round(avg_sim, 4),
        "dominant_ratio": round(dominant_ratio, 3),
        "confidence_score": confidence,
        # "runtime_sec": round(time.time() - start_time, 2)
    }



