# eye_tracking.py
import cv2
import mediapipe as mp
import numpy as np
import math

mp_face = mp.solutions.face_mesh

LEFT_EYE_LANDMARKS = [33, 133, 159, 145]
LEFT_IRIS_LANDMARKS = [469, 470, 471, 472]


# ============================================================
# FOCUS CHECK
# ============================================================
def is_focused(x_ratio, y_ratio):
    return 0.35 < x_ratio < 0.65 and 0.35 < y_ratio < 0.65


# ============================================================
# EYE DIRECTION
# ============================================================
def get_eye_direction(landmarks, frame_w, frame_h, glass_mode=False):
    def px(p):
        return int(p.x * frame_w), int(p.y * frame_h)

    Lx, _ = px(landmarks[33])
    Rx, _ = px(landmarks[133])
    _, Ty = px(landmarks[159])
    _, By = px(landmarks[145])

    eye_width = abs(Rx - Lx)
    eye_height = abs(By - Ty)

    if eye_width < 5 or eye_height < 5:
        return "CENTER", 0.5, 0.5

    iris_pts = np.array([px(landmarks[i]) for i in LEFT_IRIS_LANDMARKS])
    pupil_x, pupil_y = np.mean(iris_pts, axis=0)

    x_ratio = (pupil_x - Lx) / eye_width
    y_ratio = (pupil_y - Ty) / eye_height

    if glass_mode:
        LEFT_TH, RIGHT_TH, UP_TH, DOWN_TH = 0.38, 0.62, 0.30, 0.62
    else:
        LEFT_TH, RIGHT_TH, UP_TH, DOWN_TH = 0.42, 0.58, 0.42, 0.58

    if x_ratio < LEFT_TH:
        return "LEFT", x_ratio, y_ratio
    if x_ratio > RIGHT_TH:
        return "RIGHT", x_ratio, y_ratio
    if y_ratio < UP_TH:
        return "UP", x_ratio, y_ratio
    if y_ratio > DOWN_TH:
        return "DOWN", x_ratio, y_ratio

    return "CENTER", x_ratio, y_ratio


# ============================================================
# HEAD POSE
# ============================================================
def get_head_direction(landmarks, w, h):
    idx = [1, 152, 33, 263, 61, 291]

    model = np.array([
        [0, 0, 0],
        [0, -100, -30],
        [-60, 40, -60],
        [60, 40, -60],
        [-40, -50, -50],
        [40, -50, -50]
    ], dtype=np.float64)

    image = np.array([
        [landmarks[i].x * w, landmarks[i].y * h] for i in idx
    ], dtype=np.float64)

    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)

    _, rvec, _ = cv2.solvePnP(model, image, cam, np.zeros((4, 1)))
    R, _ = cv2.Rodrigues(rvec)

    yaw = math.degrees(math.atan2(R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)))
    pitch = math.degrees(math.atan2(-R[2, 1], R[2, 2]))

    if yaw > 15:
        return "HEAD_RIGHT"
    if yaw < -15:
        return "HEAD_LEFT"
    if pitch > 12:
        return "HEAD_DOWN"
    if pitch < -12:
        return "HEAD_UP"
    return "HEAD_CENTER"


# ============================================================
# MAIN FUNCTION (STREAMLIT READY)
# ============================================================
def analyze_eye(video_path, frame_skip=2, max_frames=1000):

    cap = cv2.VideoCapture(video_path)

    counts_eye = {k: 0 for k in ["LEFT", "RIGHT", "UP", "DOWN", "CENTER"]}
    focused_count = 0
    not_focused_count = 0
    eye_history = []
    brightness = []

    glass_mode = False
    frame_count = 0

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if not res.multi_face_landmarks:
                continue

            lm = res.multi_face_landmarks[0].landmark

            # Glass detection (brightness)
            Lx = int(lm[33].x * w)
            Rx = int(lm[133].x * w)
            Ty = int(lm[159].y * h)
            By = int(lm[145].y * h)

            roi = frame[Ty:By, Lx:Rx]
            if roi.size > 0:
                brightness.append(np.mean(roi))
                if len(brightness) > 30 and np.mean(brightness[-30:]) > 170:
                    glass_mode = True

            eye_dir, xr, yr = get_eye_direction(lm, w, h, glass_mode)
            counts_eye[eye_dir] += 1
            eye_history.append(eye_dir)

            if is_focused(xr, yr):
                focused_count += 1
            else:
                not_focused_count += 1

    cap.release()

    total_frames = focused_count + not_focused_count
    if total_frames == 0:
        return {
            "efs_percent": 0,
            "edp_percent": 0,
            "tss_percent": 0,
            "final_confidence": 0,
            "status": "NO FACE DETECTED"
        }

    # ==============================
    # CONFIDENCE CALCULATION
    # ==============================
    EFS = focused_count / total_frames

    eye_weights = {
        "LEFT": 0.7,
        "RIGHT": 0.7,
        "UP": 0.25,
        "DOWN": 0.25,
        "CENTER": 0
    }
    
    efs_percent = round(EFS * 100, 2)

    penalty_eye = sum(eye_weights[k] * v for k, v in counts_eye.items())
    EDP = max(0, 1 - penalty_eye / total_frames)
    
    edp_percent = round(EDP * 100, 2)

    changes = sum(
        1 for i in range(1, len(eye_history))
        if eye_history[i] != eye_history[i - 1]
    )

    TSS = max(0.1, 1 - changes / total_frames)

    confidence = round((0.50 * EFS + 0.25 * EDP + 0.20 * TSS) * 100, 2)

    if confidence >= 75:
        status = "Valid"
    elif confidence >= 50:
        status = "Need review"
    else:
        status = "Invalid"

    return {
        "efs_percent" : efs_percent,
        "edp_percent": edp_percent,
        "final_confidence": confidence,
        "status": status
    }
