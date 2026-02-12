# utils.py
import cv2
import numpy as np

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def crop_eye(frame, landmarks, indices):
    h, w = frame.shape[:2]

    points = [(int(landmarks[i].x * w),
               int(landmarks[i].y * h)) for i in indices]

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    eye = frame[y_min:y_max, x_min:x_max]

    if eye.size == 0:
        return np.zeros((64,64), dtype=np.uint8)

    eye = cv2.resize(eye, (64,64))
    return eye

