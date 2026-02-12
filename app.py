# app.py
import cv2
import time
import logging
import mediapipe as mp
import numpy as np
import pygame
from tensorflow.keras.models import load_model

from ear import calculate_ear
from utils import preprocess_frame, crop_eye
from head_pose import get_head_pose

logging.basicConfig(filename="performance.log", level=logging.INFO)

model = load_model("eye_cnn.h5")

pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20
counter = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    gray = preprocess_frame(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            eye_points = [(int(face_landmarks.landmark[i].x * w),
                           int(face_landmarks.landmark[i].y * h))
                          for i in LEFT_EYE]

            ear = calculate_ear(eye_points)

            pitch, yaw = get_head_pose(face_landmarks.landmark, frame)
            dynamic_threshold = EAR_THRESHOLD - abs(pitch)*0.002

            eye_crop = crop_eye(gray, face_landmarks.landmark, LEFT_EYE)
            eye_crop = eye_crop.reshape(1,64,64,1) / 255.0

            prediction = model.predict(eye_crop, verbose=0)

            if ear < dynamic_threshold or prediction[0][0] > 0.5:
                counter += 1
            else:
                counter = 0

            if counter >= FRAME_THRESHOLD:
                cv2.putText(frame, "DROWSINESS ALERT!",
                            (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0,0,255), 3)
                pygame.mixer.music.play()

    latency = (time.time() - start_time) * 1000
    fps = 1 / (time.time() - start_time)

    logging.info(f"Latency(ms): {latency:.2f}, FPS: {fps:.2f}")

    cv2.putText(frame, f"FPS: {fps:.2f}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,255,0), 2)

    cv2.imshow("Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

