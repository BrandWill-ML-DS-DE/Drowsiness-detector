🚗 Real-Time Driver Drowsiness Detection

Hybrid Computer Vision Pipeline (EAR + CNN + Head Pose)

📌 Overview

This project implements a real-time driver drowsiness detection system using a hybrid computer vision pipeline that combines:

Geometric eye modeling (Eye Aspect Ratio – EAR)

Lightweight CNN-based eye state classification

Temporal fatigue modeling

Head pose compensation

Low-light robustness

Edge deployment via ONNX export

The system achieves real-time inference (<30ms per frame on consumer hardware) and is designed for robustness to:

Low-light driving conditions

Head pose variation

Partial eye occlusion (e.g., glasses)

🧠 System Architecture
Webcam Frame
     ↓
Preprocessing (Grayscale + Histogram Equalization)
     ↓
Mediapipe Face Mesh (Facial Landmarks)
     ↓
EAR Calculation  +  CNN Eye Classifier
     ↓
Head Pose Estimation (Dynamic Thresholding)
     ↓
Temporal Frame-Based Fatigue Modeling
     ↓
Drowsiness Alert Trigger

📂 Project Structure
drowsiness-detector/
│
├── app.py              # Real-time inference loop
├── ear.py              # EAR calculation logic
├── cnn_model.py        # CNN architecture
├── train_cnn.py        # Training script
├── evaluate.py         # Precision/Recall evaluation
├── head_pose.py        # Pose estimation
├── export_onnx.py      # Edge deployment export
├── requirements.txt
├── alarm.wav
└── README.md

🛠 Tech Stack

Python

OpenCV

Mediapipe Face Mesh

TensorFlow / Keras

FAISS (optional extensions)

NumPy

ONNX

🎯 Key Engineering Decisions
Problem	Solution
Lighting variability	Histogram equalization
Head tilt distortion	Dynamic EAR threshold
False positives	Temporal smoothing
Partial occlusion	CNN eye classifier
Edge deployment	ONNX export
