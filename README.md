# 🚗 Real-Time Driver Drowsiness Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Inference-005CED)](https://onnx.ai/)

A multi-stage computer vision pipeline designed to detect driver fatigue in real-time. By synthesizing **geometric heuristics (EAR)**, **deep learning (CNN)**, and **head-pose compensation**, the system achieves high reliability across variable environmental conditions and lighting scenarios.

---

## 🧠 System Engineering & Logic

A senior-level approach necessitates more than a simple classifier. This system employs a **hybrid ensemble** to minimize false positives and maximize edge-case stability:

### 1. Geometric Analysis (EAR)
Uses `scipy` to calculate the **Eye Aspect Ratio (EAR)**. This provides a computationally inexpensive method for rapid blink frequency detection and initial fatigue screening.



### 2. Deep Learning Validation (CNN)
A custom-built CNN validates the eye state (Open vs. Closed). This secondary check provides robustness against occlusions, such as glasses or heavy shadows, where raw geometric points might jitter.

### 3. Head-Pose Compensation
Using `solvePnP` to estimate head tilt (**Pitch/Yaw/Roll**), the system dynamically adjusts the EAR threshold. This prevents "false triggers" when a driver naturally looks down at the dashboard or checks mirrors.



---

## 🛠 Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Face Tracking** | MediaPipe Face Mesh | 468-point landmark detection for sub-pixel eye mapping. |
| **Model** | Custom CNN | Binary classification of eye-crop regions (Keras/TF). |
| **Optimization** | ONNX Runtime | Model converted for cross-platform, high-speed edge inference. |
| **I/O & UI** | OpenCV / Pygame | Real-time video processing and low-latency audio alerts. |

---

## 🚀 Key Engineering Decisions

* **Histogram Equalization:** Integrated within `utils.py` to handle "low-light" night driving by normalizing frame brightness before inference—crucial for real-world safety applications.
* **Temporal Smoothing:** Implemented a frame-counter buffer (`FRAME_THRESHOLD`) to distinguish between natural blinks and sustained "micro-sleep" events, filtering out high-frequency noise.
* **Deployment-Ready Workflow:** The inclusion of `export_onnx.py` demonstrates the ability to transition from a research environment (TensorFlow) to a production-grade inference engine (ONNX) for mobile or embedded hardware.

---

## 📊 Evaluation & Metrics

The system is evaluated via `evaluate.py` using a full classification report (**Precision, Recall, F1-score**). 

> **Focus:** In a safety-critical context, we optimize for **Recall on "Closed" states**, ensuring that the system fails on the side of caution while maintaining high enough Precision to avoid "alarm fatigue" for the driver.

---

## 🏁 Getting Started

### 1. Installation
```bash
git clone [https://github.com/your-username/drowsiness-detection.git](https://github.com/your-username/drowsiness-detection.git)
cd drowsiness-detection
pip install -r requirements.txt
```
### 2. Training (Optional)

If you wish to retrain the eye classifier:
```bash
python train_cnn.py
```
### 3. Run Inference

```bash
python app.py
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Accuracy | X |
   
---

## 📉 Future Roadmap

* **[ ] Yawn Detection:** Add landmark tracking for the mouth to detect early physiological signs of fatigue.
* **[ ] Infrared (NIR) Support:** Optimize the preprocessing pipeline for Near-Infrared camera inputs for total darkness operation.
* **[ ] INT8 Quantization:** Implement post-training quantization during ONNX export for ultra-low power consumption on microcontrollers.
