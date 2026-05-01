# Real-Time Facial Landmark Detection (MediaPipe)

A real-time facial landmark detection system using MediaPipe Face Mesh.
The system extracts 468 landmarks, groups them into facial regions, and renders stable contours using temporal smoothing.

---

## 🚀 Features

* Real-time webcam inference
* 468 facial landmarks per face
* Region-based visualization (eyes, lips, nose, etc.)
* Temporal smoothing to reduce jitter
* Modular architecture
* NumPy-based data pipeline (future ML ready)

---

## 📂 Project Structure

```
.
├── main.py           # Main pipeline (capture → detect → smooth → draw)
├── face_mesh.py      # Visualization logic
├── smooth.py         # Temporal smoothing
├── face_landmarker.task   # MediaPipe model file
└── README.md
```

---

## 🧠 Pipeline Overview

```
Camera → OpenCV → MediaPipe → NumPy (F,468,2)
→ Smoothing → Region grouping → Drawing → Display
```

---

## ⚙️ Requirements

* Python 3.8+
* OpenCV
* MediaPipe
* NumPy

---

## 📦 Installation

Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / Mac
.venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

---

## ▶️ Run the Project

Make sure `face_landmarker.task` is in the project directory.

Then run:

```bash
python main.py
```

Press **q** to exit.

---

## ⚠️ Notes

* Webcam must be available and not used by another application
* `num_faces` in code controls maximum faces detected
* Performance depends mainly on MediaPipe inference


## 🛠️ Troubleshooting

**Camera not opening**

* Close other apps using webcam
* Try changing camera index in `cv.VideoCapture(0)`

**Low FPS**

* Reduce frame size
* Lower `num_faces`