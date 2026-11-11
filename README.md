# Real-Time Driver Fatigue and Distraction Detection System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange.svg)](https://developers.google.com/mediapipe)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end, real-time computer vision and machine learning system engineered to detect driver fatigue and distraction. The system analyzes live video streams, extracts facial landmarks using Google MediaPipe Face Mesh, computes physiological ratios (Eye Aspect Ratio and Mouth Aspect Ratio), estimates 3D head pose orientation (Pitch, Yaw, Roll), and classifies driver behavior into three states:

- **`ALERT`**: Attentive and actively looking at the road.
- **`DROWSY`**: Prolonged eye closure, micro-sleeps, or frequent yawning.
- **`DISTRACTED`**: Head turned away from the driving corridor (looking left, right, down, or up).

When unsafe conditions persist, non-blocking auditory alerts are dispatched to prevent road accidents.

---

## 1. Technology Stack

| Component | Technology | Version / Specification |
|---|---|---|
| **Programming Language** | Python | 3.11 |
| **Video Processing** | OpenCV (`opencv-python`) | >= 4.8.0.76 |
| **Landmark Detection** | Google MediaPipe (`mediapipe`) | >= 0.10.9 (Face Mesh 468/478 points) |
| **Numerical Processing** | NumPy | >= 1.24.0, < 2.0.0 |
| **Data Handling** | Pandas | >= 2.0.0 |
| **Machine Learning** | Scikit-learn (`scikit-learn`), Joblib | >= 1.3.0 |
| **Audio Notification** | Pygame (`pygame`) | >= 2.5.0 (Pygame Mixer) |
| **Web Dashboard** | Streamlit (`streamlit`) | >= 1.28.0 |
| **Testing** | Pytest / Unittest | >= 7.4.0 |
| **Configuration** | Python-dotenv | >= 1.0.0 |

---

## 2. Complete Project Directory Structure

```
driver-fatigue-detection/
├── main.py                     # Main application entry point for desktop OpenCV runtime
├── requirements.txt            # Python 3.11 dependencies
├── README.md                   # Project documentation and developer guide
├── .gitignore                  # Git ignore rules for venv, cache, datasets, and secrets
├── .env.example                # Template for environment variable configurations
│
├── config/                     # Centralized application configuration
│   ├── __init__.py
│   └── settings.py             # Typed dataclasses for thresholds, camera, and model paths
│
├── detection/                  # Computer vision and landmark extraction layer
│   ├── __init__.py
│   ├── video_capture.py        # VideoCamera hardware abstraction and frame capture
│   ├── face_detection.py       # MediaPipe Face Mesh 468/478 landmark detection
│   ├── eye_detection.py        # Bilateral EAR calculation and eye closure detection
│   ├── mouth_detection.py      # MAR calculation and yawn detection
│   └── head_pose.py            # 3D-to-2D solvePnP head orientation (Pitch, Yaw, Roll)
│
├── features/                   # Feature aggregation and data contracts
│   ├── __init__.py
│   └── feature_extraction.py   # DriverFeatures dataclass and pipeline aggregator
│
├── models/                     # Machine learning training and online inference
│   ├── __init__.py
│   ├── train_model.py          # Model training, validation, and serialization pipeline
│   ├── predict.py              # Real-time state predictor using serialized models
│   └── saved_models/           # Serialized model weights (.joblib, .gitkeep)
│
├── decision_engine/            # Hybrid decision logic
│   ├── __init__.py
│   ├── threshold_rules.py      # Sliding temporal counters (blinks vs micro-sleeps)
│   └── hybrid_classifier.py    # Conflict resolution fusing rules and ML inference
│
├── alerts/                     # Auditory warnings and notifications
│   ├── __init__.py
│   ├── audio_alert.py          # Asynchronous Pygame alert manager with cooldowns
│   └── sounds/                 # Audio alert assets (.wav, .gitkeep)
│
├── visualization/              # Graphical HUD and visual overlays
│   ├── __init__.py
│   └── draw_utils.py           # Landmark contours, 3D axes, and status banner drawing
│
├── dashboard/                  # Streamlit web frontend
│   ├── __init__.py
│   └── app.py                  # Streamlit interactive monitoring dashboard
│
├── dataset/                    # Feature store and offline training data
│   ├── raw/                    # Raw recordings and frame sequences (.gitkeep)
│   ├── processed/              # Processed and normalized datasets (.gitkeep)
│   └── driver_features.csv     # Schema-ready CSV for feature logging
│
├── tests/                      # Automated unit and integration test suite
│   ├── __init__.py
│   ├── test_video_capture.py   # Tests for camera abstraction contracts
│   ├── test_features.py        # Tests for DriverFeatures conversions and logic
│   └── test_classifier.py     # Tests for decision engine states and counters
│
├── screenshots/                # Visual documentation and execution captures (.gitkeep)
│
└── docs/                       # Technical architecture documentation
    └── architecture.md         # Full architectural specification and diagrams
```

---

## 3. Module Responsibilities and Architectural Flow

### Module Breakdown
- **`config/settings.py`**: Centralized single source of truth for all system hyperparameters (EAR threshold, MAR threshold, head pose angle limits, alert cooldowns, and input camera indices).
- **`detection/`**:
  - `video_capture.py`: Context-managed webcam stream handler.
  - `face_detection.py`: MediaPipe Face Mesh detector extracting 3D facial coordinates.
  - `eye_detection.py`: Computes Eye Aspect Ratio (EAR) based on Soukupová & Čech (2016).
  - `mouth_detection.py`: Computes Mouth Aspect Ratio (MAR) to monitor continuous yawning.
  - `head_pose.py`: Uses OpenCV `cv2.solvePnP` with standard 3D facial models to compute head Euler angles.
- **`features/feature_extraction.py`**: Packages computed metrics into the typed `DriverFeatures` dataclass contract.
- **`models/`**:
  - `train_model.py`: Offline dataset training and model evaluation (`RandomForest`, `SVM`).
  - `predict.py`: Low-latency online inference generating driver state probabilities.
- **`decision_engine/`**:
  - `threshold_rules.py`: Sliding temporal window counters to differentiate natural blinks from micro-sleeps.
  - `hybrid_classifier.py`: Fuses deterministic safety rules with ML predictions (safety overrides ensure sustained eye closure immediately triggers `DROWSY`).
- **`alerts/audio_alert.py`**: Non-blocking audio dispatcher with cooldown timing to prevent sound repetition.
- **`visualization/draw_utils.py`**: Renders HUD overlays, 3D pose vectors, and color-coded status banners.
- **`dashboard/app.py`**: Streamlit dashboard decoupling UI from detection logic.

---

## 4. Team Responsibility Allocation (3 Developers)

This architecture is optimized for parallel team development across three dedicated engineering streams:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        3-DEVELOPER TEAM SPREAD                         │
├─────────────────────┬──────────────────────────┬───────────────────────┤
│ Developer 1         │ Developer 2              │ Developer 3           │
│ Vision & Hardware   │ ML & Decision Engine     │ UI, Alerts & Pipeline │
├─────────────────────┼──────────────────────────┼───────────────────────┤
│ • video_capture.py  │ • feature_extraction.py  │ • settings.py         │
│ • face_detection.py │ • train_model.py         │ • audio_alert.py      │
│ • eye_detection.py  │ • predict.py             │ • draw_utils.py       │
│ • mouth_detection.py│ • threshold_rules.py     │ • dashboard/app.py    │
│ • head_pose.py      │ • hybrid_classifier.py   │ • main.py             │
│ • test_video_capture│ • test_features/classifier│ • full integration   │
│ Branch:             │ Branch:                  │ Branch:               │
│ feat/vision-pipeline│ feat/ml-decision-engine  │ feat/ui-alerts-integ  │
└─────────────────────┴──────────────────────────┴───────────────────────┘
```

---

## 5. Phased Development Roadmap

The project is structured for execution across 7 distinct phases:

- **Phase 1: Video Capture & Hardware Abstraction** (`detection/video_capture.py`)
- **Phase 2: MediaPipe Face Landmark Detection** (`detection/face_detection.py`)
- **Phase 3: Mathematical Feature Extraction (EAR, MAR, Head Pose)** (`detection/eye_detection.py`, `detection/mouth_detection.py`, `detection/head_pose.py`, `features/feature_extraction.py`)
- **Phase 4: Hybrid Decision Engine & ML Classification** (`decision_engine/`, `models/`)
- **Phase 5: Audio Alert System Integration** (`alerts/audio_alert.py`)
- **Phase 6: Interactive Streamlit Dashboard** (`dashboard/app.py`)
- **Phase 7: End-to-End Testing, Latency Optimization & Final Release** (`tests/`, `main.py`)

---

## 6. Installation & Environment Setup

### 6.1 Prerequisites
- **Python 3.11** installed on your system.
- Functional webcam or video input device.
- Standard audio output device (speakers or headphones).

### 6.2 Setup Virtual Environment
```bash
# 1. Clone repository
git clone https://github.com/<your-org>/driver-fatigue-detection.git
cd driver-fatigue-detection

# 2. Create isolated Python 3.11 virtual environment
python3.11 -m venv venv

# 3. Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install project dependencies
pip install -r requirements.txt

# 6. Initialize local environment variables
cp .env.example .env
```

---

## 7. Git Workflow & Collaboration Rules

### 7.1 Git Repository Initialization
```bash
# Initialize git repository
git init
git add .
git commit -m "chore: scaffold modular MVP architecture skeleton for driver fatigue system"

# Set up main branch
git branch -M main
```

### 7.2 Branching Model
All developers create feature branches branching from `main`:
```bash
# Developer 1:
git checkout -b feat/vision-pipeline

# Developer 2:
git checkout -b feat/ml-decision-engine

# Developer 3:
git checkout -b feat/ui-alerts-integration
```

---

## 8. Running and Testing the Application

### 8.1 Verification Test Suite
Verify that all interfaces, data models, and architectural contracts pass:
```bash
python -m unittest discover tests
```

### 8.2 Execution (Post-Implementation)
- **Desktop OpenCV Application**:
  ```bash
  python main.py
  ```
- **Streamlit Web Dashboard**:
  ```bash
  streamlit run dashboard/app.py
  ```

---

## 9. Architectural Integrity & Missing Dependencies

### Verification Checklist:
- [x] Strict separation between detection engine and user interface.
- [x] Zero circular dependencies between packages.
- [x] Strongly typed data transfer contracts (`DriverFeatures`, `DriverState`).
- [x] Decoupled asynchronous audio system with cooldown prevention.
- [x] Comprehensive test scaffolding.

### External Asset Requirements (Identified for Phase 5):
- Alert sound files: Add `drowsy_alert.wav` and `distracted_alert.wav` into `alerts/sounds/`.
- Training data: Feature logging in Phase 4 will populate `dataset/driver_features.csv`.

# driver-fatigue-
