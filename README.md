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

## 4. Team Responsibility Allocation (4 Developers)

This architecture is optimized for parallel team development across four dedicated engineering streams:

| Developer | Responsibility | Main Files | Git Branch |
|---|---|---|---|
| **Developer 1** | Computer Vision & Camera | `video_capture.py`, `face_detection.py`, `eye_detection.py`, `mouth_detection.py`, `head_pose.py`, `test_video_capture.py`, `test_face_detection.py` | `feat/computer-vision` |
| **Developer 2** | Feature Extraction & Machine Learning | `feature_extraction.py`, `train_model.py`, `predict.py`, `driver_features.csv`, `test_features.py` | `feat/feature-extraction-ml` |
| **Developer 3** | Decision Engine & Model Evaluation | `threshold_rules.py`, `hybrid_classifier.py`, ML evaluation and tests, `test_classifier.py` | `feat/decision-engine` |
| **Developer 4** | UI, Audio Alerts & Integration | `audio_alert.py`, `draw_utils.py`, `dashboard/app.py`, `main.py`, `settings.py`, integration tests | `feat/ui-alerts-integration` |

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     4-DEVELOPER TEAM SPREAD                                      │
├─────────────────────┬──────────────────────────┬─────────────────────────┬───────────────────────┤
│ Developer 1         │ Developer 2              │ Developer 3             │ Developer 4           │
│ Computer Vision &   │ Feature Extraction &     │ Decision Engine &       │ UI, Audio Alerts &    │
│ Camera              │ Machine Learning         │ Model Evaluation        │ System Integration    │
├─────────────────────┼──────────────────────────┼─────────────────────────┼───────────────────────┤
│ • video_capture.py  │ • feature_extraction.py  │ • threshold_rules.py    │ • audio_alert.py      │
│ • face_detection.py │ • train_model.py         │ • hybrid_classifier.py  │ • draw_utils.py       │
│ • eye_detection.py  │ • predict.py             │ • ML evaluation & tests │ • dashboard/app.py    │
│ • mouth_detection.py│ • driver_features.csv    │ • test_classifier.py    │ • main.py             │
│ • head_pose.py      │ • test_features.py       │ • temporal counters     │ • settings.py         │
│ • test_video_capture│                          │                         │ • integration tests   │
│ • test_face_detect  │                          │                         │                       │
│ Branch:             │ Branch:                  │ Branch:                 │ Branch:               │
│ feat/computer-vision│ feat/feature-extraction-ml│ feat/decision-engine   │ feat/ui-alerts-integ  │
└─────────────────────┴──────────────────────────┴─────────────────────────┴───────────────────────┘
```

---

## 5. Phased Development Roadmap

The project is structured for execution across 7 distinct phases:

- [x] **Phase 1: Video Capture & Hardware Abstraction** (`detection/video_capture.py`) - *Completed*
  - OpenCV camera initialization, resolution auto-negotiation, dynamic FPS calculation, screenshot capture, horizontal mirroring preview with original sensor frame preservation, and hardware resource cleanup.
- [x] **Phase 2: MediaPipe Facial Landmark Detection** (`detection/face_detection.py`) - *Completed*
  - Google MediaPipe Face Landmarker Tasks API (`RunningMode.VIDEO`), 468/478 normalized 3D landmarks, eye and mouth contour isolation, iris detection, monotonic timestamp tracking, missing-face warning banner, and mirrored preview integration.
- [x] **Phase 3: Mathematical Feature Extraction (EAR, MAR, Head Pose)** (`detection/eye_detection.py`, `detection/mouth_detection.py`, `detection/head_pose.py`, `features/feature_extraction.py`) — *Completed*
  - **Eye Aspect Ratio (EAR)**: Soukupová & Čech 6-point formula with bilateral averaging and division-by-zero epsilon guard (`EyeDetector`, `EyeDetectionResult`).
  - **Mouth Aspect Ratio (MAR)**: Vertical/horizontal ratio supporting 4-point and 6-point geometry; yawning threshold helper (`MouthDetector`, `MouthDetectionResult`).
  - **Head Pose Estimation**: `cv2.solvePnP` (ITERATIVE) + `cv2.RQDecomp3x3` → Pitch / Yaw / Roll in degrees; resolution-independent camera matrix; angles normalized to [−180°, +180°] (`HeadPoseEstimator`, `HeadPoseResult`).
  - **FeatureExtractor**: Aggregates EAR, MAR, and head-pose angles into `DriverFeatures` dataclass; `to_dict()` and `to_array(feature_order)` conversions for ML pipeline.
  - **Telemetry HUD**: Semi-transparent HUD card and RGB head-pose axis overlay rendered on live mirrored preview (`visualization/draw_utils.py`).
- [x] **Phase 4: Hybrid Decision Engine & ML Classification** (`decision_engine/`, `models/`) — *Completed*
  - **Rule-Based Detection**: `TemporalRuleEngine` tracks consecutive frame counters for EAR (< 0.25), MAR (> 0.60), and head pose angles (pitch, yaw, roll) with monotonic timing, recovery smoothing, and missing-face safety handling.
  - **Machine Learning Classification**: `ModelTrainer` and `StatePredictor` using scikit-learn (`RandomForest`, `SVM`, `GradientBoosting`); train/test split, accuracy, precision, recall, F1, confusion matrix, and joblib serialization.
  - **Dataset Schema & Synthetic Testing Generator**: Standardized feature schema `[ear, mar, pitch, yaw, roll, state]` documented in `dataset/schema.md` with optional synthetic testing generator strictly for CI/CD test verification.
  - **Hybrid Decision Engine**: `HybridDecisionEngine` fuses deterministic safety override rules with statistical ML probabilities and sliding-window temporal smoothing.
  - **Live UI & Telemetry**: Dynamic state warning banner, comprehensive HUD card displaying FPS, Face detection status, EAR, MAR, Head Pose, State, ML prediction, and Hybrid classification result.
- [x] **Phase 5: Audio Alert System Integration** (`alerts/audio_alert.py`, `alerts/sounds/`, `main.py`) — *Completed*
  - **Non-Blocking Audio Dispatch**: `AlertManager` powered by `pygame.mixer` preloads `.wav` warning clips at startup (`initialize()`) and triggers asynchronous playback for `DROWSY` (`drowsy_alert.wav`) and `DISTRACTED` (`distracted_alert.wav`) without blocking the webcam loop.
  - **Cooldown & State Transition Control**: Monotonic cooldown timer (`cooldown_seconds = 3.0s`) prevents repetitive frame-by-frame audio spam; returning to `ALERT` immediately silences active warning channels (`stop_all()`).
  - **Mute Controls & HUD Badge**: Keyboard shortcut `m` / `M` toggles audio mute/unmute live; HUD displays `AUDIO: ON`, `AUDIO: MUTED`, or `AUDIO: UNAVAILABLE` via `Visualizer.draw_audio_status()`.
  - **Graceful Degradation**: Handles missing `pygame`, missing audio hardware, corrupt/missing `.wav` files (without silent substitution), and playback errors safely without crashing the video pipeline.
- [ ] **Phase 6: Interactive Streamlit Dashboard** (`dashboard/app.py`)
- [ ] **Phase 7: End-to-End Testing, Latency Optimization & Final Release** (`tests/`, `main.py`)

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

### 6.3 Dependency & Audio Installation Notes
- **Recommended Environment**: Python 3.11 (`python3.11 -m venv venv`).
- **Core vs Optional Audio**: `pygame>=2.5.0` is used by `alerts/audio_alert.py` for audio notifications. If `pygame` is not installed or no audio hardware is attached, `main.py` continues running with visual HUD alerts (`AUDIO: UNAVAILABLE`).
- **Windows / Linux**:
  ```bash
  pip install pygame>=2.5.0
  ```
- **macOS (Apple Silicon / Python 3.14 source build fallback)**: If `pip install pygame` attempts a source build and fails on missing SDL headers, use Python 3.11 or install SDL2 dependencies first via Homebrew:
  ```bash
  brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
  pip install pygame>=2.5.0
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
# Developer 1 (Computer Vision & Camera):
git checkout -b feat/computer-vision

# Developer 2 (Feature Extraction & Machine Learning):
git checkout -b feat/feature-extraction-ml

# Developer 3 (Decision Engine & Model Evaluation):
git checkout -b feat/decision-engine

# Developer 4 (UI, Audio Alerts & Integration):
git checkout -b feat/ui-alerts-integration
```

---

## 8. Running and Testing the Application

### 8.1 Model Asset Verification
The MediaPipe Face Landmarker model asset is stored in `models/saved_models/face_landmarker.task`.
If the model file is not present, download the official Google MediaPipe task model:
```bash
curl -o models/saved_models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 8.2 Automated Unit Test Suite
Execute the automated unit and integration test suite (**115 total tests** across Phases 1–5; see [docs/phase5_testing.md](docs/phase5_testing.md) for full environment-specific verification logs):
```bash
python -m unittest discover -s tests -v
```

**Test file breakdown (Phase 1–5):**
| Test File | Tests | Coverage |
|---|---|---|
| `test_video_capture.py` | 11 | Camera init, FPS, screenshot, context manager, error handling |
| `test_face_detection.py` | 19 | MediaPipe landmark extraction, coordinate projection, monotonic timestamps *(requires `cv2`, `mediapipe`)* |
| `test_eye_detection.py` | 8 | EAR formula, bilateral averaging, edge cases, epsilon guard |
| `test_mouth_detection.py` | 7 | MAR formula, 4-pt/6-pt geometry, edge cases, yawn helper |
| `test_head_pose.py` | 9 | solvePnP pipeline, angle conventions, camera matrix, failure modes *(requires `cv2`)* |
| `test_features.py` | 7 | FeatureExtractor aggregation, to_dict, to_array *(1 test requires `cv2`)* |
| `test_classifier.py` | 3 | DriverState enum targets, counter init/reset, decision logic execution |
| `test_threshold_rules.py` | 10 | Heuristic thresholds, EAR/MAR/Pose temporal counters, recovery smoothing, missing face |
| `test_hybrid_classifier.py` | 8 | Rule-ML fusion, safety override precedence, rule fallback, sliding-window smoothing |
| `test_model_training.py` | 8 | Synthetic generator, dataset loading, ModelTrainer (RF/SVM/GB), metrics, artifact saving |
| `test_prediction.py` | 6 | StatePredictor loading, offline fallback, inference on DriverFeatures, missing face |
| `test_audio_alert.py` | 19 | AlertManager lifecycle, cooldown, mute/unmute, missing deps/WAVs, WAV validity, `main.py` integration |
| **Total** | **115** | **19/19 Phase 5 tests passing; 102/115 passing without `cv2`/`mediapipe`** |

### 8.3 Machine Learning Model Training & Evaluation
To train a classifier on feature records using scikit-learn:
```bash
# 1. Train on existing feature dataset (defaults to dataset/driver_features.csv)
python models/train_model.py --model-type random_forest

# 2. Train with a synthetic test dataset (generates 600 synthetic samples for test harness validation)
python models/train_model.py --generate-synthetic --samples 600 --output models/saved_models/driver_state_classifier.joblib

# 3. Train using Gradient Boosting or Support Vector Machine
python models/train_model.py --model-type gradient_boosting
python models/train_model.py --model-type svm
```

> [!IMPORTANT]
> **Dataset Ethics & Real-World Validation**:
> Synthetic data generated by `--generate-synthetic` is strictly intended for software integration testing and continuous integration verification. Real-world safety claims require genuine human benchmark datasets (e.g. YawDD or NTHU).

### 8.4 Phase 5 Real-Time Hybrid Decision & Audio Alert Monitoring
- **Interactive Live Feed** (Hybrid Telemetry HUD + Color-Coded State Banner + Non-Blocking Audio Alerts):
  ```bash
  python main.py
  ```
- **Start Muted or Customize Cooldown / Volume**:
  ```bash
  python main.py --no-audio
  python main.py --alert-cooldown 4.0 --alert-volume 0.6
  ```
- **Live Preview with Custom ML and Rule Weights**:
  ```bash
  python main.py --ml-weight 0.6 --rule-weight 0.4
  ```
- **Live Preview with Custom Model Path**:
  ```bash
  python main.py --model-path models/saved_models/driver_state_classifier.joblib
  ```
- **Live Preview with Full 468-Point Face Mesh**:
  ```bash
  python main.py --draw-mesh
  ```
- **Headless Console Verification** (30-frame live verification loop, prints classification telemetry & audio status, exits cleanly):
  ```bash
  python main.py --no-gui
  ```
- **Keyboard Shortcuts during Live Video**:
  - `s` / `S`: Capture mirrored preview screenshot with HUD annotations to `screenshots/`
  - `m` / `M`: Toggle audio alerts between `AUDIO: ON` and `AUDIO: MUTED` (silences active playback immediately on mute)
  - `q`, `Q`, or `ESC`: Gracefully terminate and release all camera, detector, and audio mixer resources

### 8.5 Audio Architecture & Troubleshooting
- **Audio Assets**: `alerts/sounds/drowsy_alert.wav` (urgent triple-pulse dual-tone, 0.82s) and `alerts/sounds/distracted_alert.wav` (ascending two-note chime, 0.56s).
- **Audio Architecture**: `AlertManager` (`alerts/audio_alert.py`) preloads WAV assets once during `initialize()`, enforces a monotonic cooldown (`3.0s` default) to prevent frame-by-frame repetition, silences active playback when returning to `DriverState.ALERT`, and releases the mixer cleanly in `finally`.
- **HUD Status Badges**:
  - `AUDIO: ON` — Pygame mixer initialized and alerts unmuted.
  - `AUDIO: MUTED` — Alerts silenced via `M` key, `--no-audio`, or `ENABLE_AUDIO_ALERTS=False`.
  - `AUDIO: UNAVAILABLE` — `pygame` missing or audio hardware initialization failed (visual alerts continue unaffected).
- **Troubleshooting**:
  - *No sound heard*: Check the HUD badge (`AUDIO: ON` vs `MUTED` vs `UNAVAILABLE`), verify `ALERT_VOLUME` in `.env`, and confirm system output device is active.
  - *`AUDIO: UNAVAILABLE` on startup*: Install `pygame>=2.5.0` in the project virtual environment (Python 3.11 recommended).
  - *Missing `.wav` warning in logs*: Verify `SOUND_DROWSY_PATH` and `SOUND_DISTRACTED_PATH` point to valid PCM `.wav` files; missing files are skipped rather than substituted.

---

## 9. Architectural Integrity & Verification Checklist

### Verification Checklist:
- [x] Strict separation between detection engine, classification logic, audio actuation, and user interface.
- [x] Zero circular dependencies between packages.
- [x] Strongly typed data transfer contracts (`DriverFeatures`, `DriverState`).
- [x] Deterministic safety override ensuring prolonged eye closure or yawning always triggers `DROWSY`.
- [x] Graceful degradation to pure rule engine when ML model is missing or offline.
- [x] Graceful degradation to visual-only HUD alerts when `pygame`, audio hardware, or `.wav` assets are unavailable.
- [x] Unmirrored camera sensor frame preserved for inference; preview horizontally mirrored for natural driver ergonomics.
- [x] 19/19 Phase 5 audio unit & integration tests passing (with mocked hardware/audio); 102/115 total suite tests passing in the current Python 3.14 environment lacking `cv2`/`mediapipe`.

### External Asset Requirements:
- Real labeled data: Populate `dataset/driver_features.csv` with genuine driver recordings (for Phase 6/7 ML refinement).

# driver-fatigue-
