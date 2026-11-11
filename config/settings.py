"""Centralized configuration module for the Driver Fatigue and Distraction Detection System.

This module manages all application hyperparameters, physical and temporal thresholds,
camera inputs, asset paths, and machine learning model configurations.
Values can be overridden using environment variables or a .env file.
"""

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import List, Tuple

# Try loading .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base Project Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATASET_DIR: Path = BASE_DIR / "dataset"
SAVED_MODELS_DIR: Path = BASE_DIR / "models" / "saved_models"
SOUNDS_DIR: Path = BASE_DIR / "alerts" / "sounds"
SCREENSHOTS_DIR: Path = BASE_DIR / "screenshots"


class DriverState(str, Enum):
    """Driver physiological and behavioral states."""
    ALERT = "ALERT"
    DROWSY = "DROWSY"
    DISTRACTED = "DISTRACTED"


@dataclass(frozen=True)
class CameraConfig:
    """Video capture hardware and stream parameters."""
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    frame_width: int = int(os.getenv("FRAME_WIDTH", "640"))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", "480"))
    target_fps: int = int(os.getenv("TARGET_FPS", "30"))


@dataclass(frozen=True)
class MediaPipeConfig:
    """MediaPipe Face Mesh detector parameters."""
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class LandmarkIndices:
    """Standard MediaPipe 468/478 Face Mesh landmark index mappings."""
    # Eye indices for EAR calculation (6 points per eye)
    # Left eye: [corner1, top1, top2, corner2, bottom2, bottom1]
    LEFT_EYE: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)
    # Right eye: [corner1, top1, top2, corner2, bottom2, bottom1]
    RIGHT_EYE: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)

    # Mouth inner/outer landmarks for MAR calculation
    MOUTH_TOP: int = 13
    MOUTH_BOTTOM: int = 14
    MOUTH_LEFT: int = 78
    MOUTH_RIGHT: int = 308

    # 3D Head Pose Landmark Points for solvePnP
    NOSE_TIP: int = 1
    CHIN: int = 199
    LEFT_EYE_CORNER: int = 33
    RIGHT_EYE_CORNER: int = 263
    LEFT_MOUTH_CORNER: int = 61
    RIGHT_MOUTH_CORNER: int = 291


@dataclass(frozen=True)
class ThresholdConfig:
    """Heuristic and temporal threshold parameters for rule-based engine."""
    # Eye Aspect Ratio (EAR) - baseline calibration
    ear_threshold: float = float(os.getenv("EAR_THRESHOLD", "0.25"))
    ear_consec_frames: int = int(os.getenv("EAR_CONSEC_FRAMES", "20"))  # ~1.0-1.5 sec at 20fps

    # Mouth Aspect Ratio (MAR) - yawning calibration
    mar_threshold: float = float(os.getenv("MAR_THRESHOLD", "0.60"))
    mar_consec_frames: int = int(os.getenv("MAR_CONSEC_FRAMES", "15"))

    # Head Pose (Degrees) - deviation thresholds from forward center
    head_pitch_threshold: float = float(os.getenv("HEAD_PITCH_THRESHOLD", "15.0"))  # Nodding down/up
    head_yaw_threshold: float = float(os.getenv("HEAD_YAW_THRESHOLD", "20.0"))      # Looking left/right
    head_roll_threshold: float = float(os.getenv("HEAD_ROLL_THRESHOLD", "20.0"))    # Tilting head
    distraction_consec_frames: int = int(os.getenv("DISTRACTION_CONSEC_FRAMES", "25"))


@dataclass(frozen=True)
class AlertConfig:
    """Audio alert triggers and cooldown configuration."""
    enable_audio: bool = os.getenv("ENABLE_AUDIO_ALERTS", "True").lower() in ("true", "1")
    cooldown_seconds: float = float(os.getenv("ALERT_COOLDOWN_SECONDS", "3.0"))
    volume: float = float(os.getenv("ALERT_VOLUME", "0.8"))
    sound_drowsy_path: Path = SOUNDS_DIR / "drowsy_alert.wav"
    sound_distracted_path: Path = SOUNDS_DIR / "distracted_alert.wav"


@dataclass(frozen=True)
class ModelConfig:
    """Machine learning model artifact and feature schema configuration."""
    model_path: Path = SAVED_MODELS_DIR / "driver_state_classifier.joblib"
    feature_columns: List[str] = field(default_factory=lambda: [
        "ear",
        "mar",
        "pitch",
        "yaw",
        "roll"
    ])
    target_column: str = "state"
    hybrid_weight_ml: float = 0.5  # Weight given to ML probabilities in hybrid engine
    hybrid_weight_rules: float = 0.5  # Weight given to temporal rule confidence


@dataclass(frozen=True)
class AppConfig:
    """Root configuration holding all sub-system parameter trees."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    landmarks: LandmarkIndices = field(default_factory=LandmarkIndices)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    models: ModelConfig = field(default_factory=ModelConfig)


# Global singleton instance of configuration
settings: AppConfig = AppConfig()

