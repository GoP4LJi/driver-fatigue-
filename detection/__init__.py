"""Computer Vision detection package for Driver Fatigue and Distraction Detection System.

Contains modules for video acquisition, facial landmark detection, eye analysis (EAR),
mouth analysis (MAR), head pose estimation, and the unified DetectionPipeline interface
shared between desktop OpenCV and the Streamlit dashboard.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from config.settings import AppConfig, DriverState, settings
from detection.video_capture import VideoCamera
from detection.face_detection import FaceDetector
from detection.eye_detection import EyeDetector
from detection.mouth_detection import MouthDetector
from detection.head_pose import HeadPoseEstimator
from features.feature_extraction import DriverFeatures, FeatureExtractor
from decision_engine.hybrid_classifier import HybridDecisionEngine
from alerts.audio_alert import AlertManager
from visualization.draw_utils import Visualizer


@dataclass
class FrameAnalysisResult:
    """Standardized output container produced for each processed video frame.

    Shared contract consumed identically by both main.py (OpenCV) and dashboard/app.py (Streamlit).

    Attributes:
        original_frame: Unmodified input frame (BGR NumPy ndarray).
        annotated_frame: Frame with HUD, landmark contours, and status overlays rendered.
        state: Resolved driver attentiveness state (ALERT, DROWSY, DISTRACTED).
        features: Extracted physiological and pose feature vector (EAR, MAR, Pitch, Yaw, Roll).
        landmarks: Raw MediaPipe facial landmark coordinates (or None if no face detected).
        telemetry: Dictionary containing FPS, active temporal counters, and arbitration confidence.
        alert_triggered: Whether an auditory alarm was dispatched for this frame.
    """
    original_frame: np.ndarray
    annotated_frame: np.ndarray
    state: DriverState
    features: Optional[DriverFeatures] = None
    landmarks: Optional[Any] = None
    telemetry: Optional[Dict[str, Any]] = None
    alert_triggered: bool = False


class DetectionPipeline:
    """Unified driver monitoring engine decoupled from video transport and presentation layers.

    Enables both standalone OpenCV desktop execution (main.py) and Streamlit web dashboard
    (dashboard/app.py) to process video frames using the exact same underlying logic.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize the shared detection engine components."""
        self.config = config or settings
        self.face_detector = FaceDetector(
            max_num_faces=self.config.mediapipe.max_num_faces,
            refine_landmarks=self.config.mediapipe.refine_landmarks,
            min_detection_confidence=self.config.mediapipe.min_detection_confidence,
            min_tracking_confidence=self.config.mediapipe.min_tracking_confidence,
        )
        self.feature_extractor = FeatureExtractor()
        self.decision_engine = HybridDecisionEngine()
        self.alert_manager = AlertManager(
            enabled=self.config.alerts.enable_audio,
            cooldown=self.config.alerts.cooldown_seconds,
            volume=self.config.alerts.volume,
        )
        self.visualizer = Visualizer()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> FrameAnalysisResult:
        """Process a single BGR video frame through the complete detection pipeline.

        Processing Flow:
            1. Face landmark detection (MediaPipe Face Mesh)
            2. Feature extraction (EAR, MAR, solvePnP Euler angles)
            3. Decision engine classification (fusing temporal counters and ML model)
            4. Audio alert evaluation and dispatch (Pygame mixer)
            5. HUD overlay rendering (bounding contours, 3D axes, status banner)

        Args:
            frame (np.ndarray): Input BGR video frame.
            timestamp (Optional[float]): Frame capture timestamp in epoch seconds.

        Returns:
            FrameAnalysisResult: Populated analytical result and annotated frame.
        """
        raise NotImplementedError("Phase 4/7 implementation: Frame processing pipeline.")


__all__ = [
    "VideoCamera",
    "FaceDetector",
    "EyeDetector",
    "MouthDetector",
    "HeadPoseEstimator",
    "DetectionPipeline",
    "FrameAnalysisResult",
]
