"""Feature extraction and representation module.

Aggregates individual physiological measurements (Eye Aspect Ratio, Mouth Aspect Ratio)
and 3D head orientation angles (Pitch, Yaw, Roll) into structured feature vectors
used by both the rule-based decision engine and future machine learning classification models.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from detection.eye_detection import EyeDetectionResult, EyeDetector
from detection.head_pose import HeadPoseEstimator, HeadPoseResult
from detection.mouth_detection import MouthDetectionResult, MouthDetector


@dataclass
class DriverFeatures:
    """Structured feature vector representing driver state at a discrete frame.

    Attributes:
        timestamp (float): Epoch timestamp in seconds when the frame was processed.
        left_ear (float): Left Eye Aspect Ratio.
        right_ear (float): Right Eye Aspect Ratio.
        average_ear (float): Bilateral mean Eye Aspect Ratio.
        mar (float): Mouth Aspect Ratio (vertical opening / horizontal width).
        pitch (float): Head pitch angle in degrees (positive = nodding down, negative = up).
        yaw (float): Head yaw angle in degrees (positive = turning right, negative = left).
        roll (float): Head roll angle in degrees (positive = tilting clockwise).
        face_detected (bool): True if driver's face was detected in this frame.
        valid (bool): True if all physiological features were calculated successfully.
        ear (float): Alias for average_ear for model ingestion and backward compatibility.
    """
    timestamp: float = 0.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    average_ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    face_detected: bool = False
    valid: bool = False
    ear: float = 0.0

    def __post_init__(self) -> None:
        """Synchronize ear and average_ear aliases."""
        if self.ear == 0.0 and self.average_ear != 0.0:
            self.ear = self.average_ear
        elif self.average_ear == 0.0 and self.ear != 0.0:
            self.average_ear = self.ear

    def to_dict(self) -> Dict[str, Any]:
        """Convert features dataclass to a dictionary."""
        return asdict(self)

    def to_array(self, feature_order: Optional[List[str]] = None) -> np.ndarray:
        """Convert features to an ordered 1D NumPy array for ML model inference.

        Args:
            feature_order: List of attribute names defining vector order.
                Defaults to ['ear', 'mar', 'pitch', 'yaw', 'roll'].

        Returns:
            np.ndarray: 1D array of ordered feature values (float32).
        """
        if feature_order is None:
            feature_order = ["ear", "mar", "pitch", "yaw", "roll"]
        return np.array([getattr(self, k) for k in feature_order], dtype=np.float32)


class FeatureExtractor:
    """Coordinates eye, mouth, and head pose sub-modules to produce DriverFeatures."""

    def __init__(
        self,
        eye_detector: Optional[EyeDetector] = None,
        mouth_detector: Optional[MouthDetector] = None,
        head_pose_estimator: Optional[HeadPoseEstimator] = None,
    ) -> None:
        """Initialize feature extraction pipeline sub-detectors.

        Args:
            eye_detector: Optional pre-configured EyeDetector instance.
            mouth_detector: Optional pre-configured MouthDetector instance.
            head_pose_estimator: Optional pre-configured HeadPoseEstimator instance.
        """
        self.eye_detector = eye_detector if eye_detector is not None else EyeDetector()
        self.mouth_detector = mouth_detector if mouth_detector is not None else MouthDetector()
        self.head_pose_estimator = (
            head_pose_estimator if head_pose_estimator is not None else HeadPoseEstimator()
        )

    def extract_features(
        self,
        landmarks: Any,
        frame_shape: Tuple[int, int],
        timestamp: Optional[float] = None,
    ) -> DriverFeatures:
        """Extract all physiological and pose features from detected facial landmarks.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList, list of landmarks, or None.
            frame_shape: Height and width of the unmirrored camera frame (H, W).
            timestamp: Optional timestamp; defaults to current time.time().

        Returns:
            DriverFeatures: Populated dataclass containing all computed metrics.
        """
        ts = float(timestamp if timestamp is not None else time.time())

        # Handle missing-face conditions
        if landmarks is None:
            return DriverFeatures(
                timestamp=ts,
                face_detected=False,
                valid=False,
            )

        landmark_list = getattr(landmarks, "landmark", landmarks)
        if not hasattr(landmark_list, "__getitem__") or len(landmark_list) < 468:
            return DriverFeatures(
                timestamp=ts,
                face_detected=False,
                valid=False,
            )

        # 1. Calculate Eye Aspect Ratio (EAR)
        eye_res: EyeDetectionResult = self.eye_detector.detect_eyes(landmarks, frame_shape)

        # 2. Calculate Mouth Aspect Ratio (MAR)
        mouth_res: MouthDetectionResult = self.mouth_detector.detect_mouth(landmarks, frame_shape)

        # 3. Estimate Head Pose (Pitch, Yaw, Roll)
        pose_res: HeadPoseResult = self.head_pose_estimator.estimate_head_pose(landmarks, frame_shape)

        # Valid only if all sub-detector computations completed successfully
        is_valid = bool(eye_res.valid and mouth_res.valid and pose_res.valid)

        return DriverFeatures(
            timestamp=ts,
            left_ear=eye_res.left_ear,
            right_ear=eye_res.right_ear,
            average_ear=eye_res.average_ear,
            mar=mouth_res.mar,
            pitch=pose_res.pitch,
            yaw=pose_res.yaw,
            roll=pose_res.roll,
            face_detected=True,
            valid=is_valid,
            ear=eye_res.average_ear,
        )
