"""Feature extraction and representation module.

Aggregates individual physiological measurements (EAR, MAR) and head orientation
angles (Pitch, Yaw, Roll) into structured feature vectors used by both the rule-based
decision engine and machine learning classification models.
"""

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DriverFeatures:
    """Structured feature vector representing driver state at a discrete frame.

    Attributes:
        ear: Scalar Eye Aspect Ratio (average of left and right eyes).
        mar: Scalar Mouth Aspect Ratio.
        pitch: Head pitch angle in degrees (vertical nod).
        yaw: Head yaw angle in degrees (horizontal turn).
        roll: Head roll angle in degrees (lateral tilt).
        timestamp: Epoch timestamp in seconds when the frame was processed.
    """
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert features to a dictionary format."""
        return asdict(self)

    def to_array(self, feature_order: Optional[List[str]] = None) -> np.ndarray:
        """Convert features to an ordered 1D NumPy array for ML model inference.

        Args:
            feature_order: List of attribute names defining vector order.
                Defaults to ['ear', 'mar', 'pitch', 'yaw', 'roll'].

        Returns:
            np.ndarray: 1D array of ordered feature values.
        """
        if feature_order is None:
            feature_order = ["ear", "mar", "pitch", "yaw", "roll"]
        return np.array([getattr(self, k) for k in feature_order], dtype=np.float32)


class FeatureExtractor:
    """Coordinates eye, mouth, and head pose sub-modules to produce DriverFeatures."""

    def __init__(self) -> None:
        """Initialize feature extraction pipeline dependencies."""
        pass

    def extract_features(
        self,
        landmarks: Any,
        frame_shape: Tuple[int, int],
        timestamp: Optional[float] = None,
    ) -> DriverFeatures:
        """Extract all physiological and pose features from detected facial landmarks.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList.
            frame_shape: Height and width of the image frame (H, W).
            timestamp: Optional timestamp; defaults to current time.time().

        Returns:
            DriverFeatures: Populated dataclass containing all computed metrics.
        """
        raise NotImplementedError("Phase 3 implementation: Feature extraction aggregation.")

