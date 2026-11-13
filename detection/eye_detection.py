"""Eye landmark processing and Eye Aspect Ratio (EAR) calculation module.

Implements the Eye Aspect Ratio (EAR) proposed by Soukupová and Čech (2016) to quantify
eyelid aperture geometry. Computes individual left and right EAR, bilateral averages,
and validates anatomical landmark consistency while avoiding division-by-zero errors.

Mathematical Formula:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
Where p1..p6 represent the 6 2D facial landmark coordinates bounding the eye:
    - p1, p4: Outer and inner eye corners (horizontal axis).
    - p2, p6: First vertical top-to-bottom eyelid point pair.
    - p3, p5: Second vertical top-to-bottom eyelid point pair.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np

from config.settings import settings


@dataclass
class EyeDetectionResult:
    """Structured result containing calculated eye aperture metrics.

    Attributes:
        left_ear (float): Eye Aspect Ratio for the left eye.
        right_ear (float): Eye Aspect Ratio for the right eye.
        average_ear (float): Scalar mean of left and right EAR.
        valid (bool): True if landmarks were valid and successfully evaluated.
    """
    left_ear: float = 0.0
    right_ear: float = 0.0
    average_ear: float = 0.0
    valid: bool = False

    def __iter__(self):
        """Allow tuple-like unpacking (left_ear, right_ear, average_ear)."""
        yield self.left_ear
        yield self.right_ear
        yield self.average_ear


class EyeDetector:
    """Calculates Eye Aspect Ratio (EAR) from facial landmark coordinate arrays.

    Follows the Soukupová and Čech (2016) formulation:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
    """

    EPSILON: float = 1e-6

    def __init__(self, ear_threshold: Optional[float] = None) -> None:
        """Initialize EyeDetector with baseline configuration thresholds.

        Args:
            ear_threshold: Optional closure threshold; defaults to settings.thresholds.ear_threshold.
        """
        self.ear_threshold = (
            ear_threshold
            if ear_threshold is not None
            else settings.thresholds.ear_threshold
        )
        self.left_indices = settings.landmarks.LEFT_EYE
        self.right_indices = settings.landmarks.RIGHT_EYE

    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """Compute the Eye Aspect Ratio for a single eye given 6 ordered boundary points.

        Args:
            eye_points (np.ndarray): Coordinate array of shape (6, 2) or (6, >=2)
                containing [p1, p2, p3, p4, p5, p6] in normalized or pixel coordinates.

        Returns:
            float: Calculated Eye Aspect Ratio, or 0.0 if inputs are invalid/degenerate.
        """
        if eye_points is None or not isinstance(eye_points, np.ndarray):
            return 0.0

        if eye_points.ndim < 2 or eye_points.shape[0] < 6 or eye_points.shape[1] < 2:
            return 0.0

        if np.isnan(eye_points).any() or np.isinf(eye_points).any():
            return 0.0

        pts = eye_points[:6, :2].astype(np.float64)

        # Extract landmark positions:
        # p1: corner 1, p4: corner 2
        # p2: top 1,    p6: bottom 1
        # p3: top 2,    p5: bottom 2
        p1, p2, p3, p4, p5, p6 = pts[0], pts[1], pts[2], pts[3], pts[4], pts[5]

        # Vertical distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)

        # Horizontal distance
        h = np.linalg.norm(p1 - p4)

        # Guard against zero or near-zero horizontal eye width
        if h < self.EPSILON:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)

        if np.isnan(ear) or np.isinf(ear) or ear < 0.0:
            return 0.0

        return float(ear)

    def compute_average_ear(
        self,
        left_eye_points: np.ndarray,
        right_eye_points: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute individual EAR for both eyes and their scalar mean.

        Args:
            left_eye_points (np.ndarray): Left eye coordinates of shape (6, 2).
            right_eye_points (np.ndarray): Right eye coordinates of shape (6, 2).

        Returns:
            Tuple[float, float, float]: (left_ear, right_ear, average_ear).
        """
        left_ear = self.calculate_ear(left_eye_points)
        right_ear = self.calculate_ear(right_eye_points)

        # If both are valid, average both. If only one is non-zero, average with available.
        if left_ear > 0.0 and right_ear > 0.0:
            avg_ear = (left_ear + right_ear) / 2.0
        elif left_ear > 0.0:
            avg_ear = left_ear
        elif right_ear > 0.0:
            avg_ear = right_ear
        else:
            avg_ear = 0.0

        return float(left_ear), float(right_ear), float(avg_ear)

    def detect_eyes(
        self,
        landmarks: Any,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> EyeDetectionResult:
        """Extract eye boundary landmarks and calculate bilateral EAR metrics.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList, list of NormalizedLandmarks,
                or a coordinate array.
            frame_shape: Optional (height, width) tuple for pixel coordinate projection.

        Returns:
            EyeDetectionResult: Populated result with left, right, and average EAR.
        """
        if landmarks is None:
            return EyeDetectionResult(left_ear=0.0, right_ear=0.0, average_ear=0.0, valid=False)

        # Extract landmarks list if MediaPipe object
        landmark_list = getattr(landmarks, "landmark", landmarks)
        if not hasattr(landmark_list, "__getitem__") or len(landmark_list) < 468:
            return EyeDetectionResult(left_ear=0.0, right_ear=0.0, average_ear=0.0, valid=False)

        h, w = (frame_shape[:2]) if frame_shape and len(frame_shape) >= 2 else (1.0, 1.0)

        try:
            left_pts = []
            for idx in self.left_indices:
                lm = landmark_list[idx]
                left_pts.append([lm.x * w, lm.y * h])

            right_pts = []
            for idx in self.right_indices:
                lm = landmark_list[idx]
                right_pts.append([lm.x * w, lm.y * h])

            left_arr = np.array(left_pts, dtype=np.float64)
            right_arr = np.array(right_pts, dtype=np.float64)

            left_ear, right_ear, avg_ear = self.compute_average_ear(left_arr, right_arr)
            valid = (left_ear > 0.0) and (right_ear > 0.0)

            return EyeDetectionResult(
                left_ear=left_ear,
                right_ear=right_ear,
                average_ear=avg_ear,
                valid=valid,
            )
        except (IndexError, AttributeError, TypeError):
            return EyeDetectionResult(left_ear=0.0, right_ear=0.0, average_ear=0.0, valid=False)

    def is_eye_closed(self, ear: float, threshold: Optional[float] = None) -> bool:
        """Evaluate if an EAR metric falls below the closure threshold.

        Args:
            ear (float): Calculated Eye Aspect Ratio.
            threshold (Optional[float]): Custom threshold override.

        Returns:
            bool: True if ear is strictly below threshold, False otherwise.
        """
        th = threshold if threshold is not None else self.ear_threshold
        return bool(ear < th)
