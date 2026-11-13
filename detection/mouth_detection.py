"""Mouth landmark analysis and Mouth Aspect Ratio (MAR) calculation module.

Used to quantify vertical oral cavity expansion (mouth opening) relative to mouth width.
Calculates Mouth Aspect Ratio (MAR) to establish numerical baselines for yawning detection,
an early physiological indicator of driver fatigue and drowsiness.

Mathematical Formula:
    MAR = ||p_top - p_bottom|| / ||p_left - p_right||
Where:
    - p_top (landmark 13): Upper inner lip center point.
    - p_bottom (landmark 14): Lower inner lip center point.
    - p_left (landmark 78): Left inner mouth corner.
    - p_right (landmark 308): Right inner mouth corner.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np

from config.settings import settings


@dataclass
class MouthDetectionResult:
    """Structured result containing calculated mouth aspect ratio metrics.

    Attributes:
        mar (float): Mouth Aspect Ratio (vertical opening / horizontal width).
        valid (bool): True if landmarks were valid and successfully computed.
    """
    mar: float = 0.0
    valid: bool = False

    def __iter__(self):
        """Allow tuple-like unpacking (mar,)."""
        yield self.mar


class MouthDetector:
    """Calculates Mouth Aspect Ratio (MAR) from facial landmark coordinate arrays.

    Standard Formulation:
        MAR = ||p_top - p_bottom|| / ||p_left - p_right||
    """

    EPSILON: float = 1e-6

    def __init__(self, mar_threshold: Optional[float] = None) -> None:
        """Initialize MouthDetector with yawning threshold.

        Args:
            mar_threshold: Optional ratio above which mouth is open;
                defaults to settings.thresholds.mar_threshold.
        """
        self.mar_threshold = (
            mar_threshold
            if mar_threshold is not None
            else settings.thresholds.mar_threshold
        )
        self.top_idx = settings.landmarks.MOUTH_TOP
        self.bottom_idx = settings.landmarks.MOUTH_BOTTOM
        self.left_idx = settings.landmarks.MOUTH_LEFT
        self.right_idx = settings.landmarks.MOUTH_RIGHT

    def calculate_mar(self, mouth_points: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio from facial landmark coordinate points.

        Accepts either:
            - Shape (4, 2): [p_top, p_bottom, p_left, p_right]
            - Shape (3, 2, 2): 3 vertical point pairs + 1 horizontal corner pair,
              or (6, 2): [corner1, top1, top2, corner2, bottom2, bottom1]

        Args:
            mouth_points (np.ndarray): Coordinate array containing mouth boundary landmarks.

        Returns:
            float: Scalar Mouth Aspect Ratio, or 0.0 if inputs are invalid or degenerate.
        """
        if mouth_points is None or not isinstance(mouth_points, np.ndarray):
            return 0.0

        if mouth_points.ndim < 2 or mouth_points.shape[0] < 4 or mouth_points.shape[1] < 2:
            return 0.0

        if np.isnan(mouth_points).any() or np.isinf(mouth_points).any():
            return 0.0

        pts = mouth_points[:, :2].astype(np.float64)

        if pts.shape[0] == 4:
            # Format: [top, bottom, left, right]
            p_top = pts[0]
            p_bottom = pts[1]
            p_left = pts[2]
            p_right = pts[3]

            v = np.linalg.norm(p_top - p_bottom)
            h = np.linalg.norm(p_left - p_right)

            if h < self.EPSILON:
                return 0.0

            mar = v / h

        elif pts.shape[0] >= 6:
            # 6-point formulation similar to EAR:
            # [corner1, top1, top2, corner2, bottom2, bottom1]
            p1, p2, p3, p4, p5, p6 = pts[0], pts[1], pts[2], pts[3], pts[4], pts[5]
            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            h = np.linalg.norm(p1 - p4)

            if h < self.EPSILON:
                return 0.0

            mar = (v1 + v2) / (2.0 * h)

        else:
            return 0.0

        if np.isnan(mar) or np.isinf(mar) or mar < 0.0:
            return 0.0

        return float(mar)

    def detect_mouth(
        self,
        landmarks: Any,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> MouthDetectionResult:
        """Extract mouth landmarks and calculate the MAR metric.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList or list of landmark objects.
            frame_shape: Optional (height, width) tuple for pixel coordinate projection.

        Returns:
            MouthDetectionResult: Result indicating MAR and validity flag.
        """
        if landmarks is None:
            return MouthDetectionResult(mar=0.0, valid=False)

        landmark_list = getattr(landmarks, "landmark", landmarks)
        if not hasattr(landmark_list, "__getitem__") or len(landmark_list) < 468:
            return MouthDetectionResult(mar=0.0, valid=False)

        h, w = (frame_shape[:2]) if frame_shape and len(frame_shape) >= 2 else (1.0, 1.0)

        try:
            p_top = [landmark_list[self.top_idx].x * w, landmark_list[self.top_idx].y * h]
            p_bottom = [landmark_list[self.bottom_idx].x * w, landmark_list[self.bottom_idx].y * h]
            p_left = [landmark_list[self.left_idx].x * w, landmark_list[self.left_idx].y * h]
            p_right = [landmark_list[self.right_idx].x * w, landmark_list[self.right_idx].y * h]

            pts = np.array([p_top, p_bottom, p_left, p_right], dtype=np.float64)
            mar = self.calculate_mar(pts)
            valid = mar > 0.0

            return MouthDetectionResult(mar=mar, valid=valid)
        except (IndexError, AttributeError, TypeError):
            return MouthDetectionResult(mar=0.0, valid=False)

    def is_yawning(self, mar: float, threshold: Optional[float] = None) -> bool:
        """Determine if current MAR exceeds the yawning threshold.

        Args:
            mar (float): Calculated Mouth Aspect Ratio.
            threshold (Optional[float]): Custom threshold override.

        Returns:
            bool: True if mar strictly exceeds threshold, False otherwise.
        """
        th = threshold if threshold is not None else self.mar_threshold
        return bool(mar > th)
