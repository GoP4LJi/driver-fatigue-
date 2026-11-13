"""Unit tests for Mouth Aspect Ratio (MAR) calculation and MouthDetector module.

Validates oral cavity expansion calculations, edge-case handling for zero mouth width,
different mouth opening apertures, and missing landmark inputs without camera hardware.
"""

from dataclasses import dataclass
from typing import List
import unittest
import numpy as np

from detection.mouth_detection import MouthDetectionResult, MouthDetector


@dataclass
class MockNormalizedLandmark:
    """Mock structure mimicking MediaPipe's NormalizedLandmark."""
    x: float
    y: float
    z: float = 0.0


def create_mock_mouth_landmarks(mar_state: str = "closed", frame_shape: tuple = (480, 640)) -> List[MockNormalizedLandmark]:
    """Create a 478-length list of mock landmarks with specified mouth state.

    Args:
        mar_state: 'closed', 'talking', or 'yawning'.
        frame_shape: (height, width) for normalized scaling.
    """
    landmarks = [MockNormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    h, w = frame_shape

    # Set horizontal width in pixels = 100 px (0.42 to 0.57625)
    # Width in normalized coords: dx = 100.0 / w
    dx = 100.0 / float(w)
    landmarks[78] = MockNormalizedLandmark(x=0.50 - dx / 2.0, y=0.65)   # left corner
    landmarks[308] = MockNormalizedLandmark(x=0.50 + dx / 2.0, y=0.65)  # right corner

    if mar_state == "closed":
        target_mar = 0.10  # 10 px vertical
    elif mar_state == "talking":
        target_mar = 0.30  # 30 px vertical
    else:  # yawning
        target_mar = 0.75  # 75 px vertical

    v_px = target_mar * 100.0
    dy = (v_px / 2.0) / float(h)

    landmarks[13] = MockNormalizedLandmark(x=0.50, y=0.65 - dy)  # top lip
    landmarks[14] = MockNormalizedLandmark(x=0.50, y=0.65 + dy)  # bottom lip

    return landmarks


class TestMouthDetection(unittest.TestCase):
    """Test suite for MouthDetector class and mathematical MAR computation."""

    def setUp(self) -> None:
        self.detector = MouthDetector(mar_threshold=0.60)

    def test_calculate_mar_normal_mouth_geometry(self) -> None:
        """Verify MAR on known synthetic coordinates [top, bottom, left, right].

        Points:
            p_top = (0, 10), p_bottom = (0, -10) -> vertical = 20
            p_left = (-25, 0), p_right = (25, 0) -> horizontal = 50
            Expected MAR = 20 / 50 = 0.40
        """
        mouth_pts = np.array([
            [0.0, 10.0],   # top
            [0.0, -10.0],  # bottom
            [-25.0, 0.0],  # left
            [25.0, 0.0],   # right
        ])
        mar = self.detector.calculate_mar(mouth_pts)
        self.assertAlmostEqual(mar, 0.40, places=4)

    def test_different_mouth_opening_geometries(self) -> None:
        """Verify MAR scales monotonically from closed mouth to wide yawning."""
        closed_pts = np.array([[0.0, 2.0], [0.0, -2.0], [-20.0, 0.0], [20.0, 0.0]])   # MAR = 4/40 = 0.10
        talking_pts = np.array([[0.0, 6.0], [0.0, -6.0], [-20.0, 0.0], [20.0, 0.0]]) # MAR = 12/40 = 0.30
        yawning_pts = np.array([[0.0, 16.0], [0.0, -16.0], [-20.0, 0.0], [20.0, 0.0]]) # MAR = 32/40 = 0.80

        mar_closed = self.detector.calculate_mar(closed_pts)
        mar_talking = self.detector.calculate_mar(talking_pts)
        mar_yawning = self.detector.calculate_mar(yawning_pts)

        self.assertLess(mar_closed, mar_talking)
        self.assertLess(mar_talking, mar_yawning)
        self.assertAlmostEqual(mar_closed, 0.10, places=3)
        self.assertAlmostEqual(mar_talking, 0.30, places=3)
        self.assertAlmostEqual(mar_yawning, 0.80, places=3)

    def test_invalid_landmarks_handling(self) -> None:
        """Verify detector handles None, truncated, NaN, or non-array inputs safely."""
        self.assertEqual(self.detector.calculate_mar(None), 0.0)

        # Insufficient points (< 4)
        self.assertEqual(self.detector.calculate_mar(np.array([[0.0, 0.0], [1.0, 1.0]])), 0.0)

        # Non-array input
        self.assertEqual(self.detector.calculate_mar("invalid_type"), 0.0)

        # NaN array
        nan_pts = np.full((4, 2), np.nan)
        self.assertEqual(self.detector.calculate_mar(nan_pts), 0.0)

        # Infinite array
        inf_pts = np.full((4, 2), np.inf)
        self.assertEqual(self.detector.calculate_mar(inf_pts), 0.0)

    def test_missing_landmarks_in_detect_mouth(self) -> None:
        """Verify detect_mouth returns valid=False when landmarks are None or truncated."""
        res_none = self.detector.detect_mouth(None, (480, 640))
        self.assertFalse(res_none.valid)
        self.assertEqual(res_none.mar, 0.0)

        res_short = self.detector.detect_mouth([MockNormalizedLandmark(0, 0)] * 10, (480, 640))
        self.assertFalse(res_short.valid)

    def test_zero_mouth_width(self) -> None:
        """Verify detector avoids division-by-zero when horizontal mouth width is zero."""
        degenerate = np.zeros((4, 2), dtype=np.float64)
        self.assertEqual(self.detector.calculate_mar(degenerate), 0.0)

        near_zero = np.array([[0.0, 10.0], [0.0, -10.0], [0.0, 0.0], [1e-8, 0.0]])
        self.assertEqual(self.detector.calculate_mar(near_zero), 0.0)

    def test_detect_mouth_with_mock_landmarks(self) -> None:
        """Verify end-to-end landmark extraction from mock landmark objects."""
        # Test closed mouth
        mock_closed = create_mock_mouth_landmarks("closed")
        res_closed = self.detector.detect_mouth(mock_closed, (480, 640))
        self.assertTrue(res_closed.valid)
        self.assertAlmostEqual(res_closed.mar, 0.10, places=2)
        self.assertFalse(self.detector.is_yawning(res_closed.mar))

        # Test yawning mouth
        mock_yawn = create_mock_mouth_landmarks("yawning")
        res_yawn = self.detector.detect_mouth(mock_yawn, (480, 640))
        self.assertTrue(res_yawn.valid)
        self.assertGreater(res_yawn.mar, 0.60)
        self.assertTrue(self.detector.is_yawning(res_yawn.mar))

        # Test tuple unpacking
        (mar_val,) = res_yawn
        self.assertEqual(mar_val, res_yawn.mar)

    def test_six_point_mar_calculation(self) -> None:
        """Verify detector also supports 6-point mouth geometry (like EAR)."""
        pts_6 = np.array([
            [-25.0, 0.0],  # corner 1
            [-10.0, 10.0], # top 1
            [10.0, 10.0],  # top 2
            [25.0, 0.0],   # corner 2
            [10.0, -10.0], # bottom 2
            [-10.0, -10.0] # bottom 1
        ])
        # v1 = 20, v2 = 20, h = 50 -> MAR = (20 + 20) / (2 * 50) = 40 / 100 = 0.40
        mar = self.detector.calculate_mar(pts_6)
        self.assertAlmostEqual(mar, 0.40, places=4)


if __name__ == "__main__":
    unittest.main()
