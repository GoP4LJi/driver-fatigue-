"""Unit tests for Eye Aspect Ratio (EAR) calculation and EyeDetector module.

Validates mathematical correctness of the Soukupová & Čech (2016) formula,
handles degraded/invalid landmarks, boundary edge cases, zero-width eyes,
and bilateral averaging without requiring physical camera hardware.
"""

from dataclasses import dataclass
from typing import List
import unittest
import numpy as np

from detection.eye_detection import EyeDetectionResult, EyeDetector


@dataclass
class MockNormalizedLandmark:
    """Mock structure mimicking MediaPipe's NormalizedLandmark."""
    x: float
    y: float
    z: float = 0.0


def create_mock_eye_landmarks(left_open: bool = True, right_open: bool = True) -> List[MockNormalizedLandmark]:
    """Create a 478-length list of mock landmarks with specified eye opening state."""
    landmarks = [MockNormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    # Left eye indices: (362, 385, 387, 263, 373, 380)
    # Horizontal width = |0.65 - 0.55| = 0.10
    landmarks[362] = MockNormalizedLandmark(x=0.55, y=0.35)  # p1
    landmarks[263] = MockNormalizedLandmark(x=0.65, y=0.35)  # p4
    dy_left = 0.035 if left_open else 0.003
    landmarks[385] = MockNormalizedLandmark(x=0.58, y=0.35 - dy_left)  # p2 top
    landmarks[380] = MockNormalizedLandmark(x=0.58, y=0.35 + dy_left)  # p6 bottom
    landmarks[387] = MockNormalizedLandmark(x=0.62, y=0.35 - dy_left)  # p3 top
    landmarks[373] = MockNormalizedLandmark(x=0.62, y=0.35 + dy_left)  # p5 bottom

    # Right eye indices: (33, 160, 158, 133, 153, 144)
    # Horizontal width = |0.45 - 0.35| = 0.10
    landmarks[33] = MockNormalizedLandmark(x=0.35, y=0.35)   # p1
    landmarks[133] = MockNormalizedLandmark(x=0.45, y=0.35)  # p4
    dy_right = 0.035 if right_open else 0.003
    landmarks[160] = MockNormalizedLandmark(x=0.38, y=0.35 - dy_right)  # p2 top
    landmarks[144] = MockNormalizedLandmark(x=0.38, y=0.35 + dy_right)  # p6 bottom
    landmarks[158] = MockNormalizedLandmark(x=0.42, y=0.35 - dy_right)  # p3 top
    landmarks[153] = MockNormalizedLandmark(x=0.42, y=0.35 + dy_right)  # p5 bottom

    return landmarks


class TestEyeDetection(unittest.TestCase):
    """Test suite for EyeDetector class and mathematical EAR implementation."""

    def setUp(self) -> None:
        self.detector = EyeDetector(ear_threshold=0.25)

    def test_calculate_ear_normal_valid_geometry(self) -> None:
        """Verify EAR computation on known synthetic geometric points.

        Eye geometry:
            p1 = (-15, 0), p4 = (15, 0) -> width = 30
            p2 = (-7.5, 5), p6 = (-7.5, -5) -> v1 = 10
            p3 = (7.5, 5), p5 = (7.5, -5) -> v2 = 10
            Expected EAR = (10 + 10) / (2 * 30) = 20 / 60 = 0.3333...
        """
        eye_pts = np.array([
            [-15.0, 0.0],
            [-7.5, 5.0],
            [7.5, 5.0],
            [15.0, 0.0],
            [7.5, -5.0],
            [-7.5, -5.0],
        ])
        ear = self.detector.calculate_ear(eye_pts)
        self.assertAlmostEqual(ear, 1.0 / 3.0, places=4)

    def test_different_eye_opening_geometries(self) -> None:
        """Verify EAR monotonically decreases as eye aperture narrows."""
        # Wide open: vertical=12, width=30 -> EAR = 24 / 60 = 0.40
        wide_open = np.array([
            [-15.0, 0.0], [-7.5, 6.0], [7.5, 6.0],
            [15.0, 0.0], [7.5, -6.0], [-7.5, -6.0],
        ])
        # Half open: vertical=6, width=30 -> EAR = 12 / 60 = 0.20
        half_open = np.array([
            [-15.0, 0.0], [-7.5, 3.0], [7.5, 3.0],
            [15.0, 0.0], [7.5, -3.0], [-7.5, -3.0],
        ])
        # Fully closed: vertical=0.6, width=30 -> EAR = 1.2 / 60 = 0.02
        closed = np.array([
            [-15.0, 0.0], [-7.5, 0.3], [7.5, 0.3],
            [15.0, 0.0], [7.5, -0.3], [-7.5, -0.3],
        ])

        ear_wide = self.detector.calculate_ear(wide_open)
        ear_half = self.detector.calculate_ear(half_open)
        ear_closed = self.detector.calculate_ear(closed)

        self.assertGreater(ear_wide, ear_half)
        self.assertGreater(ear_half, ear_closed)
        self.assertAlmostEqual(ear_wide, 0.40, places=3)
        self.assertAlmostEqual(ear_half, 0.20, places=3)
        self.assertAlmostEqual(ear_closed, 0.02, places=3)

    def test_invalid_landmarks_handling(self) -> None:
        """Verify detector safely returns 0.0 for corrupted or malformed inputs."""
        # None input
        self.assertEqual(self.detector.calculate_ear(None), 0.0)

        # Insufficient points (< 6)
        insufficient = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.assertEqual(self.detector.calculate_ear(insufficient), 0.0)

        # Non-array input
        self.assertEqual(self.detector.calculate_ear("invalid"), 0.0)

        # NaN values
        nan_pts = np.full((6, 2), np.nan)
        self.assertEqual(self.detector.calculate_ear(nan_pts), 0.0)

        # Infinite values
        inf_pts = np.full((6, 2), np.inf)
        self.assertEqual(self.detector.calculate_ear(inf_pts), 0.0)

    def test_missing_landmarks_in_detect_eyes(self) -> None:
        """Verify detect_eyes returns valid=False for missing or incomplete face landmarks."""
        # None landmarks
        res_none = self.detector.detect_eyes(None, (480, 640))
        self.assertFalse(res_none.valid)
        self.assertEqual(res_none.average_ear, 0.0)

        # Truncated landmark list (< 468 points)
        res_short = self.detector.detect_eyes([MockNormalizedLandmark(0, 0)] * 50, (480, 640))
        self.assertFalse(res_short.valid)

    def test_zero_horizontal_distance(self) -> None:
        """Verify detector handles zero horizontal eye width without division-by-zero errors."""
        # p1 == p4 (collinear degenerate points)
        degenerate = np.zeros((6, 2), dtype=np.float64)
        ear = self.detector.calculate_ear(degenerate)
        self.assertEqual(ear, 0.0)

        # Extremely small horizontal width (< 1e-6)
        near_zero = np.array([
            [0.0, 0.0], [0.0, 5.0], [0.0, 5.0],
            [1e-8, 0.0], [0.0, -5.0], [0.0, -5.0],
        ])
        ear_near_zero = self.detector.calculate_ear(near_zero)
        self.assertEqual(ear_near_zero, 0.0)

    def test_compute_average_ear(self) -> None:
        """Verify bilateral EAR computation and arithmetic mean."""
        pts1 = np.array([
            [-15.0, 0.0], [-7.5, 6.0], [7.5, 6.0],
            [15.0, 0.0], [7.5, -6.0], [-7.5, -6.0],
        ])  # EAR = 0.40
        pts2 = np.array([
            [-15.0, 0.0], [-7.5, 3.0], [7.5, 3.0],
            [15.0, 0.0], [7.5, -3.0], [-7.5, -3.0],
        ])  # EAR = 0.20

        l_ear, r_ear, avg_ear = self.detector.compute_average_ear(pts1, pts2)
        self.assertAlmostEqual(l_ear, 0.40, places=3)
        self.assertAlmostEqual(r_ear, 0.20, places=3)
        self.assertAlmostEqual(avg_ear, 0.30, places=3)

    def test_detect_eyes_with_mock_landmarks(self) -> None:
        """Verify end-to-end landmark extraction from mock landmark objects."""
        mock_landmarks = create_mock_eye_landmarks(left_open=True, right_open=True)
        res = self.detector.detect_eyes(mock_landmarks, (480, 640))

        self.assertTrue(res.valid)
        self.assertGreater(res.left_ear, 0.20)
        self.assertGreater(res.right_ear, 0.20)
        self.assertAlmostEqual(res.average_ear, (res.left_ear + res.right_ear) / 2.0)

        # Test tuple unpacking: (left, right, avg)
        l, r, avg = res
        self.assertEqual(l, res.left_ear)
        self.assertEqual(r, res.right_ear)
        self.assertEqual(avg, res.average_ear)

    def test_is_eye_closed_helper(self) -> None:
        """Verify is_eye_closed correctly checks thresholds."""
        self.assertTrue(self.detector.is_eye_closed(0.15))
        self.assertFalse(self.detector.is_eye_closed(0.35))
        self.assertTrue(self.detector.is_eye_closed(0.19, threshold=0.20))
        self.assertFalse(self.detector.is_eye_closed(0.22, threshold=0.20))


if __name__ == "__main__":
    unittest.main()

