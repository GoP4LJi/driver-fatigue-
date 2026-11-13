"""Unit tests for Head Pose Estimation (Pitch, Yaw, Roll) using solvePnP.

Validates 3D-to-2D perspective estimation, coordinate conventions, angle normalization,
graceful handling of solvePnP failures, and invalid landmark geometries without hardware.
"""

from dataclasses import dataclass
from typing import List
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from detection.head_pose import HeadPoseEstimator, HeadPoseResult


@dataclass
class MockNormalizedLandmark:
    """Mock structure mimicking MediaPipe's NormalizedLandmark."""
    x: float
    y: float
    z: float = 0.0


def create_mock_head_landmarks(
    pose_type: str = "neutral",
    frame_shape: tuple = (480, 640),
) -> List[MockNormalizedLandmark]:
    """Create a 478-length list of mock landmarks for head pose testing.

    Indices: [1 (Nose), 199 (Chin), 33 (RightEye/viewer-left), 263 (LeftEye/viewer-right),
              61 (RightMouth/viewer-left), 291 (LeftMouth/viewer-right)]
    """
    landmarks = [MockNormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    h, w = frame_shape

    # Baseline 2D pixel coordinates for neutral face looking straight at camera
    pts = {
        1: (320.0, 240.0),    # Nose tip
        199: (320.0, 340.0),  # Chin
        33: (250.0, 190.0),   # Right eye (viewer left)
        263: (390.0, 190.0),  # Left eye (viewer right)
        61: (270.0, 290.0),   # Right mouth (viewer left)
        291: (370.0, 290.0),  # Left mouth (viewer right)
    }

    if pose_type == "nod_down":
        # Nose and chin shift downwards
        pts[1] = (320.0, 270.0)
        pts[199] = (320.0, 350.0)
    elif pose_type == "turn_right":
        # Driver turns right -> facial features shift towards viewer left
        pts[1] = (290.0, 240.0)
        pts[199] = (300.0, 340.0)
        pts[33] = (230.0, 190.0)
        pts[263] = (360.0, 190.0)
    elif pose_type == "tilt_clockwise":
        # Roll: rotate points slightly around center (320, 240)
        angle = np.radians(15.0)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for idx, (px, py) in list(pts.items()):
            dx, dy = px - 320.0, py - 240.0
            rx = 320.0 + dx * cos_a - dy * sin_a
            ry = 240.0 + dx * sin_a + dy * cos_a
            pts[idx] = (rx, ry)

    for idx, (px, py) in pts.items():
        landmarks[idx] = MockNormalizedLandmark(x=px / float(w), y=py / float(h))

    return landmarks


class TestHeadPoseEstimation(unittest.TestCase):
    """Test suite for HeadPoseEstimator and solvePnP calculation."""

    def setUp(self) -> None:
        self.estimator = HeadPoseEstimator()
        self.frame_shape = (480, 640)

    def test_estimate_head_pose_neutral_face(self) -> None:
        """Verify neutral face looking straight at camera produces near-zero angles."""
        landmarks = create_mock_head_landmarks("neutral", self.frame_shape)
        result = self.estimator.estimate_head_pose(landmarks, self.frame_shape)

        self.assertTrue(result.valid)
        self.assertIsNotNone(result.rvec)
        self.assertIsNotNone(result.tvec)

        # Neutral face angles should be within +/- 10 degrees
        self.assertLess(abs(result.pitch), 10.0)
        self.assertLess(abs(result.yaw), 5.0)
        self.assertLess(abs(result.roll), 5.0)

        # Test tuple-like unpacking
        pitch, yaw, roll = result
        self.assertEqual(pitch, result.pitch)
        self.assertEqual(yaw, result.yaw)
        self.assertEqual(roll, result.roll)

    def test_estimate_head_pose_nod_down(self) -> None:
        """Verify nodding down generates positive pitch deviation."""
        landmarks = create_mock_head_landmarks("nod_down", self.frame_shape)
        result = self.estimator.estimate_head_pose(landmarks, self.frame_shape)

        self.assertTrue(result.valid)
        self.assertGreater(result.pitch, 5.0)

    def test_estimate_head_pose_turn_right(self) -> None:
        """Verify turning right generates positive yaw deviation."""
        landmarks = create_mock_head_landmarks("turn_right", self.frame_shape)
        result = self.estimator.estimate_head_pose(landmarks, self.frame_shape)

        self.assertTrue(result.valid)
        self.assertGreater(result.yaw, 5.0)

    def test_solvepnp_failure_handled_gracefully(self) -> None:
        """Verify solver failure returns valid=False without throwing unhandled exceptions."""
        with patch("detection.head_pose.cv2.solvePnP", return_value=(False, None, None)):
            landmarks = create_mock_head_landmarks("neutral", self.frame_shape)
            result = self.estimator.estimate_head_pose(landmarks, self.frame_shape)

            self.assertFalse(result.valid)
            self.assertEqual(result.pitch, 0.0)
            self.assertEqual(result.yaw, 0.0)
            self.assertEqual(result.roll, 0.0)

    def test_solvepnp_exception_handled_gracefully(self) -> None:
        """Verify internal OpenCV exception returns valid=False."""
        with patch("detection.head_pose.cv2.solvePnP", side_effect=RuntimeError("OpenCV error")):
            landmarks = create_mock_head_landmarks("neutral", self.frame_shape)
            result = self.estimator.estimate_head_pose(landmarks, self.frame_shape)

            self.assertFalse(result.valid)

    def test_invalid_landmark_inputs(self) -> None:
        """Verify detector handles None, truncated, or NaN landmarks safely."""
        # None landmarks
        res_none = self.estimator.estimate_head_pose(None, self.frame_shape)
        self.assertFalse(res_none.valid)

        # Insufficient points (< 291)
        res_short = self.estimator.estimate_head_pose([MockNormalizedLandmark(0, 0)] * 50, self.frame_shape)
        self.assertFalse(res_short.valid)

        # Array with NaNs
        nan_pts = np.full((6, 2), np.nan)
        res_nan = self.estimator.estimate_head_pose(nan_pts, self.frame_shape)
        self.assertFalse(res_nan.valid)

    def test_invalid_frame_dimensions(self) -> None:
        """Verify detector handles zero, negative, or malformed frame shapes safely."""
        landmarks = create_mock_head_landmarks("neutral", self.frame_shape)

        # Zero dimensions
        self.assertFalse(self.estimator.estimate_head_pose(landmarks, (0, 0)).valid)

        # Negative dimensions
        self.assertFalse(self.estimator.estimate_head_pose(landmarks, (-480, 640)).valid)

        # Malformed tuple
        self.assertFalse(self.estimator.estimate_head_pose(landmarks, (480,)).valid)

    def test_get_camera_matrix(self) -> None:
        """Verify camera matrix computation adapts to frame dimensions."""
        cam_matrix, dist_coeffs = self.estimator.get_camera_matrix((480, 640))
        self.assertEqual(cam_matrix.shape, (3, 3))
        self.assertEqual(dist_coeffs.shape, (4, 1))

        # Focal length = 640, center = (320, 240)
        self.assertAlmostEqual(cam_matrix[0, 0], 640.0)
        self.assertAlmostEqual(cam_matrix[1, 1], 640.0)
        self.assertAlmostEqual(cam_matrix[0, 2], 320.0)
        self.assertAlmostEqual(cam_matrix[1, 2], 240.0)

    def test_is_distracted_helper(self) -> None:
        """Verify is_distracted correctly flags angular deviations."""
        # Within limits
        self.assertFalse(self.estimator.is_distracted(pitch=5.0, yaw=5.0, roll=5.0))

        # Pitch exceeded (nodding down > 15 deg)
        self.assertTrue(self.estimator.is_distracted(pitch=22.0, yaw=0.0, roll=0.0))

        # Yaw exceeded (turned looking sideways > 20 deg)
        self.assertTrue(self.estimator.is_distracted(pitch=0.0, yaw=28.0, roll=0.0))

        # Custom thresholds
        custom_th = {"pitch": 10.0, "yaw": 10.0, "roll": 10.0}
        self.assertTrue(self.estimator.is_distracted(pitch=12.0, yaw=0.0, roll=0.0, thresholds=custom_th))


if __name__ == "__main__":
    unittest.main()

