"""Unit tests for Phase 2 MediaPipe Face Landmarker detection and visualization.

Tests verify detector initialization, model loading verification, coordinate transformations (2D/3D),
eye and mouth landmark extractions, missing face handling, monotonic timestamp enforcement,
and context manager lifecycle without requiring a physical camera device.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from config.settings import LandmarkIndices, settings
from detection.face_detection import FaceDetectionResult, FaceDetector
from visualization.draw_utils import Visualizer


@dataclass
class MockNormalizedLandmark:
    """Mock structure mimicking MediaPipe's NormalizedLandmark."""
    x: float
    y: float
    z: float = 0.0


def create_mock_landmarks(count: int = 478) -> List[MockNormalizedLandmark]:
    """Generate a list of mock landmarks spanning normalized space [0.1, 0.9]."""
    landmarks = []
    for i in range(count):
        # Deterministic coordinates centered around facial region
        x = 0.3 + 0.4 * (i % 20) / 20.0
        y = 0.3 + 0.4 * ((i // 20) % 20) / 20.0
        z = -0.05 + 0.1 * (i / float(count))
        landmarks.append(MockNormalizedLandmark(x=x, y=y, z=z))
    return landmarks


class TestFaceDetectorInitialization(unittest.TestCase):
    """Test FaceDetector configuration and model asset validation."""

    def test_default_initialization(self) -> None:
        """Verify FaceDetector initializes with central settings values."""
        detector = FaceDetector()
        self.assertEqual(detector.model_path, settings.mediapipe.model_path)
        self.assertEqual(detector.max_num_faces, settings.mediapipe.max_num_faces)
        self.assertEqual(detector.min_detection_confidence, settings.mediapipe.min_face_detection_confidence)
        self.assertEqual(detector.min_presence_confidence, settings.mediapipe.min_face_presence_confidence)
        self.assertEqual(detector.min_tracking_confidence, settings.mediapipe.min_tracking_confidence)
        self.assertIsNone(detector._landmarker)

    def test_custom_initialization(self) -> None:
        """Verify FaceDetector accepts custom parameters."""
        custom_path = Path("/tmp/custom_model.task")
        detector = FaceDetector(
            model_path=custom_path,
            max_num_faces=2,
            min_detection_confidence=0.8,
            min_presence_confidence=0.7,
            min_tracking_confidence=0.9,
        )
        self.assertEqual(detector.model_path, custom_path)
        self.assertEqual(detector.max_num_faces, 2)
        self.assertEqual(detector.min_detection_confidence, 0.8)
        self.assertEqual(detector.min_presence_confidence, 0.7)
        self.assertEqual(detector.min_tracking_confidence, 0.9)

    def test_missing_model_file_raises_error(self) -> None:
        """Verify initialize() raises FileNotFoundError if model file does not exist."""
        nonexistent = Path("/path/to/nonexistent/face_landmarker.task")
        detector = FaceDetector(model_path=nonexistent)
        with self.assertRaises(FileNotFoundError):
            detector.initialize()

    @patch("detection.face_detection.mp", None)
    def test_missing_mediapipe_module_raises_error(self) -> None:
        """Verify initialize() raises ImportError when MediaPipe package is unavailable."""
        detector = FaceDetector()
        with self.assertRaises(ImportError):
            detector.initialize()


class TestFaceDetectorInference(unittest.TestCase):
    """Test frame processing, invalid inputs, and detection handling."""

    def setUp(self) -> None:
        self.detector = FaceDetector()

    def tearDown(self) -> None:
        self.detector.close()

    def test_detect_landmarks_empty_or_invalid_frame(self) -> None:
        """Verify invalid or empty frames return has_face=False without throwing."""
        # Test None
        res_none = self.detector.detect_landmarks(None)
        self.assertFalse(res_none.has_face)
        self.assertIsNone(res_none.landmarks)
        self.assertEqual(res_none.num_landmarks, 0)

        # Test empty array
        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
        res_empty = self.detector.detect_landmarks(empty_frame)
        self.assertFalse(res_empty.has_face)
        self.assertIsNone(res_empty.landmarks)

    def test_monotonic_timestamps_enforced(self) -> None:
        """Verify video mode timestamps are strictly monotonically increasing."""
        # Mock _landmarker to avoid invoking real native tflite delegate
        mock_landmarker = MagicMock()
        mock_result = MagicMock()
        mock_result.face_landmarks = [create_mock_landmarks(478)]
        mock_landmarker.detect_for_video.return_value = mock_result
        self.detector._landmarker = mock_landmarker

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # First call with timestamp 1000
        res1 = self.detector.detect_landmarks(frame, timestamp_ms=1000)
        self.assertEqual(res1.timestamp_ms, 1000)

        # Second call with same or earlier timestamp (e.g. 1000 or 999)
        res2 = self.detector.detect_landmarks(frame, timestamp_ms=1000)
        self.assertGreater(res2.timestamp_ms, 1000)

        res3 = self.detector.detect_landmarks(frame, timestamp_ms=500)
        self.assertGreater(res3.timestamp_ms, res2.timestamp_ms)

    def test_detect_landmarks_face_detected(self) -> None:
        """Verify FaceDetectionResult structure when face landmarks are detected."""
        mock_landmarker = MagicMock()
        mock_landmarks = create_mock_landmarks(478)
        mock_res = MagicMock()
        mock_res.face_landmarks = [mock_landmarks]
        mock_landmarker.detect_for_video.return_value = mock_res
        self.detector._landmarker = mock_landmarker

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect_landmarks(frame, timestamp_ms=100)

        self.assertTrue(result.has_face)
        self.assertEqual(result.num_landmarks, 478)
        self.assertEqual(len(result.landmarks), 478)
        self.assertEqual(result.timestamp_ms, 100)

    def test_detect_landmarks_no_face_detected(self) -> None:
        """Verify FaceDetectionResult structure when no face is found."""
        mock_landmarker = MagicMock()
        mock_res = MagicMock()
        mock_res.face_landmarks = []  # No faces found
        mock_landmarker.detect_for_video.return_value = mock_res
        self.detector._landmarker = mock_landmarker

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect_landmarks(frame, timestamp_ms=150)

        self.assertFalse(result.has_face)
        self.assertIsNone(result.landmarks)
        self.assertEqual(result.num_landmarks, 0)


class TestLandmarkCoordinateExtraction(unittest.TestCase):
    """Test 2D pixel projection and 3D coordinate conversions."""

    def setUp(self) -> None:
        self.detector = FaceDetector()
        self.mock_landmarks = [
            MockNormalizedLandmark(x=0.0, y=0.0, z=0.0),       # Top-left
            MockNormalizedLandmark(x=0.5, y=0.5, z=-0.1),      # Center
            MockNormalizedLandmark(x=1.0, y=1.0, z=0.2),       # Bottom-right
            MockNormalizedLandmark(x=-0.2, y=1.5, z=0.0),      # Out-of-bounds
        ]

    def test_get_landmark_points_2d(self) -> None:
        """Verify 2D pixel conversion with boundary clamping."""
        shape = (480, 640, 3)  # H=480, W=640
        pts = self.detector.get_landmark_points_2d(self.mock_landmarks, shape)

        self.assertEqual(pts.shape, (4, 2))
        self.assertEqual(pts.dtype, np.int32)

        # Landmark 0: (0.0*640, 0.0*480) -> [0, 0]
        self.assertEqual(list(pts[0]), [0, 0])
        # Landmark 1: (0.5*640, 0.5*480) -> [320, 240]
        self.assertEqual(list(pts[1]), [320, 240])
        # Landmark 2: (1.0*640, 1.0*480) -> clamped to [639, 479]
        self.assertEqual(list(pts[2]), [639, 479])
        # Landmark 3: Clamped out-of-bounds -> [0, 479]
        self.assertEqual(list(pts[3]), [0, 479])

    def test_get_landmark_points_2d_with_indices(self) -> None:
        """Verify extracting a specific subset of landmark indices."""
        shape = (100, 200, 3)
        pts = self.detector.get_landmark_points_2d(self.mock_landmarks, shape, indices=[1, 2])
        self.assertEqual(len(pts), 2)
        self.assertEqual(list(pts[0]), [100, 50])

    def test_get_landmark_points_2d_empty_or_none(self) -> None:
        """Verify None or empty landmarks returns empty (0, 2) array."""
        pts = self.detector.get_landmark_points_2d(None, (480, 640, 3))
        self.assertEqual(pts.shape, (0, 2))

        pts_empty = self.detector.get_landmark_points_2d([], (480, 640, 3))
        self.assertEqual(pts_empty.shape, (0, 2))

    def test_get_landmark_points_3d(self) -> None:
        """Verify 3D coordinate conversion with pixel scaling."""
        shape = (480, 640, 3)  # W=640
        pts = self.detector.get_landmark_points_3d(self.mock_landmarks, shape)

        self.assertEqual(pts.shape, (4, 3))
        self.assertEqual(pts.dtype, np.float64)

        # Landmark 1: x=0.5*640=320, y=0.5*480=240, z=-0.1*640=-64.0
        self.assertAlmostEqual(pts[1][0], 320.0)
        self.assertAlmostEqual(pts[1][1], 240.0)
        self.assertAlmostEqual(pts[1][2], -64.0)

    def test_get_landmark_points_3d_empty_or_none(self) -> None:
        """Verify None or empty landmarks returns empty (0, 3) array."""
        pts = self.detector.get_landmark_points_3d(None, (480, 640, 3))
        self.assertEqual(pts.shape, (0, 3))

    def test_eye_and_mouth_region_helpers(self) -> None:
        """Verify left eye, right eye, and mouth region extractors."""
        landmarks = create_mock_landmarks(478)
        shape = (480, 640, 3)

        left_eye_pts = self.detector.get_left_eye_landmarks_2d(landmarks, shape)
        right_eye_pts = self.detector.get_right_eye_landmarks_2d(landmarks, shape)
        mouth_pts = self.detector.get_mouth_landmarks_2d(landmarks, shape)

        self.assertEqual(len(left_eye_pts), len(settings.landmarks.LEFT_EYE))
        self.assertEqual(len(right_eye_pts), len(settings.landmarks.RIGHT_EYE))
        self.assertEqual(len(mouth_pts), len(settings.landmarks.MOUTH))


class TestFaceDetectorLifecycle(unittest.TestCase):
    """Test resource cleanup and context manager behavior."""

    def test_close_releases_landmarker(self) -> None:
        """Verify close() terminates the underlying landmarker instance."""
        detector = FaceDetector()
        mock_landmarker = MagicMock()
        detector._landmarker = mock_landmarker

        detector.close()
        mock_landmarker.close.assert_called_once()
        self.assertIsNone(detector._landmarker)

        # Verify calling close again is safe
        detector.close()

    @patch("detection.face_detection.vision.FaceLandmarker.create_from_options")
    def test_context_manager_lifecycle(self, mock_create: MagicMock) -> None:
        """Verify context manager handles initialization and release automatically."""
        mock_instance = MagicMock()
        mock_create.return_value = mock_instance

        detector = FaceDetector()
        with detector as det:
            self.assertIsNotNone(det._landmarker)

        mock_instance.close.assert_called_once()
        self.assertIsNone(detector._landmarker)


class TestVisualizerLandmarks(unittest.TestCase):
    """Test landmark rendering and face-not-detected banner functions."""

    def setUp(self) -> None:
        self.visualizer = Visualizer()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_landmarks_with_mock_landmarks(self) -> None:
        """Verify draw_landmarks renders points and contours without throwing."""
        landmarks = create_mock_landmarks(478)
        annotated = self.visualizer.draw_landmarks(
            self.test_frame.copy(),
            landmarks,
            draw_mesh=True,
            draw_contours=True,
            draw_eyes=True,
            draw_mouth=True,
            draw_irises=True,
        )

        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, self.test_frame.shape)
        # Verify pixels were modified (non-zero)
        self.assertGreater(np.sum(annotated), 0)

    def test_draw_landmarks_safe_with_none(self) -> None:
        """Verify draw_landmarks handles None landmarks gracefully."""
        frame_copy = self.test_frame.copy()
        result = self.visualizer.draw_landmarks(frame_copy, None)
        self.assertTrue(np.array_equal(frame_copy, result))

    def test_draw_face_not_detected_banner(self) -> None:
        """Verify face not detected warning banner renders on frame."""
        frame_copy = self.test_frame.copy()
        result = self.visualizer.draw_face_not_detected_banner(frame_copy)
        self.assertIsNotNone(result)
        self.assertGreater(np.sum(result), 0)


if __name__ == "__main__":
    unittest.main()

