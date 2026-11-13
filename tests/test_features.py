"""Unit tests for the feature extraction and DriverFeatures data representations.

Validates the aggregation of Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
and 3D Head Pose (solvePnP) into unified DriverFeatures data objects, dictionary conversions,
and NumPy matrix serializations for machine learning ingestion.
"""

from dataclasses import dataclass
from typing import List
import unittest
import numpy as np

from features.feature_extraction import DriverFeatures, FeatureExtractor


@dataclass
class MockNormalizedLandmark:
    """Mock structure mimicking MediaPipe's NormalizedLandmark."""
    x: float
    y: float
    z: float = 0.0


def create_full_mock_face_landmarks(frame_shape: tuple = (480, 640)) -> List[MockNormalizedLandmark]:
    """Create a complete 478-length list of mock landmarks for integrated feature extraction."""
    landmarks = [MockNormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    h, w = frame_shape

    # 1. Eye landmarks (left & right open)
    # Left eye: (362, 385, 387, 263, 373, 380)
    landmarks[362] = MockNormalizedLandmark(x=0.55, y=0.35)
    landmarks[263] = MockNormalizedLandmark(x=0.65, y=0.35)
    landmarks[385] = MockNormalizedLandmark(x=0.58, y=0.32)
    landmarks[380] = MockNormalizedLandmark(x=0.58, y=0.38)
    landmarks[387] = MockNormalizedLandmark(x=0.62, y=0.32)
    landmarks[373] = MockNormalizedLandmark(x=0.62, y=0.38)

    # Right eye: (33, 160, 158, 133, 153, 144)
    landmarks[33] = MockNormalizedLandmark(x=0.35, y=0.35)
    landmarks[133] = MockNormalizedLandmark(x=0.45, y=0.35)
    landmarks[160] = MockNormalizedLandmark(x=0.38, y=0.32)
    landmarks[144] = MockNormalizedLandmark(x=0.38, y=0.38)
    landmarks[158] = MockNormalizedLandmark(x=0.42, y=0.32)
    landmarks[153] = MockNormalizedLandmark(x=0.42, y=0.38)

    # 2. Mouth landmarks
    # MOUTH_TOP=13, MOUTH_BOTTOM=14, MOUTH_LEFT=78, MOUTH_RIGHT=308
    landmarks[78] = MockNormalizedLandmark(x=0.42, y=0.65)
    landmarks[308] = MockNormalizedLandmark(x=0.58, y=0.65)
    landmarks[13] = MockNormalizedLandmark(x=0.50, y=0.63)
    landmarks[14] = MockNormalizedLandmark(x=0.50, y=0.67)

    # 3. Head pose landmarks (1, 199, 33, 263, 61, 291)
    landmarks[1] = MockNormalizedLandmark(x=320.0 / w, y=240.0 / h)    # Nose
    landmarks[199] = MockNormalizedLandmark(x=320.0 / w, y=340.0 / h)  # Chin
    landmarks[61] = MockNormalizedLandmark(x=270.0 / w, y=290.0 / h)   # Mouth left
    landmarks[291] = MockNormalizedLandmark(x=370.0 / w, y=290.0 / h)  # Mouth right

    return landmarks


class TestFeatures(unittest.TestCase):
    """Test suite verifying DriverFeatures data structures and FeatureExtractor aggregation."""

    def setUp(self) -> None:
        """Create sample features vector."""
        self.features = DriverFeatures(
            timestamp=1700000000.0,
            left_ear=0.28,
            right_ear=0.28,
            average_ear=0.28,
            mar=0.15,
            pitch=2.5,
            yaw=-1.0,
            roll=0.5,
            face_detected=True,
            valid=True,
            ear=0.28,
        )
        self.extractor = FeatureExtractor()

    def test_features_to_dict(self) -> None:
        """Verify dictionary conversion contains all required keys and correct values."""
        d = self.features.to_dict()
        required_keys = {
            "timestamp",
            "left_ear",
            "right_ear",
            "average_ear",
            "mar",
            "pitch",
            "yaw",
            "roll",
            "face_detected",
            "valid",
            "ear",
        }
        self.assertTrue(required_keys.issubset(set(d.keys())))
        self.assertAlmostEqual(d["average_ear"], 0.28)
        self.assertAlmostEqual(d["mar"], 0.15)
        self.assertAlmostEqual(d["pitch"], 2.5)
        self.assertTrue(d["face_detected"])
        self.assertTrue(d["valid"])

    def test_features_to_array_default_order(self) -> None:
        """Verify vector shape and numerical order for ML model ingestion."""
        arr = self.features.to_array()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (5,))
        self.assertEqual(arr.dtype, np.float32)
        np.testing.assert_allclose(arr, [0.28, 0.15, 2.5, -1.0, 0.5])

    def test_features_to_array_custom_order(self) -> None:
        """Verify extraction with custom specified feature order."""
        arr = self.features.to_array(["mar", "average_ear", "roll"])
        self.assertEqual(arr.shape, (3,))
        np.testing.assert_allclose(arr, [0.15, 0.28, 0.5])

    def test_extract_features_valid_face(self) -> None:
        """Verify FeatureExtractor combines eye, mouth, and pose metrics into DriverFeatures."""
        landmarks = create_full_mock_face_landmarks((480, 640))
        features = self.extractor.extract_features(landmarks, (480, 640), timestamp=12345.0)

        self.assertIsInstance(features, DriverFeatures)
        self.assertTrue(features.face_detected)
        self.assertTrue(features.valid)
        self.assertEqual(features.timestamp, 12345.0)

        # Check reasonable physiological ranges
        self.assertGreater(features.average_ear, 0.15)
        self.assertGreater(features.mar, 0.05)
        self.assertEqual(features.ear, features.average_ear)

    def test_extract_features_missing_face(self) -> None:
        """Verify extractor handles missing face (None input) safely without exceptions."""
        features = self.extractor.extract_features(None, (480, 640), timestamp=100.0)

        self.assertFalse(features.face_detected)
        self.assertFalse(features.valid)
        self.assertEqual(features.timestamp, 100.0)
        self.assertEqual(features.left_ear, 0.0)
        self.assertEqual(features.right_ear, 0.0)
        self.assertEqual(features.average_ear, 0.0)
        self.assertEqual(features.mar, 0.0)
        self.assertEqual(features.pitch, 0.0)
        self.assertEqual(features.yaw, 0.0)
        self.assertEqual(features.roll, 0.0)

    def test_extract_features_truncated_landmarks(self) -> None:
        """Verify extractor handles truncated/invalid landmark collections."""
        features = self.extractor.extract_features([MockNormalizedLandmark(0, 0)] * 50, (480, 640))
        self.assertFalse(features.face_detected)
        self.assertFalse(features.valid)

    def test_driver_features_post_init_alias(self) -> None:
        """Verify ear and average_ear auto-synchronize in DriverFeatures."""
        f1 = DriverFeatures(average_ear=0.33)
        self.assertEqual(f1.ear, 0.33)

        f2 = DriverFeatures(ear=0.22)
        self.assertEqual(f2.average_ear, 0.22)


if __name__ == "__main__":
    unittest.main()
