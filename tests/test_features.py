"""Unit tests for the feature extraction and data representations."""

import unittest
import numpy as np
from features.feature_extraction import DriverFeatures, FeatureExtractor


class TestFeatures(unittest.TestCase):
    """Test suite verifying DriverFeatures data structures and transformation helpers."""

    def setUp(self) -> None:
        """Create sample features vector."""
        self.features = DriverFeatures(
            ear=0.28,
            mar=0.15,
            pitch=2.5,
            yaw=-1.0,
            roll=0.5,
            timestamp=1700000000.0,
        )

    def test_features_to_dict(self) -> None:
        """Verify dictionary conversion contains all required keys."""
        d = self.features.to_dict()
        expected_keys = {"ear", "mar", "pitch", "yaw", "roll", "timestamp"}
        self.assertEqual(set(d.keys()), expected_keys)
        self.assertAlmostEqual(d["ear"], 0.28)
        self.assertAlmostEqual(d["mar"], 0.15)

    def test_features_to_array(self) -> None:
        """Verify vector shape and numerical order for ML model ingestion."""
        arr = self.features.to_array(["ear", "mar", "pitch", "yaw", "roll"])
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (5,))
        np.testing.assert_allclose(arr, [0.28, 0.15, 2.5, -1.0, 0.5])

    def test_feature_extractor_stub(self) -> None:
        """Verify extractor raises NotImplementedError prior to Phase 3."""
        extractor = FeatureExtractor()
        with self.assertRaises(NotImplementedError):
            extractor.extract_features(landmarks=None, frame_shape=(480, 640))


if __name__ == "__main__":
    unittest.main()

