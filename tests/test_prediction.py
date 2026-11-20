"""Unit tests for online model prediction (models/predict.py).

Validates model loading, graceful degradation when model file is missing,
real-time prediction from DriverFeatures, class probability distributions,
and handling of invalid/missing face features.
"""

from pathlib import Path
import tempfile
import unittest
from config.settings import DriverState
from features.feature_extraction import DriverFeatures
from models.predict import StatePredictor
from models.train_model import ModelTrainer, generate_synthetic_testing_data


class TestPrediction(unittest.TestCase):
    """Test suite for StatePredictor inference and error handling."""

    @classmethod
    def setUpClass(cls) -> None:
        """Train a temporary lightweight model for testing real predictions."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.temp_dir.name) / "test_model.joblib"

        df = generate_synthetic_testing_data(n_samples=150, random_state=42)
        trainer = ModelTrainer(model_type="random_forest", random_state=42)
        model = trainer.train(df[trainer.feature_columns], df[trainer.target_column])
        trainer.save_model(model, cls.model_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_missing_model_file_safe_fallback(self) -> None:
        """Verify predictor degrades safely when model file does not exist."""
        nonexistent = Path("/path/to/missing_model.joblib")
        predictor = StatePredictor(model_path=nonexistent, auto_load=True)

        self.assertFalse(predictor.is_loaded())

        # Test predict returns safe fallback without throwing
        features = DriverFeatures(average_ear=0.30, mar=0.20, face_detected=True, valid=True)
        state, probs = predictor.predict(features)
        self.assertEqual(state, DriverState.ALERT)
        self.assertEqual(probs[DriverState.ALERT.value], 1.0)

    def test_load_model_explicit_nonexistent_raises_error(self) -> None:
        """Verify load_model explicitly called with nonexistent path raises FileNotFoundError."""
        predictor = StatePredictor(auto_load=False)
        with self.assertRaises(FileNotFoundError):
            predictor.load_model(Path("/invalid/path/model.joblib"))

    def test_load_and_predict_valid_model(self) -> None:
        """Verify loading pre-trained artifact and performing inference on features."""
        predictor = StatePredictor(model_path=self.model_path)
        self.assertTrue(predictor.is_loaded())

        # Test alert driver features
        alert_f = DriverFeatures(
            average_ear=0.35,
            mar=0.15,
            pitch=1.0,
            yaw=0.0,
            roll=0.0,
            face_detected=True,
            valid=True,
        )
        state, probs = predictor.predict(alert_f)
        self.assertIsInstance(state, DriverState)
        self.assertIsInstance(probs, dict)
        self.assertEqual(set(probs.keys()), {s.value for s in DriverState})
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)

    def test_predict_drowsy_features(self) -> None:
        """Verify closed-eye features result in high DROWSY probability."""
        predictor = StatePredictor(model_path=self.model_path)
        drowsy_f = DriverFeatures(
            average_ear=0.12,  # Strongly closed eyes
            mar=0.15,
            pitch=20.0,
            yaw=0.0,
            roll=0.0,
            face_detected=True,
            valid=True,
        )
        state, probs = predictor.predict(drowsy_f)
        self.assertIn(state, [DriverState.DROWSY, DriverState.ALERT, DriverState.DISTRACTED])
        self.assertGreater(probs[DriverState.DROWSY.value], 0.30)

    def test_predict_missing_face_returns_safe_fallback(self) -> None:
        """Verify missing face input (face_detected=False) returns safe baseline distribution."""
        predictor = StatePredictor(model_path=self.model_path)
        missing_f = DriverFeatures(face_detected=False, valid=False)

        state, probs = predictor.predict(missing_f)
        self.assertEqual(state, DriverState.ALERT)
        self.assertEqual(probs[DriverState.ALERT.value], 1.0)


if __name__ == "__main__":
    unittest.main()
