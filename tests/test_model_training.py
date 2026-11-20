"""Unit tests for the Machine Learning training pipeline (models/train_model.py).

Validates synthetic dataset generation, dataset loading, data validation,
model training (Random Forest, SVM, Gradient Boosting), performance metric calculations,
artifact serialization, and full pipeline execution.
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from config.settings import DriverState
from models.train_model import ModelTrainer, generate_synthetic_testing_data


class TestModelTraining(unittest.TestCase):
    """Test suite for ModelTrainer and synthetic dataset generator."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_synthetic_dataset_generator(self) -> None:
        """Verify synthetic dataset generator produces well-formed data for test harness."""
        csv_path = self.temp_path / "test_synthetic.csv"
        df = generate_synthetic_testing_data(output_path=csv_path, n_samples=90, random_state=42)

        self.assertTrue(csv_path.exists())
        self.assertEqual(len(df), 90)
        expected_cols = {"timestamp", "ear", "mar", "pitch", "yaw", "roll", "state"}
        self.assertEqual(set(df.columns), expected_cols)

        # Check all 3 target classes are present
        unique_states = set(df["state"].unique())
        self.assertEqual(unique_states, {s.value for s in DriverState})

        # Check numerical sanity
        self.assertTrue((df["ear"] >= 0.0).all())
        self.assertTrue((df["mar"] >= 0.0).all())
        self.assertFalse(df.isnull().any().any())

    def test_trainer_initialization_supported_and_unsupported(self) -> None:
        """Verify ModelTrainer accepts valid algorithms and rejects invalid strings."""
        for algo in ("random_forest", "svm", "gradient_boosting"):
            trainer = ModelTrainer(model_type=algo)
            self.assertEqual(trainer.model_type, algo)

        with self.assertRaises(ValueError):
            ModelTrainer(model_type="unsupported_deep_net")

    def test_load_dataset_missing_file_raises_error(self) -> None:
        """Verify load_dataset raises FileNotFoundError on missing file."""
        trainer = ModelTrainer()
        with self.assertRaises(FileNotFoundError):
            trainer.load_dataset(self.temp_path / "nonexistent.csv")

    def test_load_dataset_missing_columns_raises_error(self) -> None:
        """Verify load_dataset raises ValueError when required columns are absent."""
        bad_csv = self.temp_path / "bad_data.csv"
        pd.DataFrame({"ear": [0.3], "mar": [0.2]}).to_csv(bad_csv, index=False)

        trainer = ModelTrainer()
        with self.assertRaises(ValueError):
            trainer.load_dataset(bad_csv)

    def test_load_dataset_empty_file_raises_error(self) -> None:
        """Verify load_dataset raises ValueError on empty CSV."""
        empty_csv = self.temp_path / "empty.csv"
        empty_csv.write_text("timestamp,ear,mar,pitch,yaw,roll,state\n")

        trainer = ModelTrainer()
        with self.assertRaises(ValueError):
            trainer.load_dataset(empty_csv)

    def test_train_and_evaluate_random_forest(self) -> None:
        """Verify fitting and evaluating Random Forest model on sample data."""
        df = generate_synthetic_testing_data(n_samples=120, random_state=123)
        trainer = ModelTrainer(model_type="random_forest", random_state=123)

        X = df[trainer.feature_columns]
        y = df[trainer.target_column]

        model = trainer.train(X, y)
        self.assertIsNotNone(model)

        metrics = trainer.evaluate(X, y)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("classification_report", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.70)

    def test_save_model_and_artifact_integrity(self) -> None:
        """Verify model serialization persists all metadata and can be inspected."""
        df = generate_synthetic_testing_data(n_samples=60, random_state=42)
        trainer = ModelTrainer(model_type="random_forest")
        X = df[trainer.feature_columns]
        y = df[trainer.target_column]
        model = trainer.train(X, y)

        artifact_path = self.temp_path / "model.joblib"
        trainer.save_model(model, artifact_path)
        self.assertTrue(artifact_path.exists())

        import joblib
        loaded = joblib.load(artifact_path)
        self.assertIsInstance(loaded, dict)
        self.assertIn("model", loaded)
        self.assertIn("feature_columns", loaded)
        self.assertIn("classes", loaded)

    def test_end_to_end_run_pipeline(self) -> None:
        """Verify run_pipeline executes dataset loading, train/test split, evaluation, and saving."""
        data_path = self.temp_path / "dataset.csv"
        generate_synthetic_testing_data(output_path=data_path, n_samples=150, random_state=99)

        model_path = self.temp_path / "output_model.joblib"
        trainer = ModelTrainer(model_type="random_forest", random_state=99)

        summary = trainer.run_pipeline(dataset_path=data_path, output_model_path=model_path, test_size=0.25)
        self.assertTrue(model_path.exists())
        self.assertEqual(summary["test_samples"], 38)
        self.assertGreaterEqual(summary["accuracy"], 0.70)


if __name__ == "__main__":
    unittest.main()
