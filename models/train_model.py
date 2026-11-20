"""Machine learning training and offline evaluation pipeline for driver state classification.

Trains, validates, and serializes classification models (e.g., Random Forest,
Support Vector Machine, or Gradient Boosting) using labeled feature records.
Includes synthetic dataset generation for software testing verification only.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Ensure project root is accessible
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import DriverState, settings

logger = logging.getLogger(__name__)


def generate_synthetic_testing_data(
    output_path: Optional[Path] = None,
    n_samples: int = 600,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic driver feature records STRICTLY for software pipeline testing.

    DISCLAIMER:
        This function generates synthetic mathematical samples solely for unit testing,
        CI/CD verification, and interface validation of the ML pipeline.
        These synthetic samples DO NOT represent genuine human driver data and must NEVER
        be used or claimed as validated real-world performance benchmarks.

    Args:
        output_path: Optional path to save the generated CSV file.
        n_samples: Total number of synthetic samples (divided equally across 3 states).
        random_state: Seed for reproducible random distributions.

    Returns:
        pd.DataFrame: DataFrame containing columns [timestamp, ear, mar, pitch, yaw, roll, state].
    """
    rng = np.random.default_rng(random_state)
    samples_per_class = max(10, n_samples // 3)

    records: List[Dict[str, Any]] = []
    base_ts = 1700000000.0

    # 1. ALERT Class: High EAR, low MAR, centered head pose
    for i in range(samples_per_class):
        records.append({
            "timestamp": base_ts + i * 0.033,
            "ear": float(rng.uniform(0.26, 0.38)),
            "mar": float(rng.uniform(0.08, 0.40)),
            "pitch": float(rng.normal(0.0, 4.0)),
            "yaw": float(rng.normal(0.0, 5.0)),
            "roll": float(rng.normal(0.0, 3.0)),
            "state": DriverState.ALERT.value,
        })

    # 2. DROWSY Class: Low EAR (closed eyes) or high MAR (yawning), head tilting downward
    for i in range(samples_per_class):
        # 60% eyes closed, 40% yawning
        if rng.random() < 0.60:
            ear_val = float(rng.uniform(0.10, 0.23))
            mar_val = float(rng.uniform(0.10, 0.45))
        else:
            ear_val = float(rng.uniform(0.20, 0.32))
            mar_val = float(rng.uniform(0.62, 0.95))

        records.append({
            "timestamp": base_ts + (samples_per_class + i) * 0.033,
            "ear": ear_val,
            "mar": mar_val,
            "pitch": float(rng.normal(18.0, 6.0)),  # Nodding forward
            "yaw": float(rng.normal(0.0, 6.0)),
            "roll": float(rng.normal(2.0, 5.0)),
            "state": DriverState.DROWSY.value,
        })

    # 3. DISTRACTED Class: Off-center yaw (turning) or high pitch (looking up/down)
    for i in range(samples_per_class):
        # Turning left or right, or looking down at phone
        distraction_type = rng.choice(["yaw", "pitch_down", "roll"])
        if distraction_type == "yaw":
            yaw_val = float(rng.choice([-1.0, 1.0]) * rng.uniform(22.0, 45.0))
            pitch_val = float(rng.normal(0.0, 5.0))
            roll_val = float(rng.normal(0.0, 5.0))
        elif distraction_type == "pitch_down":
            yaw_val = float(rng.normal(0.0, 6.0))
            pitch_val = float(rng.uniform(25.0, 40.0))  # Looking down
            roll_val = float(rng.normal(0.0, 5.0))
        else:
            yaw_val = float(rng.normal(0.0, 6.0))
            pitch_val = float(rng.normal(0.0, 5.0))
            roll_val = float(rng.choice([-1.0, 1.0]) * rng.uniform(22.0, 35.0))

        records.append({
            "timestamp": base_ts + (2 * samples_per_class + i) * 0.033,
            "ear": float(rng.uniform(0.24, 0.35)),
            "mar": float(rng.uniform(0.10, 0.45)),
            "pitch": pitch_val,
            "yaw": yaw_val,
            "roll": roll_val,
            "state": DriverState.DISTRACTED.value,
        })

    df = pd.DataFrame(records)
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    return df


class ModelTrainer:
    """Manages dataset loading, preprocessing, model training, and artifact serialization.

    Attributes:
        model_type (str): Name of classifier algorithm ('random_forest', 'svm', 'gradient_boosting').
        random_state (int): Seed for deterministic reproducible data splitting and training.
        feature_columns (List[str]): Expected numerical feature columns.
    """

    SUPPORTED_MODELS = ("random_forest", "svm", "gradient_boosting")

    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = 42,
        feature_columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize trainer with designated model architecture and random state."""
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type '{model_type}'. Supported: {self.SUPPORTED_MODELS}"
            )
        self.model_type = model_type
        self.random_state = random_state
        self.feature_columns = (
            feature_columns if feature_columns is not None else list(settings.models.feature_columns)
        )
        self.target_column = settings.models.target_column
        self.model: Optional[Any] = None

    def _create_model(self) -> Any:
        """Instantiate selected scikit-learn classifier."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=4,
                random_state=self.random_state,
                class_weight="balanced",
            )
        elif self.model_type == "svm":
            return SVC(
                kernel="rbf",
                probability=True,
                random_state=self.random_state,
                class_weight="balanced",
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
            )
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def load_dataset(self, csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
        """Load tabular feature dataset and separate feature matrix from ground truth labels.

        Args:
            csv_path (Path): Path to CSV file containing [ear, mar, pitch, yaw, roll, state].

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X_features, y_labels).

        Raises:
            FileNotFoundError: If the specified CSV does not exist.
            ValueError: If required columns are missing or if dataset contains no valid records.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Feature dataset not found at: {path}")

        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Dataset at {path} is empty.")

        # Check required columns
        required_cols = set(self.feature_columns + [self.target_column])
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        # Drop NaN or Infinite values
        df_clean = df.dropna(subset=list(required_cols)).copy()
        for col in self.feature_columns:
            df_clean = df_clean[np.isfinite(df_clean[col])]

        if df_clean.empty:
            raise ValueError("No valid numerical records remaining after dropping nulls/infinities.")

        # Filter to allowed target states
        allowed_states = {s.value for s in DriverState}
        df_clean = df_clean[df_clean[self.target_column].isin(allowed_states)]
        if df_clean.empty:
            raise ValueError(f"No records match supported states: {allowed_states}")

        X = df_clean[self.feature_columns].astype(np.float32)
        y = df_clean[self.target_column].astype(str)

        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Fit classification model on training dataset.

        Args:
            X (pd.DataFrame): Training feature matrix.
            y (pd.Series): Categorical target labels ('ALERT', 'DROWSY', 'DISTRACTED').

        Returns:
            Any: Trained scikit-learn estimator instance.
        """
        if X.empty or y.empty:
            raise ValueError("Cannot train model on empty training data.")

        self.model = self._create_model()
        self.model.fit(X, y)
        return self.model

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Compute performance metrics (Accuracy, F1, Precision, Recall, Confusion Matrix).

        Args:
            X_test (pd.DataFrame): Held-out test features.
            y_test (pd.Series): True labels for test partition.

        Returns:
            Dict[str, Any]: Dictionary of evaluation scores and classification summary.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained or loaded before evaluation.")

        y_pred = self.model.predict(X_test)
        classes = sorted(list(self.model.classes_))

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

        return {
            "model_type": self.model_type,
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "confusion_matrix": cm,
            "classes": classes,
            "classification_report": report,
            "sample_count": len(y_test),
        }

    def save_model(self, model: Any, output_path: Path) -> None:
        """Serialize trained model artifact and metadata to disk using joblib.

        Args:
            model: Trained scikit-learn estimator.
            output_path: Destination path for serialized .joblib file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": model,
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "classes": list(getattr(model, "classes_", [s.value for s in DriverState])),
            "framework": "scikit-learn",
        }
        joblib.dump(artifact, out)

    def run_pipeline(
        self,
        dataset_path: Path,
        output_model_path: Path,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Execute end-to-end dataset loading, splitting, training, evaluation, and saving.

        Args:
            dataset_path: Path to feature dataset CSV.
            output_model_path: Output target path for saved model artifact.
            test_size: Proportion of dataset reserved for evaluation (default: 0.2).

        Returns:
            Dict[str, Any]: Pipeline execution summary and test metrics.
        """
        X, y = self.load_dataset(dataset_path)

        # Stratified train/test split to preserve class distributions
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Train model
        model = self.train(X_train, y_train)

        # Evaluate performance on test set
        metrics = self.evaluate(X_test, y_test)

        # Persist trained model
        self.save_model(model, output_model_path)
        metrics["model_saved_path"] = str(output_model_path)
        metrics["train_samples"] = len(X_train)
        metrics["test_samples"] = len(X_test)

        return metrics


def main() -> None:
    """CLI execution entry point for ModelTrainer."""
    parser = argparse.ArgumentParser(
        description="Driver State Classifier Training Pipeline (scikit-learn)"
    )
    dataset_dir = settings.models.model_path.parent.parent.parent / "dataset"
    default_data = dataset_dir / "driver_features.csv"
    synthetic_data = dataset_dir / "processed" / "synthetic_driver_features.csv"

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to feature dataset CSV (default: dataset/driver_features.csv or synthetic fallback)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.models.model_path,
        help="Path where trained model artifact will be saved (.joblib)",
    )
    parser.add_argument(
        "--model-type",
        choices=ModelTrainer.SUPPORTED_MODELS,
        default="random_forest",
        help="Classification algorithm (default: random_forest)",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic dataset for testing software pipeline (NOT genuine data)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=600,
        help="Number of synthetic samples to generate if --generate-synthetic is specified",
    )
    args = parser.parse_args()

    print("=================================================================")
    print("🧠 Driver Fatigue and Distraction Detection - ML Training Pipeline")
    print("=================================================================")

    target_data = args.data
    if args.generate_synthetic:
        synthetic_path = target_data or synthetic_data
        print("⚠️ Generating synthetic testing dataset (FOR TEST HARNESS ONLY)...")
        generate_synthetic_testing_data(output_path=synthetic_path, n_samples=args.samples)
        print(f"✓ Synthetic test dataset created at: {synthetic_path}")
        target_data = synthetic_path
    elif target_data is None:
        # Check if default driver_features.csv has data
        if default_data.exists():
            try:
                test_df = pd.read_csv(default_data)
                if len(test_df) > 10:
                    target_data = default_data
            except Exception:
                pass

        if target_data is None:
            if synthetic_data.exists():
                print("ℹ️ Real dataset at dataset/driver_features.csv is empty (awaiting labeled data).")
                print(f"Using synthetic test data at: {synthetic_data.name}")
                target_data = synthetic_data
            else:
                target_data = default_data

    trainer = ModelTrainer(model_type=args.model_type)
    print(f"Algorithm: {args.model_type} | Target artifact: {args.output}")

    try:
        results = trainer.run_pipeline(dataset_path=target_data, output_model_path=args.output)
        print("-----------------------------------------------------------------")
        print(f"✓ Model trained successfully on {results['train_samples']} samples.")
        print(f"✓ Test Accuracy  : {results['accuracy'] * 100:.2f}%")
        print(f"✓ Test F1 (Macro): {results['f1_macro']:.4f}")
        print(f"✓ Test Precision : {results['precision_macro']:.4f}")
        print(f"✓ Test Recall    : {results['recall_macro']:.4f}")
        print(f"✓ Artifact saved : {results['model_saved_path']}")
        print("=================================================================")
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
