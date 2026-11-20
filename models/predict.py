"""Model inference module for online real-time driver state prediction.

Loads serialized scikit-learn model artifacts and generates driver state
classifications alongside class probability distributions given frame-level DriverFeatures.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

# Ensure project root is accessible
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import DriverState, settings
from features.feature_extraction import DriverFeatures


class StatePredictor:
    """Performs inference on real-time feature vectors using a pre-trained scikit-learn model.

    Attributes:
        model_path (Optional[Path]): Path to serialized model file (.joblib).
        feature_columns (List[str]): Expected ordered numerical feature names.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        feature_columns: Optional[List[str]] = None,
        auto_load: bool = True,
    ) -> None:
        """Initialize state predictor with optional model artifact path and auto-load toggle."""
        self.model_path = Path(model_path) if model_path is not None else settings.models.model_path
        self.feature_columns = (
            feature_columns if feature_columns is not None else list(settings.models.feature_columns)
        )
        self._model: Optional[Any] = None
        self._classes: List[str] = [s.value for s in DriverState]

        if auto_load and self.model_path.exists():
            try:
                self.load_model(self.model_path)
            except Exception:
                # Allow graceful degradation when model file is absent or corrupted
                self._model = None

    def load_model(self, model_path: Optional[Path] = None) -> None:
        """Deserialize and load model weights and schema metadata from disk.

        Args:
            model_path: Optional path to override instance model_path.

        Raises:
            FileNotFoundError: If model file does not exist at specified path.
            ValueError: If file is not a valid model artifact.
        """
        target_path = Path(model_path) if model_path is not None else self.model_path
        if not target_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at: {target_path}. "
                f"Run 'python models/train_model.py' to generate the model artifact."
            )

        loaded_obj = joblib.load(target_path)
        if isinstance(loaded_obj, dict) and "model" in loaded_obj:
            self._model = loaded_obj["model"]
            if "feature_columns" in loaded_obj and loaded_obj["feature_columns"]:
                self.feature_columns = loaded_obj["feature_columns"]
            if "classes" in loaded_obj and loaded_obj["classes"]:
                self._classes = [str(c) for c in loaded_obj["classes"]]
            elif hasattr(self._model, "classes_"):
                self._classes = [str(c) for c in self._model.classes_]
        else:
            # Direct estimator object
            self._model = loaded_obj
            if hasattr(self._model, "classes_"):
                self._classes = [str(c) for c in self._model.classes_]

        self.model_path = target_path

    def predict(self, features: DriverFeatures) -> Tuple[DriverState, Dict[str, float]]:
        """Generate predicted driver state and class probability distribution.

        Args:
            features (DriverFeatures): Computed metrics for the current frame.

        Returns:
            Tuple[DriverState, Dict[str, float]]: Tuple containing the predicted DriverState
            enum and a dictionary mapping each class name to its probability (0.0 to 1.0).
        """
        # Baseline fallback distribution
        fallback_probs: Dict[str, float] = {
            DriverState.ALERT.value: 1.0,
            DriverState.DROWSY.value: 0.0,
            DriverState.DISTRACTED.value: 0.0,
        }

        # Handle missing model or invalid features gracefully
        if not self.is_loaded():
            return DriverState.ALERT, fallback_probs

        if features is None or not features.face_detected or not features.valid:
            return DriverState.ALERT, fallback_probs

        try:
            # Build ordered feature DataFrame matching training schema
            import pandas as pd
            feature_df = pd.DataFrame(
                [features.to_array(self.feature_columns)],
                columns=self.feature_columns,
            )

            # Predict probabilities if supported, else one-hot pseudo-probabilities
            if hasattr(self._model, "predict_proba"):
                probs = self._model.predict_proba(feature_df)[0]
                prob_dict = {
                    cls_name: float(probs[i])
                    for i, cls_name in enumerate(self._classes)
                    if i < len(probs)
                }
            else:
                raw_pred = str(self._model.predict(feature_vector)[0])
                prob_dict = {cls_name: 0.0 for cls_name in self._classes}
                prob_dict[raw_pred] = 1.0

            # Determine predicted state from highest probability
            predicted_class_str = max(prob_dict.items(), key=lambda kv: kv[1])[0]
            try:
                predicted_state = DriverState(predicted_class_str)
            except ValueError:
                predicted_state = DriverState.ALERT

            # Ensure all standard states are represented in output dictionary
            for s in DriverState:
                if s.value not in prob_dict:
                    prob_dict[s.value] = 0.0

            return predicted_state, prob_dict

        except Exception:
            # Fail safely to ALERT on any unexpected mathematical/inference error
            return DriverState.ALERT, fallback_probs

    def is_loaded(self) -> bool:
        """Check whether a model has been successfully loaded into memory.

        Returns:
            bool: True if model is ready for inference, False otherwise.
        """
        return self._model is not None
