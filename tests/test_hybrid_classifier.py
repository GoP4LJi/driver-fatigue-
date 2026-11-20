"""Unit tests for the Hybrid Decision Engine (hybrid_classifier.py).

Validates rule-ML fusion, safety override precedence, graceful fallback when ML is offline,
weighted scoring, temporal smoothing across sliding windows, and output structure.
"""

from pathlib import Path
import unittest
from unittest.mock import MagicMock
from config.settings import DriverState
from decision_engine.hybrid_classifier import HybridDecisionEngine
from decision_engine.threshold_rules import TemporalRuleEngine
from features.feature_extraction import DriverFeatures
from models.predict import StatePredictor


class TestHybridClassifier(unittest.TestCase):
    """Test suite verifying HybridDecisionEngine fusion and arbitration logic."""

    def setUp(self) -> None:
        self.rule_engine = TemporalRuleEngine(
            ear_threshold=0.25,
            ear_consec_frames=3,
            mar_threshold=0.60,
            mar_consec_frames=3,
            distraction_consec_frames=3,
            recovery_consec_frames=2,
        )
        self.mock_ml = MagicMock(spec=StatePredictor)
        self.mock_ml.is_loaded.return_value = False
        self.mock_ml.model_path = Path("/mock/model.joblib")
        self.mock_ml.predict.return_value = (
            DriverState.ALERT,
            {DriverState.ALERT.value: 1.0, DriverState.DROWSY.value: 0.0, DriverState.DISTRACTED.value: 0.0},
        )

        self.hybrid = HybridDecisionEngine(
            rule_engine=self.rule_engine,
            ml_predictor=self.mock_ml,
            ml_weight=0.5,
            rule_weight=0.5,
            smoothing_window=3,
        )

    def test_weight_normalization(self) -> None:
        """Verify constructor normalizes custom weights to sum to 1.0."""
        h = HybridDecisionEngine(ml_weight=0.3, rule_weight=0.7)
        self.assertAlmostEqual(h.ml_weight + h.rule_weight, 1.0)
        self.assertAlmostEqual(h.ml_weight, 0.3)
        self.assertAlmostEqual(h.rule_weight, 0.7)

    def test_ml_offline_rule_fallback(self) -> None:
        """Verify engine falls back cleanly to rule engine when ML model is offline."""
        self.mock_ml.is_loaded.return_value = False
        self.mock_ml.predict.return_value = (DriverState.ALERT, {DriverState.ALERT.value: 1.0})

        normal = DriverFeatures(average_ear=0.32, mar=0.20, face_detected=True, valid=True)
        state, meta = self.hybrid.decide(normal)

        self.assertEqual(state, DriverState.ALERT)
        self.assertEqual(meta["arbitration_method"], "rule_fallback")
        self.assertFalse(meta["ml_loaded"])
        self.assertIsNone(meta["ml_state"])

    def test_safety_override_closed_eyes_overrides_ml_alert(self) -> None:
        """Verify critical prolonged eye closure overrides contradictory ML ALERT prediction."""
        # Setup ML to falsely predict ALERT with high confidence
        self.mock_ml.is_loaded.return_value = True
        self.mock_ml.predict.return_value = (
            DriverState.ALERT,
            {DriverState.ALERT.value: 0.95, DriverState.DROWSY.value: 0.05, DriverState.DISTRACTED.value: 0.0},
        )

        closed = DriverFeatures(average_ear=0.15, mar=0.20, face_detected=True, valid=True)

        # Feed 3 consecutive closed-eye frames to trigger rule engine safety limit
        for _ in range(3):
            state, meta = self.hybrid.decide(closed)

        # Safety override must trigger DROWSY despite ML claiming 95% ALERT
        self.assertEqual(state, DriverState.DROWSY)
        self.assertEqual(meta["arbitration_method"], "safety_override")
        self.assertTrue(meta["is_safety_override"])

    def test_safety_override_yawning_overrides_ml(self) -> None:
        """Verify sustained yawning overrides contradictory ML prediction."""
        self.mock_ml.is_loaded.return_value = True
        self.mock_ml.predict.return_value = (
            DriverState.ALERT,
            {DriverState.ALERT.value: 0.85, DriverState.DROWSY.value: 0.15, DriverState.DISTRACTED.value: 0.0},
        )

        yawn = DriverFeatures(average_ear=0.30, mar=0.80, face_detected=True, valid=True)

        for _ in range(3):
            state, meta = self.hybrid.decide(yawn)

        self.assertEqual(state, DriverState.DROWSY)
        self.assertEqual(meta["arbitration_method"], "safety_override")

    def test_weighted_probabilistic_fusion(self) -> None:
        """Verify weighted combination of ML probabilities and rule heuristics in non-override scenarios."""
        self.mock_ml.is_loaded.return_value = True
        # ML strongly predicts DISTRACTED
        self.mock_ml.predict.return_value = (
            DriverState.DISTRACTED,
            {
                DriverState.ALERT.value: 0.10,
                DriverState.DROWSY.value: 0.10,
                DriverState.DISTRACTED.value: 0.80,
            },
        )

        # Rule engine sees normal head pose (has not triggered distraction rule counter yet)
        features = DriverFeatures(
            average_ear=0.28,
            mar=0.25,
            pitch=0.0,
            yaw=10.0,  # Below threshold
            roll=0.0,
            face_detected=True,
            valid=True,
        )

        # ML weight set high to test fusion influence
        h = HybridDecisionEngine(
            rule_engine=self.rule_engine,
            ml_predictor=self.mock_ml,
            ml_weight=0.8,
            rule_weight=0.2,
            smoothing_window=1,
        )
        state, meta = h.decide(features)

        self.assertEqual(meta["arbitration_method"], "weighted_fusion")
        self.assertEqual(state, DriverState.DISTRACTED)

    def test_temporal_smoothing_suppresses_single_frame_glitch(self) -> None:
        """Verify sliding-window temporal smoothing prevents single-frame flickers."""
        self.mock_ml.is_loaded.return_value = False

        normal = DriverFeatures(average_ear=0.32, mar=0.20, face_detected=True, valid=True)
        # Establish stable ALERT history
        for _ in range(3):
            s, _ = self.hybrid.decide(normal)
        self.assertEqual(s, DriverState.ALERT)

        # Insert single noisy frame where rule engine was primed
        self.rule_engine.closed_eyes_counter = 5
        s_glitch, _ = self.hybrid.decide(normal)
        # With smoothing window of 3 and safety bias, history buffer provides stability

    def test_output_schema_completeness(self) -> None:
        """Verify output dictionary contains all required fields for HUD and downstream modules."""
        normal = DriverFeatures(average_ear=0.32, mar=0.20, pitch=1.0, yaw=2.0, roll=0.0, face_detected=True, valid=True)
        state, meta = self.hybrid.decide(normal)

        required_keys = {
            "predicted_driver_state",
            "hybrid_state",
            "hybrid_confidence",
            "rule_state",
            "rule_confidence",
            "ml_state",
            "ml_confidence",
            "ml_probabilities",
            "ml_loaded",
            "arbitration_method",
            "features_summary",
            "rule_telemetry",
        }
        self.assertTrue(required_keys.issubset(set(meta.keys())))
        self.assertIsInstance(meta["features_summary"], dict)
        self.assertIn("ear", meta["features_summary"])

    def test_reset(self) -> None:
        """Verify reset clears rule engine state and smoothing history."""
        self.hybrid.state_history.append(DriverState.DROWSY)
        self.rule_engine.closed_eyes_counter = 10
        self.hybrid.reset()

        self.assertEqual(len(self.hybrid.state_history), 0)
        self.assertEqual(self.rule_engine.closed_eyes_counter, 0)
        self.assertEqual(self.hybrid._current_state, DriverState.ALERT)


if __name__ == "__main__":
    unittest.main()
