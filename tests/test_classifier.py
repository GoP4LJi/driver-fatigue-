"""Unit tests for the decision engine and classification components."""

import unittest
from config.settings import DriverState
from decision_engine.threshold_rules import TemporalRuleEngine
from decision_engine.hybrid_classifier import HybridDecisionEngine
from features.feature_extraction import DriverFeatures


class TestClassifier(unittest.TestCase):
    """Test suite verifying rule engine, hybrid classifier, and DriverState enumerations."""

    def test_driver_state_enum(self) -> None:
        """Verify DriverState contains required primary classification targets."""
        self.assertEqual(DriverState.ALERT.value, "ALERT")
        self.assertEqual(DriverState.DROWSY.value, "DROWSY")
        self.assertEqual(DriverState.DISTRACTED.value, "DISTRACTED")

    def test_rule_engine_initialization(self) -> None:
        """Verify TemporalRuleEngine default counter initialization."""
        engine = TemporalRuleEngine()
        self.assertEqual(engine.closed_eyes_counter, 0)
        self.assertEqual(engine.yawn_counter, 0)
        self.assertEqual(engine.distraction_counter, 0)

        # Test reset
        engine.closed_eyes_counter = 10
        engine.reset()
        self.assertEqual(engine.closed_eyes_counter, 0)

    def test_implemented_decision_logic(self) -> None:
        """Verify Phase 4 implemented decision logic executes and returns valid states."""
        engine = TemporalRuleEngine()
        sample_features = DriverFeatures(
            average_ear=0.32,
            mar=0.15,
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            face_detected=True,
            valid=True,
        )
        state, telemetry = engine.update(sample_features)
        self.assertIsInstance(state, DriverState)
        self.assertEqual(state, DriverState.ALERT)
        self.assertIn("closed_eyes_counter", telemetry)

        hybrid = HybridDecisionEngine(rule_engine=engine)
        h_state, h_meta = hybrid.decide(sample_features)
        self.assertIsInstance(h_state, DriverState)
        self.assertEqual(h_state, DriverState.ALERT)
        self.assertIn("hybrid_confidence", h_meta)


if __name__ == "__main__":
    unittest.main()
