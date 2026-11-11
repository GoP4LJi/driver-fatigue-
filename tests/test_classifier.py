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

    def test_placeholder_methods_raise_not_implemented(self) -> None:
        """Verify un-implemented decision logic raises NotImplementedError prior to Phase 4."""
        engine = TemporalRuleEngine()
        sample_features = DriverFeatures()
        with self.assertRaises(NotImplementedError):
            engine.update(sample_features)

        hybrid = HybridDecisionEngine(rule_engine=engine)
        with self.assertRaises(NotImplementedError):
            hybrid.decide(sample_features)


if __name__ == "__main__":
    unittest.main()

