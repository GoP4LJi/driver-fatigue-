"""Unit tests for the temporal rule-based decision engine (threshold_rules.py).

Validates EAR threshold detection, MAR threshold detection, head pose deviations,
consecutive frame temporal counters, counter resets, state transitions, missing face handling,
and invalid feature handling without requiring hardware.
"""

import unittest
from config.settings import DriverState
from decision_engine.threshold_rules import TemporalRuleEngine
from features.feature_extraction import DriverFeatures


class TestThresholdRules(unittest.TestCase):
    """Test suite verifying heuristic rules, thresholds, and temporal counters."""

    def setUp(self) -> None:
        """Initialize engine with low consecutive frame counts for fast testing."""
        self.engine = TemporalRuleEngine(
            ear_threshold=0.25,
            ear_consec_frames=5,
            mar_threshold=0.60,
            mar_consec_frames=4,
            head_pitch_threshold=15.0,
            head_yaw_threshold=20.0,
            head_roll_threshold=20.0,
            distraction_consec_frames=5,
            recovery_consec_frames=3,
        )

    def test_alert_baseline_state(self) -> None:
        """Verify normal feature inputs maintain ALERT state and zero counters."""
        features = DriverFeatures(
            average_ear=0.32,
            mar=0.20,
            pitch=2.0,
            yaw=-1.0,
            roll=0.5,
            face_detected=True,
            valid=True,
        )
        state, meta = self.engine.update(features)
        self.assertEqual(state, DriverState.ALERT)
        self.assertEqual(self.engine.closed_eyes_counter, 0)
        self.assertEqual(self.engine.yawn_counter, 0)
        self.assertEqual(self.engine.distraction_counter, 0)
        self.assertFalse(meta["is_eye_closed"])
        self.assertFalse(meta["is_yawning"])
        self.assertFalse(meta["is_distracted"])

    def test_ear_prolonged_closure_drowsy_transition(self) -> None:
        """Verify single frame does NOT trigger DROWSY, but sustained closure does."""
        closed_features = DriverFeatures(
            average_ear=0.18,  # Below 0.25 threshold
            mar=0.20,
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            face_detected=True,
            valid=True,
        )

        # Single frame: counter increments, but state remains ALERT
        state, meta = self.engine.update(closed_features)
        self.assertEqual(self.engine.closed_eyes_counter, 1)
        self.assertEqual(state, DriverState.ALERT)
        self.assertTrue(meta["is_eye_closed"])

        # Feed 3 more frames (total 4, threshold is 5)
        for _ in range(3):
            state, meta = self.engine.update(closed_features)
            self.assertEqual(state, DriverState.ALERT)
        self.assertEqual(self.engine.closed_eyes_counter, 4)

        # 5th frame reaches ear_consec_frames: state transitions to DROWSY
        state, meta = self.engine.update(closed_features)
        self.assertEqual(self.engine.closed_eyes_counter, 5)
        self.assertEqual(state, DriverState.DROWSY)
        self.assertGreater(meta["confidence"], 0.70)
        self.assertIn("eye closure", meta["trigger_reason"])

    def test_ear_counter_reset_on_eye_opening(self) -> None:
        """Verify eye reopening resets closed_eyes_counter immediately."""
        closed = DriverFeatures(average_ear=0.18, mar=0.2, face_detected=True, valid=True)
        open_eye = DriverFeatures(average_ear=0.32, mar=0.2, face_detected=True, valid=True)

        for _ in range(3):
            self.engine.update(closed)
        self.assertEqual(self.engine.closed_eyes_counter, 3)

        # Driver opens eyes
        self.engine.update(open_eye)
        self.assertEqual(self.engine.closed_eyes_counter, 0)

    def test_mar_prolonged_yawning_drowsy_transition(self) -> None:
        """Verify sustained yawning triggers DROWSY state after threshold frames."""
        yawn_features = DriverFeatures(
            average_ear=0.30,
            mar=0.75,  # Above 0.60 threshold
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            face_detected=True,
            valid=True,
        )

        # Threshold is 4 frames
        for i in range(3):
            state, _ = self.engine.update(yawn_features)
            self.assertEqual(state, DriverState.ALERT)
            self.assertEqual(self.engine.yawn_counter, i + 1)

        # 4th frame triggers DROWSY
        state, meta = self.engine.update(yawn_features)
        self.assertEqual(state, DriverState.DROWSY)
        self.assertEqual(self.engine.yawn_counter, 4)
        self.assertTrue(meta["is_yawning"])
        self.assertIn("yawning", meta["trigger_reason"])

    def test_mar_counter_reset_on_mouth_closure(self) -> None:
        """Verify closing mouth resets yawn_counter immediately."""
        yawn = DriverFeatures(average_ear=0.30, mar=0.75, face_detected=True, valid=True)
        closed_mouth = DriverFeatures(average_ear=0.30, mar=0.20, face_detected=True, valid=True)

        for _ in range(3):
            self.engine.update(yawn)
        self.assertEqual(self.engine.yawn_counter, 3)

        self.engine.update(closed_mouth)
        self.assertEqual(self.engine.yawn_counter, 0)

    def test_head_pose_deviation_distracted_transition(self) -> None:
        """Verify sustained yaw deviation triggers DISTRACTED state."""
        turned_head = DriverFeatures(
            average_ear=0.30,
            mar=0.20,
            pitch=2.0,
            yaw=28.0,  # Exceeds head_yaw_threshold of 20.0
            roll=1.0,
            face_detected=True,
            valid=True,
        )

        # Threshold is 5 frames
        for i in range(4):
            state, _ = self.engine.update(turned_head)
            self.assertEqual(state, DriverState.ALERT)
            self.assertEqual(self.engine.distraction_counter, i + 1)

        # 5th frame triggers DISTRACTED
        state, meta = self.engine.update(turned_head)
        self.assertEqual(state, DriverState.DISTRACTED)
        self.assertEqual(self.engine.distraction_counter, 5)
        self.assertTrue(meta["is_distracted"])
        self.assertIn("head gaze deviation", meta["trigger_reason"])

    def test_pitch_and_roll_deviations(self) -> None:
        """Verify pitch (nodding) and roll (tilting) also increment distraction counter."""
        pitch_dev = DriverFeatures(
            average_ear=0.30, mar=0.20, pitch=22.0, yaw=0.0, roll=0.0, face_detected=True, valid=True
        )
        self.engine.update(pitch_dev)
        self.assertEqual(self.engine.distraction_counter, 1)

        self.engine.reset()
        roll_dev = DriverFeatures(
            average_ear=0.30, mar=0.20, pitch=0.0, yaw=0.0, roll=-25.0, face_detected=True, valid=True
        )
        self.engine.update(roll_dev)
        self.assertEqual(self.engine.distraction_counter, 1)

    def test_recovery_duration_prevents_rapid_switching(self) -> None:
        """Verify recovery frames requirement before state reverts from DROWSY back to ALERT."""
        closed = DriverFeatures(average_ear=0.18, mar=0.2, face_detected=True, valid=True)
        normal = DriverFeatures(average_ear=0.32, mar=0.2, face_detected=True, valid=True)

        # Trigger DROWSY
        for _ in range(5):
            self.engine.update(closed)
        self.assertEqual(self.engine._current_state, DriverState.DROWSY)

        # Frame 1 of normal: still DROWSY due to recovery buffer (recovery_consec_frames=3)
        state1, meta1 = self.engine.update(normal)
        self.assertEqual(state1, DriverState.DROWSY)
        self.assertEqual(self.engine.recovery_counter, 1)

        # Frame 2 of normal
        state2, meta2 = self.engine.update(normal)
        self.assertEqual(state2, DriverState.DROWSY)
        self.assertEqual(self.engine.recovery_counter, 2)

        # Frame 3 of normal: completed recovery -> transitions back to ALERT
        state3, meta3 = self.engine.update(normal)
        self.assertEqual(state3, DriverState.ALERT)
        self.assertEqual(self.engine.recovery_counter, 0)

    def test_missing_face_handling(self) -> None:
        """Verify missing face is handled safely without crashing and eventually flags distraction."""
        missing = DriverFeatures(face_detected=False, valid=False)

        # Single frame missing face
        state, meta = self.engine.update(missing)
        self.assertEqual(self.engine.missing_face_counter, 1)
        self.assertEqual(state, DriverState.ALERT)

        # Sustained missing face >= distraction_consec_frames (5 frames)
        for _ in range(4):
            state, meta = self.engine.update(missing)
        self.assertEqual(self.engine.missing_face_counter, 5)
        self.assertEqual(state, DriverState.DISTRACTED)

        # Face reappears: missing face counter resets to 0
        normal = DriverFeatures(average_ear=0.32, mar=0.2, face_detected=True, valid=True)
        self.engine.update(normal)
        self.assertEqual(self.engine.missing_face_counter, 0)

    def test_invalid_features_handling(self) -> None:
        """Verify None input or valid=False does not raise exceptions."""
        state1, meta1 = self.engine.update(None)
        self.assertIsInstance(state1, DriverState)
        self.assertEqual(self.engine.missing_face_counter, 1)

        invalid_f = DriverFeatures(average_ear=0.0, mar=0.0, face_detected=True, valid=False)
        state2, meta2 = self.engine.update(invalid_f)
        self.assertIsInstance(state2, DriverState)


if __name__ == "__main__":
    unittest.main()
