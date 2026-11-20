"""Temporal rule-based decision engine for driver state classification.

Tracks sliding frame windows and consecutive occurrence counters for eye closure,
yawning, and head gaze deviations to filter out transient natural movements (e.g. normal blinks
or quick mirror checks) and flag sustained dangerous fatigue or distraction.
"""

import time
from typing import Any, Dict, Optional, Tuple
from config.settings import DriverState, settings
from features.feature_extraction import DriverFeatures


class TemporalRuleEngine:
    """Evaluates time-series frame counters against physiological threshold rules.

    Attributes:
        ear_threshold (float): Threshold for eye closure (EAR below this indicates closed eyes).
        ear_consec_frames (int): Minimum consecutive frames of eye closure to flag drowsiness.
        mar_threshold (float): Threshold for yawning detection (MAR above this indicates yawn).
        mar_consec_frames (int): Minimum consecutive frames of mouth opening for yawn confirmation.
        head_pitch_threshold (float): Pitch threshold (degrees) for nodding down/up.
        head_yaw_threshold (float): Yaw threshold (degrees) for looking left/right.
        head_roll_threshold (float): Roll threshold (degrees) for lateral head tilt.
        distraction_consec_frames (int): Minimum consecutive frames of off-center head pose.
        recovery_consec_frames (int): Consecutive normal frames required to transition back to ALERT.
    """

    def __init__(
        self,
        ear_threshold: Optional[float] = None,
        ear_consec_frames: Optional[int] = None,
        mar_threshold: Optional[float] = None,
        mar_consec_frames: Optional[int] = None,
        head_pitch_threshold: Optional[float] = None,
        head_yaw_threshold: Optional[float] = None,
        head_roll_threshold: Optional[float] = None,
        distraction_consec_frames: Optional[int] = None,
        recovery_consec_frames: int = 5,
    ) -> None:
        """Initialize rule engine thresholds and internal frame counters."""
        cfg = settings.thresholds
        self.ear_threshold: float = ear_threshold if ear_threshold is not None else cfg.ear_threshold
        self.ear_consec_frames: int = ear_consec_frames if ear_consec_frames is not None else cfg.ear_consec_frames
        self.mar_threshold: float = mar_threshold if mar_threshold is not None else cfg.mar_threshold
        self.mar_consec_frames: int = mar_consec_frames if mar_consec_frames is not None else cfg.mar_consec_frames
        self.head_pitch_threshold: float = (
            head_pitch_threshold if head_pitch_threshold is not None else cfg.head_pitch_threshold
        )
        self.head_yaw_threshold: float = (
            head_yaw_threshold if head_yaw_threshold is not None else cfg.head_yaw_threshold
        )
        self.head_roll_threshold: float = (
            head_roll_threshold if head_roll_threshold is not None else cfg.head_roll_threshold
        )
        self.distraction_consec_frames: int = (
            distraction_consec_frames if distraction_consec_frames is not None else cfg.distraction_consec_frames
        )
        self.recovery_consec_frames: int = max(1, recovery_consec_frames)

        # Internal state counters
        self.closed_eyes_counter: int = 0
        self.yawn_counter: int = 0
        self.distraction_counter: int = 0
        self.missing_face_counter: int = 0
        self.recovery_counter: int = 0

        # Monotonic time tracking for sustained duration calculation
        self._last_timestamp: float = time.monotonic()
        self._current_state: DriverState = DriverState.ALERT

    def update(self, features: DriverFeatures) -> Tuple[DriverState, Dict[str, Any]]:
        """Update temporal state counters with the current frame's features and evaluate state.

        Args:
            features (DriverFeatures): Extracted metrics for the current frame.

        Returns:
            Tuple[DriverState, Dict[str, Any]]: Determined rule-based DriverState and
            telemetry dictionary containing active counter values, trigger reasons, and flags.
        """
        now = time.monotonic()
        dt = max(0.0, now - self._last_timestamp)
        self._last_timestamp = now

        # 1. Handle missing face or invalid landmark extraction safely
        if features is None or not features.face_detected or not features.valid:
            self.missing_face_counter += 1
            # Decay physiological counters when face is lost (missing face != eyes closed)
            self.closed_eyes_counter = max(0, self.closed_eyes_counter - 1)
            self.yawn_counter = max(0, self.yawn_counter - 1)

            # If face is missing for extended duration, treat as distraction (driver looked away completely)
            if self.missing_face_counter >= self.distraction_consec_frames:
                self._current_state = DriverState.DISTRACTED
                reason = f"Face not detected for {self.missing_face_counter} consecutive frames."
                confidence = min(1.0, self.missing_face_counter / float(self.distraction_consec_frames))
            else:
                reason = "Face not detected or landmark calculation invalid."
                confidence = 0.5

            telemetry = self._build_telemetry(
                state=self._current_state,
                confidence=confidence,
                reason=reason,
                is_eye_closed=False,
                is_yawning=False,
                is_distracted=(self.missing_face_counter >= self.distraction_consec_frames),
                dt=dt,
            )
            return self._current_state, telemetry

        # Face is valid and detected: reset missing face counter
        self.missing_face_counter = 0

        # 2. Evaluate Eye Closure (EAR)
        ear_val = features.average_ear if features.average_ear > 0 else features.ear
        is_eye_closed = bool(ear_val < self.ear_threshold)
        if is_eye_closed:
            self.closed_eyes_counter += 1
        else:
            self.closed_eyes_counter = 0

        # 3. Evaluate Yawning (MAR)
        is_yawning = bool(features.mar > self.mar_threshold)
        if is_yawning:
            self.yawn_counter += 1
        else:
            self.yawn_counter = 0

        # 4. Evaluate Head Pose Deviation (Distraction)
        is_pose_deviated = bool(
            abs(features.pitch) > self.head_pitch_threshold
            or abs(features.yaw) > self.head_yaw_threshold
            or abs(features.roll) > self.head_roll_threshold
        )
        if is_pose_deviated:
            self.distraction_counter += 1
        else:
            self.distraction_counter = 0

        # 5. Evaluate State Triggers with Safety Priority:
        # Priority 1: Prolonged Eye Closure -> Critical Drowsiness
        # Priority 2: Prolonged Yawning -> Fatigue / Drowsiness
        # Priority 3: Sustained Head Deviation -> Distraction
        # Fallback: Alert
        raw_state = DriverState.ALERT
        reason = "Driver attentive and alert."
        confidence = 1.0

        if self.closed_eyes_counter >= self.ear_consec_frames:
            raw_state = DriverState.DROWSY
            reason = (
                f"Prolonged eye closure detected: {self.closed_eyes_counter} frames "
                f"(threshold: {self.ear_consec_frames}, EAR: {ear_val:.2f} < {self.ear_threshold:.2f})."
            )
            confidence = min(1.0, 0.70 + 0.30 * (self.closed_eyes_counter / (self.ear_consec_frames * 2)))

        elif self.yawn_counter >= self.mar_consec_frames:
            raw_state = DriverState.DROWSY
            reason = (
                f"Sustained yawning detected: {self.yawn_counter} frames "
                f"(threshold: {self.mar_consec_frames}, MAR: {features.mar:.2f} > {self.mar_threshold:.2f})."
            )
            confidence = min(1.0, 0.65 + 0.35 * (self.yawn_counter / (self.mar_consec_frames * 2)))

        elif self.distraction_counter >= self.distraction_consec_frames:
            raw_state = DriverState.DISTRACTED
            reason = (
                f"Sustained head gaze deviation: {self.distraction_counter} frames "
                f"(threshold: {self.distraction_consec_frames}, "
                f"P:{features.pitch:+.1f}*, Y:{features.yaw:+.1f}*, R:{features.roll:+.1f}*)."
            )
            confidence = min(1.0, 0.70 + 0.30 * (self.distraction_counter / (self.distraction_consec_frames * 2)))

        else:
            # Check proximity to thresholds for confidence estimation when ALERT
            ear_margin = max(0.0, ear_val - self.ear_threshold)
            confidence = min(1.0, max(0.60, 0.60 + ear_margin))

        # 6. Temporal Recovery Smoothing (avoid rapid toggling back to ALERT)
        if raw_state != DriverState.ALERT:
            self._current_state = raw_state
            self.recovery_counter = 0
        else:
            if self._current_state != DriverState.ALERT:
                self.recovery_counter += 1
                if self.recovery_counter >= self.recovery_consec_frames:
                    self._current_state = DriverState.ALERT
                    self.recovery_counter = 0
                else:
                    reason = f"Recovering to ALERT ({self.recovery_counter}/{self.recovery_consec_frames} frames)."
            else:
                self._current_state = DriverState.ALERT
                self.recovery_counter = 0

        telemetry = self._build_telemetry(
            state=self._current_state,
            confidence=confidence,
            reason=reason,
            is_eye_closed=is_eye_closed,
            is_yawning=is_yawning,
            is_distracted=is_pose_deviated,
            dt=dt,
        )

        return self._current_state, telemetry

    def _build_telemetry(
        self,
        state: DriverState,
        confidence: float,
        reason: str,
        is_eye_closed: bool,
        is_yawning: bool,
        is_distracted: bool,
        dt: float,
    ) -> Dict[str, Any]:
        """Construct a standardized telemetry dictionary for the current frame evaluation."""
        return {
            "state": state,
            "confidence": round(float(confidence), 3),
            "trigger_reason": reason,
            "closed_eyes_counter": self.closed_eyes_counter,
            "yawn_counter": self.yawn_counter,
            "distraction_counter": self.distraction_counter,
            "missing_face_counter": self.missing_face_counter,
            "recovery_counter": self.recovery_counter,
            "is_eye_closed": is_eye_closed,
            "is_yawning": is_yawning,
            "is_distracted": is_distracted,
            "ear_threshold": self.ear_threshold,
            "mar_threshold": self.mar_threshold,
            "head_pitch_threshold": self.head_pitch_threshold,
            "head_yaw_threshold": self.head_yaw_threshold,
            "head_roll_threshold": self.head_roll_threshold,
            "frame_dt": round(dt, 4),
        }

    def reset(self) -> None:
        """Reset all temporal counters and state to baseline ALERT."""
        self.closed_eyes_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0
        self.missing_face_counter = 0
        self.recovery_counter = 0
        self._current_state = DriverState.ALERT
        self._last_timestamp = time.monotonic()
