"""Hybrid decision engine synthesizing rule-based heuristics and machine learning inference.

Combines the deterministic safety-guarantee of temporal heuristic thresholds
(e.g., eyes closed >= 1.5 seconds must trigger DROWSY) with statistical machine learning
predictions to maximize sensitivity while suppressing false alarms.
"""

from collections import deque
from pathlib import Path
import sys
from typing import Any, Deque, Dict, List, Optional, Tuple

# Ensure project root is accessible
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import DriverState, settings
from decision_engine.threshold_rules import TemporalRuleEngine
from features.feature_extraction import DriverFeatures
from models.predict import StatePredictor


class HybridDecisionEngine:
    """Fuses heuristic temporal rules with machine learning classification outputs.

    Implements safety arbitration, confidence weighting, and sliding-window temporal smoothing
    between deterministic safety rules and machine learning probability estimates.
    """

    def __init__(
        self,
        rule_engine: Optional[TemporalRuleEngine] = None,
        ml_predictor: Optional[StatePredictor] = None,
        ml_weight: Optional[float] = None,
        rule_weight: Optional[float] = None,
        smoothing_window: int = 5,
    ) -> None:
        """Initialize hybrid engine with rule and ML sub-components."""
        self.rule_engine: TemporalRuleEngine = rule_engine or TemporalRuleEngine()
        self.ml_predictor: StatePredictor = ml_predictor or StatePredictor()
        
        cfg = settings.models
        self.ml_weight: float = (
            float(ml_weight) if ml_weight is not None else float(cfg.hybrid_weight_ml)
        )
        self.rule_weight: float = (
            float(rule_weight) if rule_weight is not None else float(cfg.hybrid_weight_rules)
        )

        # Normalize weights to sum to 1.0
        total_w = self.ml_weight + self.rule_weight
        if total_w > 0:
            self.ml_weight /= total_w
            self.rule_weight /= total_w
        else:
            self.ml_weight = 0.5
            self.rule_weight = 0.5

        # Sliding window for temporal smoothing of state decisions
        self.smoothing_window: int = max(1, smoothing_window)
        self.state_history: Deque[DriverState] = deque(maxlen=self.smoothing_window)
        self._current_state: DriverState = DriverState.ALERT

    def decide(self, features: DriverFeatures) -> Tuple[DriverState, Dict[str, Any]]:
        """Synthesize features across rule engine and ML model to produce final classification.

        Resolution Strategy:
            1. Safety override: If temporal rules detect sustained eye closure, sustained yawning,
               or extreme head deviation, deterministic heuristic triggers take immediate precedence.
            2. ML offline fallback: If the ML model is not loaded or missing, fall back to pure rule engine.
            3. Weighted fusion: In ambiguous boundary scenarios, calculate weighted score across
               rule confidence and ML class probabilities.
            4. Temporal smoothing: Smooth transient frame-by-frame fluctuations across sliding window.

        Args:
            features (DriverFeatures): Extracted metrics for the current frame.

        Returns:
            Tuple[DriverState, Dict[str, Any]]: Final resolved DriverState and metadata
            including individual component confidences, active warnings, and arbitration notes.
        """
        # 1. Evaluate Rule Engine
        rule_state, rule_telemetry = self.rule_engine.update(features)
        rule_conf = float(rule_telemetry.get("confidence", 0.5))

        # 2. Evaluate ML Predictor
        ml_loaded = self.ml_predictor.is_loaded()
        if ml_loaded:
            ml_state, ml_probs = self.ml_predictor.predict(features)
            ml_conf = float(ml_probs.get(ml_state.value, 0.0))
        else:
            ml_state = None
            ml_probs = {
                DriverState.ALERT.value: 1.0,
                DriverState.DROWSY.value: 0.0,
                DriverState.DISTRACTED.value: 0.0,
            }
            ml_conf = 0.0

        # 3. Decision Arbitration Strategy
        arbitration_method = "rule_fallback"
        resolved_state = rule_state
        final_confidence = rule_conf

        # Scenario A: Deterministic Safety Override
        # If rules flagged DROWSY or DISTRACTED due to sustained physiological signals,
        # safety requirements demand immediate override.
        is_sustained_rule_trigger = (
            rule_telemetry.get("closed_eyes_counter", 0) >= self.rule_engine.ear_consec_frames
            or rule_telemetry.get("yawn_counter", 0) >= self.rule_engine.mar_consec_frames
            or rule_telemetry.get("distraction_counter", 0) >= self.rule_engine.distraction_consec_frames
            or rule_telemetry.get("missing_face_counter", 0) >= self.rule_engine.distraction_consec_frames
        )

        if is_sustained_rule_trigger:
            resolved_state = rule_state
            arbitration_method = "safety_override"
            final_confidence = rule_conf

        elif not ml_loaded:
            # Scenario B: ML model unavailable / offline -> pure rule-based fallback
            resolved_state = rule_state
            arbitration_method = "rule_fallback"
            final_confidence = rule_conf

        else:
            # Scenario C: Weighted Probabilistic Fusion
            arbitration_method = "weighted_fusion"

            # Construct rule probability distribution vector
            rule_probs: Dict[str, float] = {
                DriverState.ALERT.value: 0.0,
                DriverState.DROWSY.value: 0.0,
                DriverState.DISTRACTED.value: 0.0,
            }
            if rule_state == DriverState.ALERT:
                rule_probs[DriverState.ALERT.value] = rule_conf
                rem = max(0.0, 1.0 - rule_conf) / 2.0
                rule_probs[DriverState.DROWSY.value] = rem
                rule_probs[DriverState.DISTRACTED.value] = rem
            elif rule_state == DriverState.DROWSY:
                rule_probs[DriverState.DROWSY.value] = rule_conf
                rule_probs[DriverState.ALERT.value] = max(0.0, 1.0 - rule_conf)
            elif rule_state == DriverState.DISTRACTED:
                rule_probs[DriverState.DISTRACTED.value] = rule_conf
                rule_probs[DriverState.ALERT.value] = max(0.0, 1.0 - rule_conf)

            # Compute combined scores
            combined_scores: Dict[str, float] = {}
            for s in DriverState:
                s_val = s.value
                r_p = rule_probs.get(s_val, 0.0)
                m_p = ml_probs.get(s_val, 0.0)
                combined_scores[s_val] = float(self.rule_weight * r_p + self.ml_weight * m_p)

            # Pick state with maximum combined probability
            best_state_str = max(combined_scores.items(), key=lambda kv: kv[1])[0]
            resolved_state = DriverState(best_state_str)
            final_confidence = combined_scores[best_state_str]

        # 4. Temporal Smoothing
        self.state_history.append(resolved_state)
        if is_sustained_rule_trigger:
            smoothed_state = resolved_state
        else:
            smoothed_state = self._compute_smoothed_state(resolved_state)
        self._current_state = smoothed_state

        # 5. Build Comprehensive Output
        output_data: Dict[str, Any] = {
            "predicted_driver_state": smoothed_state,
            "hybrid_state": smoothed_state,
            "hybrid_confidence": round(float(final_confidence), 3),
            "rule_state": rule_state,
            "rule_confidence": round(float(rule_conf), 3),
            "ml_state": ml_state if ml_loaded else None,
            "ml_confidence": round(float(ml_conf), 3) if ml_loaded else 0.0,
            "ml_probabilities": {k: round(v, 4) for k, v in ml_probs.items()} if ml_loaded else {},
            "ml_loaded": ml_loaded,
            "arbitration_method": arbitration_method,
            "is_safety_override": is_sustained_rule_trigger,
            "features_summary": {
                "ear": round(float(features.average_ear), 3) if features else 0.0,
                "mar": round(float(features.mar), 3) if features else 0.0,
                "pitch": round(float(features.pitch), 1) if features else 0.0,
                "yaw": round(float(features.yaw), 1) if features else 0.0,
                "roll": round(float(features.roll), 1) if features else 0.0,
                "valid": bool(features.valid) if features else False,
                "face_detected": bool(features.face_detected) if features else False,
            },
            "rule_telemetry": rule_telemetry,
        }

        return smoothed_state, output_data

    def _compute_smoothed_state(self, current_candidate: DriverState) -> DriverState:
        """Apply sliding-window majority voting with safety bias.

        If any safety critical event (DROWSY / DISTRACTED) represents >= 40% of the window,
        prevent immediate fallback to ALERT until the alert condition clears across the window.
        """
        if not self.state_history:
            return current_candidate

        counts = {s: 0 for s in DriverState}
        for st in self.state_history:
            counts[st] += 1

        total = len(self.state_history)
        # Safety bias: Prioritize DROWSY if significant portion of window exhibits drowsiness
        if counts[DriverState.DROWSY] / total >= 0.40:
            return DriverState.DROWSY
        if counts[DriverState.DISTRACTED] / total >= 0.40:
            return DriverState.DISTRACTED

        # Majority vote
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def reset(self) -> None:
        """Reset internal temporal states, rule counters, and history buffer."""
        self.rule_engine.reset()
        self.state_history.clear()
        self._current_state = DriverState.ALERT
