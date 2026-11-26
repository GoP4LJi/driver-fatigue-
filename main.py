"""Main application entry point for the Real-Time Driver Fatigue and Distraction System.

Phase 5 Entry Point:
Orchestrates real-time video acquisition, MediaPipe Face Landmarker Tasks API detection,
mathematical feature extraction (Eye Aspect Ratio, Mouth Aspect Ratio, 3D Head Pose solvePnP),
Hybrid Decision Engine classification (Rule-based heuristics + scikit-learn ML inference + temporal smoothing),
non-blocking Pygame audio alert notifications with cooldown & mute controls,
color-coded warning banners, telemetry HUD, mirrored driver preview, screenshot capture,
and clean resource cleanup.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None

from alerts.audio_alert import AlertManager
from config.settings import AppConfig, DriverState, settings
from decision_engine.hybrid_classifier import HybridDecisionEngine
from detection.face_detection import FaceDetector
from detection.video_capture import VideoCamera
from features.feature_extraction import DriverFeatures, FeatureExtractor
from models.predict import StatePredictor
from visualization.draw_utils import Visualizer


class DriverMonitoringApp:
    """Central orchestrator for driver monitoring runtime.

    Attributes:
        config (AppConfig): Centralized configuration dataclass.
        camera (VideoCamera): Video acquisition hardware source.
        face_detector (FaceDetector): MediaPipe Face Landmarker Tasks API wrapper.
        feature_extractor (FeatureExtractor): Physiological and pose feature computation engine.
        hybrid_classifier (HybridDecisionEngine): Fuses temporal rules and machine learning.
        alert_manager (AlertManager): Non-blocking audio alert dispatcher with cooldown and mute.
        visualizer (Visualizer): Graphical overlay HUD and status banner renderer.
        draw_mesh (bool): Whether to render complete facial tessellation mesh points.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        camera_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        draw_mesh: bool = False,
        model_path: Optional[Path] = None,
        ml_weight: Optional[float] = None,
        rule_weight: Optional[float] = None,
        enable_audio: Optional[bool] = None,
        alert_cooldown: Optional[float] = None,
        alert_volume: Optional[float] = None,
    ) -> None:
        """Initialize configuration, camera, face landmarker, hybrid engine, and audio alerts."""
        self.config = config or settings
        self.draw_mesh = draw_mesh

        cam_idx = camera_index if camera_index is not None else self.config.camera.camera_index
        f_width = width if width is not None else self.config.camera.frame_width
        f_height = height if height is not None else self.config.camera.frame_height
        target_fps = fps if fps is not None else self.config.camera.target_fps

        self.camera = VideoCamera(
            camera_index=cam_idx,
            width=f_width,
            height=f_height,
            fps=target_fps,
        )

        self.face_detector = FaceDetector(
            model_path=self.config.mediapipe.model_path,
            max_num_faces=self.config.mediapipe.max_num_faces,
            min_detection_confidence=self.config.mediapipe.min_face_detection_confidence,
            min_presence_confidence=self.config.mediapipe.min_face_presence_confidence,
            min_tracking_confidence=self.config.mediapipe.min_tracking_confidence,
        )

        self.feature_extractor = FeatureExtractor()

        # Phase 4 Hybrid Decision Engine setup
        predictor = StatePredictor(model_path=model_path or self.config.models.model_path)
        self.hybrid_classifier = HybridDecisionEngine(
            ml_predictor=predictor,
            ml_weight=ml_weight,
            rule_weight=rule_weight,
        )

        # Phase 5 Audio Alert System setup
        self.alert_manager = AlertManager(
            enabled=enable_audio,
            cooldown=alert_cooldown,
            volume=alert_volume,
            config=self.config.alerts,
        )

        self.visualizer = Visualizer()

    def run(self, no_gui: bool = False) -> None:
        """Execute the real-time video capture, feature extraction, classification, and audio alert loop.

        Args:
            no_gui: If True, executes in console-only mode without opening an OpenCV GUI window.
        """
        print("=================================================================")
        print("🚗 Real-Time Driver Fatigue and Distraction Detection System")
        print("▶️  Phase 5: Hybrid Decision Engine + Audio Alert System")
        print("=================================================================")
        print(f"Device Index: {self.camera.camera_index} | Target: {self.camera.width}x{self.camera.height} @ {self.camera.fps} FPS")
        print(f"Landmarker Model: {self.face_detector.model_path.name} | Mesh Overlay: {self.draw_mesh}")
        ml_status = "LOADED" if self.hybrid_classifier.ml_predictor.is_loaded() else "OFFLINE (Rule Fallback)"
        print(f"ML Classifier : {self.hybrid_classifier.ml_predictor.model_path.name} [{ml_status}]")
        print(f"Fusion Weights: Rules={self.hybrid_classifier.rule_weight:.2f}, ML={self.hybrid_classifier.ml_weight:.2f}")
        print("Keyboard Controls: 's': Screenshot | 'm': Mute/Unmute Audio | 'q' or ESC: Quit")
        print("=================================================================")

        if cv2 is None:
            print("❌ OpenCV (cv2) is not installed in the active environment.")
            print("Please install requirements using: pip install -r requirements.txt")
            return

        try:
            audio_ready = self.alert_manager.initialize()
            if audio_ready:
                print(
                    f"✓ Phase 5 AlertManager initialized ({self.alert_manager.get_status_label()} | "
                    f"Cooldown: {self.alert_manager.cooldown:.1f}s | Volume: {self.alert_manager.volume:.2f})"
                )
            else:
                print(
                    f"⚠️  Audio alerts unavailable ({self.alert_manager.unavailable_reason}); "
                    "continuing with visual HUD alerts only."
                )

            with self.camera as cam, self.face_detector as detector:
                actual_w, actual_h = cam.get_resolution()
                print(f"✓ Webcam stream active. Negotiated hardware resolution: {actual_w}x{actual_h}")
                print("✓ MediaPipe Face Landmarker initialized in VIDEO mode.")
                print("✓ Phase 3 Feature Extractor (EAR, MAR, solvePnP Head Pose) initialized.")
                print("✓ Phase 4 Hybrid Decision Engine (Rules + ML + Temporal Smoothing) active.")

                if no_gui:
                    print("Executing console-only detection, classification & audio alert check (30 frames)...")
                    detected_count = 0
                    for i in range(30):
                        success, frame = cam.read_frame()
                        if not success or frame is None:
                            print(f"❌ Frame {i + 1}: Read failed.")
                            break

                        # Process unmirrored sensor frame through MediaPipe
                        result = detector.detect_landmarks(frame)
                        features: DriverFeatures = self.feature_extractor.extract_features(
                            result.landmarks if result.has_face else None,
                            frame.shape,
                            timestamp=time.time(),
                        )

                        # Evaluate hybrid classifier & dispatch audio alert if needed
                        current_state, clf_result = self.hybrid_classifier.decide(features)
                        self.alert_manager.trigger_alert(current_state)

                        if result.has_face:
                            detected_count += 1

                        if (i + 1) % 10 == 0:
                            if features.face_detected and features.valid:
                                status = (
                                    f"STATE: {current_state.value} ({clf_result['hybrid_confidence']*100:.0f}%) | "
                                    f"ML: {getattr(clf_result.get('ml_state'), 'value', 'N/A')} | "
                                    f"EAR: {features.average_ear:.2f} | MAR: {features.mar:.2f} | "
                                    f"POSE: P:{features.pitch:+.1f}*, Y:{features.yaw:+.1f}*, R:{features.roll:+.1f}*"
                                )
                            else:
                                status = f"NOT DETECTED | STATE: {current_state.value}"
                            print(
                                f"Captured {i + 1}/30 frames | {status} | "
                                f"{self.alert_manager.get_status_label()} | "
                                f"Measured FPS: {cam.get_fps():.1f}"
                            )
                        time.sleep(0.033)

                    screenshot_path = cam.save_screenshot()
                    if screenshot_path:
                        print(f"✓ Screenshot captured and saved to: {screenshot_path}")
                    print(
                        f"✓ Phase 5 console verification complete. "
                        f"Face detected in {detected_count}/30 test frames."
                    )
                    return

                window_name = "Driver Monitoring System - Phase 5 Live Feed"
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

                screenshot_notification = ""
                notification_timestamp = 0.0

                while True:
                    success, frame = cam.read_frame()
                    if not success or frame is None:
                        time.sleep(0.01)
                        continue

                    # 1. Pipeline Step: Landmark detection on unmirrored original sensor frame
                    result = detector.detect_landmarks(frame)

                    # 2. Pipeline Step: Mathematical feature extraction on unmirrored sensor landmarks
                    features: DriverFeatures = self.feature_extractor.extract_features(
                        result.landmarks if result.has_face else None,
                        frame.shape,
                        timestamp=time.time(),
                    )

                    # 3. Pipeline Step: Phase 4 Hybrid classification (Rules + ML + Temporal smoothing)
                    current_state, clf_result = self.hybrid_classifier.decide(features)

                    # 4. Pipeline Step: Phase 5 Audio alert dispatch (non-blocking, cooldown-protected)
                    self.alert_manager.trigger_alert(current_state)

                    # 5. Pipeline Step: Draw landmarks on an unmirrored frame copy
                    annotated_original = frame.copy()
                    if result.has_face:
                        annotated_original = self.visualizer.draw_landmarks(
                            annotated_original,
                            result.landmarks,
                            draw_mesh=self.draw_mesh,
                            draw_contours=True,
                            draw_eyes=True,
                            draw_mouth=True,
                            draw_irises=True,
                        )
                        status_text = f"FACE DETECTED ({result.num_landmarks} pts)"
                        status_color = (0, 255, 0)  # Green
                    else:
                        status_text = "FACE NOT DETECTED"
                        status_color = (0, 165, 255)  # Amber / Warning

                    # 6. Pipeline Step: Horizontally mirror annotated frame for natural driver preview
                    display_frame = cam.mirror_frame(annotated_original)
                    measured_fps = cam.get_fps()

                    # 7. Pipeline Step: Render top status banner indicating current driver state
                    trigger_reason = clf_result.get("rule_telemetry", {}).get("trigger_reason", "")
                    display_frame = self.visualizer.draw_state_banner(
                        display_frame,
                        current_state,
                        confidence=clf_result["hybrid_confidence"],
                        subtext=trigger_reason,
                    )

                    # 8. Pipeline Step: Overlay warning banner if no face is detected
                    if not result.has_face:
                        display_frame = self.visualizer.draw_face_not_detected_banner(display_frame)

                    # 9. Pipeline Step: Render top HUD status bar (FPS, Resolution, Controls)
                    cv2.rectangle(display_frame, (0, 0), (actual_w, 42), (25, 25, 25), -1)
                    hud_text = (
                        f"FPS: {measured_fps:.1f} | {actual_w}x{actual_h} | "
                        f"{status_text} | 's': Snap | 'm': Mute | 'q': Quit"
                    )
                    cv2.putText(
                        display_frame,
                        hud_text,
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        status_color,
                        1,
                        cv2.LINE_AA,
                    )

                    # 10. Pipeline Step: Render Telemetry HUD & Audio Status badge
                    display_frame = self.visualizer.draw_telemetry(
                        display_frame,
                        features,
                        measured_fps,
                        classification_result=clf_result,
                    )
                    display_frame = self.visualizer.draw_audio_status(
                        display_frame,
                        self.alert_manager.get_status_label(),
                    )

                    # 11. Pipeline Step: Temporary screenshot saved overlay
                    if time.time() - notification_timestamp < 2.5 and screenshot_notification:
                        cv2.putText(
                            display_frame,
                            screenshot_notification,
                            (12, actual_h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    cv2.imshow(window_name, display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                        print("\nShutdown requested by user.")
                        break
                    elif key in (ord("m"), ord("M")):  # 'm' for audio mute/unmute
                        audio_on = self.alert_manager.toggle_mute()
                        print(f"✓ Audio alerts {'UNMUTED (ON)' if audio_on else 'MUTED'}.")
                    elif key in (ord("s"), ord("S")):  # 's' for screenshot
                        saved_path = cam.save_screenshot(display_frame)
                        if saved_path:
                            screenshot_notification = f"SAVED: {saved_path.name}"
                            notification_timestamp = time.time()
                            print(f"✓ Screenshot saved: {saved_path}")

        except KeyboardInterrupt:
            print("\nStream interrupted by user (KeyboardInterrupt).")
        except Exception as e:
            print(f"\n❌ Error during video capture execution: {e}")
        finally:
            self.alert_manager.close()
            if cv2 is not None and not no_gui:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            print("✓ Video camera, FaceDetector, and AlertManager resources cleanly released.")


def main() -> None:
    """CLI application entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time Driver Fatigue and Distraction Detection System - Phase 5"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=settings.camera.camera_index,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=settings.camera.frame_width,
        help="Target frame resolution width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=settings.camera.frame_height,
        help="Target frame resolution height (default: 480)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=settings.camera.target_fps,
        help="Target frame rate per second (default: 30)",
    )
    parser.add_argument(
        "--draw-mesh",
        action="store_true",
        help="Render complete 468-point facial mesh on preview",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=settings.models.model_path,
        help="Path to trained scikit-learn model artifact (.joblib)",
    )
    parser.add_argument(
        "--ml-weight",
        type=float,
        default=settings.models.hybrid_weight_ml,
        help="Weight for ML probability distribution in hybrid engine (default: 0.5)",
    )
    parser.add_argument(
        "--rule-weight",
        type=float,
        default=settings.models.hybrid_weight_rules,
        help="Weight for rule engine heuristics in hybrid engine (default: 0.5)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Start application with audio alerts muted",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=settings.alerts.cooldown_seconds,
        help="Minimum cooldown in seconds between audio alerts (default: 3.0)",
    )
    parser.add_argument(
        "--alert-volume",
        type=float,
        default=settings.alerts.volume,
        help="Audio alert volume from 0.0 to 1.0 (default: 0.8)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in headless console mode without opening GUI window (ideal for tests/diagnostics)",
    )
    args = parser.parse_args()

    app = DriverMonitoringApp(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        draw_mesh=args.draw_mesh,
        model_path=args.model_path,
        ml_weight=args.ml_weight,
        rule_weight=args.rule_weight,
        enable_audio=False if args.no_audio else settings.alerts.enable_audio,
        alert_cooldown=args.alert_cooldown,
        alert_volume=args.alert_volume,
    )
    app.run(no_gui=args.no_gui)


if __name__ == "__main__":
    main()
