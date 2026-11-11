"""Main application entry point for the Real-Time Driver Fatigue and Distraction Detection System.

Phase 2 Entry Point:
Orchestrates real-time video acquisition, MediaPipe Face Landmarker Tasks API detection,
facial landmark and contour visualization, mirrored driver preview, HUD status telemetry,
screenshot capture, and clean resource cleanup.
"""

import argparse
import sys
import time
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None

from config.settings import AppConfig, settings
from detection.face_detection import FaceDetector
from detection.video_capture import VideoCamera
from visualization.draw_utils import Visualizer


class DriverMonitoringApp:
    """Central orchestrator for driver monitoring runtime.

    Attributes:
        config (AppConfig): Centralized configuration dataclass.
        camera (VideoCamera): Video acquisition hardware source.
        face_detector (FaceDetector): MediaPipe Face Landmarker Tasks API wrapper.
        visualizer (Visualizer): Graphical overlay HUD renderer.
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
    ) -> None:
        """Initialize application configuration, camera hardware, and face landmarker."""
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

        self.visualizer = Visualizer()

    def run(self, no_gui: bool = False) -> None:
        """Execute the real-time Phase 2 video capture and facial landmark detection loop.

        Args:
            no_gui: If True, executes in console-only mode without opening an OpenCV GUI window.
        """
        print("=================================================================")
        print("🚗 Real-Time Driver Fatigue and Distraction Detection System")
        print("▶️  Phase 2: MediaPipe Facial Landmark Detection")
        print("=================================================================")
        print(f"Device Index: {self.camera.camera_index} | Target: {self.camera.width}x{self.camera.height} @ {self.camera.fps} FPS")
        print(f"Model Asset: {self.face_detector.model_path.name} | Mesh Overlay: {self.draw_mesh}")
        print("Keyboard Controls: Press 's' to Save Screenshot | Press 'q' or ESC to Quit")
        print("=================================================================")

        if cv2 is None:
            print("❌ OpenCV (cv2) is not installed in the active environment.")
            print("Please install requirements using: pip install -r requirements.txt")
            return

        try:
            with self.camera as cam, self.face_detector as detector:
                actual_w, actual_h = cam.get_resolution()
                print(f"✓ Webcam stream active. Negotiated hardware resolution: {actual_w}x{actual_h}")
                print("✓ MediaPipe Face Landmarker initialized in VIDEO mode.")

                if no_gui:
                    print("Executing console-only detection verification (30 frames)...")
                    detected_count = 0
                    for i in range(30):
                        success, frame = cam.read_frame()
                        if not success or frame is None:
                            print(f"❌ Frame {i + 1}: Read failed.")
                            break

                        # Process unmirrored sensor frame through MediaPipe
                        result = detector.detect_landmarks(frame)
                        if result.has_face:
                            detected_count += 1

                        if (i + 1) % 10 == 0:
                            status = (
                                f"DETECTED ({result.num_landmarks} pts)"
                                if result.has_face
                                else "NOT DETECTED"
                            )
                            print(
                                f"Captured {i + 1}/30 frames | Face: {status} | "
                                f"Measured FPS: {cam.get_fps():.1f}"
                            )
                        time.sleep(0.033)

                    screenshot_path = cam.save_screenshot()
                    if screenshot_path:
                        print(f"✓ Screenshot captured and saved to: {screenshot_path}")
                    print(
                        f"✓ Phase 2 console verification complete. "
                        f"Face detected in {detected_count}/30 test frames."
                    )
                    return

                window_name = "Driver Monitoring System - Phase 2 Facial Landmark Feed"
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

                    # 2. Pipeline Step: Draw landmarks on an unmirrored frame copy
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

                    # 3. Pipeline Step: Horizontally mirror annotated frame for natural user preview
                    display_frame = cam.mirror_frame(annotated_original)
                    measured_fps = cam.get_fps()

                    # 4. Pipeline Step: Overlay warning banner if no face is detected
                    if not result.has_face:
                        display_frame = self.visualizer.draw_face_not_detected_banner(display_frame)

                    # 5. Pipeline Step: Render top HUD status bar (left-to-right text on mirrored frame)
                    cv2.rectangle(display_frame, (0, 0), (actual_w, 42), (25, 25, 25), -1)
                    hud_text = (
                        f"FPS: {measured_fps:.1f} | Res: {actual_w}x{actual_h} | "
                        f"{status_text} | 's': Screenshot | 'q': Quit"
                    )
                    cv2.putText(
                        display_frame,
                        hud_text,
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        status_color,
                        1,
                        cv2.LINE_AA,
                    )

                    # 6. Pipeline Step: Temporary screenshot saved overlay
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
                    elif key in (ord("s"), ord("S")):  # 's' for screenshot
                        # Save the mirrored display frame so saved image matches what user sees
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
            if cv2 is not None and not no_gui:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            print("✓ Video camera and FaceDetector resources cleanly released.")


def main() -> None:
    """CLI application entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time Driver Fatigue and Distraction Detection System - Phase 2"
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
    )
    app.run(no_gui=args.no_gui)


if __name__ == "__main__":
    main()
