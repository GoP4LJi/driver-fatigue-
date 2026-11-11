"""Main application entry point for the Real-Time Driver Fatigue and Distraction Detection System.

Phase 1 Entry Point:
Initializes the video acquisition pipeline, displays live camera telemetry, computes
real-time frames per second (FPS), and supports interactive screenshot capture and clean resource cleanup.
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
from detection.video_capture import VideoCamera


class DriverMonitoringApp:
    """Central orchestrator for driver monitoring runtime.

    Attributes:
        config (AppConfig): Centralized configuration dataclass.
        camera (VideoCamera): Video acquisition source.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        camera_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> None:
        """Initialize application configuration and video hardware."""
        self.config = config or settings
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

    def run(self, no_gui: bool = False) -> None:
        """Execute the real-time Phase 1 video capture loop.

        Args:
            no_gui: If True, executes in console-only mode without opening an OpenCV GUI window.
        """
        print("=================================================================")
        print("🚗 Real-Time Driver Fatigue and Distraction Detection System")
        print("▶️  Phase 1: Video Capture & Hardware Initialization")
        print("=================================================================")
        print(f"Device Index: {self.camera.camera_index} | Target: {self.camera.width}x{self.camera.height} @ {self.camera.fps} FPS")
        print("Keyboard Controls: Press 's' to Save Screenshot | Press 'q' or ESC to Quit")
        print("=================================================================")

        if cv2 is None:
            print("❌ OpenCV (cv2) is not installed in the active environment.")
            print("Please install requirements using: pip install -r requirements.txt")
            return

        try:
            with self.camera as cam:
                actual_w, actual_h = cam.get_resolution()
                print(f"✓ Webcam stream active. Negotiated hardware resolution: {actual_w}x{actual_h}")

                if no_gui:
                    print("Executing console-only frame capture verification (30 frames)...")
                    for i in range(30):
                        success, frame = cam.read_frame()
                        if not success or frame is None:
                            print(f"❌ Frame {i + 1}: Read failed.")
                            break
                        if (i + 1) % 10 == 0:
                            print(f"Captured {i + 1}/30 frames | Measured FPS: {cam.get_fps():.1f}")
                        time.sleep(0.033)

                    screenshot_path = cam.save_screenshot()
                    if screenshot_path:
                        print(f"✓ Screenshot captured and saved to: {screenshot_path}")
                    print("✓ Phase 1 console verification completed successfully.")
                    return

                window_name = "Driver Monitoring System - Phase 1 Live Webcam Stream"
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

                screenshot_notification = ""
                notification_timestamp = 0.0

                while True:
                    success, frame = cam.read_frame()
                    if not success or frame is None:
                        time.sleep(0.01)
                        continue

                    display_frame = frame.copy()
                    measured_fps = cam.get_fps()

                    # Render top telemetry status bar
                    cv2.rectangle(display_frame, (0, 0), (actual_w, 42), (25, 25, 25), -1)
                    hud_text = (
                        f"FPS: {measured_fps:.1f} | Res: {actual_w}x{actual_h} | "
                        f"Press 's': Screenshot | 'q': Quit"
                    )
                    cv2.putText(
                        display_frame,
                        hud_text,
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                    # Display temporary notification when screenshot is captured
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
                        saved_path = cam.save_screenshot(frame)
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
            print("✓ Video camera hardware resources cleanly released.")


def main() -> None:
    """CLI application entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time Driver Fatigue and Distraction Detection System - Phase 1"
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
    )
    app.run(no_gui=args.no_gui)


if __name__ == "__main__":
    main()
