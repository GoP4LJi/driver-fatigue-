"""Video capture module for webcam initialization, frame streaming, and resource management.

Provides an abstracted video source interface compatible with both standard OpenCV
desktop display loops and Streamlit image stream consumers, with macOS optimization,
real-time FPS calculation, error handling, and screenshot capture.
"""

from datetime import datetime
import os
from pathlib import Path
import platform
import sys
import time
from typing import Optional, Tuple
import numpy as np

# Ensure project root is in sys.path when executed directly as a script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import cv2
except ImportError:
    cv2 = None

from config.settings import SCREENSHOTS_DIR, settings


class VideoCamera:
    """Manages video capture hardware, frame acquisition, and cleanup.

    Attributes:
        camera_index (int): Hardware device index (default 0 for built-in camera).
        width (int): Target frame resolution width in pixels.
        height (int): Target frame resolution height in pixels.
        fps (int): Target frame rate per second.
        backend (Optional[int]): OpenCV capture API backend identifier.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: Optional[int] = None,
    ) -> None:
        """Initialize video capture configuration.

        Args:
            camera_index: Operating system video capture device index.
            width: Desired capture frame width.
            height: Desired capture frame height.
            fps: Desired capture frame rate per second.
            backend: Optional OpenCV backend identifier (e.g., cv2.CAP_AVFOUNDATION, cv2.CAP_DSHOW).
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend

        self._cap = None
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._last_frame_time: float = 0.0
        self._measured_fps: float = 0.0

    def _resolve_backend(self) -> Optional[int]:
        """Determine optimal capture backend for host operating system."""
        if cv2 is None or self.backend is not None:
            return self.backend

        # On macOS, cv2.CAP_AVFOUNDATION provides fast hardware access and correct camera permissions
        if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            return cv2.CAP_AVFOUNDATION

        return None

    def start(self) -> None:
        """Initialize and open the OpenCV video capture stream.

        Raises:
            ImportError: If OpenCV (cv2) is not installed in the environment.
            RuntimeError: If video capture device fails to open.
        """
        if cv2 is None:
            raise ImportError(
                "OpenCV (cv2) is required for VideoCamera. Please install 'opencv-python'."
            )

        if self.is_opened():
            return

        backend = self._resolve_backend()
        try:
            if backend is not None:
                self._cap = cv2.VideoCapture(self.camera_index, backend)
            else:
                self._cap = cv2.VideoCapture(self.camera_index)
        except Exception as e:
            self._cap = None
            raise RuntimeError(f"Error opening camera at index {self.camera_index}: {e}") from e

        # Fallback to default backend if platform backend failed to open
        if (self._cap is None or not self._cap.isOpened()) and backend is not None:
            self._cap = cv2.VideoCapture(self.camera_index)

        if self._cap is None or not self._cap.isOpened():
            self._cap = None
            os_info = platform.system()
            hint = ""
            if os_info == "Darwin":
                hint = (
                    " On macOS, ensure terminal/IDE has camera access under "
                    "System Settings -> Privacy & Security -> Camera."
                )
            raise RuntimeError(
                f"Failed to open video capture device at index {self.camera_index}. "
                f"Please verify device index, physical connection, and permissions.{hint}"
            )

        # Apply desired frame resolution and frame rate
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self._cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        # Retrieve actual hardware properties negotiated by driver
        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize FPS tracking
        self._frame_count = 0
        self._start_time = time.time()
        self._last_frame_time = time.time()
        self._measured_fps = float(self.fps)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera stream.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: A tuple where the first element indicates
            success (True/False) and the second element is the BGR image frame as a NumPy array.
        """
        if not self.is_opened():
            return False, None

        try:
            success, frame = self._cap.read()
        except Exception:
            return False, None

        if not success or frame is None:
            return False, None

        # Update frame metrics and measured FPS using Exponential Moving Average
        now = time.time()
        self._frame_count += 1
        dt = now - self._last_frame_time
        if dt > 0:
            instant_fps = 1.0 / dt
            self._measured_fps = 0.9 * self._measured_fps + 0.1 * instant_fps
        self._last_frame_time = now

        return True, frame

    def save_screenshot(
        self,
        frame: Optional[np.ndarray] = None,
        output_dir: Optional[Path] = None,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """Capture and save an image frame to disk.

        Args:
            frame: Optional frame to save. If None, captures a fresh frame from camera.
            output_dir: Destination folder. Defaults to project screenshots/ directory.
            filename: Target image filename. Defaults to timestamped name.

        Returns:
            Optional[Path]: Absolute path to the saved screenshot file, or None if failed.
        """
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required to save screenshots.")

        if frame is None:
            success, captured_frame = self.read_frame()
            if not success or captured_frame is None:
                return None
            frame = captured_frame

        target_dir = Path(output_dir) if output_dir else SCREENSHOTS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"screenshot_{timestamp_str}.jpg"

        file_path = target_dir / filename
        write_success = cv2.imwrite(str(file_path), frame)

        if write_success:
            return file_path
        return None

    def release(self) -> None:
        """Release camera hardware resources and close any open streams."""
        if self._cap is not None:
            try:
                if hasattr(self._cap, "isOpened") and self._cap.isOpened():
                    self._cap.release()
            except Exception:
                pass
            finally:
                self._cap = None
        self._actual_width = 0
        self._actual_height = 0
        self._measured_fps = 0.0

    def is_opened(self) -> bool:
        """Check if the video capture stream is currently open and accessible.

        Returns:
            bool: True if video capture is active, False otherwise.
        """
        return self._cap is not None and bool(self._cap.isOpened())

    def get_fps(self) -> float:
        """Get the current measured frame rate (or target FPS if idle).

        Returns:
            float: Frames per second.
        """
        if self.is_opened() and self._measured_fps > 0:
            return round(self._measured_fps, 2)
        return float(self.fps)

    def get_resolution(self) -> Tuple[int, int]:
        """Get the current frame resolution (width, height).

        Returns:
            Tuple[int, int]: (width, height) in pixels.
        """
        if self.is_opened() and self._actual_width > 0 and self._actual_height > 0:
            return self._actual_width, self._actual_height
        return self.width, self.height

    def __enter__(self) -> "VideoCamera":
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point releasing capture resources."""
        self.release()


def run_camera_preview(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    no_gui: bool = False,
) -> None:
    """Run interactive or headless camera stream verification loop.

    Args:
        camera_index: Hardware device index.
        width: Frame width.
        height: Frame height.
        fps: Target frame rate.
        no_gui: If True, executes console-only test without opening GUI window.
    """
    print("=================================================================")
    print("🎥 Phase 1: Real-Time Video Capture Diagnostics")
    print("=================================================================")
    print(f"Opening camera index: {camera_index} ({width}x{height} @ {fps} FPS)...")
    print("Controls: Press 's' to Save Screenshot | Press 'q' or ESC to Quit")
    print("=================================================================")

    camera = VideoCamera(camera_index=camera_index, width=width, height=height, fps=fps)

    try:
        with camera as cam:
            actual_w, actual_h = cam.get_resolution()
            print(f"✓ Camera connected successfully. Stream resolution: {actual_w}x{actual_h}")

            screenshot_feedback = ""
            feedback_time = 0.0

            if no_gui:
                print("Running in headless / console mode (reading 30 test frames)...")
                for i in range(30):
                    success, frame = cam.read_frame()
                    if not success or frame is None:
                        print(f"❌ Frame {i + 1}: Read failed.")
                        break
                    if (i + 1) % 10 == 0:
                        print(f"Captured {i + 1}/30 frames | Rolling FPS: {cam.get_fps():.1f}")
                    time.sleep(0.033)

                # Test screenshot capture
                saved_path = cam.save_screenshot()
                if saved_path:
                    print(f"✓ Test screenshot successfully captured: {saved_path}")
                print("✓ Console video capture verification finished successfully.")
                return

            window_name = "Driver Monitoring System - Phase 1 Live Feed"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

            while True:
                success, frame = cam.read_frame()
                if not success or frame is None:
                    print("Warning: Failed to grab frame from camera stream.")
                    time.sleep(0.01)
                    continue

                display_frame = frame.copy()
                fps_val = cam.get_fps()

                # Draw telemetry header overlay
                cv2.rectangle(display_frame, (0, 0), (actual_w, 40), (20, 20, 20), -1)
                hud_text = f"FPS: {fps_val:.1f} | Res: {actual_w}x{actual_h} | Press 's': Save | 'q': Exit"
                cv2.putText(
                    display_frame,
                    hud_text,
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                # Show temporary screenshot feedback if active
                if time.time() - feedback_time < 2.0 and screenshot_feedback:
                    cv2.putText(
                        display_frame,
                        screenshot_feedback,
                        (10, actual_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                    print("User requested exit. Stopping stream...")
                    break
                elif key in (ord("s"), ord("S")):  # 's' for screenshot
                    saved_file = cam.save_screenshot(frame)
                    if saved_file:
                        screenshot_feedback = f"SCREENSHOT SAVED: {saved_file.name}"
                        feedback_time = time.time()
                        print(f"Screenshot saved to: {saved_file}")

    except KeyboardInterrupt:
        print("\nSession interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\n❌ Video capture encountered an error: {e}")
    finally:
        if cv2 is not None and not no_gui:
            cv2.destroyAllWindows()
            # On macOS, pumping waitKey ensures the Cocoa window closes immediately
            cv2.waitKey(1)
        print("✓ Camera resources cleanly released.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Driver Fatigue Monitor - Video Capture Module (Phase 1)")
    parser.add_argument("--camera", type=int, default=settings.camera.camera_index, help="Webcam device index")
    parser.add_argument("--width", type=int, default=settings.camera.frame_width, help="Frame width in pixels")
    parser.add_argument("--height", type=int, default=settings.camera.frame_height, help="Frame height in pixels")
    parser.add_argument("--fps", type=int, default=settings.camera.target_fps, help="Target FPS")
    parser.add_argument("--no-gui", action="store_true", help="Run console-only verification without GUI window")
    cli_args = parser.parse_args()

    run_camera_preview(
        camera_index=cli_args.camera,
        width=cli_args.width,
        height=cli_args.height,
        fps=cli_args.fps,
        no_gui=cli_args.no_gui,
    )
