"""Face detection and facial landmark extraction module using MediaPipe Face Landmarker Tasks API.

Provides real-time facial landmark extraction (468/478 normalized 3D points), driver attentiveness
region extraction (eyes, mouth, irises), pixel coordinate projections, and safe hardware resource cleanup.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any, List, Optional, Tuple
import numpy as np

# Ensure project root is in sys.path when executed directly as a script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set Matplotlib cache dir to prevent writable directory warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None
    python = None
    vision = None

from config.settings import LandmarkIndices, settings


@dataclass
class FaceDetectionResult:
    """Structured result returned by FaceDetector for a single frame.

    Attributes:
        has_face (bool): True if at least one face was detected, False otherwise.
        landmarks (Optional[List[Any]]): List of 478 NormalizedLandmark objects (x, y, z), or None.
        timestamp_ms (int): Monotonically increasing frame timestamp in milliseconds.
    """
    has_face: bool = False
    landmarks: Optional[List[Any]] = None
    timestamp_ms: int = 0

    @property
    def num_landmarks(self) -> int:
        """Return count of detected landmarks (typically 468 or 478)."""
        return len(self.landmarks) if self.landmarks else 0


class FaceDetector:
    """MediaPipe Face Landmarker Tasks API wrapper for real-time driver face tracking.

    Operates in VIDEO running mode for temporal consistency across webcam frames.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize FaceDetector configuration parameters.

        Args:
            model_path: Path to official face_landmarker.task model asset.
            max_num_faces: Maximum number of faces to detect (default: 1 for driver).
            min_detection_confidence: Minimum confidence threshold for face detection.
            min_presence_confidence: Minimum confidence threshold for face presence.
            min_tracking_confidence: Minimum confidence threshold for landmark tracking.
        """
        self.model_path = Path(model_path) if model_path else settings.mediapipe.model_path
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._landmarker: Optional[Any] = None
        self._last_timestamp_ms: int = 0

    def initialize(self) -> None:
        """Instantiate underlying MediaPipe FaceLandmarker Tasks pipeline in VIDEO mode.

        Raises:
            ImportError: If mediapipe is not installed.
            FileNotFoundError: If the face_landmarker.task model file is missing.
        """
        if mp is None or vision is None:
            raise ImportError(
                "MediaPipe is required for FaceDetector. Please install 'mediapipe'."
            )

        if self._landmarker is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe Face Landmarker model asset not found at: {self.model_path}\n"
                f"Please ensure official model is downloaded from {settings.mediapipe.model_url}"
            )

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.max_num_faces,
            min_face_detection_confidence=self.min_detection_confidence,
            min_face_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def detect_landmarks(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> FaceDetectionResult:
        """Process an unmirrored BGR video frame and extract facial landmarks.

        Args:
            frame (np.ndarray): Original BGR image frame from camera sensor.
            timestamp_ms (Optional[int]): Millisecond timestamp. If None, auto-generated.

        Returns:
            FaceDetectionResult: Structured output indicating detection status and landmarks.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return FaceDetectionResult(has_face=False, landmarks=None, timestamp_ms=0)

        if self._landmarker is None:
            self.initialize()

        # Enforce monotonically increasing millisecond timestamps for VIDEO running mode
        if timestamp_ms is None:
            current_ms = int(time.time() * 1000)
        else:
            current_ms = timestamp_ms

        if current_ms <= self._last_timestamp_ms:
            current_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current_ms

        # Convert OpenCV BGR to RGB
        if cv2 is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame[:, :, ::-1].copy()

        # Wrap in MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Execute inference in VIDEO mode
        detection_result = self._landmarker.detect_for_video(mp_image, current_ms)

        if detection_result and detection_result.face_landmarks:
            primary_face = detection_result.face_landmarks[0]
            return FaceDetectionResult(
                has_face=True,
                landmarks=primary_face,
                timestamp_ms=current_ms,
            )

        return FaceDetectionResult(
            has_face=False,
            landmarks=None,
            timestamp_ms=current_ms,
        )

    def get_landmark_points_2d(
        self,
        landmarks: Any,
        frame_shape: Tuple[int, ...],
        indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Convert normalized landmark coordinates to 2D pixel coordinates (x, y).

        Args:
            landmarks: Collection of NormalizedLandmark objects.
            frame_shape: Height and width of the frame (H, W).
            indices: Optional subset of landmark indices. If None, extracts all.

        Returns:
            np.ndarray: Array of shape (N, 2) containing integer pixel coordinates [px, py].
        """
        if landmarks is None:
            return np.empty((0, 2), dtype=np.int32)

        landmark_list = getattr(landmarks, "landmark", landmarks)
        if not landmark_list:
            return np.empty((0, 2), dtype=np.int32)

        h, w = frame_shape[:2]
        target_indices = indices if indices is not None else range(len(landmark_list))

        pts = []
        for idx in target_indices:
            if idx < len(landmark_list):
                lm = landmark_list[idx]
                px = min(max(int(lm.x * w), 0), w - 1)
                py = min(max(int(lm.y * h), 0), h - 1)
                pts.append([px, py])

        return np.array(pts, dtype=np.int32)

    def get_landmark_points_3d(
        self,
        landmarks: Any,
        frame_shape: Tuple[int, ...],
        indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Convert normalized landmark coordinates to scaled 3D coordinates (x, y, z).

        Args:
            landmarks: Collection of NormalizedLandmark objects.
            frame_shape: Height and width of the frame (H, W).
            indices: Optional subset of landmark indices. If None, extracts all.

        Returns:
            np.ndarray: Array of shape (N, 3) containing float coordinates in pixel scale.
        """
        if landmarks is None:
            return np.empty((0, 3), dtype=np.float64)

        landmark_list = getattr(landmarks, "landmark", landmarks)
        if not landmark_list:
            return np.empty((0, 3), dtype=np.float64)

        h, w = frame_shape[:2]
        target_indices = indices if indices is not None else range(len(landmark_list))

        pts = []
        for idx in target_indices:
            if idx < len(landmark_list):
                lm = landmark_list[idx]
                px = lm.x * w
                py = lm.y * h
                pz = lm.z * w  # MediaPipe z is scaled proportionally to x
                pts.append([px, py, pz])

        return np.array(pts, dtype=np.float64)

    def get_left_eye_landmarks_2d(self, landmarks: Any, frame_shape: Tuple[int, ...]) -> np.ndarray:
        """Extract pixel coordinates for the driver's left eye boundary points."""
        return self.get_landmark_points_2d(landmarks, frame_shape, list(settings.landmarks.LEFT_EYE))

    def get_right_eye_landmarks_2d(self, landmarks: Any, frame_shape: Tuple[int, ...]) -> np.ndarray:
        """Extract pixel coordinates for the driver's right eye boundary points."""
        return self.get_landmark_points_2d(landmarks, frame_shape, list(settings.landmarks.RIGHT_EYE))

    def get_mouth_landmarks_2d(self, landmarks: Any, frame_shape: Tuple[int, ...]) -> np.ndarray:
        """Extract pixel coordinates for the driver's mouth boundary points."""
        return self.get_landmark_points_2d(landmarks, frame_shape, list(settings.landmarks.MOUTH))

    def close(self) -> None:
        """Release MediaPipe Tasks resources and destroy detector instances."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            finally:
                self._landmarker = None

    def __enter__(self) -> "FaceDetector":
        """Context manager entry point."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point releasing landmarker resources."""
        self.close()


def run_face_landmarker_preview(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    draw_mesh: bool = False,
    no_gui: bool = False,
) -> None:
    """Run interactive or headless MediaPipe Face Landmarker verification loop.

    Args:
        camera_index: Hardware webcam index.
        width: Video frame width.
        height: Video frame height.
        fps: Target framerate.
        draw_mesh: If True, draws complete 468/478 landmark mesh points.
        no_gui: If True, runs console-only verification without GUI window.
    """
    from detection.video_capture import VideoCamera
    from visualization.draw_utils import Visualizer

    print("=================================================================")
    print("👤 Phase 2: MediaPipe Face Landmarker Tasks API Verification")
    print("=================================================================")
    print(f"Loading model: {settings.mediapipe.model_path.name}")
    print(f"Camera index: {camera_index} ({width}x{height} @ {fps} FPS)")
    print("Controls: Press 's' to Save Screenshot | Press 'q' or ESC to Quit")
    print("=================================================================")

    if cv2 is None:
        print("❌ OpenCV (cv2) is not installed in the active environment.")
        return

    if mp is None:
        print("❌ MediaPipe is not installed in the active environment.")
        return

    camera = VideoCamera(camera_index=camera_index, width=width, height=height, fps=fps)
    detector = FaceDetector(
        model_path=settings.mediapipe.model_path,
        max_num_faces=settings.mediapipe.max_num_faces,
        min_detection_confidence=settings.mediapipe.min_face_detection_confidence,
        min_presence_confidence=settings.mediapipe.min_face_presence_confidence,
        min_tracking_confidence=settings.mediapipe.min_tracking_confidence,
    )
    visualizer = Visualizer()

    try:
        with camera as cam, detector as det:
            actual_w, actual_h = cam.get_resolution()
            print(f"✓ Stream started: {actual_w}x{actual_h} | FaceLandmarker active.")

            if no_gui:
                print("Running console-only landmark detection check (30 frames)...")
                detected_count = 0
                for i in range(30):
                    success, frame = cam.read_frame()
                    if not success or frame is None:
                        continue
                    result = det.detect_landmarks(frame)
                    if result.has_face:
                        detected_count += 1
                    if (i + 1) % 10 == 0:
                        status = f"DETECTED ({result.num_landmarks} pts)" if result.has_face else "NOT FOUND"
                        print(f"Frame {i + 1}/30 | Face: {status} | FPS: {cam.get_fps():.1f}")
                    time.sleep(0.033)
                print(f"✓ Completed: Face detected in {detected_count}/30 test frames.")
                return

            window_name = "Driver Monitoring System - Phase 2 Face Landmarker Feed"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

            screenshot_feedback = ""
            feedback_timestamp = 0.0

            while True:
                success, frame = cam.read_frame()
                if not success or frame is None:
                    time.sleep(0.01)
                    continue

                # 1. Detect landmarks on unmirrored original sensor frame
                result = det.detect_landmarks(frame)

                # 2. Draw landmarks on a copy of original frame
                annotated_original = frame.copy()
                if result.has_face:
                    annotated_original = visualizer.draw_landmarks(
                        annotated_original,
                        result.landmarks,
                        draw_mesh=draw_mesh,
                        draw_contours=True,
                    )
                    status_text = f"FACE DETECTED ({result.num_landmarks} pts)"
                    status_color = (0, 255, 0)  # Green
                else:
                    status_text = "FACE NOT DETECTED"
                    status_color = (0, 165, 255)  # Amber / Warning

                # 3. Apply horizontal mirroring for natural user preview
                display_frame = cam.mirror_frame(annotated_original)
                measured_fps = cam.get_fps()

                # 4. Render telemetry status bar on mirrored display_frame
                cv2.rectangle(display_frame, (0, 0), (actual_w, 42), (25, 25, 25), -1)
                hud_text = f"FPS: {measured_fps:.1f} | {status_text} | 's': Save | 'q': Quit"
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

                # Missing face warning banner overlay
                if not result.has_face:
                    cv2.rectangle(
                        display_frame,
                        (actual_w // 2 - 200, actual_h - 50),
                        (actual_w // 2 + 200, actual_h - 10),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        display_frame,
                        "ALIGN FACE WITH CAMERA",
                        (actual_w // 2 - 165, actual_h - 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )

                # Screenshot notification display
                if time.time() - feedback_timestamp < 2.5 and screenshot_feedback:
                    cv2.putText(
                        display_frame,
                        screenshot_feedback,
                        (12, actual_h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\nExit requested by user.")
                    break
                elif key in (ord("s"), ord("S")):
                    saved_path = cam.save_screenshot(display_frame)
                    if saved_path:
                        screenshot_feedback = f"SAVED: {saved_path.name}"
                        feedback_timestamp = time.time()
                        print(f"✓ Screenshot saved: {saved_path}")

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    except Exception as e:
        print(f"\n❌ Phase 2 execution error: {e}")
    finally:
        if cv2 is not None and not no_gui:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        print("✓ FaceDetector and camera resources cleanly released.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Driver Fatigue Monitor - Face Landmarker Module (Phase 2)")
    parser.add_argument("--camera", type=int, default=settings.camera.camera_index, help="Webcam device index")
    parser.add_argument("--width", type=int, default=settings.camera.frame_width, help="Frame width in pixels")
    parser.add_argument("--height", type=int, default=settings.camera.frame_height, help="Frame height in pixels")
    parser.add_argument("--fps", type=int, default=settings.camera.target_fps, help="Target FPS")
    parser.add_argument("--draw-mesh", action="store_true", help="Render full 468-point tessellation mesh")
    parser.add_argument("--no-gui", action="store_true", help="Run console-only verification without GUI window")
    cli_args = parser.parse_args()

    run_face_landmarker_preview(
        camera_index=cli_args.camera,
        width=cli_args.width,
        height=cli_args.height,
        fps=cli_args.fps,
        draw_mesh=cli_args.draw_mesh,
        no_gui=cli_args.no_gui,
    )
