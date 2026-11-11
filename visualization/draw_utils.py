"""Graphical overlay and visualization utilities for heads-up display (HUD).

Renders eye landmarks, mouth contours, 3D head pose projection vectors, telemetry metrics,
and real-time color-coded driver status banners onto video frames.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from config.settings import DriverState, settings
from features.feature_extraction import DriverFeatures


class Visualizer:
    """Renders visual indicators, annotations, and metric dashboards on OpenCV BGR frames.

    Provides color mappings:
        - ALERT: Green (0, 255, 0)
        - DISTRACTED: Amber/Yellow (0, 200, 255)
        - DROWSY: Crimson/Red (0, 0, 255)
    """

    COLOR_MAP: Dict[DriverState, Tuple[int, int, int]] = {
        DriverState.ALERT: (0, 255, 0),        # Green
        DriverState.DISTRACTED: (0, 200, 255), # Amber / Yellow
        DriverState.DROWSY: (0, 0, 255),       # Red
    }

    def __init__(self) -> None:
        """Initialize visualizer fonts and styling parameters."""
        pass

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: Any,
        draw_mesh: bool = False,
        draw_contours: bool = True,
        draw_eyes: bool = True,
        draw_mouth: bool = True,
        draw_irises: bool = True,
    ) -> np.ndarray:
        """Render facial landmarks and feature contours (eyes, lips) on the video frame.

        Args:
            frame (np.ndarray): Original BGR image frame.
            landmarks: MediaPipe collection of NormalizedLandmark objects.
            draw_mesh (bool): Whether to render complete tessellation mesh points.
            draw_contours (bool): Whether to highlight key eye and lip contours.
            draw_eyes (bool): Whether to highlight eye regions.
            draw_mouth (bool): Whether to highlight mouth region.
            draw_irises (bool): Whether to highlight iris points.

        Returns:
            np.ndarray: Annotated BGR frame.
        """
        if frame is None or landmarks is None:
            return frame

        landmark_list = getattr(landmarks, "landmark", landmarks)
        if not landmark_list:
            return frame

        h, w = frame.shape[:2]
        cfg = settings.visualization
        indices = settings.landmarks

        # Draw full mesh points if requested
        if draw_mesh and cv2 is not None:
            for lm in landmark_list:
                px = min(max(int(lm.x * w), 0), w - 1)
                py = min(max(int(lm.y * h), 0), h - 1)
                cv2.circle(frame, (px, py), cfg.point_radius, cfg.landmark_color, -1)

        if draw_contours and cv2 is not None:
            # Helper to extract and draw a contour polygon
            def _draw_polygon(index_tuple, color, is_closed=True):
                pts = []
                for idx in index_tuple:
                    if idx < len(landmark_list):
                        lm = landmark_list[idx]
                        px = min(max(int(lm.x * w), 0), w - 1)
                        py = min(max(int(lm.y * h), 0), h - 1)
                        pts.append([px, py])
                if len(pts) > 1:
                    cv2.polylines(
                        frame,
                        [np.array(pts, dtype=np.int32)],
                        isClosed=is_closed,
                        color=color,
                        thickness=cfg.line_thickness,
                        lineType=cv2.LINE_AA,
                    )

            if draw_eyes:
                _draw_polygon(indices.LEFT_EYE_CONTOUR, cfg.eye_contour_color, is_closed=True)
                _draw_polygon(indices.RIGHT_EYE_CONTOUR, cfg.eye_contour_color, is_closed=True)

            if draw_irises and len(landmark_list) >= 478:
                _draw_polygon(indices.LEFT_IRIS, cfg.iris_color, is_closed=True)
                _draw_polygon(indices.RIGHT_IRIS, cfg.iris_color, is_closed=True)

            if draw_mouth:
                _draw_polygon(indices.MOUTH_OUTER, cfg.mouth_contour_color, is_closed=True)
                _draw_polygon(indices.MOUTH_INNER, cfg.mouth_contour_color, is_closed=True)

        return frame

    def draw_face_not_detected_banner(self, frame: np.ndarray) -> np.ndarray:
        """Render visual warning banner when no face is detected in the camera frame."""
        if frame is None or cv2 is None:
            return frame
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w // 2 - 200, h - 55), (w // 2 + 200, h - 15), (0, 0, 0), -1)
        cv2.putText(
            frame,
            "ALIGN FACE WITH CAMERA",
            (w // 2 - 165, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def draw_head_pose_axes(
        self,
        frame: np.ndarray,
        pitch: float,
        yaw: float,
        roll: float,
        nose_point: Tuple[int, int],
        axis_length: int = 50,
    ) -> np.ndarray:
        """Render 3D coordinate axes projecting from the nose point representing head orientation.

        Args:
            frame (np.ndarray): BGR image frame.
            pitch: Nodding angle in degrees.
            yaw: Turning angle in degrees.
            roll: Tilting angle in degrees.
            nose_point: 2D pixel coordinate (x, y) of the nose tip.
            axis_length: Length of projection vectors in pixels.

        Returns:
            np.ndarray: Annotated BGR frame with 3D projection axes (X=Red, Y=Green, Z=Blue).
        """
        raise NotImplementedError("Phase 3 implementation: 3D head pose projection axes.")

    def draw_telemetry(
        self,
        frame: np.ndarray,
        features: DriverFeatures,
        fps: float,
        state_counters: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """Render real-time numerical metrics HUD (EAR, MAR, Euler angles, FPS).

        Args:
            frame: Target BGR frame.
            features: Calculated DriverFeatures for the current frame.
            fps: Measured frames per second.
            state_counters: Active consecutive trigger frame counts.

        Returns:
            np.ndarray: Annotated BGR frame.
        """
        raise NotImplementedError("Phase 3 implementation: Real-time telemetry HUD overlay.")

    def draw_state_banner(
        self,
        frame: np.ndarray,
        state: DriverState,
        confidence: float = 1.0,
        subtext: Optional[str] = None,
    ) -> np.ndarray:
        """Render color-coded top banner indicating current driver state (ALERT / DROWSY / DISTRACTED).

        Args:
            frame: Target BGR frame.
            state: Current classification state.
            confidence: Probability or confidence score (0.0 to 1.0).
            subtext: Optional descriptive alarm explanation (e.g. 'EYES CLOSED > 1.5s').

        Returns:
            np.ndarray: Frame with warning banner overlay.
        """
        raise NotImplementedError("Phase 4 implementation: Driver state status banner.")

