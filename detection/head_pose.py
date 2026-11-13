"""Head pose estimation module using 3D-to-2D Perspective-n-Point (solvePnP).

Estimates head orientation Euler angles (Pitch, Yaw, Roll) by matching canonical 3D facial
landmarks against 2D landmark projections from the unmirrored camera sensor frame.
Provides geometric metrics for driver distraction detection and nodding/slumping detection.

Coordinate Conventions & Angle Definitions:
    - Camera Matrix: Focal length approximated by frame width; optical center at image center.
    - 3D Model Coordinate System:
        * X-axis: Points toward the viewer's right (driver's left ear direction).
        * Y-axis: Points downward along the image coordinate system (toward chin).
        * Z-axis: Points away from camera into the scene.
    - Euler Angles (in degrees):
        * Pitch: Nodding movement. Positive = looking down (slumping), Negative = looking up.
        * Yaw: Horizontal turning. Positive = turning right (driver perspective), Negative = turning left.
        * Roll: Lateral head tilt. Positive = tilting clockwise, Negative = tilting counter-clockwise.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from config.settings import settings


@dataclass
class HeadPoseResult:
    """Structured result containing estimated head orientation angles and vectors.

    Attributes:
        pitch (float): Pitch angle in degrees (positive = nodding down, negative = looking up).
        yaw (float): Yaw angle in degrees (positive = turning right, negative = turning left).
        roll (float): Roll angle in degrees (positive = tilting clockwise, negative = counter-clockwise).
        valid (bool): True if pose was successfully estimated using solvePnP.
        rvec (Optional[np.ndarray]): Rotation vector from cv2.solvePnP (3, 1).
        tvec (Optional[np.ndarray]): Translation vector from cv2.solvePnP (3, 1).
    """
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    valid: bool = False
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None

    def __iter__(self):
        """Allow tuple-like unpacking (pitch, yaw, roll)."""
        yield self.pitch
        yield self.yaw
        yield self.roll


class HeadPoseEstimator:
    """Estimates head pose (Pitch, Yaw, Roll) using OpenCV solvePnP and canonical 3D facial landmarks.

    Designed to operate strictly on the original unmirrored camera frame to preserve
    optical calibration and true perspective projection.
    """

    def __init__(
        self,
        model_points_3d: Optional[np.ndarray] = None,
        landmark_indices: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Initialize 3D facial model points and corresponding landmark indices.

        Args:
            model_points_3d: Optional custom (N, 3) reference coordinates.
            landmark_indices: Optional tuple of MediaPipe landmark indices.
        """
        if model_points_3d is not None:
            self.model_points_3d = np.array(model_points_3d, dtype=np.float64)
        else:
            self.model_points_3d = np.array(
                settings.head_pose.model_points_3d,
                dtype=np.float64,
            )

        if landmark_indices is not None:
            self.landmark_indices = landmark_indices
        else:
            self.landmark_indices = settings.landmarks.HEAD_POSE

    def get_camera_matrix(self, frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate resolution-independent camera intrinsic matrix and distortion coefficients.

        Args:
            frame_shape (Tuple[int, int]): Height and width of the frame (H, W).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (camera_matrix (3, 3), dist_coeffs (4, 1)).
        """
        h, w = frame_shape[:2]
        focal_length = float(w) * settings.head_pose.focal_length_factor
        center_x = float(w) / 2.0
        center_y = float(h) / 2.0

        camera_matrix = np.array(
            [
                [focal_length, 0.0, center_x],
                [0.0, focal_length, center_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        return camera_matrix, dist_coeffs

    def estimate_head_pose(
        self,
        landmarks: Any,
        frame_shape: Tuple[int, int],
    ) -> HeadPoseResult:
        """Estimate head orientation Euler angles from 2D facial landmarks and frame dimensions.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList, list of NormalizedLandmarks,
                or pre-extracted pixel coordinate array of shape (N, 2).
            frame_shape: Frame dimensions as (height, width).

        Returns:
            HeadPoseResult: Result object containing pitch, yaw, roll, validity, rvec, and tvec.
        """
        if cv2 is None or landmarks is None or frame_shape is None:
            return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, valid=False)

        if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
            return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, valid=False)

        h, w = frame_shape[:2]

        # Extract 2D points corresponding to landmark indices
        image_points = []
        if isinstance(landmarks, np.ndarray) and landmarks.shape[0] == len(self.landmark_indices):
            image_points = landmarks[:, :2].astype(np.float64)
        else:
            landmark_list = getattr(landmarks, "landmark", landmarks)
            if not hasattr(landmark_list, "__getitem__") or len(landmark_list) < max(self.landmark_indices):
                return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, valid=False)

            try:
                for idx in self.landmark_indices:
                    lm = landmark_list[idx]
                    image_points.append([lm.x * w, lm.y * h])
                image_points = np.array(image_points, dtype=np.float64)
            except (IndexError, AttributeError, TypeError):
                return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, valid=False)

        if len(image_points) != len(self.model_points_3d):
            return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, valid=False)

        if np.isnan(image_points).any() or np.isinf(image_points).any():
            return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, valid=False)

        camera_matrix, dist_coeffs = self.get_camera_matrix(frame_shape)

        try:
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=self.model_points_3d,
                imagePoints=image_points,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success or rvec is None or tvec is None:
                return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, valid=False)

            # Convert rotation vector to 3x3 rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)

            # Decompose 3x3 rotation matrix into Euler angles (in degrees)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Align angle signs with driver monitoring convention:
            # - Pitch: Positive = nodding down, Negative = looking up
            # - Yaw: Positive = turning right, Negative = turning left (driver perspective)
            # - Roll: Positive = tilting clockwise, Negative = tilting counter-clockwise
            pitch = float(-angles[0])
            yaw = float(-angles[1])
            roll = float(angles[2])

            # Clamp angles to [-180.0, 180.0]
            def _normalize_angle(angle: float) -> float:
                return ((angle + 180.0) % 360.0) - 180.0

            pitch = _normalize_angle(pitch)
            yaw = _normalize_angle(yaw)
            roll = _normalize_angle(roll)

            return HeadPoseResult(
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                valid=True,
                rvec=rvec,
                tvec=tvec,
            )

        except Exception:
            return HeadPoseResult(pitch=0.0, yaw=0.0, roll=0.0, valid=False)

    def is_distracted(
        self,
        pitch: float,
        yaw: float,
        roll: float,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Evaluate whether head orientation angles exceed safe driving corridor thresholds.

        Args:
            pitch: Nodding angle in degrees.
            yaw: Turning angle in degrees.
            roll: Tilting angle in degrees.
            thresholds: Optional dictionary containing pitch, yaw, and roll limits.

        Returns:
            bool: True if head pose indicates distraction, False if driver is looking forward.
        """
        th = thresholds or {}
        pitch_th = th.get("pitch", settings.thresholds.head_pitch_threshold)
        yaw_th = th.get("yaw", settings.thresholds.head_yaw_threshold)
        roll_th = th.get("roll", settings.thresholds.head_roll_threshold)

        return bool(
            abs(pitch) > pitch_th
            or abs(yaw) > yaw_th
            or abs(roll) > roll_th
        )
