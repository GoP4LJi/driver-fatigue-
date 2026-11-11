"""Unit tests for the video capture module (Phase 1)."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

import detection.video_capture as vc_module
from detection.video_capture import VideoCamera


class TestVideoCapture(unittest.TestCase):
    """Test suite verifying VideoCamera class contracts, lifecycle, and frame streaming."""

    def setUp(self) -> None:
        """Set up test instances."""
        self.camera = VideoCamera(camera_index=0, width=640, height=480, fps=30)

    def test_camera_initialization(self) -> None:
        """Verify camera properties are initialized with provided parameters."""
        self.assertEqual(self.camera.camera_index, 0)
        self.assertEqual(self.camera.width, 640)
        self.assertEqual(self.camera.height, 480)
        self.assertEqual(self.camera.fps, 30)
        self.assertIsNone(self.camera.backend)
        self.assertFalse(self.camera.is_opened())

    def test_custom_camera_initialization(self) -> None:
        """Verify camera initializes with custom index, resolution, and backend."""
        custom_cam = VideoCamera(camera_index=2, width=1280, height=720, fps=60, backend=700)
        self.assertEqual(custom_cam.camera_index, 2)
        self.assertEqual(custom_cam.width, 1280)
        self.assertEqual(custom_cam.height, 720)
        self.assertEqual(custom_cam.fps, 60)
        self.assertEqual(custom_cam.backend, 700)

    @patch("detection.video_capture.cv2", None)
    def test_start_raises_import_error_when_cv2_missing(self) -> None:
        """Verify start raises ImportError if OpenCV is not installed."""
        with self.assertRaises(ImportError) as ctx:
            self.camera.start()
        self.assertIn("OpenCV (cv2) is required", str(ctx.exception))

    def test_start_raises_runtime_error_on_device_failure(self) -> None:
        """Verify start raises RuntimeError when video device fails to open."""
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.object(vc_module, "cv2", mock_cv2):
            with self.assertRaises(RuntimeError) as ctx:
                self.camera.start()
            self.assertIn("Failed to open video capture device", str(ctx.exception))
            self.assertFalse(self.camera.is_opened())

    def test_start_success(self) -> None:
        """Verify start initializes capture device, sets properties, and marks camera open."""
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640.0 if prop == mock_cv2.CAP_PROP_FRAME_WIDTH else 480.0
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.object(vc_module, "cv2", mock_cv2):
            self.camera.start()
            self.assertTrue(self.camera.is_opened())
            mock_cap.set.assert_any_call(mock_cv2.CAP_PROP_FRAME_WIDTH, 640.0)
            mock_cap.set.assert_any_call(mock_cv2.CAP_PROP_FRAME_HEIGHT, 480.0)
            mock_cap.set.assert_any_call(mock_cv2.CAP_PROP_FPS, 30.0)
            self.assertEqual(self.camera.get_resolution(), (640, 480))
            self.assertEqual(self.camera.get_fps(), 30.0)

    def test_read_frame_when_not_opened(self) -> None:
        """Verify read_frame returns (False, None) if camera is not opened."""
        success, frame = self.camera.read_frame()
        self.assertFalse(success)
        self.assertIsNone(frame)

    def test_read_frame_success_and_fps_update(self) -> None:
        """Verify read_frame returns frame and calculates metrics when open."""
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.get.side_effect = lambda prop: 640.0 if prop == mock_cv2.CAP_PROP_FRAME_WIDTH else 480.0
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.object(vc_module, "cv2", mock_cv2):
            self.camera.start()
            success, frame = self.camera.read_frame()
            self.assertTrue(success)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertEqual(self.camera._frame_count, 1)

            # Read a second frame to test moving average FPS calculation
            success, frame = self.camera.read_frame()
            self.assertTrue(success)
            self.assertEqual(self.camera._frame_count, 2)
            self.assertGreaterEqual(self.camera.get_fps(), 0.0)

    def test_read_frame_hardware_failure(self) -> None:
        """Verify read_frame handles hardware read failures gracefully."""
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.object(vc_module, "cv2", mock_cv2):
            self.camera.start()
            success, frame = self.camera.read_frame()
            self.assertFalse(success)
            self.assertIsNone(frame)

    def test_mirror_frame_horizontal_flip(self) -> None:
        """Verify mirror_frame applies horizontal axis flip (columns reversed)."""
        test_frame = np.array([
            [[10, 10, 10], [20, 20, 20]],
            [[30, 30, 30], [40, 40, 40]],
        ], dtype=np.uint8)

        mirrored = VideoCamera.mirror_frame(test_frame)
        expected = np.array([
            [[20, 20, 20], [10, 10, 10]],
            [[40, 40, 40], [30, 30, 30]],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(mirrored, expected)

    def test_save_screenshot_explicit_frame(self) -> None:
        """Verify save_screenshot writes given frame to specified directory."""
        mock_cv2 = MagicMock()
        mock_cv2.imwrite.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch.object(vc_module, "cv2", mock_cv2):
                saved = self.camera.save_screenshot(
                    frame=dummy_frame,
                    output_dir=tmp_path,
                    filename="test_capture.jpg",
                )
                self.assertIsNotNone(saved)
                self.assertEqual(saved.name, "test_capture.jpg")
                mock_cv2.imwrite.assert_called_once_with(str(tmp_path / "test_capture.jpg"), dummy_frame)

    def test_save_screenshot_from_live_stream(self) -> None:
        """Verify save_screenshot captures and writes frame automatically if frame is None."""
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.imwrite.return_value = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch.object(vc_module, "cv2", mock_cv2):
                self.camera.start()
                saved = self.camera.save_screenshot(frame=None, output_dir=tmp_path)
                self.assertIsNotNone(saved)
                self.assertTrue(saved.name.startswith("screenshot_"))
                self.assertTrue(saved.name.endswith(".jpg"))

    def test_release_and_context_manager(self) -> None:
        """Verify context manager enters and exits cleanly, releasing resources."""
        mock_cv2 = MagicMock()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640.0 if prop == mock_cv2.CAP_PROP_FRAME_WIDTH else 480.0
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.object(vc_module, "cv2", mock_cv2):
            with self.camera as cam:
                self.assertTrue(cam.is_opened())
            mock_cap.release.assert_called_once()
            self.assertFalse(self.camera.is_opened())
            self.assertEqual(self.camera.get_fps(), 30.0)


if __name__ == "__main__":
    unittest.main()
