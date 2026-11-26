"""Unit and integration tests for the Phase 5 Audio Alert System.

Verifies AlertManager initialization, pygame mixer handling, missing dependency
and missing asset resilience, DROWSY/DISTRACTED/ALERT playback behaviors,
cooldown suppression, mute/unmute controls, cleanup, WAV asset validity,
and integration with DriverMonitoringApp using mocked hardware/audio backends.
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import wave

import numpy as np

from alerts.audio_alert import AlertManager
from config.settings import AlertConfig, DriverState, SOUNDS_DIR
from detection.face_detection import FaceDetectionResult
from main import DriverMonitoringApp
from visualization.draw_utils import Visualizer


def _create_mock_pygame():
    """Create a structured MagicMock simulating pygame and pygame.mixer."""
    mock_pygame = MagicMock()
    mock_pygame.mixer.get_init.return_value = None

    def _on_init(*args, **kwargs):
        mock_pygame.mixer.get_init.return_value = (44100, -16, 2)

    def _on_quit():
        mock_pygame.mixer.get_init.return_value = None

    mock_pygame.mixer.init.side_effect = _on_init
    mock_pygame.mixer.quit.side_effect = _on_quit

    drowsy_sound = MagicMock()
    distracted_sound = MagicMock()
    drowsy_channel = MagicMock()
    distracted_channel = MagicMock()
    drowsy_channel.get_busy.return_value = True
    distracted_channel.get_busy.return_value = True
    drowsy_sound.play.return_value = drowsy_channel
    distracted_sound.play.return_value = distracted_channel

    def sound_factory(path_str: str):
        if "distracted" in Path(path_str).name:
            return distracted_sound
        return drowsy_sound

    mock_pygame.mixer.Sound.side_effect = sound_factory
    return mock_pygame, drowsy_sound, distracted_sound, drowsy_channel, distracted_channel


class TestAlertManager(unittest.TestCase):
    """Comprehensive unit tests for AlertManager with mocked pygame."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.drowsy_wav = Path(self.temp_dir.name) / "drowsy_alert.wav"
        self.distracted_wav = Path(self.temp_dir.name) / "distracted_alert.wav"
        self.drowsy_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        self.distracted_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        self.sound_paths = {
            DriverState.DROWSY: self.drowsy_wav,
            DriverState.DISTRACTED: self.distracted_wav,
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_initialization_defaults_and_clamping(self) -> None:
        """1. AlertManager initializes configuration, clamps volume, and starts uninitialized."""
        mgr = AlertManager(enabled=True, cooldown=4.5, volume=1.5, sound_paths=self.sound_paths)
        self.assertTrue(mgr.enabled)
        self.assertFalse(mgr.is_muted)
        self.assertEqual(mgr.cooldown, 4.5)
        self.assertEqual(mgr.volume, 1.0)
        self.assertFalse(mgr.is_initialized)
        self.assertFalse(mgr.is_available)
        self.assertEqual(mgr.get_status_label(), "AUDIO: UNAVAILABLE")

        mgr_low = AlertManager(volume=-0.4, cooldown=-2.0)
        self.assertEqual(mgr_low.volume, 0.0)
        self.assertEqual(mgr_low.cooldown, 0.0)

    def test_successful_audio_initialization(self) -> None:
        """2. initialize() sets up pygame mixer, loads WAVs, and sets volume."""
        mock_pygame, drowsy_sound, distracted_sound, _, _ = _create_mock_pygame()

        with patch("alerts.audio_alert.pygame", mock_pygame):
            mgr = AlertManager(volume=0.75, sound_paths=self.sound_paths)
            ok = mgr.initialize()

        self.assertTrue(ok)
        self.assertTrue(mgr.is_initialized)
        self.assertTrue(mgr.is_available)
        self.assertEqual(mgr.get_status_label(), "AUDIO: ON")
        mock_pygame.mixer.init.assert_called_once()
        drowsy_sound.set_volume.assert_called_with(0.75)
        distracted_sound.set_volume.assert_called_with(0.75)

    def test_missing_pygame_handling(self) -> None:
        """3. Missing pygame installation degrades gracefully without raising."""
        with patch("alerts.audio_alert.pygame", None):
            mgr = AlertManager(sound_paths=self.sound_paths)
            ok = mgr.initialize()

        self.assertFalse(ok)
        self.assertFalse(mgr.is_initialized)
        self.assertFalse(mgr.is_available)
        self.assertIn("pygame", mgr.unavailable_reason.lower())
        self.assertEqual(mgr.get_status_label(), "AUDIO: UNAVAILABLE")

    def test_missing_audio_file_handling_no_substitution(self) -> None:
        """4. Missing WAV file is skipped without substituting another sound."""
        mock_pygame, _, distracted_sound, _, _ = _create_mock_pygame()
        missing_drowsy = Path(self.temp_dir.name) / "nonexistent_drowsy.wav"

        with patch("alerts.audio_alert.pygame", mock_pygame):
            mgr = AlertManager(
                sound_paths={
                    DriverState.DROWSY: missing_drowsy,
                    DriverState.DISTRACTED: self.distracted_wav,
                }
            )
            ok = mgr.initialize()

            self.assertTrue(ok)
            self.assertFalse(mgr.trigger_alert(DriverState.DROWSY))
            self.assertTrue(mgr.trigger_alert(DriverState.DISTRACTED))
            distracted_sound.play.assert_called_once()

    def test_corrupt_audio_file_handled_gracefully(self) -> None:
        """Corrupt/unplayable WAV asset is caught without crashing initialize()."""
        mock_pygame, _, _, _, _ = _create_mock_pygame()
        mock_pygame.mixer.Sound.side_effect = RuntimeError("Unrecognized WAV header")

        with patch("alerts.audio_alert.pygame", mock_pygame):
            mgr = AlertManager(sound_paths=self.sound_paths)
            ok = mgr.initialize()
            self.assertTrue(ok)
            self.assertFalse(mgr.trigger_alert(DriverState.DROWSY))
            self.assertFalse(mgr.trigger_alert(DriverState.DISTRACTED))

    def test_drowsy_alert_triggering(self) -> None:
        """5. DROWSY state triggers the drowsiness warning sound."""
        mock_pygame, drowsy_sound, distracted_sound, _, _ = _create_mock_pygame()

        with patch("alerts.audio_alert.pygame", mock_pygame):
            mgr = AlertManager(sound_paths=self.sound_paths)
            mgr.initialize()
            triggered = mgr.trigger_alert(DriverState.DROWSY)

        self.assertTrue(triggered)
        drowsy_sound.play.assert_called_once()
        distracted_sound.play.assert_not_called()

    def test_distracted_alert_triggering(self) -> None:
        """6. DISTRACTED state triggers the distraction warning sound."""
        mock_pygame, drowsy_sound, distracted_sound, _, _ = _create_mock_pygame()

        with patch("alerts.audio_alert.pygame", mock_pygame):
            mgr = AlertManager(sound_paths=self.sound_paths)
            mgr.initialize()
            triggered = mgr.trigger_alert(DriverState.DISTRACTED)

        self.assertTrue(triggered)
        distracted_sound.play.assert_called_once()
        drowsy_sound.play.assert_not_called()

    def test_alert_state_stops_playback(self) -> None:
        """7. ALERT state stops active warning playback and plays no sound."""
        mock_pygame, drowsy_sound, _, drowsy_channel, _ = _create_mock_pygame()

        with patch("alerts.audio_alert.pygame", mock_pygame):
            mgr = AlertManager(sound_paths=self.sound_paths)
            mgr.initialize()
            self.assertTrue(mgr.trigger_alert(DriverState.DROWSY))
            self.assertFalse(mgr.trigger_alert(DriverState.ALERT))

        drowsy_channel.stop.assert_called_once()
        mock_pygame.mixer.stop.assert_called_once()
        self.assertEqual(drowsy_sound.play.call_count, 1)

    def test_cooldown_functionality(self) -> None:
        """8. Cooldown blocks triggers before elapsed time and allows playback afterward."""
        mock_pygame, drowsy_sound, distracted_sound, _, _ = _create_mock_pygame()

        with patch("alerts.audio_alert.pygame", mock_pygame), patch(
            "alerts.audio_alert.time.monotonic", side_effect=[10.0, 11.5, 13.1]
        ):
            mgr = AlertManager(cooldown=3.0, sound_paths=self.sound_paths)
            mgr.initialize()

            self.assertTrue(mgr.trigger_alert(DriverState.DROWSY))
            self.assertFalse(mgr.trigger_alert(DriverState.DISTRACTED))
            self.assertTrue(mgr.trigger_alert(DriverState.DISTRACTED))

        self.assertEqual(drowsy_sound.play.call_count, 1)
        self.assertEqual(distracted_sound.play.call_count, 1)

    def test_mute_functionality(self) -> None:
        """9. Muting silences active audio and suppresses subsequent alerts."""
        mock_pygame, drowsy_sound, _, drowsy_channel, _ = _create_mock_pygame()

        with patch("alerts.audio_alert.pygame", mock_pygame):
            mgr = AlertManager(sound_paths=self.sound_paths)
            mgr.initialize()
            self.assertTrue(mgr.trigger_alert(DriverState.DROWSY))

            new_state = mgr.toggle_mute()
            self.assertFalse(new_state)
            self.assertTrue(mgr.is_muted)
            self.assertEqual(mgr.get_status_label(), "AUDIO: MUTED")
            drowsy_channel.stop.assert_called_once()

            self.assertFalse(mgr.trigger_alert(DriverState.DISTRACTED))
        self.assertEqual(drowsy_sound.play.call_count, 1)

    def test_unmute_functionality(self) -> None:
        """10. Unmuting restores AUDIO: ON and resumes alert playback."""
        mock_pygame, drowsy_sound, _, _, _ = _create_mock_pygame()

        with patch("alerts.audio_alert.pygame", mock_pygame), patch(
            "alerts.audio_alert.time.monotonic", return_value=50.0
        ):
            mgr = AlertManager(enabled=False, sound_paths=self.sound_paths)
            mgr.initialize()
            self.assertEqual(mgr.get_status_label(), "AUDIO: MUTED")
            self.assertFalse(mgr.trigger_alert(DriverState.DROWSY))

            new_state = mgr.toggle_mute()
            self.assertTrue(new_state)
            self.assertEqual(mgr.get_status_label(), "AUDIO: ON")
            self.assertTrue(mgr.trigger_alert(DriverState.DROWSY))

        drowsy_sound.play.assert_called_once()

    def test_audio_resource_cleanup(self) -> None:
        """11. close() stops audio, clears loaded sounds, quits mixer, and is idempotent."""
        mock_pygame, _, _, _, _ = _create_mock_pygame()

        with patch("alerts.audio_alert.pygame", mock_pygame):
            with AlertManager(sound_paths=self.sound_paths) as mgr:
                self.assertTrue(mgr.is_initialized)
                self.assertTrue(mgr.trigger_alert(DriverState.DROWSY))
            # After context exit
            self.assertFalse(mgr.is_initialized)
            self.assertFalse(mgr.is_available)
            mgr.close()  # Idempotent second call

        mock_pygame.mixer.quit.assert_called_once()

    def test_repeated_alert_prevention_per_frame(self) -> None:
        """12. Repeated frame-by-frame DROWSY calls play only once during cooldown."""
        mock_pygame, drowsy_sound, _, _, _ = _create_mock_pygame()
        timestamps = [1.0 + i * 0.033 for i in range(30)]  # ~1 second at 30 FPS

        with patch("alerts.audio_alert.pygame", mock_pygame), patch(
            "alerts.audio_alert.time.monotonic", side_effect=timestamps
        ):
            mgr = AlertManager(cooldown=3.0, sound_paths=self.sound_paths)
            mgr.initialize()
            results = [mgr.trigger_alert(DriverState.DROWSY) for _ in range(30)]

        self.assertEqual(results.count(True), 1)
        self.assertEqual(drowsy_sound.play.call_count, 1)

    def test_playback_failure_handling(self) -> None:
        """13. Playback error in Sound.play() is caught without crashing or locking cooldown."""
        mock_pygame, drowsy_sound, _, _, _ = _create_mock_pygame()
        drowsy_sound.play.side_effect = [RuntimeError("Audio device lost"), MagicMock()]

        with patch("alerts.audio_alert.pygame", mock_pygame), patch(
            "alerts.audio_alert.time.monotonic", side_effect=[10.0, 10.1]
        ):
            mgr = AlertManager(cooldown=3.0, sound_paths=self.sound_paths)
            mgr.initialize()
            self.assertFalse(mgr.trigger_alert(DriverState.DROWSY))
            # Because playback failed, cooldown was not consumed
            self.assertTrue(mgr.trigger_alert(DriverState.DROWSY))

    def test_safe_operation_when_audio_unavailable(self) -> None:
        """14. All public methods operate safely when mixer init fails or audio is unavailable."""
        mock_pygame, _, _, _, _ = _create_mock_pygame()
        mock_pygame.mixer.init.side_effect = RuntimeError("No audio device")

        with patch("alerts.audio_alert.pygame", mock_pygame):
            mgr = AlertManager(sound_paths=self.sound_paths)
            self.assertFalse(mgr.initialize())
            self.assertEqual(mgr.get_status_label(), "AUDIO: UNAVAILABLE")
            self.assertFalse(mgr.trigger_alert(DriverState.DROWSY))
            self.assertFalse(mgr.trigger_alert(DriverState.DISTRACTED))
            self.assertFalse(mgr.trigger_alert(DriverState.ALERT))
            mgr.stop_all()
            mgr.set_volume(0.5)
            mgr.toggle_mute()
            mgr.close()

    def test_real_wav_assets_exist_and_are_valid(self) -> None:
        """Verify generated drowsy_alert.wav and distracted_alert.wav are valid, distinct WAVs."""
        drowsy_path = SOUNDS_DIR / "drowsy_alert.wav"
        distracted_path = SOUNDS_DIR / "distracted_alert.wav"
        self.assertTrue(drowsy_path.exists(), f"Missing asset: {drowsy_path}")
        self.assertTrue(distracted_path.exists(), f"Missing asset: {distracted_path}")

        durations = {}
        for name, path in (("drowsy", drowsy_path), ("distracted", distracted_path)):
            with wave.open(str(path), "rb") as wf:
                self.assertGreaterEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertGreaterEqual(wf.getframerate(), 16000)
                self.assertGreater(wf.getnframes(), 0)
                durations[name] = wf.getnframes() / wf.getframerate()

        self.assertNotEqual(drowsy_path.read_bytes(), distracted_path.read_bytes())
        self.assertGreater(durations["drowsy"], 0.2)
        self.assertGreater(durations["distracted"], 0.2)


class TestPhase5AppIntegration(unittest.TestCase):
    """Integration tests verifying DriverMonitoringApp + AlertManager wiring."""

    def test_app_triggers_audio_mute_toggle_and_cleans_up(self) -> None:
        """Verify main loop dispatches states to AlertManager, handles 'm' mute, and closes audio."""
        mock_pygame, drowsy_sound, distracted_sound, drowsy_channel, distracted_channel = _create_mock_pygame()

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2 = MagicMock()
        # Frame 1: DROWSY -> play drowsy
        # Frame 2: DISTRACTED -> play distracted, then 'm' mutes audio
        # Frame 3: DROWSY -> suppressed because muted
        # Frame 4: ALERT -> 'q' exits, plus final waitKey(1) in cleanup
        mock_cv2.waitKey.side_effect = [0xFF, ord("m"), 0xFF, ord("q"), 0]
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.LINE_AA = 16
        mock_cv2.WINDOW_AUTOSIZE = 1

        cfg = AlertConfig(
            enable_audio=True,
            cooldown_seconds=0.0,
            volume=0.8,
            sound_drowsy_path=SOUNDS_DIR / "drowsy_alert.wav",
            sound_distracted_path=SOUNDS_DIR / "distracted_alert.wav",
        )

        with patch("alerts.audio_alert.pygame", mock_pygame), patch(
            "main.cv2", mock_cv2
        ), patch("visualization.draw_utils.cv2", mock_cv2):
            app = DriverMonitoringApp()
            app.alert_manager = AlertManager(config=cfg)

            # Mock hardware & pipeline stages
            app.camera = MagicMock()
            app.camera.__enter__.return_value = app.camera
            app.camera.__exit__.return_value = None
            app.camera.get_resolution.return_value = (640, 480)
            app.camera.get_fps.return_value = 30.0
            app.camera.read_frame.return_value = (True, fake_frame)
            app.camera.mirror_frame.side_effect = lambda f: f

            app.face_detector = MagicMock()
            app.face_detector.model_path = Path("face_landmarker.task")
            app.face_detector.__enter__.return_value = app.face_detector
            app.face_detector.__exit__.return_value = None
            app.face_detector.detect_landmarks.return_value = FaceDetectionResult(
                has_face=False, landmarks=None, timestamp_ms=1
            )

            app.hybrid_classifier = MagicMock()
            app.hybrid_classifier.ml_predictor.is_loaded.return_value = False
            app.hybrid_classifier.ml_predictor.model_path = Path("model.joblib")
            app.hybrid_classifier.rule_weight = 0.5
            app.hybrid_classifier.ml_weight = 0.5
            app.hybrid_classifier.decide.side_effect = [
                (DriverState.DROWSY, {"hybrid_confidence": 0.9, "rule_telemetry": {}}),
                (DriverState.DISTRACTED, {"hybrid_confidence": 0.85, "rule_telemetry": {}}),
                (DriverState.DROWSY, {"hybrid_confidence": 0.92, "rule_telemetry": {}}),
                (DriverState.ALERT, {"hybrid_confidence": 1.0, "rule_telemetry": {}}),
            ]

            app.run(no_gui=False)

        drowsy_sound.play.assert_called_once()
        distracted_sound.play.assert_called_once()
        drowsy_channel.stop.assert_called()
        distracted_channel.stop.assert_called()
        mock_pygame.mixer.quit.assert_called_once()
        self.assertFalse(app.alert_manager.is_initialized)

    def test_app_releases_audio_on_camera_exception(self) -> None:
        """Verify AlertManager.close() is called in finally even if camera fails to open."""
        mock_pygame, _, _, _, _ = _create_mock_pygame()
        mock_cv2 = MagicMock()

        with patch("alerts.audio_alert.pygame", mock_pygame), patch("main.cv2", mock_cv2):
            app = DriverMonitoringApp()
            app.camera = MagicMock()
            app.camera.camera_index = 0
            app.camera.width = 640
            app.camera.height = 480
            app.camera.fps = 30
            app.camera.__enter__.side_effect = RuntimeError("Camera hardware busy")

            app.run(no_gui=False)

        self.assertFalse(app.alert_manager.is_initialized)
        mock_pygame.mixer.quit.assert_called_once()

    def test_visualizer_draw_audio_status_badges(self) -> None:
        """Verify Visualizer.draw_audio_status renders ON, MUTED, and UNAVAILABLE badges."""
        mock_cv2 = MagicMock()
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.LINE_AA = 16
        vis = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("visualization.draw_utils.cv2", mock_cv2):
            for label in ("AUDIO: ON", "AUDIO: MUTED", "AUDIO: UNAVAILABLE"):
                out = vis.draw_audio_status(frame, label)
                self.assertIs(out, frame)

        rendered_labels = [call.args[1] for call in mock_cv2.putText.call_args_list]
        self.assertEqual(rendered_labels, ["AUDIO: ON", "AUDIO: MUTED", "AUDIO: UNAVAILABLE"])


if __name__ == "__main__":
    unittest.main()
