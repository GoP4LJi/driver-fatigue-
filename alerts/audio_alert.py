"""Audio alert notification system powered by Pygame Mixer.

Dispatches non-blocking auditory warning alarms when hazardous driver states
(sustained eye closure, continuous yawning, or persistent distraction) are detected.
Includes cooldown enforcement, mute/unmute controls, graceful missing-dependency
handling, and clean resource teardown.
"""

import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

try:
    import pygame
except ImportError:
    pygame = None

from config.settings import AlertConfig, DriverState, settings

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages audio playback, sound assets, volume levels, and alert cooldown intervals.

    Attributes:
        enabled (bool): Global toggle to enable or mute sound playback.
        cooldown (float): Minimum delay in seconds between consecutive audible alerts.
        volume (float): Master volume scaling from 0.0 to 1.0.
        sound_paths (Dict[DriverState, Path]): Mapping from hazardous states to WAV files.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        cooldown: Optional[float] = None,
        volume: Optional[float] = None,
        sound_paths: Optional[Dict[DriverState, Path]] = None,
        config: Optional[AlertConfig] = None,
    ) -> None:
        """Initialize alert manager configuration and sound asset mappings.

        Args:
            enabled: Whether audio playback is enabled (unmuted). Defaults to config.
            cooldown: Minimum interval in seconds between consecutive alerts. Defaults to config.
            volume: Master playback volume in [0.0, 1.0]. Defaults to config.
            sound_paths: Optional state-to-WAV-path overrides for DROWSY / DISTRACTED.
            config: Optional AlertConfig instance (defaults to global settings.alerts).
        """
        cfg = config or settings.alerts
        self._config: AlertConfig = cfg

        self.enabled: bool = cfg.enable_audio if enabled is None else bool(enabled)
        self.cooldown: float = max(0.0, float(cfg.cooldown_seconds if cooldown is None else cooldown))
        raw_vol = cfg.volume if volume is None else float(volume)
        self.volume: float = max(0.0, min(1.0, raw_vol))

        if sound_paths is not None:
            self.sound_paths: Dict[DriverState, Path] = {
                state: Path(path) for state, path in sound_paths.items()
            }
        else:
            self.sound_paths = {
                DriverState.DROWSY: Path(cfg.sound_drowsy_path),
                DriverState.DISTRACTED: Path(cfg.sound_distracted_path),
            }

        self.last_alert_time: float = 0.0
        self._last_alert_time: Optional[float] = None
        self._initialized: bool = False
        self._available: bool = False
        self._unavailable_reason: str = "Not initialized"
        self._sounds: Dict[DriverState, Any] = {}
        self._active_channel: Optional[Any] = None

    @property
    def is_initialized(self) -> bool:
        """Return True if initialize() succeeded and mixer is active."""
        return self._initialized

    @property
    def is_available(self) -> bool:
        """Return True if the audio subsystem is available for playback."""
        return self._available

    @property
    def is_muted(self) -> bool:
        """Return True if audio alerts are currently muted."""
        return not self.enabled

    @property
    def unavailable_reason(self) -> str:
        """Return diagnostic description when audio subsystem is unavailable."""
        return self._unavailable_reason

    def get_status_label(self) -> str:
        """Return standardized HUD status label for the audio subsystem."""
        if not self._initialized or not self._available:
            return "AUDIO: UNAVAILABLE"
        if not self.enabled:
            return "AUDIO: MUTED"
        return "AUDIO: ON"

    def initialize(self) -> bool:
        """Initialize the Pygame audio mixer subsystem and preload audio clips.

        Returns:
            bool: True if mixer initialization succeeded, False if pygame is missing
            or audio hardware initialization failed.
        """
        if self._initialized and self._available:
            return True

        if pygame is None or getattr(pygame, "mixer", None) is None:
            self._initialized = False
            self._available = False
            self._unavailable_reason = "pygame is not installed"
            logger.warning("Audio alerts unavailable: pygame is not installed.")
            return False

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=self._config.mixer_frequency,
                    size=self._config.mixer_size,
                    channels=self._config.mixer_channels,
                    buffer=self._config.mixer_buffer,
                )
            if hasattr(pygame.mixer, "set_num_channels"):
                pygame.mixer.set_num_channels(max(1, self._config.max_channels))
        except Exception as exc:
            self._initialized = False
            self._available = False
            self._unavailable_reason = f"Audio mixer init failed: {exc}"
            logger.warning("Audio alerts unavailable: %s", exc)
            return False

        self._sounds.clear()
        for state, path in self.sound_paths.items():
            resolved = Path(path)
            if not resolved.exists() or not resolved.is_file():
                logger.warning(
                    "Audio alert asset missing for %s: %s (skipping without substitution)",
                    state.value,
                    resolved,
                )
                continue
            try:
                sound = pygame.mixer.Sound(str(resolved))
                sound.set_volume(self.volume)
                self._sounds[state] = sound
            except Exception as exc:
                logger.warning(
                    "Failed to load audio asset for %s (%s): %s",
                    state.value,
                    resolved,
                    exc,
                )

        self._initialized = True
        self._available = True
        self._unavailable_reason = ""
        return True

    def trigger_alert(self, state: DriverState) -> bool:
        """Play appropriate warning sound corresponding to the active hazardous state.

        Non-blocking playback via Pygame Mixer channels. Returning to ALERT stops
        currently active playback (when configured). Repeated triggers within the
        cooldown window are suppressed.

        Args:
            state (DriverState): The detected driver state (ALERT, DROWSY, or DISTRACTED).

        Returns:
            bool: True if an audio warning was successfully triggered; False if stopped
            on ALERT, suppressed due to cooldown, muted, unavailable, or missing asset.
        """
        if state == DriverState.ALERT:
            if self._config.stop_on_alert:
                self.stop_all()
            return False

        if not self.enabled or not self._initialized or not self._available:
            return False

        if state not in (DriverState.DROWSY, DriverState.DISTRACTED):
            return False

        sound = self._sounds.get(state)
        if sound is None:
            return False

        now = time.monotonic()
        if self._last_alert_time is not None and (now - self._last_alert_time) < self.cooldown:
            return False

        try:
            # Avoid overlapping sounds from an earlier active alert channel
            if self._active_channel is not None and hasattr(self._active_channel, "get_busy"):
                if self._active_channel.get_busy() and hasattr(self._active_channel, "stop"):
                    self._active_channel.stop()

            self._active_channel = sound.play()
            self._last_alert_time = now
            self.last_alert_time = now
            return True
        except Exception as exc:
            logger.warning("Audio playback failed for state %s: %s", state.value, exc)
            return False

    def set_enabled(self, enabled: bool) -> None:
        """Enable or mute audio alert playback.

        When muted (enabled=False), any currently playing alert sound is stopped.
        """
        self.enabled = bool(enabled)
        if not self.enabled:
            self.stop_all()

    def toggle_mute(self) -> bool:
        """Toggle audio mute state and return the resulting enabled status.

        Returns:
            bool: True if audio is now unmuted (ON); False if now muted (MUTED).
        """
        self.set_enabled(not self.enabled)
        return self.enabled

    def set_volume(self, volume: float) -> None:
        """Update master playback volume (clamped to [0.0, 1.0]) across loaded sounds."""
        self.volume = max(0.0, min(1.0, float(volume)))
        for sound in self._sounds.values():
            if hasattr(sound, "set_volume"):
                try:
                    sound.set_volume(self.volume)
                except Exception as exc:
                    logger.warning("Failed to update sound volume: %s", exc)

    def stop_all(self) -> None:
        """Immediately silence all actively playing audio channels."""
        if not self._initialized or not self._available or pygame is None:
            self._active_channel = None
            return

        try:
            if self._active_channel is not None and hasattr(self._active_channel, "stop"):
                self._active_channel.stop()
            mixer = getattr(pygame, "mixer", None)
            if mixer is not None and hasattr(mixer, "stop"):
                if not hasattr(mixer, "get_init") or mixer.get_init():
                    mixer.stop()
        except Exception as exc:
            logger.warning("Error stopping audio channels: %s", exc)
        finally:
            self._active_channel = None

    def close(self) -> None:
        """De-initialize mixer and release audio hardware resources safely."""
        self.stop_all()
        self._sounds.clear()

        if self._initialized and pygame is not None:
            mixer = getattr(pygame, "mixer", None)
            if mixer is not None and hasattr(mixer, "quit"):
                try:
                    if not hasattr(mixer, "get_init") or mixer.get_init():
                        mixer.quit()
                except Exception as exc:
                    logger.warning("Error shutting down pygame mixer: %s", exc)

        self._initialized = False
        self._available = False

    def __enter__(self) -> "AlertManager":
        """Enter runtime context and initialize audio subsystem."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit runtime context and release audio resources."""
        self.close()
