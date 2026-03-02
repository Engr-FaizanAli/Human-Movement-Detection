"""
alerts.py — Sound alarm, snapshot saving, and event persistence.

Sound: pygame.mixer (falls back silently if pygame not available)
Snapshots: saved as JPEG under %APPDATA%\\MotionGuard\\snapshots\\
Events: written to the DB via repositories
"""

import datetime
import logging
import math
import struct
import wave
from pathlib import Path

import numpy as np

from storage import repositories as repo
from utils.paths import alarm_wav_path, snapshots_dir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WAV generator — creates a fallback beep if no alarm.wav present
# ---------------------------------------------------------------------------

def _generate_beep_wav(path: Path, frequency: int = 880, duration: float = 0.6) -> None:
    """Write a simple sine-wave WAV file to path."""
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(path), "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for i in range(num_samples):
                # Fade in/out for 5% of duration
                fade = min(i, num_samples - i, int(num_samples * 0.05))
                envelope = min(1.0, fade / (num_samples * 0.05))
                value = int(32767 * envelope * math.sin(2 * math.pi * frequency * i / sample_rate))
                wav.writeframes(struct.pack("<h", value))
        log.info("Generated fallback alarm WAV: %s", path)
    except Exception as exc:
        log.warning("Could not generate alarm WAV: %s", exc)


def _ensure_alarm_wav() -> Path:
    path = alarm_wav_path()
    if not path.exists():
        _generate_beep_wav(path)
    return path


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Manages sound alarms, snapshot saving, and event logging.

    Designed to be owned by the main thread and called via Qt signal connections.
    pygame.mixer runs in the main thread; snapshots are written synchronously
    (small JPEG writes are fast enough not to block the UI appreciably).
    """

    def __init__(self) -> None:
        self._enabled: bool = True
        self._volume: float = 0.8
        self._snapshot_enabled: bool = True
        self._sound = None
        self._pygame_ok = False
        self._alarm_wav: Path = _ensure_alarm_wav()
        self._init_pygame()

    # ------------------------------------------------------------------
    def _init_pygame(self) -> None:
        try:
            import pygame  # type: ignore
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)
            self._sound = pygame.mixer.Sound(str(self._alarm_wav))
            self._sound.set_volume(self._volume)
            self._pygame_ok = True
            log.info("pygame.mixer initialised (alarm ready)")
        except Exception as exc:
            log.warning("pygame.mixer not available — alarm disabled: %s", exc)
            self._pygame_ok = False

    # ------------------------------------------------------------------
    def on_motion_detected(
        self,
        source_id: str,
        source_type: str,
        source_name: str,
        frame: np.ndarray | None = None,
    ) -> None:
        """Call when motion starts on a source."""
        log.info("Motion detected: [%s] %s", source_type, source_name)
        if self._enabled:
            self.play_alarm()
        if self._snapshot_enabled and frame is not None:
            self.save_snapshot(source_id, source_name, frame)

    def on_motion_ended(
        self,
        source_id: str,
        source_type: str,
        source_name: str,
        duration_sec: float,
        snapshot_path: str | None = None,
    ) -> None:
        """Call when motion ends — persists the event to DB."""
        timestamp = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        try:
            repo.add_event(
                source_id=source_id,
                source_type=source_type,
                source_name=source_name,
                timestamp=timestamp,
                duration_sec=duration_sec,
                snapshot_path=snapshot_path,
            )
        except Exception as exc:
            log.error("Failed to save event: %s", exc)

    # ------------------------------------------------------------------
    def play_alarm(self) -> None:
        if not self._pygame_ok or not self._sound:
            return
        try:
            self._sound.play()
        except Exception as exc:
            log.warning("Alarm play failed: %s", exc)

    def stop_alarm(self) -> None:
        if not self._pygame_ok or not self._sound:
            return
        try:
            self._sound.stop()
        except Exception:
            pass

    def test_alarm(self) -> None:
        self.play_alarm()

    # ------------------------------------------------------------------
    def save_snapshot(
        self,
        source_id: str,
        source_name: str,
        frame: np.ndarray,
    ) -> str | None:
        """Save a JPEG snapshot and return its path, or None on failure."""
        try:
            import cv2  # type: ignore

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_name)
            filename = f"{safe_name}_{ts}.jpg"
            path = snapshots_dir() / filename
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            log.debug("Snapshot saved: %s", path)
            return str(path)
        except Exception as exc:
            log.error("Snapshot save failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    def cleanup_old_snapshots(self, retention_days: int) -> int:
        """Delete snapshots older than retention_days. Returns count deleted."""
        deleted = 0
        cutoff = datetime.datetime.now() - datetime.timedelta(days=retention_days)
        snap_dir = snapshots_dir()
        for f in snap_dir.glob("*.jpg"):
            try:
                mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    f.unlink()
                    deleted += 1
            except Exception:
                pass
        if deleted:
            log.info("Deleted %d old snapshot(s) (retention=%dd)", deleted, retention_days)
        # Also clean up DB events
        try:
            repo.delete_old_events(retention_days)
        except Exception:
            pass
        return deleted

    # ------------------------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def set_volume(self, volume: float) -> None:
        """volume: 0.0 – 1.0"""
        self._volume = max(0.0, min(1.0, volume))
        if self._sound and self._pygame_ok:
            try:
                self._sound.set_volume(self._volume)
            except Exception:
                pass

    def set_snapshot_enabled(self, enabled: bool) -> None:
        self._snapshot_enabled = enabled

    def set_alarm_file(self, path: str) -> None:
        """Load a new alarm WAV file at runtime."""
        p = Path(path)
        if not p.exists():
            log.warning("Alarm file not found: %s", path)
            return
        self._alarm_wav = p
        self._init_pygame()

    @property
    def is_sound_available(self) -> bool:
        return self._pygame_ok
