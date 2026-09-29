"""Audio playback with a sample-accurate clock."""

from __future__ import annotations

import threading
import time as _time
from pathlib import Path

import numpy as np


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Returns (frames x channels float32, sr)."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, int(sr)
    except Exception:
        import librosa

        y, sr = librosa.load(str(path), sr=None, mono=False)
        y = np.atleast_2d(y).T.astype(np.float32)
        return y, int(sr)


class Player:
    """Plays an array via a sounddevice callback; time() is the song time currently at the speaker."""

    def __init__(self, data: np.ndarray, sr: int, gain: float = 1.0):
        self.data = np.ascontiguousarray(data if data.ndim == 2 else data[:, None], dtype=np.float32) * gain
        self.sr = sr
        self.frames = 0
        self.stream = None
        self.finished = threading.Event()
        self.latency = 0.0

    def _callback(self, outdata, nframes, time_info, status) -> None:
        import sounddevice as sd

        chunk = self.data[self.frames : self.frames + nframes]
        outdata[: len(chunk)] = chunk
        if len(chunk) < nframes:
            outdata[len(chunk) :] = 0
            self.frames += len(chunk)
            self.finished.set()
            raise sd.CallbackStop
        self.frames += nframes

    def open(self) -> None:
        """Create the stream without playing, so a bad audio device fails before any motion."""
        import sounddevice as sd

        self.stream = sd.OutputStream(samplerate=self.sr, channels=self.data.shape[1], dtype="float32",
                                      callback=self._callback, latency="low")
        self.latency = float(self.stream.latency)

    def start(self) -> None:
        if self.stream is None:
            self.open()
        self.stream.start()

    def time(self) -> float:
        return self.frames / self.sr - self.latency

    @property
    def done(self) -> bool:
        return self.finished.is_set()

    @property
    def duration(self) -> float:
        return len(self.data) / self.sr

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class NullPlayer:
    """Silent stand-in. Wall clock by default; simulated clock advances only via advance()."""

    def __init__(self, duration: float, simulated: bool = False):
        self._duration = duration
        self.simulated = simulated
        self.t0 = 0.0
        self.sim_t = 0.0

    def open(self) -> None:
        pass

    def start(self) -> None:
        self.t0 = _time.perf_counter()

    def advance(self, dt: float) -> None:
        self.sim_t += dt

    def time(self) -> float:
        return self.sim_t if self.simulated else _time.perf_counter() - self.t0

    @property
    def done(self) -> bool:
        return self.time() >= self._duration

    @property
    def duration(self) -> float:
        return self._duration

    def stop(self) -> None:
        pass
