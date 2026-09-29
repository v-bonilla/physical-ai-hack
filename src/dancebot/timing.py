"""Song time <-> fractional beat index."""

from __future__ import annotations

import numpy as np


def scale_beats(beats: list[float] | np.ndarray, mult: float) -> np.ndarray:
    """2 inserts midpoints between beats, 0.5 keeps every other beat, 1 leaves them unchanged."""
    b = np.asarray(sorted(beats), dtype=float)
    if mult == 2:
        if len(b) < 2:
            return b
        out = np.empty(2 * len(b) - 1)
        out[0::2] = b
        out[1::2] = (b[:-1] + b[1:]) / 2
        return out
    if mult == 0.5:
        return b[::2]
    if mult != 1:
        raise ValueError("beat multiplier must be 0.5, 1 or 2")
    return b


class BeatClock:
    def __init__(self, beats: list[float] | np.ndarray, fallback_ibi: float = 0.5):
        self.beats = np.asarray(sorted(beats), dtype=float)
        if len(self.beats) >= 2:
            self.ibi = float(np.median(np.diff(self.beats)))
        else:
            self.ibi = fallback_ibi
        if len(self.beats) == 0:
            self.beats = np.array([0.0])

    def beat_at(self, t: float) -> float:
        b = self.beats
        if t <= b[0]:
            return (t - b[0]) / self.ibi
        if t >= b[-1]:
            return (len(b) - 1) + (t - b[-1]) / self.ibi
        i = int(np.searchsorted(b, t, side="right")) - 1
        return i + (t - b[i]) / (b[i + 1] - b[i])

    def time_at(self, beat: float) -> float:
        b = self.beats
        n = len(b)
        if beat <= 0:
            return b[0] + beat * self.ibi
        if beat >= n - 1:
            return b[-1] + (beat - (n - 1)) * self.ibi
        i = int(np.floor(beat))
        return b[i] + (beat - i) * (b[i + 1] - b[i])
