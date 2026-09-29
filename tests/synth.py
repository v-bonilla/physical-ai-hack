"""Synthesized test songs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 22050


def make_beat_song(path: str | Path, bpm: float = 120.0, seconds: float = 20.0, seed: int = 0) -> Path:
    """Kick on every beat, accented downbeat, hi-hat on the off-beats, a quiet bass drone."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = 0.1 * np.sin(2 * np.pi * 55.0 * np.arange(n) / SR)
    spb = 60.0 / bpm
    kn = int(0.15 * SR)
    t = np.arange(kn) / SR
    kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-t * 30)) * t) * np.exp(-t * 18)
    hn = int(0.05 * SR)
    for k in range(int(seconds / spb)):
        s = int(k * spb * SR)
        amp = 1.0 if k % 4 == 0 else 0.7
        y[s : s + kn] += amp * kick[: n - s]
        h = int((k + 0.5) * spb * SR)
        if h < n:
            y[h : h + hn] += (0.2 * rng.standard_normal(hn) * np.exp(-np.arange(hn) / SR * 60))[: n - h]
    y /= np.abs(y).max()
    path = Path(path)
    sf.write(str(path), (0.8 * y).astype(np.float32), SR)
    return path
