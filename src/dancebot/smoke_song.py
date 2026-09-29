"""An original 12-bar test groove, synthesized offline with a known tempo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 44100
BARS = 12
PEAK_DBFS = -1.0
DEFAULT_PATH = Path("songs/smoke-groove-110.wav")

# Am, F, C, G as MIDI notes (bass root, triad)
CHORDS = [(45, (57, 60, 64)), (41, (53, 57, 60)), (48, (55, 60, 64)), (43, (55, 59, 62))]


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def _kick(rng) -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * 35)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    click = rng.standard_normal(n) * np.exp(-t * 400) * 0.3
    return body + click


def _snare(rng) -> np.ndarray:
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    noise = noise - np.convolve(noise, np.ones(8) / 8, mode="same")  # crude high-pass
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)
    # clap: three quick bursts layered on the snare
    clap = np.zeros(n)
    for k, d in enumerate((0.0, 0.011, 0.022)):
        s = int(d * SR)
        m = n - s
        clap[s:] += rng.standard_normal(m) * np.exp(-np.arange(m) / SR * (90 if k < 2 else 25))
    return 0.6 * noise * np.exp(-t * 18) + 0.5 * tone + 0.35 * clap


def _hat(rng, decay: float = 45.0, n_s: float = 0.06) -> np.ndarray:
    n = int(n_s * SR)
    x = rng.standard_normal(n)
    x = np.diff(np.diff(x, prepend=0), prepend=0)  # emphasize highs
    return 0.25 * x * np.exp(-np.arange(n) / SR * decay)


def _crash(rng) -> np.ndarray:
    n = int(1.8 * SR)
    x = np.diff(rng.standard_normal(n), prepend=0)
    return 0.35 * x * _env(n, 0.002, 0.6)


def _bass(midi: int, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    f = _hz(midi)
    wave = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 6))  # soft saw
    return 0.45 * wave * _env(n, 0.004, 0.12)


def _stab(notes, n: int) -> np.ndarray:
    t = np.arange(n) / SR
    out = np.zeros(n)
    for m in notes:
        f = _hz(m)
        out += sum(np.sin(2 * np.pi * f * k * t) * (0.5 ** k) for k in range(1, 4))
    return 0.22 * out * _env(n, 0.003, 0.09)


def _add(buf: np.ndarray, x: np.ndarray, at: int, pan: float = 0.0) -> None:
    """Mix mono x into stereo buf at sample `at`; pan in -1..1."""
    if at >= len(buf):
        return
    x = x[: len(buf) - at]
    buf[at : at + len(x), 0] += x * np.sqrt((1 - pan) / 2) * np.sqrt(2)
    buf[at : at + len(x), 1] += x * np.sqrt((1 + pan) / 2) * np.sqrt(2)


def synth(bpm: float = 110.0, seed: int = 110) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spb = 60.0 / bpm
    beat_n = spb * SR
    total = int(round(BARS * 4 * beat_n)) + int(1.0 * SR)  # one second of tail
    buf = np.zeros((total, 2))
    kick, snare, crash = _kick(rng), _snare(rng), _crash(rng)

    def at(bar: int, beat: float) -> int:
        return int(round((bar * 4 + beat) * beat_n))

    for bar in range(BARS):
        full = bar >= 2
        root, triad = CHORDS[(bar - 2) % 4] if full else CHORDS[0]
        for eighth in range(8):
            beat = eighth / 2
            _add(buf, _hat(rng) * (1.0 if eighth % 2 else 0.7), at(bar, beat), pan=0.3)
            if full:
                _add(buf, _bass(root if eighth % 2 == 0 else root + 12, int(beat_n / 2)), at(bar, beat))
        kicks = (0, 1, 2, 3) if not full else (0, 2)
        for k in kicks:
            _add(buf, kick, at(bar, k))
        if full:
            for k in (1, 3):
                _add(buf, snare, at(bar, k), pan=-0.05)
                _add(buf, _stab(triad, int(0.25 * beat_n * 2)), at(bar, k + 0.5), pan=-0.35)
        if bar == 2:
            _add(buf, crash, at(bar, 0), pan=0.2)
    return buf


def make(path: str | Path = DEFAULT_PATH, bpm: float = 110.0) -> Path:
    import soundfile as sf

    y = synth(bpm)
    y *= 10 ** (PEAK_DBFS / 20) / np.abs(y).max()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), SR, subtype="PCM_16")
    return path
