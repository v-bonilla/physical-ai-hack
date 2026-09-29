"""Audio -> tempo, beats, downbeats."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SR = 22050
HOP = 512
BEATS_PER_BAR = 4


def file_hash(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_path(audio_path: str | Path) -> Path:
    p = Path(audio_path)
    return p.with_name(p.name + ".analysis.json")


def beat_this_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("beat_this") is not None


def beats_beat_this(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    from beat_this.inference import File2Beats  # optional extra "beats"

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass
    f2b = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = f2b(str(path))
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def beats_librosa(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Beats from librosa; downbeats every 4 beats at the offset with the strongest onsets."""
    import librosa

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    _, frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    if len(beats) == 0:
        return beats, beats
    strength = onset_env[np.clip(frames, 0, len(onset_env) - 1)]
    offsets = [strength[k::BEATS_PER_BAR].mean() if len(strength[k::BEATS_PER_BAR]) else -1.0
               for k in range(BEATS_PER_BAR)]
    return beats, beats[int(np.argmax(offsets))::BEATS_PER_BAR]


def detect_beats(path: str | Path, y: np.ndarray, sr: int, backend: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    if backend in ("auto", "beat_this"):
        try:
            b, d = beats_beat_this(path)
            if len(b) >= 2:
                return b, d, "beat_this"
        except Exception as e:  # not installed, checkpoint download failed, decode error
            if backend == "beat_this":
                raise
            print(f"[analyze] Beat This unavailable ({type(e).__name__}: {e}), using librosa", file=sys.stderr)
    b, d = beats_librosa(y, sr)
    return b, d, "librosa"


def analyze(path: str | Path, backend: str = "auto", use_cache: bool = True) -> dict:
    import librosa

    path = Path(path)
    digest = file_hash(path)
    cp = cache_path(path)
    if use_cache and cp.exists():
        try:
            cached = json.loads(cp.read_text())
            cb = cached.get("backend", "")
            # in auto mode a librosa result is only a fallback: retry once Beat This is installed
            upgrade = backend == "auto" and cb.startswith("librosa") and beat_this_available()
            if cached.get("file_hash") == digest and backend in ("auto", cb) and not upgrade:
                return {**cached, "cached": True}
        except (json.JSONDecodeError, OSError):
            pass

    y, sr = librosa.load(str(path), sr=SR, mono=True)
    duration = float(len(y) / sr)
    beats, downbeats, used = detect_beats(path, y, sr, backend)
    if len(beats) < 2:
        # silence or very short audio: a 120 BPM grid keeps the rest of the pipeline working
        beats = np.arange(0.0, max(duration, 1.0), 0.5)
        downbeats = beats[::BEATS_PER_BAR]
        used += "+grid"
    ibi = float(np.median(np.diff(beats)))
    result = {
        "file": path.name,
        "file_hash": digest,
        "backend": used,
        "duration": round(duration, 4),
        "tempo_bpm": round(60.0 / ibi, 3),
        "beats": [round(float(b), 4) for b in beats],
        "downbeats": [round(float(b), 4) for b in downbeats],
    }
    if use_cache:
        try:
            cp.write_text(json.dumps(result, indent=1))
        except OSError:
            pass
    return result


def summarize(a: dict) -> str:
    return (f"{a['file']}: {a['tempo_bpm']:.1f} BPM, {a['duration']:.1f}s, {len(a['beats'])} beats, "
            f"{len(a['downbeats'])} downbeats, backend={a['backend']}{' (cached)' if a.get('cached') else ''}")
