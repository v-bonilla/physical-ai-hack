"""Choreography data model: poses stored against beat phase."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ENERGIES = ("low", "mid", "high")
MAX_SWAY = 30.0


@dataclass
class Choreo:
    name: str
    length_beats: int
    recorded_bpm: float
    samples: np.ndarray  # shape (N, 1 + n_joints), column 0 is beat phase
    mode: str = "keyframe"  # "keyframe" | "continuous"
    energy: str = "mid"
    relative: bool = False  # samples are offsets added to the pose read at connect
    tags: list[str] = field(default_factory=list)
    joints: list[str] = field(default_factory=lambda: list(JOINTS))

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=float)
        if self.samples.ndim != 2 or self.samples.shape[1] != 1 + len(self.joints):
            raise ValueError(f"{self.name}: samples must be rows of [phase, {len(self.joints)} joints]")
        if self.energy not in ENERGIES:
            raise ValueError(f"{self.name}: energy must be one of {ENERGIES}")
        if self.mode not in ("keyframe", "continuous"):
            raise ValueError(f"{self.name}: mode must be keyframe or continuous")
        if self.length_beats <= 0 or len(self.samples) == 0:
            raise ValueError(f"{self.name}: empty choreography")
        self.samples[:, 0] %= self.length_beats
        self.samples = self.samples[np.argsort(self.samples[:, 0], kind="stable")]

    @property
    def phases(self) -> np.ndarray:
        return self.samples[:, 0]

    @property
    def poses(self) -> np.ndarray:
        return self.samples[:, 1:]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "energy": self.energy,
            "tags": list(self.tags),
            "length_beats": int(self.length_beats),
            "recorded_bpm": float(self.recorded_bpm),
            "mode": self.mode,
            "relative": self.relative,
            "joints": list(self.joints),
            "samples": [[round(float(v), 4) for v in row] for row in self.samples],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Choreo":
        return cls(
            name=d["name"],
            energy=d.get("energy", "mid"),
            tags=list(d.get("tags", [])),
            length_beats=int(d["length_beats"]),
            recorded_bpm=float(d["recorded_bpm"]),
            mode=d.get("mode", "keyframe"),
            relative=bool(d.get("relative", False)),
            joints=list(d.get("joints", JOINTS)),
            samples=np.asarray(d["samples"], dtype=float),
        )


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def pose_at(choreo: Choreo, beat_phase: float) -> np.ndarray:
    """Pose at a beat phase, looping modulo length_beats. Keyframes are hit exactly on their phase."""
    L = float(choreo.length_beats)
    p = float(beat_phase) % L
    ph, poses = choreo.phases, choreo.poses
    n = len(ph)
    if n == 1:
        return poses[0].copy()
    i = int(np.searchsorted(ph, p, side="right")) - 1
    if i < 0:  # before the first keyframe: interpolate from the last one, wrapped
        i = n - 1
        p0 = ph[i] - L
    else:
        p0 = ph[i]
    j = (i + 1) % n
    p1 = ph[j] if j > i else ph[j] + L
    span = p1 - p0
    x = 0.0 if span <= 1e-9 else (p - p0) / span
    if choreo.mode == "keyframe":
        x = smoothstep(x)
    return poses[i] + (poses[j] - poses[i]) * x


def sway(amplitude: float = 15.0) -> Choreo:
    """Built-in move: shoulder_pan +A on even beats, -A on odd beats, relative to the start pose."""
    a = float(np.clip(abs(amplitude), 0.0, MAX_SWAY))
    offsets = np.zeros((2, 1 + len(JOINTS)))
    offsets[0, 0], offsets[0, 1] = 0.0, a
    offsets[1, 0], offsets[1, 1] = 1.0, -a
    return Choreo(name="sway", length_beats=2, recorded_bpm=100.0, samples=offsets, relative=True)


def save_choreo(choreo: Choreo, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(choreo.to_dict(), indent=1))
    return path


def load_choreo(path: str | Path) -> Choreo:
    return Choreo.from_dict(json.loads(Path(path).read_text()))
