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
    """Pose at a beat phase, looping modulo length_beats. Keyframes are hit exactly on their phase.

    A NaN sample means "no keyframe for this joint here": each joint then eases between its own
    keyframes, so one joint's half-beat keys do not stall another joint mid-swing.
    """
    poses = choreo.poses
    if np.isnan(poses).any():
        out = np.empty(poses.shape[1])
        for j in range(poses.shape[1]):
            keep = ~np.isnan(poses[:, j])
            out[j] = _interp(choreo.phases[keep], poses[keep, j : j + 1], choreo, beat_phase)[0]
        return out
    return _interp(choreo.phases, poses, choreo, beat_phase)


def _interp(ph: np.ndarray, poses: np.ndarray, choreo: Choreo, beat_phase: float) -> np.ndarray:
    L = float(choreo.length_beats)
    p = float(beat_phase) % L
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


def groove() -> Choreo:
    """Built-in 2-bar move, offsets from the start pose (degrees; gripper in its 0..100 units).

    Bar 1 "sway and nod": pan +12/-12 on beats, wrist_flex -10 on beats and 0 on half beats.
    Bar 2 "twist and clap": wrist_roll +20/-20 on beats, gripper opens +25 on 4.5 and 6.5, shut on beats.
    shoulder_lift and elbow_flex never move (gravity-loaded, table risk).
    """
    n = np.nan
    #        phase  pan  lift elbow wflex wroll grip
    rows = [
        [0.0,  12,  0,   0,   -10,   0,    0],
        [0.5,   n,  n,   n,     0,   n,    n],
        [1.0, -12,  n,   n,   -10,   0,    0],
        [1.5,   n,  n,   n,     0,   n,    n],
        [2.0,  12,  n,   n,   -10,   0,    0],
        [2.5,   n,  n,   n,     0,   n,    n],
        [3.0, -12,  n,   n,   -10,   0,    0],
        [3.5,   n,  n,   n,     0,   n,    n],
        [4.0,   0,  n,   n,     0,  20,    0],
        [4.5,   n,  n,   n,     n,   n,   25],
        [5.0,   0,  n,   n,     0, -20,    0],
        [6.0,   0,  n,   n,     0,  20,    0],
        [6.5,   n,  n,   n,     n,   n,   25],
        [7.0,   0,  n,   n,     0, -20,    0],
    ]
    return Choreo(name="groove", length_beats=8, recorded_bpm=110.0, samples=np.array(rows, float), relative=True)


# absolute caps on any move's offsets after --scale
OFFSET_CAPS = np.array([30.0, 0.0, 0.0, 20.0, 35.0, 40.0])


def scaled(choreo: Choreo, scale: float) -> Choreo:
    """Multiply a relative move's offsets by scale, then cap them per joint; lift and elbow stay 0."""
    s = choreo.samples.copy()
    s[:, 1:] = np.clip(s[:, 1:] * scale, -OFFSET_CAPS, OFFSET_CAPS)
    s[:, 1:] = np.where(np.isnan(choreo.samples[:, 1:]), np.nan, s[:, 1:])
    return Choreo(name=choreo.name, length_beats=choreo.length_beats, recorded_bpm=choreo.recorded_bpm,
                  samples=s, mode=choreo.mode, energy=choreo.energy, relative=choreo.relative,
                  tags=list(choreo.tags), joints=list(choreo.joints))


def save_choreo(choreo: Choreo, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(choreo.to_dict(), indent=1))
    return path


def load_choreo(path: str | Path) -> Choreo:
    return Choreo.from_dict(json.loads(Path(path).read_text()))


MOVES = ("groove", "sway")
MAX_SCALE = 1.5


def make_move(name: str = "groove", scale: float = 1.0, amplitude: float = 15.0) -> Choreo:
    if name not in MOVES:
        raise ValueError(f"unknown move {name!r}; choose from {MOVES}")
    if not (0 < scale <= MAX_SCALE):
        raise ValueError(f"scale must be in (0, {MAX_SCALE}]")
    return scaled(groove() if name == "groove" else sway(amplitude), scale)
