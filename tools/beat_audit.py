"""Check whether each move's direction changes land on the beat.

Viewers read "in sync" from the moments a joint reverses direction. This lists, per move and joint,
where those turning points sit relative to the beat grid (0.0 = on the beat, 0.5 = exactly between
two beats) and flags visible joints that turn around off the beat.

Usage:
    python tools/beat_audit.py            # all moves in moves/
"""

import json
from pathlib import Path

import numpy as np

MOVES_DIR = Path(__file__).resolve().parent.parent / "moves"
MIN_RANGE = 8.0  # joints moving less than this (deg / %) are hardly visible
OFF = 0.2  # turning points further than 0.2 beats from a beat count as off-beat


def turning_points(x: np.ndarray, ppb: int) -> list[float]:
    n, rng = len(x), x.max() - x.min()
    return [k / ppb for k in range(n)
            if (x[k] - x[k - 1]) * (x[(k + 1) % n] - x[k]) < 0 and abs(x[k] - x.mean()) > 0.2 * rng]


def audit(path: Path) -> tuple[str, list[str], bool]:
    d = json.loads(path.read_text())
    loop, ppb = np.array(d["loop"]), int(d["points_per_beat"])
    # Moves advanced on purpose so the WHOLE arm looks on the beat are checked without that advance.
    shift = float(d.get("visual_shift_beats", 0.0))
    loop = np.roll(loop, int(round(shift * ppb)), axis=0)
    lines, bad = [], False
    for j, name in enumerate(d["joints"]):
        rng = float(loop[:, j].max() - loop[:, j].min())
        if rng < MIN_RANGE:
            continue
        tps = turning_points(loop[:, j], ppb)
        offs = [t - round(t) for t in tps]
        off_beat = [t for t, o in zip(tps, offs) if abs(o) > OFF]
        flag = bool(off_beat)
        bad |= flag
        lines.append(f"    {'!!' if flag else 'ok'} {name:13s} Hub {rng:5.1f}  Wendepunkte {[round(t, 2) for t in tps]}")
    return d["name"], lines, bad


def main():
    for p in sorted(MOVES_DIR.glob("*.json")):
        name, lines, bad = audit(p)
        energy = json.loads(p.read_text()).get("energy", "?")
        print(f"{'!!' if bad else 'ok'} {name} ({energy})")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
