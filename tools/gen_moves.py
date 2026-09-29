"""Generate dance moves as building blocks - no recording needed.

Every move is a seamless loop in the same format as recorded moves (moves/gen_<name>.json), so
dance.py and a choreography agent can chain them freely. Each move carries an energy tag and a short
description for the agent.

Moves are defined as offsets around a neutral pose and are always clipped to the joint range the
arm already visited safely in your recorded moves (the "envelope"). Beats are integers: turning
points on integer beats land on the music's beat.

Usage:
    python tools/gen_moves.py            # writes moves/gen_*.json and moves/gen_*.png
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_move as rm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MOVES_DIR = ROOT / "moves"
JOINTS = rm.JOINTS
PPB = rm.POINTS_PER_BEAT
SONG_BPM = 129
#                    pan   lift  elbow wflex roll  grip
NEUTRAL = np.array([-12.0, -70.0, 30.0, 45.0, 10.0, 10.0])
# Measured with tools/video_sync.py (IMG_4171): the wave's visible reversals came ~0.3 beats late.
WAVE_VISUAL_SHIFT = 0.3


def envelope() -> tuple[np.ndarray, np.ndarray]:
    """Joint range covered by the recorded (non-generated) moves."""
    loops = []
    for f in MOVES_DIR.glob("*.json"):
        d = json.loads(f.read_text())
        if not d.get("generated"):
            loops.append(np.array(d["loop"]))
    if not loops:
        raise SystemExit("Keine aufgenommenen Moves gefunden - der Sicherheitsbereich stammt aus ihnen.")
    q = np.vstack(loops)
    return q.min(0), q.max(0)


def grid(beats: int) -> np.ndarray:
    return np.arange(beats * PPB) / PPB


def sines(beats: int, base=(0, 0, 0, 0, 0, 0), parts=()) -> np.ndarray:
    """Sum of sinusoids: parts = [(joint, amplitude, period_beats, phase_rad), ...]."""
    b = grid(beats)
    q = np.tile(NEUTRAL + np.array(base, float), (len(b), 1))
    for j, amp, period, phase in parts:
        q[:, JOINTS.index(j)] += amp * np.sin(2 * np.pi * b / period + phase)
    return q


def hits(beats: int, hit_beats, amounts: dict, attack: float = 0.45, release: float = 1.2) -> np.ndarray:
    """Percussive accent (clap, flick): snaps INTO the accent pose, arriving exactly on each hit beat,
    then eases back out over `release` beats. `amounts` = joint offsets at the moment of the hit."""
    b = grid(beats)
    w = np.zeros(len(b))
    for h in hit_beats:
        d = (b - h) % beats  # beats since this hit (wrapping)
        after = np.where(d < release, 0.5 + 0.5 * np.cos(np.pi * d / release), 0.0)
        u = np.clip((d - (beats - attack)) / attack, 0, 1)  # the attack right before the hit
        # Ease-IN: slow start, fastest at the moment of the hit -> a crisp snap instead of a soft landing.
        before = np.where(d > beats - attack, 1.0 - np.cos(0.5 * np.pi * u), 0.0)
        w = np.maximum(w, np.maximum(after, before))
    q = np.tile(NEUTRAL, (len(b), 1))
    for j, a in amounts.items():
        q[:, JOINTS.index(j)] += w * a
    return q


def keyframes(beats: int, frames: list[tuple[float, dict]], base=(0, 0, 0, 0, 0, 0)) -> np.ndarray:
    """Hold-and-hit style: poses at given beats, eased (zero speed) at each keyframe; loops around."""
    b = grid(beats)
    times = [t for t, _ in frames] + [frames[0][0] + beats]
    poses = [np.array([f.get(j, 0.0) for j in JOINTS]) for _, f in frames]
    poses.append(poses[0])
    q = np.zeros((len(b), len(JOINTS)))
    for i, t in enumerate(b):
        tt = t if t >= times[0] else t + beats
        k = max(0, np.searchsorted(times, tt, side="right") - 1)
        u = (tt - times[k]) / (times[k + 1] - times[k])
        w = 0.5 - 0.5 * np.cos(np.pi * u)
        q[i] = (1 - w) * poses[k] + w * poses[k + 1]
    return q + NEUTRAL + np.array(base, float)


def keyframes_hold(beats: int, frames: list[tuple[float, dict]], move_frac: float = 0.35, base=(0,) * 6) -> np.ndarray:
    """Robot/staccato style: snap into each pose and ARRIVE exactly at its beat, then freeze until the next.

    The eye reads the arrival (the stop) as the hit, so the motion happens in the `move_frac` beats
    BEFORE each keyframe. (Measured in video: moving after the beat looked half a beat late.)
    """
    return np.roll(_keyframes_hold_after(beats, frames, move_frac, base), -int(round(move_frac * PPB)), axis=0)


def _keyframes_hold_after(beats, frames, move_frac, base):
    b = grid(beats)
    times = [t for t, _ in frames]
    poses = [np.array([f.get(j, 0.0) for j in JOINTS]) for _, f in frames]
    q = np.zeros((len(b), len(JOINTS)))
    for i, t in enumerate(b):
        k = max(0, np.searchsorted(times, t, side="right") - 1)
        prev = poses[k - 1] if k > 0 else poses[-1]
        u = min(1.0, (t - times[k]) / move_frac)
        w = 0.5 - 0.5 * np.cos(np.pi * u)
        q[i] = (1 - w) * prev + w * poses[k]
    return q + NEUTRAL + np.array(base, float)


# Periods of 1 beat (0.46 s at 129 BPM) are too fast for the servos: measured only 60-70 % of the
# motion and up to +85 ms extra lag. Accents therefore repeat every 2 beats (on 2 and 4 = backbeat).
MOVES = {
    "gen_nod": dict(
        energy="low", beats=4,
        description="Nicken: runter auf 1 und 3, hoch auf 2 und 4 (Ellbogen + gegenlaeufiges Handgelenk).",
        loop=lambda: sines(4, parts=[("elbow_flex", 12, 2, np.pi / 2), ("wrist_flex", -10, 2, np.pi / 2)]),
    ),
    "gen_sway": dict(
        energy="low", beats=4,
        description="Ruhiges Wiegen: Basis links/rechts, ein Zyklus pro Takt, Handgelenk dreht leicht mit.",
        loop=lambda: sines(4, parts=[("shoulder_pan", 25, 4, 0), ("wrist_roll", -10, 4, 0)]),
    ),
    "gen_twist": dict(
        energy="mid", beats=4,
        description="Handgelenk-Twist: Drehung hin und her, alle 2 Beats, Arm bleibt ruhig.",
        loop=lambda: sines(4, base=(0, 0, 0, 0, 20, 0), parts=[("wrist_roll", 25, 2, np.pi / 2)]),
    ),
    "gen_clap": dict(
        energy="mid", beats=4, punch=False, accent=True,
        description="Greifer-Klatschen auf 2 und 4 (Backbeat): schnappt zu und trifft genau den Beat, "
                    "oeffnet sich danach langsam; das Handgelenk schnippt mit.",
        # Open at rest (neutral gripper 10 + 15 = 25 %), snaps shut (-> ~1 %) exactly ON beats 1 and 3 (0-based).
        loop=lambda: hits(4, [1, 3], {"gripper": -24, "wrist_flex": -10}) + np.array([0, 0, 0, 0, 0, 15]),
    ),
    "gen_flow": dict(
        energy="flow", beats=8,
        description="Fliessen (fuer ruhige Teile ohne Bass): langsame, weiche Achterbewegung ueber 2 Takte, "
                    "alle Gelenke gleiten ineinander, nichts Ruckartiges.",
        loop=lambda: sines(8, base=(0, 10, -5, 0, 0, 5), parts=[
            ("shoulder_pan", 18, 8, 0.0), ("shoulder_lift", 10, 4, 0.0), ("elbow_flex", 14, 8, np.pi / 2),
            ("wrist_flex", 18, 8, np.pi), ("wrist_roll", 15, 8, np.pi / 2), ("gripper", 5, 8, 0.0)]),
    ),
    "gen_float": dict(
        energy="flow", beats=8,
        description="Schweben/Atmen (ohne Bass): Arm hebt und senkt sich langsam ueber 2 Takte, "
                    "Handgelenk gleicht weich aus, als wuerde er atmen.",
        loop=lambda: sines(8, base=(0, 15, -10, 0, 10, 8), parts=[
            ("shoulder_lift", 14, 8, -np.pi / 2), ("elbow_flex", -12, 8, -np.pi / 2),
            ("wrist_flex", 16, 8, np.pi / 2), ("shoulder_pan", 8, 8, 0.0)]),
    ),
    "gen_look": dict(
        energy="flow", beats=8,
        description="Umschauen (ruhig): dreht langsam nach links, neigt das Handgelenk wie ein neugieriger Kopf, "
                    "dann nach rechts - weiche Uebergaenge mit kurzen Pausen.",
        loop=lambda: keyframes(8, [(0, dict(shoulder_pan=-15, wrist_flex=8, wrist_roll=15)),
                                   (2, dict(shoulder_pan=-15, wrist_flex=-5, wrist_roll=25)),
                                   (4, dict(shoulder_pan=15, wrist_flex=8, wrist_roll=5)),
                                   (6, dict(shoulder_pan=15, wrist_flex=-5, wrist_roll=0))]),
    ),
    "gen_pendulum": dict(
        energy="mid", beats=4,
        description="Pendel: Basis schwingt links/rechts, der Arm taucht in der Mitte ab und steigt an den Seiten - "
                    "alle Umkehrpunkte auf dem Beat.",
        # lift = cos(2*theta) while pan = sin(theta): up at both sides (beats 1, 3), dip in the middle (0, 2).
        loop=lambda: sines(4, base=(0, 10, -5, 0, 0, 0), parts=[
            ("shoulder_pan", 20, 4, 0.0), ("shoulder_lift", 12, 2, -np.pi / 2), ("wrist_flex", 10, 2, np.pi / 2)]),
    ),
    "gen_robot": dict(
        energy="high", beats=4, punch=False,  # already arrives hard on each beat
        description="Roboter-Tanz: auf jedem Beat ruckartig in eine neue Pose, dann einfrieren (staccato).",
        loop=lambda: keyframes_hold(4, [(0, dict(shoulder_pan=-10, shoulder_lift=8, wrist_roll=8)),
                                        (1, dict(shoulder_pan=-10, elbow_flex=-12, wrist_flex=-8)),
                                        (2, dict(shoulder_pan=10, shoulder_lift=8, wrist_roll=-4)),
                                        (3, dict(shoulder_pan=10, elbow_flex=-12, wrist_flex=8))], 0.5),
    ),
    "gen_disco": dict(
        energy="high", beats=4,
        description="Disco-Zeigen (Saturday Night Fever): Arm schraeg nach oben rechts auf 1, "
                    "schraeg nach unten links auf 3.",
        loop=lambda: keyframes(4, [(0, dict(shoulder_pan=20, shoulder_lift=40, elbow_flex=-30, wrist_flex=-10)),
                                   (2, dict(shoulder_pan=-20, shoulder_lift=-10, elbow_flex=10, wrist_flex=15))]),
    ),
    "gen_hello": dict(
        energy="mid", beats=4,
        description="Winken: Arm hoch, Hand winkt links/rechts, ein Winker pro 2 Beats.",
        # cos phase (pi/2): the hand reverses ON the beat, not between beats.
        loop=lambda: sines(4, base=(0, 60, -30, 0, 20, 10), parts=[("wrist_roll", 25, 2, np.pi / 2),
                                                                  ("wrist_flex", 10, 2, np.pi / 2)]),
    ),
    "gen_pump": dict(
        energy="high", beats=4,
        description="Faust-Pumpen: Arm schnellt auf Beat 1 und 3 hoch, auf 2 und 4 zurueck.",
        loop=lambda: keyframes(4, [(0, dict(shoulder_lift=25, elbow_flex=-25, wrist_flex=-15)), (1, {}),
                                   (2, dict(shoulder_lift=25, elbow_flex=-25, wrist_flex=-15)), (3, {})]),
    ),
    "gen_circle": dict(
        energy="mid", beats=4,
        description="Kreis: der Greifer malt pro Takt einen grossen Kreis in die Luft (Basis + Schulter), "
                    "die Hand dreht leicht mit - ganz links/rechts/oben/unten jeweils genau auf einem Beat.",
        # pan = sin, lift = cos over one bar: extremes of pan on beats 1/3, of lift on beats 0/2.
        loop=lambda: sines(4, base=(0, 20, -10, 0, 10, 10), parts=[
            ("shoulder_pan", 20, 4, 0.0), ("shoulder_lift", 16, 4, np.pi / 2), ("elbow_flex", -10, 4, np.pi / 2),
            ("wrist_roll", 12, 4, 0.0)]),
    ),
    "gen_chop": dict(
        energy="high", beats=4, punch=False, accent=True,
        description="Karate-Hieb: der Arm holt aus und schlaegt auf den Akzent-Beats (2x pro Takt) hart "
                    "nach unten, Greifer schnappt zu - danach langsam wieder hoch.",
        loop=lambda: hits(4, [1, 3], {"shoulder_lift": -18, "elbow_flex": 22, "wrist_flex": 18, "gripper": -10},
                          attack=0.5, release=1.3) + np.array([0, 18, -18, -10, 0, 12]),
    ),
    "gen_wave": dict(
        energy="high", beats=4, punch=False,  # flowing on purpose; looks right as it is
        description="Die Welle: laeuft Beat fuer Beat von der Schulter ueber Ellbogen und Handgelenk bis zur "
                    "Drehung - jedes Gelenk kehrt genau einen Beat nach dem vorigen um. Basis schwingt leicht mit.",
        # Stagger of exactly one beat (pi/2 of the 4-beat period) per joint. The eye follows the whole arm,
        # whose summed motion turned ~0.3 beats late in the video (+140..160 ms), so the loop is advanced
        # by WAVE_VISUAL_SHIFT beats.
        loop=lambda: np.roll(sines(4, base=(0, 30, -15, 0, 0, 0), parts=[
            ("shoulder_lift", 20, 4, 0.0), ("elbow_flex", 28, 4, -np.pi / 2), ("wrist_flex", 35, 4, -np.pi),
            ("wrist_roll", 10, 4, -3 * np.pi / 2), ("shoulder_pan", 12, 4, np.pi / 2)]),
            -int(round(WAVE_VISUAL_SHIFT * PPB)), axis=0),
    ),
}


def main():
    lo, hi = envelope()
    MOVES_DIR.mkdir(exist_ok=True)
    spb = 60.0 / SONG_BPM
    print(f"Sicherheitsbereich: " + ", ".join(f"{j} {a:.0f}..{b:.0f}" for j, a, b in zip(JOINTS, lo, hi)))
    print(f"\n{'Move':10s} {'Energie':7s} Spitze @{SONG_BPM} voll / halb   Hinweis")
    for name, spec in MOVES.items():
        raw = spec["loop"]()
        loop = np.clip(raw, lo, hi)
        clipped = np.abs(raw - loop).max()
        v_full, v_half = (rm.peak_speed(loop, b).max() for b in (SONG_BPM, SONG_BPM / 2))
        out = MOVES_DIR / f"{name}.json"
        out.write_text(json.dumps({
            "name": name, "generated": True, "energy": spec["energy"], "description": spec["description"],
            "recorded_bpm": SONG_BPM, "beats": spec["beats"], "points_per_beat": PPB, "joints": JOINTS,
            "loop": np.round(loop, 2).tolist(),
            # Deliberate advance for the eye (whole-arm timing measured in video), see beat_audit.py.
            "visual_shift_beats": WAVE_VISUAL_SHIFT if name == "gen_wave" else 0.0,
            # dance.py: punch = reshape back-and-forth into "accelerate, stop hard on the beat";
            # accent = hits are moved onto the song's accent beats (where its snare/clap is).
            "punch": spec.get("punch", spec["energy"] != "flow"),
            "accent": spec.get("accent", False),
        }, indent=1))
        rm.save_plot(out.with_suffix(".png"), [loop], loop, spec["beats"])
        note = f"auf Sicherheitsbereich begrenzt (bis {clipped:.0f} Grad)" if clipped > 0.5 else ""
        print(f"{name:10s} {spec['energy']:7s} {v_full:5.0f} / {v_half:5.0f} Grad/s      {note}")
    print(f"\n-> {MOVES_DIR}/gen_*.json")


if __name__ == "__main__":
    main()
