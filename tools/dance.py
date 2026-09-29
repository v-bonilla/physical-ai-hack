"""Make the SO-101 follower dance to a song using move loops (moves/*.json, recorded or generated).

Pipeline:
  1. Song analysis (analyze_song.py): beats, bar starts, energy per bar, sections (intro, verse,
     chorus, breakdown, outro) - sections of the same type are recognized as repeats.
  2. Choreography: a list of segments {start, move, tempo, amp}:
       auto  - section label picks the move family (by the moves' energy tags), the same section
               type always gets the same move sequence, section energy scales the move size (amp)
       test  - every move for 4 bars, one after another
       JSON  - a plan written by a person or an agent (see --catalog and outputs/plan_<song>.json)
  3. Playback: the song's own beat grid drives the move phase; moves are time-stretched (full / half
     tempo), scaled around their center (amp), cross-faded over one beat and speed-limited per joint.

Safety: the arm first glides from its current pose to the first dance pose, and at the end (or on
Ctrl+C) glides back to where it started before the motors are released.

Usage:
    python tools/dance.py inputs/pdoom.wav --dry-run            # plan + plot + plan JSON, no arm
    python tools/dance.py inputs/pdoom.wav --catalog            # move list for an agent
    python tools/dance.py inputs/pdoom.wav --plan test --only "gen_*" --start 0 --duration 60
    python tools/dance.py inputs/pdoom.wav --plan outputs/plan_pdoom.json
"""

import argparse
import fnmatch
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_song  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MOVES_DIR = ROOT / "moves"
FOLLOWER_PORT = "/dev/tty.usbmodem5B8E1123681"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CONTROL_HZ = 50
FADE_BEATS = 1.0
GLIDE_S = 3.0
RATES = {"full": 1.0, "half": 0.5}
LABEL_FAMILY = {"intro": "low", "breakdown": "low", "outro": "low", "verse": "mid", "chorus": "high"}
FAMILY_FALLBACK = {"flow": ["flow", "low", "mid"], "low": ["low", "flow", "mid"], "mid": ["mid", "low", "high"],
                   "high": ["high", "mid"]}
SLOT_BARS = 2  # a new move at least every 2 bars (8 beats)
CHORUS_ANCHOR = "gen_wave"  # returns on every other slot of a chorus
REST_POSE = np.array([-12.0, -106.0, 0.0, 100.0, 10.0, 0.0])  # stand-in for the arm's rest pose in --dry-run


# ----------------------------------------------------------------------------- moves
@dataclass
class Move:
    name: str
    beats: int
    loop: np.ndarray  # (beats * ppb, 6)
    ppb: int
    energy: str = "mid"
    description: str = ""
    generated: bool = False
    max_tempo: str = "full"  # "half" for moves too big to play at full song tempo

    @classmethod
    def load(cls, path: Path) -> "Move":
        d = json.loads(path.read_text())
        return cls(d["name"], int(d["beats"]), np.array(d["loop"], dtype=float), int(d["points_per_beat"]),
                   d.get("energy", "mid"), d.get("description", ""), bool(d.get("generated", False)),
                   d.get("max_tempo", "full"))

    def bars_needed(self, tempo: str) -> int:
        return int(np.ceil(self.beats / RATES[tempo] / 4))

    @property
    def center(self) -> np.ndarray:
        return self.loop.mean(axis=0)

    def pose(self, phase_beats: float, amp: float = 1.0) -> np.ndarray:
        """Joint targets inside the loop (wrapping), scaled by `amp` around the move's center."""
        x = (phase_beats % self.beats) * self.ppb
        i0 = int(np.floor(x)) % len(self.loop)
        i1 = (i0 + 1) % len(self.loop)
        f = x - np.floor(x)
        p = (1 - f) * self.loop[i0] + f * self.loop[i1]
        return self.center + min(amp, 1.0) * (p - self.center)

    def peak_speed(self, seconds_per_beat: float, rate: float, amp: float = 1.0) -> float:
        dt = seconds_per_beat / rate / self.ppb
        closed = np.vstack([self.loop, self.loop[:1]])
        return float(np.abs(np.diff(closed, axis=0)).max() / dt * min(amp, 1.0))


def load_moves(only: str | None) -> dict[str, Move]:
    moves = {p.stem: Move.load(p) for p in sorted(MOVES_DIR.glob("*.json"))}
    if only:
        pats = [s.strip() for s in only.split(",")]
        moves = {k: v for k, v in moves.items() if any(fnmatch.fnmatch(k, pat) for pat in pats)}
    if not moves:
        raise SystemExit(f"Keine passenden Moves in {MOVES_DIR}.")
    return moves


# ----------------------------------------------------------------------------- music
class Song:
    def __init__(self, analysis: dict):
        self.a = analysis
        self.beats = np.array(analysis["beats"])
        self.phase = int(analysis["downbeat_phase"])
        self.spb = float(analysis["seconds_per_beat"])
        self.sections = analysis["sections"]
        self.bars = analysis["bars"]

    def beat_pos(self, t: float) -> float:
        b = self.beats
        if t < b[0]:
            return (t - b[0]) / np.median(np.diff(b[:8]))
        if t > b[-1]:
            return len(b) - 1 + (t - b[-1]) / np.median(np.diff(b[-8:]))
        return float(np.interp(t, b, np.arange(len(b))))

    def beat_time(self, beat_index: float) -> float:
        b = self.beats
        if beat_index <= 0:
            return b[0] + beat_index * np.median(np.diff(b[:8]))
        if beat_index >= len(b) - 1:
            return b[-1] + (beat_index - len(b) + 1) * np.median(np.diff(b[-8:]))
        return float(np.interp(beat_index, np.arange(len(b)), b))

    def bar_beat(self, bar: float) -> float:
        return self.phase + 4 * bar


# ----------------------------------------------------------------------------- choreography
@dataclass
class Segment:
    start_beat: float
    move: str
    tempo: str  # "full" | "half"
    amp: float = 1.0
    label: str = ""


def choose_tempo(move: Move, spb: float, amp: float, speed_limit: float, wanted: str = "auto") -> str:
    if wanted in RATES:
        return wanted
    if move.max_tempo == "half":
        return "half"
    return "full" if move.peak_speed(spb, 1.0, amp) <= speed_limit else "half"


def plan_test(song: Song, moves, speed_limit) -> list[Segment]:
    names, segs = list(moves), []
    for k in range(len(song.bars) // 4):
        m = moves[names[k % len(names)]]
        segs.append(Segment(song.bar_beat(4 * k), m.name, choose_tempo(m, song.spb, 1.0, speed_limit), 1.0, "test"))
    return segs


def plan_auto(song: Song, moves, speed_limit) -> list[Segment]:
    """Section-driven choreography.

    * Sections that mostly lack bass get the calm "flow" family; otherwise the label decides
      (chorus -> high, verse -> mid, intro/breakdown/outro -> low).
    * A new move every SLOT_BARS bars, never the same move twice in a row.
    * In a chorus the anchor move (the wave) opens every other slot, so every chorus is recognizable;
      the slots in between rotate through the whole family for variety (same for all other levels).
    * Section energy scales the move size; moves marked energy "off" are never used.
    """
    usable = {n: m for n, m in moves.items() if m.energy != "off"}
    by_level = {lvl: [m.name for m in usable.values() if m.energy == lvl] for lvl in FAMILY_FALLBACK}
    fam = {lvl: next((by_level[f] for f in chain if by_level[f]), list(usable)) for lvl, chain in FAMILY_FALLBACK.items()}
    rotation: dict[str, int] = {}
    segs, prev = [], None
    for s in song.sections:
        level = "flow" if s.get("bass_share", 1.0) < 0.5 else LABEL_FAMILY.get(s["label"], "mid")
        family = fam[level]
        amp = float(np.clip(0.55 + 0.45 * s.get("energy_rel", 0.5), 0.5, 1.0))
        if level == "flow":
            amp = max(amp, 0.85)  # flow moves are gentle by design; keep them visible
        amp = round(amp, 2)
        anchor = CHORUS_ANCHOR if s["label"] == "chorus" and CHORUS_ANCHOR in family else None
        others = [n for n in family if n != anchor] or family
        bar, slot = s["start_bar"], 0
        while bar < s["end_bar"]:
            remaining = s["end_bar"] - bar

            def fits(n):  # the whole move must fit into the rest of the section
                return usable[n].bars_needed(choose_tempo(usable[n], song.spb, amp, speed_limit)) <= max(remaining, SLOT_BARS)

            if anchor and slot % 2 == 0 and fits(anchor):
                name = anchor
            else:
                i = rotation.get(level, 0)
                for _ in range(len(others)):
                    name = others[i % len(others)]
                    i += 1
                    if name != prev and fits(name):
                        break
                rotation[level] = i
            if name == prev and len(family) > 1:  # e.g. at a section boundary
                name = next((n for n in others + family if n != prev and fits(n)), name)
            m = usable[name]
            tempo = choose_tempo(m, song.spb, amp, speed_limit)
            segs.append(Segment(song.bar_beat(bar), name, tempo, amp, s["label"] + (" (flow)" if level == "flow" else "")))
            prev = name
            bar += max(SLOT_BARS, m.bars_needed(tempo))
            slot += 1
    return segs


def plan_from_file(path: Path, song: Song, moves, speed_limit) -> list[Segment]:
    """[{"start": seconds, "move": name, "tempo": "full|half|auto", "amp": 0.3-1.0, "label": ""}, ...]
    Start times are snapped to the nearest bar start so moves change on the "one"."""
    segs = []
    for item in json.loads(path.read_text()):
        if item["move"] not in moves:
            raise SystemExit(f"Move '{item['move']}' gibt es nicht. Vorhanden: {', '.join(moves)}")
        m = moves[item["move"]]
        b = song.beat_pos(float(item["start"]))
        bar_b = song.phase + 4 * round((b - song.phase) / 4)
        amp = float(np.clip(item.get("amp", 1.0), 0.2, 1.0))
        tempo = choose_tempo(m, song.spb, amp, speed_limit, item.get("tempo", "auto"))
        segs.append(Segment(bar_b, m.name, tempo, amp, item.get("label", "")))
    return sorted(segs, key=lambda s: s.start_beat)


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def target_pose(bp: float, segs: list[Segment], moves: dict[str, Move]) -> np.ndarray:
    k = max(0, int(np.searchsorted([s.start_beat for s in segs], bp, side="right")) - 1)
    s = segs[k]
    pose = moves[s.move].pose(max(0.0, bp - s.start_beat) * RATES[s.tempo], s.amp)
    if k > 0 and bp - s.start_beat < FADE_BEATS:
        p = segs[k - 1]
        prev = moves[p.move].pose((bp - p.start_beat) * RATES[p.tempo], p.amp)
        w = smoothstep((bp - s.start_beat) / FADE_BEATS)
        pose = (1 - w) * prev + w * pose
    return pose


# ----------------------------------------------------------------------------- song edges
@dataclass
class Edges:
    """Rise from the rest pose with the intro, settle back into it during the fade-out."""
    in0: float
    in1: float
    out0: float
    out1: float

    @classmethod
    def for_run(cls, song: Song, t_start: float, t_end: float) -> "Edges":
        a = song.a
        in0 = t_start
        in1 = max(in0 + GLIDE_S, a.get("fade_in_end", 0.0) if t_start < a.get("fade_in_end", 0.0) else 0.0)
        out1 = min(t_end, a.get("music_end", t_end))
        out0 = min(a.get("fade_out_start", out1), out1 - GLIDE_S)
        return cls(in0, in1, max(out0, in1), out1)

    def factor(self, t: float) -> float:
        """0 = rest pose, 1 = full dance."""
        w_in = smoothstep((t - self.in0) / max(1e-6, self.in1 - self.in0))
        w_out = smoothstep((t - self.out0) / max(1e-6, self.out1 - self.out0))
        return w_in * (1.0 - w_out)


def command(t: float, song, segs, moves, edges: Edges, home: np.ndarray, offset: float) -> np.ndarray:
    dance = target_pose(song.beat_pos(t + offset), segs, moves)
    return home + edges.factor(t + offset) * (dance - home)


# ----------------------------------------------------------------------------- playback
def glide(robot, start: np.ndarray, end: np.ndarray, seconds: float):
    n = int(seconds * CONTROL_HZ)
    for i in range(1, n + 1):
        t0 = time.perf_counter()
        q = start + smoothstep(i / n) * (end - start)
        robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})
        time.sleep(max(0.0, 1 / CONTROL_HZ - (time.perf_counter() - t0)))


def simulate(song, segs, moves, t_start, t_end, offset, speed_limit, edges, home):
    ts = np.arange(t_start, t_end, 1 / CONTROL_HZ)
    max_step = speed_limit / CONTROL_HZ
    q_prev = home.copy()
    out, clipped = [], np.zeros(len(JOINTS), dtype=int)
    for t in ts:
        want = command(t, song, segs, moves, edges, home, offset)
        q = q_prev + np.clip(want - q_prev, -max_step, max_step)
        clipped += np.abs(want - q) > 1.0
        out.append(q)
        q_prev = q
    return ts, np.array(out), clipped


def _lag_reach(t, c, m):
    """Lag (ms) of measured m behind commanded c and fraction of the commanded range reached."""
    dt = float(np.median(np.diff(t)))
    err = [np.abs(c[: len(c) - k] - m[k:]).mean() for k in range(0, int(0.4 / dt) + 1)]
    k = int(np.argmin(err))
    reach = (np.percentile(m, 95) - np.percentile(m, 5)) / max(1e-6, np.percentile(c, 95) - np.percentile(c, 5))
    return 1000 * k * dt, reach


def tracking_report(log_t, cmd, meas, offset_ms: float, segs, song):
    """How far behind its commands the arm is and how much of each move it reaches - overall and per move.

    The arm always lags its commands; --offset-ms sends commands that much earlier, so its remaining
    offset to the music is (lag - offset_ms). Moves reaching < ~80 % or lagging much more than the
    rest are too fast for the servos.
    """
    t, cmd, meas = np.array(log_t), np.array(cmd), np.array(meas)
    print(f"\nWie gut folgt der Arm? (Befehle {offset_ms:.0f} ms vorgezogen; Restversatz = Verzoegerung - Vorzug)")
    lags = []
    for j, name in enumerate(JOINTS):
        if cmd[:, j].std() >= 3:
            lag, reach = _lag_reach(t, cmd[:, j], meas[:, j])
            lags.append(lag)
            print(f"  {name:14s} {lag:5.0f} ms hinter Befehl ({lag - offset_ms:+4.0f} ms zur Musik) | Hub {reach * 100:4.0f} %")
    if lags:
        print(f"  -> passender Wert: --offset-ms {np.median(lags):.0f}")
    print("\nPro Move (nur bewegte Gelenke; auffaellig = Hub < 80 % oder > 40 ms mehr Verzoegerung als ueblich):")
    ref = float(np.median(lags)) if lags else offset_ms
    for k, s in enumerate(segs):
        a = song.beat_time(s.start_beat) + 0.6  # skip the cross-fade
        b = song.beat_time(segs[k + 1].start_beat) if k + 1 < len(segs) else t[-1]
        m = (t >= a) & (t < b)
        if m.sum() < 2 * CONTROL_HZ:
            continue
        parts, bad = [], False
        for j, name in enumerate(JOINTS):
            if cmd[m, j].std() < 2:
                continue
            lag, reach = _lag_reach(t[m], cmd[m, j], meas[m, j])
            flag = reach < 0.8 or lag > ref + 40
            bad |= flag
            parts.append(f"{name.split('_')[0][:5]}{'_' + name.split('_')[1][:4] if '_' in name else ''} {lag:3.0f}ms/{reach * 100:3.0f}%{'!' if flag else ''}")
        print(f"  {'!!' if bad else '  '} {a - 0.6:6.1f}s {s.move:11s} " + " | ".join(parts))


def save_plot(path, song, segs, ts, qs, t_start, t_end):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(JOINTS) + 1, 1, figsize=(14, 12), sharex=True)
    bars = song.bars
    axes[0].step([b["start"] for b in bars], [b["energy"] for b in bars], where="post", color="0.3")
    axes[0].set_ylabel("Energie", fontsize=8)
    for s in song.sections:
        axes[0].text((s["start"] + s["end"]) / 2, -0.35, s["label"], ha="center", fontsize=8, color="0.3")
    colors = {}
    for k, s in enumerate(segs):
        a = song.beat_time(s.start_beat)
        b = song.beat_time(segs[k + 1].start_beat) if k + 1 < len(segs) else t_end
        c = colors.setdefault(s.move, f"C{len(colors) % 10}")
        for ax in axes:
            ax.axvspan(a, b, color=c, alpha=0.13)
        axes[0].text(a, 1.05, f"{s.move} {s.tempo} {s.amp:.0%}", fontsize=6.5, rotation=35)
    for j, ax in enumerate(axes[1:]):
        ax.plot(ts, qs[:, j], lw=0.8)
        ax.set_ylabel(JOINTS[j], fontsize=8)
    axes[-1].set_xlim(t_start, t_end)
    axes[-1].set_xlabel("Songzeit (s)")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.05, hspace=0.25)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def catalog(moves: dict[str, Move], spb: float) -> list[dict]:
    return [{
        "name": m.name, "energy": m.energy, "beats": m.beats, "generated": m.generated,
        "description": m.description,
        "peak_speed_full_deg_s": round(m.peak_speed(spb, 1.0)), "peak_speed_half_deg_s": round(m.peak_speed(spb, 0.5)),
    } for m in moves.values()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("song", type=Path)
    p.add_argument("--plan", default="auto", help="'auto' (Abschnitte+Energie), 'test' (alle Moves) oder JSON-Datei")
    p.add_argument("--only", default=None, help="Nur diese Moves, z.B. 'gen_*' oder 'bounce,gen_wave'")
    p.add_argument("--start", type=float, default=0.0, help="Startzeit im Song (s)")
    p.add_argument("--duration", type=float, default=None, help="Laenge des Ausschnitts (s)")
    p.add_argument("--offset-ms", type=float, default=100.0, help="Arm reagiert so viele ms frueher (gemessen: 100)")
    p.add_argument("--audio-latency-ms", type=float, default=0.0,
                   help="Zusaetzliche Lautsprecher-Verzoegerung (z.B. Bluetooth), gemessen mit tools/latency_test.py")
    p.add_argument("--speed-limit", type=float, default=200.0, help="Max. Gelenktempo in Grad/s")
    p.add_argument("--bar-offset", type=int, default=None, help="Taktanfang manuell setzen (0-3)")
    p.add_argument("--p-gain", type=int, default=16, help="Motor-Steifigkeit (lerobot-Standard 16)")
    p.add_argument("--volume", type=float, default=0.8)
    p.add_argument("--catalog", action="store_true", help="Move-Katalog als JSON ausgeben und beenden")
    p.add_argument("--dry-run", action="store_true", help="Nur planen, simulieren und plotten - Arm bleibt aus")
    args = p.parse_args()

    moves = load_moves(args.only)
    print("Analysiere Song ...", file=sys.stderr)
    song = Song(analyze_song.analyze(args.song, args.bar_offset))
    if args.catalog:
        print(json.dumps(catalog(moves, song.spb), indent=1, ensure_ascii=False))
        return
    print(f"Moves: {', '.join(f'{m.name} ({m.energy})' for m in moves.values())}")
    print(f"  {song.a['tempo_bpm']:.1f} BPM | {len(song.bars)} Takte | erster Taktanfang {song.a['downbeats'][0]:.2f}s")

    if args.plan == "auto":
        segs = plan_auto(song, moves, args.speed_limit)
    elif args.plan == "test":
        segs = plan_test(song, moves, args.speed_limit)
    else:
        segs = plan_from_file(Path(args.plan), song, moves, args.speed_limit)

    audio, sr = sf.read(args.song, dtype="float32", always_2d=True)
    song_len = len(audio) / sr
    t_start = args.start
    t_end = min(song_len, t_start + args.duration) if args.duration else song_len
    offset = args.offset_ms / 1000.0

    print("\nChoreografie:")
    plan_out = []
    for k, s in enumerate(segs):
        a = song.beat_time(s.start_beat)
        b = song.beat_time(segs[k + 1].start_beat) if k + 1 < len(segs) else song_len
        plan_out.append({"start": round(a, 2), "move": s.move, "tempo": s.tempo, "amp": s.amp, "label": s.label})
        if b < t_start or a > t_end:
            continue
        v = moves[s.move].peak_speed(song.spb, RATES[s.tempo], s.amp)
        warn = "  <-- wird gebremst" if v > args.speed_limit else ""
        print(f"  {a:6.1f}s - {b:6.1f}s  {s.label:9s} {s.move:10s} {s.tempo:4s} {s.amp:4.0%}  Spitze {v:4.0f} Grad/s{warn}")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    if args.plan in ("auto", "test"):
        plan_path = out_dir / f"plan_{args.song.stem}.json"
        plan_path.write_text(json.dumps(plan_out, indent=1))
        print(f"\nPlan gespeichert (editierbar, mit --plan wieder einlesbar): {plan_path}")

    edges = Edges.for_run(song, t_start, t_end)
    print(f"Arm erhebt sich {edges.in0:.1f}-{edges.in1:.1f}s mit der Musik, kehrt {edges.out0:.1f}-{edges.out1:.1f}s "
          f"in die Ruhepose zurueck")
    ts, qs, clipped = simulate(song, segs, moves, t_start, t_end, offset, args.speed_limit, edges, REST_POSE)
    plot = out_dir / f"dance_{args.song.stem}.png"
    save_plot(plot, song, segs, ts, qs, t_start, t_end)
    print(f"Simulation: {len(ts)} Schritte | Tempo-Limit griff (>1 Grad): {dict(zip(JOINTS, clipped.tolist()))}")
    print(f"Plot: {plot}")
    if args.dry_run:
        return

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", position_p_coefficient=args.p_gain))
    robot.connect()
    obs = robot.get_observation()
    home = np.array([obs[f"{j}.pos"] for j in JOINTS])  # the arm's actual rest pose
    q_prev = home.copy()
    log_t, log_cmd, log_meas = [], [], []
    try:
        chunk = audio[int(t_start * sr): int(t_end * sr)] * args.volume
        sd.play(chunk, sr)
        # Song time 0 = the moment the first sample is actually HEARD (macOS latency + speaker delay).
        t0 = time.perf_counter() + sd.get_stream().latency + args.audio_latency_ms / 1000.0
        max_step = args.speed_limit / CONTROL_HZ
        print("\nMusik! Der Arm erhebt sich mit dem Intro. (Ctrl+C zum Stoppen)")
        while True:
            tick = time.perf_counter()
            t_song = t_start + (tick - t0)
            if t_song >= t_end:
                break
            want = command(t_song, song, segs, moves, edges, home, offset)
            q = q_prev + np.clip(want - q_prev, -max_step, max_step)
            robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q)})
            obs = robot.get_observation()
            log_t.append(t_song)
            log_cmd.append(q)
            log_meas.append([obs[f"{j}.pos"] for j in JOINTS])
            q_prev = q
            time.sleep(max(0.0, 1 / CONTROL_HZ - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        print("\nGestoppt.")
    finally:
        sd.stop()
        if len(log_cmd) > CONTROL_HZ:
            np.savez(out_dir / "dance_log.npz", t=log_t, cmd=log_cmd, meas=log_meas)
            tracking_report(log_t, log_cmd, log_meas, args.offset_ms, segs, song)
        if np.abs(q_prev - home).max() > 2:  # after Ctrl+C: glide home before releasing the motors
            print(f"Gleite in {GLIDE_S:.0f}s zurueck in die Ruhepose ...")
            glide(robot, q_prev, home, GLIDE_S)
        robot.disconnect()


if __name__ == "__main__":
    main()
