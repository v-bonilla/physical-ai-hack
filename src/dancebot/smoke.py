"""`dancebot smoke`: check every stage on this machine, ending with a short real dance."""

from __future__ import annotations

import glob
import importlib.metadata as md
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .choreo import JOINTS, groove
from .robot import DEFAULT_ROBOT_ID

SMOKE_BPM = 110.0
BPM_TOL = 0.04
PACKAGES = ["numpy", "librosa", "sounddevice", "soundfile", "torch", "lerobot", "beat-this"]
BLUETOOTH_HINTS = ("airpods", "bluetooth", "beats", "jbl", "bose", "wh-", "bt")
TEST_JOINTS = ["shoulder_pan", "wrist_flex", "wrist_roll", "gripper"]
JOINT_STEP = 3.0
JOINT_MIN_MOVE = 1.5
COUNTDOWN_S = 5


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.stages: list[tuple[str, str, str]] = []

    def say(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def stage(self, name: str, status: str, detail: str) -> str:
        first, *rest = detail.splitlines() or [""]
        self.say(f"[{status:<4}] {name:<18} {first}")
        for line in rest:
            self.say(f"       {'':<18} {line}")
        self.stages.append((name, status, first))
        return status

    @property
    def failed(self) -> bool:
        return any(s == "FAIL" for _, s, _ in self.stages)


def classify_bpm(bpm: float, expected: float = SMOKE_BPM) -> tuple[str, float | None]:
    """PASS with mult 1, WARN with the --beat-mult that fixes an octave error, or FAIL."""
    for mult, status in ((1.0, "PASS"), (2.0, "WARN"), (0.5, "WARN")):
        if abs(bpm * mult - expected) <= BPM_TOL * expected:
            return status, mult
    return "FAIL", None


def detect_ports(explicit: str | None = None) -> list[str]:
    if explicit:
        return [explicit]
    env = os.environ.get("DANCEBOT_FOLLOWER_PORT")
    if env:
        return [env]
    pattern = "/dev/tty.usbmodem*" if sys.platform == "darwin" else "/dev/ttyACM*"
    return sorted(glob.glob(pattern))


def _click_track(bpm: float = SMOKE_BPM, n: int = 4, sr: int = 44100) -> np.ndarray:
    spb = 60.0 / bpm
    out = np.zeros(int((n + 0.5) * spb * sr), dtype=np.float32)
    m = int(0.04 * sr)
    t = np.arange(m) / sr
    for k in range(n):
        f = 1600.0 if k == 0 else 1000.0
        s = int(k * spb * sr)
        out[s : s + m] += (0.6 * np.sin(2 * np.pi * f * t) * np.exp(-t * 90)).astype(np.float32)
    return out[:, None]


def stage_env(r: Report) -> None:
    versions, missing = [], []
    for pkg in PACKAGES:
        try:
            versions.append(f"{pkg} {md.version(pkg)}")
        except md.PackageNotFoundError:
            missing.append(pkg)
    mps = "unknown"
    try:
        import torch

        mps = "yes" if torch.backends.mps.is_available() else "no"
    except Exception:
        pass
    detail = (f"python {platform.python_version()} on {platform.system()} {platform.machine()}, "
              f"torch MPS: {mps}\n" + ", ".join(versions))
    if missing:
        r.stage("env", "WARN", detail + f"\nmissing: {', '.join(missing)} (run: uv sync)")
    else:
        r.stage("env", "PASS", detail)


def stage_song(r: Report, song: str | None) -> Path | None:
    if song:
        p = Path(song)
        if not p.is_file():
            r.stage("song", "FAIL", f"{p} not found")
            return None
        r.stage("song", "PASS", f"using {p}")
        return p
    from .smoke_song import DEFAULT_PATH, make

    t = time.perf_counter()
    p = make(DEFAULT_PATH, bpm=SMOKE_BPM)
    r.stage("song", "PASS", f"synthesized {p} ({SMOKE_BPM:g} BPM, 12 bars) in {time.perf_counter() - t:.1f}s")
    return p


def stage_beats(r: Report, song: Path, synthetic: bool) -> tuple[dict | None, str, float]:
    """Runs both trackers without cache. Returns (analysis, backend, beat_mult) for the dance."""
    from .analyze import analyze

    results: dict[str, tuple[dict, str, float | None]] = {}
    for backend in ("beat_this", "librosa"):
        name = f"beats/{backend}"
        t = time.perf_counter()
        try:
            a = analyze(song, backend=backend, use_cache=False)
        except Exception as e:
            hint = " (first run downloads a 77 MB checkpoint; needs network)" if backend == "beat_this" else ""
            r.stage(name, "FAIL" if backend == "librosa" or synthetic else "WARN", f"{type(e).__name__}: {e}{hint}")
            continue
        dt = time.perf_counter() - t
        bpm = a["tempo_bpm"]
        detail = f"{bpm:.1f} BPM, {len(a['beats'])} beats, {len(a['downbeats'])} downbeats, {dt:.1f}s"
        if synthetic:
            status, mult = classify_bpm(bpm)
            if status == "WARN":
                detail += f", octave error: use --beat-mult {mult:g}"
            elif status == "FAIL":
                detail += f", expected {SMOKE_BPM:g}"
        else:
            status, mult = "PASS", 1.0
        results[backend] = (a, status, mult)
        r.stage(name, status, detail)

    if not synthetic and len(results) == 2:
        b1, b2 = results["beat_this"][0]["tempo_bpm"], results["librosa"][0]["tempo_bpm"]
        ratio = max(b1, b2) / min(b1, b2)
        if 1.9 <= ratio <= 2.1:
            r.stage("beats/agree", "WARN", f"trackers disagree by 2x ({b1:.1f} vs {b2:.1f} BPM); "
                    "listen and pick --backend / --beat-mult")
        else:
            r.stage("beats/agree", "PASS", f"{b1:.1f} vs {b2:.1f} BPM")

    for want in ("PASS", "WARN"):
        for backend in ("beat_this", "librosa"):
            if backend in results and results[backend][1] == want:
                a, _, mult = results[backend]
                return a, backend, mult
    if "librosa" in results:
        return results["librosa"][0], "librosa", 1.0
    return None, "librosa", 1.0


def stage_audio(r: Report, no_audio: bool) -> bool:
    if no_audio:
        r.stage("audio", "SKIP", "--no-audio")
        return False
    try:
        import sounddevice as sd

        from .audio import Player

        dev = sd.query_devices(kind="output")
        name = str(dev["name"])
        player = Player(_click_track(), 44100)
        player.start()
        deadline = time.perf_counter() + 5.0
        while not player.done and time.perf_counter() < deadline:
            time.sleep(0.05)
        player.stop()
    except Exception as e:
        r.stage("audio", "FAIL", f"{type(e).__name__}: {e}; the dance will run silently")
        return False
    detail = (f"'{name}', {dev['default_samplerate']:.0f} Hz, output latency {player.latency * 1000:.0f} ms; "
              "you should have heard 4 clicks")
    if any(h in name.lower() for h in BLUETOOTH_HINTS):
        r.stage("audio", "WARN", detail + "\nlooks like Bluetooth (adds 150 to 300 ms); use a wired speaker")
    else:
        r.stage("audio", "PASS", detail)
    return True


def stage_joints(r: Report, perf, robot, start: np.ndarray, fast: bool) -> bool:
    """Nudge each moving joint +3 and back. Runs inside perf.session(), so every exit eases back."""
    r.say("")
    r.say("!! The arm is about to move. Arm upright, clear of the table, gripper free.")
    r.say("!! Ctrl+C to abort.")
    if not fast:
        for k in range(COUNTDOWN_S, 0, -1):
            print(f"   moving in {k}...", flush=True)
            time.sleep(1.0)
    all_ok = True
    for name in TEST_JOINTS:
        j = JOINTS.index(name)
        target = start.copy()
        target[j] += JOINT_STEP
        perf.ease(perf.last_cmd.copy(), target, seconds=0.5, settle_s=0.5)
        t_next = time.perf_counter()
        for _ in range(int(0.3 * perf.hz)):
            perf.send(target)
            t_next = perf.tick_wait(t_next + perf.dt)
        delta = float(robot.read_pose()[j] - start[j])
        perf.ease(perf.last_cmd.copy(), start, seconds=0.5, settle_s=0.5)
        ok = delta >= JOINT_MIN_MOVE
        all_ok &= ok
        r.stage(f"joint/{name}", "PASS" if ok else "FAIL",
                f"commanded +{JOINT_STEP:g}, measured {delta:+.2f}" + ("" if ok else " (servo not following)"))
    return all_ok


def run_smoke(args, arm=None, fast: bool = False, report_dir: str | Path | None = None) -> int:
    """`arm` injects a RobotIO for tests; `fast` uses a simulated clock and skips the countdown."""
    from .audio import NullPlayer, Player, load_audio
    from .perform import Performer
    from .robot import CalibrationError, FakeArm, LeRobotArm

    r = Report()
    r.say(f"dancebot smoke  {datetime.now():%Y-%m-%d %H:%M:%S}")
    stage_env(r)
    song = stage_song(r, args.song)
    analysis, backend, mult = (None, "librosa", 1.0)
    if song is not None:
        analysis, backend, mult = stage_beats(r, song, synthetic=args.song is None)
    audio_ok = stage_audio(r, args.no_audio)

    robot_id = args.robot_id or os.environ.get("DANCEBOT_FOLLOWER_ID", DEFAULT_ROBOT_ID)
    port = None
    robot = None
    if arm is not None:
        r.stage("port", "PASS", "injected test arm")
        robot = arm
    elif args.no_robot:
        r.stage("port", "SKIP", "--no-robot; the dance runs on a fake arm")
    else:
        ports = detect_ports(args.port)
        if not ports:
            r.stage("port", "SKIP", "no follower found; the dance runs on a fake arm")
        elif len(ports) > 1:
            r.stage("port", "FAIL", f"several candidates: {', '.join(ports)}\n"
                    "unplug the leader arm or pass --port")
        else:
            port = ports[0]
            r.stage("port", "PASS", port)

    if port is not None:
        try:
            robot = LeRobotArm(port=port, robot_id=robot_id)
        except Exception as e:
            r.stage("calibration", "FAIL", f"{type(e).__name__}: {e}")
    robot_ok = False
    real_arm = robot is not None and arm is None
    if robot is not None:
        try:
            r.stage("calibration", "PASS", robot.check_calibration_file())
        except CalibrationError as e:
            r.stage("calibration", "FAIL", str(e))
            robot = None
            r.stage("connect", "SKIP", "not calibrated")
            r.stage("joints", "SKIP", "not calibrated")
    else:
        for name in ("calibration", "connect", "joints"):
            if not any(s[0] == name for s in r.stages):
                r.stage(name, "SKIP", "no arm")

    beats = analysis["beats"] if analysis else []
    downbeats = analysis["downbeats"] if analysis else []
    if robot is not None and beats:
        perf = Performer(robot, NullPlayer(1.0, simulated=fast), groove(), beats, simulate=fast, hud=False,
                         beat_mult=mult, downbeats=downbeats)
        connected = False
        try:
            with perf.session() as start:
                connected = True
                r.stage("connect", "PASS", "start pose " + ", ".join(f"{n}={v:.1f}" for n, v in zip(JOINTS, start)))
                robot_ok = stage_joints(r, perf, robot, start, fast)
            if perf.interrupted:
                robot_ok = False
                r.stage("joints", "FAIL", "aborted with Ctrl+C")
        except Exception as e:
            r.stage("joints" if connected else "connect", "FAIL", f"{type(e).__name__}: {e}")
            robot_ok = False
        if not robot_ok and not any(s[0].startswith("joint") for s in r.stages):
            r.stage("joints", "SKIP", "not connected")

    # dance
    final_latency = args.latency_ms
    if analysis is None:
        r.stage("dance", "SKIP", "no beats")
    elif robot is not None and not robot_ok:
        r.stage("dance", "SKIP", "arm checks failed")
    else:
        dancer = robot if robot is not None else FakeArm()
        seconds = min(args.seconds, analysis["duration"])
        if audio_ok and not fast:
            data, sr = load_audio(song)
            player = Player(data, sr)
        else:
            player = NullPlayer(analysis["duration"], simulated=fast)
        perf = Performer(dancer, player, groove(), beats, latency_s=args.latency_ms / 1000.0, simulate=fast,
                         hud=not fast, beat_mult=mult, downbeats=downbeats, stop_after_s=seconds)
        where = "real arm" if robot is not None and real_arm else "fake arm"
        try:
            res = perf.run()
            final_latency = res["latency_s"] * 1000
            status = "WARN" if res["interrupted"] else "PASS"
            r.stage("dance", status, f"groove for {seconds:.0f}s on the {where} ({backend}, beat-mult {mult:g}), "
                    f"{res['ticks']} commands, final latency {final_latency:.0f} ms"
                    + (", stopped with Ctrl+C" if res["interrupted"] else ""))
        except Exception as e:
            r.stage("dance", "FAIL", f"{type(e).__name__}: {e}")
        if real_arm and robot is not None:
            r.say("torque is still ON: the arm holds its start pose until power is cut")

    # summary
    r.say("")
    r.say("stage              status  detail")
    for name, status, detail in r.stages:
        r.say(f"{name:<18} {status:<6}  {detail[:110]}")
    cmd = f"uv run dancebot dance {song or 'SONG'} --port {port or '/dev/tty.usbmodemXXXX'} --robot-id {robot_id}"
    if backend == "librosa":
        cmd += " --backend librosa"
    if mult != 1.0:
        cmd += f" --beat-mult {mult:g}"
    if abs(final_latency - args.latency_ms) > 1e-6:
        cmd += f" --latency-ms {final_latency:.0f}"
    r.say("")
    r.say("next: " + cmd)
    r.say("RESULT: " + ("FAIL" if r.failed else "PASS"))

    path = Path(report_dir or Path.cwd()) / f"smoke-report-{datetime.now():%Y%m%d-%H%M%S}.txt"
    try:
        path.write_text("\n".join(r.lines) + "\n")
        print(f"report written to {path}")
    except OSError as e:
        print(f"could not write report: {e}")
    return 1 if r.failed else 0
