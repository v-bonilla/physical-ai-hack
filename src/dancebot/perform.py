"""Real-time performance loop: audio clock -> beat -> pose -> servo command."""

from __future__ import annotations

import csv
import os
import select
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .choreo import JOINTS, Choreo, pose_at, smoothstep
from .timing import BeatClock, scale_beats

HZ = 50
EASE_S = 2.0
DEFAULT_MARGIN = 5.0
DEFAULT_MAX_SPEED = 180.0  # joint units per second (lerobot default units are degrees)
LATENCY_STEP = 0.010


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


@contextmanager
def _signals(handlers: dict):
    """Temporarily install signal handlers; a no-op outside the main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = {s: signal.signal(s, h) for s, h in handlers.items()}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


@contextmanager
def key_reader():
    """Yields a function returning pending keypresses. Returns no keys when stdin is not a TTY."""
    if not sys.stdin.isatty():
        yield lambda: ""
        return
    try:
        import termios
        import tty
    except ImportError:
        yield lambda: ""
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)

        def read() -> str:
            out = ""
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = os.read(fd, 1).decode(errors="ignore")
                if not ch:
                    break
                out += ch
            return out

        yield read
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class Performer:
    """Plays one choreography over the song's beats, from the first detected beat to the last."""

    def __init__(self, robot, player, choreo: Choreo, beats: list[float], latency_s: float = 0.0,
                 margin: float = DEFAULT_MARGIN, max_speed: float = DEFAULT_MAX_SPEED, hz: int = HZ,
                 simulate: bool = False, hud: bool = True, beat_mult: float = 1.0):
        self.robot = robot
        self.player = player
        self.choreo = choreo
        self.clock = BeatClock(scale_beats(beats, beat_mult))
        self.n_beats = len(self.clock.beats)
        self.tempo_bpm = 60.0 / self.clock.ibi
        self.latency_s = latency_s
        self.margin = margin
        self.max_step = max_speed / hz
        self.hz = hz
        self.dt = 1.0 / hz
        self.simulate = simulate
        self.hud = hud
        self.base = np.zeros(len(JOINTS))
        self.lo = self.hi = None
        self.log: list[tuple[float, np.ndarray]] = []
        self.last_cmd: np.ndarray | None = None
        self.song_t = 0.0

    def set_start_pose(self, start: np.ndarray) -> None:
        """Relative moves are offsets from start; clamps are the move's range plus margin."""
        start = np.asarray(start, dtype=float)
        self.base = start.copy() if self.choreo.relative else np.zeros_like(start)
        poses = self.choreo.poses + self.base
        self.lo = np.minimum(poses.min(axis=0), start) - self.margin
        self.hi = np.maximum(poses.max(axis=0), start) + self.margin

    def target_at(self, t: float) -> tuple[np.ndarray, float]:
        b = self.clock.beat_at(t)
        # hold the first pose before the first beat and the last pose after the last beat
        phase = min(max(b, 0.0), float(self.n_beats - 1))
        return self.base + pose_at(self.choreo, phase), b

    def safe(self, target: np.ndarray) -> np.ndarray:
        cmd = np.clip(target, self.lo, self.hi)
        if self.last_cmd is not None:
            cmd = self.last_cmd + np.clip(cmd - self.last_cmd, -self.max_step, self.max_step)
        return np.clip(cmd, self.lo, self.hi)

    def send(self, target: np.ndarray) -> None:
        cmd = self.safe(target)
        self.robot.send_pose(cmd)
        self.last_cmd = cmd
        self.log.append((self.song_t, cmd.copy()))

    def tick_wait(self, t_next: float) -> float:
        if self.simulate:
            return t_next
        delay = t_next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        return max(t_next, time.perf_counter() - self.dt)  # no burst of catch-up ticks after a stall

    def ease(self, a: np.ndarray, b: np.ndarray, seconds: float = EASE_S, settle_s: float = 3.0) -> None:
        """Ease commands from a to b, then keep commanding b until the step clamp has reached it."""
        n = max(1, int(seconds * self.hz))
        t_next = time.perf_counter()
        for k in range(1, n + 1):
            self.send(a + (b - a) * smoothstep(k / n))
            t_next = self.tick_wait(t_next + self.dt)
        goal = np.clip(b, self.lo, self.hi)
        for _ in range(int(settle_s * self.hz)):
            if np.allclose(self.last_cmd, goal, atol=1e-6):
                break
            self.send(b)
            t_next = self.tick_wait(t_next + self.dt)

    def hud_line(self, b: float) -> str:
        bar, beat = divmod(int(np.floor(b)), 4)
        return (f"\r t={self.song_t:6.2f}s  bar {bar + 1:3d}.{beat + 1}  {self.tempo_bpm:5.1f} BPM  "
                f"{self.choreo.name}  lat={self.latency_s * 1000:+4.0f}ms  ([ ] nudge)  ")

    def ease_back(self, start_pose: np.ndarray) -> None:
        """Return to the start pose; further Ctrl+C presses are ignored until it arrives."""
        with _signals({signal.SIGINT: signal.SIG_IGN}):
            for _ in range(10):
                try:
                    self.ease(self.last_cmd.copy(), start_pose)
                    return
                except KeyboardInterrupt:
                    continue

    def run(self, csv_path: str | Path | None = None) -> dict:
        with _signals({s: _raise_interrupt for s in (signal.SIGTERM, getattr(signal, "SIGHUP", None)) if s}):
            return self._run(csv_path)

    def _run(self, csv_path: str | Path | None) -> dict:
        self.robot.connect()
        interrupted = False
        try:
            start_pose = self.robot.read_pose()
            self.set_start_pose(start_pose)
            self.last_cmd = start_pose.copy()
            try:
                self.player.open()  # a bad audio device fails here, before any motion
                first, _ = self.target_at(self.latency_s)
                self.ease(start_pose, first)
                self.player.start()
                self._dance_loop()
            except KeyboardInterrupt:
                interrupted = True
            finally:
                # runs on any exception too; the outer finally still disconnects if this raises
                try:
                    self.player.stop()
                finally:
                    if self.hud:
                        sys.stdout.write("\n")
                    print("easing back to start pose", file=sys.stderr)
                    self.ease_back(start_pose)
        finally:
            self.robot.disconnect()
        if csv_path:
            self.dump_csv(csv_path)
        return {"interrupted": interrupted, "ticks": len(self.log), "start_pose": start_pose,
                "final_cmd": self.last_cmd, "latency_s": self.latency_s}

    def _dance_loop(self) -> None:
        with key_reader() as keys:
            t_next = time.perf_counter()
            last_hud = -1.0
            while not self.player.done:
                for ch in keys():
                    if ch == "[":
                        self.latency_s -= LATENCY_STEP
                    elif ch == "]":
                        self.latency_s += LATENCY_STEP
                self.song_t = self.player.time()
                target, b = self.target_at(self.song_t + self.latency_s)
                self.send(target)
                if self.hud and abs(self.song_t - last_hud) >= 0.1:
                    sys.stdout.write(self.hud_line(b))
                    sys.stdout.flush()
                    last_hud = self.song_t
                if self.simulate:
                    self.player.advance(self.dt)
                t_next = self.tick_wait(t_next + self.dt)

    def dump_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["song_time"] + JOINTS)
            for t, p in self.log:
                w.writerow([f"{t:.4f}"] + [f"{v:.3f}" for v in p])
