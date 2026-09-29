"""dancebot command line."""

from __future__ import annotations

import argparse
import math
import os
import sys

from .choreo import MAX_SCALE, MAX_SWAY, MOVES
from .perform import DEFAULT_MARGIN, DEFAULT_MAX_SPEED
from .robot import DEFAULT_ROBOT_ID, CalibrationError


def _finite(s: str) -> float:
    v = float(s)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError("must be a finite number")
    return v


def _positive(s: str) -> float:
    v = _finite(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def _scale(s: str) -> float:
    v = _positive(s)
    if v > MAX_SCALE:
        raise argparse.ArgumentTypeError(f"must be <= {MAX_SCALE:g}")
    return v


def _non_negative(s: str) -> float:
    v = _finite(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def cmd_dance(args) -> None:
    from .analyze import analyze, summarize
    from .audio import NullPlayer, Player, load_audio
    from .choreo import make_move
    from .perform import Performer
    from .robot import FakeArm, LeRobotArm

    if args.simulate:
        args.dry_run = True
    analysis = analyze(args.song, backend=args.backend, use_cache=not args.no_cache)
    print(summarize(analysis))
    if args.amplitude > MAX_SWAY:
        print(f"amplitude clamped to {MAX_SWAY:g}", file=sys.stderr)
    choreo = make_move(args.move, args.scale, args.amplitude)

    if args.dry_run:
        robot = FakeArm()
    else:
        port = args.port or os.environ.get("DANCEBOT_FOLLOWER_PORT")
        if not port:
            sys.exit("set --port or DANCEBOT_FOLLOWER_PORT (find it with: dancebot ports)")
        robot = LeRobotArm(port=port, robot_id=args.robot_id or os.environ.get("DANCEBOT_FOLLOWER_ID", DEFAULT_ROBOT_ID),
                           release=args.release)

    if args.simulate:
        player = NullPlayer(analysis["duration"], simulated=True)
    elif args.no_audio:
        player = NullPlayer(analysis["duration"])
    else:
        data, sr = load_audio(args.song)
        player = Player(data, sr, gain=args.gain)

    perf = Performer(robot, player, choreo, analysis["beats"], latency_s=args.latency_ms / 1000.0,
                     margin=args.margin, max_speed=args.max_speed, simulate=args.simulate, hud=not args.quiet,
                     beat_mult=args.beat_mult, downbeats=analysis["downbeats"], stop_after_s=args.seconds)
    try:
        res = perf.run(csv_path=args.csv)
        print(f"done: {res['ticks']} commands, interrupted={res['interrupted']}, "
              f"final latency {res['latency_s'] * 1000:.0f} ms")
    except CalibrationError as e:
        sys.exit(str(e))
    finally:
        if not args.dry_run and perf.last_cmd is not None:  # only once the arm was connected and read
            print("torque released, arm is limp" if args.release else
                  "torque is still ON: the arm holds its start pose until power is cut")


def cmd_smoke(args) -> None:
    from .smoke import run_smoke

    sys.exit(run_smoke(args))


def cmd_ports(args) -> None:
    from .robot import list_ports

    ports = list_ports()
    print("\n".join(ports) if ports else "no serial ports found")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dancebot", description="Make an SO-101 dance to any song.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dance", help="analyze beats and dance the arm to the song")
    p.add_argument("song")
    p.add_argument("--move", choices=MOVES, default="groove")
    p.add_argument("--scale", type=_scale, default=1.0,
                   help=f"multiply the move's offsets (0 < S <= {MAX_SCALE:g}); per-joint caps still apply")
    p.add_argument("--seconds", type=_positive, help="stop dancing after this many seconds")
    p.add_argument("--port", help="follower serial port (env DANCEBOT_FOLLOWER_PORT)")
    p.add_argument("--robot-id", help="calibration id (env DANCEBOT_FOLLOWER_ID, default dancer)")
    p.add_argument("--amplitude", type=_finite, default=15.0, help=f"shoulder_pan sway, max {MAX_SWAY:g}")
    p.add_argument("--latency-ms", type=_finite, default=float(os.environ.get("DANCEBOT_LATENCY_MS", 80)),
                   help="commands lead audio by this much (default 80, env DANCEBOT_LATENCY_MS)")
    p.add_argument("--max-speed", type=_positive, default=DEFAULT_MAX_SPEED, help="joint units per second")
    p.add_argument("--margin", type=_non_negative, default=DEFAULT_MARGIN, help="clamp margin beyond the move's range")
    p.add_argument("--dry-run", action="store_true", help="fake arm, no hardware")
    p.add_argument("--simulate", action="store_true", help="fake arm and simulated clock, faster than real time")
    p.add_argument("--beat-mult", type=float, choices=[0.5, 1.0, 2.0], default=1.0,
                   help="2 inserts midpoints (fixes half-tempo detection), 0.5 keeps every other beat")
    p.add_argument("--release", action="store_true", help="cut torque at exit; support the arm by hand first")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--backend", choices=["auto", "beat_this", "librosa"], default="auto")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--csv", help="dump commanded poses to CSV")
    p.add_argument("--quiet", action="store_true", help="no HUD")
    p.set_defaults(fn=cmd_dance)

    sub.add_parser("ports", help="list serial ports").set_defaults(fn=cmd_ports)

    p = sub.add_parser("smoke", help="check everything on this machine, ending with a short real dance")
    p.add_argument("--song", help="use this song instead of the synthesized smoke groove")
    p.add_argument("--port", help="follower serial port (env DANCEBOT_FOLLOWER_PORT, else auto-detect)")
    p.add_argument("--robot-id", help="calibration id (env DANCEBOT_FOLLOWER_ID, default dancer)")
    p.add_argument("--seconds", type=_positive, default=20.0, help="length of the final dance (default 20)")
    p.add_argument("--no-robot", action="store_true", help="skip the arm; the dance runs on a fake arm")
    p.add_argument("--no-audio", action="store_true", help="skip audio checks and play silently")
    p.add_argument("--latency-ms", type=_finite, default=float(os.environ.get("DANCEBOT_LATENCY_MS", 80)))
    p.set_defaults(fn=cmd_smoke)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
