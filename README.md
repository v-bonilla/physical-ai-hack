# dancebot

An SO-101 arm dances to any song, locked to the beat. Plan: `docs/PLAN.md`. Full design: `docs/DESIGN.md`.

## Try it on the Mac

Plug in the follower arm (power and USB), stand it upright and clear of the table, then:

```sh
uv run dancebot smoke
```

On a fresh clone the first run installs everything (torch, lerobot, Beat This!) and downloads the
Beat This! checkpoint (77 MB), which takes a few minutes depending on network. Later runs take
about 40 s.

It synthesizes a 110 BPM test groove, runs both beat trackers, plays 4 clicks, finds the arm,
checks calibration, nudges each moving joint 3 degrees (after a 5 s countdown, Ctrl+C aborts),
then dances for 20 s. It prints PASS / WARN / FAIL / SKIP per stage, the exact `dance` command to
use next, and saves everything to `smoke-report-<time>.txt` to paste back. Flags: `--song PATH`,
`--port P`, `--robot-id ID` (default `dancer`), `--seconds N`, `--no-robot`, `--no-audio`.

The last line is the verdict. Exit codes: `0` = `RESULT: PASS` (WARN allowed), `1` =
`RESULT: FAIL`, `2` = `RESULT: PARTIAL (arm not tested)` or `(audio not tested)`, meaning no FAIL
but the arm (port, calibration, connect or joints skipped) or the speaker was not exercised. Fix
PARTIAL by plugging in the arm (it looks for `/dev/tty.usbmodem*`, `/dev/tty.usbserial*`,
`/dev/tty.wchusbserial*`) or dropping `--no-robot` / `--no-audio`. Ctrl+C at any point eases the
arm back, marks the current stage FAIL and still writes the report.

| FAIL | Do this |
|---|---|
| env | `uv sync`, then rerun |
| beats/beat_this | needs network once for the checkpoint; librosa still works |
| beats (both) | report it; use `--beat-mult` from the WARN line if shown |
| audio | check the default output device in macOS Sound settings; wired speaker |
| port (several) | unplug the leader arm, or pass `--port` |
| calibration | run the printed `lerobot-calibrate` command, or pass the `--robot-id` you calibrated with |
| connect | check power and USB; close other programs using the port |
| joint/... | that servo did not follow: check its cable and power; the dance is skipped |

## Moves

- `groove` (default): 2 bars. Bar 1 sways the pan (+-12) and nods the wrist on every beat. Bar 2
  twists the wrist roll (+-20) and claps the gripper: it opens on the "and" of 1 and 3 and snaps
  shut on 2 and 4, with the snare. Shoulder lift and elbow never move. Phase 0 lands on a detected
  downbeat. Full table: `docs/PLAN.md`.
- `sway`: pan +A on even beats, -A on odd beats (`--amplitude`, max 30).
- `--scale S` (0 < S <= 1.5) multiplies any move; offsets stay capped at pan 30, wrist flex 20,
  wrist roll 35, gripper 40. The gripper also stays within 0..100.

Moves are offsets from the pose read at connect, so calibration offsets do not matter. Units are
degrees (lerobot defaults to `use_degrees=True`); the gripper uses its 0..100 range.

## Setup

```sh
uv sync          # everything, including lerobot, Beat This! and pytest
uv run pytest
uv run python -c "from beat_this.inference import File2Beats; File2Beats('final0')"   # pre-download checkpoint
```

Beat This! falls back to librosa if unavailable.

## Arm

```sh
uv run lerobot-find-port                     # unplug/replug to identify the follower
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodemXXXX --robot.id=dancer
export DANCEBOT_FOLLOWER_PORT=/dev/tty.usbmodemXXXX
export DANCEBOT_FOLLOWER_ID=dancer
uv run dancebot ports                        # list serial ports
```

## Dance

```sh
uv run dancebot dance song.mp3 --dry-run            # fake arm, real audio, HUD
uv run dancebot dance song.mp3 --simulate --csv poses.csv   # no audio, faster than real time
uv run dancebot dance song.mp3 --port /dev/tty.usbmodemXXXX --robot-id dancer --seconds 30   # real arm, groove
uv run dancebot dance song.mp3 --port /dev/tty.usbmodemXXXX --robot-id dancer --move sway --amplitude 8
```

- Start: eases from the current pose to the first target over 2 s, then starts audio.
- The audio device opens before any motion, so a bad device fails with the arm still.
- End, Ctrl+C or any error: eases back to the start pose over 2 s, then disconnects with torque
  still ON, so the arm holds its start pose. `--release` cuts torque at exit instead; support the
  arm by hand first.
- Wrong tempo octave: `--beat-mult 2` inserts midpoints (song detected at half tempo),
  `--beat-mult 0.5` keeps every other beat. `--backend librosa` skips Beat This!; compare both.
- Per-tick step clamp: `--max-speed` (default 180 degrees/s). Targets clamp to the move range plus
  `--margin`.
- Analysis caches to `song.mp3.analysis.json`. `--no-cache` recomputes; `--backend librosa` skips
  Beat This!.

## Latency

Commands lead the audio by `--latency-ms` (default 80, env `DANCEBOT_LATENCY_MS`). While dancing,
press `]` for +10 ms and `[` for -10 ms; the HUD shows the current value and the final value is
printed at exit. Tune until the moves land on the kick. Use a wired speaker; Bluetooth
adds 150 to 300 ms.
