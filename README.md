# dancebot

An SO-101 arm sways to any song, locked to the beat. Plan: `docs/PLAN.md`. Full design: `docs/DESIGN.md`.

The current build ships one built-in move: `shoulder_pan` swings to `start + A` on even beats and
`start - A` on odd beats, eased so it lands on each beat. It is relative to the pose read at
connect, so calibration offsets do not matter. All other joints hold.

## Setup (Apple Silicon Mac)

```sh
uv sync --extra robot --extra beats --extra dev   # lerobot, Beat This!, pytest
uv run pytest
```

Core (librosa beats, dry run) needs no extras. Beat This! downloads its checkpoint (77 MB) on
first use and falls back to librosa if unavailable. Download it before relying on venue network:

```sh
uv run python -c "from beat_this.inference import File2Beats; File2Beats('final0')"
```

Units are degrees: lerobot defaults to `use_degrees=True`, so `--amplitude 15` means 15 degrees
of shoulder pan each side.

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
uv run dancebot dance song.mp3 --port /dev/tty.usbmodemXXXX --robot-id dancer --amplitude 8   # real arm (A max 30)
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
printed at exit. Tune until the swing extremes land on the kick. Use a wired speaker; Bluetooth
adds 150 to 300 ms.
