# Dancing SO-101 🤖🎶

A LeRobot SO-101 arm that listens to a song, understands its structure and dances to it —
on the beat, with moves that match the energy of each part.
Built at the Physical AI Hackathon (DARE Campus Zurich, 29 Sep 2026).

**Perceive → Decide → Act**

| | What happens | Where |
|---|---|---|
| **Perceive** | Beat tracking, bar starts, energy per bar, song sections (intro / verse / chorus / breakdown / outro), bass, fades, accent beats | `tools/analyze_song.py` |
| **Decide** | Choreography: section → move family, energy → move size, a new move every 2 bars, the wave as chorus anchor, flowing moves where the bass drops out | `tools/dance.py` |
| **Act** | Moves are time-stretched onto the song's own beat grid, cross-faded, speed-limited and compensated for each joint's measured lag | `tools/dance.py` |

No training involved: moves are recorded by teleoperation or generated in code; the intelligence
is in the music analysis and the choreography.

## Setup

- SO-101 follower + leader, calibrated with LeRobot (`lerobot-calibrate`, ids `follower` / `leader`)
- Python **3.12** (3.14 breaks LeRobot's argument parser), LeRobot 0.6.1:

```bash
python3.12 -m venv venv_so101
./venv_so101/bin/pip install "lerobot[feetech]==0.6.1" librosa sounddevice soundfile matplotlib scikit-learn
```

- Serial ports are set at the top of the scripts (`FOLLOWER_PORT`, `LEADER_PORT`).
  Find yours with `lerobot-find-port`.

## Dance

```bash
# full song
./venv_so101/bin/python tools/dance.py inputs/pdoom.wav

# an excerpt (the arm rises at the start and settles back at the end)
./venv_so101/bin/python tools/dance.py inputs/tango_salon.wav --start 66 --duration 26

# plan only, no arm
./venv_so101/bin/python tools/dance.py inputs/pdoom.wav --dry-run
```

The terminal shows a song summary (BPM, overall energy, section map) and prints every move as it starts.

Useful options:

| Option | Meaning |
|---|---|
| `--lead-ms 40` | Arm arrives this many ms before the beat (0 = exactly on it) |
| `--punch 1.0` | Back-and-forth moves accelerate into the beat and stop hard (0 = smooth sine) |
| `--plan test` | Every move for 4 bars, one after another (`--only "gen_*"` to filter) |
| `--plan my_plan.json` | Your own choreography: `[{"start": 23.5, "move": "gen_wave", "amp": 0.9}, ...]` |
| `--catalog` | All moves as JSON (name, energy, description, speed) — input for a choreography agent |
| `--debug` | Speeds, simulation plot, per-move tracking report after the run |
| `--audio-latency-ms` | Extra speaker delay, e.g. Bluetooth (measure with `tools/latency_test.py`) |

**Safety:** the arm moves fast. Keep people out of its reach, clamp the base, keep a hand near
the power supply. Ctrl+C glides the arm back to its rest pose before releasing the motors.

## Moves

Moves live in `moves/*.json` as seamless loops (joint angles per 1/32 beat) with an energy tag.

- **Recorded** (teleoperation along a metronome): `bounce`, `sway`, `bigwave`, `snap` (disabled — too fast for the servos)
- **Generated** (`tools/gen_moves.py`): nod, sway, twist, clap, flow, float, look, pendulum, circle,
  robot, disco, hello, pump, chop, and the wave

Generated moves are defined as offsets around a neutral pose and clipped to the joint range the
arm already visited safely in the recorded moves.

```bash
# record a new move (4 beats, 4 repetitions at 90 BPM; averaged, looped and aligned to the beat)
./venv_so101/bin/python tools/record_move.py mymove --bpm 90 --beats 4 --loops 4

# regenerate the generated moves
./venv_so101/bin/python tools/gen_moves.py

# check that every move turns around on the beat
./venv_so101/bin/python tools/beat_audit.py
```

## Tools

| Script | Purpose |
|---|---|
| `analyze_song.py` | Song analysis → `outputs/analysis_<song>.json` + plot |
| `dance.py` | Choreography + playback on the arm |
| `record_move.py` | Record a move from the leader arm |
| `gen_moves.py` | Generate the code-defined moves |
| `beat_audit.py` | Where does each joint of each move turn around, relative to the beat? |
| `video_sync.py` | Measure the arm's *visible* timing from a phone video of a run (audio alignment + optical flow) |
| `latency_test.py` | Measure speaker delay with the laptop microphone |
| `servo_info.py` | Read-only servo check (IDs, voltage, temperature) |
| `camera_fps.py`, `loop_profile.py` | Diagnostics for the imitation-learning setup |

## What we learned

- **The servos lag ~100–135 ms** behind their commands (elbow the most) — compensated per joint.
- **Anything that reverses on every single beat is too fast** at ~130 BPM (the servos reach only 60–70 %);
  accents therefore repeat every second beat.
- **"Turns on the beat" is not enough to *look* on the beat.** A smooth sine drifts into its turning
  point and its fastest motion sits between the beats. Moves read as in sync when they accelerate into
  the pose and stop hard on the beat — and when the whole arm, not just each joint, arrives together.
- Measuring the arm on video (`video_sync.py`) found issues no joint log could show.

## Known limitations

- Section names (verse vs. chorus) are a heuristic (the loudest repeating part is the chorus).
- Overall energy % is our own index (loudness, tempo, rhythmic density, percussiveness), useful for comparing songs.
- Timing still isn't perfect for every move; `--lead-ms` and `--punch` are the knobs to tune by eye.
