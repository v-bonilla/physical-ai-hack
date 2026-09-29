# Dancebot first dance

Any song in, the SO-101 dances on its beat. One command, one simple move, locked to the music.
Everything else waits until this works on the real arm.

Code freeze 17:00. Live demo 17:30. Full system design: [DESIGN.md](DESIGN.md).

## Deliverable

```sh
uv run dancebot dance song.mp3 --port /dev/tty.usbmodemXXXX --robot-id dancer
```

Done when:

- [ ] The song is the only required argument. It plays from the laptop while the arm moves.
- [ ] The arm sways to one side on one beat and back on the next, for the whole song.
- [ ] After latency tuning, the motion visibly lands on the beat.
- [ ] At song end or Ctrl+C, the arm eases back to its start pose before the motors let go.
- [ ] `--dry-run` runs the same path with no robot attached.

## The first move: base sway

| bar.beat | 1.1 | 1.2 | 1.3 | 1.4 | 2.1 | 2.2 | ... |
|---|---|---|---|---|---|---|---|
| shoulder pan offset (deg) | +15 | -15 | +15 | -15 | +15 | -15 | ... |

- Eased between beats (smoothstep), so the arm arrives exactly on each beat.
- Offsets are relative to the pose the arm holds at connect, so the move is safe with any
  calibration. Every other joint holds still.
- Units are degrees (lerobot `use_degrees=True` default). `--amplitude` sets the swing, capped at 30.

## How a beat becomes motion

1. **Find the beats.** Beat This! (neural, ISMIR 2024) finds beats and downbeats in seconds on the
   Mac CPU. librosa is the fallback. Cached next to the song.
2. **Keep one clock.** The sound card callback counts played samples, giving exact song time. That
   time becomes a fractional beat number by interpolating between detected beats, which absorbs
   tempo drift.
3. **Look up the pose.** The move is written in beats, not seconds, so it fits any tempo.
4. **Send it.** A 50 Hz loop sends the target to the follower, leading the audio to cover servo lag.
   `[` and `]` shift the lead by 10 ms live. A speed cap and 2 s ease in and out protect the servos.

## Run of show

| Time | Step | What |
|---|---|---|
| 13:30 | S1 Checkpoint | "Any song in, the SO-101 dances on its beat. Neural beat tracking drives beat-locked motion. Next: songs generated live from a judge's prompt with ElevenLabs." |
| 13:30 to 14:15 | S2 Mac setup | Power the follower, find its port, calibrate if needed, `uv sync`, download the beat checkpoint once on venue wifi. In parallel: code lands on branch `dancebot`, reviewed and tested without hardware. |
| 14:15 to 15:00 | S3 First dance | Dry run. Then live at amplitude 8, then 15. Tune latency on a song with a hard kick drum until the swing lands on the kick. |
| **15:00** | **S4 Go or no-go** | **Solid first dance: start one upgrade. Not solid: keep fixing, nothing else matters.** |
| 15:00 to 16:15 | S5 Upgrade, rehearse | Pick three demo songs at about 90, 120 and 140 BPM. Full run-throughs, arm cools between runs. |
| 16:15 to 17:00 | S6 Demo prep | Wired speaker, songs offline, analyses cached, laptop on power, status line in a large terminal font. |
| 17:00 | Code freeze | |
| 17:30 | Live demo | 4 minutes. |

## Mac setup (S2)

Exact flags are in the README once the code lands.

```sh
git fetch origin && git switch dancebot
uv sync --extra robot --extra beats
uv run lerobot-find-port                      # unplug/replug the follower when asked
uv run lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX --robot.id=dancer   # skip if already calibrated
uv run dancebot dance songs/test.mp3 --dry-run
uv run dancebot dance songs/test.mp3 --port /dev/tty.usbmodemXXXX --robot-id dancer --amplitude 8
```

Tuning knobs:

- `--latency-ms N` sets the command lead (default 80). `[` and `]` nudge it live.
- `--beat-mult 2` doubles the detected tempo when the tracker picks half time. `0.5` halves it.
- `--backend librosa` swaps the beat tracker. Try both on each demo song.
- `--release` cuts torque at exit. By default the arm keeps holding its start pose.

## Upgrades after the first dance, cheapest win first

| Code | Upgrade | Cost | Why |
|---|---|---|---|
| U1 | Accent on the downbeat: wrist nod or gripper clap on beat 1 of every bar | ~30 min | The dance shows the bar, not only the pulse |
| U2 | Song on demand: judge gives a prompt, ElevenLabs Music writes the song, the robot dances to it | ~45 min | Biggest "look up" moment. Pre-generate a backup in case the network fails |
| U3 | Recorded choreographies with the leader arm, picked per song section | 2 h+ | Full design in DESIGN.md. Likely after today |

## Risks

- **R1 Speaker lag.** Bluetooth adds 150 to 300 ms and breaks the sync. Use a wired speaker or the
  laptop speakers.
- **R2 Venue network.** Download the beat checkpoint and every demo song before 17:00.
- **R3 Servo heat.** STS3215 servos heat up under load. Keep runs under 3 minutes, rest between.
- **R4 Start pose and exit.** The sway is relative to the pose at connect. Set the arm upright and
  clear of the table before every run. The arm holds that pose with torque on after exit, so support
  it by hand before `--release` or unplugging.
- **R5 Fast songs.** Above about 140 BPM the swing may lag. Keep demo songs at or below that.
- **R6 Wrong tempo.** The beat tracker can lock to half or double time (seen in testing). Check the BPM on the
  status line for each demo song and fix it with `--beat-mult` or `--backend`.
- **R7 Calibration id.** `--robot-id` must match the id used in `lerobot-calibrate`. lerobot 0.6.1
  reads `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`; files from older
  lerobot versions live elsewhere and are not found. dancebot refuses to start without it.
- **R8 After `--release`.** Power-cycle the arm before the next run. lerobot may drive to a stale
  goal position on connect if the arm was moved by hand.

## Roles

| Role | Owns |
|---|---|
| Arm | Power, port, calibration, start pose, cooling between runs |
| Code | Runs the command, tunes latency, builds the upgrade |
| Show | Songs, speaker, pitch, demo script, the judge's song choice |

## Demo script (4 min)

| At | Beat |
|---|---|
| 0:00 | Pitch line: any song in, the robot dances on its beat |
| 0:20 | A judge picks a song from our list, or gives a prompt if U2 shipped |
| 0:40 | Run it. Screen shows BPM, bar.beat, latency |
| 0:45 | Dance |
| 2:45 | Second song at a different tempo, to show it adapts |
| 3:40 | How it works in one sentence. Questions |
