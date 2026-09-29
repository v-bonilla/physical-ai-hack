# Dancebot plan

Any song in, the SO-101 dances on its beat. One command to check the Mac, one command to dance.

Code freeze 17:00. Live demo 17:30. Full system design: [DESIGN.md](DESIGN.md).

## Deliverable

```sh
uv run dancebot smoke                     # first run on the Mac: installs, checks, dances 20 s
uv run dancebot dance song.mp3            # the demo; smoke prints the exact flags to add
```

Done when:

- [ ] `smoke` ends with no FAIL on the Mac with the follower arm plugged in.
- [ ] `dance` takes a song as its only required argument and plays it from the laptop.
- [ ] The arm dances the groove for the whole song, landing each pose on the beat.
- [ ] After latency tuning, the motion visibly lands on the kick and snare.
- [ ] At song end, Ctrl+C, any error, or a closed terminal, the arm eases back to its start pose.

## Smoke test

`uv run dancebot smoke` on a fresh clone installs everything (first run takes a few minutes for
torch), then runs these stages and prints PASS, WARN, FAIL or SKIP for each:

| Stage | Checks |
|---|---|
| env | Python, library versions, Apple GPU availability |
| song | Synthesizes the smoke song, or uses `--song PATH` |
| beats | Beat This! and librosa on the song, BPM against the known 110; downloads the checkpoint |
| audio | Output device, latency, Bluetooth warning, plays a 4-click count-in |
| port | Finds the follower's serial port; refuses to guess between two |
| calibration | Calibration file exists for the robot id |
| connect | Reads the start pose |
| joints | 5 s countdown, then each dancing joint moves 3 degrees and back; checks it really moved |
| dance | 20 s of the groove on the song with real audio |
| summary | Stage table and the exact `dance` command to use next |

The full output is saved to `smoke-report-<timestamp>.txt`. No arm plugged in: robot stages SKIP
and the dance runs on a simulated arm.

## The dance: groove

Two bars, repeating, with bar 1 always starting on a detected downbeat. Offsets are relative to the
pose the arm holds at connect, in degrees (gripper in its 0 to 100 units). Keyframes are eased
(smoothstep), so each pose lands exactly on its beat.

| Joint | Bar 1: sway and nod | Bar 2: twist and clap |
|---|---|---|
| shoulder_pan | +12 on beats 1 and 3, -12 on 2 and 4 | 0 |
| wrist_flex | -10 on every beat, 0 on every "and" | 0 |
| wrist_roll | 0 | +20 on beats 1 and 3, -20 on 2 and 4 |
| gripper | 0 | opens +25 on the "and" of 1 and 3, snaps shut on 2 and 4 |
| shoulder_lift, elbow_flex | hold | hold |

- Shoulder lift and elbow never move: they carry the arm's weight and are the joints that can hit
  the table.
- The gripper claps on the snare (beats 2 and 4). The wrist nods on every beat. The dance shows the
  bar, not only the pulse.
- `--scale` multiplies every offset (up to 1.5, with hard caps per joint). `--move sway` is the
  minimal fallback: base only, one swing per beat.

## Songs

| Use | Song | About BPM | Why |
|---|---|---|---|
| Smoke test | "Smoke Groove", original, synthesized by `smoke` | 110 | Offline, known tempo, no license |
| Demo | Queen, "Another One Bites the Dust" | 110 | Bass and kick on every beat |
| Demo | Michael Jackson, "Billie Jean" | 117 | Relentless kick and hi-hat, instantly recognized |
| Demo | Daft Punk, "Get Lucky" | 116 | Four on the floor, clean downbeats |
| Demo | Mark Ronson ft. Bruno Mars, "Uptown Funk" | 115 | Hard snare on 2 and 4 for the gripper clap |
| Demo | Bee Gees, "Stayin' Alive" | 104 | Slower, shows the dance adapts to tempo |

- Keep demo songs between 100 and 125 BPM. Check each one with `uv run dancebot smoke --song PATH`
  and use the BPM and flags it prints.
- The organizers want to feature the demo. A copyrighted track can get their clip muted; an
  original track (the smoke song, or an ElevenLabs song, U1) avoids that.

## How a beat becomes motion

1. **Find the beats.** Beat This! (neural, ISMIR 2024) finds beats and downbeats in seconds on the
   Mac CPU. librosa is the fallback. Cached next to the song.
2. **Keep one clock.** The sound card callback counts played samples, giving exact song time. That
   time becomes a fractional beat number by interpolating between detected beats, which absorbs
   tempo drift.
3. **Look up the pose.** The dance is written in beats, not seconds, so it fits any tempo.
4. **Send it.** A 50 Hz loop sends the target to the follower, leading the audio to cover servo lag.
   `[` and `]` shift the lead by 10 ms live. A speed cap and 2 s ease in and out protect the servos.

## Run of show

| Time | Step | What |
|---|---|---|
| 13:30 | S1 Checkpoint | "Any song in, the SO-101 dances on its beat. Neural beat tracking drives beat-locked motion. Next: songs generated live from a judge's prompt with ElevenLabs." |
| until 14:30 | S2 Smoke test | Power the follower, calibrate it as `dancer` if not done, run `uv run dancebot smoke`, fix every FAIL. |
| 14:30 to 15:00 | S3 First full dance | `dance` on the smoke song, then on one demo song. Tune latency until the clap lands on the snare. |
| **15:00** | **S4 Go or no-go** | **Solid dance: start one upgrade. Not solid: keep fixing, nothing else matters.** |
| 15:00 to 16:15 | S5 Upgrade, rehearse | Run `smoke --song` on each demo song. Full run-throughs, arm cools between runs. |
| 16:15 to 17:00 | S6 Demo prep | Wired speaker, songs offline, analyses cached, laptop on power, status line in a large terminal font. |
| 17:00 | Code freeze | |
| 17:30 | Live demo | 4 minutes. |

## Mac setup (S2)

```sh
git clone -b dancebot https://github.com/v-bonilla/physical-ai-hack.git && cd physical-ai-hack
uv run lerobot-find-port                      # unplug/replug the follower when asked
uv run lerobot-calibrate --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodemXXXX --robot.id=dancer   # skip if already calibrated as dancer
uv run dancebot smoke
```

Tuning knobs for `dance`:

- `--latency-ms N` sets the command lead (default 80). `[` and `]` nudge it live.
- `--beat-mult 2` doubles the detected tempo when the tracker picks half time. `0.5` halves it.
- `--backend librosa` swaps the beat tracker.
- `--scale S` makes the dance bigger or smaller. `--move sway` is the minimal fallback.
- `--seconds N` stops after N seconds.
- `--release` cuts torque at exit. By default the arm keeps holding its start pose.

## Upgrades, cheapest win first

| Code | Upgrade | Cost | Why |
|---|---|---|---|
| U1 | Song on demand: judge gives a prompt, ElevenLabs Music writes the song, the robot dances to it | ~45 min | Biggest "look up" moment. Pre-generate a backup in case the network fails |
| U2 | Switch between groove and sway by section energy | ~1 h | The dance follows the song's build and drop |
| U3 | Recorded choreographies with the leader arm, picked per song section | 2 h+ | Full design in DESIGN.md. Likely after today |

## Risks

- **R1 Speaker lag.** Bluetooth adds 150 to 300 ms and breaks the sync. Use a wired speaker or the
  laptop speakers. `smoke` warns on Bluetooth-looking devices.
- **R2 Venue network.** `smoke` downloads the beat checkpoint. Store every demo song offline before
  17:00.
- **R3 Servo heat.** STS3215 servos heat up under load. Keep runs under 3 minutes, rest between.
- **R4 Start pose and exit.** The dance is relative to the pose at connect. Set the arm upright,
  clear of the table, gripper free, before every run. The arm holds that pose with torque on after
  exit, so support it by hand before `--release` or unplugging.
- **R5 Fast songs.** Above about 140 BPM the wrist twist nears the speed cap and lags. Keep demo
  songs at or below 125 BPM.
- **R6 Wrong tempo.** The beat tracker can lock to half or double time (seen in testing). `smoke
  --song` compares both trackers; fix with `--beat-mult` or `--backend`.
- **R7 Calibration id.** `--robot-id` (default `dancer`) must match the id used in
  `lerobot-calibrate`. lerobot 0.6.1 reads
  `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`; files from older lerobot
  versions live elsewhere and are not found. dancebot refuses to start without it.
- **R8 After `--release`.** Power-cycle the arm before the next run. lerobot may drive to a stale
  goal position on connect if the arm was moved by hand.

## Roles

| Role | Owns |
|---|---|
| Arm | Power, port, calibration, start pose, cooling between runs |
| Code | Runs smoke and dance, tunes latency, builds the upgrade |
| Show | Songs, speaker, pitch, demo script, the judge's song choice |

## Demo script (4 min)

| At | Beat |
|---|---|
| 0:00 | Pitch line: any song in, the robot dances on its beat |
| 0:20 | A judge picks a song from our list, or gives a prompt if U1 shipped |
| 0:40 | Run it. Screen shows BPM, bar.beat, latency |
| 0:45 | Dance: sway and nod, then twist and clap, on every bar |
| 2:45 | Second song at a different tempo, to show it adapts |
| 3:40 | How it works in one sentence. Questions |
