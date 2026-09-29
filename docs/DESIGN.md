# Dancebot design

An SO-101 arm dances to any song. We record a library of choreographies, the software finds the
song's beats and sections, picks a choreography per section, and plays it back locked to the beat.

## Pipeline

```
song.mp3 ──► analyze ──► plan ──► perform ──► SO-101 follower
               │           ▲         │
   beats, downbeats,   choreo library (recorded with leader arm)
   sections, energy                  └──► speaker (same clock)
```

| Stage | Input | Output | How |
|---|---|---|---|
| record | leader arm or hand-guided follower, metronome | `choreos/<name>.json` | Poses stored against beat phase, not seconds |
| analyze | audio file | `<song>.analysis.json` | Beat This! (neural beats and downbeats), librosa fallback; bar features; novelty sections; clustering labels repeats |
| plan | analysis + library | `<song>.plan.json` | Rule-based picker (below) |
| perform | audio + plan | servo commands at 50 Hz | Audio callback clock, beat interpolation, blends, safety clamps |
| generate (optional) | text prompt | song + section plan | ElevenLabs Music `compose_detailed` |

## Beat lock

This is what makes it read as dancing.

- A choreography is a function of beat phase, `pose(b)` with `b` in `[0, length_beats)`.
- Recording at a known BPM (metronome click with count-in) converts seconds to beats.
- At playback, song time maps to a fractional beat index by interpolating between detected beat
  times. Tempo drift and rubato are absorbed automatically.
- `beat_scale` in {0.5, 1, 2} song beats per choreo beat keeps moves near their recorded speed
  (half time for fast songs, double time for slow ones).
- Keyframe mode stores one pose per beat (or half beat) and eases between them, so the arm lands
  each pose exactly on the beat. Continuous mode stores a fluid 50 Hz take.
- Commands lead the audio by a tunable latency offset (servo lag plus speaker latency). Use a
  wired speaker; Bluetooth adds 150 to 300 ms.

## Choreography picker

Deterministic rules, no trained model:

1. Each choreography is tagged `low`, `mid`, or `high` energy. Section energy is ranked within the
   song (tertiles), so a quiet song still uses the whole library.
2. Sections with the same label get the same choreography. The chorus coming back with the same
   move is what an audience reads as choreography.
3. Adjacent sections differ when the library allows.
4. Tags `intro` and `finale` claim the first and last sections when present.

## Safety

- Joint targets clamp to the range seen across the recorded library, plus a small margin.
- Per-tick step clamp caps joint speed.
- Transitions blend over half a beat.
- Start: ease from the current pose to the first pose over 2 s. End or Ctrl+C: ease back to the
  starting pose before disconnecting, because torque off drops the arm.
- STS3215 servos heat up under sustained load. Rest the arm between rehearsals.

## Demo arc

1. Judge hands us a song (file or phone). Analysis takes seconds.
2. Laptop HUD shows beat, bar, section label, energy, current move.
3. Stretch: judge speaks a prompt, ElevenLabs Music generates a song with known sections, the
   robot dances to a track that did not exist a minute ago.
