"""Song analysis for the dancing arm: beats, bars, energy per bar and song sections.

  * Beats: librosa beat tracker. Bar starts ("downbeats"): the beat phase with the strongest onsets.
  * Energy per bar (0..1): loudness (60 %), rhythmic busyness (25 %) and brightness (15 %).
  * Sections: the song is cut into phrases of 4 bars; phrases that sound alike (harmony + timbre)
    get the same type letter (A, B, C ...). Runs of equal type form a section. The loudest type that
    repeats is the chorus; quiet sections become intro / breakdown / outro.

Writes outputs/analysis_<song>.json - the contract for the choreography (agents, dance.py, demo
screen) - and outputs/analysis_<song>.png.

Usage:
    python tools/analyze_song.py inputs/pdoom.wav
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
SR, HOP = 22050, 512
PHRASE_BARS = 4


def _norm(x):
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)


def analyze(path: Path, bar_offset: int | None = None, n_types: int | None = None) -> dict:
    y, sr = librosa.load(path, sr=SR, mono=True)
    duration = len(y) / sr
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    strength = onset[np.minimum(beat_frames, len(onset) - 1)]
    phase = bar_offset if bar_offset is not None else int(np.argmax([strength[p::4].mean() for p in range(4)]))
    downbeats = beats[phase::4]
    spb = float(np.median(np.diff(beats)))
    bar_edges = np.append(downbeats, min(duration, downbeats[-1] + 4 * spb))

    # Frame features
    stft = np.abs(librosa.stft(y, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    bass_power = stft[(freqs >= 30) & (freqs <= 150)].sum(0)  # kick + bass line
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
    t = librosa.times_like(rms, sr=sr, hop_length=HOP)

    bars = []
    feats = []
    for i, (a, b) in enumerate(zip(bar_edges[:-1], bar_edges[1:])):
        m = (t >= a) & (t < b)
        if not m.any():
            continue
        bars.append({"index": i, "start": round(float(a), 3), "end": round(float(b), 3),
                     "_loud": float(librosa.amplitude_to_db(rms[m]).mean()),
                     "_bass": float(10 * np.log10(bass_power[: len(t)][m].mean() + 1e-12)),
                     "_busy": float(onset[m].mean()), "_bright": float(centroid[m].mean())})
        feats.append(np.concatenate([chroma[:, m].mean(1), mfcc[1:, m].mean(1)]))
    loud, busy, bright = (_norm(np.array([b[k] for b in bars])) for k in ("_loud", "_busy", "_bright"))
    energy = 0.6 * loud + 0.25 * busy + 0.15 * bright
    loud_db = np.array([b["_loud"] for b in bars])
    bass_db = np.array([b["_bass"] for b in bars])
    # Bass relative to the song's typical bass level: a bar "has no bass" if it is >12 dB below the median.
    bass_rel = bass_db - np.median(bass_db)
    for b, e, br in zip(bars, energy, bass_rel):
        for k in ("_loud", "_bass", "_busy", "_bright"):
            b.pop(k)
        b["energy"] = round(float(e), 3)
        b["bass_db_rel"] = round(float(br), 1)
        b["bass"] = bool(br > -6)

    # Fade-in / fade-out from a 0.5 s smoothed loudness curve (fades can be shorter than a bar).
    env = uniform_filter1d(librosa.amplitude_to_db(rms, ref=np.max), size=max(1, int(0.5 * sr / HOP)))
    med_db = float(np.median(env))
    audible = np.where(env > med_db - 20)[0]
    music_start = float(t[audible[0]]) if len(audible) else 0.0
    full = np.where(env >= med_db - 3)[0]
    fade_in_end = float(t[full[0]]) if len(full) else music_start
    last_loud = np.where(env >= med_db)[0]
    fade_out_start = float(t[last_loud[-1]]) if len(last_loud) else duration
    tail = np.where((t > fade_out_start) & (env < med_db - 25))[0]
    music_end = float(t[tail[0]]) if len(tail) else duration

    # Phrases of 4 bars -> types by similarity
    feats = np.array(feats)
    n_ph = int(np.ceil(len(bars) / PHRASE_BARS))
    ph_feat = np.array([feats[p * PHRASE_BARS:(p + 1) * PHRASE_BARS].mean(0) for p in range(n_ph)])
    ph_energy = np.array([energy[p * PHRASE_BARS:(p + 1) * PHRASE_BARS].mean() for p in range(n_ph)])
    X = (ph_feat - ph_feat.mean(0)) / (ph_feat.std(0) + 1e-9)
    X = np.hstack([X, 3 * ph_energy[:, None]])  # energy counts as much as a few features
    if n_types is None:
        # Prefer 4-6 types: fewer merge verse and chorus into one long block.
        ks = range(min(4, n_ph - 1), min(6, n_ph - 1) + 1)
        scores = {k: silhouette_score(X, AgglomerativeClustering(n_clusters=k).fit_predict(X)) for k in ks}
        n_types = max(scores, key=scores.get)
    labels = AgglomerativeClustering(n_clusters=n_types).fit_predict(X)
    # Letters in order of first appearance
    order = {}
    for l in labels:
        order.setdefault(l, chr(ord("A") + len(order)))
    ph_type = [order[l] for l in labels]

    # Merge equal consecutive phrases into sections
    sections = []
    for p, typ in enumerate(ph_type):
        b0, b1 = p * PHRASE_BARS, min((p + 1) * PHRASE_BARS, len(bars))
        if sections and sections[-1]["type"] == typ:
            sections[-1]["end_bar"] = b1
        else:
            sections.append({"type": typ, "start_bar": b0, "end_bar": b1})
    runs = {}
    for s in sections:
        s["start"] = bars[s["start_bar"]]["start"]
        s["end"] = bars[s["end_bar"] - 1]["end"]
        s["energy"] = round(float(energy[s["start_bar"]:s["end_bar"]].mean()), 3)
        runs[s["type"]] = runs.get(s["type"], 0) + 1
    type_energy = {typ: np.mean([s["energy"] for s in sections if s["type"] == typ]) for typ in runs}
    repeated = [typ for typ in runs if runs[typ] >= 2]
    chorus = max(repeated or list(runs), key=lambda typ: type_energy[typ])
    med = float(np.median([s["energy"] for s in sections]))
    e_min, e_max = min(s["energy"] for s in sections), max(s["energy"] for s in sections)
    for i, s in enumerate(sections):
        s["index"] = i
        # Relative to this song: quietest section = 0, loudest = 1.
        s["energy_rel"] = round((s["energy"] - e_min) / max(e_max - e_min, 1e-9), 3)
        if s["type"] == chorus:
            s["label"] = "chorus"
        elif i == 0 and s["energy"] <= med:
            s["label"] = "intro"
        elif i == len(sections) - 1 and s["energy"] <= med:
            s["label"] = "outro"
        elif s["energy_rel"] < 0.4:
            s["label"] = "breakdown"
        else:
            s["label"] = "verse"
        s["energy_level"] = "low" if s["energy_rel"] < 0.4 else "mid" if s["energy_rel"] < 0.75 else "high"
        s["bass_share"] = round(float(np.mean([bars[i]["bass"] for i in range(s["start_bar"], s["end_bar"])])), 2)

    return {
        "song": path.name,
        "duration": round(duration, 3),
        "tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 2),
        "seconds_per_beat": round(spb, 4),
        "downbeat_phase": phase,
        # Song edges (seconds): audible -> full loudness ... last full loudness -> practically silent.
        "music_start": round(music_start, 3),
        "fade_in_end": round(fade_in_end, 3),
        "fade_out_start": round(fade_out_start, 3),
        "music_end": round(music_end, 3),
        "beats": np.round(beats, 3).tolist(),
        "downbeats": np.round(downbeats, 3).tolist(),
        "bars": bars,
        "sections": [{k: s[k] for k in ("index", "label", "type", "energy_level", "energy", "energy_rel", "bass_share",
                                        "start", "end", "start_bar", "end_bar")} for s in sections],
    }


def save_plot(result: dict, path: Path, y_env=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 3.6))
    colors = {"intro": "#9ecae1", "verse": "#a1d99b", "chorus": "#fc9272", "breakdown": "#bcbddc", "outro": "#d9d9d9"}
    for s in result["sections"]:
        ax.axvspan(s["start"], s["end"], color=colors.get(s["label"], "#eee"), alpha=0.6)
        ax.text((s["start"] + s["end"]) / 2, 1.04, f"{s['label']} ({s['type']})", ha="center", fontsize=8)
    bars = result["bars"]
    ax.step([b["start"] for b in bars] + [bars[-1]["end"]], [b["energy"] for b in bars] + [bars[-1]["energy"]],
            where="post", color="k", lw=1.2, label="Energie pro Takt")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(0, result["duration"])
    ax.set_xlabel("Zeit (s)")
    ax.set_ylabel("Energie")
    ax.set_title(f"{result['song']} - {result['tempo_bpm']:.1f} BPM", fontsize=10, loc="left")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.16)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("song", type=Path)
    p.add_argument("--bar-offset", type=int, default=None, help="Taktanfang manuell (0-3)")
    p.add_argument("--types", type=int, default=None, help="Anzahl Abschnitts-Typen erzwingen (sonst automatisch)")
    args = p.parse_args()
    result = analyze(args.song, args.bar_offset, args.types)
    out = ROOT / "outputs" / f"analysis_{args.song.stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    save_plot(result, out.with_suffix(".png"))
    print(f"{result['song']}: {result['tempo_bpm']} BPM, {len(result['bars'])} Takte, erster Taktanfang {result['downbeats'][0]}s")
    for s in result["sections"]:
        print(f"  {s['start']:6.1f}s - {s['end']:6.1f}s  {s['label']:9s} Typ {s['type']}  "
              f"Energie {s['energy_rel']:.2f} relativ ({s['energy_level']})  Bass in {s['bass_share']:.0%} der Takte")
    nb = [b for b in result["bars"] if not b["bass"]]
    print(f"Takte ohne Bass: {len(nb)} ({', '.join(f'{b['start']:.0f}s' for b in nb)})")
    print(f"Musik hoerbar ab {result['music_start']:.1f}s, volle Lautstaerke ab {result['fade_in_end']:.1f}s | "
          f"Ausblenden ab {result['fade_out_start']:.1f}s, still ab {result['music_end']:.1f}s")
    print(f"-> {out}\n-> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
