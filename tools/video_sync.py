"""Measure the arm's VISIBLE timing from a phone video of a dance run.

  1. The video's sound is aligned to the original song -> every frame gets its song time.
  2. Optical flow in the moving area of the picture -> moments where the arm reverses direction.
  3. Each reversal is compared with the nearest beat (analysis JSON) -> "turns X ms after the beat",
     per move of the plan that was danced.

Usage:
    python tools/video_sync.py inputs/IMG_4171.MOV
    python tools/video_sync.py inputs/IMG_4171.MOV --song inputs/pdoom.wav --plan outputs/plan_pdoom.json
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate

ROOT = Path(__file__).resolve().parent.parent
SR, HOP = 22050, 128


def band_flux(y, n_bands: int = 40):
    """Per-band spectral flux (vocals, bass, hi-hats separately), each band normalized."""
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=n_bands, fmin=40))
    flux = np.maximum(0.0, np.diff(S, axis=1))
    return (flux - flux.mean(1, keepdims=True)) / (flux.std(1, keepdims=True) + 1e-9)


def align(video_audio: np.ndarray, song: np.ndarray) -> tuple[float, float]:
    """Song time of the video's first sample (seconds) and a match quality (peak z-score).

    Summing the correlation over 40 frequency bands makes the match unique even in a song that
    repeats itself: rhythm alone fits at every repetition, the combination of all bands only once.
    """
    a, b = band_flux(video_audio), band_flux(song)
    c = sum(correlate(b[i], a[i], mode="full", method="fft") for i in range(len(a)))
    lags = np.arange(-a.shape[1] + 1, b.shape[1])
    k = int(np.argmax(c))
    z = (c[k] - c.mean()) / (c.std() + 1e-9)
    return lags[k] * HOP / SR, float(z)


def _flows(video: Path, width: int):
    cap = cv2.VideoCapture(str(video))
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        g = cv2.cvtColor(cv2.resize(frame, (width, int(h * width / w))), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            yield cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev = g
    cap.release()


def motion_series(video: Path, width: int = 320):
    """Per frame: time, mean arm flow (vx, vy) minus camera shake, moving area; plus the region of interest.

    Two passes to keep memory small: first find where the arm moves over the whole video, then measure
    the motion inside that region, relative to the background (hand-held camera).
    """
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30.0
    acc = None
    for f in _flows(video, width):
        mag = np.linalg.norm(f, axis=2)
        acc = mag if acc is None else acc + mag
    if acc is None:
        raise SystemExit("Konnte keine Videobilder lesen.")
    acc = cv2.GaussianBlur(acc, (0, 0), 3)
    roi = acc > 0.3 * acc.max()
    background = acc < 0.1 * acc.max()
    t, vx, vy, area = [], [], [], []
    for i, f in enumerate(_flows(video, width)):
        bg = np.median(f[background], axis=0) if background.any() else np.zeros(2)
        f = f - bg  # remove camera shake
        mag = np.linalg.norm(f, axis=2)
        moving = roi & (mag > 0.3)
        t.append((i + 1) / fps)
        area.append(moving.mean())
        vx.append(f[..., 0][moving].mean() if moving.any() else 0.0)
        vy.append(f[..., 1][moving].mean() if moving.any() else 0.0)
    return np.array(t), np.array(vx), np.array(vy), np.array(area), fps, roi


def reversals(t, v, min_speed):
    """Times where a smoothed velocity changes sign between clearly moving phases (zero crossing, interpolated)."""
    out = []
    for i in range(1, len(v)):
        if v[i - 1] * v[i] < 0:
            before = np.abs(v[max(0, i - 4):i]).max()
            after = np.abs(v[i:i + 4]).max()
            if before > min_speed and after > min_speed:
                out.append(t[i - 1] + (t[i] - t[i - 1]) * abs(v[i - 1]) / (abs(v[i - 1]) + abs(v[i])))
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--song", type=Path, default=ROOT / "inputs" / "pdoom.wav")
    p.add_argument("--plan", type=Path, default=None)
    p.add_argument("--analysis", type=Path, default=None)
    args = p.parse_args()
    plan_path = args.plan or ROOT / "outputs" / f"plan_{args.song.stem}.json"
    ana_path = args.analysis or ROOT / "outputs" / f"analysis_{args.song.stem}.json"
    ana = json.loads(ana_path.read_text())
    plan = json.loads(plan_path.read_text())
    beats = np.array(ana["beats"])
    spb = float(ana["seconds_per_beat"])

    with tempfile.TemporaryDirectory() as tmp:
        wav, small = Path(tmp) / "a.wav", Path(tmp) / "v.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.video), "-vn", "-ac", "1", "-ar", str(SR), str(wav)], check=True)
        print("Verkleinere Video fuer die Analyse ...")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.video), "-an", "-vf", "scale=480:-2",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(small)], check=True)
        va, _ = librosa.load(wav, sr=SR, mono=True)
        song, _ = librosa.load(args.song, sr=SR, mono=True)
        level = 20 * np.log10(np.sqrt(np.mean(va ** 2)) + 1e-12)
        delta, z = align(va, song)
        print(f"\nTon im Video: {level:.0f} dBFS | Abgleich mit dem Song: Guete {z:.0f} (ab ~8 sicher) | "
              f"Video-Sekunde 0 = Song-Sekunde {delta:.2f}")
        if z < 8:
            raise SystemExit("Der Ton im Video passt nicht sicher zum Song (zu leise oder anderer Song?).")
        print("Berechne Bewegung (optischer Fluss) ...")
        t, vx, vy, area, fps, roi = motion_series(small)
    ts = t + delta  # song time of each frame
    print(f"Video: {len(t) / fps:.1f} s, {fps:.0f} fps -> Song {ts[0]:.1f}-{ts[-1]:.1f} s | Arm-Bereich: {roi.mean() * 100:.0f} % des Bildes")

    k = max(1, int(round(fps * 0.1)))  # 100 ms smoothing
    rows = []
    for name, v in (("horizontal", uniform_filter1d(vx, k)), ("vertikal", uniform_filter1d(vy, k))):
        thr = 0.25 * np.percentile(np.abs(v), 95)
        for r in reversals(ts, v, thr):
            bp = np.interp(r, beats, np.arange(len(beats)))
            rows.append((r, name, (bp - np.round(bp)) * spb * 1000, bp - np.floor(bp)))
    rows.sort()
    print(f"{len(rows)} Richtungswechsel gefunden.\n")
    print("Pro Move: wann dreht der Arm SICHTBAR um, relativ zum naechsten Beat (+ = nach dem Beat)")
    print(f"  {'Start':>6s}  {'Move':11s} {'Anzahl':>6s} {'Median':>8s} {'Streuung':>9s}   Verteilung der Wende-Phase (0 = Beat, .5 = zwischen)")
    starts = [s["start"] for s in plan]
    summary = {}
    for i, s in enumerate(plan):
        a, b = s["start"] + 0.5, (plan[i + 1]["start"] if i + 1 < len(plan) else 1e9)
        sel = [r for r in rows if a <= r[0] < b]
        if len(sel) < 3:
            continue
        offs = np.array([r[2] for r in sel])
        ph = np.array([r[3] for r in sel])
        hist = np.histogram(ph, bins=4, range=(0, 1))[0]
        summary.setdefault(s["move"], []).extend(offs.tolist())
        print(f"  {s['start']:6.1f}  {s['move']:11s} {len(sel):6d} {np.median(offs):+7.0f}ms {np.percentile(offs, 75) - np.percentile(offs, 25):8.0f}ms   "
              f"[0-.25:{hist[0]} .25-.5:{hist[1]} .5-.75:{hist[2]} .75-1:{hist[3]}]")
    print("\nZusammenfassung pro Move (alle Einsaetze):")
    for m, offs in sorted(summary.items(), key=lambda kv: np.median(kv[1])):
        offs = np.array(offs)
        print(f"  {m:12s} {len(offs):4d} Wenden | Median {np.median(offs):+5.0f} ms | Anteil nahe am Beat (+-80 ms): "
              f"{np.mean(np.abs(offs) < 80) * 100:3.0f} %")
    out = ROOT / "outputs" / f"video_sync_{args.video.stem}.npz"
    np.savez(out, ts=ts, vx=vx, vy=vy, area=area, rows=np.array([(r[0], r[2]) for r in rows]))
    print(f"\nRohdaten: {out}")


if __name__ == "__main__":
    main()
