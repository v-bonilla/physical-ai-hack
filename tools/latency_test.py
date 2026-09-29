"""Measure how late the music really comes out of the speaker (e.g. Bluetooth adds 150-250 ms).

Plays a few clicks through the current output device and records them with the current input
device (MacBook microphone). The delay between "sent" and "heard", minus what macOS already reports
(and dance.py already compensates), is the extra delay to pass as --audio-latency-ms.

Run it with exactly the speaker setup of the demo, speaker ~0.5-1 m from the laptop, room quiet.
macOS asks once for microphone access for the terminal.

Usage:
    python tools/latency_test.py
"""

import numpy as np

SR = 48000
N_CLICKS = 8
GAP_S = 0.6
LEAD_S = 0.5


def click_signal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test signal, the click template, and the sample positions the clicks were sent at."""
    t = np.arange(int(0.012 * SR)) / SR
    click = (np.sin(2 * np.pi * 2000 * t) * np.hanning(len(t))).astype(np.float32)
    sent = (LEAD_S + GAP_S * np.arange(N_CLICKS)) * SR
    sig = np.zeros(int((LEAD_S + GAP_S * N_CLICKS + 1.0) * SR), dtype=np.float32)
    for s in sent.astype(int):
        sig[s: s + len(click)] += 0.8 * click
    return sig, click, sent


def detect_delays(rec: np.ndarray, click: np.ndarray, sent: np.ndarray) -> np.ndarray:
    """Delay (s) of each click in the recording, via matched filter around its expected position."""
    corr = np.abs(np.correlate(rec, click, mode="valid"))
    delays = []
    for s in sent.astype(int):
        lo, hi = s, min(len(corr), s + int(0.45 * SR))  # look up to 450 ms after sending
        if hi <= lo:
            continue
        window = corr[lo:hi]
        k = int(np.argmax(window))
        if window[k] > 6 * (np.median(window) + 1e-9):  # clearly above the room noise
            delays.append(k / SR)
    return np.array(delays)


def main():
    import sounddevice as sd

    sig, click, sent = click_signal()
    out_dev = sd.query_devices(kind="output")["name"]
    in_dev = sd.query_devices(kind="input")["name"]
    print(f"Ausgabe: {out_dev} | Mikrofon: {in_dev}\nSpiele {N_CLICKS} Klicks ab ...")
    rec = sd.playrec(sig, samplerate=SR, channels=1, dtype="float32")
    in_lat, out_lat = sd.get_stream().latency
    sd.wait()
    delays = detect_delays(rec[:, 0], click, sent)
    if len(delays) < N_CLICKS // 2:
        raise SystemExit(f"Nur {len(delays)} von {N_CLICKS} Klicks gehoert - lauter stellen / Lautsprecher naeher ans Mikro.")
    total = float(np.median(delays))
    extra = total - in_lat - out_lat
    print(f"Gehoert: {len(delays)}/{N_CLICKS} Klicks, Verzoegerung gesamt {total * 1000:.0f} ms "
          f"(Streuung {delays.std() * 1000:.0f} ms)")
    print(f"macOS meldet: Ausgabe {out_lat * 1000:.0f} ms + Mikrofon {in_lat * 1000:.0f} ms "
          f"(die Ausgabe gleicht dance.py schon aus)")
    print(f"-> zusaetzliche, unbekannte Verzoegerung: {extra * 1000:.0f} ms")
    if extra > 0.02:
        print(f"-> dance.py mit  --audio-latency-ms {extra * 1000:.0f}  starten")
    else:
        print("-> nichts zu tun, der Lautsprecher ist schnell genug")


if __name__ == "__main__":
    main()
