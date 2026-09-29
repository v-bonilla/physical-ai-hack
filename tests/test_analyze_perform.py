import numpy as np
import pytest

from dancebot.analyze import analyze, cache_path
from dancebot.audio import NullPlayer
from dancebot.choreo import sway
from dancebot.cli import main
from dancebot.perform import Performer
from dancebot.robot import FakeArm


def test_librosa_tempo_and_cache(beat_song):
    a = analyze(beat_song, backend="librosa", use_cache=True)
    assert a["backend"] == "librosa"
    assert abs(a["tempo_bpm"] - 120) / 120 < 0.03
    assert len(a["downbeats"]) >= len(a["beats"]) // 4 - 1
    assert cache_path(beat_song).exists()
    again = analyze(beat_song, backend="librosa")
    assert again.pop("cached") is True and again == a


def _simulate(amplitude=15.0, max_speed=180.0):
    start = np.array([5.0, -40.0, 60.0, 20.0, -3.0, 10.0])
    arm = FakeArm(start_pose=start)
    beats = [1.0 + 0.5 * k for k in range(16)]  # 120 BPM from t=1s
    perf = Performer(arm, NullPlayer(10.0, simulated=True), sway(amplitude), beats,
                     latency_s=0.0, max_speed=max_speed, simulate=True, hud=False)
    return perf, arm, start, beats, perf.run()


def test_simulated_dance_safety():
    perf, arm, start, beats, res = _simulate(amplitude=30.0, max_speed=150.0)
    cmds = np.array(arm.commands)
    steps = np.abs(np.diff(np.vstack([start, cmds]), axis=0))
    assert steps.max() <= perf.max_step + 1e-9
    assert np.all(cmds >= perf.lo - 1e-9) and np.all(cmds <= perf.hi + 1e-9)
    assert np.allclose(res["final_cmd"], start)
    assert np.allclose(cmds[:, 1:], start[1:])  # only the pan moves
    assert not arm.connected


def test_simulated_pan_hits_beats():
    perf, arm, start, beats, res = _simulate(amplitude=15.0)
    times = np.array([t for t, _ in perf.log])
    pans = np.array([p[0] for _, p in perf.log])
    dance = times > 0  # skip the ease-in, logged at song time 0
    for k, bt in enumerate(beats):
        i = np.argmin(np.abs(times[dance] - bt))
        expected = start[0] + (15 if k % 2 == 0 else -15)
        assert pans[dance][i] == pytest.approx(expected, abs=0.5)


def test_cli_simulate_end_to_end(beat_song, tmp_path, capsys):
    out = tmp_path / "poses.csv"
    main(["dance", str(beat_song), "--simulate", "--backend", "librosa", "--quiet", "--csv", str(out)])
    text = capsys.readouterr().out
    assert "BPM" in text and "done:" in text
    assert out.read_text().startswith("song_time,shoulder_pan")


@pytest.mark.parametrize("mult,n,bpm", [(0.5, 8, 60.0), (1.0, 16, 120.0), (2.0, 31, 240.0)])
def test_beat_mult(mult, n, bpm):
    beats = [1.0 + 0.5 * k for k in range(16)]
    perf = Performer(FakeArm(), NullPlayer(1.0, simulated=True), sway(), beats, beat_mult=mult, hud=False)
    assert perf.n_beats == n
    assert perf.tempo_bpm == pytest.approx(bpm)
    if mult == 2.0:
        assert perf.clock.beats[1] == pytest.approx(1.25)
    assert "BPM" in perf.hud_line(3.0) and f"{bpm:5.1f}" in perf.hud_line(3.0)


class _FailingPlayer(NullPlayer):
    def __init__(self, fail_on: str):
        super().__init__(10.0, simulated=True)
        self.fail_on = fail_on

    def open(self):
        if self.fail_on == "open":
            raise RuntimeError("no audio device")

    def time(self):
        if self.fail_on == "time" and self.sim_t >= 2.0:
            raise RuntimeError("stream died")
        return super().time()


@pytest.mark.parametrize("fail_on", ["open", "time"])
def test_error_paths_ease_back_and_disconnect(fail_on):
    start = np.array([5.0, -40.0, 60.0, 20.0, -3.0, 10.0])
    arm = FakeArm(start_pose=start)
    beats = [1.0 + 0.5 * k for k in range(16)]
    perf = Performer(arm, _FailingPlayer(fail_on), sway(20), beats, simulate=True, hud=False)
    with pytest.raises(RuntimeError):
        perf.run()
    assert not arm.connected
    if fail_on == "open":
        assert all(np.allclose(c, start) for c in arm.commands)  # no motion before the device check
    else:
        assert np.abs(np.array(arm.commands)[:, 0] - start[0]).max() > 10  # it was dancing
        assert np.allclose(arm.commands[-1], start)
