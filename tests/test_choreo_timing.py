import numpy as np
import pytest

from dancebot.choreo import MAX_SWAY, Choreo, load_choreo, pose_at, save_choreo, sway
from dancebot.timing import BeatClock


def test_sway_exact_on_beats():
    c = sway(15)
    for k in range(8):
        p = pose_at(c, k)
        assert p[0] == pytest.approx(15 if k % 2 == 0 else -15)
        assert np.all(p[1:] == 0)
    # eased midway: smoothstep(0.5) is 0.5, so the pan crosses zero between beats
    assert pose_at(c, 0.5)[0] == pytest.approx(0.0)
    assert pose_at(c, 0.25)[0] > 0 and pose_at(c, 1.75)[0] > -15


def test_sway_amplitude_clamped():
    assert pose_at(sway(100), 0)[0] == pytest.approx(MAX_SWAY)
    assert sway(20).relative


def test_keyframe_wrap_last_to_first():
    s = np.zeros((3, 7))
    s[:, 0] = [0, 1, 2]
    s[:, 1] = [0, 10, 20]
    c = Choreo(name="w", length_beats=3, recorded_bpm=100, samples=s)
    assert pose_at(c, 2)[0] == pytest.approx(20)
    assert pose_at(c, 3)[0] == pytest.approx(0)  # wraps back to the first keyframe
    assert pose_at(c, 2.5)[0] == pytest.approx(10)  # halfway 20 -> 0
    assert pose_at(c, -1)[0] == pytest.approx(20)


def test_continuous_linear_and_roundtrip(tmp_path):
    s = np.zeros((2, 7))
    s[:, 0] = [0, 1]
    s[:, 1] = [0, 10]
    c = Choreo(name="c", length_beats=2, recorded_bpm=90, samples=s, mode="continuous")
    assert pose_at(c, 0.25)[0] == pytest.approx(2.5)
    assert pose_at(c, 1.5)[0] == pytest.approx(5.0)
    c2 = load_choreo(save_choreo(c, tmp_path / "c.json"))
    assert c2.mode == "continuous" and np.allclose(c2.samples, c.samples)


def test_beatclock_interp_and_extrap():
    clock = BeatClock([1.0, 1.5, 2.0, 2.6, 3.1])
    assert clock.beat_at(1.0) == pytest.approx(0)
    assert clock.beat_at(1.25) == pytest.approx(0.5)
    assert clock.beat_at(2.3) == pytest.approx(2.5)  # uneven interval 2.0 -> 2.6
    assert clock.ibi == pytest.approx(0.5)
    assert clock.beat_at(0.0) == pytest.approx(-2.0)
    assert clock.beat_at(4.1) == pytest.approx(6.0)
    for b in (-1.0, 0.3, 2.5, 5.0):
        assert clock.beat_at(clock.time_at(b)) == pytest.approx(b)
