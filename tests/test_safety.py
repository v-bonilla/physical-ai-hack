from pathlib import Path

import numpy as np
import pytest

import dancebot.analyze as an
import dancebot.robot as rb
from dancebot.audio import NullPlayer
from dancebot.choreo import sway
from dancebot.cli import main
from dancebot.perform import Performer
from dancebot.robot import CalibrationError, FakeArm

BEATS = [1.0 + 0.5 * k for k in range(16)]


def _fake_follower(tmp_path, has_file: bool, calibrated: bool):
    calls = []

    class Cfg:
        def __init__(self, **kw):
            self.kw = kw

    class Follower:
        def __init__(self, cfg):
            calls.append(("init", cfg.kw))
            self.calibration_fpath = Path(tmp_path) / f"{cfg.kw['id']}.json"
            if has_file:
                self.calibration_fpath.write_text("{}")
            self.is_calibrated = calibrated

        def connect(self, calibrate=True):
            calls.append(("connect", calibrate))

        def disconnect(self):
            calls.append(("disconnect",))

    return Follower, Cfg, calls


@pytest.mark.parametrize("has_file,calibrated", [(False, True), (True, False)])
def test_uncalibrated_arm_refuses_without_motion(tmp_path, monkeypatch, has_file, calibrated):
    Follower, Cfg, calls = _fake_follower(tmp_path, has_file, calibrated)
    monkeypatch.setattr(rb, "import_follower", lambda: (Follower, Cfg))
    arm = rb.LeRobotArm(port="/dev/tty.usbmodemX", robot_id="dancer")
    assert calls[0][1]["disable_torque_on_disconnect"] is False
    with pytest.raises(CalibrationError, match="lerobot-calibrate --robot.type=so101_follower "
                                               "--robot.port=/dev/tty.usbmodemX --robot.id=dancer"):
        arm.connect()
    if has_file:
        assert ("connect", False) in calls and calls[-1] == ("disconnect",)
    else:
        assert not any(c[0] == "connect" for c in calls)


def test_cli_rejects_bad_safety_flags(beat_song):
    for flags in (["--max-speed", "0"], ["--max-speed", "-5"], ["--margin", "-1"]):
        with pytest.raises(SystemExit):
            main(["dance", str(beat_song), "--simulate", *flags])


def test_safe_clamps_after_step_cap():
    perf = Performer(FakeArm(), NullPlayer(1.0, simulated=True), sway(10), BEATS, hud=False)
    perf.set_start_pose(np.zeros(6))
    perf.last_cmd = perf.hi + 50  # somehow far outside the range
    cmd = perf.safe(perf.hi + 100)
    assert np.all(cmd <= perf.hi) and np.all(cmd >= perf.lo)


class _InterruptOnce(FakeArm):
    armed = False

    def send_pose(self, pose):
        if self.armed:
            self.armed = False
            raise KeyboardInterrupt
        super().send_pose(pose)


def test_ctrl_c_during_ease_back_still_returns(monkeypatch):
    start = np.array([5.0, -40.0, 60.0, 20.0, -3.0, 10.0])
    arm = _InterruptOnce(start_pose=start)
    perf = Performer(arm, NullPlayer(4.0, simulated=True), sway(20), BEATS, simulate=True, hud=False)
    orig = perf.ease_back

    def ease_back(sp):
        arm.armed = True
        orig(sp)

    perf.ease_back = ease_back
    res = perf.run()
    assert not arm.armed  # the interrupt fired
    assert np.allclose(res["final_cmd"], start) and np.allclose(arm.commands[-1], start)
    assert not arm.connected


def test_auto_rejects_cached_librosa_when_beat_this_available(beat_song, monkeypatch):
    an.analyze(beat_song, backend="librosa", use_cache=True)
    monkeypatch.setattr(an, "beat_this_available", lambda: True)
    monkeypatch.setattr(an, "beats_beat_this", lambda p: (np.arange(0, 20, 0.5), np.arange(0, 20, 2.0)))
    a = an.analyze(beat_song, backend="auto")
    assert a["backend"] == "beat_this" and "cached" not in a
    b = an.analyze(beat_song, backend="auto")
    assert b["cached"] and "backend=beat_this (cached)" in an.summarize(b)
