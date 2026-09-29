import argparse

import numpy as np
import pytest

import dancebot.smoke as smoke
from dancebot.audio import NullPlayer
from dancebot.choreo import JOINTS, OFFSET_CAPS, groove, make_move, pose_at
from dancebot.perform import Performer
from dancebot.robot import FakeArm

PAN, LIFT, ELBOW, WFLEX, WROLL, GRIP = range(6)


def test_groove_keyframes_exact():
    c = groove()
    expect = {  # phase: {joint: value}
        0: {PAN: 12, WFLEX: -10, WROLL: 0, GRIP: 0},
        0.5: {WFLEX: 0},
        1: {PAN: -12, WFLEX: -10},
        2.5: {WFLEX: 0},
        3: {PAN: -12, WFLEX: -10, WROLL: 0},
        4: {PAN: 0, WFLEX: 0, WROLL: 20, GRIP: 0},
        4.5: {GRIP: 25},
        5: {WROLL: -20, GRIP: 0},
        6: {WROLL: 20, GRIP: 0},
        6.5: {GRIP: 25},
        7: {PAN: 0, WROLL: -20, GRIP: 0},
        8: {PAN: 12, WFLEX: -10, WROLL: 0, GRIP: 0},  # wraps to bar 1 beat 0
    }
    for phase, joints in expect.items():
        p = pose_at(c, phase)
        for j, v in joints.items():
            assert p[j] == pytest.approx(v, abs=1e-9), (phase, JOINTS[j])


def test_groove_smooth_between_keys_and_safe_joints():
    c = groove()
    # per-joint easing: pan swings through 0 at the half beat instead of stalling at a keyframe
    assert pose_at(c, 0.5)[PAN] == pytest.approx(0.0)
    assert pose_at(c, 0.25)[PAN] == pytest.approx(12 - 24 * 0.15625)  # smoothstep(0.25) = 0.15625
    assert pose_at(c, 7.5)[WROLL] == pytest.approx(-10.0)  # -20 -> 0 over the last beat
    for ph in np.arange(0, 16, 0.01):
        p = pose_at(c, ph)
        assert p[LIFT] == 0 and p[ELBOW] == 0
        assert p[GRIP] >= -1e-9
    assert np.all(np.nan_to_num(c.poses[:, GRIP]) >= 0)


def test_scale_caps():
    big = make_move("groove", 1.5)
    assert np.nanmax(np.abs(big.poses), axis=0) == pytest.approx([18, 0, 0, 15, 30, 37.5])
    capped = make_move("sway", 1.5, amplitude=30)
    assert np.nanmax(np.abs(capped.poses[:, PAN])) == pytest.approx(OFFSET_CAPS[PAN])
    with pytest.raises(ValueError):
        make_move("groove", 2.0)


def test_downbeat_alignment_and_gripper_clamp():
    beats = [0.5 * k for k in range(40)]
    start = np.array([0.0, 0, 0, 0, 0, 90.0])
    perf = Performer(FakeArm(start), NullPlayer(1, simulated=True), groove(), beats, downbeats=[1.5, 3.5], hud=False)
    perf.set_start_pose(start)
    assert perf.d0 == 3
    assert np.allclose(perf.target_at(0.5)[0], start)  # before the lead-in beat: hold start
    assert np.allclose(perf.target_at(1.5)[0], start + pose_at(groove(), 0))  # phase 0 on the downbeat
    assert np.allclose(perf.target_at(1.5 + 4 * 0.5)[0], start + pose_at(groove(), 4))
    assert "bar    1.1" in perf.hud_line(3.0) and "--" in perf.hud_line(1.0)


@pytest.mark.parametrize("grip0,clap", [(5.0, 25.0), (95.0, -25.0)])
def test_gripper_opens_away_from_nearer_end(grip0, clap):
    beats = [0.5 * k for k in range(40)]
    start = np.array([0.0, 0, 0, 0, 0, grip0])
    arm = FakeArm(start_pose=start)
    perf = Performer(arm, NullPlayer(12.0, simulated=True), groove(), beats, downbeats=[0.0],
                     simulate=True, hud=False)
    perf.set_start_pose(start)
    assert perf.target_at(4.5 * 0.5)[0][GRIP] == pytest.approx(grip0 + clap)  # clap on beat 4.5
    perf.run()
    grips = np.array(arm.commands)[:, GRIP]
    assert grips.min() >= 0 and grips.max() <= 100
    assert (grips.max() if clap > 0 else grips.min()) == pytest.approx(grip0 + clap, abs=0.5)


def test_send_refuses_non_finite():
    start = np.array([0.0, -30, 40, 10, 0, 20])

    class NanOnce(FakeArm):
        pass

    arm = NanOnce(start_pose=start)
    perf = Performer(arm, NullPlayer(12.0, simulated=True), groove(), [0.5 * k for k in range(40)],
                     simulate=True, hud=False)
    orig = perf.target_at
    calls = {"n": 0}

    def bad_target(t):
        calls["n"] += 1
        pose, b = orig(t)
        return (pose * np.nan if calls["n"] == 60 else pose), b

    perf.target_at = bad_target
    with pytest.raises(RuntimeError, match="non-finite"):
        perf.run()
    cmds = np.array(arm.commands)
    assert np.all(np.isfinite(cmds)) and np.allclose(cmds[-1], start) and not arm.connected


def test_smoke_song(tmp_path):
    import soundfile as sf

    from dancebot.analyze import analyze
    from dancebot.smoke_song import make

    p = make(tmp_path / "s.wav", bpm=110)
    y, sr = sf.read(p)
    assert sr == 44100 and y.shape[1] == 2
    assert 12 * 4 * 60 / 110 <= len(y) / sr <= 12 * 4 * 60 / 110 + 1.5
    assert 20 * np.log10(np.abs(y).max()) == pytest.approx(-1.0, abs=0.1)
    a = analyze(p, backend="librosa", use_cache=False)
    assert abs(a["tempo_bpm"] - 110) / 110 <= 0.04


@pytest.mark.parametrize("found,expected", [([], []), (["/dev/ttyACM0"], ["/dev/ttyACM0"]),
                                            (["/dev/ttyACM1", "/dev/ttyACM0"], ["/dev/ttyACM0", "/dev/ttyACM1"])])
def test_detect_ports(monkeypatch, found, expected):
    monkeypatch.delenv("DANCEBOT_FOLLOWER_PORT", raising=False)
    monkeypatch.setattr(smoke.glob, "glob", lambda pattern: list(found))
    assert smoke.detect_ports() == expected
    assert smoke.detect_ports("/dev/x") == ["/dev/x"]


def test_classify_bpm():
    assert smoke.classify_bpm(111) == ("PASS", 1.0)
    assert smoke.classify_bpm(55.5) == ("WARN", 2.0)
    assert smoke.classify_bpm(219) == ("WARN", 0.5)
    assert smoke.classify_bpm(150)[0] == "FAIL"


def _args(song):
    return argparse.Namespace(song=str(song), port=None, robot_id=None, seconds=6.0, no_robot=False,
                              no_audio=True, latency_ms=80.0)


class CountingArm(FakeArm):
    connects = 0

    def connect(self):
        self.connects += 1
        super().connect()


@pytest.mark.parametrize("tracking,ok", [(0.6, True), (0.0, False)])
def test_smoke_end_to_end(tmp_path, monkeypatch, beat_song, tracking, ok):
    monkeypatch.chdir(tmp_path)
    # librosa only: keep the test offline and fast
    monkeypatch.setattr(smoke, "stage_beats", lambda r, song, synthetic: (
        __import__("dancebot.analyze", fromlist=["analyze"]).analyze(song, backend="librosa", use_cache=False),
        "librosa", 1.0))
    start = np.array([0.0, -30, 40, 10, 0, 20])
    arm = CountingArm(start_pose=start, tracking=tracking)
    code = smoke.run_smoke(_args(beat_song), arm=arm, fast=True, report_dir=tmp_path)
    reports = list(tmp_path.glob("smoke-report-*.txt"))
    assert len(reports) == 1
    text = reports[0].read_text()
    for name in ("env", "song", "audio", "port", "calibration", "connect", "joint/shoulder_pan",
                 "joint/gripper", "dance", "RESULT"):
        assert name in text
    assert not arm.connected and np.allclose(arm.commands[-1], start)
    assert arm.connects == 1  # connect, joints and dance share one session
    if ok:
        assert code == 2 and "RESULT: PARTIAL (audio not tested)" in text  # --no-audio
        assert "[PASS] joint/wrist_roll" in text and "[PASS] dance" in text
    else:
        assert code != 0 and "[FAIL] joint/shoulder_pan" in text
        assert "[SKIP] dance" in text


def test_smoke_gripper_near_top_steps_down(tmp_path, monkeypatch, beat_song):
    monkeypatch.setattr(smoke, "stage_beats", lambda r, song, synthetic: (
        __import__("dancebot.analyze", fromlist=["analyze"]).analyze(song, backend="librosa", use_cache=False),
        "librosa", 1.0))
    arm = CountingArm(start_pose=np.array([0.0, -30, 40, 10, 0, 99.0]))
    smoke.run_smoke(_args(beat_song), arm=arm, fast=True, report_dir=tmp_path)
    text = next(tmp_path.glob("smoke-report-*.txt")).read_text()
    assert "[PASS] joint/gripper      start 99.0, commanded -3" in text


def test_smoke_ctrl_c_before_arm(tmp_path, monkeypatch, beat_song):
    def boom(r, no_audio):
        raise KeyboardInterrupt

    monkeypatch.setattr(smoke, "stage_beats", lambda r, song, synthetic: (None, "librosa", 1.0))
    monkeypatch.setattr(smoke, "stage_audio", boom)
    arm = CountingArm()
    code = smoke.run_smoke(_args(beat_song), arm=arm, fast=True, report_dir=tmp_path)
    text = next(tmp_path.glob("smoke-report-*.txt")).read_text()
    assert code == 1 and "[FAIL] audio" in text and "aborted" in text and "RESULT: FAIL" in text
    assert arm.connects == 0 and "stage              status" in text


def test_result_line():
    r = smoke.Report()
    for n in ("env", "audio", "port", "calibration", "connect", "joints", "dance"):
        r.stages.append((n, "PASS", ""))
    assert smoke.result_line(r) == ("RESULT: PASS", 0)
    r.stages[1] = ("audio", "WARN", "")
    assert smoke.result_line(r)[1] == 0
    r.stages[2] = ("port", "SKIP", "")
    assert smoke.result_line(r) == ("RESULT: PARTIAL (arm not tested)", 2)
    r.stages[1] = ("audio", "SKIP", "")
    assert smoke.result_line(r) == ("RESULT: PARTIAL (arm and audio not tested)", 2)
    r.stages.append(("dance", "FAIL", ""))
    assert smoke.result_line(r) == ("RESULT: FAIL", 1)


def test_port_globs_mac(monkeypatch):
    monkeypatch.delenv("DANCEBOT_FOLLOWER_PORT", raising=False)
    monkeypatch.setattr(smoke.sys, "platform", "darwin")
    fs = {"/dev/tty.usbmodem*": [], "/dev/tty.usbserial*": ["/dev/tty.usbserial-1"],
          "/dev/tty.wchusbserial*": ["/dev/tty.wchusbserial-2"]}
    monkeypatch.setattr(smoke.glob, "glob", lambda p: fs.get(p, []))
    assert smoke.detect_ports() == ["/dev/tty.usbserial-1", "/dev/tty.wchusbserial-2"]
