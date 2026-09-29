"""Robot IO: real SO-101 follower via LeRobot, or a fake arm for dry runs."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .choreo import JOINTS

ROBOT_EXTRA_HINT = "LeRobot is not installed. Run: uv sync"

# lerobot 0.6.x uses so_follower; older releases used per-model module names
FOLLOWER_PATHS = [
    ("lerobot.robots.so_follower", "SO101Follower", "SO101FollowerConfig"),
    ("lerobot.robots.so101_follower", "SO101Follower", "SO101FollowerConfig"),
    ("lerobot.common.robots.so101_follower", "SO101Follower", "SO101FollowerConfig"),
]


def import_follower() -> tuple[Any, Any]:
    errors = []
    for mod, cls, cfg in FOLLOWER_PATHS:
        try:
            m = importlib.import_module(mod)
            return getattr(m, cls), getattr(m, cfg)
        except (ImportError, AttributeError) as e:
            errors.append(f"{mod}: {e}")
    try:
        importlib.import_module("lerobot")
    except ImportError:
        raise ImportError(ROBOT_EXTRA_HINT) from None
    raise ImportError("LeRobot is installed but no known SO-101 module path worked:\n  " + "\n  ".join(errors))


DEFAULT_ROBOT_ID = "dancer"


class CalibrationError(RuntimeError):
    pass


def calibration_help(port: str, robot_id: str, fpath, reason: str) -> str:
    return (f"{reason} for robot id {robot_id!r} (expected {fpath}, i.e. "
            f"~/.cache/huggingface/lerobot/calibration/robots/so_follower/{robot_id}.json).\n"
            f"The arm was not moved. Calibrate first, or pass the id you calibrated with:\n"
            f"  uv run lerobot-calibrate --robot.type=so101_follower --robot.port={port} --robot.id={robot_id}")


def to_action(pose: np.ndarray) -> dict[str, float]:
    return {f"{j}.pos": float(v) for j, v in zip(JOINTS, pose)}


def from_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.array([float(obs[f"{j}.pos"]) for j in JOINTS])


class RobotIO:
    def check_calibration_file(self) -> str:
        return "not applicable"

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def read_pose(self) -> np.ndarray: raise NotImplementedError
    def send_pose(self, pose: np.ndarray) -> None: raise NotImplementedError


class LeRobotArm(RobotIO):
    def __init__(self, port: str, robot_id: str = DEFAULT_ROBOT_ID, release: bool = False):
        Follower, FollowerConfig = import_follower()
        self.port, self.robot_id = port, robot_id
        # keeping torque on at disconnect holds the start pose instead of dropping the arm
        self.robot = Follower(FollowerConfig(port=port, id=robot_id, disable_torque_on_disconnect=release))

    def check_calibration_file(self) -> str:
        """Raise CalibrationError unless the calibration file exists; returns its path. No bus access."""
        fpath = self.robot.calibration_fpath
        if not fpath.is_file():
            raise CalibrationError(calibration_help(self.port, self.robot_id, fpath, "no calibration file"))
        return str(fpath)

    def connect(self) -> None:
        # never let lerobot calibrate interactively: it disables torque and rewrites homing
        fpath = self.check_calibration_file()
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            self.robot.bus.disconnect(disable_torque=False)  # never cut torque here, even with --release
            raise CalibrationError(calibration_help(self.port, self.robot_id, fpath,
                                                    "motor calibration does not match the file"))

    def disconnect(self) -> None:
        self.robot.disconnect()

    def read_pose(self) -> np.ndarray:
        return from_obs(self.robot.get_observation())

    def send_pose(self, pose: np.ndarray) -> None:
        self.robot.send_action(to_action(pose))


class FakeArm(RobotIO):
    """Records every command; the plant moves a fraction of the way toward each command."""

    def __init__(self, start_pose: np.ndarray | None = None, tracking: float = 0.6):
        self.pose = np.zeros(len(JOINTS)) if start_pose is None else np.asarray(start_pose, dtype=float).copy()
        self.tracking = tracking
        self.commands: list[np.ndarray] = []
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_pose(self) -> np.ndarray:
        return self.pose.copy()

    def send_pose(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=float).copy()
        self.commands.append(pose)
        self.pose = self.pose + self.tracking * (pose - self.pose)


def list_ports() -> list[str]:
    try:
        from serial.tools import list_ports as lp  # pyserial comes with lerobot[feetech]

        return sorted(p.device for p in lp.comports())
    except ImportError:
        import glob

        return sorted(glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
