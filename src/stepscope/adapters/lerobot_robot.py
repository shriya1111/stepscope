"""Run stepscope on a real LeRobot arm (SO-100 / SO-101).

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from stepscope import Tracer, TraceConfig
    from stepscope.adapters.lerobot_robot import StepscopeGuard, so101_gate, so101_kinematics, JointMap

    robot = SO101Follower(SO101FollowerConfig(port="/dev/ttyACM0", cameras={...}))
    robot.connect()
    kin, joints = so101_kinematics(), JointMap()
    tracer = Tracer("runs/so101", config=TraceConfig(max_steps=600),
                    gate=so101_gate(kin, joints, keep_out=[{"name": "operator side", "center": [0.0, -0.30, 0.10], "radius": 0.12}]),
                    shadow=True)                       # record what the gate would do; change nothing
    guard = StepscopeGuard(robot, my_policy, tracer, kinematics=kin, joint_map=joints, camera_key="front")
    guard.run_episode(task="pick up the block", max_steps=600, hz=30)

`my_policy(observation)` is your own function: it takes LeRobot's observation
dict and returns the target joint positions in the robot's units (degrees for
the five arm joints, 0 to 100 for the gripper), in SO101_MOTORS order. It may
also return a dict with "action" plus optional "samples", "internals",
"belief" and "signals" to trace more.

The guard sits in the control loop: read the observation, run the policy, pass
the action through stepscope, send it. With Tracer(shadow=True) (use this
first) the action is never changed. With shadow=False the gated action is the
one sent, and a halt holds the arm in place.

This is a software check. It does not replace a physical emergency stop, and
it has been tested against a simulated stand-in (SimSO101 below), not on
hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..gate import Gate
from ..kinematics import Kinematics
from ..tracer import Tracer

SO101_MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SO101_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]  # model joints for the five arm motors
SO101_BODIES = ["Rotation_Pitch", "Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll", "Fixed_Jaw", "Gripper_Tip"]
MODEL_PATH = Path(__file__).parent.parent / "models" / "so101_kinematic.xml"


def so101_kinematics() -> Kinematics:
    """3D positions of the SO-101's links from its five arm joint angles (radians, model convention)."""
    return Kinematics(MODEL_PATH, SO101_JOINTS, SO101_BODIES)


@dataclass
class JointMap:
    """Converts the robot's joint readings to the kinematic model's angles.

    model_angle_rad = radians(sign * (robot_value - zero_deg)), per arm joint.

    The defaults assume the robot reports degrees with the same zero pose and
    direction as the model. LeRobot's calibration sets each joint's zero at the
    middle of its range, which is NOT guaranteed to match, so check this on
    your arm before trusting any Cartesian limit: put the arm in a pose you can
    measure, call `so101_kinematics().points(joint_map.to_model(q))`, and
    compare the gripper position with a ruler. Adjust `sign` and `zero_deg`
    until they agree.
    """

    sign: list[float] = field(default_factory=lambda: [1.0] * 5)
    zero_deg: list[float] = field(default_factory=lambda: [0.0] * 5)

    def to_model(self, q_robot) -> np.ndarray:
        q = np.asarray(q_robot, dtype=float)[:5]
        return np.radians(np.asarray(self.sign) * (q - np.asarray(self.zero_deg)))

    def from_model(self, q_model) -> np.ndarray:
        return np.degrees(np.asarray(q_model, dtype=float)) / np.asarray(self.sign) + np.asarray(self.zero_deg)


def so101_gate(
    kin: Kinematics,
    joint_map: JointMap,
    max_step_deg: float = 6.0,
    workspace: tuple[list[float], list[float]] | None = ([-0.45, -0.50, -0.02], [0.45, 0.25, 0.55]),
    keep_out: list[dict] | None = None,
) -> Gate:
    """Limits for an SO-101, in the robot's own units (degrees; gripper 0 to 100).

    max_step_deg: the most any joint may move in one control step.
    workspace: box (metres, in the arm's base frame) every tracked point must stay inside.
    keep_out: spheres the arm must not enter: [{"name", "center": [x, y, z], "radius"}].
    The joint range comes from the kinematic model, converted with joint_map.
    """
    a, b = joint_map.from_model(kin.joint_low), joint_map.from_model(kin.joint_high)
    low, high = np.append(np.minimum(a, b), 0.0), np.append(np.maximum(a, b), 100.0)
    constraints: list[dict] = [
        # five arm joints in degrees; the gripper (0 to 100) may open or close fully in one step
        {"type": "max_joint_step", "name": "joint_speed", "max_abs": [max_step_deg] * 5 + [100.0]},
        {"type": "joint_limits", "name": "joint_range", "low": low.tolist(), "high": high.tolist()},
    ]
    if workspace:
        constraints.append({"type": "cartesian_box", "name": "workspace", "low": list(workspace[0]), "high": list(workspace[1])})
    for z in keep_out or []:
        constraints.append({"type": "cartesian_keep_out_sphere", "name": z.get("name", "keep_out"), "center": list(z["center"]), "radius": float(z["radius"])})
    dims = list(range(len(SO101_MOTORS)))
    return Gate({"position_dims": dims, "delta_dims": dims, "constraints": constraints}, fk=lambda q: kin.points(joint_map.to_model(q)))


class StepscopeGuard:
    """Runs a policy on a LeRobot robot with stepscope tracing and gating in the loop.

    robot: anything with LeRobot's `get_observation()` and `send_action(dict)`.
    policy: observation dict -> target joint positions (see module docstring).
    tracer: a stepscope Tracer. Its `shadow` flag decides whether the gate acts.
    """

    def __init__(
        self,
        robot: Any,
        policy: Callable[[dict], Any],
        tracer: Tracer,
        kinematics: Kinematics | None = None,
        joint_map: JointMap | None = None,
        motors: list[str] = SO101_MOTORS,
        camera_key: str | None = None,
        frame_stride: int = 4,
    ):
        self.robot, self.policy, self.tracer = robot, policy, tracer
        self.kin, self.joint_map = kinematics, joint_map or JointMap()
        self.motors, self.camera_key, self.frame_stride = motors, camera_key, frame_stride
        self.keys = [f"{m}.pos" for m in motors]

    def _send(self, q: np.ndarray) -> None:
        self.robot.send_action({k: float(v) for k, v in zip(self.keys, q)})

    def _hold(self, q: np.ndarray) -> None:
        """Stop where we are: the robot's own hold if it has one, else re-command the current position."""
        if getattr(self.robot, "supports_command_hold", False):
            self.robot.hold()
        else:
            self._send(q)

    def run_episode(
        self,
        task: str = "",
        max_steps: int = 600,
        hz: float = 30.0,
        should_stop: Callable[[dict], bool] | None = None,
        outcome: Callable[[], bool | None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict:
        """Run one episode. Returns the episode summary plus loop timing.

        should_stop(observation): optional, return True to end the episode early.
        outcome(): optional, called at the end; return True/False for success, or None if unknown.
        """
        overhead, verdicts = [], {"clamp": 0, "block": 0, "halt": 0}
        reason, period, q = "finished", 1.0 / hz, None
        with self.tracer.episode(task=task) as ep:
            for _ in range(max_steps):
                tick = time.perf_counter()
                obs = self.robot.get_observation()
                q = np.array([obs[k] for k in self.keys], dtype=float)
                if should_stop and should_stop(obs):
                    reason = "stopped"
                    break
                out = self.policy(obs)
                t0 = time.perf_counter()  # everything from here to send_action is stepscope's own cost
                out = out if isinstance(out, dict) else {"action": out}
                target = np.asarray(out["action"], dtype=float)
                samples = out.get("samples")
                if target.ndim == 2:  # several sampled actions: the first is the one to execute
                    samples, target = target, target[0]
                internals = dict(out.get("internals") or {})
                if self.kin is not None:
                    internals["arm"] = self.kin.points(self.joint_map.to_model(q)).astype(np.float32)
                frame = obs.get(self.camera_key) if self.camera_key else None
                decision = ep.step(
                    state=q,
                    action=target - q,  # traced as a move from the current position
                    action_samples=None if samples is None else np.asarray(samples, dtype=float) - q,
                    frame=None if frame is None else np.asarray(frame)[:: self.frame_stride, :: self.frame_stride],
                    belief=out.get("belief"),
                    signals=out.get("signals"),
                    internals=internals,
                )
                if decision.verdict in verdicts:
                    verdicts[decision.verdict] += 1
                overhead.append(time.perf_counter() - t0)
                if decision.halt:
                    self._hold(q)
                    reason = "halted_by_gate"
                    break
                self._send(q + decision.action)
                sleep(max(0.0, period - (time.perf_counter() - tick)))
            success = outcome() if outcome else None
            summary = ep.end(success=success, reason=reason)
        ms = np.array(overhead) * 1000 if overhead else np.zeros(1)
        summary["guard"] = {
            "shadow": self.tracer.shadow,
            "gate_verdicts": verdicts,
            "overhead_ms_mean": float(ms.mean()),
            "overhead_ms_p99": float(np.percentile(ms, 99)),
        }
        return summary


class SimSO101:
    """A stand-in with the same interface as LeRobot's SO follower, for trying the guard without hardware.

    Purely kinematic: each call to get_observation() moves every joint toward
    its last commanded target by at most `max_speed` (degrees per tick). There
    is no physics, load or backlash, so it checks the plumbing and the gate,
    not how a real arm behaves.
    """

    supports_command_hold = False

    def __init__(self, start=None, max_speed: float = 8.0):
        self.q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0] if start is None else start, dtype=float)
        self.target = self.q.copy()
        self.max_speed = max_speed
        self.sent: list[np.ndarray] = []
        self.is_connected = True

    def get_observation(self) -> dict:
        self.q = self.q + np.clip(self.target - self.q, -self.max_speed, self.max_speed)
        return {f"{m}.pos": float(v) for m, v in zip(SO101_MOTORS, self.q)}

    def send_action(self, action: dict) -> dict:
        self.target = np.array([action[f"{m}.pos"] for m in SO101_MOTORS], dtype=float)
        self.sent.append(self.target.copy())
        return action

    def hold(self) -> None:
        raise NotImplementedError("SimSO101 does not implement command hold")
