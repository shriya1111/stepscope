"""3D pieces: kinematics, whole-arm safety, joint/Cartesian gate, and the SO-101 guard."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from stepscope import Gate, Predictor, TraceConfig, Tracer  # noqa: E402
from stepscope.adapters.lerobot_robot import SO101_MOTORS, JointMap, SimSO101, StepscopeGuard, so101_gate, so101_kinematics  # noqa: E402
from stepscope.kinematics import project_to_camera  # noqa: E402
from stepscope.safety import Hazard, Scene, assess  # noqa: E402
from stepscope.store import list_episodes, read_internals, read_steps  # noqa: E402

KIN = so101_kinematics()
JM = JointMap()


def arm(q_deg):
    return KIN.points(JM.to_model(q_deg))


def test_kinematics_link_lengths_do_not_change_with_pose():
    a, b = arm([0, 0, 0, 0, 0]), arm([40, -70, 80, -30, 20])
    upper = lambda p: np.linalg.norm(p[2] - p[1])  # Upper_Arm -> Lower_Arm joint
    fore = lambda p: np.linalg.norm(p[3] - p[2])
    assert upper(a) == pytest.approx(upper(b), abs=1e-9) and upper(a) == pytest.approx(0.116, abs=0.002)
    assert fore(a) == pytest.approx(fore(b), abs=1e-9) and fore(a) == pytest.approx(0.135, abs=0.002)
    assert np.linalg.norm(a[-1] - b[-1]) > 0.1  # but the gripper does move


def test_base_rotation_swings_the_gripper_about_the_vertical_axis():
    a, b = arm([0, -60, 60, 0, 0])[-1], arm([90, -60, 60, 0, 0])[-1]
    assert a[2] == pytest.approx(b[2], abs=1e-4)  # same height
    assert np.hypot(a[0], a[1] + 0.0452) == pytest.approx(np.hypot(b[0], b[1] + 0.0452), abs=1e-3)  # same distance from the base axis


def test_joint_map_round_trip_and_offsets():
    jm = JointMap(sign=[1, -1, 1, 1, 1], zero_deg=[0, 90, 0, 0, 0])
    q = np.array([10.0, 30.0, 45.0, -20.0, 5.0])
    assert np.allclose(jm.from_model(jm.to_model(q)), q)
    assert jm.to_model(q)[1] == pytest.approx(np.radians(60))


def test_camera_projection_centre_and_direction():
    cam, eye = [0.0, 0.0, 1.0], np.eye(3)  # looking straight down from 1 m
    px = project_to_camera(np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]), cam, eye, 90, 200, 100)
    assert px[0].tolist() == [100, 50]
    assert px[1][0] > 100 and px[1][1] == 50  # +x in the world is to the right in the image
    assert px[2][1] < 50  # +y in the world is up in the image


def test_safety_scores_the_whole_arm_in_3d():
    start, end = [0, -170, 170, 0, 0], [0, -60, 10, 0, 0]
    path = KIN.path([JM.to_model(np.array(start) + (np.array(end) - start) * f) for f in np.linspace(0, 1, 12)])
    # A hazard sitting where the elbow passes mid-move; the gripper itself never comes near it.
    scene = Scene(hazards=[Hazard("hand", path[6, 2].tolist(), 0.03, severity=3)], robot_radius=0.01, speed_ref=0.01, proximity_scale=0.03)
    whole = assess(path[0], path[None, 1:], 0.01, scene)
    tip_only = assess(path[0, -1], path[None, 1:, -1], 0.01, scene)
    assert whole["time_to_contact"] is not None and whole["level"] == "high"
    assert whole["clearance"] < tip_only["clearance"]


def test_gate_joint_step_and_limits():
    gate = so101_gate(KIN, JM, max_step_deg=5, workspace=None)
    q = np.array([0.0, -60, 60, 0, 0, 50])
    d = gate.check(q, np.array([20.0, 0, 0, 0, 0, 0]))
    assert d.verdict == "clamp" and np.abs(d.action).max() == pytest.approx(5)
    low = np.degrees(KIN.joint_low[1])
    d = gate.check(np.array([0.0, low + 1, 60, 0, 0, 50]), np.array([0.0, -4, 0, 0, 0, 0]))
    assert d.verdict == "clamp" and d.action[1] == pytest.approx(-1)
    assert gate.check(q, np.array([1.0, 1, 1, 0, 0, 0])).verdict == "allow"


def test_cartesian_constraints_need_kinematics():
    with pytest.raises(ValueError, match="kinematics"):
        Gate({"position_dims": [0], "delta_dims": [0], "constraints": [{"type": "cartesian_box", "low": [0, 0, 0], "high": [1, 1, 1]}]})


START, GOAL = np.array([-50.0, -60, 60, 0, 0, 50]), np.array([50.0, -60, 60, 0, 0, 50])
ZONE = {"name": "operator side", "center": arm([0, -60, 60, 0, 0])[-1].tolist(), "radius": 0.07}  # right on the gripper's path


def reach(obs):
    """Scripted stand-in for a policy: head for GOAL, a few degrees per step."""
    q = np.array([obs[f"{m}.pos"] for m in SO101_MOTORS])
    return q + np.clip(GOAL - q, -4, 4)


def run_guard(tmp_path, shadow, predictor=None, steps=60):
    robot = SimSO101(start=START)
    tracer = Tracer(tmp_path / "run", config=TraceConfig(max_steps=steps), gate=so101_gate(KIN, JM, keep_out=[ZONE]), predictor=predictor, shadow=shadow)
    guard = StepscopeGuard(robot, reach, tracer, kinematics=KIN, joint_map=JM)
    summary = guard.run_episode(task="reach across", max_steps=steps, hz=1000, sleep=lambda s: None)
    meta = list_episodes(tmp_path / "run")[0]
    arm_path = read_internals(tmp_path / "run", meta["episode_id"])["arm"]
    inside = np.linalg.norm(arm_path - np.array(ZONE["center"]), axis=-1).min(axis=1) < ZONE["radius"]
    return robot, summary, read_steps(tmp_path / "run", meta["episode_id"]), inside


def test_shadow_mode_records_but_never_changes_the_action(tmp_path):
    robot, summary, df, inside = run_guard(tmp_path, shadow=True)
    assert np.allclose(robot.q, GOAL)  # the robot went exactly where the policy sent it
    assert inside.any()  # ...straight through the zone
    flagged = df[df["verdict"] != "allow"]
    assert len(flagged) > 0 and summary["guard"]["shadow"] is True
    assert all(np.allclose(a, e) for a, e in zip(df["action"], df["executed"]))
    assert "operator side" in flagged["violations"].iloc[0]
    assert summary["reason"] == "finished"


def test_enforce_mode_keeps_the_arm_out_of_the_zone(tmp_path):
    robot, summary, df, inside = run_guard(tmp_path, shadow=False)
    assert not inside.any()
    assert summary["guard"]["gate_verdicts"]["clamp"] + summary["guard"]["gate_verdicts"]["block"] > 0
    assert robot.q[0] < GOAL[0] - 5  # held short of the goal, on the near side of the zone
    assert summary["guard"]["overhead_ms_mean"] < 20


def test_halt_holds_the_arm(tmp_path):
    always_risky = Predictor(features=["elapsed"])
    always_risky.intercept = 10.0
    robot = SimSO101(start=START)
    gate = so101_gate(KIN, JM)
    gate.constraints.append({"type": "risk_limit", "name": "failure_risk", "max_risk": 0.9, "sustain": 3})
    tracer = Tracer(tmp_path / "run", config=TraceConfig(max_steps=60), gate=gate, predictor=always_risky)
    summary = StepscopeGuard(robot, reach, tracer, kinematics=KIN, joint_map=JM).run_episode(max_steps=60, hz=1000, sleep=lambda s: None)
    assert summary["reason"] == "halted_by_gate" and summary["n_steps"] == 3
    held_at = robot.sent[-1]
    for _ in range(5):
        robot.get_observation()
    assert np.allclose(robot.q, held_at) and robot.q[0] < 0  # stopped where it was, far from the goal
