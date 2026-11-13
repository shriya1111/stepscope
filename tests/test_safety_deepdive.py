import numpy as np
import pytest

from stepscope import TraceConfig, Tracer
from stepscope.deepdive import formed_at, occlusion_batch, plan_change, read_deepdive, turn_degrees, write_deepdive
from stepscope.safety import Hazard, Scene, annotate_run, assess, choose_plan, path_clearance, risk_grid
from stepscope.server import episode_payload, run_payload

HAND = Hazard("hand", [100.0, 0.0], 10.0, severity=3, kind="person")
CUP = Hazard("cup", [0.0, 100.0], 10.0, severity=1, kind="fragile")
SCENE = Scene(hazards=[HAND, CUP], extent=200, robot_radius=5, speed_ref=10, proximity_scale=20)


def test_path_clearance_checks_segments_not_just_points():
    # the two points straddle the hazard: the segment between them passes through it
    clear, hit = path_clearance(np.array([[100.0, -50.0], [100.0, 50.0]]), HAND)
    assert clear == pytest.approx(-10) and hit == 0
    clear, hit = path_clearance(np.array([[0.0, 0.0], [50.0, 0.0]]), HAND, robot_radius=5)
    assert clear == pytest.approx(35) and hit is None


def straight(to, n=5):
    return np.linspace([0, 0], to, n + 1)[1:]


def test_assess_likelihood_times_consequence():
    away = np.stack([straight([-80, -80])] * 4)
    at_hand = np.stack([straight([100, 0])] * 4)
    at_cup = np.stack([straight([0, 100])] * 4)
    safe = assess([0, 0], away, 10, SCENE)
    assert safe["level"] == "low" and safe["time_to_contact"] is None and safe["clearance"] > 80
    hand = assess([0, 0], at_hand, 10, SCENE)
    assert hand["hazard"] == "hand" and hand["p_contact"] == 1 and hand["risk"] == 1 and hand["level"] == "high"
    assert hand["time_to_contact"] == 4 and hand["clearance"] < 0
    # same geometry, minor hazard: much lower risk
    cup = assess([0, 0], at_cup, 10, SCENE)
    assert cup["hazard"] == "cup" and cup["risk"] == pytest.approx(0.25)
    # same hazard, slow approach: consequence scales with speed squared
    assert assess([0, 0], at_hand, 5, SCENE)["risk"] == pytest.approx(0.25)
    # only half the sampled plans head for the hand
    mixed = np.concatenate([away[:2], at_hand[:2]])
    assert assess([0, 0], mixed, 10, SCENE)["p_contact"] == pytest.approx(0.5)


def test_risk_grid_peaks_at_the_serious_hazard():
    g = np.array(risk_grid(SCENE, n=20))
    iy, ix = np.unravel_index(g.argmax(), g.shape)
    assert ix >= 9 and iy == 0  # the hand is at x=100 (mid-right), y=0 (top row)
    assert g.max() == pytest.approx(1.0) and g.min() < 0.05


def test_choose_plan_prefers_safe_then_progress():
    f = lambda gain, contact, risk=0.0: {"gain": gain, "contact": contact, "risk": risk}
    assert choose_plan([f(0.5, True, 1.0), f(0.1, False), f(0.3, False)]) == 2
    assert choose_plan([f(0.2, False), f(0.2, False)]) == 0  # ties keep the model's own first choice
    assert choose_plan([f(0.5, True, 0.9), f(0.1, True, 0.4)]) == 1  # nothing safe: least risky


def test_annotate_run_and_payloads(tmp_path):
    tracer = Tracer(tmp_path / "run", config=TraceConfig(position_dims=[0, 1]))
    for target in ([100.0, 0.0], [-100.0, -100.0]):  # one episode drives into the hand, one away
        with tracer.episode() as ep:
            pos = np.zeros(2)
            for t in range(12):
                step = np.array(target) / 12
                plans = np.stack([pos + step * np.arange(0, 5)[:, None]] * 3)  # index 0 = current position
                ep.step(state=pos, action=step, internals={"plans": plans} if t % 4 == 0 else None)
                pos = pos + step
            ep.end(success=True)
    totals = annotate_run(tmp_path / "run", SCENE)
    assert totals["episodes"] == 2 and totals["steps"] == 24 and totals["contacts"] > 0
    run = run_payload(tmp_path / "run")
    assert run["scene"]["hazards"][0]["name"] == "hand" and len(run["scene"]["risk_grid"]) == 32
    into, away = run["episodes"]
    assert into["safety"]["peak_level"] == "high" and into["safety"]["contact_steps"] > 0
    assert away["safety"]["peak_level"] == "low" and away["safety"]["contact_steps"] == 0
    s = episode_payload(tmp_path / "run" / "episodes" / "ep_00000")["safety"]
    assert len(s["risk"]) == 12 and s["risk"][-1] > s["risk"][0] and s["inside"][-1] is True


def test_deepdive_helpers(tmp_path):
    frames = np.zeros((2, 12, 12, 3), np.uint8)
    occ = occlusion_batch(frames, grid=3)
    assert occ.shape == (18, 2, 12, 12, 3)
    assert occ[4, 0, 4:8, 4:8].min() == 255 and occ[4, 0].sum() == 255 * 16 * 3 and occ[4, 1].sum() == 0
    assert occ[9 + 8, 1, 8:, 8:].min() == 255 and occ[9 + 8, 0].sum() == 0
    base = np.zeros((4, 2))
    assert plan_change(np.stack([base, base + [3, 4]]), base).tolist() == [0, 5]
    stages = np.stack([base + 50, base + 20, base + 4, base])
    assert formed_at(stages, tolerance=10) == 2
    assert turn_degrees(np.array([[0, 1.0]]), np.array([[1.0, 0]]), np.zeros(2)) == pytest.approx(90)
    write_deepdive(tmp_path, {8: {"formed_at": 1}})
    write_deepdive(tmp_path, {16: {"formed_at": 2}})
    assert read_deepdive(tmp_path) == {"8": {"formed_at": 1}, "16": {"formed_at": 2}}
