import numpy as np
import pytest

from stepscope import Gate, Predictor, TraceConfig, Tracer
from stepscope.metrics import OnlineMetrics
from stepscope.predictor import auroc, first_alert
from stepscope.store import list_episodes, read_steps


def run_metrics(actions, positions, samples=None, plan=None):
    m = OnlineMetrics(TraceConfig(window=4, max_steps=len(actions)))
    out = []
    for i, (a, p) in enumerate(zip(actions, positions)):
        out.append(m.update(np.array(p, float), np.array(a, float), None if samples is None else samples[i], None if plan is None else plan[i]))
    return out


def test_straight_motion_is_efficient_and_has_no_reversals():
    out = run_metrics([[1, 0]] * 6, [[i, 0] for i in range(6)])
    assert out[-1]["reversal_rate"] == 0
    assert out[-1]["path_efficiency"] == pytest.approx(1.0)
    assert out[-1]["net_speed"] == pytest.approx(1.0)


def test_zigzag_shows_up_as_reversals_and_poor_efficiency():
    out = run_metrics([[1, 0], [-1, 0]] * 4, [[i % 2, 0] for i in range(8)])
    assert out[-1]["reversal_rate"] == 1.0
    assert out[-1]["path_efficiency"] < 0.3


def test_sample_disagreement():
    agree = np.tile([[1.0, 0.0]], (6, 1))
    split = np.array([[1.0, 0.0], [-1.0, 0.0]] * 3)
    a = run_metrics([[1, 0]], [[0, 0]], samples=[agree])[0]
    b = run_metrics([[1, 0]], [[0, 0]], samples=[split])[0]
    assert a["direction_disagreement"] == pytest.approx(0, abs=1e-9) and a["sample_spread"] == 0
    assert b["direction_disagreement"] == pytest.approx(1.0) and b["sample_spread"] > 0.4


def test_plan_regressions_and_dwell():
    out = run_metrics([[0, 0]] * 6, [[0, 0]] * 6, plan=[0, 1, 1, 1, 0, 1])
    assert [o["plan_regressions"] for o in out] == [0, 0, 0, 0, 1, 1]
    assert out[3]["subgoal_dwell"] == 2


GATE = {
    "position_dims": [0, 1],
    "delta_dims": [0, 1],
    "constraints": [
        {"type": "max_step", "max_norm": 0.1},
        {"type": "workspace_box", "low": [0, 0], "high": [1, 1]},
        {"type": "keep_out_box", "low": [0.4, 0.4], "high": [0.6, 0.6]},
        {"type": "risk_limit", "max_risk": 0.8, "sustain": 2},
    ],
}


def test_gate_allows_safe_action_unchanged():
    d = Gate(GATE).check(np.array([0.1, 0.1]), np.array([0.05, 0.0, 1.0]))
    assert d.verdict == "allow" and np.allclose(d.action, [0.05, 0, 1])


def test_gate_clamps_speed_and_workspace():
    d = Gate(GATE).check(np.array([0.1, 0.1]), np.array([0.5, 0.0, 1.0]))
    assert d.verdict == "clamp" and np.linalg.norm(d.action[:2]) == pytest.approx(0.1)
    assert d.action[2] == 1.0  # non-motion dims untouched
    d = Gate(GATE).check(np.array([0.98, 0.5]), np.array([0.05, 0.0, 0.0]))
    assert d.verdict == "clamp" and d.action[0] == pytest.approx(0.02)


def test_gate_never_lets_position_enter_keep_out():
    rng = np.random.default_rng(0)
    gate, pos = Gate(GATE), np.array([0.1, 0.5])
    hit = 0
    for _ in range(2000):
        d = gate.check(pos, np.append(np.array([0.5, 0.5]) - pos + rng.normal(0, 0.2, 2), 0))
        hit += d.intervened
        pos = pos + d.action[:2]
        assert not (0.4 < pos[0] < 0.6 and 0.4 < pos[1] < 0.6)
        assert np.all(pos >= 0) and np.all(pos <= 1)
    assert hit > 100


def test_gate_halts_only_on_sustained_risk():
    gate = Gate(GATE)
    s, a = np.array([0.1, 0.1]), np.array([0.01, 0, 0])
    assert gate.check(s, a, risk=0.9).verdict == "allow"
    assert gate.check(s, a, risk=0.5).verdict == "allow"
    assert gate.check(s, a, risk=0.9).verdict == "allow"
    d = gate.check(s, a, risk=0.95)
    assert d.halt and np.allclose(d.action[:2], 0)


def test_first_alert_and_auroc():
    assert first_alert(np.array([0.9, 0.1, 0.9, 0.9, 0.9]), 0.8, 3) == 4
    assert first_alert(np.array([0.9, 0.1, 0.9]), 0.8, 3) is None
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.5


def synthetic_episode(rng, fail):
    T = 40
    return {
        "success": not fail,
        "states": rng.normal(0, 1, (T, 3)),
        "metrics": {"reversal_rate": rng.normal(0.6 if fail else 0.1, 0.1, T), "elapsed": np.linspace(0, 1, T)},
    }


def test_predictor_learns_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    eps = [synthetic_episode(rng, i % 3 == 0) for i in range(90)]
    p = Predictor().fit(eps)
    bad, good = synthetic_episode(rng, True), synthetic_episode(rng, False)
    assert p.score(bad["metrics"], bad["states"]).mean() > 0.9
    assert p.score(good["metrics"], good["states"]).mean() < 0.1
    assert p.explain(bad["metrics"], bad["states"], 20)[0]["label"] == "oscillation"
    p.save(tmp_path / "m.json")
    q = Predictor.load(tmp_path / "m.json")
    assert np.allclose(q.score(bad["metrics"], bad["states"]), p.score(bad["metrics"], bad["states"]))
    assert q.threshold == p.threshold


def test_tracer_writes_readable_episode(tmp_path):
    tracer = Tracer(tmp_path / "run", config=TraceConfig(position_dims=[0, 1], max_steps=5), gate=Gate(GATE))
    with tracer.episode(task="demo") as ep:
        for t in range(5):
            d = ep.step(state=[0.1 * t, 0.1, 7.0], action=[0.5, 0, 0], frame=np.full((4, 4, 3), t, np.uint8), belief={"subtask": "go"}, plan_step=0)
            assert d.verdict == "clamp"
        ep.end(success=True, reason="success")
    with tracer.episode() as ep:  # abandoned episodes are still saved
        ep.step(state=[0, 0, 0], action=[0, 0, 0])
    metas = list_episodes(tmp_path / "run")
    assert [m["success"] for m in metas] == [True, None]
    assert metas[0]["gate"]["clamp"] == 5 and metas[0]["frames"]["count"] == 5
    df = read_steps(tmp_path / "run", metas[0]["episode_id"])
    assert len(df) == 5 and np.linalg.norm(df["executed"][0][:2]) == pytest.approx(0.1)
    png = (tmp_path / "run" / "episodes" / metas[0]["episode_id"] / "frames.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
