"""End to end on the built-in simulator: collect, train, score live, gate, serve."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stepscope.cli import _collect
from stepscope.evaluate import compare_gated, evaluate, train
from stepscope.server import create_app
from stepscope.sim import GATE_CONFIG, TRACE_CONFIG, NoisyScriptedPolicy, TabletopEnv, rollout
from stepscope.sim.tabletop import KEEP_OUT_HIGH, KEEP_OUT_LOW
from stepscope.store import list_episodes, read_steps
from stepscope import Gate, Tracer


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    _collect(root / "train", range(150))
    model, baseline = train(root / "train")
    seeds = range(5000, 5080)
    _collect(root / "eval", seeds, predictor=model)
    report = evaluate(root / "eval", model, baseline)
    _collect(root / "gated", seeds, predictor=model, halt_on_risk=True)
    return root, model, report


def test_simulator_is_deterministic(tmp_path):
    def run(d):
        tracer = Tracer(d, config=TRACE_CONFIG, gate=Gate(GATE_CONFIG))
        return rollout(TabletopEnv(), NoisyScriptedPolicy(), tracer, seed=3)

    a, b = run(tmp_path / "a"), run(tmp_path / "b")
    assert a["n_steps"] == b["n_steps"] and a["reason"] == b["reason"]


def test_clean_policy_succeeds(tmp_path):
    tracer = Tracer(tmp_path / "r", config=TRACE_CONFIG, gate=Gate(GATE_CONFIG))
    env, policy = TabletopEnv(difficulty=0.0), NoisyScriptedPolicy(difficulty=0.0)
    results = [rollout(env, policy, tracer, s) for s in range(15)]
    assert all(r["success"] for r in results)


def test_executed_path_never_enters_keep_out_zone(demo):
    root, _, _ = demo
    interventions = 0
    for meta in list_episodes(root / "train"):
        df = read_steps(root / "train", meta["episode_id"])
        pos = np.stack(df["state"].to_numpy())[:, :3]
        inside = np.all(pos > KEEP_OUT_LOW, axis=1) & np.all(pos < KEEP_OUT_HIGH, axis=1)
        assert not inside.any()
        interventions += meta["gate"]["clamp"] + meta["gate"]["block"]
    assert interventions > 0


def test_predictor_beats_chance_and_warns_earlier_than_the_clock(demo):
    _, _, report = demo
    m, b = report["model"], report["time_only_baseline"]
    assert m["recall"] >= 0.5
    assert m["false_alarm_rate"] <= 0.2
    assert m["median_lead_steps"] > b["median_lead_steps"]
    assert report["auroc_at_step"]["20"]["auroc"] > 0.7


def test_live_risk_matches_offline_scoring(demo):
    root, model, _ = demo
    from stepscope.evaluate import load_episodes

    for e in load_episodes(root / "eval")[:10]:
        assert np.allclose(e["risk"], model.score(e["metrics"], e["states"]), atol=1e-9)


def test_risk_halt_stops_failures_early(demo):
    root, _, _ = demo
    g = compare_gated(root / "eval", root / "gated")
    assert g["failures_halted_early"] > 0
    assert g["robot_steps_spent_on_failures"]["with_halt"] < g["robot_steps_spent_on_failures"]["without_halt"]


def test_viewer_api(demo):
    root, _, _ = demo
    client = TestClient(create_app(root))
    runs = client.get("/api/runs").json()
    assert {r["name"] for r in runs} == {"train", "eval", "gated"}
    detail = client.get("/api/runs/eval").json()
    assert detail["report"]["episodes"] == 80 and len(detail["episodes"]) == 80
    ep = detail["episodes"][0]["episode_id"]
    data = client.get(f"/api/runs/eval/episodes/{ep}").json()
    assert len(data["steps"]["risk"]) == data["meta"]["n_steps"]
    assert client.get(f"/api/runs/eval/episodes/{ep}/frames.png").content[:4] == b"\x89PNG"
    assert "stepscope" in client.get("/").text
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/eval/episodes/..%2F..%2Fx").status_code == 404


def test_gym_adapter(tmp_path):
    gym = pytest.importorskip("gymnasium")
    from stepscope.adapters.gym import trace_rollout

    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(0)
    tracer = Tracer(tmp_path / "gym")
    summary = trace_rollout(env, lambda obs: rng.uniform(-2, 2, (6, 1)), tracer, seed=0, max_steps=30)
    assert summary["n_steps"] == 30
    df = read_steps(tmp_path / "gym", summary["episode_id"])
    assert df["m_sample_spread"].notna().all()
