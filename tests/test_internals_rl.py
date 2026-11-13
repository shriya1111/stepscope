import numpy as np
import pandas as pd
import pytest

from stepscope import Predictor, TraceConfig, Tracer
from stepscope.explain import SituationIndex, describe_decision, plan_modes
from stepscope.rl import _label_steps, export_dataset, export_situations, load_situations, reset_to_situation_wrapper
from stepscope.server import episode_payload, run_payload
from stepscope.store import forward_fill, list_episodes, read_internals


def test_plan_modes_separates_two_options():
    line = np.linspace(0, 100, 8)
    left = np.stack([line, np.zeros(8)], axis=1)
    right = np.stack([line, np.full(8, 200.0)], axis=1)
    plans = np.stack([left, left + 3, right, left - 2, right + 4, left + 1])
    assert plan_modes(plans, threshold=30).tolist() == [0, 0, 1, 0, 1, 0]
    assert plan_modes(plans[[0, 1, 3]], threshold=30).tolist() == [0, 0, 0]
    assert "Torn between 2 options (4 vs 2 of 6" in describe_decision(6, [4, 2], 12.0, None)


def test_forward_fill():
    assert forward_fill([0, 3], np.array([[1.0], [2.0]]), 5).ravel().tolist() == [1, 1, 1, 2, 2]


def make_run(root, n_eps=4, with_predictor=False):
    """Tiny synthetic run: 1-D robot; failing episodes carry a distinct embedding."""
    predictor = None
    if with_predictor:
        predictor = Predictor(features=["elapsed"])
        predictor.coef, predictor.mean, predictor.std = np.array([8.0]), np.array([0.5]), np.array([0.25])
        predictor.threshold, predictor.sustain = 0.9, 2
    tracer = Tracer(root, config=TraceConfig(max_steps=20), predictor=predictor)
    for e in range(n_eps):
        fail = e % 2 == 1
        with tracer.episode(task="t", meta={"seed": e}) as ep:
            reward = None
            for t in range(20):
                internals = None
                if t % 4 == 0:
                    internals = {
                        "embedding": np.full(6, 5.0 if fail else 0.0) + 0.01 * t,
                        "plans": np.zeros((3, 4, 2)) + t,
                        "modes": np.array([0, 0, 1]),
                        "denoise_spread": np.array([9.0, 3.0, 1.0]),
                    }
                ep.step(state=[float(t)], action=[1.0], reward=reward, internals=internals, snapshot=np.array([t, e], float))
                reward = t / 20
            ep.end(success=not fail, reason="success" if not fail else "timeout", final_reward=1.0)
    return root


def test_internals_round_trip_and_decision_payload(tmp_path):
    run = make_run(tmp_path / "run")
    z = read_internals(run, "ep_00001")
    assert z["embedding"].shape == (5, 6) and z["embedding__t"].tolist() == [0, 4, 8, 12, 16]
    assert z["snapshot"].shape == (20, 2)
    assert list_episodes(run)[0]["internals"] == ["denoise_spread", "embedding", "modes", "plans", "snapshot"]

    index = SituationIndex(run)
    assert len(index.points()) == 4 * 5
    # look-alikes of a failing episode's moments come from the other failing episode
    assert all(nb["ep"] == "ep_00003" and nb["success"] is False for nb in index.neighbours("ep_00001", k=3)[8])

    payload = episode_payload(run / "episodes" / "ep_00001")
    d = payload["decisions"]["decisions"][2]
    assert d["t"] == 8 and d["modes"] == [0, 0, 1] and len(d["plans"]) == 3 and d["map"] is not None
    assert "Torn between 2 options" in d["summary"] and "failed" in d["summary"]
    assert len(run_payload(run)["map"]) == 20
    assert payload["steps"]["reward"][1] == 0.0


def test_labels():
    assert _label_steps(4, True, None) == ["normal"] * 4
    assert _label_steps(3, False, None) == ["failed_no_alert"] * 3
    assert _label_steps(4, True, 2) == ["normal", "normal", "recovery", "recovery"]
    lab = _label_steps(30, False, 20)
    assert lab[3] == "failed_no_alert" and lab[4] == "pre_alert" and lab[19] == "pre_alert" and lab[20] == "post_alert_failed"


def test_dataset_export(tmp_path):
    run = make_run(tmp_path / "run", with_predictor=True)
    info = export_dataset(run, tmp_path / "d.parquet")
    df = pd.read_parquet(tmp_path / "d.parquet")
    assert info["rows"] == len(df) == 80 and df["done"].sum() == 4
    first = df[df.episode_id == "ep_00000"]
    assert first["reward"].tolist()[:2] == [0.0, 0.05] and first["reward"].iloc[-1] == 1.0  # reward follows its action
    assert first["next_state"].iloc[0][0] == 1.0
    failed = df[df.episode_id == "ep_00001"]
    alert = int(failed["alerted"].to_numpy().argmax())
    assert set(failed["label"]) == {"pre_alert", "post_alert_failed"}
    assert (failed[failed.t >= alert]["weight"] == 0).all() and (failed[failed.t < alert]["weight"] == 3).all()
    assert set(first["label"]) == {"normal", "recovery"}
    assert list(failed["snapshot"].iloc[5]) == [5.0, 1.0]


def test_situations_and_reset_wrapper(tmp_path):
    gym = pytest.importorskip("gymnasium")
    run = make_run(tmp_path / "run", with_predictor=True)
    sits = export_situations(run, tmp_path / "s.json", lead=3)
    assert [s["episode_id"] for s in sits] == ["ep_00001", "ep_00003"]
    alert = list_episodes(run)[1]["first_alert_step"]
    assert sits[0]["step"] == alert - 3 and sits[0]["state"] == [alert - 3, 1.0] and sits[0]["anchor"] == "alert"

    class Env(gym.Env):
        observation_space = gym.spaces.Box(-1e3, 1e3, (2,))
        action_space = gym.spaces.Box(-1, 1, (1,))

        def reset(self, *, seed=None, options=None):
            self.state = np.array((options or {}).get("reset_to_state", [0.0, 0.0]), float)
            return self.state.copy(), {}

        def step(self, action):
            self.state[0] += 1
            return self.state.copy(), 0.0, False, False, {}

    env = reset_to_situation_wrapper()(Env(), load_situations(tmp_path / "s.json"), shuffle=False)
    obs, info = env.reset()
    assert obs.tolist() == sits[0]["state"] and info["situation"] == sits[0]["id"]
    obs, info = env.reset()
    assert obs.tolist() == sits[1]["state"]
    assert env.step(np.zeros(1))[0][0] == sits[1]["state"][0] + 1


def test_embedding_novelty_used_live(tmp_path):
    run = make_run(tmp_path / "train", n_eps=8)
    from stepscope.evaluate import load_episodes, train

    model, _ = train(run, novelty_source="embedding")
    assert model.novelty_source == "embedding" and model.novelty_mean.shape == (6,)
    eps = load_episodes(run, "embedding")
    assert eps[0]["states"].shape == (20, 6)
    risk_fail = model.score(eps[1]["metrics"], eps[1]["states"]).mean()
    risk_ok = model.score(eps[0]["metrics"], eps[0]["states"]).mean()
    assert risk_fail > risk_ok
    model.save(tmp_path / "m.json")
    assert Predictor.load(tmp_path / "m.json").novelty_source == "embedding"
