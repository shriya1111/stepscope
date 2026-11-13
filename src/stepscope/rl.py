"""Turn traces into training material.

  * export_dataset: every step as a transition (state, action, reward, next
    state, done) with risk, labels and sampling weights attached.
  * export_situations: a library of saved environment states from just before
    things went wrong, to restart training or evaluation from.
  * ResetToSituation: a Gymnasium wrapper that resets into those states.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .store import dump_json, list_episodes, read_internals, read_run_meta, read_steps

PRE_ALERT_WINDOW = 16

# What each label means for a learner, and the default sampling weight.
LABELS = {
    "normal": ("step from an episode that succeeded without an alert", 1.0),
    "pre_alert": ("the run-up to an alert in a failed episode: the hard moments worth practising", 3.0),
    "post_alert_failed": ("after the alert in an episode that went on to fail: do not imitate", 0.0),
    "recovery": ("after an alert in an episode that still succeeded: how to get out of trouble", 2.0),
    "failed_no_alert": ("failed episode that never raised an alert: outcome known, cause not localised", 0.5),
}


def _label_steps(n: int, success: bool, alert: int | None) -> list[str]:
    if alert is None:
        return ["normal" if success else "failed_no_alert"] * n
    if success:
        return ["normal"] * alert + ["recovery"] * (n - alert)
    start = max(0, alert - PRE_ALERT_WINDOW)
    return ["failed_no_alert"] * start + ["pre_alert"] * (alert - start) + ["post_alert_failed"] * (n - alert)


def export_dataset(run_dir: str | Path, out: str | Path) -> dict:
    """Write one Parquet file of transitions for offline RL or filtered imitation.

    Columns: episode_id, t, state, action (what the policy asked for), executed
    (what the gate let through), reward, next_state, done, success, risk,
    gate_verdict, alerted, label, weight, frame_index, and snapshot if recorded.
    Rewards are the ones passed to ep.step(reward=...), aligned so that row t
    holds the reward that followed action t (the last one comes from ep.end(final_reward=...)).
    """
    run_dir = Path(run_dir)
    parts = []
    for meta in list_episodes(run_dir):
        if meta["success"] is None:
            continue
        df = read_steps(run_dir, meta["episode_id"])
        n = len(df)
        states = df["state"].tolist()
        reward = df["reward"].to_numpy(dtype=float) if "reward" in df else np.full(n, np.nan)
        part = pd.DataFrame(
            {
                "episode_id": meta["episode_id"],
                "t": np.arange(n),
                "state": states,
                "action": df["action"].tolist(),
                "executed": df["executed"].tolist(),
                "reward": np.append(reward[1:], np.nan if meta.get("final_reward") is None else meta["final_reward"]),
                "next_state": states[1:] + [states[-1]],
                "done": np.arange(n) == n - 1,
                "success": bool(meta["success"]),
                "risk": df["risk"].to_numpy(dtype=float),
                "gate_verdict": df["verdict"].tolist(),
                "frame_index": np.arange(n),
            }
        )
        alert = meta.get("first_alert_step")
        part["alerted"] = np.arange(n) >= alert if alert is not None else False
        part["label"] = _label_steps(n, bool(meta["success"]), alert)
        part["weight"] = part["label"].map(lambda k: LABELS[k][1])
        snap = read_internals(run_dir, meta["episode_id"]).get("snapshot")
        if snap is not None and len(snap) == n:
            part["snapshot"] = snap.tolist()
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out, index=False)
    return {"rows": len(data), "episodes": len(parts), "labels": data["label"].value_counts().to_dict()}


def export_situations(run_dir: str | Path, out: str | Path, lead: int = 8, include_successes: bool = False) -> list[dict]:
    """Save restart points: the environment state `lead` steps before each alert.

    Failed episodes with no alert use the step where risk peaked (or the
    midpoint if the run has no risk scores). Needs snapshots in the traces.
    """
    run_dir = Path(run_dir)
    situations = []
    for meta in list_episodes(run_dir):
        if meta["success"] is None or (meta["success"] and not include_successes):
            continue
        z = read_internals(run_dir, meta["episode_id"])
        if "snapshot" not in z:
            continue
        alert = meta.get("first_alert_step")
        if alert is not None:
            anchor, why = alert, "alert"
        else:
            risk = read_steps(run_dir, meta["episode_id"])["risk"].to_numpy(dtype=float)
            if np.all(np.isnan(risk)):
                anchor, why = meta["n_steps"] // 2, "midpoint"
            else:
                anchor, why = int(np.nanargmax(risk)), "peak_risk"
        step = max(0, anchor - lead)
        i = int(np.searchsorted(z["snapshot__t"], step, side="right") - 1)
        causes = meta.get("causes") or []
        situations.append(
            {
                "id": f"{run_dir.name}/{meta['episode_id']}@{step}",
                "episode_id": meta["episode_id"],
                "step": step,
                "anchor": why,
                "outcome": "success" if meta["success"] else meta.get("reason") or "failed",
                "symptom": causes[0]["label"] if causes else None,
                "state": z["snapshot"][max(i, 0)].tolist(),
            }
        )
    dump_json(Path(out), {"run": run_dir.name, "label": read_run_meta(run_dir).get("label"), "lead_steps": lead, "situations": situations})
    return situations


def load_situations(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())["situations"]


def reset_to_situation_wrapper():
    """Returns the ResetToSituation class (imports gymnasium only when used)."""
    import gymnasium as gym

    class ResetToSituation(gym.Wrapper):
        """Each reset() starts from a saved situation instead of a random start.

        `apply(env, state, seed)` restores a state and returns (obs, info). The
        default works for environments that accept
        reset(options={"reset_to_state": state}), such as gym-pusht.
        """

        def __init__(self, env, situations: list[dict], apply=None, shuffle: bool = True, seed: int = 0):
            super().__init__(env)
            if not situations:
                raise ValueError("no situations to reset to")
            self.situations = situations
            self._apply = apply or (lambda e, state, seed: e.reset(seed=seed, options={"reset_to_state": state}))
            self._rng = np.random.default_rng(seed)
            self._shuffle = shuffle
            self._next = 0
            self.current: dict | None = None

        def reset(self, *, seed=None, options=None):
            if self._shuffle:
                self.current = self.situations[int(self._rng.integers(len(self.situations)))]
            else:
                self.current = self.situations[self._next % len(self.situations)]
                self._next += 1
            obs, info = self._apply(self.env, np.asarray(self.current["state"], dtype=float), seed)
            return obs, {**info, "situation": self.current["id"]}

    return ResetToSituation
