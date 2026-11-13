"""The SDK entry point: wrap your policy loop with a Tracer.

    tracer = Tracer("runs/my-run", config=TraceConfig(position_dims=[0, 1, 2]))
    with tracer.episode(task="pick up the cube") as ep:
        obs = env.reset()
        while not done:
            samples = policy.sample_actions(obs, n=8)
            decision = ep.step(state=obs["state"], action=samples[0],
                               action_samples=samples, frame=obs["image"])
            if decision.halt:
                break
            obs, done = env.step(decision.action)
        ep.end(success=...)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .gate import Decision, Gate
from .metrics import OnlineMetrics, TraceConfig
from .predictor import Predictor, first_alert
from .store import dump_json, write_episode


class Tracer:
    def __init__(
        self,
        run_dir: str | Path,
        config: TraceConfig | None = None,
        gate: Gate | None = None,
        predictor: Predictor | None = None,
        run_meta: dict | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.config = config or TraceConfig()
        self.gate = gate
        self.predictor = predictor
        (self.run_dir / "episodes").mkdir(parents=True, exist_ok=True)
        self._count = sum(1 for _ in (self.run_dir / "episodes").iterdir())
        dump_json(
            self.run_dir / "run.json",
            {
                "name": self.run_dir.name,
                "created_at": time.time(),
                "config": self.config.to_dict(),
                "gate": gate.config if gate else None,
                "alert": {"threshold": predictor.threshold, "sustain": predictor.sustain} if predictor else None,
                **(run_meta or {}),
            },
        )

    def episode(self, task: str = "", meta: dict | None = None, episode_id: str | None = None) -> "EpisodeTrace":
        eid = episode_id or f"ep_{self._count:05d}"
        self._count += 1
        return EpisodeTrace(self, eid, task, meta or {})


class EpisodeTrace:
    def __init__(self, tracer: Tracer, episode_id: str, task: str, meta: dict):
        self.tracer = tracer
        self.episode_id = episode_id
        self.task = task
        self.meta = dict(meta)
        self._metrics = OnlineMetrics(tracer.config)
        self._rows: list[dict] = []
        self._frames: list[np.ndarray] = []
        self._internals: dict[str, tuple[list[int], list[np.ndarray]]] = {}
        self._novelty_vec: np.ndarray | None = None
        self._started = time.time()
        self._ended = False
        if tracer.gate:
            tracer.gate.reset()

    def __enter__(self) -> "EpisodeTrace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._ended:
            self.end(success=None, reason="aborted" if exc_type is None else f"error: {exc_type.__name__}")

    def step(
        self,
        state: Any,
        action: Any,
        action_samples: Any = None,
        frame: np.ndarray | None = None,
        belief: dict | None = None,
        plan_step: int | None = None,
        signals: dict[str, float] | None = None,
        reward: float | None = None,
        internals: dict[str, Any] | None = None,
        snapshot: Any = None,
    ) -> Decision:
        """Record one step and return the gate's decision. Execute `decision.action`.

        state: what the policy observed (1-D array).
        action: the action the policy wants to take (1-D array).
        action_samples: optional (N, action_dim) array of alternative samples from
            the policy; their disagreement is the main uncertainty signal.
        frame: optional HxWx3 uint8 camera image.
        belief: optional dict of whatever the policy believes (subtask, detected
            object pose, plan text). Stored as-is and shown in the viewer.
        plan_step: optional index of the current plan step / subtask.
        signals: optional extra per-step numbers of your own (for example a
            model's token entropy). Stored as metrics and usable as predictor features.
        reward: optional reward received for the previous action (for RL export).
        internals: optional arrays from inside the model at this step (an embedding,
            attention keypoints, sampled plans, token probabilities...). Pass them
            only on the steps where the model actually ran. Saved to internals.npz.
        snapshot: optional environment state that can be restored later, so
            training can restart from this exact situation.
        """
        tr = self.tracer
        state = np.asarray(state, dtype=float).ravel()
        action = np.asarray(action, dtype=float).ravel()
        samples = None if action_samples is None else np.asarray(action_samples, dtype=float)

        t = len(self._rows)
        internals = dict(internals or {})
        if snapshot is not None:
            internals["snapshot"] = snapshot
        for key, value in internals.items():
            self._internals.setdefault(key, ([], []))
            self._internals[key][0].append(t)
            self._internals[key][1].append(np.asarray(value))
        src = getattr(tr.predictor, "novelty_source", None) if tr.predictor else None
        if src and src in internals:
            self._novelty_vec = np.asarray(internals[src], dtype=float).ravel()

        m = self._metrics.update(state, action, samples, plan_step)
        for k, v in (signals or {}).items():
            m[k] = float(v)
        ref = self._novelty_vec if (src and self._novelty_vec is not None) else state
        risk = tr.predictor.score_step(m, ref) if tr.predictor else float("nan")
        if tr.gate:
            decision = tr.gate.check(state, action, risk)
        else:
            decision = Decision(action=action.copy(), risk=risk)
        self._metrics.record_intervention(decision.intervened)

        row = {
            "t": len(self._rows),
            "state": state.tolist(),
            "action": action.tolist(),
            "executed": decision.action.tolist(),
            "verdict": decision.verdict,
            "violations": decision.violations,
            "belief": belief or {},
            "plan_step": -1 if plan_step is None else int(plan_step),
            "risk": risk,
            "reward": float("nan") if reward is None else float(reward),
        }
        for name, value in m.items():
            row[f"m_{name}"] = value
        self._rows.append(row)
        if frame is not None:
            self._frames.append(np.asarray(frame, dtype=np.uint8))
        return decision

    def end(self, success: bool | None, reason: str = "", meta: dict | None = None, final_reward: float | None = None) -> dict:
        """Close the episode and write it to disk. Returns the episode summary.

        final_reward: the reward that followed the last action (there is no later
        step() call to carry it).
        """
        if self._ended:
            raise RuntimeError("episode already ended")
        self._ended = True
        tr = self.tracer
        self.meta.update(meta or {})
        risk = np.array([r["risk"] for r in self._rows], dtype=float)
        verdicts = [r["verdict"] for r in self._rows]
        summary: dict[str, Any] = {
            "episode_id": self.episode_id,
            "task": self.task,
            "success": success,
            "reason": reason,
            "n_steps": len(self._rows),
            "started_at": self._started,
            "duration_s": time.time() - self._started,
            "max_risk": float(np.nanmax(risk)) if len(risk) and not np.all(np.isnan(risk)) else None,
            "first_alert_step": None,
            "causes": [],
            "gate": {v: verdicts.count(v) for v in ("clamp", "block", "halt")},
            "final_reward": final_reward,
            "meta": self.meta,
        }
        if tr.predictor and len(self._rows):
            alert = first_alert(risk, tr.predictor.threshold, tr.predictor.sustain)
            summary["first_alert_step"] = alert
            if alert is not None:
                names = [k[2:] for k in self._rows[0] if k.startswith("m_")]
                metrics = {n: np.array([r.get(f"m_{n}", np.nan) for r in self._rows], float) for n in names}
                states = np.array([r["state"] for r in self._rows], float)
                if getattr(tr.predictor, "novelty_source", None) in self._internals:
                    from .store import forward_fill

                    ts, vals = self._internals[tr.predictor.novelty_source]
                    states = forward_fill(ts, np.stack(vals).reshape(len(vals), -1), len(self._rows))
                summary["causes"] = tr.predictor.explain(metrics, states, alert)
        write_episode(tr.run_dir / "episodes" / self.episode_id, self._rows, self._frames, summary, self._internals)
        return summary
