"""Step-level process metrics, computed online from the policy loop.

Every metric here is embodiment-agnostic: it needs only the state vector, the
action the policy proposed, and (optionally) several sampled actions and the
index of the plan step the policy believes it is on.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

METRIC_NAMES = [
    "sample_spread",
    "sample_spread_ema",
    "direction_disagreement",
    "direction_disagreement_ema",
    "action_jerk",
    "action_jerk_ema",
    "reversal_rate",
    "path_efficiency",
    "net_speed",
    "plan_regressions",
    "subgoal_dwell",
    "gate_interventions",
    "elapsed",
]

_EPS = 1e-9


@dataclass
class TraceConfig:
    """Tells stepscope how to read your state and action vectors.

    position_dims: indices of the state vector that are a position (used for
        progress metrics and by the gate). Default: the whole state.
    motion_dims: indices of the action vector that are motion commands
        (deltas or velocities). Default: the whole action.
    max_steps: episode step budget, if there is one.
    window: number of steps the rolling metrics look back over.
    """

    position_dims: Sequence[int] | None = None
    motion_dims: Sequence[int] | None = None
    max_steps: int | None = None
    window: int = 8
    ema_alpha: float = 0.25

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("position_dims", "motion_dims"):
            if d[k] is not None:
                d[k] = [int(i) for i in d[k]]
        return d


class OnlineMetrics:
    """Rolling metric state for one episode. Call update() once per step."""

    def __init__(self, config: TraceConfig):
        self.cfg = config
        w = config.window
        self._pos: deque[np.ndarray] = deque(maxlen=w + 1)
        self._reversals: deque[float] = deque(maxlen=w)
        self._interventions: deque[float] = deque(maxlen=w)
        self._prev_motion: np.ndarray | None = None
        self._prev_plan: int | None = None
        self._plan_regressions = 0
        self._dwell = 0
        self._ema: dict[str, float] = {}
        self._t = 0

    def _ema_update(self, key: str, value: float) -> float:
        if np.isnan(value):
            return self._ema.get(key, float("nan"))
        prev = self._ema.get(key)
        a = self.cfg.ema_alpha
        new = value if prev is None else a * value + (1 - a) * prev
        self._ema[key] = new
        return new

    def record_intervention(self, intervened: bool) -> None:
        """Tell the metrics whether the gate changed the action on this step."""
        self._interventions.append(1.0 if intervened else 0.0)

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        action_samples: np.ndarray | None = None,
        plan_step: int | None = None,
    ) -> dict[str, float]:
        cfg = self.cfg
        nan = float("nan")
        pos = state if cfg.position_dims is None else state[list(cfg.position_dims)]
        motion = action if cfg.motion_dims is None else action[list(cfg.motion_dims)]
        m: dict[str, float] = {}

        # --- disagreement between sampled actions (policy uncertainty) ---
        if action_samples is not None and len(action_samples) > 1:
            s = np.asarray(action_samples, dtype=float)
            sm = s if cfg.motion_dims is None else s[:, list(cfg.motion_dims)]
            m["sample_spread"] = float(sm.std(axis=0).mean())
            norms = np.linalg.norm(sm, axis=1, keepdims=True)
            units = sm / np.maximum(norms, _EPS)
            m["direction_disagreement"] = float(1.0 - np.linalg.norm(units.mean(axis=0)))
        else:
            m["sample_spread"] = nan
            m["direction_disagreement"] = nan
        m["sample_spread_ema"] = self._ema_update("spread", m["sample_spread"])
        m["direction_disagreement_ema"] = self._ema_update("disagree", m["direction_disagreement"])

        # --- smoothness of the executed command stream ---
        if self._prev_motion is not None:
            m["action_jerk"] = float(np.linalg.norm(motion - self._prev_motion))
            n1, n0 = np.linalg.norm(motion), np.linalg.norm(self._prev_motion)
            reversed_ = n1 > 1e-6 and n0 > 1e-6 and float(motion @ self._prev_motion) < 0
            self._reversals.append(1.0 if reversed_ else 0.0)
        else:
            m["action_jerk"] = nan
        m["action_jerk_ema"] = self._ema_update("jerk", m["action_jerk"])
        m["reversal_rate"] = float(np.mean(self._reversals)) if self._reversals else 0.0
        self._prev_motion = np.array(motion, dtype=float)

        # --- progress through state space ---
        self._pos.append(np.array(pos, dtype=float))
        if len(self._pos) >= 2:
            pts = np.stack(self._pos)
            net = float(np.linalg.norm(pts[-1] - pts[0]))
            path = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            m["net_speed"] = net / (len(pts) - 1)
            m["path_efficiency"] = net / path if path > 1e-6 else nan
        else:
            m["net_speed"] = nan
            m["path_efficiency"] = nan

        # --- progress through the plan ---
        if plan_step is not None:
            if self._prev_plan is not None:
                if plan_step < self._prev_plan:
                    self._plan_regressions += 1
                self._dwell = self._dwell + 1 if plan_step == self._prev_plan else 0
            self._prev_plan = int(plan_step)
            m["plan_regressions"] = float(self._plan_regressions)
            m["subgoal_dwell"] = float(self._dwell)
        else:
            m["plan_regressions"] = nan
            m["subgoal_dwell"] = nan

        m["gate_interventions"] = float(np.mean(self._interventions)) if self._interventions else 0.0
        m["elapsed"] = self._t / cfg.max_steps if cfg.max_steps else nan
        self._t += 1
        return m
