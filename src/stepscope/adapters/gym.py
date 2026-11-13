"""Trace any Gymnasium-style environment.

    from stepscope.adapters.gym import trace_rollout
    summary = trace_rollout(env, policy, tracer, seed=0)

`policy(obs)` returns either one action, or an (N, action_dim) array of sampled
actions, in which case the first sample is executed and the rest feed the
uncertainty metrics.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..tracer import Tracer


def trace_rollout(
    env: Any,
    policy: Callable[[Any], Any],
    tracer: Tracer,
    seed: int | None = None,
    task: str = "",
    max_steps: int = 1000,
    state_fn: Callable[[Any], Any] = lambda obs: obs,
    frame_fn: Callable[[Any], np.ndarray | None] = lambda obs: None,
    success_fn: Callable[[dict, float], bool] = lambda info, total_reward: bool(info.get("is_success", False)),
) -> dict:
    obs, info = env.reset(seed=seed)
    total, reason = 0.0, "max_steps"
    with tracer.episode(task=task, meta={"seed": seed}) as ep:
        for _ in range(max_steps):
            out = np.asarray(policy(obs), dtype=float)
            samples = out if out.ndim == 2 else None
            action = out[0] if out.ndim == 2 else out
            decision = ep.step(state=state_fn(obs), action=action, action_samples=samples, frame=frame_fn(obs))
            if decision.halt:
                reason = "halted_by_gate"
                break
            obs, reward, terminated, truncated, info = env.step(decision.action.astype(env.action_space.dtype))
            total += float(reward)
            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                break
        success = reason != "halted_by_gate" and success_fn(info, total)
        return ep.end(success=success, reason=reason, meta={"total_reward": total})
