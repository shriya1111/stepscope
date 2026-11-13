"""Run traced episodes of the built-in simulator."""

from __future__ import annotations

from ..tracer import Tracer
from .policy import NoisyScriptedPolicy
from .tabletop import TabletopEnv

TASK = "Pick up the blue cube and place it on the green ring"


def _sim_cause(success: bool, failure: str | None, truth: dict) -> str | None:
    """What actually went wrong, from simulator ground truth (for scoring the diagnosis)."""
    if success:
        return None
    if failure == "object_off_table":
        return "knocked off table"
    cost = {
        "oscillation": truth["zigzag_steps"] * 0.65,
        "stalled": truth["frozen_steps"],
        "retry loop": 13 * (truth["grasp_misses"] + truth["drops"]),
    }
    best = max(cost, key=cost.get)
    return best if cost[best] > 0 else "unknown"


def rollout(env: TabletopEnv, policy: NoisyScriptedPolicy, tracer: Tracer, seed: int) -> dict:
    obs = env.reset(seed)
    policy.reset(seed)
    success, reason, info = False, "timeout", {}
    with tracer.episode(task=TASK, meta={"seed": seed}) as ep:
        while True:
            samples, belief = policy.sample_actions(obs)
            decision = ep.step(
                state=obs["state"],
                action=samples[0],
                action_samples=samples,
                frame=obs["image"],
                belief=belief,
                plan_step=policy.plan_step,
            )
            if decision.halt:
                reason = "halted_by_gate"
                break
            obs, _, terminated, truncated, info = env.step(decision.action)
            if terminated or truncated:
                success = bool(info["success"])
                reason = "success" if success else info["failure"]
                break
        truth = {**env.ground_truth(), **policy.ground_truth()}
        truth["cause"] = _sim_cause(success, info.get("failure"), truth)
        return ep.end(success=success, reason=reason, meta={"sim": truth})
