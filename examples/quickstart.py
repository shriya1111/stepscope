"""Trace your own policy loop. Runs on the built-in simulator; swap in your env and policy.

    python examples/quickstart.py
    stepscope serve --runs runs
"""

from stepscope import Gate, TraceConfig, Tracer
from stepscope.sim import NoisyScriptedPolicy, TabletopEnv

env, policy = TabletopEnv(), NoisyScriptedPolicy()

tracer = Tracer(
    "runs/quickstart",
    # Which parts of your state/action vectors are position and motion.
    config=TraceConfig(position_dims=[0, 1, 2], motion_dims=[0, 1, 2], max_steps=90),
    # Hard limits checked before every action is sent to the robot.
    gate=Gate(
        {
            "position_dims": [0, 1, 2],
            "delta_dims": [0, 1, 2],
            "constraints": [
                {"type": "max_step", "max_norm": 0.05},
                {"type": "workspace_box", "low": [0, 0, 0], "high": [1, 1, 0.3]},
                {"type": "keep_out_box", "name": "operator_zone", "low": [0, 0.7, 0], "high": [0.24, 1, 0.3]},
            ],
        }
    ),
    # predictor=Predictor.load("models/predictor.json"),  # adds live risk scores once you have trained one
)

for seed in range(20):
    obs = env.reset(seed)
    policy.reset(seed)
    with tracer.episode(task="Pick up the cube and place it on the ring") as ep:
        while True:
            samples, belief = policy.sample_actions(obs)  # (N, action_dim) samples from the policy
            decision = ep.step(
                state=obs["state"],
                action=samples[0],
                action_samples=samples,   # optional: enables the uncertainty metrics
                frame=obs["image"],       # optional: camera frame for the viewer
                belief=belief,            # optional: anything the policy believes
                plan_step=policy.plan_step,  # optional: enables plan-progress metrics
            )
            if decision.halt:
                ep.end(success=False, reason="halted_by_gate")
                break
            obs, _, terminated, truncated, info = env.step(decision.action)  # execute the gated action
            if terminated or truncated:
                ep.end(success=info["success"], reason=info["failure"] or "success")
                break

print("Traced 20 episodes to runs/quickstart")
