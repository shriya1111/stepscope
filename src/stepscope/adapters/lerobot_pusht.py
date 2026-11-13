"""Trace a real pretrained policy: LeRobot's diffusion policy on the PushT task.

    pip install "lerobot[diffusion,pusht]"
    python -m stepscope.adapters.lerobot_pusht --run runs/pusht --episodes 100

The policy (`lerobot/diffusion_pusht`, 263M parameters) looks at a 96x96 camera
image and the pusher position, and outputs a chunk of target positions by
denoising random noise. We draw several chunks per decision from different
noise; how much they disagree is the policy's own uncertainty, read directly
from the model rather than assumed.

Three things differ from the stock LeRobot evaluation, all to make CPU runs
practical and the traces richer:
  * DDIM sampling with 10 steps instead of DDPM with 100
  * N chunks are sampled per decision (the first one is executed)
  * model internals are captured at each decision (see DiffusionPushTPolicy.decide)
  * normalisation constants are read from the checkpoint file itself
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..explain import commit_step, plan_modes
from ..gate import Gate
from ..metrics import TraceConfig
from ..predictor import Predictor
from ..tracer import Tracer

REPO = "lerobot/diffusion_pusht"
MAX_STEPS = 300
TASK = "Push the T-shaped block onto the target outline"
TRACE_CONFIG = TraceConfig(position_dims=[0, 1], motion_dims=[0, 1], max_steps=MAX_STEPS)
GATE_CONFIG = {
    "position_dims": [0, 1],
    "delta_dims": [0, 1],
    "constraints": [{"type": "workspace_box", "name": "workspace", "low": [0, 0], "high": [512, 512]}],
}
# Extra per-step signal this adapter logs, on top of the built-in metrics.
EXTRA_FEATURES = ["plan_shift", "n_modes", "top_mode_share", "commit_step", "final_spread"]
# Sampled plans closer than this (mean point distance, pixels of a 512 px workspace) count as one option.
MODE_THRESHOLD_PX = 30.0


class DiffusionPushTPolicy:
    def __init__(self, n_samples: int = 8, ddim_steps: int = 10):
        import torch
        from diffusers import DDIMScheduler
        from huggingface_hub import hf_hub_download
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from safetensors.torch import load_file

        self.torch = torch
        self.n_samples = n_samples
        self.policy = DiffusionPolicy.from_pretrained(REPO)
        self.policy.eval()
        cfg = self.policy.config
        self.n_obs, self.n_exec = cfg.n_obs_steps, cfg.n_action_steps
        diffusion = self.policy.diffusion
        diffusion.noise_scheduler = DDIMScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            clip_sample_range=cfg.clip_sample_range,
            prediction_type=cfg.prediction_type,
        )
        diffusion.num_inference_steps = ddim_steps
        # This checkpoint predates LeRobot's processor files; its normalisation
        # constants live in the weights file.
        sd = load_file(hf_hub_download(REPO, "model.safetensors"))
        g = lambda k: sd[k].float()
        self.img_mean = g("normalize_inputs.buffer_observation_image.mean").view(3, 1, 1)
        self.img_std = g("normalize_inputs.buffer_observation_image.std").view(3, 1, 1)
        self.s_min, self.s_max = g("normalize_inputs.buffer_observation_state.min"), g("normalize_inputs.buffer_observation_state.max")
        self.a_min, self.a_max = g("unnormalize_outputs.buffer_action.min"), g("unnormalize_outputs.buffer_action.max")
        # The vision encoder ends in a spatial softmax: 32 (x, y) image points. Capture them.
        self._keypoints = None
        diffusion.rgb_encoder.pool.register_forward_hook(
            lambda _m, _i, out: setattr(self, "_keypoints", out.detach().numpy().copy())
        )

    def reset(self, seed: int, obs: dict) -> None:
        self.torch.manual_seed(seed)
        self.history = deque([obs] * self.n_obs, maxlen=self.n_obs)
        self.prev_tail: np.ndarray | None = None

    def observe(self, obs: dict) -> None:
        self.history.append(obs)

    def decide(self) -> dict:
        """Run the model once. Returns:

        chunks        (N, n_exec, 2) target positions for the next steps; chunks[0] is executed
        plan_shift    how far the new plan moved from what the previous plan predicted (px)
        internals     arrays from inside the model, for the viewer and for training data:
            plans           (N, horizon, 2) every sampled plan, in pixels
            modes           (N,) which option each sampled plan belongs to
            keypoints       (32, 2) image points the vision encoder's last layer attends to
            embedding       the vector the action head is conditioned on
            denoise_spread  (n_denoise,) disagreement between samples after each denoising step (px)
            denoise_path    (n_denoise, horizon, 2) the executed plan forming out of noise
        """
        torch = self.torch
        img = torch.stack([torch.from_numpy(o["pixels"]).permute(2, 0, 1).float() / 255 for o in self.history])
        img = (img - self.img_mean) / self.img_std
        state = torch.stack([torch.from_numpy(o["agent_pos"]).float() for o in self.history])
        state = (state - self.s_min) / (self.s_max - self.s_min) * 2 - 1
        batch = {"observation.state": state[None], "observation.images": img[None, :, None]}
        unnorm = lambda a: (a + 1) / 2 * (self.a_max - self.a_min) + self.a_min
        with torch.no_grad():
            diffusion = self.policy.diffusion
            cond = diffusion._prepare_global_conditioning(batch)  # image encoder runs once
            # Same loop as DiffusionModel.conditional_sample, but keeping the intermediate steps.
            n = self.n_samples
            sample = torch.randn(n, diffusion.config.horizon, 2)
            diffusion.noise_scheduler.set_timesteps(diffusion.num_inference_steps)
            gc = cond.expand(n, -1)
            spread, path = [], []
            for t in diffusion.noise_scheduler.timesteps:
                out = diffusion.unet(sample, torch.full((n,), int(t), dtype=torch.long), global_cond=gc)
                sample = diffusion.noise_scheduler.step(out, t, sample).prev_sample
                px = unnorm(sample)
                spread.append(float(px.std(dim=0).mean()))
                path.append(px[0].numpy().copy())
        horizon = unnorm(sample).numpy()
        start = self.n_obs - 1
        chunks = horizon[:, start : start + self.n_exec]
        overlap = horizon.shape[1] - (start + self.n_exec)
        shift = float("nan")
        if self.prev_tail is not None and overlap > 0:
            shift = float(np.linalg.norm(horizon[0, start : start + overlap] - self.prev_tail, axis=1).mean())
        self.prev_tail = horizon[0, start + self.n_exec :].copy()
        # Keypoints come out in [-1, 1] over the centre crop the encoder sees.
        crop = self.policy.config.crop_shape or (96, 96)
        size = self.history[-1]["pixels"].shape[0]
        kp = (self._keypoints[-1] + 1) / 2 * np.array([crop[1], crop[0]]) + (size - np.array([crop[1], crop[0]])) / 2
        internals = {
            "plans": horizon.astype(np.float32),
            "modes": plan_modes(chunks, MODE_THRESHOLD_PX),
            "keypoints": kp.astype(np.float32),
            "embedding": cond[0].numpy().astype(np.float32),
            "denoise_spread": np.array(spread, np.float32),
            "denoise_path": np.stack(path).astype(np.float32),
        }
        return {"chunks": chunks, "plan_shift": shift, "internals": internals}


def rollout(env, policy: DiffusionPushTPolicy, tracer: Tracer, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    policy.reset(seed, obs)
    success, reason, best = False, "timeout", 0.0
    with tracer.episode(task=TASK, meta={"seed": seed}) as ep:
        done = False
        reward = None
        while not done:
            d = policy.decide()
            chunks, z = d["chunks"], d["internals"]
            sizes = np.bincount(z["modes"])
            signals = {
                "plan_shift": d["plan_shift"],
                "n_modes": float(len(sizes)),
                "top_mode_share": float(sizes[0] / sizes.sum()),
                "commit_step": float(commit_step(z["denoise_spread"])),
                "final_spread": float(z["denoise_spread"][-1]),
            }
            for k in range(chunks.shape[1]):
                pos = obs["agent_pos"].astype(float)
                # The policy outputs absolute targets; trace them as moves from the current position.
                decision = ep.step(
                    state=pos,
                    action=chunks[0, k] - pos,
                    action_samples=chunks[:, k] - pos,
                    frame=obs["pixels"],
                    belief={"chunk_step": k, "target": [round(float(v), 1) for v in chunks[0, k]]},
                    signals=signals,
                    reward=reward,
                    internals=z if k == 0 else None,
                    snapshot=np.concatenate([info["pos_agent"], info["block_pose"]]),
                )
                if decision.halt:
                    reason, done = "halted_by_gate", True
                    break
                obs, reward, terminated, truncated, info = env.step((pos + decision.action).astype(np.float32))
                policy.observe(obs)
                best = max(best, float(reward))
                if terminated or truncated:
                    success = bool(info.get("is_success", False))
                    reason = "success" if success else "timeout"
                    done = True
                    break
        final = None if reason == "halted_by_gate" else reward
        return ep.end(success=success, reason=reason, meta={"sim": {"best_coverage": best, "cause": None}}, final_reward=final)


def main(argv: list[str] | None = None) -> None:
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the environment)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--ddim-steps", type=int, default=10)
    ap.add_argument("--model", help="stepscope predictor to score risk live")
    args = ap.parse_args(argv)

    policy = DiffusionPushTPolicy(n_samples=args.samples, ddim_steps=args.ddim_steps)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=MAX_STEPS)
    tracer = Tracer(
        args.run,
        config=TRACE_CONFIG,
        gate=Gate(GATE_CONFIG),
        predictor=Predictor.load(args.model) if args.model else None,
        run_meta={
            "label": f"LeRobot diffusion policy on PushT ({REPO}), DDIM {args.ddim_steps} steps, {args.samples} samples",
            # Tells the viewer how to draw plans (workspace units) over the camera frame.
            "overlay": {"plan_extent": 512},
        },
    )
    done = sum(1 for _ in (Path(args.run) / "episodes").iterdir())  # resume where a previous call stopped
    wins = 0
    for i in range(done, args.episodes):
        t0 = time.time()
        s = rollout(env, policy, tracer, args.seed + i)
        wins += bool(s["success"])
        print(f"episode {i + 1}/{args.episodes} seed {args.seed + i}: {s['reason']} in {s['n_steps']} steps, "
              f"best coverage {s['meta']['sim']['best_coverage']:.2f}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()


def restore_pusht(env, state, seed=None):
    """Put a gym-pusht environment into a saved snapshot [agent x, y, block x, y, angle].

    Use as ResetToSituation(env, situations, apply=restore_pusht). gym-pusht's own
    reset_to_state sets the block position before its angle, which shifts the
    block; setting the angle first restores the snapshot exactly.
    """
    env.reset(seed=seed)
    u = env.unwrapped
    u.agent.position = [float(state[0]), float(state[1])]
    u.agent.velocity = (0, 0)
    u.block.angle = float(state[4])
    u.block.position = [float(state[2]), float(state[3])]
    u.block.velocity = (0, 0)
    u.block.angular_velocity = 0
    u.space.reindex_shapes_for_body(u.block)
    u.space.reindex_shapes_for_body(u.agent)
    info = u._get_info()
    info["is_success"] = False
    return u.get_obs(), info
