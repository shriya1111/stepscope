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
from ..safety import Scene, assess, choose_plan, path_clearance
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
BLOCK_RADIUS = 45.0  # rough half-extent of the T block, for hazard clearance


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

    def infer(self, pixels: np.ndarray, agent_pos: np.ndarray, noise=None, repeat: int = 1, keep_stages: bool = False) -> dict:
        """One batched forward pass of the whole policy.

        pixels (B, n_obs, H, W, 3) uint8, agent_pos (B, n_obs, 2). `repeat` draws
        that many plans per observation. `noise` (B*repeat, horizon, 2) fixes the
        starting noise, so two calls differ only by their inputs.
        Returns plans (B*repeat, horizon, 2) in pixels, the conditioning vectors,
        the spread between samples after each denoising step and, with
        keep_stages, `stages` (n_denoise, B*repeat, horizon, 2): the plan the
        model would output if it stopped after each denoising step.
        """
        torch = self.torch
        img = torch.from_numpy(np.ascontiguousarray(pixels)).permute(0, 1, 4, 2, 3).float() / 255
        img = (img - self.img_mean) / self.img_std
        state = torch.from_numpy(np.asarray(agent_pos, dtype=np.float32))
        state = (state - self.s_min) / (self.s_max - self.s_min) * 2 - 1
        unnorm = lambda a: (a + 1) / 2 * (self.a_max - self.a_min) + self.a_min
        with torch.no_grad():
            diffusion = self.policy.diffusion
            cond = diffusion._prepare_global_conditioning({"observation.state": state, "observation.images": img[:, :, None]})
            gc = cond.repeat_interleave(repeat, dim=0)
            n = len(gc)
            sample = torch.randn(n, diffusion.config.horizon, 2) if noise is None else torch.as_tensor(noise, dtype=torch.float32)
            # Same loop as DiffusionModel.conditional_sample, but keeping the intermediate steps.
            diffusion.noise_scheduler.set_timesteps(diffusion.num_inference_steps)
            spread, path, stages = [], [], []
            for t in diffusion.noise_scheduler.timesteps:
                out = diffusion.unet(sample, torch.full((n,), int(t), dtype=torch.long), global_cond=gc)
                step = diffusion.noise_scheduler.step(out, t, sample)
                sample = step.prev_sample
                px = unnorm(sample)
                spread.append(float(px.std(dim=0).mean()))
                path.append(px[0].numpy().copy())
                if keep_stages:
                    stages.append(unnorm(step.pred_original_sample.clamp(-1, 1)).numpy().copy())
        return {
            "plans": unnorm(sample).numpy(),
            "cond": cond.numpy(),
            "spread": np.array(spread, np.float32),
            "path": np.stack(path).astype(np.float32),
            "stages": np.stack(stages) if keep_stages else None,
        }

    def decide(self) -> dict:
        """Run the model once on the current observation history. Returns:

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
        r = self.infer(
            np.stack([o["pixels"] for o in self.history])[None],
            np.stack([o["agent_pos"] for o in self.history])[None],
            repeat=self.n_samples,
        )
        horizon = r["plans"]
        start = self.n_obs - 1
        chunks = horizon[:, start : start + self.n_exec]
        overlap = horizon.shape[1] - (start + self.n_exec)
        shift = float("nan")
        if self.prev_tail is not None and overlap > 0:
            shift = float(np.linalg.norm(horizon[0, start : start + overlap] - self.prev_tail, axis=1).mean())
        self.prev_tail = horizon[0, start + self.n_exec :].copy()
        internals = {
            "plans": horizon.astype(np.float32),
            "modes": plan_modes(chunks, MODE_THRESHOLD_PX),
            "keypoints": self.keypoints_px().astype(np.float32),
            "embedding": r["cond"][0].astype(np.float32),
            "denoise_spread": r["spread"],
            "denoise_path": r["path"],
        }
        return {"chunks": chunks, "plan_shift": shift, "internals": internals}

    def keypoints_px(self) -> np.ndarray:
        """Attention points of the most recent frame, in image pixels."""
        # They come out in [-1, 1] over the centre crop the encoder sees.
        crop = self.policy.config.crop_shape or (96, 96)
        box = np.array([crop[1], crop[0]])
        return (self._keypoints[-1] + 1) / 2 * box + (96 - box) / 2

    def use_plan(self, chunks: np.ndarray, internals: dict, index: int) -> None:
        """Make sampled plan `index` the executed one (moves it to position 0 in place)."""
        if index == 0:
            return
        order = [index] + [i for i in range(len(chunks)) if i != index]
        chunks[:] = chunks[order]
        internals["plans"] = internals["plans"][order]
        internals["modes"] = plan_modes(chunks, MODE_THRESHOLD_PX)
        start = self.n_obs - 1
        self.prev_tail = internals["plans"][0, start + self.n_exec :].copy()


def simulate_plans(sim_env, snapshot: np.ndarray, chunks: np.ndarray, scene: Scene | None = None) -> list[dict]:
    """Predicted future of each sampled plan: run it in a copy of the simulator.

    sim_env is a second gym-pusht environment used as the world model (on a real
    robot this would be a learned or physics model). Returns one dict per plan:
    reward before/after, `gain`, the pusher and block paths, and how close each
    came to the scene's hazards.
    """
    futures = []
    for chunk in chunks:
        _, info = restore_pusht(sim_env, snapshot)
        u = sim_env.unwrapped
        start_reward = None
        agent_path, block_path, reward = [info["pos_agent"].copy()], [info["block_pose"][:2].copy()], 0.0
        for a in chunk:
            _, reward, terminated, _, info = sim_env.step(np.asarray(a, dtype=np.float32))
            if start_reward is None:
                start_reward = float(reward)
            agent_path.append(info["pos_agent"].copy())
            block_path.append(info["block_pose"][:2].copy())
            if terminated:
                break
        f = {
            "reward_after": float(reward),
            "gain": float(reward) - float(start_reward or 0.0),
            "success": bool(info.get("is_success", False)),
            "agent_path": np.array(agent_path),
            "block_path": np.array(block_path),
            "contact": False,
            "risk": 0.0,
            "clearance": None,
            "hazard": None,
        }
        if scene and scene.hazards:
            speed = float(np.linalg.norm(np.diff(f["agent_path"], axis=0), axis=1).max())
            pusher = assess(f["agent_path"][0], f["agent_path"][None, 1:], speed, scene)
            # The block is pushed, so it can be shoved into a hazard even if the pusher stays clear.
            block = min((path_clearance(f["block_path"], h, BLOCK_RADIUS) + (h.name,) for h in scene.hazards), key=lambda r: r[0])
            f["clearance"] = float(min(pusher["clearance"], block[0]))
            f["contact"] = pusher["time_to_contact"] is not None or block[1] is not None
            f["risk"] = float(max(pusher["risk"], 1.0 if block[1] is not None else 0.0))
            f["hazard"] = block[2] if block[0] < pusher["clearance"] else pusher["hazard"]
        futures.append(f)
    return futures


def rollout(env, policy: DiffusionPushTPolicy, tracer: Tracer, seed: int, sim_env=None, scene: Scene | None = None, shield: bool = False) -> dict:
    """One traced episode.

    With sim_env, the future of every sampled plan is predicted at each decision
    and stored. With shield=True the plan with the best predicted future is the
    one executed (see safety.choose_plan), instead of the first sample.
    """
    obs, info = env.reset(seed=seed)
    policy.reset(seed, obs)
    success, reason, best = False, "timeout", 0.0
    with tracer.episode(task=TASK, meta={"seed": seed}) as ep:
        done = False
        reward = None
        while not done:
            d = policy.decide()
            chunks, z = d["chunks"], d["internals"]
            snapshot = np.concatenate([info["pos_agent"], info["block_pose"]])
            extra = {}
            if sim_env is not None:
                futures = simulate_plans(sim_env, snapshot, chunks, scene)
                pick = choose_plan(futures) if shield else 0
                extra = {"shield_switched": float(pick != 0), "predicted_gain": futures[pick]["gain"], "plans_in_contact": float(sum(f["contact"] for f in futures))}
                if pick:
                    policy.use_plan(chunks, z, pick)
                    futures = [futures[pick]] + [f for i, f in enumerate(futures) if i != pick]
                z = {**z, "future_gain": np.array([f["gain"] for f in futures], np.float32),
                     "future_contact": np.array([f["contact"] for f in futures]),
                     "future_block": np.stack([_pad(f["block_path"], chunks.shape[1] + 1) for f in futures]).astype(np.float32)}
            sizes = np.bincount(z["modes"])
            signals = {
                "plan_shift": d["plan_shift"],
                "n_modes": float(len(sizes)),
                "top_mode_share": float(sizes[0] / sizes.sum()),
                "commit_step": float(commit_step(z["denoise_spread"])),
                "final_spread": float(z["denoise_spread"][-1]),
                **extra,
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
    ap.add_argument("--scene", help="YAML scene file with hazards")
    ap.add_argument("--futures", action="store_true", help="predict and store the future of every sampled plan")
    ap.add_argument("--shield", action="store_true", help="execute the sampled plan with the best predicted future")
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
            "shield": bool(args.shield),
        },
    )
    done = sum(1 for _ in (Path(args.run) / "episodes").iterdir())  # resume where a previous call stopped
    wins = 0
    scene = Scene.from_file(args.scene) if args.scene else None
    sim_env = gym.make("gym_pusht/PushT-v0", obs_type="state", max_episode_steps=10_000) if (args.futures or args.shield) else None
    for i in range(done, args.episodes):
        t0 = time.time()
        s = rollout(env, policy, tracer, args.seed + i, sim_env=sim_env, scene=scene, shield=args.shield)
        wins += bool(s["success"])
        print(f"episode {i + 1}/{args.episodes} seed {args.seed + i}: {s['reason']} in {s['n_steps']} steps, "
              f"best coverage {s['meta']['sim']['best_coverage']:.2f}, {time.time() - t0:.0f}s", flush=True)




def _pad(path: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([path, np.repeat(path[-1:], n - len(path), axis=0)]) if len(path) < n else path[:n]


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


if __name__ == "__main__":
    main()
