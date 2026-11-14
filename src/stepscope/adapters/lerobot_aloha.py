"""Trace a real ACT policy on a 3D simulated robot: ALOHA cube transfer in MuJoCo.

    pip install "lerobot" mujoco dm_control gym-aloha
    MUJOCO_GL=egl python -m stepscope.adapters.lerobot_aloha --run runs/aloha --episodes 50

Two 6-joint arms with grippers; the right arm picks up a cube and hands it to
the left. The policy (`lerobot/act_aloha_sim_transfer_cube_human`) is an Action
Chunking Transformer: from one camera image and the 14 joint positions it
outputs the next 100 joint targets in a single pass.

What is captured at each decision, from inside the model:
  * candidate plans: plan 0 is the model's standard output and is the one
    executed. The others are the same forward pass with dropout left on
    (Monte Carlo dropout), a standard way to see how firmly a network holds its
    answer. ACT's own latent "style" variable was the obvious source of
    alternatives, but this checkpoint ignores it: plans drawn from different
    latents differ by about 0.0001 rad.
  * attention map: the decoder's cross-attention over the camera image
  * embedding: the encoder's output, averaged over tokens
and, from the simulator, the 3D position of every arm link (forward kinematics)
plus the full physics state, so any moment can be restored exactly.

Differences from the stock LeRobot evaluation: the policy replans every
`--replan` steps (stock: every 100), and stored camera frames are downscaled.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..explain import plan_modes
from ..gate import Gate
from ..kinematics import Kinematics, project_to_camera
from ..metrics import TraceConfig
from ..predictor import Predictor
from ..safety import Scene, assess, choose_plan
from ..tracer import Tracer

REPO = "lerobot/act_aloha_sim_transfer_cube_human"
ENV_ID = "gym_aloha/AlohaTransferCube-v0"
MAX_STEPS = 400
TASK = "Pick up the cube with the right arm and hand it to the left arm"
FRAME_STRIDE = 4  # camera frames are stored at 1/4 resolution (160 x 120)
PATH_STRIDE = 5  # 3D paths of plans are stored every 5th step
ARM_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # the 12 arm joints inside the 14-d action (6 and 13 are grippers)
_LINKS = ["shoulder_link", "upper_arm_link", "upper_forearm_link", "lower_forearm_link", "wrist_link", "gripper_link"]
BODIES = [f"vx300s_{side}/{link}" for side in ("left", "right") for link in _LINKS]
LINKS = [[i, i + 1] for i in range(5)] + [[i, i + 1] for i in range(6, 11)]
GRIPPERS = [5, 11]  # indices into BODIES
JOINT_NAMES = [f"vx300s_{side}/{j}" for side in ("left", "right") for j in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")]
MODE_THRESHOLD_RAD = 0.08  # plans whose arm joints differ by less than this on average count as one option
TRACE_CONFIG = TraceConfig(max_steps=MAX_STEPS)
EXTRA_FEATURES = ["plan_shift", "n_modes", "top_mode_share", "final_spread"]


def make_env():
    import gymnasium as gym
    import gym_aloha  # noqa: F401  (registers the environment)

    return gym.make(ENV_ID, obs_type="pixels_agent_pos", max_episode_steps=MAX_STEPS + 50)


def physics_of(env):
    return env.unwrapped._env.physics


def make_kinematics(env) -> Kinematics:
    ph = physics_of(env)
    return Kinematics(ph.model.ptr, JOINT_NAMES, BODIES, links=LINKS, base_qpos=ph.data.qpos.copy())


def gate_config(kin: Kinematics) -> dict:
    """Limits for the simulated arms: joint speed, joint range, and a box the arms must stay inside."""
    low, high = np.full(14, -np.inf), np.full(14, np.inf)
    low[ARM_JOINTS], high[ARM_JOINTS] = kin.joint_low, kin.joint_high
    return {
        "position_dims": list(range(14)),
        "delta_dims": list(range(14)),
        "constraints": [
            {"type": "max_joint_step", "name": "joint_speed", "max_abs": 0.25},
            {"type": "joint_limits", "name": "joint_range", "low": low.tolist(), "high": high.tolist()},
            {"type": "cartesian_box", "name": "workspace", "low": [-0.95, -0.2, -0.02], "high": [0.95, 1.3, 1.0]},
        ],
    }


def restore_aloha(env, state: np.ndarray) -> None:
    """Put the simulator back into a saved physics state (positions and velocities of everything)."""
    ph = physics_of(env)
    with ph.reset_context():
        ph.set_state(np.asarray(state, dtype=float))


def observe(env) -> dict:
    """The observation the policy would get in the simulator's current state."""
    u = env.unwrapped
    return u._format_raw_obs(u._env.task.get_observation(u._env.physics))


class Camera:
    """Projects 3D points into the stored (downscaled) top camera frame."""

    def __init__(self, env, name: str = "top"):
        import mujoco

        ph = physics_of(env)
        cid = mujoco.mj_name2id(ph.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, name)
        self.pos, self.mat, self.fovy = ph.data.cam_xpos[cid].copy(), ph.data.cam_xmat[cid].copy(), float(ph.model.cam_fovy[cid])
        self.width, self.height = 640 // FRAME_STRIDE, 480 // FRAME_STRIDE

    def pixels(self, points: np.ndarray) -> np.ndarray:
        return project_to_camera(points, self.pos, self.mat, self.fovy, self.width, self.height)


class ACTAlohaPolicy:
    def __init__(self, n_samples: int = 6, replan: int = 25):
        import torch
        from huggingface_hub import hf_hub_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from safetensors.torch import load_file

        self.torch = torch
        self.n_samples, self.replan = n_samples, replan
        cfg = PreTrainedConfig.from_pretrained(REPO)
        cfg.pretrained_backbone_weights = None  # the checkpoint already holds the backbone; skip the ImageNet download
        self.policy = ACTPolicy.from_pretrained(REPO, config=cfg)
        self.policy.eval()
        self.model = self.policy.model
        self.chunk = cfg.chunk_size
        self.latent_dim = cfg.latent_dim
        # This checkpoint predates LeRobot's processor files; its normalisation constants are in the weights file.
        sd = load_file(hf_hub_download(REPO, "model.safetensors"))
        g = lambda k: sd[k].float()
        self.img_mean, self.img_std = g("normalize_inputs.buffer_observation_images_top.mean"), g("normalize_inputs.buffer_observation_images_top.std")
        self.s_mean, self.s_std = g("normalize_inputs.buffer_observation_state.mean"), g("normalize_inputs.buffer_observation_state.std")
        self.a_mean, self.a_std = g("unnormalize_outputs.buffer_action.mean"), g("unnormalize_outputs.buffer_action.std")
        # Capture the decoder's cross-attention over the encoder tokens (most of which are image patches).
        self._attn = None
        self.model.decoder.layers[0].multihead_attn.register_forward_hook(lambda _m, _i, out: setattr(self, "_attn", out[1].detach()))

    def reset(self, seed: int) -> None:
        self.torch.manual_seed(seed)
        self.prev_plan: np.ndarray | None = None

    def infer(self, images: np.ndarray, states: np.ndarray, latents=None, repeat: int = 1, attention_steps: int | None = None, dropout: bool = False) -> dict:
        """One batched forward pass.

        images (B, H, W, 3) uint8, states (B, 14). `latents` (B*repeat, latent_dim)
        sets the style variable for each output (default: zeros, the standard
        behaviour). Returns plans (B*repeat, chunk, 14) of joint targets, the
        attention map over the image (B*repeat, h, w) averaged over the first
        `attention_steps` planned steps, and the encoder embedding.
        With dropout=True the transformer's dropout layers stay active, so each
        of the `repeat` outputs is a different sample.
        This mirrors ACT.forward at inference time, with the latent exposed.
        """
        torch, m = self.torch, self.model
        for mod in list(m.encoder.modules()) + list(m.decoder.modules()):
            if isinstance(mod, torch.nn.Dropout):
                mod.train(dropout)
        with torch.no_grad():
            img = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255
            img = (img - self.img_mean) / self.img_std
            state = (torch.from_numpy(np.asarray(states, dtype=np.float32)) - self.s_mean) / self.s_std
            n = len(img) * repeat
            z = torch.zeros(n, self.latent_dim) if latents is None else torch.as_tensor(latents, dtype=torch.float32)
            feat = m.backbone(img)["feature_map"]
            h, w = feat.shape[-2:]
            pos = m.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)
            feat = m.encoder_img_feat_input_proj(feat)
            feat = feat.flatten(2).permute(2, 0, 1).repeat_interleave(repeat, dim=1)  # (hw, n, C)
            pos = pos.flatten(2).permute(2, 0, 1)  # (hw, 1, C)
            tokens = torch.cat([m.encoder_latent_input_proj(z)[None], m.encoder_robot_state_input_proj(state.repeat_interleave(repeat, dim=0))[None], feat])
            pos_embed = torch.cat([m.encoder_1d_feature_pos_embed.weight.unsqueeze(1), pos])
            enc = m.encoder(tokens, pos_embed=pos_embed)
            dec = m.decoder(
                torch.zeros(self.chunk, n, tokens.shape[-1]), enc, encoder_pos_embed=pos_embed, decoder_pos_embed=m.decoder_pos_embed.weight.unsqueeze(1)
            )
            plans = m.action_head(dec.transpose(0, 1)) * self.a_std + self.a_mean
            k = attention_steps or self.replan
            attn = self._attn[:, :k, 2:].mean(dim=1).reshape(n, h, w)  # tokens 0 and 1 are the latent and the robot state
            other = self._attn[:, :k, :2].mean(dim=1)  # attention on the latent and robot-state tokens
        return {"plans": plans.numpy(), "attention": attn.numpy(), "token_attention": other.numpy(), "embedding": enc.mean(dim=0).numpy()}

    def decide(self, obs: dict, kin: Kinematics, cam: Camera) -> dict:
        """Run the model once. Returns the chunk to execute and the internals to trace."""
        image, state = obs["pixels"]["top"][None], obs["agent_pos"][None]
        r = self.infer(image, state)  # plan 0: the standard output
        alt = self.infer(image, state, repeat=self.n_samples - 1, dropout=True)
        self.infer(image[:, :32, :32], state)  # leave dropout off and the hook holding nothing stale
        plans = np.concatenate([r["plans"], alt["plans"]])  # (N, chunk, 14)
        ahead = plans[:, : self.replan]
        shift = float("nan")
        if self.prev_plan is not None:
            overlap = self.chunk - self.replan
            shift = float(np.abs(plans[0][:overlap, ARM_JOINTS] - self.prev_plan[self.replan :][:, ARM_JOINTS]).mean())
        self.prev_plan = plans[0].copy()
        xyz = np.stack([kin.path(p[::PATH_STRIDE, ARM_JOINTS]) for p in ahead])  # (N, T', 12, 3)
        internals = {
            "plans_joint": plans.astype(np.float32),
            "plans_xyz": xyz.astype(np.float32),
            # Gripper paths projected into the camera frame, for drawing on the image: (N, 2 grippers, T', 2)
            "plans": cam.pixels(xyz[:, :, GRIPPERS]).transpose(0, 2, 1, 3).astype(np.float32),
            "modes": plan_modes(ahead[:, :, ARM_JOINTS], MODE_THRESHOLD_RAD),
            "attention": r["attention"][0].astype(np.float32),
            "embedding": r["embedding"][0].astype(np.float32),
        }
        spread = float(ahead[:, :, ARM_JOINTS].std(axis=0).mean())
        return {"plans": plans, "plan_shift": shift, "final_spread": spread, "internals": internals}


def simulate_plans(sim_env, state: np.ndarray, plans: np.ndarray, steps: int, kin: Kinematics, scene: Scene | None = None) -> list[dict]:
    """Predicted future of each candidate plan, from a copy of the simulator.

    Steps the physics directly (no camera rendering), so it is fast. On a real
    robot a learned or physics model would take the simulator's place.
    """
    import mujoco

    u = sim_env.unwrapped
    task, ph = u._env.task, u._env.physics
    box = mujoco.mj_name2id(ph.model.ptr, mujoco.mjtObj.mjOBJ_BODY, "box")
    bodies = kin._bid
    futures = []
    for plan in plans:
        restore_aloha(sim_env, state)
        start = float(task.get_reward(ph))
        arm, cube, best = [ph.data.xpos[bodies].copy()], [ph.data.xpos[box].copy()], start
        for k, a in enumerate(plan[:steps]):
            task.before_step(a, ph)
            ph.step(u._env._n_sub_steps)
            best = max(best, float(task.get_reward(ph)))
            if (k + 1) % PATH_STRIDE == 0:
                arm.append(ph.data.xpos[bodies].copy())
                cube.append(ph.data.xpos[box].copy())
        f = {"reward_after": best, "gain": best - start, "arm_path": np.array(arm), "cube_path": np.array(cube), "contact": False, "risk": 0.0, "clearance": None, "hazard": None}
        if scene and scene.hazards:
            speed = float(np.linalg.norm(np.diff(f["arm_path"], axis=0), axis=-1).max()) / PATH_STRIDE
            a = assess(f["arm_path"][0], f["arm_path"][None, 1:], speed, scene)
            f.update(clearance=a["clearance"], contact=a["time_to_contact"] is not None, risk=a["risk"], hazard=a["hazard"])
        futures.append(f)
    return futures


def rollout(env, policy: ACTAlohaPolicy, tracer: Tracer, seed: int, kin: Kinematics, cam: Camera, sim_env=None, scene: Scene | None = None, shield: bool = False) -> dict:
    obs, info = env.reset(seed=seed)
    policy.reset(seed)
    ph = physics_of(env)
    success, reason, best, reward = False, "timeout", 0.0, None
    with tracer.episode(task=TASK, meta={"seed": seed}) as ep:
        for t in range(MAX_STEPS):
            k = t % policy.replan
            internals = {}
            if k == 0:
                d = policy.decide(obs, kin, cam)
                plans, z = d["plans"], d["internals"]
                extra = {}
                if sim_env is not None:
                    futures = simulate_plans(sim_env, ph.get_state(), plans, policy.replan, kin, scene)
                    pick = choose_plan(futures) if shield else 0
                    extra = {"shield_switched": float(pick != 0), "predicted_gain": futures[pick]["gain"], "plans_in_contact": float(sum(f["contact"] for f in futures))}
                    if pick:
                        order = [pick] + [i for i in range(len(plans)) if i != pick]
                        plans = plans[order]
                        z = {key: (v[order] if key in ("plans_joint", "plans_xyz", "plans") else v) for key, v in z.items()}
                        z["modes"] = plan_modes(plans[:, : policy.replan, ARM_JOINTS], MODE_THRESHOLD_RAD)
                        policy.prev_plan = plans[0].copy()
                sizes = np.bincount(z["modes"])
                signals = {"plan_shift": d["plan_shift"], "n_modes": float(len(sizes)), "top_mode_share": float(sizes[0] / sizes.sum()), "final_spread": d["final_spread"], **extra}
                internals = dict(z)
            joints = obs["agent_pos"].astype(float)
            internals["arm"] = kin.points(joints[ARM_JOINTS]).astype(np.float32)
            decision = ep.step(
                state=joints,
                action=plans[0, k] - joints,  # the policy outputs absolute joint targets; trace them as moves
                action_samples=plans[:, k] - joints,
                frame=obs["pixels"]["top"][::FRAME_STRIDE, ::FRAME_STRIDE],
                belief={"plan_step": k, "stage": ["reaching", "cube grasped", "cube lifted", "handing over", "transferred"][int(reward or 0)]},
                plan_step=int(reward or 0),
                signals=signals,
                reward=reward,
                internals=internals,
                snapshot=ph.get_state(),
            )
            if decision.halt:
                reason = "halted_by_gate"
                break
            obs, reward, terminated, _, info = env.step((joints + decision.action).astype(np.float32))
            best = max(best, float(reward))
            if terminated:
                success, reason = True, "success"
                break
        final = None if reason == "halted_by_gate" else reward
        return ep.end(success=success, reason=reason, meta={"sim": {"best_reward": best, "cause": None}}, final_reward=final)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--replan", type=int, default=25, help="run the model every this many steps")
    ap.add_argument("--model", help="stepscope predictor to score risk live")
    ap.add_argument("--scene", help="YAML scene file with 3D hazards")
    ap.add_argument("--futures", action="store_true", help="predict the future of every candidate plan")
    ap.add_argument("--shield", action="store_true", help="execute the candidate plan with the best predicted future")
    args = ap.parse_args(argv)

    env = make_env()
    env.reset(seed=0)
    kin, cam = make_kinematics(env), Camera(env)
    policy = ACTAlohaPolicy(n_samples=args.samples, replan=args.replan)
    scene = Scene.from_file(args.scene) if args.scene else None
    sim_env = make_env() if (args.futures or args.shield) else None
    if sim_env is not None:
        sim_env.reset(seed=0)
    tracer = Tracer(
        args.run,
        config=TRACE_CONFIG,
        gate=Gate(gate_config(kin), fk=lambda q: kin.points(q[ARM_JOINTS])),
        predictor=Predictor.load(args.model) if args.model else None,
        run_meta={
            "label": f"ACT on ALOHA cube transfer, MuJoCo ({REPO}), replanning every {args.replan} steps, {args.samples} candidate plans",
            "overlay": {"plan_extent": cam.width, "unit": "m"},
            "scene3d": {"links": LINKS, "grippers": GRIPPERS, "table": [[-0.6, 0.0], [0.6, 1.2]], "path_stride": PATH_STRIDE},
            "shield": bool(args.shield),
        },
    )
    done = sum(1 for _ in (Path(args.run) / "episodes").iterdir())  # resume where a previous call stopped
    for i in range(done, args.episodes):
        t0 = time.time()
        s = rollout(env, policy, tracer, args.seed + i, kin, cam, sim_env=sim_env, scene=scene, shield=args.shield)
        print(f"episode {i + 1}/{args.episodes} seed {args.seed + i}: {s['reason']} in {s['n_steps']} steps, best reward {s['meta']['sim']['best_reward']:.0f}/4, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
