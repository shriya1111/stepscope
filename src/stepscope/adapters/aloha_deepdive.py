"""Deep dives into single decisions of the ACT policy on the ALOHA cube-transfer task.

    MUJOCO_GL=egl python -m stepscope.adapters.aloha_deepdive --run runs/aloha --auto 3 --every 2 \\
        --scene examples/aloha_scene.yaml

For each chosen decision the saved physics state is restored, the real model is
re-run, and these are stored in <episode>/deepdive.json:

  candidates  the standard plan and dropout alternatives (ACT plans in one pass,
              so there are no intermediate steps to scrub through)
  cause       how far the grippers' planned path moves when each patch of the
              camera image is hidden, and when a whole input is removed
  whatif      the cube is moved a few centimetres (or taken away), the model is
              re-run, and the new plan is recorded
  futures     each candidate plan is played forward in the simulator: predicted
              task progress and closest approach to the scene's hazards

Distances are in metres, measured at the two grippers.
"""

from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import numpy as np

from ..deepdive import occlusion_batch, write_deepdive
from ..safety import Scene, choose_plan
from ..store import encode_png, list_episodes, read_internals
from .lerobot_aloha import (
    ARM_JOINTS,
    FRAME_STRIDE,
    GRIPPERS,
    PATH_STRIDE,
    ACTAlohaPolicy,
    Camera,
    make_env,
    make_kinematics,
    observe,
    physics_of,
    restore_aloha,
    simulate_plans,
)

GRID = (6, 8)  # rows, cols of image patches for the cause map
CUBE_QPOS = slice(16, 19)  # the cube's x, y, z inside the physics state
WHAT_IFS = [
    ("Cube 4 cm toward the right arm", (0.04, 0.0)),
    ("Cube 4 cm toward the left arm", (-0.04, 0.0)),
    ("Cube 4 cm farther away", (0.0, 0.04)),
    ("Cube 4 cm closer", (0.0, -0.04)),
    ("Cube taken away", None),
]


def _uri(frame: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(frame[::FRAME_STRIDE, ::FRAME_STRIDE])).decode()


def analyse_decision(policy: ACTAlohaPolicy, env, sim_env, kin, cam: Camera, state: np.ndarray, t: int, scene: Scene | None) -> dict:
    steps = policy.replan
    restore_aloha(env, state)
    obs = observe(env)
    image, joints = obs["pixels"]["top"], obs["agent_pos"]

    def gripper_paths(plans: np.ndarray) -> np.ndarray:
        """(B, T', 2 grippers, 3): where each plan takes the two grippers."""
        return np.stack([kin.path(p[:steps:PATH_STRIDE, ARM_JOINTS])[:, GRIPPERS] for p in plans])

    def change(paths: np.ndarray, base: np.ndarray) -> np.ndarray:
        return np.linalg.norm(paths - base[None], axis=-1).mean(axis=(1, 2))

    def pixels(paths: np.ndarray) -> np.ndarray:
        return cam.pixels(paths).transpose(0, 2, 1, 3)  # (B, grippers, T', 2)

    # --- baseline and candidates ---
    policy.torch.manual_seed(int(t))
    base = policy.infer(image[None], joints[None])
    alt = policy.infer(image[None], joints[None], repeat=policy.n_samples - 1, dropout=True)
    plans = np.concatenate([base["plans"], alt["plans"]])
    paths = gripper_paths(plans)

    # --- cause map: hide each image patch; remove each input ---
    fill = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)  # the table colour, so a hidden patch looks empty
    occ = occlusion_batch(image[None], GRID, fill=fill)[:, 0]
    blank = np.broadcast_to(fill, image.shape)
    variants = np.concatenate([occ, blank[None], image[None]])
    states = np.repeat(joints[None], len(variants), axis=0)
    states[-1] = policy.s_mean.numpy()  # robot state replaced by the training-set average pose
    r = policy.infer(variants, states)
    moved = change(gripper_paths(r["plans"]), paths[0])
    cause = {
        "grid": list(GRID),
        "current_frame": moved[: len(occ)].reshape(GRID).tolist(),
        "inputs": [
            {"label": "Camera image (blanked)", "change": float(moved[len(occ)])},
            {"label": "Joint positions (set to an average pose)", "change": float(moved[len(occ) + 1])},
        ],
    }

    # --- what-if: move the cube ---
    whatif = []
    for label, delta in WHAT_IFS:
        edited = np.array(state, dtype=float)
        if delta is None:
            edited[CUBE_QPOS] = [0.0, 1.15, 0.05]  # far corner of the table, out of the working area
        else:
            edited[CUBE_QPOS][:2] += delta
        restore_aloha(env, edited)
        o = observe(env)
        new = gripper_paths(policy.infer(o["pixels"]["top"][None], o["agent_pos"][None])["plans"])
        whatif.append({"label": label, "kind": "cube", "frame": _uri(o["pixels"]["top"]), "plan": np.round(pixels(new)[0], 1).tolist(), "shift": float(change(new, paths[0])[0])})
    policy.infer(image[None, :32, :32], joints[None])  # leave dropout off

    # --- predicted futures ---
    futures = simulate_plans(sim_env, state, plans, steps, kin, scene)
    out_futures = [
        {
            "gain": f["gain"],
            "reward_after": f["reward_after"],
            "contact": f["contact"],
            "clearance": f["clearance"],
            "hazard": f["hazard"],
            "risk": f["risk"],
            "agent_path": np.round(cam.pixels(f["arm_path"][:, GRIPPERS]).transpose(1, 0, 2), 1).tolist(),
            "block_path": np.round(cam.pixels(f["cube_path"]), 1).tolist(),
        }
        for f in futures
    ]
    return {
        "frame": _uri(image),
        "plans": np.round(pixels(paths), 1).tolist(),
        "stages": None,
        "candidate_spread": float(change(paths[1:], paths[0]).mean()),
        "attention": np.round(base["attention"][0], 4).tolist(),
        "cause": cause,
        "whatif": whatif,
        "futures": out_futures,
        "safest": choose_plan(futures),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", nargs="*", help="episode ids to analyse")
    ap.add_argument("--auto", type=int, default=0, help="or: pick this many failed and this many successful episodes")
    ap.add_argument("--every", type=int, default=1, help="analyse every k-th decision")
    ap.add_argument("--scene")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--replan", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="redo episodes that already have a deep dive")
    args = ap.parse_args(argv)

    run = Path(args.run)
    metas = [m for m in list_episodes(run) if "snapshot" in (m.get("internals") or [])]
    chosen = list(args.episodes or [])
    if args.auto:
        chosen += [m["episode_id"] for m in metas if m["success"] is False][: args.auto]
        chosen += [m["episode_id"] for m in metas if m["success"] is True][: args.auto]
    scene = Scene.from_file(args.scene) if args.scene else None
    env, sim_env = make_env(), make_env()
    env.reset(seed=0)
    sim_env.reset(seed=0)
    kin, cam = make_kinematics(env), Camera(env)
    policy = ACTAlohaPolicy(n_samples=args.samples, replan=args.replan)
    for ep in chosen:
        if (run / "episodes" / ep / "deepdive.json").exists() and not args.force:
            print(f"{ep}: already analysed, skipping", flush=True)
            continue
        z = read_internals(run, ep)
        steps = z["plans__t"][:: args.every].tolist()
        t0 = time.time()
        out = {t: analyse_decision(policy, env, sim_env, kin, cam, z["snapshot"][t], t, scene) for t in steps}
        write_deepdive(run / "episodes" / ep, out)
        print(f"{ep}: {len(steps)} decisions analysed in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
