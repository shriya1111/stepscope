"""Deep dives into single decisions of the LeRobot diffusion policy on PushT.

    python -m stepscope.adapters.pusht_deepdive --run runs/pusht --auto 3 --every 2 \\
        --scene examples/pusht_scene.yaml

For each chosen decision the saved environment state is restored, the real
model is re-run, and four analyses are stored in <episode>/deepdive.json:

  stages   decision debugger: the plan the model would output if it stopped
           after each denoising step, and the step at which the plan was formed
  cause    cause map: how far the plan moves when each patch of the image is
           hidden, and when each input (camera, robot position, motion) is removed
  whatif   what-if console: the scene is edited (block moved or turned, pusher
           moved), the model is re-run, and the new plan is recorded
  futures  each sampled plan is played forward in the simulator: predicted
           progress, closest approach to hazards, and which plan is safest

Every re-run uses the same starting noise, so differences come from the inputs.
"""

from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import numpy as np

from ..deepdive import formed_at, occlusion_batch, plan_change, turn_degrees, write_deepdive
from ..safety import Scene, choose_plan
from ..store import encode_png, list_episodes, read_internals
from .lerobot_pusht import DiffusionPushTPolicy, restore_pusht, simulate_plans

GRID = 6
FORMED_TOLERANCE_PX = 10.0
WHAT_IFS = [
    ("Block 40 px left", "block", (-40, 0, 0)),
    ("Block 40 px right", "block", (40, 0, 0)),
    ("Block 40 px up", "block", (0, -40, 0)),
    ("Block 40 px down", "block", (0, 40, 0)),
    ("Block turned 25° one way", "block", (0, 0, np.radians(25))),
    ("Block turned 25° the other way", "block", (0, 0, -np.radians(25))),
    ("Pusher 40 px left", "pusher", (-40, 0)),
    ("Pusher 40 px right", "pusher", (40, 0)),
    ("Pusher 40 px up", "pusher", (0, -40)),
    ("Pusher 40 px down", "pusher", (0, 40)),
]


def _edit(state: np.ndarray, kind: str, delta) -> np.ndarray:
    s = np.array(state, dtype=float)
    if kind == "block":
        s[2:4] = np.clip(s[2:4] + delta[:2], 40, 472)
        s[4] += delta[2]
    else:
        s[0:2] = np.clip(s[0:2] + delta, 20, 492)
    return s


def _observe(env, prev_state: np.ndarray, state: np.ndarray):
    """Camera frames and pusher positions the policy would see for these two consecutive states."""
    a, _ = restore_pusht(env, prev_state)
    b, _ = restore_pusht(env, state)
    return np.stack([a["pixels"], b["pixels"]]), np.stack([a["agent_pos"], b["agent_pos"]]).astype(np.float32)


def _uri(frame: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(frame)).decode()


def analyse_decision(policy: DiffusionPushTPolicy, env, sim_env, snapshots: np.ndarray, t: int, scene: Scene | None) -> dict:
    torch = policy.torch
    now, prev = snapshots[t], snapshots[max(t - 1, 0)]
    pixels, pos = _observe(env, prev, now)
    n = policy.n_samples
    noise = torch.randn(n, policy.policy.config.horizon, 2, generator=torch.Generator().manual_seed(int(t)))
    one = noise[:1]

    # --- baseline + decision debugger ---
    base = policy.infer(pixels[None], pos[None], noise=noise, repeat=n, keep_stages=True)
    plans, plan = base["plans"], base["plans"][0]
    stages = base["stages"]  # (S, N, H, 2)
    stage_dist = np.linalg.norm(stages[:, 0] - plan[None], axis=-1).mean(axis=-1)

    # --- cause map: hide each image patch, and remove each input ---
    occ = occlusion_batch(pixels, GRID)
    blank = np.full_like(pixels, 255)
    variants = np.concatenate([occ, blank[None], pixels[None], np.stack([pixels[1], pixels[1]])[None]])
    positions = np.repeat(pos[None], len(variants), axis=0)
    positions[len(occ) + 1] = 256.0  # robot position replaced by the table centre
    positions[len(occ) + 2] = pos[1]  # no motion: previous observation = current one
    r = policy.infer(variants, positions, noise=one.repeat(len(variants), 1, 1))
    change = plan_change(r["plans"], plan)
    cells = change[: len(occ)].reshape(2, GRID, GRID)
    cause = {
        "grid": GRID,
        "previous_frame": cells[0].tolist(),
        "current_frame": cells[1].tolist(),
        "inputs": [
            {"label": "Camera image (blanked)", "change": float(change[len(occ)])},
            {"label": "Robot position (set to table centre)", "change": float(change[len(occ) + 1])},
            {"label": "Motion (previous frame replaced by current)", "change": float(change[len(occ) + 2])},
        ],
    }

    # --- what-if console ---
    edited = [(_edit(prev, kind, np.array(d, float)), _edit(now, kind, np.array(d, float))) for _, kind, d in WHAT_IFS]
    obs = [_observe(env, p, s) for p, s in edited]
    r = policy.infer(np.stack([o[0] for o in obs]), np.stack([o[1] for o in obs]), noise=one.repeat(len(obs), 1, 1))
    whatif = []
    for (label, kind, _), (_, s), o, new in zip(WHAT_IFS, edited, obs, r["plans"]):
        whatif.append(
            {
                "label": label,
                "kind": kind,
                "frame": _uri(o[0][1]),
                "plan": np.round(new, 1).tolist(),
                "shift": float(plan_change(new[None], plan)[0]),
                "turn": turn_degrees(new, plan, s[:2]) if kind == "block" else turn_degrees(new - s[:2], plan - now[:2], np.zeros(2)),
            }
        )

    # --- predicted futures ---
    start = policy.n_obs - 1
    futures = simulate_plans(sim_env, now, plans[:, start : start + policy.n_exec], scene)
    out_futures = [
        {
            "gain": f["gain"],
            "reward_after": f["reward_after"],
            "contact": f["contact"],
            "clearance": f["clearance"],
            "hazard": f["hazard"],
            "risk": f["risk"],
            "agent_path": np.round(f["agent_path"], 1).tolist(),
            "block_path": np.round(f["block_path"], 1).tolist(),
        }
        for f in futures
    ]
    return {
        "frame": _uri(pixels[1]),
        "plans": np.round(plans, 1).tolist(),
        "stages": np.round(stages, 1).tolist(),
        "stage_distance": stage_dist.tolist(),
        "formed_at": formed_at(stages[:, 0], FORMED_TOLERANCE_PX),
        "keypoints": policy.keypoints_px().tolist(),
        "cause": cause,
        "whatif": whatif,
        "futures": out_futures,
        "safest": choose_plan(futures),
    }


def main(argv: list[str] | None = None) -> None:
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", nargs="*", help="episode ids to analyse")
    ap.add_argument("--auto", type=int, default=0, help="or: pick this many failed and this many successful episodes")
    ap.add_argument("--every", type=int, default=2, help="analyse every k-th decision")
    ap.add_argument("--scene")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--ddim-steps", type=int, default=8)
    args = ap.parse_args(argv)

    run = Path(args.run)
    metas = [m for m in list_episodes(run) if "snapshot" in (m.get("internals") or [])]
    chosen = list(args.episodes or [])
    if args.auto:
        chosen += [m["episode_id"] for m in metas if m["success"] is False][: args.auto]
        chosen += [m["episode_id"] for m in metas if m["success"] is True][: args.auto]
    scene = Scene.from_file(args.scene) if args.scene else None
    policy = DiffusionPushTPolicy(n_samples=args.samples, ddim_steps=args.ddim_steps)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=10_000)
    sim_env = gym.make("gym_pusht/PushT-v0", obs_type="state", max_episode_steps=10_000)
    for ep in chosen:
        z = read_internals(run, ep)
        steps = z["plans__t"][:: args.every].tolist()
        t0 = time.time()
        out = {t: analyse_decision(policy, env, sim_env, z["snapshot"], t, scene) for t in steps}
        write_deepdive(run / "episodes" / ep, out)
        print(f"{ep}: {len(steps)} decisions analysed in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
