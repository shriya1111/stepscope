"""Model-agnostic pieces of a "deep dive" into one decision.

The adapter for a specific model supplies the forward passes; these helpers
build the perturbations and turn the outputs into something readable.
Results are stored per episode in deepdive.json, keyed by decision step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .store import dump_json


def occlusion_batch(frames: np.ndarray, grid: int, fill: int = 255) -> np.ndarray:
    """Every way of hiding one cell of a grid x grid tiling in each frame.

    frames: (F, H, W, 3). Returns (F * grid * grid, F, H, W, 3): variant
    f*grid*grid + gy*grid + gx hides cell (gy, gx) of frame f only.
    """
    F, H, W, _ = frames.shape
    out = np.repeat(frames[None], F * grid * grid, axis=0).copy()
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    i = 0
    for f in range(F):
        for gy in range(grid):
            for gx in range(grid):
                out[i, f, ys[gy] : ys[gy + 1], xs[gx] : xs[gx + 1]] = fill
                i += 1
    return out


def plan_change(plans: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Mean point-to-point distance of each plan (B, T, 2) from a baseline plan (T, 2)."""
    return np.linalg.norm(np.asarray(plans) - np.asarray(baseline)[None], axis=-1).mean(axis=-1)


def formed_at(stage_plans: np.ndarray, tolerance: float) -> int:
    """First stage whose predicted plan is already within `tolerance` of the final one.

    stage_plans: (S, T, 2), the plan the model would output if it stopped after each stage.
    """
    d = np.linalg.norm(stage_plans - stage_plans[-1][None], axis=-1).mean(axis=-1)
    return int(np.argmax(d <= tolerance))


def turn_degrees(plan: np.ndarray, baseline: np.ndarray, origin: np.ndarray) -> float:
    """Angle between where two plans head (their endpoints, seen from the robot)."""
    a, b = np.asarray(plan)[-1] - origin, np.asarray(baseline)[-1] - origin
    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
        return 0.0
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def write_deepdive(ep_dir: Path, decisions: dict[int, dict]) -> None:
    path = Path(ep_dir) / "deepdive.json"
    existing = json.loads(path.read_text())["decisions"] if path.exists() else {}
    existing.update({str(t): d for t, d in decisions.items()})
    dump_json(path, {"decisions": existing})


def read_deepdive(ep_dir: Path, compact: bool = False) -> dict | None:
    """compact=True thins the per-stage plans (whole units, every other point) for sending to a browser."""
    path = Path(ep_dir) / "deepdive.json"
    if not path.exists():
        return None
    decisions = json.loads(path.read_text())["decisions"]
    if compact:
        for d in decisions.values():
            if "stages" in d:
                d["stages"] = np.round(np.asarray(d["stages"])[:, :, ::2]).astype(int).tolist()
    return decisions
