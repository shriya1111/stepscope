"""Turning model internals into things an operator can read.

  * plan_modes: are the policy's sampled plans one idea or several?
  * SituationIndex: where does this moment sit among all traced moments, and
    how did the most similar past moments end?
  * describe_decision: one plain sentence per decision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .store import list_episodes, read_internals


def plan_modes(plans: np.ndarray, threshold: float) -> np.ndarray:
    """Group sampled plans into modes.

    plans: (N, H, D) sampled action sequences. Two plans are the same mode when
    their mean point-to-point distance is under `threshold` (or they are linked
    through other plans that are). Returns a mode id per plan; mode 0 is always
    the mode of plans[0], the plan that gets executed.
    """
    plans = np.asarray(plans, dtype=float)
    n = len(plans)
    dist = np.linalg.norm(plans[:, None] - plans[None], axis=-1).mean(axis=-1)
    labels = -np.ones(n, dtype=int)
    current = 0
    for seed in range(n):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = current
        while stack:
            i = stack.pop()
            for j in np.where((dist[i] < threshold) & (labels < 0))[0]:
                labels[j] = current
                stack.append(j)
        current += 1
    return labels


def commit_step(spread_per_denoise_step: np.ndarray, tolerance: float = 1.25) -> int:
    """First denoising step at which the samples are as close together as they end up."""
    s = np.asarray(spread_per_denoise_step, dtype=float)
    return int(np.argmax(s <= s[-1] * tolerance))


class SituationIndex:
    """All decision-time embeddings of a run, for a 2-D map and look-alike search."""

    def __init__(self, run_dir: str | Path, key: str = "embedding"):
        self.episodes, self.steps, self.success, vecs = [], [], [], []
        for meta in list_episodes(Path(run_dir)):
            z = read_internals(Path(run_dir), meta["episode_id"])
            if key not in z:
                continue
            v = z[key].reshape(len(z[key]), -1)
            vecs.append(v)
            self.episodes += [meta["episode_id"]] * len(v)
            self.steps += z[f"{key}__t"].tolist()
            self.success += [meta["success"]] * len(v)
        self.empty = not vecs
        if self.empty:
            return
        X = np.concatenate(vecs).astype(float)
        self.mean, self.scale = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.Z = (X - self.mean) / self.scale
        # PCA via SVD: the two directions along which situations differ most.
        _, _, vt = np.linalg.svd(self.Z - self.Z.mean(axis=0), full_matrices=False)
        self.xy = self.Z @ vt[:2].T
        self.episodes = np.array(self.episodes)
        self.steps = np.array(self.steps)

    def points(self) -> list[dict]:
        if self.empty:
            return []
        return [
            {"ep": e, "t": int(t), "x": float(x), "y": float(y), "success": s}
            for e, t, (x, y), s in zip(self.episodes, self.steps, self.xy, self.success)
        ]

    def neighbours(self, episode_id: str, k: int = 5) -> dict[int, list[dict]]:
        """For each decision in an episode: the k most similar moments from OTHER episodes."""
        if self.empty:
            return {}
        mine = np.where(self.episodes == episode_id)[0]
        other = np.where(self.episodes != episode_id)[0]
        out: dict[int, list[dict]] = {}
        if not len(other):
            return out
        for i in mine:
            d = np.linalg.norm(self.Z[other] - self.Z[i], axis=1)
            best = other[np.argsort(d)[:k]]
            out[int(self.steps[i])] = [
                {"ep": str(self.episodes[j]), "t": int(self.steps[j]), "success": self.success[j]} for j in best
            ]
        return out


def describe_decision(n_plans: int, mode_sizes: list[int], final_spread: float | None, neighbours: list[dict] | None) -> str:
    """One plain sentence about a decision, from the pieces above."""
    parts = []
    if len(mode_sizes) <= 1:
        parts.append(f"All {n_plans} sampled plans agree")
    else:
        split = " vs ".join(str(s) for s in sorted(mode_sizes, reverse=True))
        parts.append(f"Torn between {len(mode_sizes)} options ({split} of {n_plans} sampled plans)")
    if final_spread is not None:
        parts.append(f"plans differ by about {final_spread:.0f} px after denoising")
    if neighbours:
        failed = sum(nb["success"] is False for nb in neighbours)
        parts.append(f"{failed} of the {len(neighbours)} most similar past moments were in episodes that failed")
    return "; ".join(parts) + "."
