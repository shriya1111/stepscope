"""A stand-in for a learned policy, with the pathologies learned policies have.

It is a scripted pick-and-place controller that, like a diffusion policy or a
VLA, can be asked for several action samples per step. On top of the scripted
behaviour it has three failure-prone quirks, drawn per episode:

  * indecision: its samples split between two detour directions, so the action
    it executes zig-zags and makes slow progress
  * freezes: for a stretch of steps the samples collapse to near-zero motion
  * uncertainty that grows when the object is near the table edge

The policy only sees the (noisy) observation, never simulator ground truth.
"""

from __future__ import annotations

import numpy as np

PHASES = ["reach", "descend", "grasp", "lift", "transport", "lower", "release"]
Z_HOVER, Z_GRASP, MAX_SPEED = 0.14, 0.02, 0.04


class NoisyScriptedPolicy:
    def __init__(self, n_samples: int = 8, difficulty: float = 1.0):
        self.n_samples = n_samples
        self.difficulty = difficulty

    def reset(self, seed: int) -> None:
        self.rng = rng = np.random.default_rng(seed + 7_919)
        d = self.difficulty
        self.phase = "reach"
        self.retries = 0
        self.indecision = float(np.deg2rad(rng.uniform(50, 80))) if rng.random() < 0.12 * d else 0.0
        self.freeze_hazard = 0.05 if rng.random() < 0.16 * d else 0.0015
        self.freeze_left = 0
        self.frozen_steps = 0
        self.zigzag_steps = 0

    def ground_truth(self) -> dict:
        return {
            "indecision_deg": float(np.rad2deg(self.indecision)),
            "freeze_hazard": self.freeze_hazard,
            "frozen_steps": self.frozen_steps,
            "zigzag_steps": self.zigzag_steps,
            "retries": self.retries,
        }

    def sample_actions(self, obs: dict) -> tuple[np.ndarray, dict]:
        """Returns (samples [N, 4], belief). The policy executes samples[0]."""
        rng = self.rng
        s = obs["state"]
        ee, obj, target, holding = s[0:3], s[4:6], s[6:8], s[8] > 0.5

        # ----- plan: which subtask am I on? -----
        if self.phase in ("lift", "transport", "lower") and not holding:
            self.phase, self.retries = "reach", self.retries + 1  # dropped it
        elif self.phase == "reach":
            if np.linalg.norm(ee[:2] - obj) < 0.012 and abs(ee[2] - Z_HOVER) < 0.03:
                self.phase = "descend"
        elif self.phase == "descend":
            if ee[2] < Z_GRASP + 0.01:
                self.phase = "grasp"
        elif self.phase == "grasp" and s[3] > 0.5:  # gripper has closed
            if holding:
                self.phase = "lift"
            else:
                self.phase, self.retries = "reach", self.retries + 1  # missed
        elif self.phase == "lift":
            if ee[2] > Z_HOVER - 0.03:
                self.phase = "transport"
        elif self.phase == "transport":
            if np.linalg.norm(ee[:2] - target) < 0.012:
                self.phase = "lower"
        elif self.phase == "lower":
            if ee[2] < Z_GRASP + 0.01:
                self.phase = "release"
        elif self.phase == "release" and s[3] < 0.5:
            self.phase, self.retries = "reach", self.retries + 1  # placed off target

        goal = {
            "reach": np.array([obj[0], obj[1], Z_HOVER]),
            "descend": np.array([obj[0], obj[1], Z_GRASP]),
            "grasp": ee.copy(),
            "lift": np.array([ee[0], ee[1], Z_HOVER]),
            "transport": np.array([target[0], target[1], Z_HOVER]),
            "lower": np.array([target[0], target[1], Z_GRASP]),
            "release": ee.copy(),
        }[self.phase]
        grip = 1.0 if self.phase in ("grasp", "lift", "transport", "lower") else 0.0

        # ----- action: mean command plus the policy's quirks -----
        to_goal = goal - ee
        dist = float(np.linalg.norm(to_goal))
        mean = to_goal if dist <= MAX_SPEED else to_goal * MAX_SPEED / dist
        moving = self.phase in ("reach", "transport")

        if self.freeze_left > 0:
            self.freeze_left -= 1
        elif moving and rng.random() < self.freeze_hazard:
            self.freeze_left = int(rng.integers(12, 40))
        frozen = self.freeze_left > 0 and self.phase not in ("grasp", "release")

        edge = max(0.0, float(np.max(np.abs(obj - np.array([0.5, 0.6])))) - 0.22) / 0.18
        sigma = 0.003 * (1.0 + 1.5 * edge)
        n = self.n_samples
        samples = np.zeros((n, 4))
        samples[:, 3] = grip
        if frozen:
            samples[:, :3] = rng.normal(0, 0.006, (n, 3))
            self.frozen_steps += 1
        else:
            samples[:, :3] = mean + rng.normal(0, sigma, (n, 3))
            if self.indecision and moving and dist > 0.06:
                signs = rng.choice([-1.0, 1.0], n)
                c, sn = np.cos(self.indecision), np.sin(self.indecision) * signs
                x, y = samples[:, 0].copy(), samples[:, 1].copy()
                samples[:, 0], samples[:, 1] = c * x - sn * y, sn * x + c * y
                self.zigzag_steps += 1

        belief = {
            "subtask": self.phase,
            "object_estimate": [round(float(obj[0]), 3), round(float(obj[1]), 3)],
            "holding": bool(holding),
            "retries": self.retries,
        }
        return samples, belief

    @property
    def plan_step(self) -> int:
        return PHASES.index(self.phase)
