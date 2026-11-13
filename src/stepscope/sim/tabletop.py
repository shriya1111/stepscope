"""A small tabletop pick-and-place simulator, in plain numpy.

It exists so the whole stack runs on a laptop with no robot, GPU or model
download. It is NOT a physics engine. What it does have is the set of things
that make real manipulation policies fail in ways you cannot see from the
final outcome alone:

  * perception error that gets worse near the table edge and in bad lighting,
    so the gripper closes next to the object instead of on it
  * objects that get knocked away, sometimes off the table, by a near miss
  * a slippery grasp that drops the object in transit

State vector (9): ee x, y, z | gripper closed | perceived object x, y |
                  target x, y | holding
Action vector (4): dx, dy, dz | gripper command (>0.5 = close)
"""

from __future__ import annotations

import numpy as np

from ..metrics import TraceConfig

MAX_STEPS = 90
TABLE_LOW = np.array([0.0, 0.0, 0.0])
TABLE_HIGH = np.array([1.0, 1.0, 0.3])
KEEP_OUT_LOW = np.array([0.0, 0.70, 0.0])  # operator hand-off corner
KEEP_OUT_HIGH = np.array([0.24, 1.0, 0.3])
GRASP_RADIUS = 0.025
PLACE_TOLERANCE = 0.05
IMAGE_SIZE = 96

TRACE_CONFIG = TraceConfig(position_dims=[0, 1, 2], motion_dims=[0, 1, 2], max_steps=MAX_STEPS)

GATE_CONFIG = {
    "position_dims": [0, 1, 2],
    "delta_dims": [0, 1, 2],
    "constraints": [
        {"type": "max_step", "name": "speed_limit", "max_norm": 0.05},
        {"type": "workspace_box", "name": "workspace", "low": TABLE_LOW.tolist(), "high": TABLE_HIGH.tolist()},
        {
            "type": "keep_out_box",
            "name": "operator_zone",
            "low": KEEP_OUT_LOW.tolist(),
            "high": KEEP_OUT_HIGH.tolist(),
        },
    ],
}


def _in_keep_out(xy: np.ndarray, margin: float = 0.0) -> bool:
    return bool(
        KEEP_OUT_LOW[0] - margin < xy[0] < KEEP_OUT_HIGH[0] + margin
        and KEEP_OUT_LOW[1] - margin < xy[1] < KEEP_OUT_HIGH[1] + margin
    )


class TabletopEnv:
    def __init__(self, max_steps: int = MAX_STEPS, difficulty: float = 1.0):
        self.max_steps = max_steps
        self.difficulty = difficulty

    # ---------- episode setup ----------
    def reset(self, seed: int) -> dict:
        self.rng = rng = np.random.default_rng(seed)
        self.t = 0
        self.ee = np.array([0.5, 0.08, 0.2])
        self.gripper_closed = False
        self.holding = False

        def sample_xy() -> np.ndarray:
            while True:
                xy = rng.uniform([0.10, 0.30], [0.90, 0.90])
                if not _in_keep_out(xy, margin=0.04):
                    return xy

        self.obj = sample_xy()
        while True:
            self.target = sample_xy()
            if np.linalg.norm(self.target - self.obj) >= 0.25:
                break

        d = self.difficulty
        # Perception gets worse toward the table edge and under bad lighting.
        edge = max(0.0, float(np.max(np.abs(self.obj - np.array([0.5, 0.6])))) - 0.22) / 0.18
        lighting = float(rng.lognormal(0.0, 0.45))
        self.perception_sigma = 0.0095 * lighting * (1.0 + 2.2 * edge) * d
        self.bias = rng.normal(0, self.perception_sigma, 2)
        self.slip = float(rng.uniform(0.06, 0.16)) * d if rng.random() < 0.15 else 0.002 * d
        self.events = {"grasp_misses": 0, "knocks": 0, "drops": 0}
        self.failure = None
        return self._obs()

    # ---------- dynamics ----------
    def step(self, action: np.ndarray):
        rng = self.rng
        action = np.asarray(action, dtype=float)
        delta = action[:3]
        norm = np.linalg.norm(delta)
        if norm > 0.06:  # actuator limit
            delta = delta * 0.06 / norm
        self.ee = np.clip(self.ee + delta, TABLE_LOW, TABLE_HIGH)
        speed = float(np.linalg.norm(delta))
        want_closed = action[3] > 0.5
        terminated, success = False, False

        if self.holding:
            if rng.random() < self.slip * (speed / 0.04) ** 2:  # slipped out of the gripper
                self.holding = False
                self.obj = self.ee[:2] + rng.normal(0, 0.02, 2)
                self.events["drops"] += 1
            elif not want_closed:  # deliberate release
                self.holding = False
                self.obj = self.ee[:2] + rng.normal(0, 0.15 * self.ee[2], 2)
                if np.linalg.norm(self.obj - self.target) <= PLACE_TOLERANCE:
                    terminated, success = True, True
            else:
                self.obj = self.ee[:2].copy()
        elif want_closed and not self.gripper_closed and self.ee[2] < 0.035:
            err = float(np.linalg.norm(self.ee[:2] - self.obj))
            if err < GRASP_RADIUS:
                self.holding = True
                self.obj = self.ee[:2].copy()
            else:
                self.events["grasp_misses"] += 1
                if err < 2.4 * GRASP_RADIUS:  # fingers clip the object and shove it away
                    away = (self.obj - self.ee[:2]) / max(err, 1e-9)
                    self.obj = self.obj + away * 0.04 * rng.uniform(0.5, 1.5)
                    self.events["knocks"] += 1
                # A fresh look after a miss: part of the old error persists.
                self.bias = 0.5 * self.bias + rng.normal(0, self.perception_sigma * 0.85, 2)
        self.gripper_closed = bool(want_closed)

        if not self.holding and (np.any(self.obj < 0.03) or np.any(self.obj > 0.97)):
            terminated, self.failure = True, "object_off_table"
        self.t += 1
        truncated = not terminated and self.t >= self.max_steps
        if truncated:
            self.failure = "timeout"
        info = {"success": success, "failure": self.failure, "events": dict(self.events)}
        return self._obs(), float(success), terminated, truncated, info

    # ---------- observation ----------
    def _obs(self) -> dict:
        if self.holding:
            seen = self.ee[:2].copy()
        else:
            seen = self.obj + self.bias + self.rng.normal(0, 0.0015, 2)
        state = np.concatenate(
            [self.ee, [float(self.gripper_closed)], seen, self.target, [float(self.holding)]]
        )
        return {"state": state, "image": self.render()}

    def ground_truth(self) -> dict:
        return {
            "perception_sigma": self.perception_sigma,
            "slip": self.slip,
            "object": self.obj.tolist(),
            "target": self.target.tolist(),
            **self.events,
        }

    def render(self) -> np.ndarray:
        """Top-down camera view."""
        n = IMAGE_SIZE
        img = np.full((n, n, 3), (236, 233, 226), np.uint8)

        def px(xy):
            return int(round(xy[0] * (n - 1))), int(round((1 - xy[1]) * (n - 1)))

        x0, y1 = px(KEEP_OUT_LOW[:2])
        x1, y0 = px(KEEP_OUT_HIGH[:2])
        img[y0 : y1 + 1, x0 : x1 + 1] = (224, 196, 190)
        img[y0 : y1 + 1 : 3, x0 : x1 + 1] = (205, 160, 152)

        yy, xx = np.mgrid[0:n, 0:n]

        def disc(xy, r, color, ring=False):
            cx, cy = px(xy)
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            mask = (d2 <= r * r) & (d2 >= (r - 1.6) ** 2) if ring else d2 <= r * r
            img[mask] = color

        disc(self.target, 6, (40, 130, 60), ring=True)
        ox, oy = px(self.obj)
        img[max(oy - 3, 0) : oy + 4, max(ox - 3, 0) : ox + 4] = (42, 120, 214)
        r = 3.5 + 12 * self.ee[2]  # higher gripper looks bigger
        disc(self.ee[:2], r, (30, 30, 30) if self.gripper_closed else (110, 110, 110), ring=True)
        disc(self.ee[:2], 1.2, (30, 30, 30))
        return img
