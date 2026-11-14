"""Pre-actuation gate: declarative constraints checked before every action.

The gate never trusts the policy. Each step it predicts where the proposed
action would take the robot and returns a Decision: allow the action, clamp it
to something safe, block it (hold still), or halt the episode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VERDICT_ORDER = {"allow": 0, "clamp": 1, "block": 2, "halt": 3}


@dataclass
class Decision:
    """What the gate decided for one step. Send `action` to the robot."""

    action: np.ndarray
    verdict: str = "allow"  # allow | clamp | block | halt
    violations: list[dict[str, Any]] = field(default_factory=list)
    risk: float = float("nan")
    shadow: bool = False  # True: the verdict is advisory and `action` is the policy's own action

    @property
    def halt(self) -> bool:
        """True when the gate says to stop the episode (never in shadow mode)."""
        return self.verdict == "halt" and not self.shadow

    @property
    def intervened(self) -> bool:
        return self.verdict != "allow"


def _inside(p: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    return bool(np.all(p > low) and np.all(p < high))


class Gate:
    """Checks a proposed action against a list of constraints.

    Config (dict, or a YAML/JSON file):

        position_dims: [0, 1, 2]   # state dims that are the position
        delta_dims: [0, 1, 2]      # action dims that are position deltas
        constraints:
          - {type: max_step, max_norm: 0.05}
          - {type: workspace_box, low: [0, 0, 0], high: [1, 1, 0.3]}
          - {type: keep_out_box, name: operator_zone, low: [...], high: [...]}
          - {type: risk_limit, max_risk: 0.9, sustain: 5}

    For arms driven by joint targets (position_dims/delta_dims are then the joints):

          - {type: max_joint_step, max_abs: 0.1}
          - {type: joint_limits, low: [...], high: [...]}
          - {type: cartesian_box, low: [x, y, z], high: [x, y, z]}          # needs fk
          - {type: cartesian_keep_out_sphere, center: [x, y, z], radius: r}  # needs fk
    """

    def __init__(self, config: dict, fk=None):
        """fk: for the cartesian_* constraints, a function mapping the next
        joint vector to the (P, 3) positions of the robot's tracked points
        (see stepscope.kinematics.Kinematics.points)."""
        self.config = config
        self.fk = fk
        self.position_dims = list(config.get("position_dims", []))
        self.delta_dims = list(config.get("delta_dims", self.position_dims))
        if len(self.position_dims) != len(self.delta_dims):
            raise ValueError("position_dims and delta_dims must have the same length")
        self.constraints = [dict(c) for c in config.get("constraints", [])]
        known = {"max_step", "workspace_box", "keep_out_box", "risk_limit", "joint_limits", "max_joint_step", "cartesian_box", "cartesian_keep_out_sphere"}
        for c in self.constraints:
            if c.get("type") not in known:
                raise ValueError(f"unknown constraint type: {c.get('type')!r}")
            c.setdefault("name", c["type"])
            if c["type"].startswith("cartesian") and fk is None:
                raise ValueError(f"constraint {c['name']!r} needs a kinematics function: Gate(config, fk=...)")
        self._risk_streak = 0

    @classmethod
    def from_file(cls, path: str | Path) -> "Gate":
        text = Path(path).read_text()
        if str(path).endswith(".json"):
            return cls(json.loads(text))
        import yaml

        return cls(yaml.safe_load(text))

    def reset(self) -> None:
        """Call at the start of each episode."""
        self._risk_streak = 0

    def check(self, state: np.ndarray, action: np.ndarray, risk: float = float("nan")) -> Decision:
        action = np.array(action, dtype=float)
        pos = np.asarray(state, dtype=float)[self.position_dims]
        delta = action[self.delta_dims].copy()
        verdict = "allow"
        violations: list[dict[str, Any]] = []

        def escalate(v: str) -> None:
            nonlocal verdict
            if VERDICT_ORDER[v] > VERDICT_ORDER[verdict]:
                verdict = v

        for c in self.constraints:
            kind = c["type"]
            if kind == "max_step":
                norm = float(np.linalg.norm(delta))
                if norm > c["max_norm"] + 1e-12:
                    violations.append({"constraint": c["name"], "detail": f"step {norm:.3f} > {c['max_norm']}"})
                    delta = delta * (c["max_norm"] / norm)
                    escalate("clamp")

            elif kind == "workspace_box":
                low, high = np.asarray(c["low"], float), np.asarray(c["high"], float)
                nxt = pos + delta
                if np.any(nxt < low) or np.any(nxt > high):
                    violations.append({"constraint": c["name"], "detail": "next position leaves workspace"})
                    if c.get("on_violation", "clamp") == "block":
                        delta = np.zeros_like(delta)
                        escalate("block")
                    else:
                        delta = np.clip(nxt, low, high) - pos
                        escalate("clamp")

            elif kind == "keep_out_box":
                low, high = np.asarray(c["low"], float), np.asarray(c["high"], float)
                # Already inside (should not happen): let it move so it can leave.
                if _inside(pos + delta, low, high) and not _inside(pos, low, high):
                    violations.append({"constraint": c["name"], "detail": "next position enters keep-out zone"})
                    slid = None
                    if c.get("on_violation", "clamp") == "clamp":
                        # Slide along the boundary: drop one axis of motion at a time
                        # and keep the option closest to what the policy asked for.
                        best = None
                        for axis in range(len(delta)):
                            cand = delta.copy()
                            cand[axis] = 0.0
                            if not _inside(pos + cand, low, high):
                                err = float(np.linalg.norm(cand - delta))
                                if best is None or err < best[0]:
                                    best = (err, cand)
                        slid = best[1] if best else None
                    if slid is not None:
                        delta = slid
                        escalate("clamp")
                    else:
                        delta = np.zeros_like(delta)
                        escalate("block")

            elif kind == "max_joint_step":
                # No single joint may move more than this in one step; the whole move is scaled to keep its direction.
                biggest = float(np.abs(delta).max()) if len(delta) else 0.0
                if biggest > c["max_abs"] + 1e-12:
                    violations.append({"constraint": c["name"], "detail": f"joint step {biggest:.3f} > {c['max_abs']}"})
                    delta = delta * (c["max_abs"] / biggest)
                    escalate("clamp")

            elif kind == "joint_limits":
                low, high = np.asarray(c["low"], float), np.asarray(c["high"], float)
                nxt = pos + delta
                if np.any(nxt < low) or np.any(nxt > high):
                    violations.append({"constraint": c["name"], "detail": "target is past a joint limit"})
                    delta = np.clip(nxt, low, high) - pos
                    escalate("clamp")

            elif kind in ("cartesian_box", "cartesian_keep_out_sphere"):
                # Where would this move put the arm? Checked for every tracked point, not just the gripper.
                def bad(d):
                    pts = self.fk(pos + d)
                    if kind == "cartesian_box":
                        return bool(np.any(pts < np.asarray(c["low"], float)) or np.any(pts > np.asarray(c["high"], float)))
                    return bool(np.any(np.linalg.norm(pts - np.asarray(c["center"], float), axis=1) < c["radius"]))

                if bad(delta) and not bad(np.zeros_like(delta)):  # if already in violation, let it move so it can leave
                    what = "leaves the allowed space" if kind == "cartesian_box" else "enters a keep-out zone"
                    violations.append({"constraint": c["name"], "detail": f"the arm {what}"})
                    # Shorten the move until it is safe; hold still if no shorter move is.
                    for scale in (0.5, 0.25, 0.1):
                        if not bad(delta * scale):
                            delta = delta * scale
                            escalate("clamp")
                            break
                    else:
                        delta = np.zeros_like(delta)
                        escalate("block")

            elif kind == "risk_limit":
                if not np.isnan(risk) and risk >= c["max_risk"]:
                    self._risk_streak += 1
                else:
                    self._risk_streak = 0
                if self._risk_streak >= int(c.get("sustain", 1)):
                    violations.append(
                        {"constraint": c["name"], "detail": f"failure risk {risk:.2f} >= {c['max_risk']}"}
                    )
                    escalate("halt")

        if verdict == "halt":
            delta = np.zeros_like(delta)
        action[self.delta_dims] = delta
        return Decision(action=action, verdict=verdict, violations=violations, risk=risk)
