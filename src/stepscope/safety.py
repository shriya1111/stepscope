"""Environment-aware safety: score each action against what is around the robot.

A Scene is a list of hazards (a person's hand, a fragile object, a hot plate),
each with a position, a size and a severity class. In a deployment the list
comes from a detector on the camera feed; here it is loaded from a YAML file.

For every step we look at where the policy's sampled plans would take the robot
and compute:

  clearance   closest the executed plan comes to any hazard (workspace units)
  time_to_contact  steps until the executed plan first touches a hazard
  p_contact   chance of contact: the share of sampled plans that touch the
              hazard, or a proximity term if that is larger
  severity    1 (minor) to 3 (serious), from the hazard's class
  risk        p_contact x severity weight x speed factor, in [0, 1]

Risk is likelihood times consequence, so a slow move near a cup and a fast move
toward a hand are not scored the same. The numbers are a transparent heuristic,
not a certified hazard analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .store import dump_json, list_episodes, read_internals, read_steps

SEVERITY_WEIGHT = {1: 0.25, 2: 0.6, 3: 1.0}
SEVERITY_NAME = {1: "minor", 2: "moderate", 3: "serious"}
LEVELS = [(0.4, "high"), (0.1, "medium"), (0.0, "low")]


def level(risk: float) -> str:
    return next(name for cut, name in LEVELS if risk >= cut)


@dataclass
class Hazard:
    name: str
    center: list[float]
    radius: float
    severity: int = 2  # 1 minor, 2 moderate, 3 serious
    kind: str = "object"  # person | fragile | object ...


@dataclass
class Scene:
    hazards: list[Hazard] = field(default_factory=list)
    extent: float = 1.0  # workspace size, for drawing
    robot_radius: float = 0.0  # added to every hazard radius
    speed_ref: float = 1.0  # speed (units per step) treated as "full speed"
    proximity_scale: float = 1.0  # distance at which proximity risk has fallen to ~37%

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        d = dict(d)
        d["hazards"] = [Hazard(**h) for h in d.get("hazards", [])]
        return cls(**d)

    @classmethod
    def from_file(cls, path: str | Path) -> "Scene":
        text = Path(path).read_text()
        if str(path).endswith(".json"):
            return cls.from_dict(json.loads(text))
        import yaml

        return cls.from_dict(yaml.safe_load(text))

    def to_dict(self) -> dict:
        return asdict(self)


def path_clearance(points: np.ndarray, hazard: Hazard, robot_radius: float = 0.0) -> tuple[float, int | None]:
    """(closest approach to the hazard's edge, first index that touches it or None).

    points: (T, 2) positions along a path, checked along the segments between them.
    Negative clearance means the path goes inside the hazard.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    c = np.asarray(hazard.center, dtype=float)
    r = hazard.radius + robot_radius
    if len(pts) == 1:
        d = np.array([np.linalg.norm(pts[0] - c)])
    else:
        a, b = pts[:-1], pts[1:]
        ab = b - a
        t = np.clip(((c - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-12), 0, 1)
        d = np.linalg.norm(a + t[:, None] * ab - c, axis=1)
    clear = d - r
    hit = np.where(clear <= 0)[0]
    return float(clear.min()), (int(hit[0]) if len(hit) else None)


def assess(position: np.ndarray, plans: np.ndarray, speed: float, scene: Scene) -> dict:
    """Safety of acting from `position` given sampled future paths.

    plans: (N, T, 2) future positions for each sampled plan; plans[0] is the one
    being executed. Returns the worst hazard's numbers.
    """
    pos = np.asarray(position, dtype=float).reshape(1, 2)
    plans = np.asarray(plans, dtype=float)
    speed_factor = float(np.clip((speed / scene.speed_ref) ** 2, 0.1, 1.0))  # consequence grows with kinetic energy
    out = {"clearance": float("inf"), "time_to_contact": None, "p_contact": 0.0, "severity": 0, "risk": 0.0, "hazard": None}
    for h in scene.hazards:
        results = [path_clearance(np.concatenate([pos, p]), h, scene.robot_radius) for p in plans]
        clear, ttc = results[0]
        hit_share = float(np.mean([r[1] is not None for r in results]))
        p = max(hit_share, float(np.exp(-max(clear, 0.0) / scene.proximity_scale)))
        risk = p * SEVERITY_WEIGHT[h.severity] * speed_factor
        out["clearance"] = min(out["clearance"], clear)
        if ttc is not None:
            out["time_to_contact"] = ttc if out["time_to_contact"] is None else min(out["time_to_contact"], ttc)
        if out["hazard"] is None or risk > out["risk"]:
            out.update(p_contact=p, severity=h.severity, risk=risk, hazard=h.name)
    if not scene.hazards:
        out["clearance"] = None
    out["level"] = level(out["risk"])
    return out


def risk_grid(scene: Scene, n: int = 32, speed: float | None = None) -> list[list[float]]:
    """Risk of standing still at each cell of an n x n grid over the workspace (for the map overlay)."""
    speed = scene.speed_ref if speed is None else speed
    out = []
    for iy in range(n):
        row = []
        for ix in range(n):
            p = np.array([(ix + 0.5) / n, (iy + 0.5) / n]) * scene.extent
            row.append(assess(p, p[None, None, :], speed, scene)["risk"] if scene.hazards else 0.0)
        out.append(row)
    return out


def annotate_run(run_dir: str | Path, scene: Scene, position_dims=(0, 1), plans_key: str = "plans", plan_offset: int = 1) -> dict:
    """Score every step of every episode in a run against the scene.

    Needs sampled plans in the traces (internals[plans_key], shape (N, H, 2) in
    workspace units). plan_offset is the index in a plan of the first action
    that is executed. Writes <episode>/safety.json and <run>/scene.json.
    """
    run_dir = Path(run_dir)
    dump_json(run_dir / "scene.json", {**scene.to_dict(), "risk_grid": risk_grid(scene)})
    totals = {"episodes": 0, "steps": 0, "high": 0, "medium": 0, "contacts": 0}
    for meta in list_episodes(run_dir):
        ep = meta["episode_id"]
        z = read_internals(run_dir, ep)
        if plans_key not in z:
            continue
        df = read_steps(run_dir, ep)
        pos = np.stack(df["state"].to_numpy())[:, list(position_dims)].astype(float)
        speed = np.linalg.norm(np.stack(df["executed"].to_numpy())[:, :2], axis=1)
        ts = z[f"{plans_key}__t"]
        cols: dict[str, list] = {k: [] for k in ("clearance", "time_to_contact", "p_contact", "severity", "risk", "hazard", "level", "inside")}
        for t in range(len(df)):
            i = int(np.searchsorted(ts, t, side="right") - 1)
            k = t - int(ts[i])  # steps into the current plan
            plans = z[plans_key][i][:, plan_offset + k :]
            if plans.shape[1] == 0:
                plans = z[plans_key][i][:, -1:]
            a = assess(pos[t], plans, float(speed[t]), scene)
            inside = any(np.linalg.norm(pos[t] - np.asarray(h.center)) <= h.radius + scene.robot_radius for h in scene.hazards)
            for key in cols:
                cols[key].append(inside if key == "inside" else a[key])
        peak = float(np.max(cols["risk"])) if cols["risk"] else 0.0
        summary = {
            "peak_risk": peak,
            "peak_level": level(peak),
            "high_steps": int(sum(lv == "high" for lv in cols["level"])),
            "medium_steps": int(sum(lv == "medium" for lv in cols["level"])),
            "contact_steps": int(sum(cols["inside"])),
            "min_clearance": float(np.min(cols["clearance"])) if scene.hazards else None,
        }
        dump_json(run_dir / "episodes" / ep / "safety.json", {"summary": summary, **cols})
        totals["episodes"] += 1
        totals["steps"] += len(df)
        totals["high"] += summary["high_steps"]
        totals["medium"] += summary["medium_steps"]
        totals["contacts"] += summary["contact_steps"]
    return totals


def choose_plan(futures: list[dict]) -> int:
    """Pick which sampled plan to execute, given its predicted future.

    futures[i] has `contact` (bool: predicted to touch a hazard), `risk` and
    `gain` (predicted task progress). Safe plans beat unsafe ones; among safe
    plans the one with the most predicted progress wins; if none is safe, the
    least risky.
    """
    safe = [i for i, f in enumerate(futures) if not f["contact"]]
    if safe:
        return max(safe, key=lambda i: (round(futures[i]["gain"], 3), -i))
    return min(range(len(futures)), key=lambda i: (futures[i]["risk"], i))
