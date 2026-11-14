"""Trace viewer backend: a small read-only JSON API over the run directories."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .deepdive import read_deepdive
from .explain import SituationIndex, describe_decision
from .store import list_episodes, list_runs, read_internals, read_run_meta, read_steps

_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
VIEWER_DIR = Path(__file__).parent / "viewer"


def _clean(values) -> list:
    """NaN/inf are not valid JSON; send them as null."""
    return [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in values]


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def runs_payload(root: Path) -> list[dict]:
    out = []
    for d in list_runs(root):
        eps = list_episodes(d)
        out.append(
            {
                "name": d.name,
                "label": read_run_meta(d).get("label") or "",
                "episodes": len(eps),
                "successes": sum(bool(e["success"]) for e in eps),
            }
        )
    return out


def _with_safety(d: Path) -> list[dict]:
    eps = list_episodes(d)
    for e in eps:
        s = _read_json(d / "episodes" / e["episode_id"] / "safety.json")
        e["safety"] = s["summary"] if s else None
        e["deepdives"] = sorted(int(t) for t in (read_deepdive(d / "episodes" / e["episode_id"]) or {}))
    return eps


def run_payload(d: Path) -> dict:
    return {
        "run": read_run_meta(d),
        "report": _read_json(d / "report.json"),
        "gated_report": _read_json(d / "gated_report.json"),
        "episodes": _with_safety(d),
        "map": situation_index(d).points(),
        "scene": _read_json(d / "scene.json"),
    }


_INDEX_CACHE: dict = {}


def situation_index(run_dir: Path) -> SituationIndex:
    """Embedding index for a run, rebuilt when episodes are added."""
    key = (str(run_dir), sum(1 for _ in (run_dir / "episodes").iterdir()))
    if key not in _INDEX_CACHE:
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = SituationIndex(run_dir)
    return _INDEX_CACHE[key]


def decisions_payload(d: Path, index: SituationIndex | None = None) -> dict | None:
    """What the model was deciding, at each step where it ran (None if no internals were logged)."""
    z = read_internals(d.parent.parent, d.name)
    if "plans" not in z:
        return None
    index = index or situation_index(d.parent.parent)
    grippers = (read_run_meta(d.parent.parent).get("scene3d") or {}).get("grippers") or [-1]
    near = index.neighbours(d.name)
    xy = {}
    if not index.empty:
        mine = index.episodes == d.name
        xy = {int(t): [float(a), float(b)] for t, (a, b) in zip(index.steps[mine], index.xy[mine])}
    out = []
    for i, t in enumerate(z["plans__t"].tolist()):
        modes = z["modes"][i].tolist() if "modes" in z else [0] * len(z["plans"][i])
        spread = z["denoise_spread"][i].tolist() if "denoise_spread" in z else None
        nb = near.get(t, [])
        out.append(
            {
                "t": t,
                "plans": z["plans"][i].tolist(),
                "modes": modes,
                "keypoints": z["keypoints"][i].tolist() if "keypoints" in z else None,
                "attention": np.round(z["attention"][i], 4).tolist() if "attention" in z else None,
                # 3D path of each gripper under each candidate plan: (plans, grippers, points, 3)
                "plans_xyz": np.round(z["plans_xyz"][i][:, :, grippers].transpose(0, 2, 1, 3), 3).tolist() if "plans_xyz" in z else None,
                "denoise_spread": spread,
                "denoise_path": z["denoise_path"][i][:, ::3].tolist() if "denoise_path" in z else None,
                "map": xy.get(t),
                "neighbours": nb,
                "summary": describe_decision(len(modes), np.bincount(modes).tolist(), spread[-1] if spread else None, nb),
            }
        )
    return {"decisions": out}


def scene3d_payload(d: Path) -> dict | None:
    """Arm link positions per step, and tracked objects, for the 3D view (None if not recorded)."""
    z = read_internals(d.parent.parent, d.name)
    if "arm" not in z:
        return None
    cfg = read_run_meta(d.parent.parent).get("scene3d") or {}
    objects = []
    for o in cfg.get("objects", []):  # an object's position is read out of the saved simulator state
        if "snapshot" in z:
            objects.append({"name": o["name"], "size": o.get("size", 0.02), "path": np.round(z["snapshot"][:, o["snapshot_index"]], 3).tolist()})
    return {"arm": np.round(z["arm"].astype(float), 3).tolist(), "objects": objects}


def episode_payload(d: Path, index: SituationIndex | None = None) -> dict:
    df = read_steps(d.parent.parent, d.name)
    return {
        "decisions": decisions_payload(d, index),
        "scene3d": scene3d_payload(d),
        "safety": _read_json(d / "safety.json"),
        "deepdive": read_deepdive(d, compact=True),
        "meta": _read_json(d / "meta.json"),
        "steps": {
            "risk": _clean(df["risk"].tolist()),
            "reward": _clean(df["reward"].tolist()) if "reward" in df else None,
            "verdict": df["verdict"].tolist(),
            "violations": [json.loads(v) for v in df["violations"]],
            "belief": [json.loads(v) for v in df["belief"]],
            "plan_step": df["plan_step"].tolist(),
            "state": [list(map(float, s)) for s in df["state"]],
            "action": [list(map(float, s)) for s in df["action"]],
            "executed": [list(map(float, s)) for s in df["executed"]],
            "metrics": {c[2:]: _clean(df[c].tolist()) for c in df.columns if c.startswith("m_")},
        },
    }


def create_app(runs_root: str | Path) -> FastAPI:
    root = Path(runs_root)
    app = FastAPI(title="stepscope", docs_url=None, redoc_url=None)

    def run_dir(name: str) -> Path:
        if not _NAME.match(name) or name in (".", ".."):
            raise HTTPException(404, "no such run")
        for d in list_runs(root):
            if d.name == name:
                return d
        raise HTTPException(404, "no such run")

    def episode_dir(run: str, ep: str) -> Path:
        if not _NAME.match(ep) or ep in (".", ".."):
            raise HTTPException(404, "no such episode")
        d = run_dir(run) / "episodes" / ep
        if not (d / "meta.json").exists():
            raise HTTPException(404, "no such episode")
        return d

    @app.get("/api/runs")
    def runs():
        return runs_payload(root)

    @app.get("/api/runs/{run}")
    def run_detail(run: str):
        return run_payload(run_dir(run))

    @app.get("/api/runs/{run}/episodes/{ep}")
    def episode(run: str, ep: str):
        return episode_payload(episode_dir(run, ep))

    @app.get("/api/runs/{run}/episodes/{ep}/frames.png")
    def frames(run: str, ep: str):
        p = episode_dir(run, ep) / "frames.png"
        if not p.exists():
            raise HTTPException(404, "no frames recorded")
        return FileResponse(p, media_type="image/png")

    @app.get("/")
    def index():
        return FileResponse(VIEWER_DIR / "index.html", media_type="text/html")

    return app
