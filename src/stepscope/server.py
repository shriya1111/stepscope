"""Trace viewer backend: a small read-only JSON API over the run directories."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .store import list_episodes, list_runs, read_run_meta, read_steps

_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
VIEWER_DIR = Path(__file__).parent / "viewer"


def _clean(values) -> list:
    """NaN/inf are not valid JSON; send them as null."""
    return [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in values]


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

    def read_json(path: Path):
        return json.loads(path.read_text()) if path.exists() else None

    @app.get("/api/runs")
    def runs():
        out = []
        for d in list_runs(root):
            eps = list_episodes(d)
            meta = read_run_meta(d)
            out.append(
                {
                    "name": d.name,
                    "label": meta.get("label") or "",
                    "episodes": len(eps),
                    "successes": sum(bool(e["success"]) for e in eps),
                }
            )
        return out

    @app.get("/api/runs/{run}")
    def run_detail(run: str):
        d = run_dir(run)
        return {
            "run": read_run_meta(d),
            "report": read_json(d / "report.json"),
            "gated_report": read_json(d / "gated_report.json"),
            "episodes": list_episodes(d),
        }

    @app.get("/api/runs/{run}/episodes/{ep}")
    def episode(run: str, ep: str):
        d = episode_dir(run, ep)
        df = read_steps(d.parent.parent, ep)
        return {
            "meta": read_json(d / "meta.json"),
            "steps": {
                "risk": _clean(df["risk"].tolist()),
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
