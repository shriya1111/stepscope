"""Export runs as one self-contained HTML file: the viewer with the data inlined.

Open it from disk or host it anywhere; it needs no server.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from .server import VIEWER_DIR, episode_payload, run_payload, runs_payload
from .store import list_runs


def _round(o, digits: int = 4):
    if isinstance(o, float):
        return round(o, digits)
    if isinstance(o, dict):
        return {k: _round(v, digits) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, digits) for v in o]
    return o


def export_html(runs_root: str | Path, out: str | Path, runs: list[str] | None = None, max_mb: float = 14.0) -> dict:
    """Writes `out`. Step data is included for failed, alerted and halted episodes
    first, then the rest, until the size budget is used up."""
    root = Path(runs_root)
    dirs = [d for d in list_runs(root) if runs is None or d.name in runs]
    api: dict = {"runs": [r for r in runs_payload(root) if runs is None or r["name"] in runs]}
    frames: dict = {}
    queue = []
    for d in dirs:
        payload = run_payload(d)
        api[f"runs/{d.name}"] = payload
        for i, e in enumerate(payload["episodes"]):
            interesting = e["success"] is not True or e["first_alert_step"] is not None
            queue.append((0 if interesting else 1, i, d, e["episode_id"]))
    budget, used, included = max_mb * 1e6, 0, 0
    used += len(json.dumps(_round(api)))
    for _, _, d, ep in sorted(queue, key=lambda q: q[:2]):
        ep_dir = d / "episodes" / ep
        body = json.dumps(_round(episode_payload(ep_dir)), separators=(",", ":"))
        png = ep_dir / "frames.png"
        uri = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode() if png.exists() else ""
        if used + len(body) + len(uri) > budget:
            continue
        api[f"runs/{d.name}/episodes/{ep}"] = json.loads(body)
        if uri:
            frames[f"{d.name}/{ep}"] = uri
        used += len(body) + len(uri)
        included += 1
    data = json.dumps({"api": _round(api), "frames": frames}, separators=(",", ":")).replace("</", "<\\/")
    html = (VIEWER_DIR / "index.html").read_text().replace("/*STEPSCOPE_DATA*/", f"window.STEPSCOPE_DATA = {data};", 1)
    Path(out).write_text(html)
    return {"episodes_total": len(queue), "episodes_with_steps": included, "bytes": len(html.encode())}
