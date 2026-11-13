"""On-disk trace format.

    <run_dir>/run.json
    <run_dir>/episodes/<episode_id>/meta.json     outcome, summary, your metadata
    <run_dir>/episodes/<episode_id>/steps.parquet one row per step
    <run_dir>/episodes/<episode_id>/frames.png    camera frames as one sprite sheet

Plain Parquet + JSON + PNG, so traces open in pandas, DuckDB or an image viewer
without stepscope installed.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPRITE_COLUMNS = 16


def encode_png(img: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as PNG (no imaging library needed)."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w, c = img.shape
    if c != 3:
        raise ValueError("expected an RGB image")
    raw = np.concatenate([np.zeros((h, 1), np.uint8), img.reshape(h, w * 3)], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def make_sprite(frames: list[np.ndarray]) -> np.ndarray:
    h, w, _ = frames[0].shape
    rows = (len(frames) + SPRITE_COLUMNS - 1) // SPRITE_COLUMNS
    sheet = np.zeros((rows * h, SPRITE_COLUMNS * w, 3), np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, SPRITE_COLUMNS)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = f
    return sheet


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2))


def forward_fill(ts, values: np.ndarray, n_steps: int) -> np.ndarray:
    """Spread values recorded at steps `ts` over every step (each holds until the next)."""
    idx = np.clip(np.searchsorted(np.asarray(ts), np.arange(n_steps), side="right") - 1, 0, len(ts) - 1)
    return np.asarray(values)[idx]


def read_internals(run_dir: Path, episode_id: str) -> dict[str, np.ndarray]:
    """Arrays logged via `internals=`. For each key K: K (stacked values) and K__t (their steps)."""
    p = Path(run_dir) / "episodes" / episode_id / "internals.npz"
    if not p.exists():
        return {}
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def write_episode(
    ep_dir: Path, rows: list[dict], frames: list[np.ndarray], meta: dict, internals: dict | None = None
) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ("belief", "violations"):
        if col in df:
            df[col] = df[col].map(lambda v: json.dumps(_jsonable(v)))
    df.to_parquet(ep_dir / "steps.parquet", index=False)
    if frames:
        h, w, _ = frames[0].shape
        meta = {**meta, "frames": {"count": len(frames), "width": w, "height": h, "columns": SPRITE_COLUMNS}}
        (ep_dir / "frames.png").write_bytes(encode_png(make_sprite(frames)))
    if internals:
        arrays = {}
        for key, (ts, vals) in internals.items():
            arrays[key] = np.stack(vals)
            arrays[f"{key}__t"] = np.asarray(ts, dtype=np.int32)
        np.savez_compressed(ep_dir / "internals.npz", **arrays)
        meta = {**meta, "internals": sorted(internals)}
    dump_json(ep_dir / "meta.json", meta)


def list_runs(root: Path) -> list[Path]:
    root = Path(root)
    if (root / "episodes").is_dir():
        return [root]
    return sorted(p for p in root.iterdir() if (p / "episodes").is_dir()) if root.is_dir() else []


def read_run_meta(run_dir: Path) -> dict:
    p = Path(run_dir) / "run.json"
    return json.loads(p.read_text()) if p.exists() else {}


def list_episodes(run_dir: Path) -> list[dict]:
    out = []
    ep_root = Path(run_dir) / "episodes"
    if not ep_root.is_dir():
        return out
    for d in sorted(ep_root.iterdir()):
        mp = d / "meta.json"
        if mp.exists():
            out.append(json.loads(mp.read_text()))
    return out


def read_steps(run_dir: Path, episode_id: str) -> pd.DataFrame:
    return pd.read_parquet(Path(run_dir) / "episodes" / episode_id / "steps.parquet")


def metrics_matrix(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c[2:]: df[c].to_numpy(dtype=float) for c in df.columns if c.startswith("m_")}


def states_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["state"].to_numpy()).astype(float)
