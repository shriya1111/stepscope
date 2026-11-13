"""Train the predictor from a traced run and measure how early it flags failures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .predictor import Predictor, auroc, first_alert
from .store import dump_json, list_episodes, metrics_matrix, read_steps, states_matrix

def load_episodes(run_dir: str | Path) -> list[dict]:
    out = []
    for meta in list_episodes(Path(run_dir)):
        if meta["success"] is None:
            continue
        df = read_steps(Path(run_dir), meta["episode_id"])
        out.append(
            {
                "id": meta["episode_id"],
                "success": bool(meta["success"]),
                "metrics": metrics_matrix(df),
                "states": states_matrix(df),
                "risk": df["risk"].to_numpy(dtype=float),
                "meta": meta,
            }
        )
    return out


def train(
    run_dir: str | Path, false_alarm_rate: float = 0.05, extra_features: list[str] | None = None
) -> tuple[Predictor, Predictor]:
    """Returns (predictor, time-only baseline) fitted on a traced run.

    extra_features: names of your own `signals` to use alongside the built-in metrics.
    """
    from .predictor import DEFAULT_FEATURES

    eps = load_episodes(run_dir)
    if not any(e["success"] for e in eps) or all(e["success"] for e in eps):
        raise ValueError("training needs both successful and failed episodes")
    model = Predictor(DEFAULT_FEATURES + list(extra_features or [])).fit(eps, false_alarm_rate=false_alarm_rate)
    baseline = Predictor(features=["elapsed"]).fit(eps, false_alarm_rate=false_alarm_rate)
    return model, baseline


def _alert_stats(eps: list[dict], risks: list[np.ndarray], threshold: float, sustain: int) -> dict:
    alerts = [first_alert(r, threshold, sustain) for r in risks]
    failed = [(e, a) for e, a in zip(eps, alerts) if not e["success"]]
    ok = [(e, a) for e, a in zip(eps, alerts) if e["success"]]
    leads = [len(e["states"]) - 1 - a for e, a in failed if a is not None]
    return {
        "failures": len(failed),
        "failures_flagged": sum(a is not None for _, a in failed),
        "recall": float(np.mean([a is not None for _, a in failed])) if failed else None,
        "successes": len(ok),
        "false_alarms": sum(a is not None for _, a in ok),
        "false_alarm_rate": float(np.mean([a is not None for _, a in ok])) if ok else None,
        "median_lead_steps": float(np.median(leads)) if leads else None,
        "median_alert_step": float(np.median([a for _, a in failed if a is not None])) if leads else None,
    }


def evaluate(run_dir: str | Path, model: Predictor, baseline: Predictor | None = None) -> dict:
    """Score a held-out run. Writes <run_dir>/report.json and returns it."""
    eps = load_episodes(run_dir)
    risks = [model.score(e["metrics"], e["states"]) for e in eps]
    report: dict = {
        "episodes": len(eps),
        "success_rate": float(np.mean([e["success"] for e in eps])),
        "threshold": model.threshold,
        "sustain": model.sustain,
        "model": _alert_stats(eps, risks, model.threshold, model.sustain),
    }
    if baseline is not None:
        b_risks = [baseline.score(e["metrics"], e["states"]) for e in eps]
        report["time_only_baseline"] = _alert_stats(eps, b_risks, baseline.threshold, baseline.sustain)

    # How well does risk at step t separate episodes that will fail from ones that will not?
    report["auroc_at_step"] = {}
    for t in (10, 20, 30, 40, 50):
        alive = [(r[t], not e["success"]) for e, r in zip(eps, risks) if len(r) > t]
        if len(alive) >= 20:
            s, y = zip(*alive)
            report["auroc_at_step"][str(t)] = {"auroc": auroc(np.array(s), np.array(y)), "episodes_running": len(alive)}

    # Does the top reason given at alert time match what the simulator says went wrong?
    hits = total = 0
    confusion: dict[str, dict[str, int]] = {}
    for e, r in zip(eps, risks):
        truth = (e["meta"].get("meta", {}).get("sim") or {}).get("cause")
        a = first_alert(r, model.threshold, model.sustain)
        if e["success"] or a is None or truth not in ("oscillation", "stalled", "retry loop"):
            continue
        why = model.explain(e["metrics"], e["states"], a, top=1)
        guess = why[0]["label"] if why else "none"
        confusion.setdefault(truth, {}).setdefault(guess, 0)
        confusion[truth][guess] += 1
        total += 1
        hits += guess == truth
    if total:
        report["diagnosis"] = {"agreement": hits / total, "episodes": total, "confusion": confusion}

    report["feature_weights"] = dict(zip(model.features, model.coef.tolist()))
    dump_json(Path(run_dir) / "report.json", report)
    return report


def compare_gated(baseline_run: str | Path, gated_run: str | Path) -> dict:
    """Pair episodes by seed: what did halting on high risk change?"""
    base = {e["meta"]["seed"]: e for e in list_episodes(Path(baseline_run))}
    gated = {e["meta"]["seed"]: e for e in list_episodes(Path(gated_run))}
    seeds = sorted(set(base) & set(gated))
    halted = [s for s in seeds if gated[s]["reason"] == "halted_by_gate"]
    halted_fail = [s for s in halted if not base[s]["success"]]
    halted_ok = [s for s in halted if base[s]["success"]]
    saved = [base[s]["n_steps"] - gated[s]["n_steps"] for s in halted_fail]
    n_fail = sum(not base[s]["success"] for s in seeds)
    n_ok = len(seeds) - n_fail
    report = {
        "paired_episodes": len(seeds),
        "failures_without_halt": n_fail,
        "failures_halted_early": len(halted_fail),
        "median_steps_saved_per_halted_failure": float(np.median(saved)) if saved else None,
        "robot_steps_spent_on_failures": {
            "without_halt": int(sum(base[s]["n_steps"] for s in seeds if not base[s]["success"])),
            "with_halt": int(sum(gated[s]["n_steps"] for s in seeds if not base[s]["success"])),
        },
        "successes_without_halt": n_ok,
        "successes_wrongly_halted": len(halted_ok),
    }
    dump_json(Path(gated_run) / "gated_report.json", report)
    return report
