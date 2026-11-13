"""Command line: stepscope demo | collect | train | evaluate | serve."""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

from .gate import Gate
from .predictor import Predictor
from .tracer import Tracer


def _collect(run_dir: Path, seeds: range, predictor: Predictor | None = None, halt_on_risk: bool = False, label: str = "") -> list[dict]:
    from .sim import GATE_CONFIG, TRACE_CONFIG, NoisyScriptedPolicy, TabletopEnv, rollout

    if run_dir.exists():
        shutil.rmtree(run_dir)
    gate_cfg = copy.deepcopy(GATE_CONFIG)
    if halt_on_risk:
        if predictor is None:
            raise SystemExit("--halt-on-risk needs --model")
        gate_cfg["constraints"].append(
            {"type": "risk_limit", "name": "failure_risk", "max_risk": predictor.threshold, "sustain": predictor.sustain}
        )
    tracer = Tracer(run_dir, config=TRACE_CONFIG, gate=Gate(gate_cfg), predictor=predictor, run_meta={"label": label})
    env, policy = TabletopEnv(), NoisyScriptedPolicy()
    results = []
    for i, seed in enumerate(seeds):
        results.append(rollout(env, policy, tracer, seed))
        if (i + 1) % 100 == 0 or i + 1 == len(seeds):
            ok = sum(r["success"] for r in results)
            print(f"  {run_dir.name}: {i + 1}/{len(seeds)} episodes, {ok} succeeded", flush=True)
    return results


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _print_report(rep: dict) -> None:
    m = rep["model"]
    print(f"\nHeld-out run: {rep['episodes']} episodes, {_pct(rep['success_rate'])} succeeded")
    rows = [("stepscope", m)]
    if "time_only_baseline" in rep:
        rows.append(("time-only baseline", rep["time_only_baseline"]))
    print(f"  {'':20s} {'failures flagged':>18s} {'false alarms':>14s} {'median warning':>16s}")
    for name, s in rows:
        flagged = f"{s['failures_flagged']}/{s['failures']} ({_pct(s['recall'])})"
        fa = f"{s['false_alarms']}/{s['successes']} ({_pct(s['false_alarm_rate'])})"
        lead = "n/a" if s["median_lead_steps"] is None else f"{s['median_lead_steps']:.0f} steps early"
        print(f"  {name:20s} {flagged:>18s} {fa:>14s} {lead:>16s}")
    if rep.get("auroc_at_step"):
        parts = [f"step {t}: {v['auroc']:.2f}" for t, v in rep["auroc_at_step"].items()]
        print("  AUROC of risk for 'will this episode fail', " + ", ".join(parts))
    if "diagnosis" in rep:
        d = rep["diagnosis"]
        print(f"  Top reason matches simulator ground truth in {_pct(d['agreement'])} of {d['episodes']} flagged failures")


def cmd_demo(args: argparse.Namespace) -> None:
    from .evaluate import compare_gated, evaluate, train

    root = Path(args.out)
    print(f"1/4 Collecting {args.train_episodes} training episodes")
    _collect(root / "runs" / "train", range(args.train_episodes), label="Training run (no predictor)")
    print("2/4 Training the failure predictor")
    model, baseline = train(root / "runs" / "train")
    model.save(root / "models" / "predictor.json")
    baseline.save(root / "models" / "baseline_time_only.json")
    print(f"3/4 Collecting {args.eval_episodes} held-out episodes with live risk scoring")
    seeds = range(200_000, 200_000 + args.eval_episodes)
    _collect(root / "runs" / "eval", seeds, predictor=model, label="Held-out run, risk scored live")
    report = evaluate(root / "runs" / "eval", model, baseline)
    print("4/4 Re-running the same episodes with the gate halting on high risk")
    _collect(root / "runs" / "eval-gated", seeds, predictor=model, halt_on_risk=True, label="Same episodes, gate halts on high risk")
    gated = compare_gated(root / "runs" / "eval", root / "runs" / "eval-gated")
    _print_report(report)
    spent = gated["robot_steps_spent_on_failures"]
    print(
        f"\nWith the gate halting on risk: {gated['failures_halted_early']}/{gated['failures_without_halt']} failures stopped early "
        f"(median {gated['median_steps_saved_per_halted_failure']:.0f} steps saved each; "
        f"{spent['without_halt']} -> {spent['with_halt']} robot steps spent on failing episodes), "
        f"{gated['successes_wrongly_halted']}/{gated['successes_without_halt']} would-be successes halted by mistake."
    )
    print(f"\nOpen the viewer:  stepscope serve --runs {root / 'runs'}")


def cmd_collect(args: argparse.Namespace) -> None:
    predictor = Predictor.load(args.model) if args.model else None
    _collect(Path(args.run), range(args.seed, args.seed + args.episodes), predictor, args.halt_on_risk)


def cmd_train(args: argparse.Namespace) -> None:
    from .evaluate import train

    model, baseline = train(args.run, false_alarm_rate=args.false_alarm_rate, extra_features=args.extra_features)
    model.save(args.out)
    baseline.save(Path(args.out).with_name("baseline_time_only.json"))
    print(f"Saved {args.out} (alert threshold {model.threshold:.2f}, sustained {model.sustain} steps)")
    for f, c in sorted(zip(model.features, model.coef), key=lambda fc: -abs(fc[1])):
        print(f"  {f:28s} {c:+.2f}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluate import evaluate

    baseline = Predictor.load(args.baseline) if args.baseline else None
    _print_report(evaluate(args.run, Predictor.load(args.model), baseline))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .server import create_app

    print(f"stepscope viewer on http://{args.host}:{args.port}")
    uvicorn.run(create_app(args.runs), host=args.host, port=args.port, log_level="warning")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="stepscope", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the full pipeline on the built-in simulator")
    d.add_argument("--out", default=".", help="where to put runs/ and models/")
    d.add_argument("--train-episodes", type=int, default=400)
    d.add_argument("--eval-episodes", type=int, default=200)
    d.set_defaults(fn=cmd_demo)

    c = sub.add_parser("collect", help="trace simulator episodes into a run directory")
    c.add_argument("--run", required=True)
    c.add_argument("--episodes", type=int, default=100)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--model", help="predictor to score risk live")
    c.add_argument("--halt-on-risk", action="store_true", help="gate halts the episode when risk stays high")
    c.set_defaults(fn=cmd_collect)

    t = sub.add_parser("train", help="fit the failure predictor on a traced run")
    t.add_argument("--run", required=True)
    t.add_argument("--out", default="models/predictor.json")
    t.add_argument("--false-alarm-rate", type=float, default=0.05)
    t.add_argument("--extra-features", nargs="*", default=[], help="names of your own signals to use as features")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="score a held-out run")
    e.add_argument("--run", required=True)
    e.add_argument("--model", default="models/predictor.json")
    e.add_argument("--baseline")
    e.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("serve", help="open the trace viewer")
    s.add_argument("--runs", default="runs")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8321)
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
