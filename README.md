# stepscope

Step-level decision tracing, failure prediction and pre-actuation gating for robot policies.

Robot data tools record what the robot sensed and did. stepscope records what the
policy *decided* at every step (its belief, its plan step, how much its sampled
actions disagreed), turns that into a live failure-risk score, and puts a hard
safety gate between the policy and the actuators.

![Episode view](docs/episode.png)

## What is in the box

| Piece | What it does | Where |
|---|---|---|
| Tracing SDK | Wraps a policy loop in a few lines. Writes plain Parquet + JSON + PNG. | `src/stepscope/tracer.py`, `store.py` |
| Step metrics | Sample spread, direction disagreement, reversals, path efficiency, net progress, plan regressions, subgoal dwell. Computed online. | `metrics.py` |
| Failure predictor | Logistic regression over the step metrics. Gives a risk score per step, an alert threshold calibrated to a false-alarm rate, and the leading symptoms at alert time. | `predictor.py`, `evaluate.py` |
| Gate | Declarative constraints checked before every action: speed limit, workspace box, keep-out zones, halt on sustained risk. | `gate.py` |
| Viewer | Fleet overview and a per-episode timeline with camera, belief, risk, gate events and metrics in sync. | `server.py`, `viewer/index.html` |
| Simulator | A numpy tabletop pick-and-place task and a scripted policy with realistic failure modes, so everything runs with no robot or GPU. | `sim/` |
| Gymnasium adapter | Trace any Gymnasium-style environment. | `adapters/gym.py` |

## Quick start

```bash
pip install -e ".[dev]"
```

```bash
stepscope demo
```

```bash
stepscope serve --runs runs
```

`stepscope export --runs runs --out stepscope.html` writes the viewer and its data as
one self-contained page. `stepscope demo` takes under a minute. It collects 400 training episodes, trains
the predictor, scores 200 held-out episodes live, then re-runs those same
episodes with the gate halting on high risk. The viewer opens on
http://127.0.0.1:8321.

## Results on the built-in simulator

From `stepscope demo` (200 held-out episodes, 42 of which fail):

| | Failures flagged | False alarms | Median warning before the end |
|---|---|---|---|
| stepscope | 34 / 42 (81%) | 7 / 158 (4%) | 58 steps |
| Elapsed-time-only baseline | 40 / 42 (95%) | 3 / 158 (2%) | 8 steps |

- Episodes last at most 90 steps, so the median alert comes about a third of the way in.
- With the gate halting on sustained risk, 34 of 42 failures stop early and robot
  steps spent on failing episodes drop from 3,722 to 1,978. The cost is 7 of 158
  would-be successes halted by mistake.
- The leading symptom at alert time matches the simulator's ground-truth cause in
  59% of flagged failures.

**Read these numbers as a pipeline check, not as evidence about real robots.** The
simulator and its failure modes were written alongside the metrics, so the
metrics are well matched to them. The numbers that matter will come from a real
policy (see the roadmap).

## Results on a real policy (LeRobot diffusion policy, PushT)

`lerobot/diffusion_pusht` (263M parameters), run on CPU with DDIM 8 steps and 6
sampled plans per decision. 60 episodes, 55% task success.

- **Failure prediction does not work here yet.** Cross-validated over the 60
  episodes, the risk score ranks failing episodes no better than chance (AUROC
  0.44 to 0.54 at steps 20, 60 and 100), with the built-in metrics and with the
  model-internal signals (plan shift, plan options, sample spread, embedding
  novelty). The simulator numbers above did not transfer.
- **The shield made no measurable difference.** On 30 paired seeds with the
  demo hazard scene, executing the sampled plan with the best predicted future
  gave 20/30 task successes against 18/30 without it (6 episodes flipped to
  success, 4 to failure), and the share of steps spent inside a hazard zone was
  12.8% against 12.5%. The shield switched plans at 40% of decisions, but this
  policy's six sampled plans are usually within a few pixels of each other, so
  there is rarely a safer one to switch to.
- **What does work on the real policy** is the inspection: sampled plans and
  attention points on the camera frame, the situation map, deep dives
  (debugger, cause map, what-if, predicted futures) and per-step safety scores.

The demo hazard scene (`examples/pusht_scene.yaml`) sits where the pusher often
travels and the policy cannot see it, so nearly every episode reaches high
safety risk. It shows the scoring working, not a realistic risk level.

## Trace your own policy

```python
from stepscope import Gate, Predictor, TraceConfig, Tracer

tracer = Tracer(
    "runs/my-run",
    config=TraceConfig(position_dims=[0, 1, 2], motion_dims=[0, 1, 2], max_steps=300),
    gate=Gate.from_file("examples/gate.yaml"),          # optional
    predictor=Predictor.load("models/predictor.json"),  # optional
)

with tracer.episode(task="pick up the cube") as ep:
    obs = env.reset()
    while True:
        samples = policy.sample_actions(obs, n=8)      # (N, action_dim)
        decision = ep.step(
            state=obs["state"],
            action=samples[0],
            action_samples=samples,   # optional: enables uncertainty metrics
            frame=obs["image"],       # optional: camera frame
            belief={"subtask": "reach"},  # optional: anything the policy believes
            plan_step=0,              # optional: enables plan-progress metrics
        )
        if decision.halt:
            ep.end(success=False, reason="halted_by_gate")
            break
        obs, done, success = env.step(decision.action)  # execute the gated action
        if done:
            ep.end(success=success)
            break
```

A runnable version is in `examples/quickstart.py`. To train a predictor on your
own traces: `stepscope train --run runs/my-run --out models/predictor.json`. It
needs both successful and failed episodes.

## Trace format

```
runs/<run>/run.json
runs/<run>/episodes/<id>/meta.json      outcome, alert step, symptoms, your metadata
runs/<run>/episodes/<id>/steps.parquet  one row per step: state, action, executed
                                        action, gate verdict, belief, risk, m_* metrics
runs/<run>/episodes/<id>/frames.png     camera frames as one sprite sheet
```

## Seeing inside the model

Pass arrays from inside your model to `ep.step(internals={...})` on the steps
where it runs. They are saved per episode in `internals.npz` and the viewer
turns them into:

- **Sampled plans drawn on the camera frame**, grouped into options ("4 of 6
  plans go left, 2 go right"), with the executed plan highlighted.
- **Attention points**: where the vision encoder's last layer is looking.
- **Plans forming out of noise**: disagreement between samples after each
  denoising step.
- **A situation map**: every decision placed by the model's internal
  representation, coloured by how the episode ended, and for each decision the
  most similar past moments and their outcomes.
- **A one-line summary** of each decision.

The LeRobot PushT harness (`src/stepscope/adapters/lerobot_pusht.py`) records all
of these from a real pretrained diffusion policy:

```bash
pip install "lerobot[diffusion,pusht]"
```

```bash
python -m stepscope.adapters.lerobot_pusht --run runs/pusht --episodes 40
```

## Deep dives: how did the model get to this action?

For chosen decisions, stepscope restores the saved environment state, re-runs the
real model with fixed starting noise, and stores four analyses the viewer shows
as tabs:

- **Decision debugger**: the plan the model would output if it stopped after
  each denoising step. Drag a slider to watch the decision form.
- **Cause map**: hide each patch of the camera image and measure how far the
  plan moves; also remove whole inputs (camera, robot position, motion).
- **What-if**: move or turn the block, or move the pusher, and see the new plan
  next to the original.
- **Predicted futures**: play every sampled plan forward in a copy of the
  simulator and report predicted progress and closest approach to hazards.

```bash
python -m stepscope.adapters.pusht_deepdive --run runs/pusht --auto 3 --every 4 --scene examples/pusht_scene.yaml
```

The what-if edits are a fixed set computed ahead of time, not free-form
editing in the browser. The model-agnostic parts are in `src/stepscope/deepdive.py`;
the PushT-specific forward passes are in `src/stepscope/adapters/pusht_deepdive.py`.

## Environment-aware safety

Declare what is around the robot in a scene file (`examples/pusht_scene.yaml`):
hazards with a position, size and severity class. Then score every traced step:

```bash
stepscope safety --run runs/pusht --scene examples/pusht_scene.yaml
```

For each step this looks at where the sampled plans would take the robot and
records clearance, time to contact, chance of contact (the share of sampled
plans that touch the hazard), severity, and a combined risk:
chance of contact x severity weight x speed factor. The viewer draws the hazards
on the camera frame and adds safety-risk and clearance timelines.

**Shield.** With `--shield`, the PushT harness predicts the future of every
sampled plan in a simulator copy and executes the one with the best predicted
outcome that stays clear of hazards (`safety.choose_plan`):

```bash
python -m stepscope.adapters.lerobot_pusht --run runs/pusht-shield --episodes 30 --shield --scene examples/pusht_scene.yaml
```

Limits: hazards come from a file, not from a detector on the camera feed; the
simulator stands in for a world model, which a real robot would need; the risk
formula is a transparent heuristic, not a certified hazard analysis.

## Training data from traces

```bash
stepscope export-dataset --run runs/pusht --out dataset.parquet
```

```bash
stepscope situations --run runs/pusht --out situations.json
```

- `export-dataset` writes every step as a transition (state, action, executed
  action, reward, next state, done) with the risk score, a label and a sampling
  weight. Labels: `normal`, `pre_alert` (the run-up to an alert in a failed
  episode, weight 3), `post_alert_failed` (do not imitate, weight 0),
  `recovery` (weight 2), `failed_no_alert`.
- `situations` saves the environment state a few steps before each alert (or
  risk peak) as a library of restart points.
- `ResetToSituation` is a Gymnasium wrapper that starts each episode from one of
  them, so a learner practises the hard moments:

```python
from stepscope.rl import load_situations, reset_to_situation_wrapper
from stepscope.adapters.lerobot_pusht import restore_pusht

env = reset_to_situation_wrapper()(env, load_situations("situations.json"), apply=restore_pusht)
```

Log `reward=` and `snapshot=` in `ep.step()` to enable these. No RL training
loop is included; this is the data and the restart mechanism.

## Tests

```bash
pytest
```

## Roadmap

1. Run a real open policy (LeRobot diffusion policy or a VLA on LIBERO / PushT)
   through the Gymnasium adapter and see which metrics survive contact with it.
   This needs a GPU and is the first real test of the idea.
2. Model-internal signals: token log-probs and attention for VLAs, denoising
   variance for diffusion policies.
3. MCAP export so traces open in Foxglove.
4. Escalation hooks: on alert, call a larger model or hand off to a human.
5. Fleet storage and comparison across policy versions.

## Limitations

- Only the built-in simulator has been run end to end. The Gymnasium adapter is
  tested on Pendulum with a random policy, not on a learned manipulation policy.
- The gate assumes actions are position deltas on the dims you declare. It is a
  software check, not certified functional safety.
- The predictor is linear and trained per task. It has not been tested across tasks.
