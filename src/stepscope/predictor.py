"""Failure predictor: a per-step risk score from the step-level metrics.

Deliberately simple: a standardised logistic regression. It trains in under a
second, runs online in microseconds, and every score decomposes into
per-feature contributions, which is what powers the "why" in the viewer.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

DEFAULT_FEATURES = [
    "sample_spread_ema",
    "direction_disagreement_ema",
    "action_jerk_ema",
    "reversal_rate",
    "path_efficiency",
    "net_speed",
    "plan_regressions",
    "subgoal_dwell",
    "gate_interventions",
    "novelty",
    "elapsed",
]

# Symptoms: (label, kind, [(feature, sign), ...]). The score is the mean of the
# signed z-scores, so "stalled" means: barely moving AND not thrashing about.
SYMPTOMS = [
    ("oscillation", "behaviour", [("action_jerk_ema", 1), ("reversal_rate", 1)]),
    ("stalled", "behaviour", [("net_speed", -1), ("action_jerk_ema", -1)]),
    ("retry loop", "behaviour", [("plan_regressions", 1)]),
    ("stuck on a subgoal", "behaviour", [("subgoal_dwell", 1)]),
    ("fighting the gate", "behaviour", [("gate_interventions", 1)]),
    ("plan keeps changing", "behaviour", [("plan_shift", 1)]),
    ("torn between options", "factor", [("n_modes", 1)]),
    ("sampled plans far apart", "factor", [("final_spread", 1)]),
    ("policy uncertainty", "factor", [("sample_spread_ema", 1)]),
    ("indecision between actions", "factor", [("direction_disagreement_ema", 1)]),
    ("unfamiliar state", "factor", [("novelty", 1)]),
    ("over time budget", "factor", [("elapsed", 1)]),
]


def first_alert(risk: np.ndarray, threshold: float, sustain: int) -> int | None:
    """First step at which risk has been >= threshold for `sustain` steps in a row."""
    streak = 0
    for t, r in enumerate(risk):
        streak = streak + 1 if (not np.isnan(r) and r >= threshold) else 0
        if streak >= sustain:
            return t
    return None


def _sustained_peak(risk: np.ndarray, sustain: int) -> float:
    """Highest threshold at which this risk trace would still raise an alert."""
    r = np.nan_to_num(risk, nan=-np.inf)
    if len(r) < sustain:
        return -np.inf
    windows = np.lib.stride_tricks.sliding_window_view(r, sustain)
    return float(windows.min(axis=1).max())


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):  # average ranks over ties
        idx = scores == v
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class Predictor:
    def __init__(self, features: list[str] | None = None, sustain: int = 3):
        self.features = list(features or DEFAULT_FEATURES)
        self.sustain = sustain
        self.mean = np.zeros(len(self.features))
        self.std = np.ones(len(self.features))
        self.coef = np.zeros(len(self.features))
        self.intercept = 0.0
        self.threshold = 0.5
        self.novelty_mean: np.ndarray | None = None
        self.novelty_prec: np.ndarray | None = None
        self.info: dict = {}
        # Name of an `internals` key (e.g. "embedding") to measure novelty on,
        # instead of the raw state. Set before fit().
        self.novelty_source: str | None = None

    # ---------- featurisation ----------
    def novelty(self, states: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of each state from the states seen in successful runs."""
        states = np.atleast_2d(states)
        if self.novelty_mean is None:
            return np.full(len(states), np.nan)
        d = states - self.novelty_mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.novelty_prec, d))

    def _matrix(self, metrics: dict[str, np.ndarray], states: np.ndarray) -> np.ndarray:
        n = len(states)
        cols = []
        for f in self.features:
            if f == "novelty":
                cols.append(self.novelty(states))
            else:
                cols.append(np.asarray(metrics.get(f, np.full(n, np.nan)), float))
        return np.stack(cols, axis=1)

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std
        return np.nan_to_num(Z, nan=0.0)  # missing value -> training mean

    # ---------- training ----------
    def fit(self, episodes: list[dict], false_alarm_rate: float = 0.05, l2: float = 1.0, seed: int = 0) -> "Predictor":
        """episodes: dicts with `metrics` (name -> array), `states` (T x D), `success` (bool)."""
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(episodes))
        n_cal = max(1, len(episodes) // 5)
        cal = [episodes[i] for i in order[:n_cal]]
        train = [episodes[i] for i in order[n_cal:]]

        if "novelty" in self.features:
            good = [e["states"] for e in train if e["success"]]
            if good:
                S = np.concatenate(good)
                self.novelty_mean = S.mean(axis=0)
                cov = np.cov(S, rowvar=False) + 1e-6 * np.eye(S.shape[1])
                self.novelty_prec = np.linalg.pinv(cov)

        X = np.concatenate([self._matrix(e["metrics"], e["states"]) for e in train])
        y = np.concatenate([np.full(len(e["states"]), 0.0 if e["success"] else 1.0) for e in train])
        with warnings.catch_warnings():  # a feature that was never logged is all-NaN; handled below
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
        self.mean = np.nan_to_num(self.mean, nan=0.0)
        self.std = np.where(np.isnan(self.std) | (self.std < 1e-9), 1.0, self.std)
        Z = np.hstack([self._standardise(X), np.ones((len(X), 1))])

        # Balance classes so the score is not dominated by whichever outcome has more steps.
        pos = max(y.sum(), 1.0)
        neg = max(len(y) - y.sum(), 1.0)
        sw = np.where(y == 1, 0.5 * len(y) / pos, 0.5 * len(y) / neg)
        reg = l2 * np.eye(Z.shape[1])
        reg[-1, -1] = 0.0
        w = np.zeros(Z.shape[1])
        for _ in range(30):  # Newton's method
            p = 1 / (1 + np.exp(-Z @ w))
            g = Z.T @ (sw * (p - y)) + reg @ w
            H = (Z * (sw * p * (1 - p))[:, None]).T @ Z + reg + 1e-8 * np.eye(Z.shape[1])
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.coef, self.intercept = w[:-1], float(w[-1])

        # Pick the alert threshold on held-out episodes: the lowest one that keeps
        # false alarms on successful episodes at or under the target rate.
        peaks = np.array([_sustained_peak(self.score(e["metrics"], e["states"]), self.sustain) for e in cal])
        ok = np.array([e["success"] for e in cal])
        good_peaks = np.sort(peaks[ok])
        if len(good_peaks):
            k = int(np.ceil((1 - false_alarm_rate) * len(good_peaks))) - 1
            self.threshold = float(min(max(good_peaks[min(max(k, 0), len(good_peaks) - 1)] + 1e-6, 0.05), 0.999))
        self.info = {
            "train_episodes": len(train),
            "calibration_episodes": len(cal),
            "train_failure_rate": float(np.mean([not e["success"] for e in train])),
            "target_false_alarm_rate": false_alarm_rate,
        }
        return self

    # ---------- inference ----------
    def score(self, metrics: dict[str, np.ndarray], states: np.ndarray) -> np.ndarray:
        Z = self._standardise(self._matrix(metrics, np.atleast_2d(states)))
        return 1 / (1 + np.exp(-(Z @ self.coef + self.intercept)))

    def score_step(self, metrics: dict[str, float], state: np.ndarray) -> float:
        return float(self.score({k: np.array([v]) for k, v in metrics.items()}, state[None])[0])

    def contributions(self, metrics: dict[str, np.ndarray], states: np.ndarray) -> np.ndarray:
        """Per-feature push on the risk logit, shape (T, n_features). Positive = riskier."""
        return self._standardise(self._matrix(metrics, np.atleast_2d(states))) * self.coef

    def explain(self, metrics: dict[str, np.ndarray], states: np.ndarray, t: int, top: int = 3) -> list[dict]:
        """Why is risk high around step t? Returns the strongest symptoms, strongest first.

        Each symptom is a signature over the metrics, measured in standard
        deviations from what the training runs looked like and averaged over the
        last few steps. "behaviour" symptoms say what the robot is doing wrong and
        are listed before "factor" symptoms, which say what makes it more likely.
        """
        Z = self._standardise(self._matrix(metrics, np.atleast_2d(states)))[max(0, t - 4) : t + 1].mean(axis=0)
        z = dict(zip(self.features, Z))
        out = []
        for label, kind, terms in SYMPTOMS:
            if all(f in z for f, _ in terms):
                score = float(np.mean([sign * z[f] for f, sign in terms]))
                if score > 0.5:
                    out.append({"label": label, "kind": kind, "score": score})
        out.sort(key=lambda o: (o["kind"] != "behaviour", -o["score"]))
        return out[:top]

    # ---------- persistence ----------
    def to_dict(self) -> dict:
        return {
            "features": self.features,
            "sustain": self.sustain,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "threshold": self.threshold,
            "novelty_mean": None if self.novelty_mean is None else self.novelty_mean.tolist(),
            "novelty_prec": None if self.novelty_prec is None else self.novelty_prec.tolist(),
            "info": self.info,
            "novelty_source": self.novelty_source,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "Predictor":
        p = cls(d["features"], d.get("sustain", 3))
        p.mean, p.std, p.coef = np.array(d["mean"]), np.array(d["std"]), np.array(d["coef"])
        p.intercept, p.threshold = d["intercept"], d["threshold"]
        if d.get("novelty_mean") is not None:
            p.novelty_mean, p.novelty_prec = np.array(d["novelty_mean"]), np.array(d["novelty_prec"])
        p.info = d.get("info", {})
        p.novelty_source = d.get("novelty_source")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Predictor":
        return cls.from_dict(json.loads(Path(path).read_text()))
