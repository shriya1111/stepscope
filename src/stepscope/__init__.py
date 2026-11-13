"""stepscope: step-level decision tracing, failure prediction and gating for robot policies."""

from .gate import Decision, Gate
from .metrics import TraceConfig
from .predictor import Predictor
from .tracer import EpisodeTrace, Tracer

__all__ = ["Tracer", "EpisodeTrace", "TraceConfig", "Gate", "Decision", "Predictor"]
__version__ = "0.1.0"
