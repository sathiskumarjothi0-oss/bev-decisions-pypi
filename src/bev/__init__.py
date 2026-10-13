"""Bev: open-weight models that answer typed questions about a document with calibrated probabilities.

    import bev
    model = bev.load()                              # Bev-Large (laptop or GPU); bev.load(bev.PRO_MODEL) for Bev-Pro (GPU)
    answers = model.decide(
        "Invoice INV-2291. Total: $9,412.50. Approval limit: $8,900.",
        {"over_limit": bev.presets.over_limit("$8,900")},
    )
    sure, review = bev.gate(answers, threshold=0.8)

Question types: "noul" (yes/no), "choice" (criteria: {key: description}), "score" (criteria: levels, lowest first).
Up to 16 questions per call, up to 300 options per question, 4,096 input tokens.
"""

from . import presets
from .evaluate import evaluate, evaluate_file
from .gating import confidence, gate, value
from .model import DEFAULT_MODEL, PRO_MODEL, Bev, load
from .router import Router
from .schema import decide_as, questions_for

__all__ = ["DEFAULT_MODEL", "PRO_MODEL", "Bev", "Router", "confidence", "decide_as", "evaluate", "evaluate_file", "gate", "load",
           "presets", "questions_for", "value"]
__version__ = "0.2.3"
