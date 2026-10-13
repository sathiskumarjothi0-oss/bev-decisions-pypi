"""A score question: ordered levels, lowest first. "score" is the expected level (Σ level index × probability),
so 0.0 means surely the lowest level and len(levels) - 1 surely the highest.

    python examples/03_score_question.py
"""

import os

import bev

model = bev.load(os.environ.get("BEV_MODEL") or bev.PRO_MODEL)

answers = model.decide(
    "ALERT checkout-api p99 latency 4.8 s (SLO 800 ms) for 25 minutes; error rate 12%; customers cannot pay.",
    {"severity": {"type": "score", "instructions": "How severe is this incident?", "criteria": ["low", "medium", "high", "critical"]}},
)
a = answers["severity"]
print(a["label"], round(a["score"], 2), round(a["confidence"], 3))
