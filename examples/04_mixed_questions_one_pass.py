"""Several questions of different types about one document, answered together (up to 16 per call).

    python examples/04_mixed_questions_one_pass.py
"""

import json
import os

import bev

model = bev.load(os.environ.get("BEV_MODEL") or bev.PRO_MODEL)

ticket = """From: maria@example.com
Subject: Charged twice for order 88412
I was billed $49.99 twice on 3 October for the same order. Please refund one payment today, this is urgent."""

answers = model.decide(ticket, {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "payments, invoices, refunds", "tech": "bugs, outages, errors", "sales": "pricing and plans"}},
    "urgent": {"type": "noul", "instructions": "Does the customer ask for urgent handling?"},
    "severity": {"type": "score", "instructions": "How severe is the problem for the customer?", "criteria": ["low", "medium", "high"]},
})
print(json.dumps(answers, indent=1))
