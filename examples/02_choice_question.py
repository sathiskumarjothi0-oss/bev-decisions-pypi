"""A choice question: Beveridge picks one of the options you describe and gives a probability for each.
Describe each option in a few words; the descriptions matter more than the keys.

    python examples/02_choice_question.py
"""

import os

import bev

model = bev.load(os.environ.get("BEV_MODEL") or bev.DEFAULT_MODEL)

answers = model.decide(
    "The app crashes every time I open the settings page since the last update.",
    {"team": {"type": "choice", "instructions": "Which team should handle this message?",
              "criteria": {"billing": "payments, invoices, refunds", "tech": "bugs, crashes, outages", "sales": "pricing and plans"}}},
)
a = answers["team"]
print(a["choice"], round(a["confidence"], 3))
for option, p in sorted(a["probabilities"].items(), key=lambda kv: -kv[1]):
    print(f"  {option:8s} {p:.3f}")
