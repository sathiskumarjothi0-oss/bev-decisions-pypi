"""Act automatically on confident answers and send the rest to a person.
Choose the threshold on a few hundred labelled examples of your own (see 09_evaluate_your_data.py).

    python examples/05_confidence_gate.py
"""

import os

import bev

model = bev.load(os.environ.get("BEV_MODEL") or bev.DEFAULT_MODEL)

answers = model.decide("Hi, can you tell me if the premium plan includes phone support? Thanks", {
    "team": {"type": "choice", "instructions": "Which team should handle this message?",
             "criteria": {"billing": "payments, invoices, refunds", "tech": "bugs, outages, errors", "sales": "pricing and plans"}},
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
})
sure, review = bev.gate(answers, threshold=0.8)
print("act on:", sure)
print("send to a person:", list(review))
