"""The smallest useful call: one yes/no question about one text.

    python examples/01_first_decision.py          # BEV_MODEL=<folder or repo id> to choose the model
"""

import os

import bev

model = bev.load(os.environ.get("BEV_MODEL") or bev.PRO_MODEL)  # downloads once, then works offline

answers = model.decide("I was charged twice for my order, please refund one payment.",
                       {"refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}})
print(answers["refund"]["noul"])  # probability that the answer is yes
