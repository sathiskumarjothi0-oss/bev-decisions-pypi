"""Many documents at once: faster than calling decide() in a loop.

    python examples/08_batch.py
"""

import os

import bev

model = bev.load(os.environ.get("BEV_MODEL") or bev.DEFAULT_MODEL)
questions = bev.presets.get("ticket_triage")
texts = ["My parcel has not arrived after two weeks.", "Please cancel my subscription.", "The login page shows error 500."]
for text, answers in zip(texts, model.decide_batch([(t, questions) for t in texts], batch_size=8), strict=True):
    print(text, "->", bev.value(answers["department"]))
