"""Automation: pre-check invoices before approval. Invoices clearly under the limit are approved automatically; anything
over the limit, or any answer below the confidence threshold, goes to a person.

    python examples/12_automation_invoice_approval.py
"""

import os

import bev

LIMIT = "$1,500"
invoices = {
    "INV-2291": "Invoice INV-2291\nLaptop stand $89.00\nMonitor $1,240.00\nDocking station $310.50\nTotal: $1,639.50",
    "INV-2292": "Invoice INV-2292\nKeyboard $79.00\nMouse $29.00\nTotal: $108.00",
}

model = bev.load(os.environ.get("BEV_MODEL") or bev.DEFAULT_MODEL)
questions = {"over_limit": bev.presets.over_limit(LIMIT)}
for number, answers in zip(invoices, model.decide_batch([(t, questions) for t in invoices.values()]), strict=True):
    over, conf = bev.value(answers["over_limit"]), bev.confidence(answers["over_limit"])
    if conf < 0.9:
        decision = "send to a person (not sure)"
    elif over:
        decision = f"needs manager approval (over {LIMIT})"
    else:
        decision = "approve automatically"
    print(f"{number}: {decision}  | confidence {conf:.3f}")
# Due dates and lateness: compute them in code from the dates, not with the model (see Limits in the README).
