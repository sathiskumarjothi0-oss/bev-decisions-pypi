"""Measure Bev on your own labelled examples before relying on it: accuracy, calibration and the share of answers
above a confidence threshold. Each case: {"state": ..., "questions": {...}, "labels": {name: answer}}.

    python examples/09_evaluate_your_data.py      # or: bev eval my_cases.jsonl --threshold 0.8
"""

import json
import os

import bev

model = bev.load(os.environ.get("BEV_MODEL") or bev.DEFAULT_MODEL)
q = {"refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"}}
cases = [{"state": "Please refund my last payment.", "questions": q, "labels": {"refund": True}},
         {"state": "How do I change my password?", "questions": q, "labels": {"refund": False}},
         {"state": "I want my money back for the broken charger.", "questions": q, "labels": {"refund": True}}]
print(json.dumps(bev.evaluate(model, cases, threshold=0.8), indent=1))
