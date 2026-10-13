"""Automation: triage a support inbox. Each message gets a team, an urgency flag and a sentiment level; confident
answers are routed automatically, the rest go to a person. Writes routing.csv.

    python examples/11_automation_support_inbox.py inbox/        # a folder of .txt or .eml files (one message each)
    python examples/11_automation_support_inbox.py               # runs on three built-in sample messages
"""

import csv
import os
import sys
from pathlib import Path

import bev

SAMPLES = {
    "m1.txt": "Subject: Charged twice for order 88412\nI was billed $49.99 twice. Please refund one payment today.",
    "m2.txt": "Subject: App crashes\nSince the last update the app closes as soon as I open settings.",
    "m3.txt": "Subject: Question\nDo you ship to Ireland, and how long does it take?",
}

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
messages = {p.name: p.read_text(errors="ignore") for p in sorted(folder.glob("*")) if p.suffix in (".txt", ".eml")} if folder else SAMPLES

model = bev.load(os.environ.get("BEV_MODEL") or bev.DEFAULT_MODEL)   # Bev-Large-C: runs on a laptop CPU
questions = bev.presets.get("ticket_triage")                         # department, urgent, sentiment
results = model.decide_batch([(text, questions) for text in messages.values()], batch_size=8)

with open("routing.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["message", "action", "team", "urgent", "sentiment", "confidence"])
    for name, answers in zip(messages, results, strict=True):
        sure, review = bev.gate(answers, threshold=0.8)
        action = f"route to {sure['department']}" if "department" in sure else "send to a person"
        if sure.get("urgent"):
            action += ", flag urgent"
        w.writerow([name, action, bev.value(answers["department"]), bev.value(answers["urgent"]),
                    bev.value(answers["sentiment"]), round(bev.confidence(answers["department"]), 3)])
        print(f"{name}: {action}")
print("wrote routing.csv")
