"""Evaluate Bev on your own labelled examples before relying on it (accuracy, calibration, confidence threshold).

    bev eval my_cases.jsonl [--model ...] [--threshold 0.8]

Each line of the file: {"state": "...", "questions": {...}, "labels": {"question name": answer}} — answer is true/false
(or 1/0) for yes/no questions, the option key for choice, the level for score.
Reported per question type and overall:
  accuracy      = correct ÷ questions                 (example: 412 of 500 -> 82.4%)
  ECE           = Σ over 10 confidence bins of (share of answers in the bin × |accuracy − mean confidence| in the bin)
                  (example: a bin with 40% of the answers at confidence 0.95 and 80% right adds 0.4 × 0.15 = 0.06)
  at threshold  = share of answers with confidence ≥ threshold, and their accuracy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gating import confidence, value


def evaluate(model: Any, cases: list[dict], threshold: float = 0.8) -> dict:
    answers = model.decide_batch([(c["state"], c["questions"]) for c in cases])
    rows: dict[str, list[tuple[float, int]]] = {}
    for c, ans in zip(cases, answers, strict=True):
        for n, a in ans.items():
            y = c["labels"].get(n)
            if y is None:
                continue
            got = value(a)
            ok = (bool(got) == (y in (True, 1, "1", "true", "yes"))) if "noul" in a else str(got) == str(y)
            rows.setdefault(c["questions"][n]["type"], []).append((confidence(a), int(ok)))
    rows["all"] = [r for t, v in rows.items() for r in v]
    out = {}
    for t, rs in rows.items():
        if not rs:
            continue
        ece = 0.0
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            bin_ = [r for r in rs if lo <= r[0] < hi or (b == 9 and r[0] == 1.0)]
            if bin_:
                ece += len(bin_) / len(rs) * abs(sum(x[1] for x in bin_) / len(bin_) - sum(x[0] for x in bin_) / len(bin_))
        sure = [r for r in rs if r[0] >= threshold]
        out[t] = {"questions": len(rs), "accuracy": round(sum(r[1] for r in rs) / len(rs), 4), "ece": round(ece, 4),
                  f"share_at_{threshold}": round(len(sure) / len(rs), 4),
                  f"accuracy_at_{threshold}": round(sum(r[1] for r in sure) / len(sure), 4) if sure else None}
    return out


def evaluate_file(model: Any, path: str | Path, threshold: float = 0.8) -> dict:
    cases = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
    return evaluate(model, cases, threshold)
