"""Confidence helpers: act on confident answers, send the rest to a person or a larger model.

    from bev import confidence, gate
    sure, review = gate(answers, threshold=0.8)
"""

from __future__ import annotations


def confidence(answer: dict) -> float:
    """The probability of the answer given: for yes/no, the larger of P(yes) and P(no); otherwise the chosen option's."""
    if "noul" in answer:
        return max(answer["noul"], 1 - answer["noul"])
    return float(answer.get("confidence", max(answer["probabilities"].values())))


def value(answer: dict):  # type: ignore[no-untyped-def]
    """The answer itself: True/False for yes/no, the chosen option key, or the score level."""
    if "noul" in answer:
        return answer["noul"] >= 0.5
    return answer.get("choice", answer.get("label"))


def gate(answers: dict[str, dict], threshold: float = 0.8) -> tuple[dict, dict]:
    """Split answers into (confident: {name: value}, review: {name: answer}) at the given confidence."""
    sure, review = {}, {}
    for name, a in answers.items():
        if confidence(a) >= threshold:
            sure[name] = value(a)
        else:
            review[name] = a
    return sure, review
