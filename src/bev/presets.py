"""Ready-made question sets, worded as in the question families Beveridge is tested on (Beveridge answers most reliably when a
question is phrased like these). Use them as they are or as a starting point.

    from bev import presets
    answers = model.decide(ticket_text, presets.get("ticket_triage"))
    answers = model.decide(invoice_text, {"over_limit": presets.over_limit("$8,900")})
    presets.names()

Not offered on purpose: invoice payment status and lateness against today's date (Beveridge-Large models are unreliable at
it; compute due dates in your own code).
"""

from __future__ import annotations

import copy

PRESETS: dict[str, dict[str, dict]] = {
    "ticket_triage": {
        "department": {"type": "choice", "instructions": "Which department should handle this message?",
                       "criteria": {"billing": "billing team", "refunds": "refunds team", "shipping": "shipping team",
                                    "technical": "technical team", "account": "account team", "cancellation": "cancellation team"}},
        "urgent": {"type": "noul", "instructions": "Does the customer explicitly ask for urgent or same-day handling?"},
        "sentiment": {"type": "score", "instructions": "What is the customer's sentiment?",
                      "criteria": {"very_negative": "very negative", "negative": "negative", "neutral": "neutral", "positive": "positive", "very_positive": "very positive"}},
    },
    "security_log": {
        "category": {"type": "choice", "instructions": "Which incident category best describes this log?",
                     "criteria": {"brute_force": "many failed logins against accounts, typically from one source",
                                  "phishing": "a user clicked a link in a suspicious email and entered credentials",
                                  "malware": "an unknown executable ran and made outbound connections",
                                  "exfiltration": "unusually large data transfers to an external destination",
                                  "benign": "normal activity, no incident"}},
        "severity": {"type": "score", "instructions": "How severe is this incident?",
                     "criteria": {"low": "routine, no action", "medium": "investigate today", "high": "contain now", "critical": "major incident"}},
    },
    "service_alert": {
        "severity": {"type": "score", "instructions": "How severe is the incident, judged by the highest error rate?",
                     "criteria": {"low": "below 2%", "medium": "2% to 5%", "high": "5% to 8%", "critical": "8% or more"}},
    },
    "request_safety": {
        "refuse": {"type": "noul", "instructions": "Should an AI assistant refuse this request because carrying it out would cause harm?"},
    },
    "review_sentiment": {
        "sentiment": {"type": "choice", "instructions": "Is this product review positive or negative?",
                      "criteria": {"positive": "positive review", "negative": "negative review"}},
    },
}


def names() -> list[str]:
    return sorted(PRESETS)


def get(name: str) -> dict[str, dict]:
    """A copy of the preset's questions (safe to edit)."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}'; available: {', '.join(names())}")
    return copy.deepcopy(PRESETS[name])


def over_limit(limit: str) -> dict:
    """Does an invoice total exceed an approval limit? limit as written, e.g. "$8,900" or "EUR 12.5k"."""
    return {"type": "noul", "instructions": f"Does the invoice total exceed the approval limit of {limit}?"}


def count_above(threshold: str, max_count: int = 5) -> dict:
    """How many line items cost more than a threshold? e.g. count_above("$250.00")."""
    return {"type": "choice", "instructions": f"How many line items cost more than {threshold} each?",
            "criteria": {str(i): f"exactly {i}" for i in range(max_count + 1)}}


def failed_logins_from(ip: str, max_count: int = 5) -> dict:
    """How many failed logins came from one IP address in a log?"""
    return {"type": "choice", "instructions": f"How many failed logins came from IP {ip}?",
            "criteria": {str(i): f"exactly {i}" for i in range(max_count + 1)}}


def mentions(word: str, what: str = "tickets", max_count: int = 5) -> dict:
    """How many items in a list mention a word? e.g. mentions("crash")."""
    return {"type": "choice", "instructions": f"How many {what} mention '{word}'?",
            "criteria": {str(i): f"exactly {i}" for i in range(max_count + 1)}}
