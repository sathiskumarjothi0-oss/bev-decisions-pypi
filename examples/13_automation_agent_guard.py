"""Automation: a guard in front of an AI agent. Before the agent acts, Beveridge answers plain yes/no questions about the
user's request and the action the agent proposes; your code decides what is allowed. Harmful requests are refused,
actions that move money, delete things or contact people wait for a person, and everything else runs.
(Agents that speak MCP can call the same checks as tools: pip install "bev-decisions[mcp]" && bev mcp.)

    python examples/13_automation_agent_guard.py
"""

import os

import bev

model = bev.load(os.environ.get("BEV_MODEL") or bev.DEFAULT_MODEL)

QUESTIONS = {
    **bev.presets.get("request_safety"),   # "refuse": should an assistant refuse this request?
    "money": {"type": "noul", "instructions": "Does the proposed action pay, transfer or spend money?"},
    "delete": {"type": "noul", "instructions": "Does the proposed action delete or overwrite files or data?"},
    "contact": {"type": "noul", "instructions": "Does the proposed action send a message to another person?"},
}
NEEDS_A_PERSON = ("money", "delete", "contact")


def guard(request: str, proposed_action: str) -> str:
    answers = model.decide(f"User request: {request}\nProposed action: {proposed_action}", QUESTIONS)
    if bev.value(answers["refuse"]) and bev.confidence(answers["refuse"]) >= 0.8:
        return "refuse"
    flagged = [q for q in NEEDS_A_PERSON if bev.value(answers[q]) or bev.confidence(answers[q]) < 0.8]
    return f"ask a person first ({', '.join(flagged)})" if flagged else "run"


print(guard("Find the opening hours of the Swansea library", "search: Swansea central library opening hours"))
print(guard("Pay the supplier invoice INV-2291", "payment: transfer $1,639.50 to the supplier"))
print(guard("Tidy my downloads folder", "files: delete every file in ~/Downloads"))
print(guard("Tell Maria the report is ready", "email: send 'The report is ready' to maria@example.com"))
