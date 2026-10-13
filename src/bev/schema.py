"""Schema-driven decisions: describe the decisions as a Pydantic model and get a typed object back.

    from typing import Literal
    from pydantic import BaseModel, Field
    from bev import decide_as

    class Ticket(BaseModel):
        urgent: bool = Field(description="Does the customer ask for urgent handling?")
        team: Literal["billing", "tech", "sales"] = Field(description="Which team should handle this ticket?")

    ticket = decide_as(model, text, Ticket)        # Ticket(urgent=True, team="billing")

bool fields become yes/no questions; Literal[...] and Enum fields become choice questions (their values are the options).
Field descriptions are the instructions (required). Needs pip install "bev-decisions[structured]".
"""

from __future__ import annotations

import enum
import typing
from typing import Any


def questions_for(schema: type) -> dict[str, dict]:
    qs: dict[str, dict] = {}
    for name, f in schema.model_fields.items():  # type: ignore[attr-defined]
        if not f.description:
            raise ValueError(f"field '{name}' needs a description: it is the question Bev is asked")
        t = f.annotation
        if t is bool:
            qs[name] = {"type": "noul", "instructions": f.description}
        elif typing.get_origin(t) is typing.Literal:
            qs[name] = {"type": "choice", "instructions": f.description, "criteria": {str(v): str(v) for v in typing.get_args(t)}}
        elif isinstance(t, type) and issubclass(t, enum.Enum):
            qs[name] = {"type": "choice", "instructions": f.description, "criteria": {str(m.value): m.name.replace("_", " ").lower() for m in t}}
        else:
            raise ValueError(f"field '{name}': use bool, Literal[...] or an Enum")
    return qs


def decide_as(model: Any, state: Any, schema: type, threshold: float | None = None) -> Any:
    """Ask the schema's questions and build the object. With threshold, fields below it are set to None."""
    from .gating import confidence, value

    answers = model.decide(state, questions_for(schema))
    data = {}
    for name, a in answers.items():
        v = value(a) if threshold is None or confidence(a) >= threshold else None
        t = schema.model_fields[name].annotation  # type: ignore[attr-defined]
        data[name] = t(v) if v is not None and isinstance(t, type) and issubclass(t, enum.Enum) else v
    return schema.model_construct(**data) if threshold is not None else schema(**data)  # type: ignore[attr-defined]
