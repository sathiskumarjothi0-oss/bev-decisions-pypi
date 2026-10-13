"""Describe the decisions as a Pydantic model and get a typed object back.
bool -> yes/no question; Literal[...] or Enum -> choice question. The field description is the question.

    pip install "bev-decisions[structured]"
    python examples/06_pydantic_schema.py
"""

import os
from typing import Literal

from pydantic import BaseModel, Field

import bev


class Ticket(BaseModel):
    urgent: bool = Field(description="Does the customer ask for urgent handling?")
    team: Literal["billing", "tech", "sales"] = Field(description="Which team should handle this ticket?")


model = bev.load(os.environ.get("BEV_MODEL") or bev.PRO_MODEL)
print(bev.decide_as(model, "Charged twice, refund one payment today please!", Ticket))
