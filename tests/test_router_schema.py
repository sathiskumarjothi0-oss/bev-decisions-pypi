from typing import Literal

import pytest
from conftest import FakeModel

import bev


def test_cascade_sends_only_unsure_answers_to_the_strong_model():
    fast, strong = FakeModel(yes=0.95, first=0.5), FakeModel(first=0.9)
    r = bev.Router(fast=fast, strong=strong, threshold=0.8)
    out = r.decide("t", {"u": {"type": "noul", "instructions": "x"},
                         "c": {"type": "choice", "instructions": "x", "criteria": {"a": "a", "b": "b"}}})
    assert out["routing"] == {"u": "fast", "c": "strong"}
    assert out["answers"]["c"]["confidence"] == 0.9


def test_strong_mode_needs_a_strong_model(fake):
    with pytest.raises(ValueError):
        bev.Router(fast=fake, mode="strong")


def test_decide_as_builds_the_object(fake):
    pydantic = pytest.importorskip("pydantic")

    class Ticket(pydantic.BaseModel):
        urgent: bool = pydantic.Field(description="Urgent?")
        team: Literal["billing", "tech"] = pydantic.Field(description="Which team?")

    assert bev.questions_for(Ticket)["team"]["criteria"] == {"billing": "billing", "tech": "tech"}
    t = bev.decide_as(fake, "text", Ticket)
    assert t.urgent is True and t.team == "billing"


def test_evaluate_reports_accuracy(fake):
    q = {"u": {"type": "noul", "instructions": "x"}}
    rep = bev.evaluate(fake, [{"state": "a", "questions": q, "labels": {"u": True}},
                              {"state": "b", "questions": q, "labels": {"u": False}}], threshold=0.8)
    assert rep["all"]["accuracy"] == 0.5
