import pytest

from bev.model import _format, _item


def test_valid_request_is_converted():
    item = _item("text", {"u": {"type": "noul", "instructions": "Urgent?"},
                          "s": {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]}})
    assert item["state"] == "text"
    assert item["questions"]["s"]["criteria"] == {"0": "low", "1": "high"}


@pytest.mark.parametrize("questions, message", [
    ({}, "at least one"),
    ({"a": {"type": "maybe", "instructions": "x"}}, "type must be"),
    ({"a": {"type": "noul"}}, "instructions are required"),
    ({"a": {"type": "choice", "instructions": "x"}}, "needs criteria"),
    ({"a": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}, "only for score"),
    ({f"q{i}": {"type": "noul", "instructions": "x"} for i in range(17)}, "at most 16"),
    ({"a": {"type": "choice", "instructions": "x", "criteria": {str(i): "o" for i in range(301)}}}, "at most 300"),
])
def test_invalid_requests_are_rejected(questions, message):
    with pytest.raises(ValueError, match=message):
        _item("text", questions)


def test_json_state_is_serialised():
    assert _item({"id": 7}, {"a": {"type": "noul", "instructions": "x"}})["state"] == '{"id": 7}'


def test_score_levels_are_named_in_answers():
    qs = {"s": {"type": "score", "instructions": "How bad?", "criteria": ["low", "medium", "high"]}}
    out = _format(_item("t", qs), {"s": {"pred": "2", "probs": {"0": 0.1, "1": 0.2, "2": 0.7}}}, qs)
    assert out["s"]["label"] == "high"
    assert out["s"]["score"] == pytest.approx(0.2 + 1.4)
