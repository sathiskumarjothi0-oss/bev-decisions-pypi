import bev


def test_confidence_and_value():
    yes = {"noul": 0.2, "probabilities": {"true": 0.2, "false": 0.8}}
    assert bev.confidence(yes) == 0.8 and bev.value(yes) is False
    choice = {"choice": "billing", "confidence": 0.65, "probabilities": {"billing": 0.65, "tech": 0.35}}
    assert bev.confidence(choice) == 0.65 and bev.value(choice) == "billing"


def test_gate_splits_at_threshold(fake):
    answers = fake.decide("t", {"u": {"type": "noul", "instructions": "x"},
                                "c": {"type": "choice", "instructions": "x", "criteria": {"a": "a", "b": "b"}}})
    sure, review = bev.gate(answers, threshold=0.8)
    assert sure == {"u": True} and list(review) == ["c"]


def test_presets_are_valid_requests():
    from bev.model import _item

    for name in bev.presets.names():
        _item("text", bev.presets.get(name))
