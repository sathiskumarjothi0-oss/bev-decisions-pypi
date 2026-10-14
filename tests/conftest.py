"""A fake model with Beveridge's interface, so the tests run without downloading weights."""

import pytest


class FakeModel:
    """Answers every question with fixed probabilities: yes 0.9; the first option 0.7, the rest share 0.3."""

    def __init__(self, yes: float = 0.9, first: float = 0.7) -> None:
        self.yes, self.first, self.calls = yes, first, 0

    def decide(self, state, questions):
        return self.decide_batch([(state, questions)])[0]

    def decide_batch(self, requests, batch_size=8):
        self.calls += 1
        out = []
        for _, qs in requests:
            ans = {}
            for n, q in qs.items():
                if q["type"] == "noul":
                    ans[n] = {"noul": self.yes, "probabilities": {"true": self.yes, "false": 1 - self.yes}}
                    continue
                crit = q["criteria"]
                keys = [str(c) for c in crit] if isinstance(crit, list) else list(crit)
                rest = (1 - self.first) / max(1, len(keys) - 1)
                probs = {k: (self.first if i == 0 else rest) for i, k in enumerate(keys)}
                key = "choice" if q["type"] == "choice" else "label"
                ans[n] = {key: keys[0], "confidence": self.first, "probabilities": probs,
                          **({"score": sum(i * p for i, p in enumerate(probs.values()))} if key == "label" else {})}
            out.append(ans)
        return out


@pytest.fixture
def fake():
    return FakeModel()
