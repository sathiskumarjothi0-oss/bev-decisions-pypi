"""Route questions between a fast model and a stronger one.

    from bev import Router
    router = Router()                        # Beveridge-Large first; unsure answers go to Beveridge-Pro when it is available
    result = router.decide(text, questions)
    result["answers"], result["routing"]     # routing: which model answered each question

Mode "cascade" (default): every question goes to the fast model; answers below `threshold` confidence are asked again
on the strong model, if one is loaded. Mode "fast" / "strong": one model for everything.
"""

from __future__ import annotations

from typing import Any

from .gating import confidence


class Router:
    def __init__(self, fast: Any = None, strong: Any = None, threshold: float = 0.8, mode: str = "cascade",
                 load_strong: bool = False, device: str | None = None) -> None:
        from .model import DEFAULT_MODEL, PRO_MODEL, load

        self.fast = fast if fast is not None else load(DEFAULT_MODEL, device=device)
        self.strong = strong if strong is not None else (load(PRO_MODEL, device=device) if load_strong else None)
        if mode not in ("cascade", "fast", "strong"):
            raise ValueError("mode must be cascade, fast or strong")
        if mode == "strong" and self.strong is None:
            raise ValueError("mode 'strong' needs a strong model (strong=... or load_strong=True)")
        self.threshold, self.mode = threshold, mode

    def decide(self, state: Any, questions: dict[str, dict]) -> dict:
        return self.decide_batch([(state, questions)])[0]

    def decide_batch(self, requests: list[tuple[Any, dict[str, dict]]]) -> list[dict]:
        if self.mode == "strong":
            return [{"answers": a, "routing": dict.fromkeys(a, "strong")} for a in self.strong.decide_batch(requests)]
        first = self.fast.decide_batch(requests)
        out = [{"answers": a, "routing": dict.fromkeys(a, "fast")} for a in first]
        if self.mode == "fast" or self.strong is None:
            return out
        redo = [(i, {n: requests[i][1][n] for n, a in r["answers"].items() if confidence(a) < self.threshold}) for i, r in enumerate(out)]
        redo = [(i, qs) for i, qs in redo if qs]
        if redo:
            for (i, _), ans in zip(redo, self.strong.decide_batch([(requests[i][0], qs) for i, qs in redo]), strict=True):
                for n, a in ans.items():
                    out[i]["answers"][n] = a
                    out[i]["routing"][n] = "strong"
        return out
