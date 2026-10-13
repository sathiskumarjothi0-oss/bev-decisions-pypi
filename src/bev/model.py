"""Load Bev and answer typed questions locally. The network code is in bev/_core."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MAX_QUESTIONS, MAX_OPTIONS = 16, 300
DEFAULT_MODEL = "sathiskumarjothi0/Bev-Large-C"  # Hugging Face repo ids of the released models
PRO_MODEL = "sathiskumarjothi0/Bev-Pro-C"
TYPES = ("noul", "choice", "score")


def _resolve(path_or_repo: str | Path) -> Path:
    p = Path(path_or_repo)
    if (p / "config.json").exists() or (p / "bev_config.json").exists():
        return p
    from huggingface_hub import snapshot_download  # a Hugging Face repo id
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        return Path(snapshot_download(str(path_or_repo)))
    except (RepositoryNotFoundError, GatedRepoError) as e:
        raise RuntimeError(f"model '{path_or_repo}' is not available on Hugging Face (not released yet, private, or misspelled). "
                           f"Released models: {PRO_MODEL} (GPU, ~60 GB) and {DEFAULT_MODEL} (laptop or GPU) once published.") from e


def _item(state: Any, questions: dict[str, dict]) -> dict:
    """Request -> the model's input format. Raises ValueError with a readable message."""
    if not questions:
        raise ValueError("ask at least one question")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"at most {MAX_QUESTIONS} questions per call")
    qs = {}
    for n, q in questions.items():
        t, crit = q.get("type"), q.get("criteria")
        if t not in TYPES:
            raise ValueError(f"question '{n}': type must be one of {TYPES}")
        if not q.get("instructions"):
            raise ValueError(f"question '{n}': instructions are required")
        if t in ("choice", "score") and not crit:
            raise ValueError(f"question '{n}': {t} needs criteria")
        if isinstance(crit, list):
            if t != "score":
                raise ValueError(f"question '{n}': list criteria are only for score questions")
            crit = {str(i): str(c) for i, c in enumerate(crit)}  # numbered levels; names restored in _format
        if crit and len(crit) > MAX_OPTIONS:
            raise ValueError(f"question '{n}': at most {MAX_OPTIONS} options")
        qs[n] = {"type": t, "instructions": q["instructions"], **({"criteria": {str(k): str(v) for k, v in crit.items()}} if crit else {})}
    return {"id": "request", "state": state if isinstance(state, str) else json.dumps(state), "questions": qs}


def _format(item: dict, row: dict, questions: dict[str, dict]) -> dict:
    out = {}
    for n, q in item["questions"].items():
        a = row[n]
        if isinstance(questions[n].get("criteria"), list):  # list levels: report level names, not their numbers
            names = [str(c) for c in questions[n]["criteria"]]
            a = {"pred": names[int(a["pred"])], "probs": {names[int(k)]: v for k, v in a["probs"].items()}}
            q = {**q, "criteria": {nm: nm for nm in names}}
        if q["type"] == "noul":
            p = float(a["probs"]["1"])
            out[n] = {"noul": p, "probabilities": {"true": p, "false": 1 - p}}
        elif q["type"] == "choice":
            out[n] = {"choice": a["pred"], "confidence": float(a["probs"][a["pred"]]), "probabilities": a["probs"]}
        else:
            levels = list(q["criteria"])
            out[n] = {"label": a["pred"], "score": sum(i * float(a["probs"][lv]) for i, lv in enumerate(levels)),
                      "confidence": float(a["probs"][a["pred"]]), "probabilities": a["probs"]}
    return out


class Bev:
    """A loaded Bev-Large model. Use bev.load(...) to create one."""

    def __init__(self, folder: Path, device: str | None = None) -> None:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        from ._core.network import BevNetwork

        self.config = json.loads((folder / "config.json").read_text())
        enc = str(folder / "encoder")  # encoder configuration and tokenizer ship with the weights
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        m = BevNetwork(enc)
        state = {k: v for k, v in load_file(str(folder / "model.safetensors")).items() if not k.startswith("noul.")}  # unused head
        m.load_state_dict(state)
        for t, v in self.config["temperatures"].items():
            m.temp[t].fill_(v)
        self.model = m.to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(enc)
        self.max_len = int(self.config["max_len"])

    def decide(self, state: Any, questions: dict[str, dict]) -> dict:
        """Answer the questions about one document. Returns {question name: answer}."""
        return self.decide_batch([(state, questions)])[0]

    def decide_batch(self, requests: list[tuple[Any, dict[str, dict]]], batch_size: int = 8) -> list[dict]:
        """Answer many (state, questions) pairs; faster than calling decide() in a loop."""
        import torch

        from ._core.network import predict

        items = [{**_item(s, q), "id": str(i)} for i, (s, q) in enumerate(requests)]
        with torch.inference_mode():
            rows = predict(self.model, self.tok, items, batch_size, self.max_len)
        by_id = {r["id"]: r["answers"] for r in rows}
        return [_format(it, by_id[it["id"]], q) for it, (_, q) in zip(items, requests, strict=True)]


def load(path_or_repo: str | Path = DEFAULT_MODEL, device: str | None = None):  # type: ignore[no-untyped-def]
    """Load a Bev model from a local folder or a Hugging Face repo id: Bev-Large models (laptop or GPU) or Bev-Pro models
    (GPU; pip install "bev-decisions[pro]"). device: "cuda", "mps" or "cpu" (default: best available)."""
    folder = _resolve(path_or_repo)
    if (folder / "bev_config.json").exists() or json.loads((folder / "config.json").read_text()).get("model_type") == "bev-pro":
        from .pro import BevPro

        return BevPro(folder, device)
    return Bev(folder, device)
