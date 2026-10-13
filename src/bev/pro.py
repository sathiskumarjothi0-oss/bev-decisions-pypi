"""Bev-Pro: a larger Bev built on Google Gemma 4 (Apache-2.0). Same questions and answers as Bev.

    import bev
    pro = bev.load("sathiskumarjothi0/Bev-Pro-C")      # needs: pip install "bev-decisions[pro]" and a GPU with ~60 GB
    pro.decide(text, questions)

A Bev-Pro folder holds bev_config.json (settings and calibration temperatures), the Gemma 4 model files and Bev's
decision weights (adapter/). Everything is loaded from that one folder or Hugging Face repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import MAX_OPTIONS, MAX_QUESTIONS, TYPES

LETTERS = [chr(65 + i) for i in range(26)]


def _options(q: dict) -> list[tuple[str, str]]:
    if q["type"] == "noul":
        return [("1", "yes"), ("0", "no")]
    crit = q["criteria"]
    if isinstance(crit, list):
        return [(str(c), str(c)) for c in crit]
    return [(str(k), str(v)) for k, v in crit.items()]


class BevPro:
    """A loaded Bev-Pro model. Use bev.load(...) to create one."""

    def __init__(self, folder: Path, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local = (folder / "bev_config.json").exists()  # bundled or merged: the base model is in the folder
        self.config = json.loads((folder / ("bev_config.json" if local else "config.json")).read_text())
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        src = str(folder) if local else self.config["base_model"]
        self.tok = AutoTokenizer.from_pretrained(src)
        self.tok.padding_side = "left"
        self.letters = [self.tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
        m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, **({"device_map": "cuda"} if device == "cuda" else {}))
        if not (folder / "adapter").exists():  # merged
            self.model = m.to(device).eval()
        else:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(m, str(folder / "adapter")).to(device).eval()
        self.temps = self.config.get("temperatures", {})
        self.max_len = int(self.config.get("max_len", 4096))
        self.tokens_per_batch = int(self.config.get("tokens_per_batch", 16384))

    def _prompt(self, state: str, instructions: str, opts: list[tuple[str, str]]) -> list[int]:
        tail = ("\n\nQuestion: " + instructions + "\nOptions:\n" + "\n".join(f"{LETTERS[j]}. {d}" for j, (_, d) in enumerate(opts))
                + "\nAnswer with the letter only.")

        def build(doc: str) -> list[int]:
            return self.tok.apply_chat_template([{"role": "user", "content": "Document:\n" + doc + tail}], add_generation_prompt=True,
                                                tokenize=True, return_dict=False)

        ids = build(state)
        if len(ids) <= self.max_len:
            return ids
        doc = self.tok(state, add_special_tokens=False)["input_ids"]
        keep = max(16, len(doc) - (len(ids) - self.max_len) - 8)
        return build(self.tok.decode(doc[:keep]) + " …")[-self.max_len:]

    def _score(self, jobs: list[tuple[str, str, str, list[tuple[str, str]]]]) -> list[list[float]]:
        """jobs: (state, instructions, type, options); returns temperature-scaled logits over each job's options."""
        import torch

        enc = sorted(((i, self._prompt(s, ins, o)) for i, (s, ins, _, o) in enumerate(jobs)), key=lambda x: len(x[1]))
        out: dict[int, list[float]] = {}
        k = 0
        with torch.inference_mode():
            while k < len(enc):
                b = [enc[k]]
                while k + len(b) < len(enc) and (len(b) + 1) * len(enc[k + len(b)][1]) <= self.tokens_per_batch:
                    b.append(enc[k + len(b)])
                k += len(b)
                L = max(len(ids) for _, ids in b)
                inp = torch.full((len(b), L), self.tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(b), L), dtype=torch.long)
                for r, (_, ids) in enumerate(b):
                    inp[r, L - len(ids):] = torch.tensor(ids)
                    att[r, L - len(ids):] = 1
                lg = self.model(input_ids=inp.to(self.device), attention_mask=att.to(self.device), logits_to_keep=1,
                                use_cache=False).logits[:, -1, :][:, self.letters].float().cpu()
                for r, (i, _) in enumerate(b):
                    n = len(jobs[i][3])
                    out[i] = (lg[r, :n] / self.temps.get(jobs[i][2], 1.0)).tolist()
        return [out[i] for i in range(len(jobs))]

    def decide(self, state: Any, questions: dict[str, dict]) -> dict:
        """Answer the questions about one document. Returns {question name: answer}."""
        return self.decide_batch([(state, questions)])[0]

    def decide_batch(self, requests: list[tuple[Any, dict[str, dict]]], batch_size: int = 8) -> list[dict]:
        """Answer many (state, questions) pairs. Questions with more than 26 options are answered as a tournament
        (a winner per group of 26, then a final among the winners)."""
        import torch

        flat = []
        for ri, (state, qs) in enumerate(requests):
            if not qs or len(qs) > MAX_QUESTIONS:
                raise ValueError(f"ask 1 to {MAX_QUESTIONS} questions")
            text = state if isinstance(state, str) else json.dumps(state)
            for n, q in qs.items():
                if q.get("type") not in TYPES or not q.get("instructions"):
                    raise ValueError(f"question '{n}': needs a type in {TYPES} and instructions")
                if q["type"] != "noul" and not q.get("criteria"):
                    raise ValueError(f"question '{n}': {q['type']} needs criteria")
                opts = _options(q)
                if len(opts) > MAX_OPTIONS:
                    raise ValueError(f"question '{n}': at most {MAX_OPTIONS} options")
                flat.append((ri, n, text, q, opts))
        jobs, owners = [], []
        for fi, (_, _, text, q, opts) in enumerate(flat):
            groups = [opts[k:k + 26] for k in range(0, len(opts), 26)]
            for g in groups:
                jobs.append((text, q["instructions"], q["type"], g))
                owners.append(fi)
        first = self._score(jobs)
        winners: dict[int, list] = {}
        logits: dict[int, list[float]] = {}
        for fi, job, lg in zip(owners, jobs, first, strict=True):
            winners.setdefault(fi, []).append((job[3][max(range(len(lg)), key=lg.__getitem__)], lg))
        finals = [(fi, flat[fi]) for fi in winners if len(winners[fi]) > 1]
        if finals:
            second = self._score([(text, q["instructions"], q["type"], [w for w, _ in winners[fi]]) for fi, (_, _, text, q, _) in finals])
            for (fi, _), lg in zip(finals, second, strict=True):
                logits[fi] = lg
        out: list[dict] = [{} for _ in requests]
        for fi, (ri, n, _, q, opts) in enumerate(flat):
            if fi in logits:
                keys = [k for (k, _), _ in winners[fi]]
                p = torch.softmax(torch.tensor(logits[fi]), 0).tolist()
            else:
                keys = [k for k, _ in opts]
                p = torch.softmax(torch.tensor(winners[fi][0][1]), 0).tolist()
            probs = dict(zip(keys, p, strict=True))
            best = max(probs, key=probs.get)
            if q["type"] == "noul":
                out[ri][n] = {"noul": probs.get("1", 0.0), "probabilities": {"true": probs.get("1", 0.0), "false": probs.get("0", 0.0)}}
            elif q["type"] == "choice":
                out[ri][n] = {"choice": best, "confidence": probs[best], "probabilities": probs}
            else:
                levels = [k for k, _ in opts]
                out[ri][n] = {"label": best, "score": sum(levels.index(k) * v for k, v in probs.items()), "confidence": probs[best],
                              "probabilities": probs}
        return out
