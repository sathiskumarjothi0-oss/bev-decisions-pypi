"""The Bev-Large network (inference only).

A ModernBERT encoder reads the questions, their options and the document in one sequence; typed values (amounts, dates,
times, numbers) are added to the token embeddings. Each question becomes a vector that gathers evidence from the
document, is refined over three recursive steps together with the facts read from the text, and scores its options.
The three steps are combined with learned halting weights, and the per-type temperature turns scores into calibrated
probabilities.
"""

from __future__ import annotations

import bisect
import math
import re

import torch
from torch import nn

from .features import FACT_DIM, FACT_KINDS, VAL_DIM, extract, facts, option_feats, state_facts, value_features

D = 256  # width of the decision layers
OPT_FEATS = 5
STEPS = 3
QTYPES = {"noul": 0, "choice": 1, "score": 2}
MIN_STATE_TOKENS = 160  # long option lists are shortened so that at least this much of the document fits
ALL_QUESTIONS = 15  # fact tag for facts about the document as a whole


class Prompt:
    """One request as a single token sequence: [Q1] type: instructions [O] option ... [STATE] document."""

    def __init__(self, tok, item: dict, max_len: int) -> None:
        self.names = list(item["questions"])
        qs = item["questions"]
        self.late: set[str] = set()  # questions whose options are encoded separately (too many to fit)
        self.compact: set[str] = set()  # questions whose option descriptions are shortened
        while True:
            self._build(item)
            n_head = len(tok(self.head, add_special_tokens=False)["input_ids"])
            joint = [n for n in self.names if qs[n]["type"] != "noul" and n not in self.late]
            if n_head <= max_len - MIN_STATE_TOKENS or not joint:
                break
            big = max(joint, key=lambda n: len(qs[n]["criteria"]))
            (self.late if big in self.compact else self.compact).add(big)
        self.state_off = len(self.head)
        enc = tok(self.head + item["state"], truncation=True, max_length=max_len, return_offsets_mapping=True)
        self.ids, self.offsets = enc["input_ids"], enc["offset_mapping"]
        self.ends, m = [], 0
        for _, e in self.offsets:
            m = max(m, e)
            self.ends.append(m)
        self.values = [(v, self.state_off) for v in extract(item["state"])]
        for base, txt in self.value_texts:
            self.values += [(v, base) for v in extract(txt)]

    def _build(self, item: dict) -> None:
        parts: list[str] = []
        pos = 0
        self.qspans, self.instr_off, self.ospans, self.value_texts = [], [], {}, []

        def add(s: str) -> int:
            nonlocal pos
            start = pos
            parts.append(s)
            pos += len(s)
            return start

        for i, n in enumerate(self.names):
            q = item["questions"][n]
            add(f"[Q{i + 1}] {q['type']}: ")
            s = add(q["instructions"])
            self.instr_off.append(s)
            self.value_texts.append((s, q["instructions"]))
            self.qspans.append((s, pos))
            crit = q.get("criteria")
            if q["type"] == "noul":
                c = crit if isinstance(crit, dict) else {}
                spans = []
                for txt in (f"yes: {c.get('true', 'true')}", f"no: {c.get('false', 'false')}"):
                    add(" [O] ")
                    o = add(txt)
                    spans.append((o, o + len(txt)))
                self.ospans[n] = spans
            elif n not in self.late:
                spans = []
                for k, desc in crit.items():
                    add(" [O] ")
                    txt = f"{k}: {desc}"
                    if n in self.compact:
                        same = re.sub(r"[\W_]+", " ", str(k)).strip().lower() == re.sub(r"[\W_]+", " ", str(desc)).strip().lower()
                        txt = str(k) if same else f"{k}: {' '.join(str(desc).split()[:6])}"
                    o = add(txt)
                    spans.append((o, o + len(txt)))
                    self.value_texts.append((o, txt))
                self.ospans[n] = spans
            add(" ")
        add("[STATE] ")
        self.head = "".join(parts)

    def tokens_in(self, a: int, b: int) -> list[int]:
        """Token indices covering characters a..b of the prompt."""
        out = []
        k = bisect.bisect_right(self.ends, a)
        while k < len(self.offsets) and self.offsets[k][0] < b:
            s, e = self.offsets[k]
            if e > s:
                out.append(k)
            k += 1
        return out


class BevNetwork(nn.Module):
    """Bev-Large: ModernBERT-large encoder plus the decision layers. Build it from the encoder configuration that ships
    with the weights (no download), then load model.safetensors."""

    def __init__(self, encoder_dir: str) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.enc = AutoModel.from_config(AutoConfig.from_pretrained(encoder_dir), attn_implementation="sdpa")
        H = self.enc.config.hidden_size
        self.H, self.pad_id = H, self.enc.config.pad_token_id
        self.val_proj = nn.Linear(VAL_DIM, H)
        self.q_proj, self.o_proj, self.o_late = nn.Linear(H, D), nn.Linear(H, D), nn.Linear(H, D)
        self.qtype, self.qid = nn.Embedding(3, D), nn.Embedding(16, D)
        self.ev_k, self.ev_v, self.ev_out = nn.Linear(H, D), nn.Linear(H, D), nn.Linear(2 * D, D)
        self.fact_in = nn.Sequential(nn.Linear(FACT_DIM + len(FACT_KINDS) + 2 * H, D), nn.GELU(), nn.Linear(D, D))
        self.core = nn.TransformerEncoderLayer(D, 8, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.core_gate = nn.Parameter(torch.zeros(1))
        self.halt = nn.Linear(D, 1)
        self.opt_mlp = nn.Sequential(nn.Linear(3 * D, D), nn.GELU(), nn.Linear(D, 1))
        self.op_gate, self.opt_lex = nn.Linear(D, 1), nn.Linear(OPT_FEATS, 1)
        self.fact_att = nn.Linear(D, D, bias=False)
        self.num_w = nn.Parameter(torch.tensor(1.0))
        self.fact_q, self.fact_mix = nn.Linear(D, D, bias=False), nn.Linear(D, D)
        self.dec_sel = nn.Sequential(nn.Linear(3 * D, D), nn.GELU(), nn.Linear(D, 1))
        self.dec_none, self.dec_w = nn.Linear(D, 1), nn.Linear(D, 1)
        self.temp = nn.ParameterDict({t: nn.Parameter(torch.tensor(1.0), requires_grad=False) for t in QTYPES})

    # yes/no questions whose answer follows from an exact comparison of numbers or dates found in the text
    def _exact_yes_no(self, base: torch.Tensor, zq: torch.Tensor, Fc: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        zz = zq.unsqueeze(0).expand(Fc.shape[0], -1)
        a = torch.softmax(torch.cat([self.dec_none(zq), self.dec_sel(torch.cat([zz, Fc, zz * Fc], -1)).squeeze(-1)]), 0)
        comp = torch.sigmoid(torch.cat([base, base + self.dec_w(zq) * (2 * bits - 1)]))
        p = (a * comp).sum().clamp(1e-06, 1 - 1e-06)
        return torch.log(p / (1 - p)).reshape(1)

    def _facts(self, it: dict, e: Prompt, L: int):
        nq = len(e.names)
        fl: list[tuple[int, object]] = [(ALL_QUESTIONS, f) for f in state_facts(it["state"])]
        for i, n in enumerate(e.names):
            fl += [(i, f) for f in facts(it["state"], it["questions"][n]["instructions"])]
        if not fl:
            return None
        feats = torch.zeros(len(fl), FACT_DIM + len(FACT_KINDS))
        smat = torch.zeros(2 * len(fl), L)
        tags, res_v = [], []
        for r_, (i, f) in enumerate(fl):
            feats[r_] = torch.tensor(f.feats + [float(f.kind == k) for k in FACT_KINDS])
            wheres = {"state": ("state", "state"), "both": ("state", "question"), "question": ("question", "question")}[f.where]
            for sidx, ((s0, s1), where) in enumerate(zip(f.spans[:2], wheres, strict=False)):
                base = e.state_off if where == "state" else e.instr_off[i if i < nq else 0]
                tk = e.tokens_in(base + s0, base + s1)
                if tk:
                    smat[2 * r_ + sidx, tk] = 1.0 / len(tk)
            tags.append(i)
            res_v.append(f.result if f.result is not None else float("nan"))
        return feats, smat, tags, res_v

    def _encode(self, prompts: list[Prompt], dev: torch.device, L: int) -> torch.Tensor:
        ids = torch.full((len(prompts), L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(prompts), L), dtype=torch.long)
        vfeat = torch.zeros((len(prompts), L, VAL_DIM))
        for b, e in enumerate(prompts):
            ids[b, : len(e.ids)] = torch.tensor(e.ids)
            mask[b, : len(e.ids)] = 1
            for v, off in e.values:
                for k in e.tokens_in(off + v.start, off + v.end):
                    vfeat[b, k] = torch.tensor(value_features(v))
        ids, mask, vfeat = ids.to(dev), mask.to(dev), vfeat.to(dev)
        emb = self.enc.get_input_embeddings()(ids) + self.val_proj(vfeat)
        return self.enc(inputs_embeds=emb, attention_mask=mask).last_hidden_state

    def _encode_options(self, texts: list[str], tok, dev: torch.device) -> torch.Tensor:
        """Mean-pooled encoder vectors of option texts encoded on their own (questions with very many options)."""
        uniq = sorted(set(texts))
        out = {}
        for i in range(0, len(uniq), 256):
            chunk = uniq[i : i + 256]
            n_real = len(chunk)
            chunk = chunk + [""] * (next(bk for bk in (32, 64, 128, 256) if bk >= n_real) - n_real)
            t = tok(chunk, padding="max_length", truncation=True, max_length=32, return_tensors="pt").to(dev)
            h = self.enc(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state
            pooled = (h * t["attention_mask"].unsqueeze(-1)).sum(1) / t["attention_mask"].sum(1, keepdim=True)
            for s, v in zip(chunk[:n_real], pooled[:n_real], strict=True):
                out[s] = v
        return torch.stack([out[s] for s in texts])

    def forward(self, items: list[dict], tok, max_len: int = 4096) -> list[dict[str, torch.Tensor]]:
        """Per request: {question: logits per step, "__halt__": halting weights per step and question}."""
        dev = next(self.parameters()).device
        prompts = [Prompt(tok, it, max_len) for it in items]
        L = max(256, -(-max(len(e.ids) for e in prompts) // 256) * 256)
        fdata = [self._facts(it, e, L) for it, e in zip(items, prompts, strict=True)]
        hid = self._encode(prompts, dev, L)
        late_texts, late_index = [], []
        for b, (it, e) in enumerate(zip(items, prompts, strict=True)):
            for n in e.late:
                for k, desc in it["questions"][n]["criteria"].items():
                    late_index.append((b, n))
                    late_texts.append(f"{k}: {desc}")
        late_emb = self.o_late(self._encode_options(late_texts, tok, dev)) if late_texts else None
        late_map: dict[tuple[int, str], list[int]] = {}
        for j, key in enumerate(late_index):
            late_map.setdefault(key, []).append(j)
        outs = []
        for b, it in enumerate(items):
            e, h = prompts[b], hid[b]
            nq = len(e.names)
            rows = [e.tokens_in(*sp) or [0] for sp in e.qspans]
            orows, owner = [], []
            for i, n in enumerate(e.names):
                for sp in e.ospans.get(n, []):
                    orows.append(e.tokens_in(*sp) or [0])
                    owner.append(i)
            pm = torch.zeros(nq + len(orows), L)
            for r_, toks in enumerate(rows + orows):
                pm[r_, toks] = 1.0 / len(toks)
            pooled = pm.to(dev) @ h
            qtypes = torch.tensor([QTYPES[it["questions"][n]["type"]] for n in e.names], device=dev)
            q = self.q_proj(pooled[:nq]) + self.qid.weight[:nq] + self.qtype(qtypes)
            o_joint = self.o_proj(pooled[nq:]) if orows else None
            valid = torch.zeros(L, dtype=torch.bool, device=dev)
            valid[: len(e.ids)] = True
            att = torch.softmax((q @ self.ev_k(h).T / math.sqrt(D)).masked_fill(~valid.unsqueeze(0), float("-inf")), -1)
            q = self.ev_out(torch.cat([q, att @ self.ev_v(h)], -1))  # question + evidence from the document
            F = fq_t = fres_t = None
            if fdata[b] is not None:
                feats, smat, tags, res_v = fdata[b]
                span_h = (smat.to(dev) @ h).view(len(tags), 2 * self.H)
                F = self.fact_in(torch.cat([feats.to(dev), span_h], 1)) + self.qid.weight[torch.tensor(tags, device=dev)]
                fq_t = torch.tensor(tags, device=dev)
                fres_t = torch.tensor(res_v, device=dev, dtype=torch.float32)
            x = (torch.cat([q, F]) if F is not None else q).unsqueeze(0)
            zs, lams = [], []
            for _ in range(STEPS):  # recursive refinement with learned halting
                x = self.core(x)
                zs.append(q + self.core_gate * x[0, :nq])
                lams.append(torch.sigmoid(self.halt(zs[-1])).squeeze(-1))
            remain, ps = torch.ones(nq, device=dev), []
            for k, lam in enumerate(lams):
                ps.append(remain * (lam if k < len(lams) - 1 else torch.ones_like(lam)))
                remain = remain * (1 - lam)
            p_halt = torch.stack(ps)
            dec_ok = dec_bits = None
            if F is not None:  # each question reads the facts that belong to it
                own = (fq_t.unsqueeze(0) == torch.arange(nq, device=dev).unsqueeze(1)) | (fq_t == ALL_QUESTIONS).unsqueeze(0)
                has = own.any(1, keepdim=True)
                zs = [
                    z
                    + has.float()
                    * self.fact_mix(
                        torch.softmax((self.fact_q(z) @ F.T / math.sqrt(D)).masked_fill(~own, float("-inf")), -1).nan_to_num(0.0) @ F
                    )
                    for z in zs
                ]
                ft = fdata[b][0].to(dev)
                kind = ft[:, FACT_DIM : FACT_DIM + len(FACT_KINDS)]
                dec_ok = (
                    (kind[:, FACT_KINDS.index("cmp")] > 0)
                    | (kind[:, FACT_KINDS.index("pair")] > 0)
                    | (kind[:, FACT_KINDS.index("sum")] > 0) & (ft[:, 4] > 0)
                )
                dec_bits = ft[:, 0]
            res: dict[str, torch.Tensor] = {}
            for i, n in enumerate(e.names):
                qd = it["questions"][n]
                if qd["type"] == "noul":
                    o = o_joint[torch.tensor([j for j, w in enumerate(owner) if w == i], device=dev)]
                else:
                    o = (
                        late_emb[torch.tensor(late_map[b, n], device=dev)]
                        if n in e.late
                        else o_joint[torch.tensor([j for j, w in enumerate(owner) if w == i], device=dev)]
                    )
                    ofe = torch.tensor(option_feats(it["state"], list(qd["criteria"])), dtype=torch.float32, device=dev)
                step_logits = []
                for z in zs:
                    zq = z[i]
                    zz = zq.unsqueeze(0).expand(o.shape[0], -1)
                    lg = self.opt_mlp(torch.cat([zz, o, zz * o], -1)).squeeze(-1)
                    if qd["type"] == "noul":
                        base = (lg[0] - lg[1]).reshape(1)
                        if dec_ok is not None:
                            cand = dec_ok & ((fq_t == i) | (fq_t == ALL_QUESTIONS))
                            if bool(cand.any()):
                                base = self._exact_yes_no(base, zq, F[cand], dec_bits[cand])
                        step_logits.append(base)
                        continue
                    lex = ofe.clone()
                    lex[:, 2] = 0
                    op = self.opt_lex(lex).squeeze(-1)
                    numeric = ofe[:, 4] > 0
                    if F is not None and bool(numeric.any()):  # numeric options matched against computed results
                        sel = (fq_t == i) & ~torch.isnan(fres_t)
                        if bool(sel.any()):
                            a_ = torch.softmax(F[sel] @ self.fact_att(zq) / math.sqrt(D), 0)
                            match = torch.exp(-(ofe[:, 2:3] - fres_t[sel].unsqueeze(0)).abs())
                            op = op + self.num_w * numeric.float() * (match @ a_) * 10
                    step_logits.append(lg + torch.sigmoid(self.op_gate(zq)) * op)
                res[n] = torch.stack(step_logits)
            res["__halt__"] = p_halt
            outs.append(res)
        return outs


def probabilities(model: BevNetwork, out: dict[str, torch.Tensor], qtype: str, name: str) -> torch.Tensor:
    """Calibrated probabilities of one question: per step, temperature-scaled; steps combined by their halting weights."""
    i = [k for k in out if k != "__halt__"].index(name)
    w = out["__halt__"][:, i : i + 1]
    t = model.temp[qtype]
    if qtype == "noul":
        return (w.squeeze(-1) * torch.sigmoid(out[name].squeeze(-1) / t)).sum().reshape(1)
    return (w * torch.softmax(out[name] / t, -1)).sum(0)


def predict(model: BevNetwork, tok, items: list[dict], batch: int, max_len: int) -> list[dict]:
    """[{"id", "answers": {question: {"pred", "probs"}}}] for items {"id", "state", "questions"}."""
    rows = []
    for k in range(0, len(items), batch):
        chunk = items[k : k + batch]
        for it, out in zip(chunk, model(chunk, tok, max_len), strict=True):
            ans = {}
            for n, q in it["questions"].items():
                p = probabilities(model, out, q["type"], n).float().cpu().numpy()
                if q["type"] == "noul":
                    ans[n] = {"pred": int(p[0] >= 0.5), "probs": {"1": float(p[0]), "0": float(1 - p[0])}}
                else:
                    opts = list(q["criteria"])
                    ans[n] = {"pred": opts[int(p.argmax())], "probs": {o: float(v) for o, v in zip(opts, p, strict=True)}}
            rows.append({"id": it["id"], "answers": ans})
    return rows
