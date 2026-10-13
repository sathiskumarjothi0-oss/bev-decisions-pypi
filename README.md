<div align="center">

# Bev

**System One decision models with open weights.** Typed yes/no, choice and score decisions over any text, with a
calibrated probability for every option, in one pass, on your own machines.

[![PyPI](https://img.shields.io/pypi/v/bev-decisions?label=PyPI)](https://pypi.org/project/bev-decisions/)
[![Python](https://img.shields.io/pypi/pyversions/bev-decisions)](https://pypi.org/project/bev-decisions/)
[![Bev-Pro-C](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Bev--Pro--C-yellow)](https://huggingface.co/sathiskumarjothi0/Bev-Pro-C)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-sjothi-FFDD00?logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/sjothi)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

Give Bev a document (text, email, ticket, log or JSON) and typed questions. It returns typed answers: yes/no (`noul`),
one of the options you describe (`choice`), or a level on a scale you define (`score`), each with calibrated
probabilities. It never generates text, so there is nothing to parse and nothing to hallucinate, and it runs locally:
your documents never leave your machines.

| Model | Size | Runs on | Best for | Status |
|---|---|---|---|---|
| Bev-Large-C | 0.4B (ModernBERT-large base) | laptop CPU/GPU, edge boxes, any GPU | operations documents, logs and alerts, counting, amounts and limits, ticket routing, request safety | coming soon |
| [Bev-Pro-C](https://huggingface.co/sathiskumarjothi0/Bev-Pro-C) | 26B (Gemma 4 26B-A4B base) | one GPU with about 60 GB | knowledge-heavy decisions: contracts, finance topics, large intent sets, dates and payment status | available |


## Installation

```bash
python -m pip install "bev-decisions[pro]"     # Bev-Pro-C (GPU with about 60 GB)
python -m pip install bev-decisions            # Bev-Large-C (laptop or any GPU)
```

Python 3.10 or newer. The first `bev.load()` downloads the model from Hugging Face and caches it; after that it works
offline. Optional extras:

| Extra | Adds |
|---|---|
| `bev-decisions[pro]` | Bev-Pro models (GPU) |
| `bev-decisions[serve]` | the self-hosted HTTP server |
| `bev-decisions[mcp]` | the MCP server for AI assistants and agents |
| `bev-decisions[structured]` | schema-driven decisions with Pydantic |
| `bev-decisions[all]` | everything above |

## Quickstart

```python
import bev

model = bev.load(bev.PRO_MODEL)   # sathiskumarjothi0/Bev-Pro-C

answers = model.decide(
    "Subject: Charged twice for order 88412. I was billed $49.99 twice on 3 October. Please refund one payment today, it's urgent.",
    {
        "team":   {"type": "choice", "instructions": "Which department should handle this message?",
                   "criteria": {"billing": "billing team", "refunds": "refunds team", "technical": "technical team"}},
        "urgent": {"type": "noul",   "instructions": "Does the customer explicitly ask for urgent or same-day handling?"},
        "severity": {"type": "score", "instructions": "How severe is the problem for the customer?",
                     "criteria": ["low", "medium", "high"]},
    },
)
answers["team"]      # {"choice": ..., "confidence": ..., "probabilities": {...}}
answers["urgent"]    # {"noul": P(yes), "probabilities": {"true": ..., "false": ...}}
answers["severity"]  # {"label": ..., "score": expected level, "confidence": ..., "probabilities": {...}}
```

More in [examples/](https://github.com/sathiskumarjothi0-oss/bev-decisions-pypi/tree/main/examples/): ten short scripts from a first yes/no question to an HTTP client.

## Decision primitives

| Type | Use it for | Answer |
|---|---|---|
| `noul` | yes or no | `noul`: probability of yes |
| `choice` | one of a set you define (`criteria`: key → description, up to 300) | `choice`, `confidence`, `probabilities` |
| `score` | ordered levels (`criteria`: list or key → description, lowest first) | `label`, `score` (expected level: Σ index × probability), `confidence`, `probabilities` |

Up to 16 questions per call share one read of the document. Keep each question narrow and combine the answers in your
code. Describe options in a few words that tell them apart, add an "other" option when nothing may fit, and keep
the option order stable between calls.

## Confidence gating

```python
sure, review = bev.gate(answers, threshold=0.8)   # sure: {name: value}; review: answers below the threshold
if "team" in sure:
    route(ticket, sure["team"])
else:
    send_to_person(ticket)
```

Choose the threshold on a few hundred labelled examples of your own (see [Evaluate on your data](#evaluate-on-your-data)).

## Route mode

```python
router = bev.Router(load_strong=True, threshold=0.8)    # Bev-Large-C first, unsure answers asked again on Bev-Pro-C
result = router.decide(text, questions)
result["answers"], result["routing"]                    # routing: {"team": "fast", "urgent": "strong"}
```

Modes: `"cascade"` (default), `"fast"`, `"strong"`. Route mode needs Bev-Large-C (coming soon).

## Schema-driven decisions

```python
from typing import Literal
from pydantic import BaseModel, Field

class Ticket(BaseModel):
    urgent: bool = Field(description="Does the customer explicitly ask for urgent or same-day handling?")
    team: Literal["billing", "technical", "shipping"] = Field(description="Which department should handle this message?")

ticket = bev.decide_as(model, text, Ticket)     # Ticket(urgent=True, team='billing')
```

## Workflow presets

```python
bev.presets.names()   # ['request_safety', 'review_sentiment', 'security_log', 'service_alert', 'ticket_triage']
model.decide(text, bev.presets.get("security_log"))
model.decide(invoice, {"over_limit": bev.presets.over_limit("$8,900"), "big": bev.presets.count_above("$250.00")})
```

## Batch processing

```python
results = model.decide_batch([(text, questions) for text in texts], batch_size=8)
```

## Command line

```bash
bev "I was charged twice, please refund" --noul "Does the customer ask for a refund?" --model sathiskumarjothi0/Bev-Pro-C
bev --file ticket.txt --preset ticket_triage --min-confidence 0.8
bev --file alert.txt --score "How severe is this?" --levels low medium high
bev --batch tickets.txt --preset ticket_triage
bev presets
```

## Self-hosting

```bash
pip install "bev-decisions[serve]"
bev serve --model sathiskumarjothi0/Bev-Pro-C --port 8000     # POST /v1/decide, /v1/batch; GET /v1/presets, /health
docker compose -f compose.cuda.yaml up                         # the same server in a GPU container
pip install "bev-decisions[mcp]"
bev mcp --model sathiskumarjothi0/Bev-Pro-C                    # MCP tools: decide, decide_preset, list_presets
```

| Method and path | Body | Response |
|---|---|---|
| `POST /v1/decide` | `{"state": ..., "questions": {...}}` or `{"state": ..., "preset": "ticket_triage"}` | `{"answers": {...}}` |
| `POST /v1/batch` | `{"requests": [{"state": ..., "questions": ...}, ...]}` | `{"results": [{"answers": ...}, ...]}` |
| `GET /v1/presets`, `GET /health` | | preset names; `{"ok": true}` |

MCP client configuration: `{"mcpServers": {"bev": {"command": "bev", "args": ["mcp", "--model", "sathiskumarjothi0/Bev-Pro-C"]}}}`.
The server listens on 127.0.0.1 and has no authentication; put it behind your own gateway (TLS, authentication, rate
limits) before exposing it.

## Benchmarks

Accuracy (%) = correct answers ÷ questions on held-out test sets; the same cases for every system (October 2026).
Bold = best in the row; – = not run. * = an area Bev was not built for. Question-by-question results are on the
Hugging Face model card.

### Decisions on documents

![Decisions on documents](https://huggingface.co/sathiskumarjothi0/Bev-Pro-C/resolve/main/assets/benchmark.png)

| Test | Bev-Pro-C | Jev | Mistral Large 4 | Mistral Large 3 | gpt-oss 120B | Gemma 4 31B | Nemotron 3 Super |
|---|---|---|---|---|---|---|---|
| Bev-Bench (operations documents) | **91.9** | 87.9 | 83.7 | 74.6 | 91.1 | 88.9 | 70.6 |
| Typed decisions | **79.2** | 73.8 | 72.0 | 68.0 | 68.8 | 75.0 | 62.7 |
| Rules-Bench (rules shown) | 92.0 | **92.3** | – | – | – | – | – |
| Aegis 2.0 request safety | **78.2** | 71.5 | 70.5 | 66.9 | 69.5 | 73.1 | 69.3 |
| JailbreakBench refusal | 84.0 | 83.5 | 77.5 | 62.0 | 68.0 | **87.5** | 65.0 |

### Public classification tests

![Public classification tests](https://huggingface.co/sathiskumarjothi0/Bev-Pro-C/resolve/main/assets/benchmark_classification.png)

| Test | Bev-Pro-C | Jev | Mistral Large 4 | Mistral Large 3 | gpt-oss 120B | Gemma 4 31B | Nemotron 3 Super |
|---|---|---|---|---|---|---|---|
| AG News | 90.6 | **91.8** | 90.8 | 88.8 | 88.8 | **91.8** | 87.8 |
| Emotion | 56.2 | 57.6 | 55.8 | 55.6 | **57.8** | 55.4 | 50.2 |
| Banking77 (77 intents) | 70.6 | 79.2 | 75.8 | 71.2 | **81.0** | 78.8 | 69.2 |
| SST-5 (5-level sentiment) | 51.8 | **56.8** | 51.4 | 49.4 | 49.6 | 53.2 | 46.8 |
| XNLI (English) | 86.4 | **89.2** | 84.8 | 77.6 | 80.2 | **89.2** | 75.0 |
| MASSIVE intents | 74.8 | 80.8 | 78.4 | 75.2 | 81.6 | **83.8** | 69.4 |

### Areas Bev was not built for

![Areas Bev was not built for](https://huggingface.co/sathiskumarjothi0/Bev-Pro-C/resolve/main/assets/benchmark_outside.png)

| Test | Bev-Pro-C | Jev | Mistral Large 4 | Mistral Large 3 | gpt-oss 120B | Gemma 4 31B | Nemotron 3 Super |
|---|---|---|---|---|---|---|---|
| Product reviews* | 96.4 | 96.4 | 96.4 | 96.6 | 95.4 | **97.2** | 94.8 |
| Unfair contract terms* | **76.0** | 62.0 | 54.6 | 52.8 | 47.6 | 74.4 | 44.0 |
| Financial-news sentiment* | 59.2 | 66.4 | 64.0 | 50.4 | 68.4 | **71.2** | 64.0 |
| Poem sentiment* | **65.1** | 44.5 | 42.6 | 32.1 | 59.8 | 62.7 | 45.5 |
| Support intents* | 88.0 | 89.8 | 85.4 | 86.2 | 90.0 | **90.4** | 76.6 |
| Toxic comments* | 79.2 | 78.8 | 68.0 | 73.0 | **80.8** | 76.4 | 69.6 |
| Financial-news topics* | **69.8** | 65.0 | 61.8 | 61.6 | 63.4 | 64.2 | 57.2 |
| Legal case holdings* | 71.2 | **77.6** | 69.2 | 72.2 | 61.2 | 71.6 | 55.0 |


Calibration of Bev-Pro-C (confidence = the probability of the answer given; ECE 0 is perfect):

![Bev-Pro-C calibration](https://huggingface.co/sathiskumarjothi0/Bev-Pro-C/resolve/main/assets/reliability.png)

| Test | Accuracy | Answered at confidence ≥ 0.8 | Accuracy of those | ECE |
|---|---|---|---|---|
| Bev-Bench (operations documents) | 91.9% | 89.7% | 95.0% | 0.032 |
| Typed decisions | 79.2% | 61.7% | 91.5% | 0.040 |
| Rules-Bench (rules shown) | 92.0% | 87.2% | 97.0% | 0.011 |
| Aegis 2.0 request safety | 78.2% | 51.1% | 92.6% | 0.024 |
| Product reviews | 96.4% | 93.0% | 97.9% | 0.014 |


## Evaluate on your data

```bash
bev eval my_cases.jsonl --threshold 0.8
```

Each line: `{"state": "...", "questions": {...}, "labels": {"name": answer}}`. Reports, per question type: accuracy
(correct ÷ questions), ECE (Σ over 10 confidence bins of share of answers × |accuracy − mean confidence|; 0 is perfect;
example: a bin with 40% of the answers at mean confidence 0.95 and 90% right adds 0.4 × 0.05 = 0.02) and, at the
threshold, the share of answers and their accuracy.

## Limits

- English only; 4,096 tokens per call (longer text is cut at the end); no images.
- Knowledge-heavy decisions (case law, finance topic labels) are the weakest area; check them on your own data.
- Answers can change when the options are shuffled for a small share of questions; keep option order stable.
- For decisions with serious consequences, validate on your own data and keep a person in the loop.
- Questions with more than 26 options run as a tournament on Bev-Pro-C: lower accuracy, less calibrated confidence.

## Contributing and security

Issues and pull requests are welcome; run `pip install -e ".[all,dev]"`, `ruff check src tests examples` and `pytest`
(the tests need no model download). Report security problems privately through GitHub: Security → Report a
vulnerability.

## Support

If Bev is useful to you, you can support its development: [buymeacoffee.com/sjothi](https://www.buymeacoffee.com/sjothi)

## Licence

Apache-2.0, © 2026 Dr. Sathiskumar JOTHI: the package and the models, commercial use allowed. Bev-Pro-C is built on
Gemma 4 (Apache-2.0) by Google; Bev-Large-C on ModernBERT-large (Apache-2.0) by Answer.AI and LightOn.
