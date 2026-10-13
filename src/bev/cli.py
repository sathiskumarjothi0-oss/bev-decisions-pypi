"""Command line for Bev.

    bev "I was charged twice, please refund" --noul "Does the customer ask for a refund?"
    bev --file ticket.txt --choice "Which team?" --options billing="payments, refunds" tech="bugs, outages"
    bev --file alert.txt --score "How severe is this?" --levels low medium high
    bev --file ticket.txt --preset ticket_triage
    bev --batch tickets.txt --preset ticket_triage --min-confidence 0.8      # one text per line
    bev "..." --questions questions.json --pro                               # unsure answers go to Bev-Pro (GPU)
    bev presets                                   # list the ready-made question sets
    bev eval my_cases.jsonl [--threshold 0.8]     # accuracy and calibration on your own labelled examples
    bev serve [--host 127.0.0.1] [--port 8000]    # HTTP server   (pip install "bev-decisions[serve]")
    bev mcp                                       # MCP server    (pip install "bev-decisions[mcp]")
Common options: --model (folder or Hugging Face repo id), --device (cuda, mps, cpu).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _model(a: argparse.Namespace):  # type: ignore[no-untyped-def]
    import bev

    return bev.load(a.model or bev.DEFAULT_MODEL, device=a.device)


def _sub(argv: list[str]) -> None:
    import bev

    ap = argparse.ArgumentParser(prog=f"bev {argv[0]}")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    if argv[0] == "presets":
        for n in bev.presets.names():
            print(n, "->", ", ".join(bev.presets.get(n)))
        return
    if argv[0] == "eval":
        ap.add_argument("file")
        ap.add_argument("--threshold", type=float, default=0.8)
        a = ap.parse_args(argv[1:])
        print(json.dumps(bev.evaluate_file(_model(a), a.file, a.threshold), indent=1))
        return
    if argv[0] == "serve":
        ap.add_argument("--host", default="127.0.0.1")
        ap.add_argument("--port", type=int, default=8000)
        a = ap.parse_args(argv[1:])
        from .server import serve

        serve(_model(a), a.host, a.port)
        return
    if argv[0] == "mcp":
        a = ap.parse_args(argv[1:])
        from .mcp_server import run

        run(_model(a))


def _questions(a: argparse.Namespace) -> dict:
    import bev

    if a.questions:
        return json.loads(Path(a.questions).read_text())
    if a.preset:
        return bev.presets.get(a.preset)
    qs: dict = {}
    if a.noul:
        qs["answer"] = {"type": "noul", "instructions": a.noul}
    if a.choice:
        if not a.options:
            sys.exit("--choice needs --options key=description ...")
        qs["choice"] = {"type": "choice", "instructions": a.choice, "criteria": dict(o.split("=", 1) for o in a.options)}
    if a.score:
        if not a.levels:
            sys.exit("--score needs --levels lowest ... highest")
        qs["score"] = {"type": "score", "instructions": a.score, "criteria": a.levels}
    if not qs:
        sys.exit("ask something: --noul, --choice with --options, --score with --levels, --preset NAME, or --questions file.json")
    return qs


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("serve", "mcp", "eval", "presets"):
        _sub(sys.argv[1:])
        return
    ap = argparse.ArgumentParser(prog="bev", description="Ask Bev typed questions about a text. Subcommands: presets, eval, serve, mcp.")
    ap.add_argument("text", nargs="?", help="the text to decide about (or use --file / --batch)")
    ap.add_argument("--file", help="read the text from a file")
    ap.add_argument("--batch", help="a file with one text per line")
    ap.add_argument("--noul", help="a yes/no question")
    ap.add_argument("--choice", help="a choice question (with --options)")
    ap.add_argument("--options", nargs="+", help="key=description for --choice")
    ap.add_argument("--score", help="a score question (with --levels)")
    ap.add_argument("--levels", nargs="+", help="score levels, lowest first")
    ap.add_argument("--questions", help="a JSON file of questions in Bev's format")
    ap.add_argument("--preset", help="a ready-made question set (see: bev presets)")
    ap.add_argument("--min-confidence", type=float, default=None, help='mark answers below this confidence with "review": true')
    ap.add_argument("--pro", action="store_true", help="send unsure answers to Bev-Pro (needs a GPU and bev-decisions[pro])")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.batch:
        texts = [x for x in Path(a.batch).read_text().split("\n") if x.strip()]
    elif a.file:
        texts = [Path(a.file).read_text()]
    elif a.text:
        texts = [a.text]
    else:
        sys.exit("give a text, --file or --batch")
    qs = _questions(a)
    import bev

    if a.pro:
        router = bev.Router(fast=_model(a), load_strong=True, threshold=a.min_confidence or 0.8, device=a.device)
        results = router.decide_batch([(t, qs) for t in texts])
    else:
        results = [{"answers": ans} for ans in _model(a).decide_batch([(t, qs) for t in texts])]
    for r in results:
        if a.min_confidence is not None:
            for ans in r["answers"].values():
                ans["review"] = bev.confidence(ans) < a.min_confidence
        print(json.dumps(r if a.pro else r["answers"]))


if __name__ == "__main__":
    main()
