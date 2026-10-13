"""Typed values and facts read from the document and the questions: amounts, dates, times, numbers, and the comparisons,
sums, counts and time windows they imply. The network uses them alongside the text."""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field

MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
STOP = set(
    "a an the of to in on at by for from with and or is are was were be been did does do how many much what which who whose whether any at least more than less each this that these those it its as per within single there their ask asks".split()
)
_money = re.compile("(?:USD\\s*|\\$)\\s?(\\d[\\d,]*(?:\\.\\d+)?)(k)?|(\\d[\\d,]*(?:\\.\\d+)?)\\s*USD", re.I)
_iso_ts = re.compile("\\b(\\d{4})-(\\d{2})-(\\d{2})(?:[ T](\\d{2}):(\\d{2})(?::(\\d{2}))?)?\\b")
_dmy = re.compile("\\b(\\d{1,2})\\s+([A-Za-z]{3,9})\\.?\\s+(\\d{4})\\b")
_mdy = re.compile("\\b([A-Za-z]{3,9})\\.?\\s+(\\d{1,2})(?:st|nd|rd|th)?,?\\s+(\\d{4})\\b")
_clock = re.compile("(?<![\\d-])\\b(\\d{1,2}):(\\d{2})\\b(?!:\\d{2}\\s*\\d)")
_num = re.compile("(?<![\\w.])(\\d+(?:\\.\\d+)?)(?=(?:ms|%|\\s|$|[^\\w.]))")
_ip = re.compile("\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b")
_word = re.compile("[A-Za-z][A-Za-z0-9_\\-]*")


@dataclass
class Value:
    kind: str
    value: float
    start: int
    end: int
    line: int


@dataclass
class Fact:
    kind: str
    feats: list[float]
    spans: list[tuple[int, int]] = field(default_factory=list)
    where: str = "state"
    result: float | None = None


FACT_KINDS = ["cmp", "pair", "sum", "count", "window", "run"]
FACT_DIM = 8
VAL_DIM = 4 + 2 + 16


def _month(s: str) -> int | None:
    return MONTHS.get(s[:3].lower())


def _minutes(d: dt.datetime) -> float:
    return (d - dt.datetime(2000, 1, 1)).total_seconds() / 60


def _line_of(text: str) -> list[int]:
    starts = [0] + [m.end() for m in re.finditer("\n", text)]
    return starts


def extract(text: str) -> list[Value]:
    out: list[Value] = []
    taken: list[tuple[int, int]] = []
    starts = _line_of(text)

    def line(pos: int) -> int:
        lo, hi = (0, len(starts) - 1)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def free(a: int, b: int) -> bool:
        return all((b <= x or a >= y for x, y in taken))

    def add(kind: str, v: float, a: int, b: int) -> None:
        if free(a, b):
            out.append(Value(kind, v, a, b, line(a)))
            taken.append((a, b))

    for m in _money.finditer(text):
        raw = m.group(1) or m.group(3)
        v = float(raw.replace(",", ""))
        if m.group(2):
            v *= 1000
        add("money", v, m.start(), m.end())
    for m in _iso_ts.finditer(text):
        y, mo, d, hh, mm, ss = m.groups()
        try:
            t = dt.datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), int(ss or 0))
        except ValueError:
            continue
        add("date", _minutes(t), m.start(), m.end())
    for rx, order in ((_dmy, "dmy"), (_mdy, "mdy")):
        for m in rx.finditer(text):
            g = m.groups()
            d, mo, y = (g[0], g[1], g[2]) if order == "dmy" else (g[1], g[0], g[2])
            mon = _month(mo)
            if mon is None:
                continue
            try:
                add("date", _minutes(dt.datetime(int(y), mon, int(d))), m.start(), m.end())
            except ValueError:
                continue
    for m in _clock.finditer(text):
        add("time", int(m.group(1)) * 60 + int(m.group(2)), m.start(), m.end())
    for m in _num.finditer(text):
        add("number", float(m.group(1)), m.start(), m.end())
    out.sort(key=lambda v: v.start)
    return out


def key_terms(question: str, state_vocab: set[str]) -> list[str]:
    terms = []
    for w in _word.findall(question.lower()):
        if w in STOP or len(w) < 3:
            continue
        stem = w[:-1] if w.endswith("s") and len(w) > 4 else w
        if any(v.startswith(stem) for v in state_vocab):
            terms.append(stem)
    terms += _ip.findall(question)
    return sorted(set(terms))


def _lines_with(lines: list[str], terms: list[str]) -> list[int]:
    low = [x.lower() for x in lines]
    return [i for i, x in enumerate(low) if all(t.lower() in x for t in terms)]


def _signed_log(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def facts(state: str, question: str) -> list[Fact]:
    sv, qv = (extract(state), extract(question))
    lines = state.split("\n")
    vocab = {w for w in _word.findall(state.lower())}
    terms = key_terms(question, vocab)
    return _question_facts(sv, qv, lines, vocab, terms, question)


def state_facts(state: str) -> list[Fact]:
    sv = extract(state)
    fs: list[Fact] = []
    for kind in ("date", "money"):
        vals = [s for s in sv if s.kind == kind][:12]
        for i in range(len(vals)):
            for j in range(len(vals)):
                if i != j:
                    diff = vals[j].value - vals[i].value
                    unit = diff / 1440 if kind == "date" else diff
                    fs.append(
                        Fact(
                            "pair",
                            [float(diff > 0), float(diff == 0), _signed_log(unit), float(kind == "date"), 0, 0, 0, 0],
                            [(vals[i].start, vals[i].end), (vals[j].start, vals[j].end)],
                        )
                    )
    return fs


def _question_facts(sv: list[Value], qv: list[Value], lines: list[str], vocab: set[str], terms: list[str], question: str) -> list[Fact]:
    fs: list[Fact] = []
    for q in qv:
        same = [s for s in sv if s.kind == q.kind]
        for s in same[:32]:
            diff = s.value - q.value
            fs.append(
                Fact(
                    "cmp",
                    [float(diff > 0), float(diff == 0), _signed_log(diff), _signed_log(s.value), 0, 0, 0, 0],
                    [(s.start, s.end), (q.start, q.end)],
                    "both",
                )
            )
    money = [s for s in sv if s.kind == "money"]
    if money:
        tot = sum(s.value for s in money)
        for q in [q for q in qv if q.kind == "money"] or [None]:
            ref = q.value if q else 0.0
            fs.append(
                Fact(
                    "sum",
                    [
                        float(tot > ref) if q else 0.0,
                        _signed_log(tot - ref),
                        _signed_log(tot),
                        float(len(money)),
                        float(q is not None),
                        0,
                        0,
                        0,
                    ],
                    [(q.start, q.end)] if q else [],
                    "question" if q else "state",
                )
            )
    quoted = [m.lower() for m in re.findall("'([^']+)'|\\\"([^\\\"]+)\\\"", question) for m in m if m]
    if terms or quoted:
        k = len(_lines_with(lines, sorted(set(terms + quoted))))
        fs.append(Fact("count", [float(k), math.log1p(k), float(len(terms)), 0, 0, 0, 0, 0], result=float(k)))
        allw = sorted(
            {w[:-1] if w.endswith("s") and len(w) > 4 else w for w in _word.findall(question.lower()) if w not in STOP and len(w) > 2}
            | set(quoted)
            | set(_ip.findall(question))
        )
        k_all = len(_lines_with(lines, allw))
        fs.append(Fact("count", [float(k_all), math.log1p(k_all), float(len(allw)), 0, 0, 1.0, 0, 0], result=float(k_all)))
        for drop in allw:
            sub = [w for w in allw if w != drop]
            if sub:
                kd = len(_lines_with(lines, sub))
                fs.append(Fact("count", [float(kd), math.log1p(kd), float(len(sub)), 0, 0, 0, 1.0, 0], result=float(kd)))
        for t in terms:
            kt = len(_lines_with(lines, [t]))
            fs.append(Fact("count", [float(kt), math.log1p(kt), 1.0, 1.0, 0, 0, 0, 0], result=float(kt)))
    for q in qv:
        same = [s for s in sv if s.kind == q.kind]
        if same:
            above = sum(s.value > q.value for s in same)
            fs.append(
                Fact(
                    "count",
                    [float(above), math.log1p(above), 0, 0, 1.0, float(len(same)), 0, 0],
                    [(q.start, q.end)],
                    "question",
                    result=float(above),
                )
            )
    wq = re.search("(\\d+)\\s*-?\\s*minute", question)
    if wq and terms:
        w = float(wq.group(1))
        stamps: dict[str, list[float]] = {}
        content_terms = [t for t in terms if not _ip.fullmatch(t)]
        for i in _lines_with(lines, content_terms):
            ts = [v.value for v in sv if v.line == i and v.kind == "date"]
            ip = _ip.findall(lines[i])
            if ts:
                stamps.setdefault(ip[0] if ip else "*", []).append(ts[0])
        best = 0
        for ts in stamps.values():
            ts.sort()
            j = 0
            for i in range(len(ts)):
                while ts[i] - ts[j] > w:
                    j += 1
                best = max(best, i - j + 1)
        fs.append(Fact("window", [float(best), math.log1p(best), w, 0, 0, 0, 0, 0], result=float(best)))
    ents = [t for t in terms if t in vocab or "-" in t]
    for q in [q for q in qv if q.kind == "number"]:
        for ent in ents[:3]:
            idx = _lines_with(lines, [ent])
            if len(idx) < 2:
                continue
            per_line = [[v.value for v in sv if v.line == i and v.kind == "number"] for i in idx]
            width = min((len(x) for x in per_line), default=0)
            for col in range(min(width, 4)):
                run = best = 0
                for nums in per_line:
                    run = run + 1 if nums[col] > q.value else 0
                    best = max(best, run)
                fs.append(
                    Fact(
                        "run",
                        [float(best), math.log1p(best), float(col), float(len(idx)), 0, 0, 0, 0],
                        [(q.start, q.end)],
                        "question",
                        result=float(best),
                    )
                )
    return fs[:128]


def option_feats(state: str, options: list[str]) -> list[list[float]]:
    low = state.lower()
    lines = low.split("\n")
    vocab = set(_word.findall(low))
    out = []
    for o in options:
        ol = o.lower()
        toks = [t for t in _word.findall(ol) if t not in STOP]
        overlap = sum(t in vocab for t in toks) / max(len(toks), 1)
        try:
            num = float(o)
        except ValueError:
            num = -1.0
        first = 1.0
        idx = [i for i, x in enumerate(lines) if ol in x] if len(ol) > 2 else []
        if len(idx) >= 4:
            cols = [[float(n) for n in _num.findall(lines[i])] for i in idx]
            width = min(len(c) for c in cols)
            for col in range(min(width, 4)):
                series = [c[col] for c in cols]
                head = sorted(series[: max(3, len(series) // 4)])
                base = head[len(head) // 2]
                hit = next((k for k, v in enumerate(series) if base > 0 and v > 2.5 * base and (v - base > 0.5)), None)
                if hit is not None:
                    first = min(first, idx[hit] / max(len(lines), 1))
        out.append([float(ol in low), overlap, num, first, float(num >= 0)])
    return out


def value_features(v: Value) -> list[float]:
    """A typed value as 22 numbers: its kind, signed log magnitude, a day ordinal for dates, and 16 Fourier features."""
    kinds = ["money", "date", "time", "number"]
    raw = math.log1p(abs(v.value)) if math.isfinite(v.value) else 709.78 if not math.isnan(v.value) else 0.0
    lv = math.copysign(min(raw, 709.78), v.value) if not math.isnan(v.value) else 0.0
    four = [f(lv * 2**k / 4) for k in range(8) for f in (math.sin, math.cos)]
    ordinal = (v.value / 1440 - 9500) / 1000 if v.kind == "date" else 0.0
    return [float(v.kind == k) for k in kinds] + [lv / 10, ordinal] + four
