#!/usr/bin/env python3
"""Build a diverse seed corpus for Stage B paraphrase triple generation.

Pulls text from five HF datasets across distinct registers (all parquet-native;
current `datasets` no longer loads script-based datasets, so the obvious
canonical names like `cnn_dailymail` / `daily_dialog` are not used):

    wikipedia  encyclopedia      wikimedia/wikipedia (20231101.en)
    news       journalism        abisee/cnn_dailymail (3.0.0)
    arxiv      academic abstract gfissore/arxiv-abstracts-2021
    consumer   reviews / pitch   fancyzhx/amazon_polarity
    dialogue   spoken exchange   benjaminbeilharz/better_daily_dialog

Each document is sentence-split, length- and shape-filtered, deduped across
all domains, and written to data/seeds.jsonl as one row per sentence:

    {"text": str, "domain": str, "source_id": str}

Reproducible via --seed (controls the streaming order skip). Re-running with
--domain X appends only that domain and respects dedupe against the existing
file.

Smoke test:
    python build_seeds.py --per-domain 20 --domain dialogue

Full default run (~5k sentences):
    python build_seeds.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Iterator

# ---------------- sentence splitter ----------------
# Prefer nltk if available (handles abbreviations); fall back to a regex that
# splits on sentence-final punctuation followed by whitespace + capital/quote.

try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize

    def split_sentences(text: str) -> list[str]:
        return sent_tokenize(text)

except ImportError:
    _SENT_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')

    def split_sentences(text: str) -> list[str]:
        return [s.strip() for s in _SENT_END.split(text) if s.strip()]


# ---------------- filters ----------------

WORD_MIN_DEFAULT = 10
WORD_MAX_DEFAULT = 40

_BAD_PATTERNS = re.compile(r"https?://|www\.|@\w+|[<>{}]|\[.*?]\(.*?\)")
_ALLCAPS = re.compile(r"\b[A-Z]{5,}\b")


def is_clean_sentence(s: str, word_min: int, word_max: int) -> bool:
    s = s.strip()
    if not s:
        return False
    if not (s[0].isupper() or s[0] in '"\'('):
        return False
    if s[-1] not in ".!?":
        return False
    words = s.split()
    if not (word_min <= len(words) <= word_max):
        return False
    if _BAD_PATTERNS.search(s):
        return False
    if _ALLCAPS.search(s):
        return False
    # filter out lines that are mostly digits / punctuation (tables, refs)
    letters = sum(c.isalpha() for c in s)
    if letters < 0.6 * len(s):
        return False
    return True


def normalize_for_dedupe(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


_DETOK_PUNCT = re.compile(r"\s+([,.!?;:'’‘])")


def detokenize(s: str) -> str:
    """Light cleanup for pre-tokenized text (DailyDialog has 'word , word')."""
    s = _DETOK_PUNCT.sub(r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------------- per-domain loaders ----------------
# Each yields (raw_text, source_id) tuples until the caller stops pulling.
# Streaming randomization: .shuffle(buffer_size) fills a window and draws
# uniformly from it (windowed shuffle, NOT full shuffle); .skip(N) advances
# the underlying stream by N before any shuffling.

SHUFFLE_BUFFER = 10_000


def _stream(name: str, *args, seed: int, skip: int, **kwargs):
    from datasets import load_dataset
    ds = load_dataset(name, *args, split="train", streaming=True, **kwargs)
    if skip:
        ds = ds.skip(skip)
    return ds.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)


def load_wikipedia(seed: int, skip: int) -> Iterator[tuple[str, str]]:
    for i, row in enumerate(_stream("wikimedia/wikipedia", "20231101.en",
                                    seed=seed, skip=skip)):
        text = (row.get("text") or "")[:2000]  # cap per-article work
        if text:
            yield text, f"wiki:{row.get('id', i)}"


def load_cnn(seed: int, skip: int) -> Iterator[tuple[str, str]]:
    for i, row in enumerate(_stream("abisee/cnn_dailymail", "3.0.0",
                                    seed=seed, skip=skip)):
        text = row.get("article") or ""
        if text:
            yield text[:2000], f"cnn:{row.get('id', i)}"


def load_arxiv(seed: int, skip: int) -> Iterator[tuple[str, str]]:
    # gfissore/arxiv-abstracts-2021 keeps natural prose (proper case, real
    # punctuation); ccdv/arxiv-summarization is lowercased + tokenized.
    for i, row in enumerate(_stream("gfissore/arxiv-abstracts-2021",
                                    seed=seed, skip=skip)):
        text = row.get("abstract") or ""
        if text:
            yield text, f"arxiv:{row.get('id', i)}"


def load_amazon(seed: int, skip: int) -> Iterator[tuple[str, str]]:
    for i, row in enumerate(_stream("fancyzhx/amazon_polarity",
                                    seed=seed, skip=skip)):
        text = row.get("content") or ""
        if text:
            yield text, f"amazon:{i}"


def load_dailydialog(seed: int, skip: int) -> Iterator[tuple[str, str]]:
    """One utterance per row in this mirror (column: utterance)."""
    for i, row in enumerate(_stream("benjaminbeilharz/better_daily_dialog",
                                    seed=seed, skip=skip)):
        utt = (row.get("utterance") or "").strip()
        if utt:
            yield utt, f"dd:{row.get('dialog_id', i)}:{i}"


DOMAINS: dict[str, Callable[[int, int], Iterator[tuple[str, str]]]] = {
    "wikipedia": load_wikipedia,
    "news":      load_cnn,
    "arxiv":     load_arxiv,
    "consumer":  load_amazon,
    "dialogue":  load_dailydialog,
}


# ---------------- pipeline ----------------

def collect_domain(domain: str, target: int, word_min: int, word_max: int,
                   seed: int, skip: int, seen: set[str]) -> list[dict]:
    out: list[dict] = []
    for text, source_id in DOMAINS[domain](seed, skip):
        for sent in split_sentences(text):
            sent = detokenize(sent)
            if not is_clean_sentence(sent, word_min, word_max):
                continue
            key = normalize_for_dedupe(sent)
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": sent, "domain": domain, "source_id": source_id})
            if len(out) >= target:
                return out
    return out


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_DIR / "data" / "seeds.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output JSONL path (default: {DEFAULT_OUT.relative_to(PROJECT_DIR.parent.parent)})")
    ap.add_argument("--per-domain", type=int, default=1000,
                    help="target clean sentences per domain (default: 1000)")
    ap.add_argument("--word-min", type=int, default=WORD_MIN_DEFAULT)
    ap.add_argument("--word-max", type=int, default=WORD_MAX_DEFAULT)
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed for the streaming buffer (default: 0)")
    ap.add_argument("--skip", type=int, default=0,
                    help="docs to .skip() before shuffling, to land in a "
                         "different chunk of the stream (default: 0)")
    ap.add_argument("--domain", choices=list(DOMAINS), action="append",
                    help="restrict to one or more domains (default: all)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    domains = args.domain or list(DOMAINS)
    seen: set[str] = set()

    appending = out_path.exists() and args.domain is not None
    if appending:
        with out_path.open() as f:
            for line in f:
                seen.add(normalize_for_dedupe(json.loads(line)["text"]))
        print(f"[init] loaded {len(seen)} existing keys for dedupe", file=sys.stderr)

    mode = "a" if appending else "w"
    with out_path.open(mode) as f:
        for d in domains:
            print(f"[{d}] collecting up to {args.per_domain} sentences...", file=sys.stderr)
            rows = collect_domain(d, args.per_domain, args.word_min, args.word_max,
                                  args.seed, args.skip, seen)
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{d}] wrote {len(rows)}", file=sys.stderr)

    print(f"done -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
