"""Stage B filter pass: validate (source, instruction, target) triples before SFT.

First a normalization step, then two filter layers ordered by how much we trust them.

  0. NORMALIZE (always) -> the teacher sometimes echoes the axis hint into the
     instruction text as a trailing "(axes: genre, register)". The axes already live
     in the row's `axes` field, so this is stripped from `instruction` (with whitespace
     tidied) in both the trainable output and the dedup key. Done at the filter stage,
     not in the raw store, so the canonical jsonl stays a faithful record of the teacher.

  1. HARD constraint checks (deterministic, axis-specific) -> failing rows are DROPPED.

  1. HARD constraint checks (deterministic, axis-specific) -> failing rows are DROPPED.
     These verify instruction compliance we can check exactly:
       - "exactly N words" / "N-word ..."   -> word count must equal N
       - "no more than N words" (also "at most"/"fewer than"/"up to")
                                            -> word count must be <= N
       - "exactly one ... sentence" / "single ... sentence"
                                            -> exactly one sentence
                                            -> if "declarative", reject ? and !

  2. FAITHFULNESS flags (soft) -> rows are KEPT but listed for review (--strict drops them):
       - number retention: a quantity in the SOURCE (digit, or a number word >= "two",
         or an ordinal) that has no surface form in the TARGET.
       - optional BERTScore F1(target vs source) via bert_score.BERTScorer (pass --bertscore;
         needs `pip install bert_score`). Loads roberta-large through transformers on the local
         device, with rescale_with_baseline=True so scores are interpretable (~0 unrelated,
         ~1 near-identical). CAVEAT: similarity legitimately drops for heavy style transforms
         (a wedding-toast or magical-realist rewrite adds framing), so this is a soft *outlier*
         signal, NOT a hard gate — the flag threshold is axis-dependent (see STRICT_AXES).

  Word counting splits on whitespace, so a hyphenated token like "carbon-fiber" counts
  as one word — matching how the instructions tend to be read.

Usage:
    python filter_triples.py                     # filters data/triples.claude-code.jsonl
    python filter_triples.py --input other.jsonl
    python filter_triples.py --bertscore         # add semantic layer (needs: pip install bert_score; ~1.4GB model)
    python filter_triples.py --strict            # also drop faithfulness-flagged rows

Output (next to the input):
    <stem>.filtered.jsonl   rows passing the hard checks (safe to train on)
    <stem>.flagged.jsonl    rows dropped or soft-flagged, each annotated with `filter_flags`
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
FILTER_VERSION = "v1"

# Rescaled BERTScore F1 (rescale_with_baseline=True): ~0 for unrelated text, ~1 for
# near-identical (can dip slightly negative). Heavy style transforms legitimately score
# lower, so the soft-flag threshold is axis-dependent. These are STARTING POINTS — tune
# after a first --bertscore run on real data.
STRICT_AXES = {"length", "structural"}
BERTSCORE_MIN_STRICT = 0.45    # length/structural rewrites should stay close to the source
BERTSCORE_MIN_LENIENT = 0.15   # voice/genre/tone/register/etc. may diverge and still be faithful

CARDINALS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}
# value -> surface forms to look for in a target
INT_FORMS: dict[int, set[str]] = {}
for _w, _v in {**CARDINALS, **ORDINALS}.items():
    INT_FORMS.setdefault(_v, set()).add(_w)
for _v in list(INT_FORMS):
    INT_FORMS[_v].add(str(_v))
    INT_FORMS[_v].add(f"{_v}th")  # crude ordinal digit form (3rd handled below)
for _v, _suf in {1: "1st", 2: "2nd", 3: "3rd"}.items():
    INT_FORMS.setdefault(_v, set()).add(_suf)


def get_device() -> str:
    import torch  # only needed for the optional --bertscore layer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


AXES_SUFFIX_RE = re.compile(r"\s*\(axes:[^)]*\)\s*$", re.IGNORECASE)


def clean_instruction(instruction: str) -> str:
    """Strip the `(axes: ...)` annotation the teacher sometimes echoes into the
    instruction text, and tidy whitespace.

    The axes already live in the row's `axes` field, so the trailing
    "(axes: genre, register)" is redundant: it would leak the internal taxonomy
    into the student's training prompts, and — because the model emits it
    nondeterministically — it fractures the same instruction into multiple surface
    forms (with/without the suffix, plus stray double spaces), corrupting the
    (source, instruction) dedup key. Returns the cleaned single-line text.
    """
    s = AXES_SUFFIX_RE.sub("", instruction or "")
    return re.sub(r"\s+", " ", s).strip()


def dedup_key(row: dict) -> tuple[str, str]:
    """Normalized (source, instruction) key for de-duplication.

    Re-emits regenerate the same job, and the stored `source` is the model's echo,
    which varies by quote style, whitespace, and the missing-space-after-period
    detokenizer artifact in some seeds ("staring.he" vs "staring. he"). Normalize
    all three so echo-variants of one job collapse to a single key. The instruction
    is additionally stripped of trailing sentence punctuation so the period jitter
    that pairs with the (axes: ...) echo ("…the end" vs "…the end.") collapses too.
    """
    def n(s: str) -> str:
        s = s or ""
        s = re.sub(r"[‘’“”'\"`]", '"', s)
        s = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", s)  # insert missing space after sentence punctuation
        s = re.sub(r"\s+", " ", s)
        return s.strip().lower().rstrip(".!? ")
    return (n(row.get("source", "")), n(clean_instruction(row.get("instruction", ""))))


def word_to_int(tok: str):
    tok = tok.lower().strip()
    if tok.isdigit():
        return int(tok)
    return CARDINALS.get(tok)


def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def bertscore_threshold(axes: list[str], override) -> float:
    if override is not None:
        return override
    return BERTSCORE_MIN_STRICT if any(a in STRICT_AXES for a in axes) else BERTSCORE_MIN_LENIENT


def check_constraints(instruction: str, target: str) -> tuple[list[str], dict]:
    """Hard, deterministic checks. Returns (failures, extras-to-record)."""
    instr = instruction.lower()
    failures: list[str] = []
    extras: dict = {}

    # --- word-count constraints ---
    n_eq = None
    m = re.search(r"exactly (\w+) words?", instr) or re.search(r"in exactly (\w+) words?", instr)
    if m:
        n_eq = word_to_int(m.group(1))
    if n_eq is None:
        m = re.search(r"\b(\w+)-word\b", instr)  # "five-word", "eight-word"
        if m:
            n_eq = word_to_int(m.group(1))

    n_max = None
    m = re.search(r"(?:no more than|at most|fewer than|up to) (\w+) words?", instr)
    if m:
        n_max = word_to_int(m.group(1))

    if n_eq is not None or n_max is not None:
        wc = word_count(target)
        extras["word_count"] = wc
        if n_eq is not None and wc != n_eq:
            failures.append(f"constraint:exactly-{n_eq}-words (got {wc})")
        if n_max is not None and wc > n_max:
            failures.append(f"constraint:max-{n_max}-words (got {wc})")

    # --- single-sentence constraints ---
    wants_single = bool(
        re.search(r"\b(one|single)\b[^.]*\bsentence\b", instr)
    )
    if wants_single:
        sc = sentence_count(target)
        extras["sentence_count"] = sc
        if sc != 1:
            failures.append(f"constraint:one-sentence (got {sc})")
        if "declarative" in instr and ("?" in target or "!" in target):
            failures.append("constraint:declarative (found ? or !)")

    return failures, extras


def check_number_retention(source: str, target: str) -> list[str]:
    """Soft faithfulness check: quantities in source that vanished from the target."""
    tgt = target.lower()
    flags: list[str] = []
    seen: set[int] = set()
    for tok in re.findall(r"\d+", source):
        seen.add(int(tok))
    for w, v in {**CARDINALS, **ORDINALS}.items():
        if v >= 2 and re.search(rf"\b{w}\b", source.lower()):  # skip 0/1 (too pronoun-like)
            seen.add(v)
    for v in sorted(seen):
        forms = INT_FORMS.get(v, {str(v)})
        if not any(re.search(rf"\b{re.escape(f)}\b", tgt) for f in forms):
            flags.append(f"faithfulness:dropped-number {v}")
    return flags


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(HERE / "data" / "triples.claude-code.jsonl"))
    p.add_argument("--bertscore", action="store_true", help="Compute BERTScore F1 (needs: pip install bert_score).")
    p.add_argument("--bertscore-min", type=float, default=None,
                   help="Global override for the BERTScore soft-flag threshold; default is per-axis (see STRICT_AXES).")
    p.add_argument("--strict", action="store_true", help="Also DROP faithfulness-flagged rows, not just flag them.")
    p.add_argument("--dedupe", action="store_true",
                   help="Collapse duplicate (source, instruction) pairs (keeps the latest) before filtering.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No rows in {in_path}")

    # Normalize the instruction field first: strip the redundant `(axes: ...)` echo
    # and tidy whitespace, so the trainable output is clean and the dedup below
    # collapses surface-form variants of the same instruction. The axes metadata is
    # untouched (it lives in r["axes"]).
    n_cleaned = 0
    for r in rows:
        cleaned = clean_instruction(r.get("instruction", ""))
        if cleaned != r.get("instruction", ""):
            r["instruction"] = cleaned
            n_cleaned += 1
    if n_cleaned:
        print(f"normalize: cleaned `(axes: …)`/whitespace from {n_cleaned} instruction fields")

    if args.dedupe:
        seen: dict[tuple[str, str], dict] = {}
        for r in rows:
            seen[dedup_key(r)] = r  # keep the last (most recent run) occurrence
        n_before = len(rows)
        rows = list(seen.values())
        print(f"dedupe:   {n_before} -> {len(rows)} distinct (source, instruction) "
              f"({n_before - len(rows)} duplicates removed)")

    base = str(in_path)[:-6] if str(in_path).endswith(".jsonl") else str(in_path)
    out_path = Path(base + ".filtered.jsonl")
    flagged_path = Path(base + ".flagged.jsonl")

    # Optional semantic layer. Uses bert_score.BERTScorer directly (caches the model, vs
    # evaluate.load(...).compute() which reloads it each call) with rescale_with_baseline so
    # the F1 values are interpretable. roberta-large (~1.4GB) is pulled to ~/.cache/huggingface
    # on first use and runs through transformers on `device`. Scored once over all rows.
    bert_f1 = None
    if args.bertscore:
        try:
            from bert_score import BERTScorer  # noqa: PLC0415 — optional dep
        except ImportError as e:
            raise SystemExit("--bertscore needs: pip install bert_score") from e
        scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=get_device())
        _, _, f1 = scorer.score([r["target"] for r in rows], [r["source"] for r in rows])
        bert_f1 = f1.tolist()

    kept, flagged = [], []
    reason_counts: Counter = Counter()
    for i, r in enumerate(rows):
        hard, extras = check_constraints(r["instruction"], r["target"])
        soft = check_number_retention(r["source"], r["target"])
        if bert_f1 is not None:
            f1 = round(float(bert_f1[i]), 4)
            extras["bertscore_f1"] = f1
            thresh = bertscore_threshold(r.get("axes", []), args.bertscore_min)
            if f1 < thresh:
                soft.append(f"faithfulness:bertscore {f1}<{thresh}")

        dropped = bool(hard) or (args.strict and bool(soft))
        annotated = {**r, **extras, "filter_version": FILTER_VERSION,
                     "filter_flags": hard + soft, "dropped": dropped}
        for flag in hard + soft:
            reason_counts[flag.split(" ")[0]] += 1
        if dropped:
            flagged.append(annotated)
        else:
            kept.append(annotated)
            if soft:  # kept-but-noted: also surface in the review file
                flagged.append(annotated)

    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    flagged_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in flagged))

    n = len(rows)
    print(f"input:    {n} triples ({in_path.name})")
    print(f"kept:     {len(kept)}  ({len(kept) / n:.0%} yield)  -> {out_path.name}")
    print(f"dropped:  {sum(1 for r in flagged if r['dropped'])}  (hard-constraint failures)")
    print(f"flagged:  {len(flagged)}  (dropped + soft-flagged for review)  -> {flagged_path.name}")
    if reason_counts:
        print("by reason:")
        for reason, c in reason_counts.most_common():
            print(f"  {reason}: {c}")
    if not args.bertscore:
        print("note: BERTScore layer skipped (run with --bertscore to enable; needs `pip install bert_score`).")


if __name__ == "__main__":
    main()
