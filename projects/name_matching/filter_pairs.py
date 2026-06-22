"""Stage B filter pass: validate (source_name, condition, variant_name) rows before use.

Name-matching analog of the paraphraser's filter_triples.py. Two layers, ordered by
how much we can trust them:

  1. HARD constraint checks (deterministic) -> failing rows are DROPPED:
       - variant_name missing/empty.
       - label not in {match, non-match}.
       - no-op: variant identical to source (the model applied no change at all).
       - order-flip (single-axis 'order', match): the variant must contain the SAME
         multiset of name tokens as the source, just reordered. If tokens were added
         or changed, the stated transformation did not actually happen -> drop.

  2. FAITHFULNESS flags (soft) -> rows KEPT but listed for review (--strict drops them).
     Surface similarity between source and variant should sit in an axis-dependent BAND
     (mirrors filter_triples.py's per-axis BERTScore thresholds — same idea, names need
     string similarity not embeddings):
       - spelling axes {typo, nickname, phonetic, translit}: the variant respells tokens,
         so flag only when it drifts TOO far (char-similarity < SIM_MIN_SPELLING) — likely
         a different name, not a variant.
       - structural axes {order, initial, format, suffix, cultural}: spelling is preserved
         but reordered/abbreviated, so char-similarity is unreliable; band on TOKEN overlap
         (Jaccard < JACCARD_MIN_STRUCTURAL) instead.
       - negative (non-match): the variant is a DIFFERENT person, so it SHOULD be
         dissimilar; flag when it is TOO similar (char-similarity > SIM_MAX_NEGATIVE),
         i.e. an ambiguous/under-distinguished negative.
     With --jaro (needs `pip install jellyfish`) char-similarity uses Jaro-Winkler and an
     extra phonetic check flags {phonetic, nickname} match rows whose Metaphone codes
     diverge. Default uses a pure-python normalized Levenshtein ratio (always available).

Usage:
    python filter_pairs.py                 # filters data/pairs.claude-code.jsonl
    python filter_pairs.py --input other.jsonl
    python filter_pairs.py --jaro          # Jaro-Winkler + phonetic layer (needs jellyfish)
    python filter_pairs.py --strict        # also drop soft-flagged rows
    python filter_pairs.py --dedupe        # collapse duplicate (source_name, condition)

Output (next to the input):
    <stem>.filtered.jsonl   rows passing the hard checks (safe to use)
    <stem>.flagged.jsonl    rows dropped or soft-flagged, annotated with `filter_flags`
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
FILTER_VERSION = "v1"

SPELLING_AXES = {"typo", "nickname", "phonetic", "translit"}
STRUCTURAL_AXES = {"order", "initial", "format", "suffix", "cultural"}

# Soft-flag thresholds. STARTING POINTS — tune after a first real run (see pairs.runs.md).
SIM_MIN_SPELLING = 0.45        # below this, a "respelled" variant probably became a different name
JACCARD_MIN_STRUCTURAL = 0.34  # below this, a reorder/abbreviation lost too many name tokens
SIM_MAX_NEGATIVE = 0.90        # above this, a "different person" negative is ambiguously similar


def norm(s: str) -> str:
    s = s or ""
    s = re.sub(r"[‘’“”'\"`]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def alpha_tokens(name: str) -> list[str]:
    """Lowercased name tokens with commas/periods stripped (so 'J.' -> 'j')."""
    cleaned = re.sub(r"[.,]", " ", norm(name).lower())
    return [t for t in cleaned.split() if t]


def levenshtein_ratio(a: str, b: str) -> float:
    """Pure-python normalized similarity = 1 - lev(a,b)/max(len). Always available."""
    a, b = norm(a).lower(), norm(b).lower()
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(alpha_tokens(a)), set(alpha_tokens(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def dedup_key(row: dict) -> tuple[str, str]:
    return (norm(row.get("source_name", "")).lower(), norm(row.get("condition", "")).lower())


def check_constraints(row: dict) -> list[str]:
    """Hard, deterministic checks -> drop on any failure."""
    failures: list[str] = []
    src = norm(row.get("source_name", ""))
    var = norm(row.get("variant_name", ""))
    axes = row.get("axes", [])
    label = row.get("label")

    if not var:
        failures.append("constraint:empty-variant")
        return failures
    if label not in ("match", "non-match"):
        failures.append(f"constraint:bad-label ({label})")
    if var == src:
        failures.append("constraint:no-op (variant identical to source)")

    # Pure reorder must preserve the exact name-token multiset.
    if axes == ["order"] and label == "match":
        if sorted(alpha_tokens(src)) != sorted(alpha_tokens(var)):
            failures.append("constraint:order-tokens-changed")

    return failures


def check_similarity(row: dict, sim_fn, metaphone) -> tuple[list[str], dict]:
    """Soft, axis-banded similarity flags. Returns (flags, extras-to-record)."""
    flags: list[str] = []
    src, var = row.get("source_name", ""), row.get("variant_name", "")
    axes = set(row.get("axes", []))
    label = row.get("label")

    char_sim = round(sim_fn(src, var), 4)
    jac = round(token_jaccard(src, var), 4)
    extras = {"char_sim": char_sim, "token_jaccard": jac}

    if label == "non-match":
        if char_sim > SIM_MAX_NEGATIVE:
            flags.append(f"faithfulness:negative-too-similar {char_sim}>{SIM_MAX_NEGATIVE}")
    elif axes & SPELLING_AXES:
        if char_sim < SIM_MIN_SPELLING:
            flags.append(f"faithfulness:spelling-drift {char_sim}<{SIM_MIN_SPELLING}")
    else:  # structural-only match
        if jac < JACCARD_MIN_STRUCTURAL:
            flags.append(f"faithfulness:lost-tokens jaccard {jac}<{JACCARD_MIN_STRUCTURAL}")

    if metaphone is not None and label == "match" and axes & {"phonetic", "nickname"}:
        # Compare the most-distinctive token (the surname is usually last).
        st, vt = alpha_tokens(src), alpha_tokens(var)
        if st and vt and metaphone(st[-1]) != metaphone(vt[-1]):
            flags.append("faithfulness:phonetic-code-diverged")

    return flags, extras


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=str(HERE / "data" / "pairs.claude-code.jsonl"))
    p.add_argument("--jaro", action="store_true",
                   help="Use Jaro-Winkler + Metaphone phonetic check (needs: pip install jellyfish).")
    p.add_argument("--strict", action="store_true", help="Also DROP soft-flagged rows.")
    p.add_argument("--dedupe", action="store_true",
                   help="Collapse duplicate (source_name, condition) pairs (keeps the latest) before filtering.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No rows in {in_path}")

    if args.dedupe:
        seen: dict[tuple[str, str], dict] = {}
        for r in rows:
            seen[dedup_key(r)] = r
        n_before = len(rows)
        rows = list(seen.values())
        print(f"dedupe:   {n_before} -> {len(rows)} distinct (source_name, condition) "
              f"({n_before - len(rows)} duplicates removed)")

    sim_fn = levenshtein_ratio
    metaphone = None
    if args.jaro:
        try:
            import jellyfish  # noqa: PLC0415 — optional dep
        except ImportError as e:
            raise SystemExit("--jaro needs: pip install jellyfish") from e
        sim_fn = jellyfish.jaro_winkler_similarity
        metaphone = jellyfish.metaphone

    base = str(in_path)[:-6] if str(in_path).endswith(".jsonl") else str(in_path)
    out_path = Path(base + ".filtered.jsonl")
    flagged_path = Path(base + ".flagged.jsonl")

    kept, flagged = [], []
    reason_counts: Counter = Counter()
    for r in rows:
        hard = check_constraints(r)
        soft, extras = check_similarity(r, sim_fn, metaphone)
        dropped = bool(hard) or (args.strict and bool(soft))
        annotated = {**r, **extras, "filter_version": FILTER_VERSION,
                     "filter_flags": hard + soft, "dropped": dropped}
        for flag in hard + soft:
            reason_counts[flag.split(" ")[0]] += 1
        if dropped:
            flagged.append(annotated)
        else:
            kept.append(annotated)
            if soft:
                flagged.append(annotated)

    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    flagged_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in flagged))

    n = len(rows)
    print(f"input:    {n} name pairs ({in_path.name})")
    print(f"kept:     {len(kept)}  ({len(kept) / n:.0%} yield)  -> {out_path.name}")
    print(f"dropped:  {sum(1 for r in flagged if r['dropped'])}  (hard-constraint failures)")
    print(f"flagged:  {len(flagged)}  (dropped + soft-flagged for review)  -> {flagged_path.name}")
    if reason_counts:
        print("by reason:")
        for reason, c in reason_counts.most_common():
            print(f"  {reason}: {c}")
    if not args.jaro:
        print("note: similarity used pure-python Levenshtein ratio (run with --jaro for "
              "Jaro-Winkler + phonetic check; needs `pip install jellyfish`).")


if __name__ == "__main__":
    main()
