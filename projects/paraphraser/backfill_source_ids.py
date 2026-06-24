#!/usr/bin/env python3
"""Re-key data generated under the OLD (document-level) source_id scheme.

Context: build_seeds.py used to assign source_id at the document level, so every
sentence split from one document shared an id (amazon:50 = review #50, etc.). The
manifest had 10,000 jobs but only 1,650 distinct ids, 814 of them colliding (one
id -> up to 20 sentences). See triples.runs.md "Data-integrity finding". The
generator is now fixed (source_id = "<doc-id>#<sentence-index>"); this script
brings already-generated data into the same scheme so source_id is a usable key.

What it does: builds a canonical map from data/seeds.jsonl — for each old doc-level
id, distinct source texts are numbered in FIRST-APPEARANCE order (which is document
sentence order), giving "<old-id>#<k>". That map is applied to seeds.jsonl,
run_manifest.jsonl, and triples.claude-code.jsonl so the same sentence gets the
same new id in every file. The triples store the model's ECHO of the source (minor
punctuation/spacing drift), so matching is on an alphanumeric-only signature, with a
fallback via (old-id, instruction) through the manifest for the few echoes that
altered words.

Triples are NOT regenerated and their text is untouched — only the source_id tag
changes. The triples themselves were always valid (the model rewrote the real fed
text); only the id was non-unique.

Idempotent-ish: rows whose source_id already contains "#" are treated as already
keyed and skipped. Run ONCE over the complete corpus (i.e. after the chunk queue is
finished) so seeds.jsonl still carries the old ids the map is built from.

Usage:
    python backfill_source_ids.py --dry-run     # report only, no writes
    python backfill_source_ids.py               # rewrite in place (.bak-preBackfill backups)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
SEEDS = DATA / "seeds.jsonl"
MANIFEST = DATA / "run_manifest.jsonl"
TRIPLES = DATA / "triples.claude-code.jsonl"

NEW_DELIM = "#"


def sig(s: str) -> str:
    """Alphanumeric-only signature: collapses quote/space/punctuation echo drift."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def build_map(seeds: list[dict]) -> tuple[dict, dict]:
    """(old_id, sig(text)) -> new_id, plus old_id -> count of distinct sentences."""
    order: dict[str, list[str]] = defaultdict(list)  # old_id -> [sig, ...] first-seen order
    for r in seeds:
        oid, sg = r["source_id"], sig(r["text"])
        if sg not in order[oid]:
            order[oid].append(sg)
    newid: dict[tuple[str, str], str] = {}
    for oid, sigs in order.items():
        for k, sg in enumerate(sigs):
            newid[(oid, sg)] = f"{oid}{NEW_DELIM}{k}"
    return newid, {oid: len(sigs) for oid, sigs in order.items()}


def collisions(rows: list[dict], text_field: str) -> int:
    by_id: dict[str, set] = defaultdict(set)
    for r in rows:
        oid = r.get("source_id")
        if oid is None:  # pre-schema pilot rows carry no source_id
            continue
        by_id[oid].add(sig(r.get(text_field, "")))
    return sum(1 for v in by_id.values() if len(v) > 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args()

    seeds = read_jsonl(SEEDS)
    manifest = read_jsonl(MANIFEST)
    triples = read_jsonl(TRIPLES)
    print(f"loaded: seeds={len(seeds)}  manifest={len(manifest)}  triples={len(triples)}")
    print(f"collisions BEFORE: manifest={collisions(manifest, 'source')}  "
          f"triples={collisions(triples, 'source')}")

    newid, _ = build_map(seeds)
    # fallback for echo-drifted triples: (old_id, instruction) -> new_id, via the manifest
    instr_fb: dict[tuple[str, str], str] = {}
    for r in manifest:
        oid = r["source_id"]
        nid = newid.get((oid, sig(r["source"])))
        if nid is None:
            continue
        for ins in r.get("instructions", []):
            instr_fb.setdefault((oid, ins["instruction"]), nid)

    def rekey(rows: list[dict], text_field: str, use_fallback: bool) -> Counter:
        stats: Counter = Counter()
        for r in rows:
            oid = r.get("source_id")
            if oid is None:  # pre-schema pilot rows: nothing to re-key
                stats["no_id"] += 1
                continue
            if NEW_DELIM in oid:
                stats["already"] += 1
                continue
            nid = newid.get((oid, sig(r.get(text_field, ""))))
            if nid is None and use_fallback:
                nid = instr_fb.get((oid, r.get("instruction", "")))
                if nid is not None:
                    stats["fallback"] += 1
            if nid is None:
                stats["UNMATCHED"] += 1
                continue
            r["source_id"] = nid
            stats["rekeyed"] += 1
        return stats

    s_seeds = rekey(seeds, "text", use_fallback=False)
    s_man = rekey(manifest, "source", use_fallback=False)
    s_tri = rekey(triples, "source", use_fallback=True)
    print(f"seeds:    {dict(s_seeds)}")
    print(f"manifest: {dict(s_man)}")
    print(f"triples:  {dict(s_tri)}")
    print(f"collisions AFTER:  manifest={collisions(manifest, 'source')}  "
          f"triples={collisions(triples, 'source')}")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return
    for path, rows in [(SEEDS, seeds), (MANIFEST, manifest), (TRIPLES, triples)]:
        shutil.copy(path, str(path) + ".bak-preBackfill")
        write_jsonl(path, rows)
        print(f"wrote {path.name}  (backup: {path.name}.bak-preBackfill)")


if __name__ == "__main__":
    main()
