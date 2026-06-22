#!/usr/bin/env python3
"""Build a seed corpus of canonical PERSON NAMES for Stage B variant generation.

Analogous to the paraphraser's build_seeds.py, but the unit is a full name, not a
sentence. Names are composed from two genuinely public-domain frequency lists:

    first names   US SSA national baby names ... https://www.ssa.gov/oact/babynames/names.zip
    surnames      US Census 2010 surnames ...... https://www2.census.gov/topics/genealogy/2010surnames/names.zip

Each canonical name is first + (optional) middle initial + last, e.g. "John L. Smith".
We stratify by surname frequency into two "domains" so the matcher sees both
collision-prone common names and rare ones (this matters for the `negative` /
collision conditions downstream):

    frequent   surname in the top FREQ_RANK of the Census list
    rare       everything below it

Output: data/seeds.jsonl, one row per name:

    {"name": "John L. Smith", "domain": "frequent", "source_id": "ssa:John|census:Smith",
     "components": {"first": "John", "middle": "L.", "last": "Smith"}}

`components` is stored so the deterministic filter checks (filter_pairs.py) and the
teacher prompt can reason about parts without re-parsing.

Network is BEST-EFFORT: if the public files can't be fetched, an embedded fallback
list of common names is used so smoke tests run offline. Reproducible via --seed.

Smoke test (offline-safe):
    python build_seeds.py --per-domain 25

Full default run:
    python build_seeds.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_DIR / "data" / "seeds.jsonl"

SSA_URL = "https://www.ssa.gov/oact/babynames/names.zip"
CENSUS_URL = "https://www2.census.gov/topics/genealogy/2010surnames/names.zip"
FREQ_RANK = 2000  # surnames ranked <= this are "frequent"
MIDDLE_INITIAL_P = 0.6  # fraction of composed names that carry a middle initial

# ---------------- embedded fallback (offline) ----------------
# Small, deliberately diverse pool so translit/cultural/nickname conditions have
# realistic inputs even with no network. Surnames roughly ordered common -> rarer.
_FALLBACK_FIRST = [
    "John", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "William", "Linda",
    "David", "Elizabeth", "Catherine", "Stephen", "José", "Sofía", "Mohammed", "Wei",
    "Ananya", "Dmitri", "Aoife", "Lars",
]
_FALLBACK_LAST = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "O'Brien", "van der Berg", "Müller", "Nguyen", "Okafor",
    "Kowalski", "Tchaikovsky", "Singh", "Haddad", "Ferrari",
]


def _fetch_zip_member(url: str, predicate) -> bytes | None:
    """Download a zip and return the first member whose name matches `predicate`."""
    try:
        # Gov sites (SSA) 403 the default urllib agent; send a browser-ish UA.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (name-matching seed builder)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            blob = resp.read()
        zf = zipfile.ZipFile(io.BytesIO(blob))
        name = next((n for n in zf.namelist() if predicate(n)), None)
        return zf.read(name) if name else None
    except Exception as e:  # network/parse failure -> caller falls back
        print(f"[warn] fetch failed for {url}: {e}", file=sys.stderr)
        return None


def load_first_names() -> list[str]:
    """SSA yobYYYY.txt rows are 'Name,Sex,Count'; take names in frequency order."""
    data = _fetch_zip_member(SSA_URL, lambda n: n.lower().startswith("yob") and n.endswith(".txt"))
    if not data:
        print("[info] using embedded fallback first names", file=sys.stderr)
        return list(_FALLBACK_FIRST)
    names: list[str] = []
    seen: set[str] = set()
    for row in csv.reader(io.StringIO(data.decode("utf-8"))):
        if len(row) >= 1 and row[0] and row[0] not in seen:
            seen.add(row[0])
            names.append(row[0])
    return names


def load_surnames() -> list[tuple[str, int]]:
    """Census file rows are 'NAME,rank,count,...'; return [(Titlecased, rank)]."""
    data = _fetch_zip_member(CENSUS_URL, lambda n: n.lower().endswith(".csv"))
    if not data:
        print("[info] using embedded fallback surnames", file=sys.stderr)
        return [(s, i + 1) for i, s in enumerate(_FALLBACK_LAST)]
    out: list[tuple[str, int]] = []
    reader = csv.reader(io.StringIO(data.decode("utf-8")))
    header = next(reader, None)  # skip 'name,rank,count,...'
    for row in reader:
        if len(row) >= 2 and row[0] and row[0].upper() != "ALL OTHER NAMES":
            try:
                rank = int(row[1])
            except ValueError:
                continue
            out.append((row[0].title(), rank))
    return out


def normalize_for_dedupe(name: str) -> str:
    return " ".join(name.lower().split())


def compose(first: str, last: str, rng: random.Random) -> dict:
    middle = f"{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}." if rng.random() < MIDDLE_INITIAL_P else ""
    parts = [first] + ([middle] if middle else []) + [last]
    return {
        "name": " ".join(parts),
        "components": {"first": first, "middle": middle, "last": last},
        "first": first, "last": last,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--per-domain", type=int, default=2500,
                    help="target composed names per frequency domain (default: 2500)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    firsts = load_first_names()
    surnames = load_surnames()
    rng.shuffle(firsts)

    buckets = {
        "frequent": [s for s in surnames if s[1] <= FREQ_RANK],
        "rare":     [s for s in surnames if s[1] > FREQ_RANK],
    }
    if not buckets["rare"]:  # fallback list is short -> split it in half
        mid = len(surnames) // 2 or 1
        buckets = {"frequent": surnames[:mid], "rare": surnames[mid:]}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total = 0
    with out_path.open("w") as f:
        for domain, pool in buckets.items():
            if not pool:
                continue
            print(f"[{domain}] composing up to {args.per_domain} names "
                  f"from {len(pool)} surnames x {len(firsts)} first names...", file=sys.stderr)
            wrote = 0
            attempts = 0
            cap = args.per_domain * 20  # bound the loop if the pool is tiny
            while wrote < args.per_domain and attempts < cap:
                attempts += 1
                first = rng.choice(firsts)
                last = rng.choice(pool)[0]
                rec = compose(first, last, rng)
                key = normalize_for_dedupe(rec["name"])
                if key in seen:
                    continue
                seen.add(key)
                row = {
                    "name": rec["name"],
                    "domain": domain,
                    "source_id": f"ssa:{rec['first']}|census:{rec['last']}",
                    "components": rec["components"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                wrote += 1
                total += 1
            print(f"[{domain}] wrote {wrote}", file=sys.stderr)

    print(f"done -> {out_path} ({total} names)", file=sys.stderr)


if __name__ == "__main__":
    main()
