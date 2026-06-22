#!/usr/bin/env python3
"""Stratified condition sampler — produce a per-(name x tier) job manifest.

Mirror of the paraphraser's sample_instructions.py. For each row in seeds.jsonl,
draw N_easy + N_hard conditions from conditions.json, respecting the tier rule:

    hard:  any knowledge axis {nickname, phonetic, translit, cultural, negative},
           OR multi-axis (combinatorial)
    easy:  everything else (single-axis order / initial / format / suffix / typo)

Each pool is cycled round-robin with a fixed seed (shuffle, draw, reshuffle when
empty), so conditions are used evenly across the corpus. Within one name, the draw
is deduped on condition text.

Output: data/run_manifest.jsonl — one row per (seed_idx, tier):
    {seed_idx, source_name, domain, source_id, components, tier, model, model_id, conditions}
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SEEDS = PROJECT_DIR / "data" / "seeds.jsonl"
DEFAULT_BANK = PROJECT_DIR / "conditions.json"
DEFAULT_OUT = PROJECT_DIR / "data" / "run_manifest.jsonl"

# Tier -> (subagent model alias, full model id stamped into provenance).
TIER_MODELS = {
    "easy": ("haiku",  "claude-haiku-4-5"),
    "hard": ("sonnet", "claude-sonnet-4-6"),
}

HARD_AXES = {"nickname", "phonetic", "translit", "cultural", "negative"}


def tier_for(axes: list[str]) -> str:
    if any(a in HARD_AXES for a in axes):
        return "hard"
    if len(axes) > 1:
        return "hard"
    return "easy"


class CyclingSampler:
    """Round-robin without replacement: shuffle once, draw, reshuffle when empty.

    `take(n)` returns n entries with distinct `condition` text."""

    def __init__(self, pool: list[dict], rng: random.Random):
        self._pool = list(pool)
        self._rng = rng
        self._queue: list[dict] = []

    def _refill(self) -> None:
        self._queue = list(self._pool)
        self._rng.shuffle(self._queue)

    def take(self, n: int) -> list[dict]:
        out: list[dict] = []
        seen: set[str] = set()
        while len(out) < n:
            if not self._queue:
                self._refill()
            cand = self._queue.pop()
            if cand["condition"] in seen:
                continue
            seen.add(cand["condition"])
            out.append(cand)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=str(DEFAULT_SEEDS))
    ap.add_argument("--conditions", default=str(DEFAULT_BANK))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--per-seed-easy", type=int, default=4)
    ap.add_argument("--per-seed-hard", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bank = json.loads(Path(args.conditions).read_text())
    easy_pool = [e for e in bank if tier_for(e["axes"]) == "easy"]
    hard_pool = [e for e in bank if tier_for(e["axes"]) == "hard"]

    if args.per_seed_easy > len(easy_pool):
        raise SystemExit(f"--per-seed-easy {args.per_seed_easy} > easy pool ({len(easy_pool)})")
    if args.per_seed_hard > len(hard_pool):
        raise SystemExit(f"--per-seed-hard {args.per_seed_hard} > hard pool ({len(hard_pool)})")

    samplers = {
        "easy": CyclingSampler(easy_pool, rng),
        "hard": CyclingSampler(hard_pool, rng),
    }

    seeds = [json.loads(l) for l in Path(args.seeds).read_text().splitlines() if l]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    usage: Counter = Counter()
    with out_path.open("w") as f:
        for i, seed in enumerate(seeds):
            for tier_name, n in [("easy", args.per_seed_easy),
                                  ("hard", args.per_seed_hard)]:
                if n == 0:
                    continue
                model, model_id = TIER_MODELS[tier_name]
                picks = samplers[tier_name].take(n)
                row = {
                    "seed_idx": i,
                    "source_name": seed["name"],
                    "domain": seed["domain"],
                    "source_id": seed["source_id"],
                    "components": seed.get("components", {}),
                    "tier": tier_name,
                    "model": model,
                    "model_id": model_id,
                    "conditions": picks,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                for p in picks:
                    usage[p["condition"]] += 1

    print(f"wrote {n_rows} jobs -> {out_path}")
    if usage:
        counts = sorted(usage.values())
        print(f"condition coverage: {len(usage)}/{len(bank)} bank entries used")
        print(f"  uses per condition: min={counts[0]} median={counts[len(counts)//2]} max={counts[-1]}")


if __name__ == "__main__":
    main()
