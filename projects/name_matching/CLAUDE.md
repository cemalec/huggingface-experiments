# Name-matching experiment

## Goal

Build a labelled dataset of person-name pairs for training/evaluating a **name-matching**
model — one that decides whether two differently-written names refer to the same individual.
The dataset is generated, not scraped: take a canonical name, apply a known *mismatch
condition*, and record the resulting variant plus a match / non-match label.

This is a structural sibling of the `paraphraser/` project. Where the paraphraser turns
(sentence, style instruction) → rewrite, this turns (name, mismatch condition) → variant name.
The machinery (seed builder → condition bank → stratified sampler → batched Claude Code
Workflow → filter pass) is deliberately the same so the two can share conventions.

## What a row is

Each output row (kept in the schema's `triples` array for parallelism with the paraphraser) is
a labelled name **pair** with the rule that produced it:

    source_name "John L. Smith"  + "first and last name flipped"            -> "Smith L. John"      (match)
    source_name "John L. Smith"  + "alternate first + letter sub in surname"-> "Johnathan L. Smlth" (match)
    source_name "John L. Smith"  + "different person, same surname"          -> "Jane R. Smith"      (non-match)

The student learns to see through `match` variations and to reject confusable `non-match` ones.

## Two-stage data pipeline

### Stage A — condition bank (DONE, deterministic)

`conditions.json` is the bank of mismatch conditions, tagged with `axes` and a `label`
(match / non-match). Axes:

- **order** — token reordering ("Smith, John", Eastern order)
- **initial** — full ↔ initial, middle add/drop
- **format** — case / punctuation / spacing ("JOHN SMITH", "OBrien")
- **suffix** — generational suffix or title add/drop
- **typo** — OCR / keyboard noise ("Smlth", "Jonh")
- **nickname** — diminutive / formal / alternate ("Bob" ↔ "Robert")  *(knowledge)*
- **phonetic** — homophone respelling ("Catherine" / "Katherine")  *(knowledge)*
- **translit** — diacritics / romanization ("Jose" / "José")  *(knowledge)*
- **cultural** — particles / patronymic / maiden ("van der Berg")  *(knowledge)*
- **negative** — a genuinely DIFFERENT but confusable person  *(label = non-match)*

Composition: single-axis conditions (each axis covered with several phrasings) + curated
two-axis combinatorial conditions (e.g. nickname + typo). `generate_conditions.py` builds it
deterministically — no API needed — and writes a `review/` sample. **Tier rule** (in
`sample_conditions.tier_for`): a condition is **hard** if it touches a knowledge axis
{nickname, phonetic, translit, cultural, negative} OR combines >1 axis; otherwise **easy**.
Tier → model: easy → Haiku, hard → Sonnet.

### Stage B — variant generation (SCAFFOLDED; no bulk run yet)

Three Python stages produce chunked Claude Code Workflow files, then those workflows are
invoked one at a time (one subagent per batch of names):

1. **`build_seeds.py`** — composes canonical names (first + optional middle initial + last)
   from public-domain sources: **US SSA** first names + **US Census 2010** surnames, stratified
   by surname frequency into `frequent` / `rare` domains. Writes `data/seeds.jsonl`:
   `{name, domain, source_id, components}`. Network is best-effort (Census downloads fine; SSA
   may 403 a datacenter IP) with an embedded fallback name pool so it runs offline.

2. **`sample_conditions.py`** — for each seed name draws N_easy + N_hard conditions from the
   bank via per-tier round-robin samplers → `data/run_manifest.jsonl` (one row per name × tier).

3. **`emit_chunks_batched.py`** — partitions the manifest into batched workflow chunks
   (`--batch-sources` names per subagent, `--max-bytes` cap per file), skipping
   `(source_name, tier)` pairs already in the `--done-file`. Writes
   `data/runs/batched/chunk_NNN.workflow.js`.

The workflow (`generate_pairs_batched_workflow.js`) fans out one subagent per batch of names;
each emits one variant per (name, condition), echoing `source_name`/`axes`/`label` and tagging
provenance. It RETURNS `{run_id, n_subagents, count, mismapped, failed, triples}` and writes
nothing (sandbox has no FS) — the orchestrator persists `triples` to
`data/pairs.claude-code.jsonl` and logs the run in `pairs.runs.md`. Run procedure and the
`mismapped` false-positive caveat are in `pairs.runs.md`.

**Filter pass** (`filter_pairs.py`): Layer 1 deterministic constraint checks — empty/invalid
label, no-op (variant == source), and order-flip token-multiset preservation — **drop**
violations. Layer 2 soft, axis-banded **faithfulness flags** (kept for review unless `--strict`):
string-similarity bands per axis group (spelling axes flagged when they drift *too far*;
structural axes banded on token overlap; negatives flagged when *too similar*). Default uses a
pure-python Levenshtein ratio; `--jaro` adds Jaro-Winkler + a Metaphone phonetic check (needs
`jellyfish`). Thresholds are starting points — see "Threshold tuning" in `pairs.runs.md`.

## Files

| Path | Purpose |
|---|---|
| `conditions.json` | Mismatch-condition bank (Stage A output): `{condition, axes, label}` |
| `generate_conditions.py` | Stage A: deterministically build the condition bank + review sample |
| `review/conditions_review_sample.json` | Review subset of the bank, seed=0 |
| `build_seeds.py` | Stage B step 1: compose canonical names from public SSA + Census lists → `data/seeds.jsonl` |
| `sample_conditions.py` | Stage B step 2: stratified condition draw per (name × tier) → `data/run_manifest.jsonl` |
| `emit_chunks_batched.py` | Stage B step 3: partition manifest, emit batched workflow chunks |
| `generate_pairs_batched_workflow.js` | Stage B Claude Code Workflow template (`BATCHES` array; emit substitutes its EDIT-PER-RUN block) |
| `filter_pairs.py` | Stage B filter pass: constraint checks (drop) + similarity-band faithfulness flags (review) |
| `data/` | **Gitignored** bulk datasets (regenerable; push full runs to the Hub) |
| `data/seeds.jsonl` | Canonical-name seed corpus |
| `data/run_manifest.jsonl` | Jobs (name × tier) with per-job condition draws |
| `data/runs/batched/chunk_NNN.workflow.js` | Per-chunk workflow files; inputs to the Workflow tool |
| `data/pairs.claude-code.jsonl` | Stage B output, Claude teacher, fully tagged |
| `data/pairs.claude-code.filtered.jsonl` | Filter output: rows passing hard checks (use this) |
| `data/pairs.claude-code.flagged.jsonl` | Filter output: dropped + soft-flagged rows, with reasons |
| `pairs.runs.md` | Run log: one row per `run_id` with its params, + filter runs |

## Status / known gaps

- **Pipeline is scaffolded and smoke-tested** end to end on the Python side (condition bank →
  seeds → manifest → chunk emit → filter all run); **no bulk Workflow run has been executed** yet.
- **SSA download** 403s from datacenter IPs — the embedded fallback first-name pool keeps it
  runnable, but a real run on a residential connection (or a cached `names.zip`) gets the full list.
- **Filter thresholds** in `filter_pairs.py` are untuned starting points; tune against a real
  distribution before relying on `--strict` (see `pairs.runs.md`).
- **Label balance**: negatives live only on the `negative` axis (hard pool), so the match/non-match
  ratio is governed by `--per-seed-hard` and how often negatives are drawn — check and adjust if a
  specific positive:negative ratio is wanted for training.

## Hugging Face docs

Doc retrieval goes through the **Context7 MCP server** (see the repo-root `CLAUDE.md`). Prefer it
over training-data recall for HF API signatures.
