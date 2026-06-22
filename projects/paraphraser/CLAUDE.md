# Paraphraser experiment

## Goal

Train a small student model to paraphrase arbitrary text following natural-language style instructions. Must run on a 16GB Apple Silicon laptop for both training and inference. Style adaptation happens via prompt at inference time, not via retraining per style.

## Design decisions

- **Free-form instruction tuning**, not fixed style tags or a multi-axis tag scheme. The student learns the mapping from natural-language style instruction → rewrite. New styles at inference time require no retraining — just a new instruction string.
- **Teacher**: Claude. With no Anthropic API budget on this account, both stages run Claude *inside Claude Code* (subagents on the Claude Code plan, no API key): Stage A was generated directly in a Claude Code session, Stage B via `generate_triples_workflow.js`. An earlier local-model (Qwen) teacher for Stage B was tried and dropped — it failed the hard style axes.
- **Student plan**: LoRA on a small causal instruct model (1.5-3B class) is preferred over FLAN-T5 with prefixes once we committed to free-form instructions. FLAN-T5 with fixed prefixes is a worse fit for natural-language instructions than a causal instruct base that's already been instruction-tuned.

## Two-stage data pipeline

### Stage A — instruction bank (DONE)

`instructions.json` is a 250-entry bank of rewrite instructions spanning seven style axes:

- **register**: technical / plain / academic / colloquial / archaic
- **audience**: expert / novice / child / executive / skeptic
- **tone**: neutral / persuasive / urgent / reassuring / skeptical / playful / formal
- **length & density**: terse / expansive / hedged / blunt
- **genre / framing**: tweet / headline / abstract / FAQ / dialogue / marketing / textbook
- **structural**: active↔passive / nominalize↔verbalize / conclusion↔evidence
- **voice / persona**: research paper / startup pitch / news wire / instructional manual / personal essay

Composition: 120 single-axis instructions (each axis category covered ≥2× with different phrasings, for student robustness to prompt wording) + 130 combinatorial instructions composing two axes (e.g. *"startup pitch to a skeptical investor"* combines voice + tone). Each entry is tagged with the axes it exercises, for downstream diversity tracking.

`review/instructions_review_sample.json` — random 70-item subset for human inspection (`random.Random(0).sample(...)`, reproducible). 70 is the right sample size for a 95% CL / 10% MOE estimate of bank quality (finite-population corrected from the textbook 384).

`generate_instructions.py` is the reference script that *would* generate the bank via the Anthropic API. **Not run** — no API key on this account. The script is kept as documentation of the prompt design and as the path forward if API access becomes available.

### Stage B — rewrite generation (IN PROGRESS)

For each seed sentence, sample N instructions from the bank, ask the teacher model to rewrite preserving meaning, save the triple. Output: `(source, instruction, target)` ready for SFT.

The pipeline runs in three Python stages that produce chunked Claude Code Workflow files, then those workflows are invoked one at a time:

1. **`build_seeds.py`** — builds `data/seeds.jsonl` (5000 sentences, 1000/domain) from five parquet-native HF datasets: `wikimedia/wikipedia`, `abisee/cnn_dailymail`, `gfissore/arxiv-abstracts-2021`, `fancyzhx/amazon_polarity`, `benjaminbeilharz/better_daily_dialog`. (The canonical names — `cnn_dailymail`, `daily_dialog`, `amazon_polarity`, `ccdv/arxiv-summarization` — are all unusable in current `datasets`: script-based, or pre-tokenized/lowercased.) Streams + windowed-shuffles each source (`buffer_size=10000`); `--seed` and `--skip` jump to different chunks. Sentence splitter prefers nltk, falls back to a regex. Filters: 10–40 words, no URLs/markdown/all-caps, mostly-letters, light detokenizer for `"word , word"`. Cross-domain dedupe.

2. **`sample_instructions.py`** — builds `data/run_manifest.jsonl` (10000 jobs = 5000 seeds × 2 tiers). For each seed, draws N_easy + N_hard (default 4+4) instructions from the bank via per-tier round-robin shuffle samplers, so every one of the 250 instructions gets ~125–223 uses across the corpus. Tier rule: `hard` if any axis is `structural` / `voice` OR multi-axis; else `easy`. Tier → model: easy→Haiku, hard→Sonnet.

3. **`emit_chunks.py`** — partitions the manifest into chunks of ≤1000 subagents (Claude Code subagent lifetime cap; whole seeds stay together), and writes `data/runs/chunk_NNN.workflow.js` per chunk by substituting the EDIT-PER-RUN block of `generate_triples_workflow.js`. Each emitted file is a self-contained workflow with its own `run_id` and `gen_timestamp`. At 5000 seeds × 2 tiers = 10 chunks.

The workflow itself (`generate_triples_workflow.js`):

- **Teacher model**: Claude inside a Claude Code Workflow (no API key/budget — subagents run on the Claude Code plan). Fans out one fresh subagent per (seed × tier) job, each with its own per-subagent instruction draw. Easy axes (register / length / audience / tone / single-axis genre) → Haiku. Hard axes (structural / voice / any multi-axis combinatorial) → Sonnet. Run procedure, per-row provenance schema, and sandbox gotchas are documented in the script's header.
  - *Dropped:* an earlier local Qwen teacher (HF Transformers + MPS; MLX 4-bit) failed the hard axes — a chiasmus instruction just got a clause appended, "magical-realist narrator" got a bland rephrase — so that tooling was removed in favor of the Claude workflow.
  - *Earlier pilot structure (now refactored):* a single shared `SOURCES` + `TIERS` arrays where every easy subagent saw the same 5 easy instructions. Replaced by a flat `JOBS` array with per-job instructions, so bulk runs cover the whole bank.

- **Provenance**: Claude-generated triples go in `data/triples.claude-code.jsonl`, every row tagged `domain`, `source_id`, `tier`, `gen_model`, `gen_backend`, `prompt_version`, `run_id`, `gen_timestamp`. Each chunk run gets a fresh `run_id`, logged in `triples.runs.md`. Bulk data lives under `data/` (gitignored — regenerable, push to the Hub for sharing). A legacy `data/triples.jsonl` from the dropped Qwen attempt may linger locally; it has no tags and is not used.
- **Filter pass**: `filter_triples.py` runs after generation. Layer 1 — deterministic constraint checks (exact/max word counts, single-sentence/declarative) — **drops** instruction violations. Layer 2 — soft faithfulness flags (number retention always; optional rescaled BERTScore via `--bertscore`, needs `bert_score`, axis-dependent thresholds) — **flags for review** without dropping, because source↔target similarity legitimately drops for heavy voice/genre rewrites. Writes `*.filtered.jsonl` (safe to train on) + `*.flagged.jsonl` (review). Pilot: 59/60 kept.

## Files

| Path | Purpose |
|---|---|
| `instructions.json` | 250-item instruction bank (Stage A output) |
| `review/instructions_review_sample.json` | 70-item review subset, seed=0 |
| `generate_instructions.py` | Anthropic-API reference script (not run) |
| `build_seeds.py` | Stage B step 1: pull/clean ~5k sentences from 5 HF datasets → `data/seeds.jsonl` |
| `sample_instructions.py` | Stage B step 2: stratified instruction draw per (seed × tier) → `data/run_manifest.jsonl` |
| `emit_chunks.py` | Stage B step 3: partition manifest, emit `data/runs/chunk_NNN.workflow.js` |
| `generate_triples_workflow.js` | Stage B Claude Code Workflow template (`JOBS` array; emit_chunks substitutes its EDIT-PER-RUN block) |
| `filter_triples.py` | Stage B filter pass: constraint checks (drop) + faithfulness flags (review) |
| `data/` | **Gitignored** bulk datasets (regenerable; push full runs to the Hub) |
| `data/seeds.jsonl` | 5000-sentence seed corpus, 1000/domain |
| `data/run_manifest.jsonl` | 10000 jobs (seed × tier) with per-job instruction draws |
| `data/runs/chunk_NNN.workflow.js` | Per-chunk workflow files (≤1000 subagents each); inputs to Workflow tool |
| `data/triples.jsonl` | *(legacy)* Stage B output from the dropped Qwen teacher; untagged, unused |
| `data/triples.claude-code.jsonl` | Stage B output, **Claude** teacher, fully tagged |
| `data/triples.claude-code.filtered.jsonl` | Filter output: rows passing hard checks (train on this) |
| `data/triples.claude-code.flagged.jsonl` | Filter output: dropped + soft-flagged rows, with reasons |
| `triples.runs.md` | Run log: one row per `run_id` with its params |
| `review/triples_review_sample.jsonl` | *(legacy)* Qwen Stage B review subset |

## Known gaps before training

1. **Faithfulness filtering**: `filter_triples.py` does deterministic constraint checks (drop) + soft faithfulness flags (number retention + BERTScore via `--bertscore`) + `--dedupe` (collapses normalized `(source, instruction)`). `bert_score` (0.3.12, roberta-large, runs on MPS) **is now installed in `venv/`** — run with `./venv/bin/python filter_triples.py --dedupe --bertscore`. **Per-axis thresholds need tuning**: the strict (length/structural) threshold 0.45 sits at the median and over-flags compression rewrites — see "Filter runs" in `triples.runs.md`. Flags are soft (no drop) until `--strict`.
2. **Instruction bank review**: the 70-item review subset is still pending hand-review. Looking for (a) instructions that change meaning rather than style, (b) near-duplicates.
3. **Bulk run execution (IN PROGRESS — ~48% done)**: as of 2026-06-19, **4,813 / 10,000 manifest jobs** generated (normalized coverage; easy 69%, hard 27% — earlier emits over-weighted easy). Duplication is minimal (20,492 rows → 20,121 distinct `(source, instruction)`, only 371 dupes — the `--done-file` dedup worked). See `triples.runs.md` for the reconciled per-run accounting. **Run procedure — do NOT run all chunks in parallel** (each batched chunk is ~1.3M+ subagent tokens; the full set is >10M tokens in minutes and will hit rate limits): invoke **one chunk at a time** via `Workflow({ scriptPath: ".../data/runs/batched/chunk_NNN.workflow.js" })`, persist the returned `triples` to `data/triples.claude-code.jsonl` after each, check `failed`/`mismapped` (note: `mismapped` is dominated by benign echo-variant false positives — verify against the manifest before worrying), then start the next. At most 1–2 chunks in flight, never the whole directory. The `…013508Z` queue concentrates the hard backlog in chunks 004–009 — run those next.
