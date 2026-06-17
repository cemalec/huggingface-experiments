# Paraphraser experiment

## Goal

Train a small student model to paraphrase arbitrary text following natural-language style instructions. Must run on a 16GB Apple Silicon laptop for both training and inference. Style adaptation happens via prompt at inference time, not via retraining per style.

## Design decisions

- **Free-form instruction tuning**, not fixed style tags or a multi-axis tag scheme. The student learns the mapping from natural-language style instruction → rewrite. New styles at inference time require no retraining — just a new instruction string.
- **Teacher**: Claude in principle (highest quality); in practice we don't have API budget on this account, so Stage A was generated directly in a Claude Code session and Stage B uses a local instruct model.
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

- **Teacher model**: three backends, in increasing quality:
  - `generate_rewrites.py` — HF Transformers + MPS, default `Qwen2.5-3B-Instruct` (bf16, ~6GB resident). Works on any platform; the 3B ceiling caps quality.
  - `generate_rewrites_mlx.py` — MLX (Apple Silicon native) + 4-bit quantization, default `mlx-community/Qwen2.5-7B-Instruct-4bit` (~4GB resident). Substantially faster than 7B-bf16 on MPS and fits in unified memory without swap pressure. 7B-class quality at 3B-class memory footprint. Preferred *local* path.
  - `generate_triples_workflow.js` — **Claude Code Workflow** (teacher = Claude, no API key/budget — subagents run on the Claude Code plan). Fans out one fresh subagent per (source × difficulty tier), routing easy axes (register/length/simple genre) → Haiku and hard axes (structural/voice/combinatorial) → Sonnet. This is the **quality path**: the local Qwen teachers fail the hard axes (a chiasmus instruction just got a clause appended; "magical-realist narrator" got a bland rephrase), whereas Claude obeys them. Run procedure, the per-row provenance schema, and the sandbox gotchas are documented in the script's header.

- **Provenance & output separation**: Claude-generated triples go in `data/triples.claude-code.jsonl` and are **never mixed** with the Qwen `data/triples.jsonl`. Every Claude row is tagged `tier`, `gen_model`, `gen_backend`, `prompt_version`, `run_id`, `gen_timestamp`. Each run gets a fresh `run_id`, logged in `triples.runs.md`. Bulk data lives under `data/` (gitignored — regenerable, push to the Hub for sharing).
- **Seed corpus**: needs domain diversity (Wikipedia, news, arXiv abstracts, marketing, dialogue). Currently uses a small built-in demo set for pipeline smoke-testing; real run will use `--seeds path/to/file.txt`.
- **Filter pass**: `filter_triples.py` runs after generation. Layer 1 — deterministic constraint checks (exact/max word counts, single-sentence/declarative) — **drops** instruction violations. Layer 2 — soft faithfulness flags (number retention always; optional rescaled BERTScore via `--bertscore`, needs `bert_score`, axis-dependent thresholds) — **flags for review** without dropping, because source↔target similarity legitimately drops for heavy voice/genre rewrites. Writes `*.filtered.jsonl` (safe to train on) + `*.flagged.jsonl` (review). Pilot: 59/60 kept.

#### Why MLX over HF Transformers on this machine

Tried `Qwen2.5-7B-Instruct` in bf16 via HF Transformers on MPS first. The math: 14GB model + OS + Python runtime + KV cache > 16GB unified RAM, so the OS pages the weights to swap and every forward pass thrashes. Real-world result: ~20-40 minutes per rewrite. MLX 4-bit packs the same model class into ~4GB and uses Apple's GPU directly, so generation runs at native speed without swap pressure.

## Files

| Path | Purpose |
|---|---|
| `instructions.json` | 250-item instruction bank (Stage A output) |
| `review/instructions_review_sample.json` | 70-item review subset, seed=0 |
| `generate_instructions.py` | Anthropic-API reference script (not run) |
| `generate_rewrites.py` | Stage B local-model script |
| `generate_triples_workflow.js` | Stage B Claude Code Workflow (teacher = Claude); see its header to run |
| `filter_triples.py` | Stage B filter pass: constraint checks (drop) + faithfulness flags (review) |
| `data/` | **Gitignored** bulk triple datasets (regenerable; push full runs to the Hub) |
| `data/triples.jsonl` | Stage B output, **Qwen** teacher (no provenance tags) |
| `data/triples.claude-code.jsonl` | Stage B output, **Claude** teacher, fully tagged |
| `data/triples.claude-code.filtered.jsonl` | Filter output: rows passing hard checks (train on this) |
| `data/triples.claude-code.flagged.jsonl` | Filter output: dropped + soft-flagged rows, with reasons |
| `triples.runs.md` | Run log: one row per `run_id` with its params |
| `review/triples_review_sample.jsonl` | Stage B review subset (created on run) |

## Known gaps before training

1. **Seed corpus**: still using the in-code demo set. Real run needs ~5k diverse sentences.
2. **Faithfulness filtering**: `filter_triples.py` does deterministic constraint checks (drop) + soft faithfulness flags (number retention now; BERTScore via `--bertscore`). BERTScore dep (`bert_score`, ~1.4GB roberta-large via transformers) not yet installed — enable when triple count grows and per-axis thresholds are worth tuning.
3. **Instruction bank review**: the 70-item review subset is still pending hand-review. Looking for (a) instructions that change meaning rather than style, (b) near-duplicates.
4. **Quality ceiling of local teacher**: Qwen2.5-3B/7B fail the hard axes (structural/voice/combinatorial). Resolved path: `generate_triples_workflow.js` uses Claude as the teacher via a Claude Code Workflow — no API budget needed. Local backends remain for offline/portable use. Open scaling question: feed the real ~5k seed corpus by chunking into multiple workflow runs (subagent lifetime cap is 1000/run).
