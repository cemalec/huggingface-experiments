# Hugging Face Experiments

## Purpose

This repo is a learning sandbox for the Hugging Face ecosystem (`transformers`, `datasets`, `huggingface_hub`, `accelerate`, `peft`, `tokenizers`, `evaluate`, etc.). The work here is **educational, not production**. Scripts and notebooks exist to explore how the libraries behave, not to ship a system.

Expect two kinds of requests:

1. **Code tasks** — write/modify scripts and notebooks that fine-tune, evaluate, upload, or otherwise exercise HF models and datasets.
2. **Conceptual questions** — explain how a HF library works, what the right tool/abstraction is for a given LLM task (classification, token classification, summarization, RAG, fine-tuning vs. LoRA vs. prompting, etc.), and the tradeoffs between approaches.

When answering conceptual questions, prefer concrete, runnable examples grounded in this repo's conventions (the scripts below) over abstract description. When the user is learning a new concept, name the underlying HF class/function (e.g., `Trainer`, `AutoModelForSequenceClassification`, `DataCollatorWithPadding`) so they can search the docs themselves.

## Repo layout

- `fine_tuning.py` — reference fine-tuning script using `Trainer`, defaults to BERT on GLUE/MRPC. Accepts `--checkpoint`, `--dataset`, `--dataset-config`.
- `upload_to_hub.py` — pushes a checkpoint directory to the Hub. Reads `HF_TOKEN` env var or stored credentials.
- `utils.py` — small helpers (e.g., device selection: CUDA / MPS / CPU).
- `ClassicalNLP/` — notebooks and scripts for masked LM, summarization, token classification.
- `demo_notebook.ipynb` — top-level demo.
- `model_checkpoints/`, `test-trainer/` — local checkpoint outputs (gitignored).
- `course_certs/` — HF course completion certs (reference only, don't edit).

## Conventions

- Python 3.8+, venv at `.venv/` or `venv/`.
- Use `utils.get_device()` (or similar) for device selection — this is an Apple Silicon machine, so MPS is often the right backend; don't hardcode `cuda`.
- Prefer the high-level HF APIs (`AutoModelFor*`, `AutoTokenizer`, `Trainer`, `pipeline`) over hand-rolled training loops unless the user explicitly wants the lower-level version for learning.
- New experiments: a standalone script or notebook is fine. Don't build framework abstractions across experiments — each one should be readable on its own.
- Checkpoints land in `test-trainer/` or `model_checkpoints/` and are not committed.

## When suggesting tools for a task

Default recommendations to anchor on (mention alternatives when relevant):

| Task | First-reach tool |
|------|------------------|
| Quick inference | `transformers.pipeline` |
| Classification fine-tune | `Trainer` + `AutoModelForSequenceClassification` |
| Token classification (NER) | `Trainer` + `AutoModelForTokenClassification` + `DataCollatorForTokenClassification` |
| Causal LM fine-tune | `Trainer` or `SFTTrainer` (TRL) |
| Parameter-efficient fine-tuning | `peft` (LoRA/QLoRA) |
| RLHF / preference tuning | `trl` (DPO, PPO, GRPO) |
| Evaluation | `evaluate` library |
| Dataset loading/streaming | `datasets` (`load_dataset`, `.map`, `.with_format`) |
| Tokenizer training | `tokenizers` |
| Multi-GPU / mixed precision | `accelerate` |
| Hub uploads | `huggingface_hub` (`HfApi`, `upload_folder`) |

Call out when a task would be better served outside HF (e.g., vLLM/TGI for serving, llama.cpp for local quantized inference).

## Hugging Face docs

Doc retrieval goes through the **Context7 MCP server** (configured in `.mcp.json` at project scope). It exposes two tools:

- `resolve-library-id` — map a name like "transformers" or "peft" to a Context7 library ID.
- `get-library-docs` — fetch docs (with optional topic filter) for that ID.

Workflow: resolve once per library, then query `get-library-docs` with a `topic` (e.g., `"Trainer"`, `"LoRA"`, `"DataCollator"`) rather than reading a whole library. Always prefer Context7 over training-data recall for API signatures — HF APIs change across minor versions. Fall back to WebFetch on `https://huggingface.co/docs/<library>/...` only if Context7 doesn't have what you need.
