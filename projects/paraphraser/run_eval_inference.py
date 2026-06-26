"""
Holdout evaluation — Step 1: inference.

Runs 50 holdout (source, instruction) pairs through the epoch-1 and epoch-3
checkpoints and writes data/eval_judge_inputs.jsonl.

"Holdout" means:
  - Sources: val-split rows starting at --skip (never seen during training).
  - Instructions: eval_instructions.json (50 new framings not in the
    250-instruction training bank).

Usage:
    python run_eval_inference.py                          # batch A (rows 20-69)
    python run_eval_inference.py --skip 70 \
        --output data/eval_judge_inputs_b.jsonl           # batch B (rows 70-119)
"""
import argparse
import json
import os
import random
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA              = "data/triples.claude-code.filtered.jsonl"
EVAL_INSTRUCTIONS = "eval_instructions.json"
BASE              = "Qwen/Qwen2.5-1.5B-Instruct"
N_EVAL            = 50

CHECKPOINTS = [
    ("epoch1", "cemalec/paraphraser-adapter", "checkpoint-1236"),
    ("epoch3", "cemalec/paraphraser-adapter", "checkpoint-3708"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--skip", type=int, default=20,
                    help="Val rows to skip before taking N_EVAL sources (default: 20)")
parser.add_argument("--output", default="data/eval_judge_inputs.jsonl")
args = parser.parse_args()

N_SKIP = args.skip
OUT    = args.output

# ── Reproduce the exact val split from train.py ───────────────────────────────
with open(DATA) as f:
    rows = [json.loads(line) for line in f]

rng = random.Random(42)
rng.shuffle(rows)
val_n = max(1, int(len(rows) * 0.02))
val_rows = rows[:val_n]
print(f"Val split: {val_n} rows")

# Take rows [N_SKIP : N_SKIP + N_EVAL] as holdout sources.
# These sources were never seen during training AND the paired instructions are
# entirely new (not in the training bank), so the combination is doubly held out.
sources = val_rows[N_SKIP : N_SKIP + N_EVAL]
if len(sources) < N_EVAL:
    raise ValueError(
        f"Val split too small: needed {N_SKIP + N_EVAL} rows, got {val_n}. "
        "Reduce N_SKIP or N_EVAL."
    )
print(f"Holdout sources: {len(sources)} (val rows {N_SKIP}–{N_SKIP + N_EVAL - 1})")

# ── Load holdout instructions ─────────────────────────────────────────────────
with open(EVAL_INSTRUCTIONS) as f:
    eval_instructions = json.load(f)
assert len(eval_instructions) == N_EVAL, f"Expected {N_EVAL} eval instructions, got {len(eval_instructions)}"
print(f"Holdout instructions: {len(eval_instructions)}")

# ── Inference helper ──────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
token  = os.environ.get("HF_TOKEN")
print(f"Device: {device}")


def run_checkpoint(repo_id, subfolder, sources, instructions):
    """Run all (source, instruction) pairs through one checkpoint."""
    print(f"\nLoading {repo_id}/{subfolder} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map={"": device}, token=token
    )
    model = PeftModel.from_pretrained(model, repo_id, subfolder=subfolder, token=token)
    model.eval()

    outputs = []
    for i, (row, instr_entry) in enumerate(zip(sources, instructions)):
        messages = [{"role": "user", "content": (
            "Rewrite the following text according to the instruction.\n\n"
            f"Instruction: {instr_entry['instruction']}\n\nText: {row['source']}"
        )}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        outputs.append(generated)
        print(f"  {i+1}/{N_EVAL}: {instr_entry['instruction'][:55]}...", flush=True)

    del model
    if device == "mps":
        torch.mps.empty_cache()
    return outputs


# ── Run both checkpoints ──────────────────────────────────────────────────────
all_outputs = {}
for label, repo_id, subfolder in CHECKPOINTS:
    all_outputs[label] = run_checkpoint(repo_id, subfolder, sources, eval_instructions)

# ── Write judge inputs ────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
with open(OUT, "w") as f:
    for i, (row, instr_entry) in enumerate(zip(sources, eval_instructions)):
        record = {
            "id": i,
            "source": row["source"],
            "instruction": instr_entry["instruction"],
            "axes": instr_entry["axes"],
            "epoch1_output": all_outputs["epoch1"][i],
            "epoch3_output": all_outputs["epoch3"][i],
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\nWrote {N_EVAL} judge inputs → {OUT}")
