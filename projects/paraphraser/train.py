"""
LoRA SFT of a causal instruct model on (instruction, source) → target triples.

Install missing deps first:
    pip install trl peft

Usage:
    python train.py                                       # all defaults
    python train.py --model Qwen/Qwen2.5-3B-Instruct     # larger model
    python train.py --rank 32 --epochs 2                 # tune LoRA rank / epochs
    python train.py --batch-size 2 --grad-accum 16       # if memory is tight

Outputs:
    checkpoints/paraphraser/   — LoRA adapter weights + tokenizer
    (base model weights are NOT saved; load base + adapter at inference time)

Inference after training:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/paraphraser")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, "checkpoints/paraphraser")
    # Then build messages with the same format as make_messages() below and generate.
"""

import argparse
import json
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATA = "data/triples.claude-code.filtered.jsonl"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_triples(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def make_messages(row: dict) -> list[dict]:
    """Return a conversational messages list for one triple.

    SFTTrainer (TRL ≥1.0) auto-detects a 'messages' column and applies the
    model's chat template; assistant_only_loss=True then masks the user turn
    so loss is computed only on the target rewrite.
    """
    return [
        {
            "role": "user",
            "content": (
                "Rewrite the following text according to the instruction.\n\n"
                f"Instruction: {row['instruction']}\n\n"
                f"Text: {row['source']}"
            ),
        },
        {"role": "assistant", "content": row["target"]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for paraphraser student")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HF model id or local path (default: Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank r; increase to 32 if underfitting (default: 16)")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default="checkpoints/paraphraser")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Effective batch = batch-size × grad-accum (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=256,
                        help="Sequence length cap; 256 covers ~99% of our triples (default: 256)")
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub-dataset", default=None,
                        help="HF Hub dataset ID to load instead of local --data file")
    parser.add_argument("--push-adapter-to", default=None,
                        help="HF Hub model ID to push adapter after training")
    parser.add_argument("--resume-from", default=None,
                        help="Hub checkpoint to resume: 'repo_id:subfolder', "
                             "e.g. cemalec/paraphraser-adapter:checkpoint-1236")
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}  model: {args.model}  rank: {args.rank}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Data ───────────────────────────────────────────────────────────────────
    if args.hub_dataset:
        from datasets import load_dataset as _hf_load
        _ds = _hf_load(args.hub_dataset, split="train")
        rows = [dict(r) for r in _ds]
        print(f"loaded {len(rows)} rows from {args.hub_dataset}")
    else:
        rows = load_triples(args.data)
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    messages = [make_messages(r) for r in rows]
    val_n = max(1, int(len(messages) * args.val_split))
    train_ds = Dataset.from_dict({"messages": messages[val_n:]})
    val_ds   = Dataset.from_dict({"messages": messages[:val_n]})
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    # bfloat16 weights throughout; no quantization (bitsandbytes requires CUDA).
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.config.use_cache = True

    # ── LoRA ───────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training config ─────────────────────────────────────────────────────────
    # SFTConfig extends TrainingArguments with SFT-specific knobs.
    # assistant_only_loss=True: loss computed only on the assistant turn (the target
    # rewrite), not on the user prompt. Replaces DataCollatorForCompletionOnlyLM
    # from TRL <1.0.
    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=False,   # GC is extremely slow on MPS (~16× slowdown); keep off
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=(device == "cuda"),        # AMP only on CUDA; MPS trains in native bfloat16
        fp16=False,
        max_length=args.max_seq_len,
        assistant_only_loss=True,       # mask user-turn tokens from loss
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=(device == "cuda"),
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,     # replaces 'tokenizer=' in TRL ≥1.0
    )

    resume_checkpoint = None
    if args.resume_from:
        repo_id, _, subfolder = args.resume_from.partition(":")
        local_dir = "/tmp/resume_ckpt"
        print(f"Downloading checkpoint from {args.resume_from} ...", flush=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subfolder}/*"] if subfolder else None,
            local_dir=local_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        resume_checkpoint = os.path.join(local_dir, subfolder) if subfolder else local_dir
        print(f"Resuming from {resume_checkpoint}", flush=True)

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved adapter + tokenizer → {args.output}")

    if args.push_adapter_to:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_adapter_to, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=args.output,
            repo_id=args.push_adapter_to,
            repo_type="model",
        )
        print(f"pushed adapter → huggingface.co/{args.push_adapter_to}")


if __name__ == "__main__":
    main()
