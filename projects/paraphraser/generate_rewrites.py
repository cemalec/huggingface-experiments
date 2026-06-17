"""Stage B: generate (source, instruction, target) triples using a local instruct model.

First run downloads the model (~6GB for the default Qwen2.5-3B-Instruct in bf16).
Runs on Apple Silicon MPS by default; falls back to CUDA / CPU.

Usage:
    # Smoke test with built-in demo seeds (~18 triples, a few minutes)
    python generate_rewrites.py

    # Real run with a seed file (one sentence per line)
    python generate_rewrites.py --seeds path/to/seeds.txt --n-per-seed 3

    # Quick sanity-check run capped at N triples
    python generate_rewrites.py --limit 10

Output:
    data/triples.jsonl                       all generated triples
    review/triples_review_sample.jsonl       random subset for human inspection
"""

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline

HERE = Path(__file__).parent

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

DEMO_SEEDS = [
    "Quantum entanglement is a phenomenon in which two particles share a state such that measuring one instantaneously determines the state of the other.",
    "Inflation has remained above the central bank's target for the third consecutive quarter, prompting renewed debate about whether further rate hikes are warranted.",
    "The new running shoe uses a carbon-fiber plate embedded in the midsole to improve energy return during long-distance races.",
    "Most birds migrate in response to changes in day length rather than to changes in temperature.",
    "The proposed legislation would require all publicly listed companies to disclose their direct and indirect greenhouse gas emissions annually.",
    "She finally admitted that she had been wrong about the deadline, though she insisted it didn't change the overall plan.",
]

PROMPT_TEMPLATE = (
    "Rewrite the sentence below according to the instruction. "
    "Preserve the original meaning exactly — do not add, remove, or change any facts. "
    "Output only the rewrite, with no preamble, no surrounding quotes, and no commentary.\n\n"
    "Instruction: {instruction}\n\n"
    "Sentence: {source}"
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model id (default: {DEFAULT_MODEL})")
    p.add_argument("--instructions", default=str(HERE / "instructions.json"))
    p.add_argument("--seeds", default=None, help="Text file, one seed sentence per line. Defaults to built-in demo set.")
    p.add_argument("--n-per-seed", type=int, default=3, help="Instructions sampled per seed.")
    p.add_argument("--limit", type=int, default=None, help="Cap total triples (for quick test runs).")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--review-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=str(HERE / "data" / "triples.jsonl"))
    return p.parse_args()


def load_seeds(path: str | None) -> list[str]:
    if path is None:
        return DEMO_SEEDS
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def build_work(seeds: list[str], instructions: list[dict], n_per_seed: int, rng: random.Random) -> list[dict]:
    work = []
    for seed in seeds:
        for inst in rng.sample(instructions, k=min(n_per_seed, len(instructions))):
            work.append({"source": seed, "instruction": inst["instruction"], "axes": inst["axes"]})
    return work


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    instructions = json.loads(Path(args.instructions).read_text())
    seeds = load_seeds(args.seeds)
    work = build_work(seeds, instructions, args.n_per_seed, rng)
    if args.limit:
        work = work[: args.limit]
    print(f"Planned {len(work)} triples from {len(seeds)} seeds x {args.n_per_seed} instructions each.")

    device = get_device()
    print(f"Loading {args.model} on {device}...")
    pipe = pipeline(
        "text-generation",
        model=args.model,
        torch_dtype=torch.bfloat16,
        device=device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    triples = []
    with out_path.open("w", buffering=1) as f:
        for item in tqdm(work, desc="generating"):
            messages = [{
                "role": "user",
                "content": PROMPT_TEMPLATE.format(instruction=item["instruction"], source=item["source"]),
            }]
            out = pipe(messages, max_new_tokens=args.max_new_tokens, do_sample=False, return_full_text=False)
            target = out[0]["generated_text"].strip()

            triple = {
                "source": item["source"],
                "instruction": item["instruction"],
                "axes": item["axes"],
                "target": target,
            }
            triples.append(triple)
            f.write(json.dumps(triple) + "\n")
            f.flush()

    print(f"\nWrote {len(triples)} triples to {out_path}")

    review_dir = HERE / "review"
    review_dir.mkdir(exist_ok=True)
    review = rng.sample(triples, min(args.review_size, len(triples)))
    review_path = review_dir / "triples_review_sample.jsonl"
    with review_path.open("w") as f:
        for t in review:
            f.write(json.dumps(t) + "\n")
    print(f"Wrote {len(review)}-item review sample to {review_path}")


if __name__ == "__main__":
    main()
