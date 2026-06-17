"""Stage A: ask Claude to generate a diverse bank of paraphrase-rewrite instructions.

Run once. Output (`instructions.json`) is then fed into Stage B, which pairs each
seed sentence with 3-5 randomly sampled instructions and asks Claude for the
rewrite + faithfulness/style ratings. Stage B produces the SFT training set.

Usage:
    export ANTHROPIC_API_KEY=...
    python generate_instructions.py
"""

import json
import random
from pathlib import Path

import anthropic

COUNT = 250
REVIEW_SAMPLE_SIZE = 70
REVIEW_SEED = 0
OUTPUT_PATH = Path(__file__).parent / "instructions.json"
REVIEW_DIR = Path(__file__).parent / "review"
REVIEW_PATH = REVIEW_DIR / "instructions_review_sample.json"

SYSTEM_PROMPT = """You are helping design the training data for a small paraphraser model. The student model will be fine-tuned on (instruction, source_sentence, rewrite) triples so that, at inference time, a user can ask it to rewrite arbitrary text in an arbitrary style via natural-language instruction. Your job here is to design the *instruction bank* that defines the style space the student will learn."""

USER_PROMPT = f"""Generate exactly {COUNT} diverse rewrite instructions for a paraphraser.

# What a good instruction looks like

Each instruction is a single imperative sentence telling a writer how to rewrite a given source sentence. The instruction must specify *style*, not *content* — the rewrite should preserve the source's meaning while changing how it's expressed.

Aim for a deliberate mix of two kinds of instructions:

1. **Single-axis instructions** that exercise one stylistic dimension cleanly (e.g. *"Rewrite in a hedged, academic register"* — register only). These give the student model clean signal for each individual axis. Cover every axis category below at least twice with single-axis instructions, using different phrasings the second time.

2. **Combinatorial instructions** that compose two axes into a recognizable real-world style (e.g. *"Rewrite as a confident startup pitch to a skeptical investor"* combines tone + voice + audience). The interesting stylistic territory lives in these combinations — be generous with them. Roughly half of the bank should be combinatorial.

Span these axes:
- **Register**: technical / plain / academic / colloquial / archaic
- **Audience**: expert / novice / child / executive / skeptic
- **Tone**: neutral / persuasive / urgent / reassuring / skeptical / playful / formal
- **Length & density**: terse / expansive / hedged / blunt
- **Genre / framing**: tweet / headline / abstract / FAQ answer / dialogue line / marketing copy / textbook sentence
- **Structural shifts**: active↔passive / nominalize↔verbalize / lead-with-conclusion↔lead-with-evidence
- **Voice / persona**: research paper / startup pitch / news wire / instructional manual / personal essay

# What to avoid

- Anything that changes the meaning: no "translate", "summarize the key points", "fact-check", "add an example", "remove the qualifier".
- Anything pure-format with no prose impact: no "output as JSON", "wrap in quotes".
- Vague non-instructions: no "make it better", "improve clarity", "polish it".
- Near-duplicates. "Rewrite for a non-technical audience" and "Rewrite so a layperson can understand" are duplicates; pick one.
- Instructions that bundle three or more axes — keep each one focused.

# Examples of the right shape

- Rewrite as a single sentence suitable for a research-paper abstract.
- Rewrite to sound like a confident startup pitch to a skeptical investor.
- Rewrite in the plainest possible English, as if explaining to a curious 12-year-old.
- Rewrite so the main conclusion comes first and the supporting reason follows.
- Rewrite as a punchy news headline of at most 12 words.
- Rewrite in a hedged, academic register that signals appropriate uncertainty.
- Rewrite as a line of dialogue a frustrated user might say out loud.
- Rewrite to be ~30% shorter without losing any factual content.

# Output

For each instruction, also tag which axes it primarily targets (1-2 axis labels from the list above, lowercased, e.g. "register", "tone", "genre"). This tagging is for downstream diversity tracking — be honest about which axes the instruction actually exercises.

Produce exactly {COUNT} instructions. Aim for even coverage across the axes; do not let any single axis dominate."""

SCHEMA = {
    "type": "object",
    "properties": {
        "instructions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "axes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["instruction", "axes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["instructions"],
    "additionalProperties": False,
}


def main() -> None:
    client = anthropic.Anthropic()

    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=32000,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        final = stream.get_final_message()

    print()

    payload_text = next(b.text for b in final.content if b.type == "text")
    payload = json.loads(payload_text)
    instructions = payload["instructions"]

    OUTPUT_PATH.write_text(json.dumps(instructions, indent=2))

    REVIEW_DIR.mkdir(exist_ok=True)
    rng = random.Random(REVIEW_SEED)
    review_sample = rng.sample(instructions, min(REVIEW_SAMPLE_SIZE, len(instructions)))
    REVIEW_PATH.write_text(json.dumps(review_sample, indent=2))

    axis_counts: dict[str, int] = {}
    for entry in instructions:
        for axis in entry["axes"]:
            axis_counts[axis] = axis_counts.get(axis, 0) + 1

    print(f"\nWrote {len(instructions)} instructions to {OUTPUT_PATH}")
    print(f"Wrote {len(review_sample)}-item review sample to {REVIEW_PATH}")
    print(f"Token usage: input={final.usage.input_tokens}, output={final.usage.output_tokens}")
    print("\nAxis coverage:")
    for axis, n in sorted(axis_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {axis:24s} {n}")


if __name__ == "__main__":
    main()
