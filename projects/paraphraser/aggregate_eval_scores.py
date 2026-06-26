"""
Holdout evaluation — Step 3: aggregate scores.

Reads the judge scores returned by the workflow and writes a per-axis summary
table to review/eval_judge_summary.md.

Usage (two equivalent paths):

  A. Direct from the workflow result JSON file (path printed by the workflow
     completion notification):
       python aggregate_eval_scores.py --scores-json /path/to/workflow-result.json

  B. From a JSONL file if you already saved the scores:
       python aggregate_eval_scores.py --scores-jsonl data/eval_judge_scores.jsonl

Either form also saves data/eval_judge_scores.jsonl so both inputs are always
available for later re-runs.
"""
import argparse
import json
import os
import statistics
from collections import defaultdict

OUT_MD   = "review/eval_judge_summary.md"
OUT_JSONL = "data/eval_judge_scores.jsonl"

AXES = ["register", "audience", "tone", "length", "genre", "structural", "voice"]
DIMS = [("adherence", "Adherence"), ("faithfulness", "Faithfulness"), ("fluency", "Fluency")]


def load_scores(args):
    if args.scores_json:
        with open(args.scores_json) as f:
            data = json.load(f)
        # Support both the raw workflow result object and its "result" subkey
        if "result" in data:
            data = data["result"]
        return data.get("scores", data) if isinstance(data, dict) else data
    elif args.scores_jsonl:
        with open(args.scores_jsonl) as f:
            return [json.loads(line) for line in f]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-json", default=None, action="append",
                        help="Path to raw workflow result JSON file (repeatable)")
    parser.add_argument("--scores-jsonl", default=None, action="append",
                        help="Path to scores JSONL file (repeatable)")
    parser.add_argument("--inputs", default=None, action="append",
                        help="eval_judge_inputs.jsonl file(s) to backfill missing source sentences (repeatable)")
    args = parser.parse_args()

    # Collect from all provided sources
    scores = []
    if args.scores_json:
        for path in args.scores_json:
            args_single = argparse.Namespace(scores_json=path, scores_jsonl=None)
            scores.extend(load_scores(args_single))
    if args.scores_jsonl:
        for path in args.scores_jsonl:
            args_single = argparse.Namespace(scores_json=None, scores_jsonl=path)
            scores.extend(load_scores(args_single))
    if not scores:
        raise ValueError("Provide --scores-json or --scores-jsonl")
    print(f"Loaded {len(scores)} scored examples")

    # Backfill source sentences from inputs files when scores lack them
    if args.inputs:
        source_lookup = {}
        for path in args.inputs:
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    # Key by (id, instruction) to handle multi-batch runs safely
                    source_lookup[(row["id"], row["instruction"])] = row["source"]
        filled = 0
        for row in scores:
            if "source" not in row:
                key = (row.get("id"), row.get("instruction", ""))
                if key in source_lookup:
                    row["source"] = source_lookup[key]
                    filled += 1
        if filled:
            print(f"Backfilled source for {filled} rows from --inputs")

    # Always save combined JSONL
    os.makedirs("data", exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for row in scores:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {OUT_JSONL}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    # Group scores by primary axis (multi-axis examples counted in each of their axes)
    by_axis = defaultdict(list)
    for row in scores:
        for ax in row.get("axes", ["unknown"]):
            by_axis[ax].append(row)

    # Overall pool (all examples)
    by_axis["overall"] = scores

    # ── Build markdown ────────────────────────────────────────────────────────
    lines = ["# Holdout Evaluation — LLM-as-Judge Summary\n"]
    lines.append("**Eval set**: 50 examples with instructions not present in the 250-instruction training bank.")
    lines.append("Multi-axis examples are counted in each of their axes.")
    lines.append("Scores are Opus-assigned on 0–5 scales (adherence / faithfulness / fluency).\n")

    def fmt(vals):
        if not vals:
            return "—"
        return f"{statistics.mean(vals):.2f}"

    # Per-axis table
    lines.append("## Per-axis mean scores\n")
    header = "| Axis | N | E1 Adh | E1 Faith | E1 Flu | E3 Adh | E3 Faith | E3 Flu | Δ Adh | Δ Faith | Δ Flu |"
    divider = "|------|---|--------|----------|--------|--------|----------|--------|-------|---------|-------|"
    lines.append(header)
    lines.append(divider)

    ordered_axes = AXES + ["overall"]
    for ax in ordered_axes:
        rows = by_axis.get(ax, [])
        n = len(rows)
        if n == 0:
            continue
        e1_adh  = [r["e1_adherence"] for r in rows]
        e1_fth  = [r["e1_faithfulness"] for r in rows]
        e1_flu  = [r["e1_fluency"] for r in rows]
        e3_adh  = [r["e3_adherence"] for r in rows]
        e3_fth  = [r["e3_faithfulness"] for r in rows]
        e3_flu  = [r["e3_fluency"] for r in rows]
        d_adh   = statistics.mean(e3_adh) - statistics.mean(e1_adh)
        d_fth   = statistics.mean(e3_fth) - statistics.mean(e1_fth)
        d_flu   = statistics.mean(e3_flu) - statistics.mean(e1_flu)
        label = f"**{ax}**" if ax == "overall" else ax
        lines.append(
            f"| {label} | {n} | {fmt(e1_adh)} | {fmt(e1_fth)} | {fmt(e1_flu)} "
            f"| {fmt(e3_adh)} | {fmt(e3_fth)} | {fmt(e3_flu)} "
            f"| {d_adh:+.2f} | {d_fth:+.2f} | {d_flu:+.2f} |"
        )

    # Overall summary
    e1_total = [r["e1_adherence"] + r["e1_faithfulness"] + r["e1_fluency"] for r in scores]
    e3_total = [r["e3_adherence"] + r["e3_faithfulness"] + r["e3_fluency"] for r in scores]
    lines.append(
        f"\n**Epoch-1 mean composite** (sum of 3 dims): {fmt(e1_total)} / 15"
    )
    lines.append(
        f"**Epoch-3 mean composite**: {fmt(e3_total)} / 15  "
        f"(Δ {statistics.mean(e3_total) - statistics.mean(e1_total):+.2f})"
    )

    # Win/loss/tie
    e3_wins = sum(1 for r in scores if r["e3_adherence"] + r["e3_faithfulness"] + r["e3_fluency"]
                  > r["e1_adherence"] + r["e1_faithfulness"] + r["e1_fluency"])
    e1_wins = sum(1 for r in scores if r["e3_adherence"] + r["e3_faithfulness"] + r["e3_fluency"]
                  < r["e1_adherence"] + r["e1_faithfulness"] + r["e1_fluency"])
    ties    = len(scores) - e3_wins - e1_wins
    lines.append(f"\n**Win/Tie/Loss (epoch-3 vs epoch-1, by composite)**: "
                 f"{e3_wins} W / {ties} T / {e1_wins} L\n")

    # Per-example details
    lines.append("## Per-example scores\n")
    lines.append("| # | Axes | E1 Adh | E1 Faith | E1 Flu | E3 Adh | E3 Faith | E3 Flu | Winner |")
    lines.append("|---|------|--------|----------|--------|--------|----------|--------|--------|")
    for row in sorted(scores, key=lambda r: r["id"]):
        e1 = row["e1_adherence"] + row["e1_faithfulness"] + row["e1_fluency"]
        e3 = row["e3_adherence"] + row["e3_faithfulness"] + row["e3_fluency"]
        winner = "E3" if e3 > e1 else ("E1" if e1 > e3 else "tie")
        ax_str = ", ".join(row.get("axes", []))
        lines.append(
            f"| {row['id']+1} | {ax_str} "
            f"| {row['e1_adherence']} | {row['e1_faithfulness']} | {row['e1_fluency']} "
            f"| {row['e3_adherence']} | {row['e3_faithfulness']} | {row['e3_fluency']} "
            f"| {winner} |"
        )

    # Notes section
    lines.append("\n## Judge notes\n")
    for row in sorted(scores, key=lambda r: r["id"]):
        instr = row.get("instruction", "")
        source = row.get("source", "")
        lines.append(f"**{row['id']+1}. {instr}**")
        if source:
            lines.append(f"*Source: {source}*")
        lines.append(f"- E1 ({row.get('e1_adherence','?')}/{row.get('e1_faithfulness','?')}/{row.get('e1_fluency','?')}): {row.get('e1_notes', '')}")
        lines.append(f"- E3 ({row.get('e3_adherence','?')}/{row.get('e3_faithfulness','?')}/{row.get('e3_fluency','?')}): {row.get('e3_notes', '')}\n")

    os.makedirs("review", exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
