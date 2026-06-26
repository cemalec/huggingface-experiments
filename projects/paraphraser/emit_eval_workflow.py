"""
Holdout evaluation — Step 2: emit judge workflow.

Reads data/eval_judge_inputs.jsonl and writes data/eval_judge_workflow.js —
a self-contained Claude Code Workflow that fans out 50 Opus subagents, one per
example, each scoring both checkpoints on:
  - Instruction adherence (0–5): did the output follow the style instruction?
  - Semantic faithfulness (0–5): does the output preserve the source meaning?
  - Fluency (0–5): is the output natural, well-formed prose?

All three scores are assessed for epoch-1 and epoch-3 outputs side-by-side in
one subagent call, so the judge always has both outputs in context and can score
them relative to the same standard.

Usage:
    python emit_eval_workflow.py
    # then in Claude Code:
    Workflow({ scriptPath: "projects/paraphraser/data/eval_judge_workflow.js" })
    # After the workflow completes, save the returned scores:
    #   python aggregate_eval_scores.py --scores-json '<workflow output path>'
"""
import argparse
import json
import os
import textwrap
from datetime import datetime, timezone

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", default="data/eval_judge_inputs.jsonl")
parser.add_argument("--output", default=None,
                    help="JS output path; defaults to inputs path with .workflow.js suffix")
args = parser.parse_args()

INPUTS = args.inputs
OUT    = args.output or INPUTS.replace(".jsonl", ".workflow.js")


def get_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


with open(INPUTS) as f:
    jobs = [json.loads(line) for line in f]
print(f"Loaded {len(jobs)} holdout examples")

ts = get_timestamp()
run_id = f"holdout-judge-{ts.replace(':', '').replace('-', '')}"

# Serialize the jobs array for embedding in JS.
# We JSON-dump the full list and do minimal formatting — no Date.now or Math.random.
jobs_json = json.dumps(jobs, ensure_ascii=False, indent=2)

script = textwrap.dedent(f"""\
/*
 * Holdout LLM-as-judge evaluation — Claude Code Workflow.
 * =======================================================
 *
 * Fans out 50 Opus subagents, one per holdout example.  Each subagent
 * receives both checkpoint outputs and scores three dimensions per checkpoint:
 *   adherence  — did the output follow the style instruction?  (0–5)
 *   faithfulness — does it preserve the source meaning?       (0–5)
 *   fluency    — is it natural, well-formed prose?             (0–5)
 *
 * run_id:  {run_id}
 * generated: {ts}
 *
 * After this workflow completes, call aggregate_eval_scores.py on the result.
 */

export const meta = {{
  name: 'holdout-judge-eval',
  description: 'LLM-as-judge holdout eval: score epoch-1 vs epoch-3 on 50 unseen (source, instruction) pairs',
  phases: [{{ title: 'Judge', detail: 'one Opus subagent per holdout example' }}],
}}

const RUN_ID = '{run_id}';

const JOBS = {jobs_json};

const SCORE_SCHEMA = {{
  type: 'object',
  properties: {{
    id:                  {{ type: 'integer' }},
    e1_adherence:        {{ type: 'integer', minimum: 0, maximum: 5 }},
    e1_faithfulness:     {{ type: 'integer', minimum: 0, maximum: 5 }},
    e1_fluency:          {{ type: 'integer', minimum: 0, maximum: 5 }},
    e1_notes:            {{ type: 'string' }},
    e3_adherence:        {{ type: 'integer', minimum: 0, maximum: 5 }},
    e3_faithfulness:     {{ type: 'integer', minimum: 0, maximum: 5 }},
    e3_fluency:          {{ type: 'integer', minimum: 0, maximum: 5 }},
    e3_notes:            {{ type: 'string' }},
  }},
  required: ['id','e1_adherence','e1_faithfulness','e1_fluency','e1_notes',
             'e3_adherence','e3_faithfulness','e3_fluency','e3_notes'],
}};

phase('Judge');

const results = await parallel(JOBS.map(job => async () => {{
  const prompt = `You are a rigorous style-transfer evaluator.  Your task is to score two candidate rewrites of a source sentence.  Both rewrites attempted to follow the SAME style instruction.  They were produced by the same model trained for different numbers of epochs: EPOCH-1 is the earlier checkpoint, EPOCH-3 is the later checkpoint.

SOURCE:
"${{job.source}}"

STYLE INSTRUCTION:
"${{job.instruction}}"

EPOCH-1 OUTPUT:
"${{job.epoch1_output}}"

EPOCH-3 OUTPUT:
"${{job.epoch3_output}}"

Score each output on three dimensions.  Use the full 0–5 range; do not cluster scores around 3.

ADHERENCE (0–5) — how faithfully did the output follow the style instruction?
  5 = nails the instruction: register, persona, format, structural constraint all satisfied
  4 = strong, minor gap
  3 = partial — some axes obeyed, others ignored
  2 = superficial — label applied but style not convincingly rendered
  1 = barely attempted
  0 = ignored the instruction entirely

FAITHFULNESS (0–5) — does the output preserve the meaning of the source?
  5 = all core information retained, no additions or distortions
  4 = minor loss or slight addition, meaning intact
  3 = some information dropped or slightly changed
  2 = significant loss or distortion
  1 = most meaning lost
  0 = hallucinated / unrelated content

FLUENCY (0–5) — is the output natural, grammatically correct, well-formed prose?
  5 = polished, natural in its register
  4 = clearly readable, minor awkwardness
  3 = understandable but noticeably stilted
  2 = disfluent, hard to follow
  1 = fragmentary or near-ungrammatical
  0 = incoherent

Return a JSON object matching the schema exactly.  id must equal ${{job.id}}.  Provide one short sentence of notes per checkpoint (e1_notes, e3_notes) identifying the main strength or weakness.`;

  const score = await agent(prompt, {{
    label: `judge:${{job.id}}`,
    phase: 'Judge',
    model: 'claude-opus-4-8',
    schema: SCORE_SCHEMA,
  }});
  if (!score) return null;
  return {{ ...score, source: job.source, axes: job.axes, instruction: job.instruction }};
}}));

const valid = results.filter(Boolean);
log(`Scored ${{valid.length}} / ${{JOBS.length}} examples`);

return {{
  run_id: RUN_ID,
  count: valid.length,
  scores: valid,
}};
""")

os.makedirs("data", exist_ok=True)
with open(OUT, "w") as f:
    f.write(script)

print(f"Wrote {OUT}  ({len(jobs)} jobs, run_id={run_id})")
print()
print("Next steps:")
print("  1. Run the workflow in Claude Code:")
print(f'       Workflow({{ scriptPath: "projects/paraphraser/{OUT}" }})')
print("  2. After it completes, save the scores JSON from the workflow output")
print("     to data/eval_judge_scores.jsonl, then:")
print("       python aggregate_eval_scores.py")
