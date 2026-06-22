/*
 * Stage B triple generation — Claude Code Workflow (teacher = Claude, no API key).
 * ============================================================================
 *
 * WHAT THIS IS
 *   A Claude Code *Workflow* script (NOT a node script — do not `node` it).
 *   It fans out one fresh subagent per (source x difficulty tier) and asks each
 *   to rewrite a source sentence under several style instructions, returning
 *   validated (source, instruction, axes, target) triples. Subagents run on the
 *   Claude Code plan, so this needs no ANTHROPIC_API_KEY / API budget.
 *
 *   Why this exists: a local Qwen teacher (since removed) failed the
 *   hard instruction axes (structural / voice / combinatorial). Claude obeys
 *   them. See CLAUDE.md "Stage B".
 *
 * BULK MODE (real ~5k seed corpus)
 *   Don't hand-edit this file for each chunk. Instead:
 *     1. Build seeds:        python build_seeds.py
 *     2. Sample instructions: python sample_instructions.py
 *     3. Emit chunk files:    python emit_chunks.py
 *        -> writes data/runs/chunk_NNN.workflow.js (one per <=1000-subagent batch),
 *           each with its own JOBS array, run_id, and gen_timestamp.
 *     4. Run each chunk:      Workflow({ scriptPath: "...chunk_NNN.workflow.js" })
 *   The chunk files are self-contained copies of this script with the
 *   EDIT-PER-RUN block replaced. This file's pilot JOBS below is a small
 *   sample kept for hand-editing / debugging.
 *
 * HOW TO RUN A SINGLE-CHUNK / PILOT (hand-edited, from a Claude Code session)
 *   1. Get a UTC timestamp for the run:   date -u +%Y-%m-%dT%H:%M:%SZ
 *   2. Edit the EDIT-PER-RUN block below: RUN.run_id, RUN.gen_timestamp, JOBS.
 *   3. Invoke the Workflow tool with this file:
 *        Workflow({ scriptPath: "projects/paraphraser/generate_triples_workflow.js" })
 *   4. It runs in the background; watch with /workflows. On completion it RETURNS
 *      { run_id, count, failed, triples } — it does NOT write any file (the
 *      sandbox has no filesystem access). The orchestrator persists the result:
 *        # `OUT` = the tool's output file path from the completion notification
 *        python3 - <<'PY'
 *        import json, os
 *        os.makedirs("projects/paraphraser/data", exist_ok=True)
 *        d = json.load(open("OUT"))
 *        with open("projects/paraphraser/data/triples.claude-code.jsonl","a") as f:  # append; gitignored bulk data
 *            for t in d["result"]["triples"]:
 *                f.write(json.dumps(t, ensure_ascii=False) + "\n")
 *        PY
 *   5. Record the run in triples.runs.md (one row per run_id).
 *   6. Filter:  python filter_triples.py          (defaults to triples.claude-code.jsonl)
 *      -> writes *.filtered.jsonl (train on this) + *.flagged.jsonl (review).
 *      Add --bertscore for the semantic layer (needs: pip install bert_score).
 *
 * PROVENANCE (stamped on every row, do not drop these on future runs)
 *   tier, gen_model, gen_backend, prompt_version, run_id, gen_timestamp
 *
 * GOTCHAS (learned the hard way)
 *   - The sandbox statically REJECTS the literal tokens `Date.now`, `Math.random`,
 *     `new Date` — even inside comments. That's why timestamps are generated
 *     outside and inlined (step 1 above).
 *   - The Workflow `args` parameter did NOT bind into the script's `args` global
 *     in this harness, so config is inlined as consts below instead.
 *   - For the real ~5k-seed corpus: don't put 5k sources in one run (subagent
 *     lifetime cap is 1000). Chunk into multiple invocations, each with its own
 *     run_id, appending to triples.claude-code.jsonl.
 */

export const meta = {
  name: 'paraphrase-triples',
  description: 'Generate tagged (source, instruction, target) paraphrase triples with Claude as teacher (Haiku=easy, Sonnet=hard)',
  phases: [{ title: 'Generate', detail: 'one fresh subagent per (source x tier) job' }],
}

// ============================ EDIT PER RUN ==================================
// JOBS holds one entry per subagent. Each entry carries its own instruction
// draw (per-subagent sample from the bank), tier (model routing), and the
// seed-corpus metadata that's stamped onto every emitted triple.
//
// For bulk runs this whole block is replaced by emit_chunks.py.

const RUN = {
  run_id: 'claude-pilot-20260617T112732Z',   // unique per run; convention: claude-<label>-<timestamp>
  gen_timestamp: '2026-06-17T11:27:32Z',      // from: date -u +%Y-%m-%dT%H:%M:%SZ
  gen_backend: 'claude-code-workflow',
  prompt_version: 'v1',                        // bump if buildPrompt() changes, so rows stay distinguishable
}

const JOBS = [
  { seed_idx: 0, source: 'Most birds migrate in response to changes in day length rather than to changes in temperature.',
    domain: 'pilot', source_id: 'pilot:0',
    tier: 'easy', model: 'haiku', model_id: 'claude-haiku-4-5',
    instructions: [
      { instruction: 'Recast as a weather-report line.', axes: ['genre'] },
      { instruction: 'Compress into exactly one short declarative sentence.', axes: ['length'] },
    ] },
  { seed_idx: 0, source: 'Most birds migrate in response to changes in day length rather than to changes in temperature.',
    domain: 'pilot', source_id: 'pilot:0',
    tier: 'hard', model: 'sonnet', model_id: 'claude-sonnet-4-6',
    instructions: [
      { instruction: 'Render as a chiasmus where the second half inverts the first.', axes: ['structural'] },
      { instruction: 'Recast in the laconic, present-tense voice of a hardboiled detective narrator.', axes: ['voice'] },
    ] },
]
// ========================== END EDIT PER RUN ================================

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['triples'],
  properties: {
    triples: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['source', 'instruction', 'axes', 'target'],
      properties: {
        source:      { type: 'string' },
        instruction: { type: 'string' },
        axes:        { type: 'array', items: { type: 'string' } },
        target:      { type: 'string' },
      },
    }},
  },
}

function buildPrompt(job) {
  const list = job.instructions
    .map((it, i) => `${i + 1}. ${it.instruction}  (axes: ${it.axes.join(', ')})`).join('\n')
  return `You are a careful rewriting teacher generating training data for a paraphrase model.

SOURCE:
"""${job.source}"""

For EACH instruction below, rewrite the SOURCE so that it:
- genuinely BECOMES the thing the instruction asks for (an actual weather report, a real chiasmus with ABBA inversion, exact word counts when demanded, etc.),
- preserves the core meaning and facts of the source (introduce no new claims, drop nothing essential),
- is a single self-contained rewrite.

INSTRUCTIONS:
${list}

Return one triple per instruction. Echo the source text and the instruction's axes verbatim in each triple.`
}

phase('Generate')
log(`Generating ${JOBS.length} subagents`)

const results = await parallel(
  JOBS.map((job) => () =>
    agent(buildPrompt(job), {
      label: `${job.tier}:${job.source.slice(0, 32)}`,
      phase: 'Generate',
      model: job.model,
      schema: SCHEMA,
    })
  )
)

const triples = results.flatMap((r, i) => {
  const job = JOBS[i]
  const rows = (r && r.triples) ? r.triples : []
  return rows.map((t) => ({
    source: t.source,
    instruction: t.instruction,
    axes: t.axes,
    target: t.target,
    domain: job.domain,
    source_id: job.source_id,
    tier: job.tier,
    gen_model: job.model_id,
    gen_backend: RUN.gen_backend,
    prompt_version: RUN.prompt_version,
    run_id: RUN.run_id,
    gen_timestamp: RUN.gen_timestamp,
  }))
})

const failed = results.filter((r) => !r).length
log(`Collected ${triples.length} triples (${failed} subagents returned nothing)`)
return { run_id: RUN.run_id, count: triples.length, failed, triples }
