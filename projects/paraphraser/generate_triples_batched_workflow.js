/*
 * Stage B triple generation — BATCHED Claude Code Workflow (teacher = Claude, no API key).
 * ============================================================================
 *
 * WHAT THIS IS
 *   The batched successor to generate_triples_workflow.js. Each subagent now
 *   rewrites SEVERAL sources (same tier) instead of one, amortizing the large
 *   fixed per-subagent overhead (~11k tokens of system prompt + tool schema)
 *   across many more triples. Measured: ~2,800 tok/triple at 1 source/subagent
 *   -> ~775 tok/triple at 4, and lower still at 8. See triples.runs.md.
 *
 *   Each triple self-identifies its source via "source_ref" (S1, S2, ...), so
 *   rewrites map back to the right source even with mixed domains in one prompt.
 *   The batch test measured 0/128 mis-mappings; this script logs `mismapped` so
 *   regressions surface immediately.
 *
 * BULK MODE
 *     1. Build seeds:         python build_seeds.py
 *     2. Sample instructions: python sample_instructions.py
 *     3. Emit batched chunks: python emit_chunks_batched.py --batch-sources 8
 *        -> writes data/runs/batched/chunk_NNN.workflow.js (each <512KB, the
 *           Workflow script-size cap), skipping already-generated (source_id,tier).
 *     4. Run each chunk:       Workflow({ scriptPath: "<abs path>/chunk_NNN.workflow.js" })
 *   This file's BATCHES below is a 2-batch sample kept for hand-debugging.
 *
 * PERSIST + LOG (same as the single-source flow)
 *   On completion it RETURNS { run_id, n_subagents, count, mismapped, failed, triples }
 *   and writes nothing (sandbox has no FS). Orchestrator appends result.triples to
 *   data/triples.claude-code.jsonl and records the run in triples.runs.md.
 *
 * PROVENANCE (stamped on every row): tier, gen_model, gen_backend, prompt_version,
 *   run_id, gen_timestamp. prompt_version is 'v1-batch' to distinguish batched rows.
 *
 * GOTCHAS: the sandbox statically rejects literal `Date.now`/`Math.random`/`new Date`
 *   even in comments (timestamps inlined from Python); pass scriptPath as an ABSOLUTE
 *   path (relative paths get doubled against the project dir).
 */

export const meta = {
  name: 'paraphrase-triples-batched',
  description: 'Generate tagged paraphrase triples, multiple sources per subagent (Haiku=easy, Sonnet=hard)',
  phases: [{ title: 'Generate', detail: 'one subagent per (batch-of-sources x tier)' }],
}

// ============================ EDIT PER RUN ==================================
// Replaced by emit_chunks_batched.py. BATCHES holds one entry per subagent;
// each entry has a `sources` array (its own instruction draws) + tier/model.

const RUN = {
  run_id: 'batched-pilot-PLACEHOLDER',
  gen_timestamp: '2026-06-18T00:00:00Z',
  gen_backend: 'claude-code-workflow',
  prompt_version: 'v1-batch',
}

const BATCHES = [
  { tier: 'easy', model: 'haiku', model_id: 'claude-haiku-4-5', sources: [
    { ref: 'S1', source: 'Most birds migrate in response to changes in day length rather than to changes in temperature.',
      domain: 'pilot', source_id: 'pilot:0',
      instructions: [ { instruction: 'Recast as a weather-report line.', axes: ['genre'] },
                      { instruction: 'Compress into exactly one short declarative sentence.', axes: ['length'] } ] },
    { ref: 'S2', source: 'The library extends its hours during final exams.',
      domain: 'pilot', source_id: 'pilot:1',
      instructions: [ { instruction: 'Phrase as a cheerful announcement.', axes: ['tone'] } ] },
  ] },
  { tier: 'hard', model: 'sonnet', model_id: 'claude-sonnet-4-6', sources: [
    { ref: 'S1', source: 'Most birds migrate in response to changes in day length rather than to changes in temperature.',
      domain: 'pilot', source_id: 'pilot:0',
      instructions: [ { instruction: 'Render as a chiasmus where the second half inverts the first.', axes: ['structural'] } ] },
  ] },
]
// ========================== END EDIT PER RUN ================================

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['triples'],
  properties: {
    triples: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['source_ref', 'source', 'instruction', 'axes', 'target'],
      properties: {
        source_ref:  { type: 'string' },
        source:      { type: 'string' },
        instruction: { type: 'string' },
        axes:        { type: 'array', items: { type: 'string' } },
        target:      { type: 'string' },
      },
    }},
  },
}

function buildPrompt(batch) {
  const blocks = batch.sources.map((s) => {
    const list = s.instructions
      .map((it, i) => `${i + 1}. ${it.instruction}  (axes: ${it.axes.join(', ')})`).join('\n')
    return `=== SOURCE ${s.ref} ===\n"""${s.source}"""\nInstructions for ${s.ref}:\n${list}`
  }).join('\n\n')
  return `You are a careful rewriting teacher generating training data for a paraphrase model.

You are given SEVERAL sources, each with its own rewrite instructions. For EACH
(source, instruction) pair, produce one rewrite that genuinely BECOMES the form the
instruction asks for (a real chiasmus, an actual weather report, exact word counts when
demanded) while preserving that source's meaning and facts (add no new claims, drop
nothing essential). Keep each source's rewrites tied to THAT source only.

${blocks}

Return one triple per (source, instruction). In each triple, set "source_ref" to the
matching source label (e.g. "S1"), and echo the source text and the instruction's axes verbatim.`
}

phase('Generate')
const nSources = BATCHES.reduce((a, b) => a + b.sources.length, 0)
log(`Generating ${BATCHES.length} batched subagents over ${nSources} sources`)

const results = await parallel(
  BATCHES.map((batch) => () =>
    agent(buildPrompt(batch), {
      label: `${batch.tier}:batch[${batch.sources.length}]`,
      phase: 'Generate',
      model: batch.model,
      schema: SCHEMA,
    })
  )
)

// Map each returned triple back to its source via source_ref; flag mismatches.
// Normalize quotes/whitespace/case before comparing: models often echo the source
// with straight<->curly or single<->double quote swaps, which are not mis-mappings.
// Compare a 60-char normalized prefix so genuine cross-contamination still stands out.
const normSrc = (s) => (s || '')
  .replace(/[‘’“”'"`]/g, '"')
  .replace(/\s+/g, ' ')
  .trim().toLowerCase().slice(0, 60)
let mismapped = 0
const triples = results.flatMap((r, bi) => {
  const batch = BATCHES[bi]
  const byRef = Object.fromEntries(batch.sources.map((s) => [s.ref, s]))
  const rows = (r && r.triples) ? r.triples : []
  return rows.map((t) => {
    const src = byRef[t.source_ref]
    const ok = src && t.source && normSrc(src.source) === normSrc(t.source)
    if (!ok) mismapped++
    const m = src || batch.sources[0]
    return {
      source: t.source,
      instruction: t.instruction,
      axes: t.axes,
      target: t.target,
      domain: m.domain,
      source_id: m.source_id,
      tier: batch.tier,
      gen_model: batch.model_id,
      gen_backend: RUN.gen_backend,
      prompt_version: RUN.prompt_version,
      run_id: RUN.run_id,
      gen_timestamp: RUN.gen_timestamp,
      mapping_ok: !!ok,
    }
  })
})

const failed = results.filter((r) => !r).length
log(`Collected ${triples.length} triples; ${mismapped} mis-mapped; ${failed} subagents empty`)
return { run_id: RUN.run_id, n_subagents: BATCHES.length, count: triples.length, mismapped, failed, triples }
