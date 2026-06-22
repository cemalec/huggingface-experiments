/*
 * Stage B name-variant generation — BATCHED Claude Code Workflow (teacher = Claude, no API key).
 * ============================================================================
 *
 * WHAT THIS IS
 *   The name-matching analog of the paraphraser's generate_triples_batched_workflow.js.
 *   Each subagent ("batch") handles SEVERAL canonical names of the same tier. For each
 *   (name, condition) pair it emits one VARIANT name that exhibits that mismatch, plus
 *   the match/non-match label the condition implies. Batching amortizes the fixed
 *   per-subagent overhead across many more pairs (see triples.runs.md note in the
 *   paraphraser; same economics here).
 *
 *   Each emitted row self-identifies its source via "source_ref" (S1, S2, ...), so
 *   variants map back to the right canonical name even with many names in one prompt.
 *   The script logs `mismapped` (echoed source_name != expected) so regressions surface.
 *
 * THE "TRIPLE"  (kept the paraphraser's array name for parallelism)
 *   Each element is (source_name, condition, variant_name) + label + axes — a labelled
 *   name PAIR with the rule that generated it. Example:
 *     source_name "John L. Smith" + condition "first/last flipped" -> variant "Smith L. John" (match)
 *     source_name "John L. Smith" + "alternate first + letter sub in surname" -> "Johnathan L. Smlth" (match)
 *     source_name "John L. Smith" + "different person, same surname"  -> "Jane R. Smith" (non-match)
 *
 * BULK MODE
 *     1. Build the condition bank:  python generate_conditions.py   -> conditions.json
 *     2. Build name seeds:          python build_seeds.py            -> data/seeds.jsonl
 *     3. Sample conditions:         python sample_conditions.py      -> data/run_manifest.jsonl
 *     4. Emit batched chunks:       python emit_chunks_batched.py --batch-sources 16
 *        -> data/runs/batched/chunk_NNN.workflow.js (each <512KB, the Workflow size cap),
 *           skipping already-generated (source_name, tier).
 *     5. Run each chunk:            Workflow({ scriptPath: "<abs path>/chunk_NNN.workflow.js" })
 *   The BATCHES below is a 2-batch sample kept for hand-debugging.
 *
 * PERSIST + LOG (same flow as the paraphraser)
 *   On completion it RETURNS { run_id, n_subagents, count, mismapped, failed, triples }
 *   and writes nothing (sandbox has no FS). Orchestrator appends result.triples to
 *   data/pairs.claude-code.jsonl and records the run in pairs.runs.md.
 *
 * PROVENANCE (stamped on every row): tier, label, gen_model, gen_backend,
 *   prompt_version, run_id, gen_timestamp.
 *
 * GOTCHAS: the sandbox statically rejects literal `Date.now`/`Math.random`/`new Date`
 *   even in comments (timestamps inlined from Python); pass scriptPath as an ABSOLUTE path.
 */

export const meta = {
  name: 'name-variants-batched',
  description: 'Generate labelled name-match variants, multiple names per subagent (Haiku=easy, Sonnet=hard)',
  phases: [{ title: 'Generate', detail: 'one subagent per (batch-of-names x tier)' }],
}

// ============================ EDIT PER RUN ==================================
// Replaced by emit_chunks_batched.py. BATCHES holds one entry per subagent;
// each entry has a `sources` array (its own condition draws) + tier/model.

const RUN = {
  run_id: 'batched-pilot-PLACEHOLDER',
  gen_timestamp: '2026-06-21T00:00:00Z',
  gen_backend: 'claude-code-workflow',
  prompt_version: 'v1-batch',
}

const BATCHES = [
  { tier: 'easy', model: 'haiku', model_id: 'claude-haiku-4-5', sources: [
    { ref: 'S1', source_name: 'John L. Smith', domain: 'frequent', source_id: 'ssa:John|census:Smith',
      components: { first: 'John', middle: 'L.', last: 'Smith' },
      conditions: [ { condition: "First and last name are swapped (given/family order reversed).", axes: ['order'], label: 'match' },
                    { condition: "First name reduced to its initial.", axes: ['initial'], label: 'match' } ] },
    { ref: 'S2', source_name: 'Mary Johnson', domain: 'frequent', source_id: 'ssa:Mary|census:Johnson',
      components: { first: 'Mary', middle: '', last: 'Johnson' },
      conditions: [ { condition: "Whole name rendered in ALL CAPS.", axes: ['format'], label: 'match' } ] },
  ] },
  { tier: 'hard', model: 'sonnet', model_id: 'claude-sonnet-4-6', sources: [
    { ref: 'S1', source_name: 'John L. Smith', domain: 'frequent', source_id: 'ssa:John|census:Smith',
      components: { first: 'John', middle: 'L.', last: 'Smith' },
      conditions: [ { condition: "Apply two changes at once: replace the first name with an accepted alternate or nickname; and substitute one letter in the surname.", axes: ['nickname', 'typo'], label: 'match' },
                    { condition: "A DIFFERENT person who happens to share the same surname (different first name).", axes: ['negative'], label: 'non-match' } ] },
  ] },
]
// ========================== END EDIT PER RUN ================================

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['triples'],
  properties: {
    triples: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['source_ref', 'source_name', 'condition', 'axes', 'label', 'variant_name'],
      properties: {
        source_ref:   { type: 'string' },
        source_name:  { type: 'string' },
        condition:    { type: 'string' },
        axes:         { type: 'array', items: { type: 'string' } },
        label:        { type: 'string', enum: ['match', 'non-match'] },
        variant_name: { type: 'string' },
      },
    }},
  },
}

function buildPrompt(batch) {
  const blocks = batch.sources.map((s) => {
    const list = s.conditions
      .map((c, i) => `${i + 1}. [${c.label}] ${c.condition}  (axes: ${c.axes.join(', ')})`).join('\n')
    return `=== NAME ${s.ref} ===\n"""${s.source_name}"""\nConditions for ${s.ref}:\n${list}`
  }).join('\n\n')
  return `You are building training data for a person-name MATCHING model. The model must
learn to recognize when two differently-written names refer to the same individual,
and when they do NOT.

You are given SEVERAL canonical names, each with its own list of mismatch conditions.
For EACH (name, condition) pair, produce ONE variant name that realistically exhibits
exactly that condition:

- If the condition's label is "match", the variant MUST still refer to the SAME person
  — apply only the surface change described (reorder, initialize, nickname, typo,
  transliterate, etc.). Keep it realistic, as if it came from a real record or OCR.
- If the label is "non-match", produce a DIFFERENT but confusably-similar person's name
  as the condition describes (e.g. same surname, different first name). It must be
  genuinely a different person, not the same one rewritten.

Make exactly the change(s) the condition names and no others; do not also "fix" or
otherwise alter unrelated parts of the name. Keep each name's variants tied to THAT name.

${blocks}

Return one triple per (name, condition). In each triple set "source_ref" to the matching
name label (e.g. "S1"), echo the source_name and the condition's axes and label verbatim,
and put the produced variant in "variant_name".`
}

phase('Generate')
const nSources = BATCHES.reduce((a, b) => a + b.sources.length, 0)
log(`Generating ${BATCHES.length} batched subagents over ${nSources} names`)

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

// Map each returned row back to its source via source_ref; flag mismatches.
// Normalize quotes/whitespace/case before comparing (models echo with quote/case
// swaps that are not mis-mappings); compare a 60-char normalized prefix.
const normName = (s) => (s || '')
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
    const ok = src && t.source_name && normName(src.source_name) === normName(t.source_name)
    if (!ok) mismapped++
    const m = src || batch.sources[0]
    return {
      source_name: t.source_name,
      condition: t.condition,
      axes: t.axes,
      label: t.label,
      variant_name: t.variant_name,
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
log(`Collected ${triples.length} name pairs; ${mismapped} mis-mapped; ${failed} subagents empty`)
return { run_id: RUN.run_id, n_subagents: BATCHES.length, count: triples.length, mismapped, failed, triples }
