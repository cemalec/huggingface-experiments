#!/usr/bin/env python3
"""Emit two throwaway test workflows to size the token-saving levers before the bulk run.

  test_batch.workflow.js  — packs N sources per subagent (vs 1 today) to measure the
                            token drop from amortizing per-subagent overhead, and to
                            check whether multiple sources in one prompt cause the
                            model to mis-map rewrites to the wrong source.
  test_haiku_ab.workflow.js — runs the SAME hard-tier jobs through BOTH Haiku and
                            Sonnet, then a Sonnet judge rates whether each rewrite
                            genuinely realizes the (structural/voice/combinatorial)
                            instruction. Tells us if hard->Sonnet routing is necessary.

Both inline their JOBS (the Workflow sandbox has no FS access) and inline timestamps
(the sandbox statically rejects Date.now/Math.random/new Date). Draws fresh seeds
(NOT chunk_001's 0-319) spread across all five domains.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "data" / "run_manifest.jsonl"
OUT_DIR = HERE / "data" / "runs" / "tests"

# Mixed-domain seed picks (avoid chunk_001's wikipedia 0-319).
BATCH_SEEDS = [500, 1500, 2500, 3500,   # batch 1: wiki/news/arxiv/consumer
               501, 1501, 2501, 4500,   # batch 2: + dialogue
               502, 1502, 3501, 4501,   # batch 3
               503, 2502, 3502, 4502]   # batch 4
BATCH_SIZE = 4                          # sources per subagent

AB_SEEDS = [600, 601, 602, 603,         # wikipedia
            1600, 1601, 1602, 1603,     # news
            2600, 2601, 2602, 2603,     # arxiv
            3600, 3601, 3602, 3603,     # consumer
            4600, 4601, 4602, 4603]     # dialogue


def load_manifest():
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l]
    by_seed: dict[int, dict] = {}
    for r in rows:
        by_seed.setdefault(r["seed_idx"], {})[r["tier"]] = r
    return by_seed


def emit_batch(by_seed, ts_compact, ts_iso) -> str:
    # Build batches: for each tier, group the picked seeds into BATCH_SIZE-source subagents.
    batches = []
    for tier in ("easy", "hard"):
        seeds = [by_seed[s][tier] for s in BATCH_SEEDS]
        for i in range(0, len(seeds), BATCH_SIZE):
            group = seeds[i:i + BATCH_SIZE]
            sources = []
            for k, job in enumerate(group, 1):
                sources.append({
                    "ref": f"S{k}",
                    "source": job["source"],
                    "domain": job["domain"],
                    "source_id": job["source_id"],
                    "instructions": job["instructions"],
                })
            batches.append({
                "tier": tier,
                "model": group[0]["model"],
                "model_id": group[0]["model_id"],
                "sources": sources,
            })
    return TEMPLATE_BATCH % {
        "run_id": f"test-batch-{ts_compact}",
        "ts": ts_iso,
        "batch_size": BATCH_SIZE,
        "jobs": json.dumps(batches, ensure_ascii=False, indent=0),
    }


def emit_ab(by_seed, ts_compact, ts_iso) -> str:
    jobs = []
    for s in AB_SEEDS:
        h = by_seed[s]["hard"]
        jobs.append({
            "seed_idx": s,
            "source": h["source"],
            "domain": h["domain"],
            "source_id": h["source_id"],
            "instructions": h["instructions"],
        })
    return TEMPLATE_AB % {
        "run_id": f"test-haiku-ab-{ts_compact}",
        "ts": ts_iso,
        "jobs": json.dumps(jobs, ensure_ascii=False, indent=0),
    }


TEMPLATE_BATCH = r'''/* TEST: batch N sources per subagent. Throwaway — measures token drop + source-mapping fidelity. */
export const meta = {
  name: 'test-batch-sources',
  description: 'Test packing %(batch_size)s sources per subagent to amortize per-subagent overhead',
  phases: [{ title: 'Generate' }],
}

const RUN = {
  run_id: "%(run_id)s",
  gen_timestamp: "%(ts)s",
  gen_backend: 'claude-code-workflow',
  prompt_version: 'v1-batch',
}

const BATCHES = %(jobs)s

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
log(`Generating ${BATCHES.length} batched subagents (${%(batch_size)s} sources each)`)

const results = await parallel(
  BATCHES.map((batch) => () =>
    agent(buildPrompt(batch), {
      label: `${batch.tier}:batch-${batch.sources.map(s => s.domain).join('/')}`,
      phase: 'Generate',
      model: batch.model,
      schema: SCHEMA,
    })
  )
)

// Map each returned triple back to its source via source_ref; flag mismatches.
let mismapped = 0
const triples = results.flatMap((r, bi) => {
  const batch = BATCHES[bi]
  const byRef = Object.fromEntries(batch.sources.map(s => [s.ref, s]))
  const rows = (r && r.triples) ? r.triples : []
  return rows.map((t) => {
    const src = byRef[t.source_ref]
    // mismatch check: echoed source text should match the referenced source
    const ok = src && t.source && src.source.slice(0, 40) === t.source.slice(0, 40)
    if (!ok) mismapped++
    const meta = src || batch.sources[0]
    return {
      source_ref: t.source_ref, mapping_ok: !!ok,
      source: t.source, instruction: t.instruction, axes: t.axes, target: t.target,
      domain: meta.domain, source_id: meta.source_id,
      tier: batch.tier, gen_model: batch.model_id, gen_backend: RUN.gen_backend,
      prompt_version: RUN.prompt_version, run_id: RUN.run_id, gen_timestamp: RUN.gen_timestamp,
    }
  })
})

const failed = results.filter((r) => !r).length
log(`Collected ${triples.length} triples; ${mismapped} mis-mapped to wrong source; ${failed} subagents empty`)
return { run_id: RUN.run_id, n_subagents: BATCHES.length, count: triples.length, mismapped, failed, triples }
'''


TEMPLATE_AB = r'''/* TEST: Haiku vs Sonnet on HARD axes, with a Sonnet judge. Throwaway. */
export const meta = {
  name: 'test-haiku-ab',
  description: 'A/B hard-tier rewrites: Haiku vs Sonnet, judged for genuine style realization',
  phases: [{ title: 'Generate' }, { title: 'Judge' }],
}

const RUN = {
  run_id: "%(run_id)s",
  gen_timestamp: "%(ts)s",
}

const JOBS = %(jobs)s

const GEN_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['rewrites'],
  properties: {
    rewrites: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['instruction', 'target'],
      properties: { instruction: { type: 'string' }, target: { type: 'string' } },
    }},
  },
}

const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['verdicts'],
  properties: {
    verdicts: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['instruction', 'a_ok', 'b_ok', 'better', 'reason'],
      properties: {
        instruction: { type: 'string' },
        a_ok: { type: 'boolean' },          // candidate A (Haiku) genuinely realizes the instruction
        b_ok: { type: 'boolean' },          // candidate B (Sonnet) genuinely realizes the instruction
        better: { type: 'string', enum: ['A', 'B', 'tie'] },
        reason: { type: 'string' },
      },
    }},
  },
}

function genPrompt(job) {
  const list = job.instructions
    .map((it, i) => `${i + 1}. ${it.instruction}  (axes: ${it.axes.join(', ')})`).join('\n')
  return `You are a careful rewriting teacher. Rewrite the SOURCE under each instruction so it
genuinely BECOMES the form requested (a real chiasmus with ABBA inversion, an actual
maximalist-essayist voice, etc.) while preserving the source's meaning and facts.

SOURCE:
"""${job.source}"""

INSTRUCTIONS:
${list}

Return one rewrite per instruction; echo the instruction verbatim.`
}

function judgePrompt(job, aRewrites, bRewrites) {
  const byInstr = (arr) => Object.fromEntries((arr || []).map(x => [x.instruction, x.target]))
  const A = byInstr(aRewrites), B = byInstr(bRewrites)
  const blocks = job.instructions.map((it, i) => {
    // align by index as primary, instruction-text as fallback
    const a = (aRewrites[i] && aRewrites[i].target) || A[it.instruction] || '(missing)'
    const b = (bRewrites[i] && bRewrites[i].target) || B[it.instruction] || '(missing)'
    return `${i + 1}. INSTRUCTION: ${it.instruction}  (axes: ${it.axes.join(', ')})\n   A: ${a}\n   B: ${b}`
  }).join('\n\n')
  return `You are a strict evaluator of style-transfer rewrites. Two anonymous models (A and B)
each rewrote the same SOURCE under each instruction. For each instruction decide:
- a_ok: does candidate A GENUINELY become the form the instruction asks for (not just a
  reworded sentence) AND keep the source's meaning? Be strict on structural/voice axes —
  a chiasmus must actually invert; a named voice must actually sound like it.
- b_ok: same judgement for candidate B.
- better: which is the stronger realization (A, B, or tie).
- reason: one short clause.

SOURCE:
"""${job.source}"""

${blocks}

Return one verdict per instruction, echoing the instruction verbatim.`
}

phase('Generate')
log(`A/B over ${JOBS.length} hard jobs x {haiku, sonnet}`)

const gen = await parallel(
  JOBS.map((job) => () =>
    Promise.all([
      agent(genPrompt(job), { label: `haiku:${job.source_id}`, phase: 'Generate', model: 'haiku', schema: GEN_SCHEMA }),
      agent(genPrompt(job), { label: `sonnet:${job.source_id}`, phase: 'Generate', model: 'sonnet', schema: GEN_SCHEMA }),
    ])
  )
)

phase('Judge')
const verdicts = await parallel(
  JOBS.map((job, ji) => () => {
    const [a, b] = gen[ji] || [null, null]
    const aR = (a && a.rewrites) || [], bR = (b && b.rewrites) || []
    if (aR.length !== job.instructions.length || bR.length !== job.instructions.length) {
      return Promise.resolve({ skipped: true, source_id: job.source_id, aLen: aR.length, bLen: bR.length })
    }
    return agent(judgePrompt(job, aR, bR), {
      label: `judge:${job.source_id}`, phase: 'Judge', model: 'sonnet', schema: JUDGE_SCHEMA,
    }).then((v) => ({
      source_id: job.source_id, domain: job.domain,
      verdicts: (v && v.verdicts) || [],
      axes: job.instructions.map(it => it.axes),
    }))
  })
)

// Tally
let haikuOk = 0, sonnetOk = 0, n = 0, betterA = 0, betterB = 0, betterTie = 0, skipped = 0
for (const j of verdicts) {
  if (!j || j.skipped) { skipped++; continue }
  for (const v of j.verdicts) {
    n++
    if (v.a_ok) haikuOk++
    if (v.b_ok) sonnetOk++
    if (v.better === 'A') betterA++
    else if (v.better === 'B') betterB++
    else betterTie++
  }
}
log(`Judged ${n} hard instructions: Haiku ok=${haikuOk}, Sonnet ok=${sonnetOk}; better A/B/tie=${betterA}/${betterB}/${betterTie}; skipped jobs=${skipped}`)
return {
  run_id: RUN.run_id, n_instructions: n,
  haiku_ok: haikuOk, sonnet_ok: sonnetOk,
  better_haiku: betterA, better_sonnet: betterB, better_tie: betterTie,
  skipped, verdicts,
}
'''


def main() -> None:
    by_seed = load_manifest()
    ts_compact = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    batch_js = emit_batch(by_seed, ts_compact, ts_iso)
    ab_js = emit_ab(by_seed, ts_compact, ts_iso)
    (OUT_DIR / "test_batch.workflow.js").write_text(batch_js)
    (OUT_DIR / "test_haiku_ab.workflow.js").write_text(ab_js)

    for name in ("test_batch.workflow.js", "test_haiku_ab.workflow.js"):
        p = OUT_DIR / name
        print(f"  {p}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
