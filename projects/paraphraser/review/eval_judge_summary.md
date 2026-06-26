# Holdout Evaluation — LLM-as-Judge Summary

**Eval set**: 50 examples with instructions not present in the 250-instruction training bank.
Multi-axis examples are counted in each of their axes.
Scores are Opus-assigned on 0–5 scales (adherence / faithfulness / fluency).

## Per-axis mean scores

| Axis | N | E1 Adh | E1 Faith | E1 Flu | E3 Adh | E3 Faith | E3 Flu | Δ Adh | Δ Faith | Δ Flu |
|------|---|--------|----------|--------|--------|----------|--------|-------|---------|-------|
| register | 18 | 4.00 | 3.39 | 4.72 | 4.06 | 3.28 | 4.56 | +0.06 | -0.11 | -0.17 |
| audience | 18 | 3.89 | 3.56 | 4.72 | 3.61 | 3.33 | 4.33 | -0.28 | -0.22 | -0.39 |
| tone | 22 | 3.45 | 3.27 | 4.55 | 4.05 | 3.50 | 4.50 | +0.59 | +0.23 | -0.05 |
| length | 14 | 4.14 | 4.00 | 4.57 | 4.07 | 3.64 | 4.36 | -0.07 | -0.36 | -0.21 |
| genre | 28 | 3.71 | 3.18 | 4.71 | 3.96 | 3.50 | 4.50 | +0.25 | +0.32 | -0.21 |
| structural | 12 | 2.75 | 3.50 | 4.75 | 2.67 | 3.83 | 4.58 | -0.08 | +0.33 | -0.17 |
| voice | 18 | 4.00 | 3.00 | 4.78 | 4.06 | 3.39 | 4.50 | +0.06 | +0.39 | -0.28 |
| **overall** | 100 | 3.65 | 3.45 | 4.68 | 3.84 | 3.44 | 4.49 | +0.19 | -0.01 | -0.19 |

**Epoch-1 mean composite** (sum of 3 dims): 11.78 / 15
**Epoch-3 mean composite**: 11.77 / 15  (Δ -0.01)

**Win/Tie/Loss (epoch-3 vs epoch-1, by composite)**: 41 W / 18 T / 41 L

## Per-example scores

| # | Axes | E1 Adh | E1 Faith | E1 Flu | E3 Adh | E3 Faith | E3 Flu | Winner |
|---|------|--------|----------|--------|--------|----------|--------|--------|
| 1 | register | 4 | 4 | 3 | 4 | 4 | 4 | E3 |
| 1 | register | 5 | 3 | 5 | 5 | 2 | 5 | E1 |
| 2 | register | 4 | 4 | 5 | 3 | 2 | 5 | E1 |
| 2 | register | 3 | 4 | 4 | 4 | 3 | 4 | tie |
| 3 | register | 3 | 3 | 5 | 5 | 4 | 5 | E3 |
| 3 | register | 5 | 3 | 5 | 4 | 3 | 5 | E1 |
| 4 | register | 3 | 3 | 4 | 5 | 4 | 5 | E3 |
| 4 | register | 5 | 4 | 5 | 5 | 2 | 5 | E1 |
| 5 | register | 4 | 3 | 5 | 3 | 3 | 3 | E1 |
| 5 | register | 2 | 5 | 5 | 4 | 4 | 5 | E3 |
| 6 | audience | 4 | 3 | 5 | 4 | 3 | 5 | tie |
| 6 | audience | 5 | 4 | 5 | 4 | 2 | 3 | E1 |
| 7 | audience | 4 | 4 | 5 | 3 | 3 | 2 | E1 |
| 7 | audience | 3 | 3 | 5 | 5 | 4 | 5 | E3 |
| 8 | audience | 3 | 5 | 5 | 3 | 3 | 5 | E1 |
| 8 | audience | 4 | 2 | 5 | 2 | 5 | 5 | E3 |
| 9 | audience | 5 | 5 | 5 | 4 | 5 | 5 | E1 |
| 9 | audience | 3 | 4 | 5 | 3 | 3 | 4 | E1 |
| 10 | audience | 4 | 4 | 5 | 4 | 2 | 5 | E1 |
| 10 | audience | 3 | 3 | 4 | 4 | 2 | 4 | tie |
| 11 | tone | 4 | 3 | 5 | 4 | 5 | 5 | E3 |
| 11 | tone | 2 | 3 | 4 | 4 | 3 | 4 | E3 |
| 12 | tone | 2 | 4 | 3 | 5 | 3 | 5 | E3 |
| 12 | tone | 3 | 3 | 5 | 4 | 3 | 5 | E3 |
| 13 | tone | 4 | 5 | 5 | 5 | 5 | 4 | tie |
| 13 | tone | 4 | 3 | 5 | 4 | 4 | 5 | E3 |
| 14 | tone | 2 | 2 | 4 | 5 | 4 | 5 | E3 |
| 14 | tone | 4 | 5 | 4 | 5 | 3 | 4 | E1 |
| 15 | tone | 2 | 5 | 5 | 4 | 3 | 5 | tie |
| 15 | tone | 3 | 5 | 4 | 3 | 2 | 4 | E1 |
| 16 | length | 5 | 5 | 5 | 5 | 4 | 5 | E1 |
| 16 | length | 5 | 5 | 3 | 5 | 4 | 3 | E1 |
| 17 | length | 4 | 4 | 5 | 3 | 2 | 5 | E1 |
| 17 | length | 4 | 5 | 5 | 5 | 4 | 5 | tie |
| 18 | length | 5 | 4 | 5 | 5 | 4 | 5 | tie |
| 18 | length | 2 | 5 | 5 | 4 | 5 | 5 | E3 |
| 19 | length | 5 | 3 | 4 | 3 | 2 | 4 | E1 |
| 19 | length | 4 | 3 | 5 | 5 | 2 | 3 | E1 |
| 20 | length | 3 | 3 | 3 | 4 | 4 | 5 | E3 |
| 20 | length | 4 | 4 | 5 | 4 | 4 | 5 | tie |
| 21 | genre | 3 | 2 | 5 | 4 | 4 | 5 | E3 |
| 21 | genre | 3 | 4 | 5 | 3 | 4 | 5 | tie |
| 22 | genre | 5 | 3 | 5 | 5 | 3 | 5 | tie |
| 22 | genre | 5 | 4 | 5 | 3 | 3 | 4 | E1 |
| 23 | genre | 3 | 4 | 5 | 5 | 4 | 4 | E3 |
| 23 | genre | 2 | 1 | 4 | 4 | 3 | 4 | E3 |
| 24 | genre | 4 | 4 | 5 | 3 | 3 | 3 | E1 |
| 24 | genre | 4 | 3 | 5 | 4 | 1 | 5 | E1 |
| 25 | genre | 4 | 4 | 5 | 5 | 5 | 5 | E3 |
| 25 | genre | 3 | 2 | 4 | 4 | 4 | 4 | E3 |
| 26 | structural | 3 | 4 | 5 | 4 | 4 | 5 | E3 |
| 26 | structural | 4 | 3 | 5 | 2 | 4 | 4 | E1 |
| 27 | structural | 2 | 4 | 5 | 2 | 4 | 5 | tie |
| 27 | structural | 1 | 5 | 5 | 1 | 5 | 5 | tie |
| 28 | structural | 5 | 4 | 5 | 5 | 3 | 5 | E1 |
| 28 | structural | 3 | 2 | 5 | 3 | 2 | 5 | tie |
| 29 | structural | 2 | 4 | 5 | 4 | 3 | 4 | tie |
| 29 | structural | 4 | 3 | 4 | 2 | 5 | 5 | E3 |
| 30 | structural | 2 | 3 | 4 | 2 | 2 | 3 | E1 |
| 30 | structural | 1 | 5 | 5 | 0 | 5 | 5 | E1 |
| 31 | voice | 4 | 2 | 5 | 5 | 3 | 5 | E3 |
| 31 | voice | 4 | 3 | 4 | 4 | 4 | 4 | E3 |
| 32 | voice | 3 | 3 | 4 | 4 | 4 | 4 | E3 |
| 32 | voice | 3 | 5 | 5 | 5 | 5 | 5 | E3 |
| 33 | voice | 4 | 2 | 5 | 3 | 2 | 4 | E1 |
| 33 | voice | 4 | 2 | 5 | 4 | 3 | 5 | E3 |
| 34 | voice | 5 | 4 | 5 | 4 | 3 | 5 | E1 |
| 34 | voice | 4 | 4 | 5 | 3 | 2 | 5 | E1 |
| 35 | voice | 1 | 3 | 5 | 5 | 4 | 5 | E3 |
| 35 | voice | 3 | 4 | 4 | 2 | 2 | 4 | E1 |
| 36 | genre, audience | 4 | 4 | 4 | 4 | 2 | 5 | E1 |
| 36 | genre, audience | 4 | 4 | 5 | 4 | 2 | 5 | E1 |
| 37 | register, tone | 5 | 2 | 5 | 3 | 3 | 4 | E1 |
| 37 | register, tone | 4 | 2 | 4 | 4 | 2 | 4 | tie |
| 38 | genre, voice | 5 | 3 | 5 | 5 | 4 | 4 | tie |
| 38 | genre, voice | 4 | 1 | 5 | 5 | 4 | 5 | E3 |
| 39 | audience, tone | 3 | 2 | 5 | 4 | 4 | 5 | E3 |
| 39 | audience, tone | 4 | 2 | 5 | 4 | 5 | 5 | E3 |
| 40 | length, audience | 4 | 3 | 4 | 4 | 4 | 5 | E3 |
| 40 | length, audience | 5 | 4 | 5 | 3 | 3 | 3 | E1 |
| 41 | register, voice | 5 | 4 | 5 | 5 | 4 | 5 | tie |
| 41 | register, voice | 5 | 3 | 5 | 4 | 2 | 5 | E1 |
| 42 | genre, tone | 3 | 4 | 4 | 5 | 2 | 5 | E3 |
| 42 | genre, tone | 3 | 3 | 5 | 3 | 2 | 5 | E1 |
| 43 | audience, genre | 4 | 4 | 3 | 4 | 4 | 4 | E3 |
| 43 | audience, genre | 4 | 4 | 5 | 2 | 4 | 3 | E1 |
| 44 | genre, tone | 4 | 5 | 5 | 4 | 5 | 4 | E1 |
| 44 | genre, tone | 4 | 3 | 5 | 3 | 3 | 4 | E1 |
| 45 | genre, register | 3 | 2 | 5 | 3 | 3 | 5 | E3 |
| 45 | genre, register | 4 | 4 | 5 | 5 | 5 | 5 | E3 |
| 46 | voice, tone | 5 | 3 | 5 | 3 | 4 | 3 | E1 |
| 46 | voice, tone | 4 | 2 | 4 | 4 | 4 | 4 | E3 |
| 47 | genre, voice | 5 | 4 | 5 | 4 | 3 | 4 | E1 |
| 47 | genre, voice | 4 | 2 | 5 | 4 | 4 | 5 | E3 |
| 48 | genre, structural | 2 | 2 | 4 | 2 | 4 | 4 | E3 |
| 48 | genre, structural | 4 | 3 | 5 | 5 | 5 | 5 | E3 |
| 49 | genre, tone | 4 | 4 | 5 | 5 | 4 | 5 | E3 |
| 49 | genre, tone | 3 | 2 | 4 | 4 | 4 | 5 | E3 |
| 50 | register, length | 4 | 3 | 5 | 4 | 4 | 4 | tie |
| 50 | register, length | 4 | 5 | 5 | 3 | 5 | 4 | E1 |

## Judge notes

**1. Render as a line from a corporate earnings call transcript — measured, forward-looking.**
*Source: The story was a good one, maybe a bit farfetched...or was it?*
- E1 (4/4/3): Convincing earnings-call register and clear forward-looking close, but the phrase "continue informing our ongoing process understanding" is wordy and slightly stilted.
- E3 (4/4/4): Tighter, more natural corporate phrasing ("data point we need for informed decision-making moving forward") that nails the measured, forward-looking tone with minor added framing.

**1. Render as a line from a corporate earnings call transcript — measured, forward-looking.**
*Source: The story was a good one, maybe a bit farfetched...or was it?*
- E1 (5/3/5): Nails the measured, forward-looking earnings-call register and reasonably maps the "farfetched/or was it?" doubt onto "elements may warrant further review."
- E3 (5/2/5): Equally polished earnings-call voice, but invents "scalability concerns" that do not correspond to the source's farfetchedness, distorting the meaning.

**2. Rewrite in the clipped, imperative register of a field operations checklist.**
*Source: New signing: Ferdinand poses with Harry Redknapp and his new No 5 shirt at QPR on Monday.*
- E1 (4/4/5): Strong clipped checklist register with a genuine imperative ("Verify presence") and Perthshire retained; only minor loss of the "abounds" intensity.
- E3 (3/2/5): Clipped tone but reads as a passive status report rather than imperative, and it drops the Perthshire location entirely.

**2. Rewrite in the clipped, imperative register of a field operations checklist.**
*Source: New signing: Ferdinand poses with Harry Redknapp and his new No 5 shirt at QPR on Monday.*
- E1 (3/4/4): Clipped and label-driven with a "Status: signed" tag, but reads as a flat entity list rather than a structured checklist and uses no imperative phrasing.
- E3 (4/3/4): Convincingly renders the checklist register with item-by-item confirmed/assigned lines, but reinterprets "poses with" into arrival/positioning, a minor meaning distortion.

**3. Translate into the careful, qualification-heavy register of a scientific abstract.**
*Source: I know this book is fiction, but come on - do black women and men need to be portrayed as stupid for the sake of making money?*
- E1 (3/3/5): Formal abstract register and fluent, but the closing clause asserts ("fundamental boundary condition") rather than qualifies, missing the instruction's qualification-heavy demand and adding an uncalled-for characterization.
- E3 (5/4/5): Nails the qualification-heavy register with "holds under standard assumptions regarding thermodynamic equilibrium and gravitational collapse," a genuine hedge; only minor added scope conditions keep it just shy of perfect faithfulness.

**3. Translate into the careful, qualification-heavy register of a scientific abstract.**
*Source: I know this book is fiction, but come on - do black women and men need to be portrayed as stupid for the sake of making money?*
- E1 (5/3/5): Excellent qualification-heavy abstract register ("may inadvertently," "remain controversial"), but drops the profit motive and adds extraneous concepts (social equity, public perception).
- E3 (4/3/5): Clean, polished abstract phrasing but less hedged than the "qualification-heavy" instruction demands, and omits both the money motive and the gender specificity.

**4. Render in the breezy, first-person register of a lifestyle blog post.**
*Source: Unfortunatelly, didn't have a chance to listen to original version (this one is remastered version).*
- E1 (3/3/4): Casual first-person voice is present, but the dismissive "who cares" framing and the pivot to vague "how it lives its life" drift from both the lifestyle-blog warmth and the source's factual content.
- E3 (5/4/5): Nails the breezy lifestyle-blog register and retains both species name and family, with only a light "insect pedigree" embellishment.

**4. Render in the breezy, first-person register of a lifestyle blog post.**
*Source: Unfortunatelly, didn't have a chance to listen to original version (this one is remastered version).*
- E1 (5/4/5): Nails the breezy first-person lifestyle voice while preserving the core point (original unavailable, got the remaster) with only mild embellishment.
- E3 (5/2/5): Strong lifestyle register and fluent prose, but hallucinates that the original is freely available "if you hunt around," distorting the source meaning.

**5. Recast in the terse, timestamped register of a system log entry.**
*Source: Moreover, we calculate the values of topological free entropy dimensions of families of generators of some unital C$^*$ algebras (for example: irrational rotation C$^*$ algebras or minimal tensor product of two reduced C$^*$ algebras of free groups).*
- E1 (4/3/5): Terse and convincingly log-like with date/timestamp and parenthetical kickoffs, but invents a date and match times not in the source and carries the spurious "TURNOVER" label.
- E3 (3/3/3): Has a timestamp but undercuts the "terse" constraint with sentence-like padding, adds an unsupported "schedule confirmed" status, and reads awkwardly ("schedule...scheduled").

**5. Recast in the terse, timestamped register of a system log entry.**
*Source: Moreover, we calculate the values of topological free entropy dimensions of families of generators of some unital C$^*$ algebras (for example: irrational rotation C$^*$ algebras or minimal tensor product of two reduced C$^*$ algebras of free groups).*
- E1 (2/5/5): Log-style labels evoke the register and meaning is fully preserved, but it omits the explicitly required timestamp.
- E3 (4/4/5): Satisfies the timestamped log register ("Completed at 14:37.") and is terser, with only minor loss of source qualifiers ("unital", "minimal", "two").

**6. Explain as if to a bright ten-year-old encountering this idea for the first time.**
*Source: The Monarch could be confronted with divided loyalties as head of state for both the remainder of the UK and a breakaway nation north of the border.*
- E1 (4/3/5): Warm, age-appropriate register with a relatable analogy and retains the four-or-six count, but conflates the teeth with the calli and drops the distinct labellum/teeth-on-the-sides detail.
- E3 (4/3/5): Correctly separates teeth from bumps and names the labellum, but loses the specific four-or-six count and misplaces the base as "where the stem meets the leaf."

**6. Explain as if to a bright ten-year-old encountering this idea for the first time.**
*Source: The Monarch could be confronted with divided loyalties as head of state for both the remainder of the UK and a breakaway nation north of the border.*
- E1 (5/4/5): Clean, vivid child-level analogy that captures the divided-loyalty idea; drops the explicit Monarch/head-of-state framing but meaning is intact.
- E3 (4/2/3): Good child framing but introduces a self-contradiction ("the Queen ... becomes king") and an invented "special powers over Scotland" detail not in the source.

**7. Write for a deeply skeptical reader who has heard promises before and wants proof.**
*Source: The S-factors from the two methods do not show any discrepancy within the experimental errors.*
- E1 (4/4/5): The "measured rather than assumed" framing directly addresses a proof-wanting skeptic and the prose is clean and faithful, with only a mild expansion.
- E3 (3/3/2): Has a subject-verb agreement error ("people... demonstrates"), an incoherent "more than any statistic you've ever seen" addition that even undercuts the proof-seeking angle, and several unsupported additions.

**7. Write for a deeply skeptical reader who has heard promises before and wants proof.**
*Source: The S-factors from the two methods do not show any discrepancy within the experimental errors.*
- E1 (3/3/5): Good skeptic framing ("no cherry-picking"), but drops the "within experimental errors" qualifier and adds an unsupported "given how they were constructed" rationale.
- E3 (5/4/5): Strong skeptic persona ("That's why I'm still here") and correctly retains the measurement-uncertainty qualifier; "match up exactly" only mildly overstates "no discrepancy."

**8. Render for a domain expert who needs only the key claim — no scaffolding, no examples.**
*Source: Picking up right from Halloween 4, Halloween 5 was one of the last '80s slasher films and its got high standards of horror.*
- E1 (3/5/5): Preserves both source claims accurately, but the trailing "in this work" is residual scaffolding the instruction asked to strip.
- E3 (3/3/5): Cleaner and more compressed, but "associated physical properties...in detail" subtly re-attributes the model's physical aspects to the tensor and adds emphasis not in the source.

**8. Render for a domain expert who needs only the key claim — no scaffolding, no examples.**
*Source: Picking up right from Halloween 4, Halloween 5 was one of the last '80s slasher films and its got high standards of horror.*
- E1 (4/2/5): Achieves the terse, single-key-claim expert framing, but "eighth-grade" is a distortion of "one of the last '80s slasher films," damaging faithfulness.
- E3 (2/5/5): Preserves meaning fully and reads naturally, but keeps the "continued from Halloween 4" scaffolding and expands rather than reducing to the key claim.

**9. Write for a reader who is overwhelmed and needs only the single most important thing.**
*Source: I've enjoyed several of her other books but this was just tedious.*
- E1 (5/5/5): The label format ("Saturday hours:") is maximally stripped-down and scannable, ideal for an overwhelmed reader needing only the essential fact.
- E3 (4/5/5): Accurate and concise with the reassuring "That's it," but the conversational "runs from" is marginally less minimal than a bare label for an overwhelmed reader.

**9. Write for a reader who is overwhelmed and needs only the single most important thing.**
*Source: I've enjoyed several of her other books but this was just tedious.*
- E1 (3/4/5): Tightly compressed and leads with the verdict, though it keeps the praise rather than reducing to the single most important point; reads naturally.
- E3 (3/3/4): Similarly terse but the added temporal "before" subtly shifts meaning and reads slightly more awkwardly than E1.

**10. Phrase for a general audience curious enough to click but unwilling to work hard.**
*Source: If the cold spot does originate from structures at modest redshifts, as we suggest, then there is no remaining need for non-Gaussian processes at the last scattering surface of the CMB to explain the cold spot.*
- E1 (4/4/5): Plain, accessible phrasing that accurately conveys the grommet keeping the fan positioned; natural and easy to read.
- E3 (4/2/5): Equally casual and fluent, but "keeps it from spinning off" distorts the source's meaning of positioning into preventing rotation/detachment.

**10. Phrase for a general audience curious enough to click but unwilling to work hard.**
*Source: If the cold spot does originate from structures at modest redshifts, as we suggest, then there is no remaining need for non-Gaussian processes at the last scattering surface of the CMB to explain the cold spot.*
- E1 (3/3/4): Plainly accessible and readable, but flat rather than enticing, and misreads "modest redshifts" as the young universe while reducing the CMB non-Gaussian claim to vague "math tricks."
- E3 (4/2/4): The hedging "maybe... just an old story that doesn't fit" hits the curious-clickable register better, but "early universe" and especially "weird extra-curvature effects" distort the source's modest-redshift structures and non-Gaussian-at-last-scattering claim.

**11. Render with barely concealed excitement that keeps breaking through a composed surface.**
*Source: As the name suggests it is located centrally on the west side of the city.*
- E1 (4/3/5): Restrained excitement reads well via em-dashes and "almost breathtaking," but it drops the potential energy surface and the two dihedral angles, losing key source content.
- E3 (4/5/5): Far more complete on meaning (recovers energy surface and dihedral angles) and vividly excited, though the exclamation tips slightly past "barely concealed" toward overt enthusiasm.

**11. Render with barely concealed excitement that keeps breaking through a composed surface.**
*Source: As the name suggests it is located centrally on the west side of the city.*
- E1 (2/3/4): Composed and analytical but little excitement breaks through; "western edge of the heart, right in the middle" muddles the central-location claim.
- E3 (4/3/4): The "well," hedge plus exclamation and "smack-dab" convincingly stages restraint giving way to excitement, though the central/western-edge tension slightly distorts the source.

**12. Write with the resigned fatalism of someone who has watched this happen many times before.**
*Source: Mary-Kate stars as Chloe, and Ashley as Riley (NOT like So Little Tiime!) and they are heading off to London, to compete as China in the Model United Nations.*
- E1 (2/4/3): Preserves the source's puzzlement at sympathy for stars but barely renders resigned fatalism, and the wordy "after all...after all" repetition is clumsy.
- E3 (5/3/5): Nails the world-weary "seen it all before" fatalism fluently, but shifts the source's non-comprehension into resigned acceptance, slightly altering meaning.

**12. Write with the resigned fatalism of someone who has watched this happen many times before.**
*Source: Mary-Kate stars as Chloe, and Ashley as Riley (NOT like So Little Tiime!) and they are heading off to London, to compete as China in the Model United Nations.*
- E1 (3/3/5): Conveys mild weariness ("not surprising, really") but the fatalism is understated; drops the "So Little Time" aside and flattens "compete as China" to "Model UN."
- E3 (4/3/5): "Same casting choices as always" plus "predictable, really" render the resigned-fatalism persona more convincingly, but it shares the same dropped aside and invented recurrence framing.

**13. Phrase with the measured, conflict-averse diplomacy of someone trying to please everyone.**
*Source: Late rally: Chris Ashton goes over for England's third try in Dunedin.*
- E1 (4/5/5): Strong diplomatic hedging ("While there is considerable...", "we can acknowledge") in clean, polished prose; faithful, though slightly less effusively people-pleasing than the persona invites.
- E3 (5/5/4): Nails the conflict-averse, please-everyone register with layered softeners ("it's worth noting", "if you will", "simply"); preserves all facts but the rambling em-dash aside adds minor awkwardness.

**13. Phrase with the measured, conflict-averse diplomacy of someone trying to please everyone.**
*Source: Late rally: Chris Ashton goes over for England's third try in Dunedin.*
- E1 (4/3/5): Strong conflict-averse hedging ("I understand if you'd prefer not to dwell"), but drops the Dunedin location and England attribution.
- E3 (4/4/5): Retains Dunedin and adds a measured please-everyone gloss, though the tone reads more as positive PR than truly conflict-averse diplomacy.

**14. Rewrite in a tone of dry, bone-tired exasperation.**
*Source: In order to do so we establish a variational principle for the topological pressure of certain noncompact sets.*
- E1 (2/2/4): Reads as mild resignation rather than bone-tired exasperation, and drops the source's central frequent-pregnancy consequence into a vague euphemism.
- E3 (5/4/5): The "apparently... just how you get knocked up all the time" framing nails the dry exasperation while preserving both the Italian-man and frequent-pregnancy core, with only a slight generalization from Italian marriage to marriage.

**14. Rewrite in a tone of dry, bone-tired exasperation.**
*Source: In order to do so we establish a variational principle for the topological pressure of certain noncompact sets.*
- E1 (4/5/4): "Whatever that means exactly" and "here we go again" convey weary exasperation while fully preserving the source's mathematical content.
- E3 (5/3/4): "Fine, it's there now, and yes" nails the bone-tired tone, but "had to write down every single one of them" adds a detail not in the source.

**15. Render with conspiratorial urgency, as if sharing something you're not supposed to know.**
*Source: The sticks that she had were 10 inches long and 1/8 inch thick.*
- E1 (2/5/5): Faithful and fluent, but the tone reads as understated insider-aside rather than the demanded conspiratorial urgency.
- E3 (4/3/5): Convincingly conspiratorial ("nobody wants to talk about openly"), but adds unsupported claims and recasts the apoptosis example as something being "tested against."

**15. Render with conspiratorial urgency, as if sharing something you're not supposed to know.**
*Source: The sticks that she had were 10 inches long and 1/8 inch thick.*
- E1 (3/5/4): Preserves both measurements precisely and gestures at hidden knowledge, but the conspiratorial urgency is muted and somewhat stilted ("something about which I'm told").
- E3 (3/2/4): Reads more naturally and dramatically but distorts the source: drops the 1/8-inch thickness ("thick like butter") and hallucinates a "dangerous" detail.

**16. Compress to the fewest words that preserve the core claim — nothing else.**
*Source: Then we can take a walk and enjoy the scenery in the park.*
- E1 (5/5/5): Maximally compressed telegraphic triplet that still preserves all three core elements including the "strong" qualifier on the meridional flow.
- E3 (5/4/5): Equally tight and fluent, but drops the "strong" magnitude qualifier, a minor loss of the source's emphasis on flow intensity.

**16. Compress to the fewest words that preserve the core claim — nothing else.**
*Source: Then we can take a walk and enjoy the scenery in the park.*
- E1 (5/5/3): Tight 5-word compression that keeps both core claims exactly ("scenery" retained), though the noun fragment "Walks" reads slightly less natural than an imperative.
- E3 (5/4/3): Equally tight compression with a more imperative phrasing, but "views" narrows "scenery" slightly and the dropped article keeps it telegraphic.

**17. Expand until every implied assumption is made fully explicit.**
*Source: Do you think you can tell me where it is?*
- E1 (4/4/5): Faithfully expands the source's implied "morning is cooler" assumption and preserves the actual time-of-day preference; mild faithfulness loss from framing it as "mode of transportation."
- E3 (3/2/5): Polished prose but distorts meaning: drops the "morning" element and invents a running-vs-walking contrast not implied by the source, making explicit an assumption that isn't there.

**17. Expand until every implied assumption is made fully explicit.**
*Source: Do you think you can tell me where it is?*
- E1 (4/5/5): Strongly unpacks the implied assumptions (knowledge/capability, accuracy/reliability) while preserving meaning; "its location" leaves the referent slightly less explicit than the instruction demands.
- E3 (5/4/5): Makes more assumptions explicit ("cognitive capacity," "this particular object or place"), best fulfilling the instruction, though the added object/place specification is a slight inference beyond the source.

**18. Reduce to a single punchy sentence with no hedging and no softening.**
*Source: I just returned from a backpack trip into the Sierra's.*
- E1 (5/4/5): Single punchy declarative sentence with all hedging removed; drops the "energies may be large" premise but keeps the core claim intact.
- E3 (5/4/5): Identical to epoch-1: clean, hedge-free single sentence; same minor omission of the "large energies" detail.

**18. Reduce to a single punchy sentence with no hedging and no softening.**
*Source: I just returned from a backpack trip into the Sierra's.*
- E1 (2/5/5): Faithful and fluent, but "I've been back" is vague and softening, and expanding to "Sierra Nevada mountains" works against the punchy/tightening goal.
- E3 (4/5/5): Direct declarative "I have returned" with no hedging and a tighter, punchier "expedition," cleanly matching the instruction.

**19. Add qualifying clauses and hedges until the original certainty is thoroughly undermined.**
*Source: Well, the first settlers of Massachusetts arrived there because of religious persecution from England and King James.*
- E1 (5/3/4): Saturates the hedging instruction ("perhaps," "I suppose one could say," "around")—the strongest stylistic match—but drops the key "another reviewer" detail and the "less than an hour" precision.
- E3 (3/2/4): More fluent and concise but only lightly hedged ("perhaps," "approximately"), and distorts the source by changing "less than an hour" to "within mere minutes."

**19. Add qualifying clauses and hedges until the original certainty is thoroughly undermined.**
*Source: Well, the first settlers of Massachusetts arrived there because of religious persecution from England and King James.*
- E1 (4/3/5): Strong, natural hedging that undermines certainty, but introduces an unwarranted "drawn westward toward the New World" addition.
- E3 (5/2/3): Densest hedging best satisfies the instruction, but the "not entirely out of their own volition" clause distorts the source meaning and the stacked clauses read convolutedly.

**20. Distill to a three-to-five word phrase a journalist would use as a headline.**
*Source: His excellent understanding of the complex character shines through every line he delivers.In addition, we get lavish sets, ornate costumes, a star-studded cast, and a sword-clashing, heart-pounding, tear-jerking finale.*
- E1 (3/3/3): Headline register is present but it runs to six words (over the 3-5 cap) and "Careful Every Step" drifts from the source's focus on words.
- E3 (4/4/5): Punchy, idiomatic em-dash headline that closely tracks the source meaning, though it slightly exceeds the strict 3-5 word limit.

**20. Distill to a three-to-five word phrase a journalist would use as a headline.**
*Source: His excellent understanding of the complex character shines through every line he delivers.In addition, we get lavish sets, ornate costumes, a star-studded cast, and a sword-clashing, heart-pounding, tear-jerking finale.*
- E1 (4/4/5): Reads as a punchy journalistic headline capturing acting, production, and finale, though at six words it slightly overshoots the 3-5 word target.
- E3 (4/4/5): Essentially identical to epoch-1 (only "Splendid" swapped for "Magnificent"); strong headline phrasing but likewise runs to six words, just over the requested length.

**21. Write as a pull quote set in large type for a magazine feature on this topic.**
*Source: I was a bit skeptical that this tape would work, but it does.*
- E1 (3/2/5): Punchy, pull-quote-shaped phrasing, but distorts the meaning by converting "$350,000 worth" into "350,000 copies" of a single book.
- E3 (4/4/5): Preserves the "$350,000 worth annually" meaning and reads as a polished pull quote, though "person...person" repetition is slightly clumsy.

**21. Write as a pull quote set in large type for a magazine feature on this topic.**
*Source: I was a bit skeptical that this tape would work, but it does.*
- E1 (3/4/5): Punchy, quotable line that fits the pull-quote format, but renders no typographic/large-type framing and drops the specific subject ("this tape").
- E3 (3/4/5): "And yet" sharpens the concessive skeptic-to-believer arc, slightly more striking as a feature pull quote, though it likewise omits the tape and any large-type cue.

**22. Render as a caption beneath a diagram in a technical manual.**
*Source: "River Season" is a wonderfully written story about small town life in America.*
- E1 (5/3/5): Convincing technical-manual figure caption that preserves the writer-director dual role, but hallucinates a "90-minute" runtime and drops the film title.
- E3 (5/3/5): Equally strong caption register and slightly cleaner phrasing, but "developed by" collapses the written-and-directed roles and it shares the same 90-minute hallucination and dropped title.

**22. Render as a caption beneath a diagram in a technical manual.**
*Source: "River Season" is a wonderfully written story about small town life in America.*
- E1 (5/4/5): Convincingly clinical technical-caption register (Figure label, genre: metadata, terse noun phrases); only minor loss is dropping the "wonderfully written" praise.
- E3 (3/3/4): Keeps the caption frame but leaks marketing/review prose ("audience engagement", "character development") that breaks the technical register and adds content not in the source.

**23. Phrase as a one-sentence entry in a subject glossary.**
*Source: Norton lost both rematches, but only by the slimmest of margins each time.*
- E1 (3/4/5): Reads as a clean declarative sentence but lacks the term-definition format that signals a true glossary entry.
- E3 (5/4/4): The colon term:definition structure convincingly renders the glossary format; meaning preserved with only minor elision of the "two possibilities" framing.

**23. Phrase as a one-sentence entry in a subject glossary.**
*Source: Norton lost both rematches, but only by the slimmest of margins each time.*
- E1 (2/1/4): Reads as a glossary-style noun phrase but is a fragment, not a sentence, and inverts the meaning by reporting victories instead of losses.
- E3 (4/3/4): Correctly preserves the narrow losses in an encyclopedic one-sentence register, but adds an unsupported claim that the margins were identical.

**24. Write as the subtitle that would appear on a documentary film about this.**
*Source: In the 1970s, the reasons for prohibition were that it could lead to confusion about gender.*
- E1 (4/4/5): Strong evocative documentary-tagline register with the colon-led phrasing; adds interpretive "grace and celebration" but meaning stays intact.
- E3 (3/3/3): Attempts cinematic framing but "captured in live action" is awkward and a mild distortion, weakening both the documentary feel and fidelity.

**24. Write as the subtitle that would appear on a documentary film about this.**
*Source: In the 1970s, the reasons for prohibition were that it could lead to confusion about gender.*
- E1 (4/3/5): Convincing documentary-subtitle register and clean colon format; weakness is "the Birth of Prohibition," which reframes the source's causal reasoning into an origin story and loses the gender-confusion rationale.
- E3 (4/1/5): Polished, dramatic documentary subtitle, but "Why Prohibition Was Banned" inverts the source (prohibition was the argued-for measure, not something banned) and "Gender Identity" adds unsupported content, so meaning is largely lost.

**25. Render as a warning label on a product that involves this subject.**
*Source: Applications of these results include an extension theorem for Mori contractions of fiber type and a classification theorem in the case Y has a structure of projective bundle or quadric fibration.*
- E1 (4/4/5): Convincing WARNING-label register with risk disclaimer, but softens "the attack" to the vaguer "the incident," a slight faithfulness slip.
- E3 (5/5/5): Strongest rendering: "CAUTION —" header plus elaborated label boilerplate, and it retains the source's "the attack" verbatim, preserving meaning fully.

**25. Render as a warning label on a product that involves this subject.**
*Source: Applications of these results include an extension theorem for Mori contractions of fiber type and a classification theorem in the case Y has a structure of projective bundle or quadric fibration.*
- E1 (3/2/4): Compact warning-label voice reads well, but it drops the projective-bundle case and conflates the two distinct theorems, distorting the source.
- E3 (4/4/4): Retains both the extension and the classification cases (projective bundle and quadric fibration) with stronger warning-label phrasing, though slightly verbose for a label.

**26. Rewrite so the conclusion appears first and the supporting evidence follows.**
*Source: Fisher Price will NOT give us a new doll to replace the broken doll.*
- E1 (3/4/5): Leads with the conclusion but compresses everything into one clause, so the conclusion-then-evidence ordering reads as a within-sentence flip rather than a clear two-part structure; fluent and faithful.
- E3 (4/4/5): Renders the structural constraint more convincingly with a distinct conclusion sentence followed by an evidence/elaboration sentence; slight expansion ("both your business and theirs") but meaning intact and polished.

**26. Rewrite so the conclusion appears first and the supporting evidence follows.**
*Source: Fisher Price will NOT give us a new doll to replace the broken doll.*
- E1 (4/3/5): Conclusion-first with a following reason clause cleanly satisfies the structural constraint, but the appended "they have no obligation to do so" is an invented justification not in the source, and it switches our/us to your/you.
- E3 (2/4/4): Preserves the original "our" framing faithfully, but the second clause merely restates the refusal rather than providing supporting evidence, so the conclusion-then-evidence structure is only superficially rendered.

**27. Convert every passive construction to active voice without changing the meaning.**
*Source: The constitution won wide support among the many Egyptians who backed the army's removal of Mursi.*
- E1 (2/4/5): Fluent and meaning-preserving, but ignored the instruction: "has been documented" is still passive voice, so the active-voice conversion was not performed.
- E3 (2/4/5): Identical to epoch-1: still passive ("has been documented"), so it likewise failed the passive-to-active conversion despite being fluent.

**27. Convert every passive construction to active voice without changing the meaning.**
*Source: The constitution won wide support among the many Egyptians who backed the army's removal of Mursi.*
- E1 (1/5/5): Fluent and faithful, but ignored the passive-to-active instruction: changed "won" to "received," which is if anything less active, and converted no actual passive construction.
- E3 (1/5/5): Identical to epoch-1: polished and meaning-preserving, but the active-voice instruction was not followed (no passive was converted, and "won"→"received" reduces agency).

**28. Restructure so each discrete claim becomes its own standalone sentence.**
*Source: He had only a brief stint at a film school.*
- E1 (5/4/5): Cleanly splits every claim into its own standalone sentence and stays within the source's comfort/fit meaning, though it mildly embellishes "looks fantastic."
- E3 (5/3/5): Tight, well-structured one-claim-per-sentence rewrite, but "holds up well over time" invents a durability claim absent from the source's "wears much better."

**28. Restructure so each discrete claim becomes its own standalone sentence.**
*Source: He had only a brief stint at a film school.*
- E1 (3/2/5): Produces standalone sentences as asked, but fabricates new claims instead of decomposing the single source claim, adding unsupported detail.
- E3 (3/2/5): Identical to epoch-1 output; satisfies the standalone-sentence form but pads with inferred claims not present in the source.

**29. Rewrite so the least important detail is introduced first and the key point is saved for last.**
*Source: The company buildings, after some time empty, have since been re-developed as office space in the Arena District.*
- E1 (2/4/5): Reads as a fluent normal sentence but the least-important-first / key-point-last reordering is barely realized, and "works exactly as expected" softens the saved punchline.
- E3 (4/3/4): "First... then, surprisingly" scaffolding convincingly saves the field-theory-agreement result for last, but "surprisingly" injects an editorial framing absent from the source and "from operators" slightly distorts the meaning.

**29. Rewrite so the least important detail is introduced first and the key point is saved for last.**
*Source: The company buildings, after some time empty, have since been re-developed as office space in the Arena District.*
- E1 (4/3/4): Genuinely reorders to lead with the empty-building detail and end on the Arena District redevelopment, but adds unsupported content ("needed new use," "someone decided").
- E3 (2/5/5): Polished and fully faithful, but keeps the source's original ordering—no real demotion of a minor detail or deferral of the key point, so the structural instruction is barely rendered.

**30. Nominalize every main verb — turn all actions into abstract nouns.**
*Source: The thermodynamic properties, such as the chemical potential of each component and the pressure of the overall system, are incorporated in the model.*
- E1 (2/3/4): Leans on abstract noun phrases but leaves the main verbs un-nominalized and drops the Daft Punk reference; otherwise clean prose.
- E3 (2/2/3): Same un-nominalized main verbs as e1 but adds unsupported elaboration that distorts the source and reads more inflated.

**30. Nominalize every main verb — turn all actions into abstract nouns.**
*Source: The thermodynamic properties, such as the chemical potential of each component and the pressure of the overall system, are incorporated in the model.*
- E1 (1/5/5): Meaning preserved and prose is polished, but it ignores the nominalization instruction entirely — the main verb "are integrated" is left as a verb, no actions turned into abstract nouns.
- E3 (0/5/5): Faithful and fluent, but worse on the task than epoch-1: it reverts to the source's own verb "incorporated," making no nominalization attempt at all.

**31. Render in the voice of a management consultant presenting findings on a slide.**
*Source: It seems so strange to be here, burying you, but it's not you.*
- E1 (4/2/5): Strong consultant slide register ("Key takeaways from our analysis"), but distorts a first-person future intention into a past empirical finding about "participants," shifting meaning.
- E3 (5/3/5): Nails the consultant-slide persona ("Key finding," "The solution? ... a mandate"), and keeps the post-retirement structured-participation core, though it inflates a personal plan into a prescriptive client strategy.

**31. Render in the voice of a management consultant presenting findings on a slide.**
*Source: It seems so strange to be here, burying you, but it's not you.*
- E1 (4/3/4): Strong consultant framing ("Key takeaways," "implications," "proceed forward") but invents "your death" and "identity or intent," distorting the source's ambiguous meaning.
- E3 (4/4/4): Good slide-deck register ("Key finding," "engagement") that better preserves the source's "feels strange / it's not you" contrast without over-asserting facts not in the original.

**32. Write as a skeptical peer reviewer's note scrawled in the margin of a draft.**
*Source: Political career Kumari belonged to the former royal family of Bijolia.*
- E1 (3/3/4): Skeptical reviewer voice is present but reads as a full analytic sentence rather than a scrawled margin note, and it invents an unsupported claim ("may have missed something important elsewhere").
- E3 (4/4/4): Captures the terse, fragmentary margin-note register convincingly ("seems redundant?") while preserving both degrees and institutions without fabrication.

**32. Write as a skeptical peer reviewer's note scrawled in the margin of a draft.**
*Source: Political career Kumari belonged to the former royal family of Bijolia.*
- E1 (3/5/5): Preserves the source fact fluently, but the commentary ("not entirely surprising") reads as mild musing rather than a convincingly skeptical reviewer's critique.
- E3 (5/5/5): Nails the skeptical peer-reviewer persona by actually critiquing the draft's coverage ("doesn't really flesh it out... need more context") while retaining the royal-family fact.

**33. Phrase in the voice of a founder pitching this to a room of investors.**
*Source: These tools should allow a phenomenological study of string models in anticipation of upcoming experiments sensitive to cosmic string radiation.*
- E1 (4/2/5): Strong founder-pitch cadence ("What if I told you...", "systemic risk", "it's a pattern"), but it drops every specific fact (names, the cheating-with-ex betrayal) and invents "client"/"systemic risk" framing not in the source.
- E3 (3/2/4): Has some pitch register ("What we're seeing here", "our case") but reads more like a reflective monologue than an investor pitch, and it loses the same concrete facts while vaguely abstracting the betrayal into "trust erodes over time."

**33. Phrase in the voice of a founder pitching this to a room of investors.**
*Source: These tools should allow a phenomenological study of string models in anticipation of upcoming experiments sensitive to cosmic string radiation.*
- E1 (4/2/5): Strong founder-pitch voice with rhetorical hooks, but drops cosmic string radiation and the phenomenological-study specifics, leaning on hype over the source content.
- E3 (4/3/5): Equally convincing pitch register while retaining more of the core (toolkit, study string models, next-gen detectors, anticipation), though still omits cosmic string radiation.

**34. Render as a museum docent explaining this to a slow-moving tour group.**
*Source: Cyclotron resonance is detected using the photoconductive response of the sample for several different Landau level occupancies.*
- E1 (5/4/5): Nails the docent-to-tour-group persona with explicit pacing cues ("take your time to absorb that... before we move on"); minor liberty adding "right now."
- E3 (4/3/5): Good docent register, but adds editorializing not in the source ("shows us how democracy works when people actually go out and vote for something important"), and lacks e1's tour-group pacing cues.

**34. Render as a museum docent explaining this to a slow-moving tour group.**
*Source: Cyclotron resonance is detected using the photoconductive response of the sample for several different Landau level occupancies.*
- E1 (4/4/5): Convincing docent voice with good tour-group pacing; weakened by inventing the term "Landau-level-resonance spectroscopy."
- E3 (3/2/5): Fluent but drifts to generic lecturer and distorts the source — adds a magnetic field and inverts detection to light "emitted" rather than the photoconductive response.

**35. Write as an anxious grad student summarizing this for their supervisor.**
*Source: Maybe you can correct it by going to a driving range before you play again.*
- E1 (1/3/5): Polished but ignores the persona—third-person marketing tone with no first-person voice, anxiety, or supervisor-summary framing, plus an unsupported "this is working" claim.
- E3 (5/4/5): Nails the anxious-grad-student-to-supervisor persona via first-person hedging and deference while preserving the core meaning; only minor casual embellishment.

**35. Write as an anxious grad student summarizing this for their supervisor.**
*Source: Maybe you can correct it by going to a driving range before you play again.*
- E1 (3/4/4): Conveys anxious, hedging deference and retains the driving-range golf-practice meaning, but reads as advice-giving rather than a grad student summarizing for a supervisor.
- E3 (2/2/4): Has an anxious tone but distorts the source by inventing a "test" context and a shared "together" outing, dropping the original play-again/practice framing.

**36. Render as a listicle aimed at a curious general reader who knows nothing about this.**
*Source: She's got her hands full: Miranda scored three big wins that night.*
- E1 (4/4/4): Strong listicle that even glosses "loanee" for the naive reader, retaining all core facts; minor invented nuance in item 3 and the ingredients dropped.
- E3 (4/2/5): Polished, engaging listicle voice, but distorts meaning: fabricates "before he joined the team," drops the loan fact, and invents the "because it stuck" reasoning.

**36. Render as a listicle aimed at a curious general reader who knows nothing about this.**
*Source: She's got her hands full: Miranda scored three big wins that night.*
- E1 (4/4/5): Clean listicle with simple accessible items; preserves the "hands full" busyness and keeps timing consistent, though list points are somewhat generic.
- E3 (4/2/5): Engaging, well-formed listicle but hallucinates a detail ("without dropping any points"), drops the "hands full" element, and contradicts itself on timing (one day vs. one night).

**37. Write in the urgent, breathless register of a breaking-news push notification.**
*Source: The back of the sculpture is hollowed out, although closed with a board.*
- E1 (5/2/5): Nails the breathless breaking-news register (all-caps WARNING, NOW, IMMEDIATELY), but hallucinates "God's Love"/"Saving Lives"/"lose faith again" and drops the personal frame.
- E3 (3/3/4): Less breathless and reads more like a product-availability alert than breaking news, but adds less fabricated content while retaining the core "comforting psalms quotes from authors."

**37. Write in the urgent, breathless register of a breaking-news push notification.**
*Source: The back of the sculpture is hollowed out, although closed with a board.*
- E1 (4/2/4): Convincing all-caps push-alert register, but "THIS IS NOT A SIGNATURE" is a non-sequitur hallucination absent from the source.
- E3 (4/2/4): Strong breaking-news urgency that reads coherently, but "DO NOT TOUCH / SECURITY ISSUE" invents content not present in the source.

**38. Phrase as a product warning written by a cautious corporate lawyer.**
*Source: Toronto FC striker Jermain Defoe has not had his faith shattered by his World Cup snub earlier this year and showed it by putting his latest tattoo on Instagram on Tuesday.*
- E1 (5/3/5): Convincingly renders the cautious-lawyer warning register, but drops the explicit tensor-product definitions (T_l = T ⊗_R (-), T_r) and the R-bimodule category, losing the source's core construction.
- E3 (5/4/4): Same strong legal-warning persona while preserving the explicit monad formulas and R-bimodule category, so meaning is far better retained; the dense technical insertion is slightly heavier prose but remains grammatical.

**38. Phrase as a product warning written by a cautious corporate lawyer.**
*Source: Toronto FC striker Jermain Defoe has not had his faith shattered by his World Cup snub earlier this year and showed it by putting his latest tattoo on Instagram on Tuesday.*
- E1 (4/1/5): Convincing cautious-lawyer disclaimer register, but drops nearly all source content (no Defoe, tattoo, World Cup snub, or Tuesday) so meaning is almost entirely lost.
- E3 (5/4/5): Nails the corporate-lawyer warning persona while retaining the key facts (Mr. Defoe, recent tattoo, posted Tuesday); only minor loss is the World Cup snub framing.

**39. Render as a pop-science explainer for someone enthusiastic but uninformed.**
*Source: Thiessen aged off the team, and was not a member of the team when Bottcher won the 2012 Canadian Junior Curling Championships.*
- E1 (3/2/5): Has the conversational explainer voice but offers no "science"/mechanism and inverts the source — frames the husband as wanting her to use it rather than hiding it because she can't stop.
- E3 (4/4/5): Genuinely pop-science — explains a mechanism ("your brain stops noticing how much you use it") while preserving the compulsion and the husband hiding the item; the brain claim is a small genre-appropriate addition.

**39. Render as a pop-science explainer for someone enthusiastic but uninformed.**
*Source: Thiessen aged off the team, and was not a member of the team when Bottcher won the 2012 Canadian Junior Curling Championships.*
- E1 (4/2/5): Strong, enthusiastic explainer voice with analogies, but drops Bottcher's name and the championship name and adds invented "missing out" framing, distorting the facts.
- E3 (4/5/5): Slightly less teacherly enthusiasm but a clean casual explainer that preserves all core facts: Bottcher, the 2012 Canadian Junior Championships, and the aged-off-before-the-win sequence.

**40. Write as the single sentence a busy executive would skim in a briefing memo.**
*Source: I worked in the morning then I played tennis with my husband.*
- E1 (4/3/4): Crisp memo register with date parenthetical and a strategic "leverage going forward" framing, but it drops the explicit Ryder Cup revenge context and overstates a "match play" win into general "leverage against Europe."
- E3 (4/4/5): Smooth single-sentence memo line that correctly preserves the Ryder Cup "modest compensation/revenge" framing and the championship win, though it omits Donaldson by name and the knockout detail.

**40. Write as the single sentence a busy executive would skim in a briefing memo.**
*Source: I worked in the morning then I played tennis with my husband.*
- E1 (5/4/5): Clean telegraphic noun-phrase memo line, fully skimmable; only minor loss is "husband"→"spouse".
- E3 (3/3/3): "Morning shift" and "followed up with spouse for tennis" distort the source and the semicolon splice reads more like a status update than a skimmable line.

**41. Rewrite in the dense, jargon-saturated register of a patent application claim.**
*Source: The stream flows south to south-southwest parallel to County Road 14-531.*
- E1 (5/4/5): Convincing patent register ("present invention discloses," "said configuration," "prior art"); faithful, with only mild elaboration beyond the source's "briefly discussed."
- E3 (5/4/5): Equally strong patent diction with idiomatic touches ("(Class IV)," "wherein said," "aforementioned"); "plays a significant functional role" slightly overstates the source's modest "role briefly discussed."

**41. Rewrite in the dense, jargon-saturated register of a patent application claim.**
*Source: The stream flows south to south-southwest parallel to County Road 14-531.*
- E1 (5/3/5): Excellent dense patent-claim register with "wherein said"/"thereby" phrasing; captures south and southwest endpoints but invents a spurious "northwardly-oriented" origin.
- E3 (4/2/5): Clean patent register, but "approximately 270 degrees" means due west, contradicting the source's south/south-southwest direction, and the "parallel to" relationship is lost.

**42. Render as a sardonic tweet from a domain expert who has lost all patience.**
*Source: I have spent a lot of money on that club, it's cost me a lot.*
- E1 (3/4/4): Retains the core 10-15% lower finding faithfully, but the tone reads as vague confusion ("isn't really real") rather than the requested sardonic, impatient expert bite.
- E3 (5/2/5): Nails the sardonic exhausted-expert persona and tweet format, but hallucinates specific values (73 +/- 12, 69 +/- 3) absent from the source, a significant fabrication of quantitative data.

**42. Render as a sardonic tweet from a domain expert who has lost all patience.**
*Source: I have spent a lot of money on that club, it's cost me a lot.*
- E1 (3/3/5): Sardonic tweet tone lands ("That's how you do business"), but the domain-expert persona is absent and "broke" adds info not in the source.
- E3 (3/2/5): Sardonic and fluent, but "still owe them money" is a distortion not in the source, and the expert persona is still missing.

**43. Write as the opening line of a bedtime story for a very young child.**
*Source: All this book accomplished was to make me want to read "Jane Eyre" again.*
- E1 (4/4/3): Solid bedtime-story register, but the redundant tacked-on "it was not good enough" reads as an awkward afterthought.
- E3 (4/4/4): Same storybook opener with a more natural closing clause; adds an emotional reaction ("very unhappy") that is a reasonable inference from the source.

**43. Write as the opening line of a bedtime story for a very young child.**
*Source: All this book accomplished was to make me want to read "Jane Eyre" again.*
- E1 (4/4/5): Strength: "Once upon a time" plus warm second-person address nails the bedtime-story opener, though it runs long for a very young child.
- E3 (2/4/3): Weakness: childlike register but no storybook opening framing, and the trailing "which is exactly what happens sometimes!" is vague and slightly awkward.

**44. Phrase as a cautionary footnote in a peer-reviewed paper.**
*Source: Shia LaBeouf has pleaded guilty to disorderly conduct stemming from an incident at a Broadway show in June.*
- E1 (4/5/5): Polished academic-footnote register with the proper hedged "It should be noted that" framing; renders the caution cleanly and concisely.
- E3 (4/5/4): Same strong register but the added "will find it inadequate" is redundant with the earlier "does not constitute an adequate resource," making it slightly wordy and circular.

**44. Phrase as a cautionary footnote in a peer-reviewed paper.**
*Source: Shia LaBeouf has pleaded guilty to disorderly conduct stemming from an incident at a Broadway show in June.*
- E1 (4/3/5): Strong cautionary-footnote register with a fitting "does not reflect current legal standing" disclaimer, but hallucinates a year (June 2018) and drops the first name.
- E3 (3/3/4): Preserves all source facts and adds a nice "(see record)" citation touch, but the second sentence editorializes ("warrant careful consideration") and reads as commentary rather than a tight footnote.

**45. Render as the opening sentence of the Wikipedia article about this subject.**
*Source: Flora There have been plenty of nature in the forest, which are rubber, takian, wha, wild rose apple, tung fah, kor, daeng khuan, kradoan, tuanghon, waii, lumpi, palm and orchid.*
- E1 (3/2/5): Encyclopedic register and naming attempt, but no true "X is a..." defining structure and it hallucinates "natural hot springs" not in the source.
- E3 (3/3/5): Same fabricated "hot springs" subject, but retains more source content (region draw, health/illness, locals and tourists) in fluent encyclopedic prose.

**45. Render as the opening sentence of the Wikipedia article about this subject.**
*Source: Flora There have been plenty of nature in the forest, which are rubber, takian, wha, wild rose apple, tung fah, kor, daeng khuan, kradoan, tuanghon, waii, lumpi, palm and orchid.*
- E1 (4/4/5): Clean encyclopedic register, but the added "each contributing to its rich biodiversity" is an unsupported editorializing flourish that is slightly off-tone for a neutral Wikipedia lead.
- E3 (5/5/5): Neutral, factual encyclopedic lead that preserves the species list exactly with no additions—an ideal Wikipedia opening sentence.

**46. Write with the flat, numbered-step tone of a procedure in an instruction manual.**
*Source: Sinclair (left) has struggled to make his mark on Manuel Pellegrini's multi-million pound Man City squad.*
- E1 (5/3/5): Clean three-step manual format with polished flat register, but invents a "document in records" step not present in the source.
- E3 (3/4/3): Sticks closer to the source ("institution known as L.") with no fabricated step, but runs the two steps onto one line, weakening the numbered-step format, and reads slightly awkward.

**46. Write with the flat, numbered-step tone of a procedure in an instruction manual.**
*Source: Sinclair (left) has struggled to make his mark on Manuel Pellegrini's multi-million pound Man City squad.*
- E1 (4/2/4): Strong manual register with conditional branching steps, but recasts the content as a hypothetical evaluation procedure and never asserts the source's actual claim, dropping Pellegrini and the multi-million-pound squad.
- E3 (4/4/4): Preserves the core meaning including Pellegrini and Sinclair's failure to establish himself; numbered procedural framing is clear though run inline rather than on separate lines, and the cost descriptor is dropped.

**47. Render as the back-cover blurb of a self-help book on this topic.**
*Source: To me, this variety makes the album more interesting - I see it as a Good Thing.I didn't care much for the drum machine in Track 6 - Moonlight Sonata (is this the hip-hop that the other reviewer mentioned?).*
- E1 (5/4/5): Convincing back-cover self-help blurb ("This guide shows you how...") that keeps the literal tree referent while extending into the genre metaphor.
- E3 (4/3/4): Hits the self-help book register but the "What you don't want to know" hook is slightly off for a blurb, and it drops the literal tree, abstracting entirely to "growth."

**47. Render as the back-cover blurb of a self-help book on this topic.**
*Source: To me, this variety makes the album more interesting - I see it as a Good Thing.I didn't care much for the drum machine in Track 6 - Moonlight Sonata (is this the hip-hop that the other reviewer mentioned?).*
- E1 (4/2/5): Convincing self-help blurb voice, but drops the Track 6 / Moonlight Sonata / drum-machine content entirely and invents "extraordinary," losing most of the source's specific meaning.
- E3 (4/4/5): Hits the back-cover blurb register ("every time you turn the page," "don't let one track ruin your experience") while retaining both the variety point and the drum-machine Moonlight Sonata complaint.

**48. Rewrite as a stage direction in a theatrical script.**
*Source: I received my package in a timely manner and within the specified time that was stated.*
- E1 (2/2/4): Fluent present-tense action prose, but reads as literary narration rather than a true stage direction, and "not entirely convinced of its worth" distorts the source's enthusiasm while dropping the page count and afternoon detail.
- E3 (2/4/4): Retains the sixty-three-page detail and the passing-it-on idea without contradicting additions, but like E1 it is narrative prose rather than a conventionally formatted stage direction (no cue/parenthetical/scene framing).

**48. Rewrite as a stage direction in a theatrical script.**
*Source: I received my package in a timely manner and within the specified time that was stated.*
- E1 (4/3/5): Convincing present-tense stage direction, but swaps the package for a "receipt" and adds an audience reaction not in the source, distorting meaning.
- E3 (5/5/5): Clean, disciplined stage direction that keeps the actual package and retains timeliness ("on schedule, without delay, as promised") with no additions.

**49. Phrase as a 30-second mindfulness prompt read aloud in a meditation app.**
*Source: The concept of selective control is developed using cell death or apoptosis in heterogeneous cell populations as an example.*
- E1 (4/4/5): Good meditation register with a breath cue, but the "that's how we know what we believe" line is slightly didactic and it drops the "rigorous" emphasis.
- E3 (5/4/5): Nails the meditation-app format with "Breathe in... / Breathe out" bookends and preserves "tested rigorously"; minor drift in framing testing as "proven true / something real."

**49. Phrase as a 30-second mindfulness prompt read aloud in a meditation app.**
*Source: The concept of selective control is developed using cell death or apoptosis in heterogeneous cell populations as an example.*
- E1 (3/2/4): Soft, app-like opening ("Breathe with me now") but pivots into a vague humanistic moral about "being different" that drops the core concept of selective control and distorts the apoptosis example.
- E3 (4/4/5): Strong meditation cadence ("Breathe in...", "Let that settle for one breath") that names the actual concept of selective control and the differing cell responses, preserving meaning far better than EPOCH-1 despite a slight "we can choose which survive" overstatement.

**50. Write as a text message from someone who understands but doesn't want to explain everything.**
*Source: In October 2012, Lopez was spotted wearing Real Madrid's famous white jersey while in the Spanish capital to perform a concert.*
- E1 (4/3/5): Best captures the terse, withholding "doesn't want to explain" persona ("But honestly... nope."), but drops the $3 first-purchase detail and shipping, losing meaning.
- E3 (4/4/4): Retains far more source detail ($3 first one, shipping, gold-plated) and stays texty, though explaining more sits in mild tension with the "doesn't want to explain everything" framing.

**50. Write as a text message from someone who understands but doesn't want to explain everything.**
*Source: In October 2012, Lopez was spotted wearing Real Madrid's famous white jersey while in the Spanish capital to perform a concert.*
- E1 (4/5/5): Strong texting register with "just wanted you to know it happened" nailing the understated, withholding persona; all source facts intact.
- E3 (3/5/4): Casual and faithful, but the interrogative "?" framing reads as someone seeking confirmation rather than someone who understands and is withholding.

