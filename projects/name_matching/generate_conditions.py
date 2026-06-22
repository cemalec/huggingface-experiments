#!/usr/bin/env python3
"""Stage A: build the bank of name-mismatch CONDITIONS that guides Stage B.

Analogous to the paraphraser's `generate_instructions.py`, but DETERMINISTIC —
name-mismatch categories are enumerable, so no model/API call is needed. Run once;
the output (`conditions.json`) is fed into Stage B, which pairs each seed NAME with
several sampled conditions and asks the teacher to emit the corresponding variant.

A condition is a natural-language description of ONE way two names can differ while
(usually) still referring to the same person — the kind of near-match a dumb string
comparison rejects but a good matcher must accept. Each entry carries:

    {"condition": str, "axes": [str, ...], "label": "match" | "non-match"}

Axes (the mismatch dimensions the student must learn to see through):

    order      token reordering ............... "Smith, John" / "Smith John"
    initial    full <-> initial, middle add/drop "J. L. Smith"
    format     case / punctuation / spacing ..... "JOHN SMITH", "OBrien"
    suffix     generational suffix / title ...... drop "Jr.", add "Dr."
    typo       OCR / keyboard noise ............. "Smlth", "Jonh"
    nickname   diminutive / formal / alternate .. "Bob" <-> "Robert"      (knowledge)
    phonetic   homophone respelling ............. "Catherine" / "Katherine" (knowledge)
    translit   diacritics / romanization ........ "Jose" / "José"          (knowledge)
    cultural   particles / patronymic / maiden .. "van der Berg"           (knowledge)
    negative   a DIFFERENT person (true mismatch) "John Smith" vs "Jane Smith"

`label` is "match" everywhere except the `negative` axis: negatives are confusable
*non-*matches (same surname, initials line up, etc.) that the matcher must REJECT.

Tier (set downstream by sample_conditions.tier_for): a condition is HARD if it
touches a knowledge axis {nickname, phonetic, translit, cultural, negative} or
combines >1 axis; otherwise EASY. Easy -> Haiku, hard -> Sonnet.

Usage:
    python generate_conditions.py            # writes conditions.json + review sample
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
OUT_PATH = HERE / "conditions.json"
REVIEW_DIR = HERE / "review"
REVIEW_PATH = REVIEW_DIR / "conditions_review_sample.json"
REVIEW_SAMPLE_SIZE = 40
REVIEW_SEED = 0

# Single-axis conditions. Each axis is covered with several phrasings so the
# student is robust to how a mismatch is described (mirrors the paraphraser's
# "cover every axis >=2x with different phrasings").
SINGLE: dict[str, list[str]] = {
    "order": [
        "First and last name are swapped (given/family order reversed).",
        "Written in 'Last, First' order with a comma.",
        "Family name placed first with no comma (Eastern name order).",
    ],
    "initial": [
        "First name reduced to its initial.",
        "Middle name reduced to a single middle initial.",
        "Middle name or middle initial dropped entirely.",
        "A middle initial added that was not in the original.",
        "Both first and middle names reduced to initials.",
    ],
    "format": [
        "Whole name rendered in ALL CAPS.",
        "Whole name rendered in all lowercase.",
        "Hyphen removed from a hyphenated surname.",
        "Apostrophe removed from the surname (e.g., O'Brien -> OBrien).",
        "The period dropped from an initial (e.g., 'J.' -> 'J').",
        "Inconsistent spacing introduced between name parts.",
    ],
    "suffix": [
        "Generational suffix dropped (Jr., Sr., II, III).",
        "Generational suffix added that was not present.",
        "Honorific or professional title prepended (Dr., Mr., Ms.).",
    ],
    "typo": [
        "A single letter substituted in the surname, OCR-style (e.g., i -> l).",
        "Two adjacent letters transposed in the first name.",
        "A letter doubled somewhere in the surname.",
        "A letter dropped from the first name.",
        "A keyboard-adjacent letter substituted in the surname.",
    ],
    "nickname": [
        "First name replaced with a common diminutive or nickname (e.g., Robert -> Bob).",
        "Nickname expanded back to the formal given name (e.g., Bill -> William).",
        "First name replaced with an accepted longer/alternate form (e.g., John -> Johnathan).",
    ],
    "phonetic": [
        "Surname respelled to a phonetically equivalent variant (e.g., Smith -> Smyth).",
        "First name given a homophone spelling (e.g., Catherine -> Katherine).",
        "A silent-letter or vowel variant that sounds identical.",
    ],
    "translit": [
        "Diacritics stripped from the name (e.g., Jose for José, Muller for Müller).",
        "Name re-romanized via a different transliteration system (e.g., Tchaikovsky -> Chaykovsky).",
        "An accented letter expanded (ü -> ue, ß -> ss).",
    ],
    "cultural": [
        "Surname given with vs. without a name particle (e.g., 'van der Berg' -> 'Vanderberg').",
        "A patronymic added or dropped.",
        "Maiden and married surnames interchanged or hyphenated.",
        "A compound surname collapsed to a single part.",
    ],
    "negative": [
        "A DIFFERENT person who happens to share the same surname (different first name).",
        "A DIFFERENT person who shares the same first name but a different surname.",
        "A common-name collision: a plausibly different individual with a very similar full name.",
        "Initials line up but the spelled-out names differ (a different person).",
    ],
}

# Knowledge axes -> hard tier; also any multi-axis combo is hard (see sample_conditions).
HARD_AXES = {"nickname", "phonetic", "translit", "cultural", "negative"}

# Short imperative clause per axis, used to auto-compose natural combinatorial
# conditions ("apply two changes at once: A; B"). Negatives are never combined.
AX_CLAUSE = {
    "order":    "swap the given and family name order",
    "initial":  "reduce the first name to an initial",
    "format":   "uppercase the entire name",
    "suffix":   "drop any generational suffix",
    "typo":     "substitute one letter in the surname",
    "nickname": "replace the first name with an accepted alternate or nickname",
    "phonetic": "respell the surname to a phonetic equivalent",
    "translit": "strip the diacritics from the name",
    "cultural": "collapse or alter a name particle",
}

# Curated axis pairs that produce realistic two-change near-matches. The first entry
# is the user's motivating example (nickname + typo -> "Johnathan L. Smlth").
COMBO_PAIRS = [
    ("nickname", "typo"), ("order", "initial"), ("translit", "format"),
    ("nickname", "order"), ("phonetic", "suffix"), ("typo", "initial"),
    ("cultural", "format"), ("translit", "initial"), ("nickname", "format"),
    ("phonetic", "order"), ("nickname", "initial"), ("typo", "order"),
    ("translit", "suffix"), ("cultural", "initial"), ("phonetic", "initial"),
    ("typo", "format"), ("nickname", "suffix"), ("order", "format"),
    ("translit", "order"), ("phonetic", "format"), ("cultural", "order"),
    ("suffix", "initial"), ("nickname", "phonetic"), ("typo", "suffix"),
    ("order", "suffix"),
]


def build_bank() -> list[dict]:
    bank: list[dict] = []
    seen: set[str] = set()

    def add(condition: str, axes: list[str], label: str) -> None:
        if condition in seen:
            return
        seen.add(condition)
        bank.append({"condition": condition, "axes": axes, "label": label})

    for axis, phrasings in SINGLE.items():
        label = "non-match" if axis == "negative" else "match"
        for cond in phrasings:
            add(cond, [axis], label)

    for a, b in COMBO_PAIRS:
        cond = f"Apply two changes at once: {AX_CLAUSE[a]}; and {AX_CLAUSE[b]}."
        add(cond, [a, b], "match")

    return bank


def main() -> None:
    bank = build_bank()
    OUT_PATH.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n")

    singles = sum(1 for e in bank if len(e["axes"]) == 1)
    combos = len(bank) - singles
    negatives = sum(1 for e in bank if e["label"] == "non-match")
    print(f"wrote {len(bank)} conditions -> {OUT_PATH.name} "
          f"({singles} single-axis, {combos} combinatorial; {negatives} non-match)")

    REVIEW_DIR.mkdir(exist_ok=True)
    sample = random.Random(REVIEW_SEED).sample(bank, min(REVIEW_SAMPLE_SIZE, len(bank)))
    REVIEW_PATH.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(sample)}-item review sample -> {REVIEW_PATH.relative_to(HERE)}")


if __name__ == "__main__":
    main()
