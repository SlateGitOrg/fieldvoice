"""A DETERMINISTIC SIMULATED ASR BACKEND.

There is no speech recogniser in this repository.  There is no audio, no
model weights, no network and no GPU.  What exists is a clean backend
interface plus a seeded corruption model: given a reference transcript and a
noise / accent / vocabulary profile it emits a plausible hypothesis using
phonetic confusion classes, number and unit mangling, and out-of-vocabulary
spell-out of domain identifiers.

The simulator stands in for the acoustic model.  The things under test are
the INTERFACE and the EVALUATION, which would be unchanged if a real
whisper.cpp binding were dropped in behind ``AsrBackend``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

from .entities import (
    DATE,
    FAULT_CODE,
    LOCATION,
    LOCATION_GAZETTEER,
    PART_ID,
    QUANTITY,
    extract_entities,
    tokenize,
)

# Word-level phonetic confusions: pairs a recogniser genuinely trips on in
# noise.  These are narrative words -- they move WER and touch no field.
PHONETIC_CONFUSIONS = {
    "inspected": "expected", "expected": "inspected",
    "the": "a", "and": "an", "was": "is", "we": "he", "of": "off",
    "then": "than", "applied": "a-plied", "checked": "chucked",
    "turbine": "turban", "bearing": "barren", "greased": "creased",
    "cleared": "clean", "before": "be-for", "reading": "readied",
    "replaced": "re-placed", "torque": "talk", "blade": "played",
    "tested": "tasted", "on": "in", "at": "that", "crew": "grew",
}

# Teens and tens are the classic magnitude confusion.  "fifteen" heard as
# "fifty" is a factor of ~3.3 on a dose or a torque spec; WER counts it the
# same as hearing "the" as "a".
TEEN_TENS = {
    "thirteen": "thirty", "fourteen": "forty", "fifteen": "fifty",
    "sixteen": "sixty", "seventeen": "seventy", "eighteen": "eighty",
    "nineteen": "ninety",
    "thirty": "thirteen", "forty": "fourteen", "fifty": "fifteen",
    "sixty": "sixteen", "seventy": "seventeen", "eighty": "eighteen",
    "ninety": "nineteen",
}

# Units confuse within an acoustic neighbourhood, not within a physical one:
# that is exactly why a unit slip is silent to a listener and fatal to a work
# order.
UNIT_CONFUSIONS = {
    "mg": "ml", "ml": "mg", "nm": "mm", "mm": "nm",
    "cm": "mm", "l": "ml", "m": "mm",
}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


@dataclass(frozen=True)
class NoiseProfile:
    """Per-class error propensities.

    Split by class on purpose: real failure modes are not uniform across a
    sentence, and a single global "error rate" knob could not produce the two
    systems whose ranking flips.
    """

    name: str
    narrative_error_rate: float = 0.0
    identifier_error_rate: float = 0.0
    number_error_rate: float = 0.0
    unit_error_rate: float = 0.0
    date_error_rate: float = 0.0
    location_error_rate: float = 0.0
    # Share of identifier errors that become an out-of-vocabulary spell-out
    # ("GB-4471" -> "gb forty four seventy one") rather than a digit slip.
    # Spell-out is the failure domain-constrained decoding is meant to remove,
    # and it is also the one that inflates token count, so WER and entity
    # error rate disagree hardest on it.
    spellout_share: float = 0.5


CLEAN = NoiseProfile("clean")

# System A: a strong general recogniser.  Crisp on English narrative, no
# domain vocabulary, so it mangles identifiers, magnitudes and units.
NARRATIVE_STRONG = NoiseProfile(
    name="A-general",
    narrative_error_rate=0.02,
    identifier_error_rate=0.60,
    number_error_rate=0.45,
    unit_error_rate=0.45,
    date_error_rate=0.40,
    location_error_rate=0.30,
    spellout_share=0.5,
)

# System B: a domain-constrained recogniser.  Accented and noisy on narrative,
# but decoding is pinned to the parts / fault-code / unit vocabulary.
DOMAIN_CONSTRAINED = NoiseProfile(
    name="B-domain",
    # 0.30 is not a free parameter: it is the accented-speaker-in-turbine-noise
    # condition, and it is above the measured crossover (~0.28) at which B
    # stops looking better than A on WER while still being ~12x better on
    # entity error rate.  Below the crossover B wins on both metrics and the
    # corpus makes no argument at all; see test_flip_needs_narrative_noise.
    narrative_error_rate=0.30,
    identifier_error_rate=0.04,
    number_error_rate=0.05,
    unit_error_rate=0.03,
    date_error_rate=0.05,
    location_error_rate=0.04,
    spellout_share=0.0,
)

PROFILES = {p.name: p for p in (CLEAN, NARRATIVE_STRONG, DOMAIN_CONSTRAINED)}


class AsrBackend:
    """The seam a real recogniser would slot into."""

    name = "abstract"

    def transcribe(self, reference: str) -> str:
        raise NotImplementedError

    def transcribe_batch(self, references: Sequence[str]) -> List[str]:
        return [self.transcribe(r) for r in references]


class SimulatedAsr(AsrBackend):
    """Seeded, reproducible corruption of a known reference transcript."""

    def __init__(self, profile: NoiseProfile, seed: int = 0):
        self.profile = profile
        self.seed = seed
        self.name = profile.name

    def transcribe(self, reference: str) -> str:
        # Seeding from (seed, profile, reference) rather than carrying rng
        # state across calls keeps every utterance independently reproducible:
        # transcribing one utterance twice, or in a different batch order,
        # yields the same hypothesis.
        rng = random.Random("%d|%s|%s" % (self.seed, self.profile.name, reference))
        tokens = tokenize(reference)
        spans = self._entity_spans(tokens)
        out: List[str] = []
        i = 0
        while i < len(tokens):
            span = spans.get(i)
            if span is None:
                out.append(self._corrupt_narrative(tokens[i], rng))
                i += 1
                continue
            etype, width = span
            out.extend(self._corrupt_entity(etype, tokens[i:i + width], rng))
            i += width
        return " ".join(out)

    def _entity_spans(self, tokens: Sequence[str]):
        spans = {}
        for e in extract_entities(list(tokens)):
            if e.etype == QUANTITY:
                width = 2
            elif e.etype == LOCATION:
                width = len(e.value.split())
            else:
                width = 1
            spans[e.pos] = (e.etype, width)
        return spans

    def _corrupt_narrative(self, token: str, rng: random.Random) -> str:
        if rng.random() >= self.profile.narrative_error_rate:
            return token
        if token in PHONETIC_CONFUSIONS:
            return PHONETIC_CONFUSIONS[token]
        return _consonant_slip(token, rng)

    def _corrupt_entity(self, etype, toks, rng: random.Random) -> List[str]:
        toks = list(toks)
        if etype in (PART_ID, FAULT_CODE):
            if rng.random() >= self.profile.identifier_error_rate:
                return toks
            if rng.random() < self.profile.spellout_share:
                return _spell_out_identifier(toks[0])
            return [_slip_digit(toks[0], rng)]
        if etype == QUANTITY:
            magnitude, unit = toks[0], toks[1]
            if rng.random() < self.profile.number_error_rate:
                magnitude = _corrupt_magnitude(magnitude, rng)
            if rng.random() < self.profile.unit_error_rate:
                unit = UNIT_CONFUSIONS.get(unit, unit)
            return [magnitude, unit]
        if etype == DATE:
            if rng.random() >= self.profile.date_error_rate:
                return toks
            return [_corrupt_date(toks[0], rng)]
        if etype == LOCATION:
            if rng.random() >= self.profile.location_error_rate:
                return toks
            others = [g for g in LOCATION_GAZETTEER if g != " ".join(toks)]
            return rng.choice(others).split()
        return toks


def _consonant_slip(token: str, rng: random.Random) -> str:
    """Fallback phonetic confusion: swap one consonant for a close one."""
    classes = {"b": "p", "p": "b", "d": "t", "t": "d", "m": "n", "n": "m",
               "v": "f", "f": "v", "s": "z", "z": "s", "g": "k", "k": "g"}
    idxs = [k for k, ch in enumerate(token) if ch in classes]
    if not idxs:
        return token
    k = rng.choice(idxs)
    return token[:k] + classes[token[k]] + token[k + 1:]


def _slip_digit(token: str, rng: random.Random) -> str:
    idxs = [k for k, ch in enumerate(token) if ch.isdigit()]
    if not idxs:
        return token
    k = rng.choice(idxs)
    digit = int(token[k])
    new = (digit + rng.choice([1, -1, 2])) % 10
    return token[:k] + str(new) + token[k + 1:]


def _spell_out_identifier(token: str) -> List[str]:
    """Turn ``gb-4471`` into ``gb forty four seventy one``.

    This is the out-of-vocabulary failure this models: the identifier
    survives acoustically but is written as narrative, so the structured field
    is simply gone while WER records only a handful of word errors.
    """
    letters, _, digits = token.partition("-")
    out = [letters]
    digits = "".join(ch for ch in digits if ch.isdigit())
    for k in range(0, len(digits), 2):
        out.extend(_number_words(int(digits[k:k + 2])))
    return out


def _number_words(value: int) -> List[str]:
    if value < 20:
        return [_ONES[value]]
    tens, ones = divmod(value, 10)
    if ones == 0:
        return [_TENS[tens]]
    return [_TENS[tens], _ONES[ones]]


def _corrupt_magnitude(token: str, rng: random.Random) -> str:
    if token in TEEN_TENS:
        return TEEN_TENS[token]
    if token.isdigit():
        value = int(token)
        if 13 <= value <= 19:
            return str(value % 10 * 10)
        if value in (30, 40, 50, 60, 70, 80, 90):
            return str(10 + value // 10)
        return _slip_digit(token, rng)
    return token


def _corrupt_date(token: str, rng: random.Random) -> str:
    parts = token.split("-")
    if len(parts) != 3:
        return token
    year, month, day = parts
    # Digit transposition in the day is the characteristic date failure of a
    # recogniser reading a spoken date back into ISO form.
    if len(day) == 2 and day[0] != day[1]:
        return "-".join([year, month, day[::-1]])
    return "-".join([year, month, "%02d" % ((int(day) % 28) + 1)])
