"""Domain-constrained decoding against a parts-and-fault-code catalogue.

This is a POST-DECODING constraint pass, not a beam search: the repository has
no acoustic model, so there are no lattice scores to constrain.  It takes the
text a recogniser emitted and enforces the one rule the maintenance system
needs: **no identifier is committed unless it is in the catalogue.**

For every identifier-shaped field in a hypothesis it decides one of:

  ACCEPTED   exact catalogue hit                         -> committed
  CORRECTED  exactly one catalogue entry within edit
             distance 1 (or a spelled-out form that
             re-composes to exactly one entry)            -> committed
  CONFIRM    zero or several candidates                  -> NOT committed,
                                                            routed to a human

A real whisper.cpp integration would apply the same catalogue as a grammar /
logit mask during decoding; the evaluation and the gating below would not
change.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .entities import (
    FAULT_CODE,
    PART_ID,
    Entity,
    _FAULT_RE,
    _PART_RE,
    extract_entities,
    tokenize,
)

ACCEPTED = "accepted"
CORRECTED = "corrected"
CONFIRM = "confirm"

IDENTIFIER_TYPES = (PART_ID, FAULT_CODE)

# Single spoken number words and how many digits each may stand for.  "seven"
# can be the chunk "07" or the lone digit "7"; both readings are enumerated and
# the catalogue decides, rather than a hard-coded guess.
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_NUMBER_WORDS = set(_ONES) | set(_TEENS) | set(_TENS)

# Max digit-substitution distance for a snap.  1 because the simulated failure
# mode is a single digit slip.  Measured (not assumed): radius 2 does NOT commit
# more wrong parts - uniqueness gating catches the extra collisions - but it
# roughly doubles-to-triples the share routed to a human (see
# test_radius_two_routes_more), which costs technician time for no accuracy.
SNAP_RADIUS = 1


@dataclass(frozen=True)
class Catalogue:
    parts: FrozenSet[str]
    faults: FrozenSet[str]

    def ids(self, etype: str) -> FrozenSet[str]:
        return self.parts if etype == PART_ID else self.faults

    @property
    def part_prefixes(self) -> FrozenSet[str]:
        return frozenset(p.split("-")[0] for p in self.parts)


@dataclass
class Field:
    etype: str
    value: Optional[str]          # committed value (None when CONFIRM)
    heard: str                    # what the recogniser produced
    status: str
    candidates: Tuple[str, ...] = ()


@dataclass
class DecodedReport:
    fields: List[Field]
    other_entities: List[Entity] = field(default_factory=list)

    def committed(self) -> List[Entity]:
        out = [Entity(f.etype, f.value, k) for k, f in enumerate(self.fields)
               if f.status != CONFIRM]
        return out + list(self.other_entities)

    def pending(self) -> List[Field]:
        return [f for f in self.fields if f.status == CONFIRM]


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _digit_readings(words: Sequence[str]) -> List[str]:
    """All digit strings a run of number words can denote."""
    options: List[List[str]] = []
    k = 0
    while k < len(words):
        w = words[k]
        if w in _TENS:
            if k + 1 < len(words) and words[k + 1] in _ONES:
                options.append(["%d" % (_TENS[w] + _ONES[words[k + 1]])])
                k += 2
                continue
            options.append(["%d" % _TENS[w]])
        elif w in _TEENS:
            options.append(["%d" % _TEENS[w]])
        else:
            options.append(["%02d" % _ONES[w], "%d" % _ONES[w]])
        k += 1
    return ["".join(c) for c in itertools.product(*options)]


def _resolve(etype: str, candidates: Iterable[str], catalogue: Catalogue,
             heard: str, radius: int) -> Field:
    ids = catalogue.ids(etype)
    candidates = list(dict.fromkeys(candidates))
    exact = [c for c in candidates if c in ids]
    if len(exact) == 1 and heard == exact[0]:
        return Field(etype, exact[0], heard, ACCEPTED, tuple(exact))
    if len(exact) == 1:
        return Field(etype, exact[0], heard, CORRECTED, tuple(exact))
    if len(exact) > 1:
        return Field(etype, None, heard, CONFIRM, tuple(sorted(exact)))
    # Same-length substitution distance: the identifier failure mode is a
    # digit slip, which never changes length (length-changing damage shows up
    # as a spell-out and is handled by re-composition above).
    near = sorted({i for c in candidates for i in ids
                   if len(i) == len(c) and i[:3] == c[:3]
                   and _hamming(c, i) <= radius})
    if len(near) == 1:
        return Field(etype, near[0], heard, CORRECTED, tuple(near))
    return Field(etype, None, heard, CONFIRM, tuple(near))


def constrained_decode(hypothesis: str, catalogue: Catalogue,
                       radius: int = SNAP_RADIUS) -> DecodedReport:
    tokens = tokenize(hypothesis)
    prefixes = {p.lower() for p in catalogue.part_prefixes}
    fields: List[Field] = []
    narrative: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        up = tok.upper()
        if _PART_RE.match(up):
            fields.append(_resolve(PART_ID, [up], catalogue, up, radius))
            narrative.append("<id>")
            i += 1
            continue
        if _FAULT_RE.match(up):
            fields.append(_resolve(FAULT_CODE, [up], catalogue, up, radius))
            narrative.append("<id>")
            i += 1
            continue
        if tok in prefixes or tok == "e":
            j = i + 1
            while j < len(tokens) and tokens[j] in _NUMBER_WORDS:
                j += 1
            words = tokens[i + 1:j]
            if words:
                etype, width = (FAULT_CODE, 3) if tok == "e" else (PART_ID, 4)
                cands = ["%s-%s" % (up, d) for d in _digit_readings(words)
                         if len(d) == width]
                if cands:
                    fields.append(_resolve(etype, cands, catalogue,
                                           " ".join(tokens[i:j]), radius))
                    narrative.append("<id>")
                    i = j
                    continue
        narrative.append(tok)
        i += 1
    others = [e for e in extract_entities(narrative)
              if e.etype not in IDENTIFIER_TYPES]
    return DecodedReport(fields=fields, other_entities=others)


def naive_nearest_decode(hypothesis: str, catalogue: Catalogue) -> DecodedReport:
    """The generic approach: always snap to the closest catalogue entry.

    It never emits an out-of-catalogue code, so it *looks* safe - but it turns
    every unknown part into a confident, validated, wrong one.
    """
    rep = constrained_decode(hypothesis, catalogue, radius=99)
    for f in rep.fields:
        if f.status == CONFIRM:
            pool = f.candidates or tuple(catalogue.ids(f.etype))
            key = f.heard.upper() if "-" in f.heard else f.heard
            f.value = min(sorted(pool), key=lambda c: levenshtein(key, c))
            f.status = CORRECTED
    return rep


def build_catalogue(corpus, off_catalogue_share: float = 0.05,
                    distractors: int = 400, seed: int = 7):
    """Catalogue from the corpus's planted IDs, with planted omissions.

    Returns (catalogue, off_catalogue_indices).  The omitted utterances carry
    a part number the catalogue has never seen - a newly fitted component -
    and the only correct outcome for them is CONFIRM.  Distractor IDs fill the
    catalogue out to a realistic density so snapping is not trivially unique.
    """
    rng = random.Random(seed)
    idx = list(range(len(corpus)))
    rng.shuffle(idx)
    off = set(idx[:int(round(off_catalogue_share * len(corpus)))])
    parts, faults = set(), set()
    for k, utt in enumerate(corpus):
        for e in utt.gold:
            if e.etype == FAULT_CODE:
                faults.add(e.value)
            elif e.etype == PART_ID and k not in off:
                parts.add(e.value)
    off_parts = {e.value for k in off for e in corpus[k].gold
                 if e.etype == PART_ID}
    # A part number can recur across utterances.  Without this, an omitted
    # part re-enters via another report and the planted "unknown" is not
    # unknown at all (the test suite caught exactly that on utterance 64).
    parts -= off_parts
    while len(parts) < len(corpus) - len(off) + distractors:
        cand = "%s-%d" % (rng.choice(("GB", "YD", "PT", "HV", "BL")),
                          rng.randint(1000, 9999))
        if cand not in off_parts:
            parts.add(cand)
    return Catalogue(frozenset(parts), frozenset(faults)), off


@dataclass
class IdentifierOutcome:
    gold: int = 0
    committed_correct: int = 0
    committed_wrong: int = 0      # the dangerous one: a wrong ID in the record
    routed: int = 0               # sent for confirmation
    missing: int = 0              # no field produced at all
    unvalidated: int = 0          # committed ID absent from catalogue

    @property
    def committed_error_rate(self) -> float:
        return (self.committed_wrong / self.gold) if self.gold else 0.0

    @property
    def routed_rate(self) -> float:
        return (self.routed / self.gold) if self.gold else 0.0


def score_identifiers(gold: Sequence[Entity], report: DecodedReport,
                      catalogue: Catalogue, expect_confirm: bool,
                      acc: Dict[str, IdentifierOutcome]) -> None:
    """Score per identifier type, aligned by order within the type.

    For an off-catalogue utterance the gold PART_ID is only "handled
    correctly" if it is routed; committing even its true value would violate
    the never-emit-unvalidated rule.
    """
    for etype in IDENTIFIER_TYPES:
        o = acc.setdefault(etype, IdentifierOutcome())
        g = [e.value for e in gold if e.etype == etype]
        f = [x for x in report.fields if x.etype == etype]
        for k, gv in enumerate(g):
            o.gold += 1
            if k >= len(f):
                o.missing += 1
                continue
            fk = f[k]
            if fk.status == CONFIRM:
                o.routed += 1
                continue
            if fk.value not in catalogue.ids(etype):
                o.unvalidated += 1
            if fk.value == gv and not (expect_confirm and etype == PART_ID):
                o.committed_correct += 1
            else:
                o.committed_wrong += 1


def score_raw_identifiers(gold: Sequence[Entity], hypothesis: str,
                          catalogue: Catalogue,
                          acc: Dict[str, IdentifierOutcome],
                          expect_confirm: bool = False) -> None:
    """Unconstrained baseline: every extracted ID is committed as heard."""
    fields = [Field(e.etype, e.value, e.value, ACCEPTED)
              for e in extract_entities(hypothesis)
              if e.etype in IDENTIFIER_TYPES]
    score_identifiers(gold, DecodedReport(fields), catalogue, expect_confirm,
                      acc)
