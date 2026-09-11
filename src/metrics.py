"""Two metrics over the same transcript pair: word error rate and entity
error rate.

WER is implemented properly (Levenshtein alignment by dynamic programming
with substitution / insertion / deletion, plus a backtrace so the demo can
name the offending words).  Entity error rate runs the same alignment idea
over TYPED FIELDS instead of words, per type, so a system that is fine on
dates and catastrophic on part IDs cannot hide behind an average.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .entities import (
    ENTITY_TYPES,
    Entity,
    extract_entities,
    tokenize,
)

MATCH = "match"
SUB = "sub"
INS = "ins"
DEL = "del"


@dataclass
class ErrorCounts:
    hits: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def reference_count(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def rate(self) -> float:
        """Errors per reference item.

        Undefined with no reference items; 0.0 is returned only when the
        hypothesis is also empty, otherwise the insertions still count and the
        rate is reported against a denominator of 1 so a system cannot score
        0.0 by hallucinating fields into an empty reference.
        """
        denom = self.reference_count
        if denom == 0:
            return 0.0 if self.insertions == 0 else float(self.insertions)
        return self.errors / denom


@dataclass
class Alignment:
    """Edit operations taking the reference to the hypothesis."""

    ops: List[Tuple[str, object, object]] = field(default_factory=list)
    counts: ErrorCounts = field(default_factory=ErrorCounts)

    def errors_only(self):
        return [op for op in self.ops if op[0] != MATCH]


def _align(ref: Sequence, hyp: Sequence, equal) -> Alignment:
    """Levenshtein DP with backtrace.

    Unit costs for substitution, insertion and deletion.  Ties are broken
    substitution-first so an aligned pair is reported as one substitution
    rather than as a deletion plus an insertion, which would double-count the
    same mistake.
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = DEL
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = INS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = equal(ref[i - 1], hyp[j - 1])
            diag = dp[i - 1][j - 1] + (0 if same else 1)
            up = dp[i - 1][j] + 1
            left = dp[i][j - 1] + 1
            best = min(diag, up, left)
            dp[i][j] = best
            if best == diag:
                back[i][j] = MATCH if same else SUB
            elif best == up:
                back[i][j] = DEL
            else:
                back[i][j] = INS

    ops: List[Tuple[str, object, object]] = []
    counts = ErrorCounts()
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == MATCH:
            ops.append((MATCH, ref[i - 1], hyp[j - 1]))
            counts.hits += 1
            i -= 1
            j -= 1
        elif op == SUB:
            ops.append((SUB, ref[i - 1], hyp[j - 1]))
            counts.substitutions += 1
            i -= 1
            j -= 1
        elif op == DEL:
            ops.append((DEL, ref[i - 1], None))
            counts.deletions += 1
            i -= 1
        else:
            ops.append((INS, None, hyp[j - 1]))
            counts.insertions += 1
            j -= 1
    ops.reverse()
    return Alignment(ops=ops, counts=counts)


def align_words(reference, hypothesis) -> Alignment:
    ref = reference if isinstance(reference, (list, tuple)) else tokenize(reference)
    hyp = hypothesis if isinstance(hypothesis, (list, tuple)) else tokenize(hypothesis)
    return _align(list(ref), list(hyp), lambda a, b: a == b)


def word_error_rate(reference, hypothesis) -> float:
    return align_words(reference, hypothesis).counts.rate


@dataclass
class EntityReport:
    overall: ErrorCounts
    per_type: Dict[str, ErrorCounts]
    alignments: Dict[str, Alignment]

    @property
    def rate(self) -> float:
        return self.overall.rate

    def type_rate(self, etype: str) -> float:
        return self.per_type[etype].rate

    def worst_type(self):
        """Type with the highest rate among types that occur in the reference.

        Reporting this alongside the average is the point of the per-type
        breakdown: an average over five field types can look survivable while
        one class of field is unusable.
        """
        present = [
            (t, c) for t, c in self.per_type.items() if c.reference_count > 0
        ]
        if not present:
            return None
        return max(present, key=lambda tc: (tc[1].rate, tc[1].reference_count))[0]


def entity_error_report(gold: Sequence[Entity], predicted: Sequence[Entity]) -> EntityReport:
    """Score extracted fields against planted ground-truth fields.

    Alignment is per type and by ORDER, not by absolute token position, so an
    entity that has shifted because the ASR inserted or dropped narrative
    words ahead of it still matches its gold counterpart.  Aligning by
    position would punish a correct field for the sins of the words around it.
    """
    per_type: Dict[str, ErrorCounts] = {}
    alignments: Dict[str, Alignment] = {}
    overall = ErrorCounts()
    for etype in ENTITY_TYPES:
        g = [e for e in gold if e.etype == etype]
        p = [e for e in predicted if e.etype == etype]
        al = _align(g, p, lambda a, b: a.value == b.value)
        per_type[etype] = al.counts
        alignments[etype] = al
        overall.hits += al.counts.hits
        overall.substitutions += al.counts.substitutions
        overall.insertions += al.counts.insertions
        overall.deletions += al.counts.deletions
    return EntityReport(overall=overall, per_type=per_type, alignments=alignments)


def evaluate(reference: str, hypothesis: str, gold: Sequence[Entity]):
    """Both metrics for one utterance."""
    words = align_words(reference, hypothesis)
    predicted = extract_entities(hypothesis)
    entities = entity_error_report(gold, predicted)
    return words, entities


def entity_token_positions(reference: str, gold: Sequence[Entity]) -> set:
    """Token indices in the reference that belong to a scored field.

    Used by the demo to split word errors into "damaged a field" versus
    "damaged narrative", which is the whole divergence story.
    """
    tokens = tokenize(reference)
    marked = set()
    for e in gold:
        width = 2 if e.etype == "QUANTITY" else len(e.value.split())
        if e.etype in ("PART_ID", "FAULT_CODE", "DATE"):
            width = 1
        for k in range(e.pos, min(e.pos + width, len(tokens))):
            marked.add(k)
    return marked


def classify_word_errors(reference: str, hypothesis: str, gold: Sequence[Entity]):
    """Split the word-level edit list into entity-token and narrative errors."""
    marked = entity_token_positions(reference, gold)
    al = align_words(reference, hypothesis)
    entity_errors = []
    narrative_errors = []
    ri = 0
    for op, r, h in al.ops:
        in_entity = op != INS and ri in marked
        if op != MATCH:
            record = (op, r, h)
            (entity_errors if in_entity else narrative_errors).append(record)
        if op != INS:
            ri += 1
    return entity_errors, narrative_errors
