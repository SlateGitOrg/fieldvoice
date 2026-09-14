"""The 60-second artefact: the same two systems ranked two ways.

Run with ``python -m src.demo``.  ASCII only, because the target console is
Windows cp1252.
"""

from __future__ import annotations

from .asr import DOMAIN_CONSTRAINED, NARRATIVE_STRONG, SimulatedAsr
from .entities import ENTITY_TYPES, extract_entities
from .generator import headline_case, make_corpus
from .metrics import (
    ErrorCounts,
    align_words,
    classify_word_errors,
    entity_error_report,
    word_error_rate,
)

BAR = "=" * 74
CORPUS_SIZE = 200
SEED = 4


def rule(title: str) -> None:
    print()
    print(BAR)
    print(title)
    print(BAR)


def wrap(label: str, text: str, width: int = 62) -> None:
    words = text.split()
    line = ""
    first = True
    for w in words:
        if len(line) + len(w) + 1 > width:
            print("%-10s %s" % (label if first else "", line))
            first = False
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        print("%-10s %s" % (label if first else "", line))


def headline() -> None:
    rule("HEADLINE: the ranking flips")
    reference, gold, hyp_a, hyp_b = headline_case()
    wrap("REFERENCE", reference)
    print()
    wrap("SYSTEM A", hyp_a)
    print()
    wrap("SYSTEM B", hyp_b)

    wer_a = word_error_rate(reference, hyp_a)
    wer_b = word_error_rate(reference, hyp_b)
    rep_a = entity_error_report(gold, extract_entities(hyp_a))
    rep_b = entity_error_report(gold, extract_entities(hyp_b))

    print()
    print("%-22s %8s %9s %11s" % ("metric", "A", "B", "prefers"))
    print("-" * 60)
    print("%-22s %8.3f %9.3f %11s" % (
        "word error rate", wer_a, wer_b, "A" if wer_a < wer_b else "B"))
    print("%-22s %8.3f %9.3f %11s" % (
        "entity error rate", rep_a.rate, rep_b.rate,
        "A" if rep_a.rate < rep_b.rate else "B"))
    print()
    print("WER prefers the transcript whose every mistake lands on a field.")

    for name, hyp in (("A", hyp_a), ("B", hyp_b)):
        ent, narr = classify_word_errors(reference, hyp, gold)
        print()
        print("System %s: %d word errors -> %d on scored fields, %d on narrative"
              % (name, len(ent) + len(narr), len(ent), len(narr)))
        for op, r, h in ent:
            print("    FIELD     %-14s %-4s %s" % (r, op, h))
        for op, r, h in narr:
            print("    narrative %-14s %-4s %s" % (r, op, h))


def corpus_evaluation() -> None:
    rule("CORPUS: %d simulated field reports, two ASR profiles" % CORPUS_SIZE)
    corpus = make_corpus(CORPUS_SIZE)
    systems = [
        SimulatedAsr(NARRATIVE_STRONG, seed=SEED),
        SimulatedAsr(DOMAIN_CONSTRAINED, seed=SEED),
    ]
    results = {}
    for asr in systems:
        word_errors = 0
        word_total = 0
        totals = {t: ErrorCounts() for t in ENTITY_TYPES}
        overall = ErrorCounts()
        for utt in corpus:
            hyp = asr.transcribe(utt.reference)
            al = align_words(utt.reference, hyp)
            word_errors += al.counts.errors
            word_total += al.counts.reference_count
            rep = entity_error_report(utt.gold, extract_entities(hyp))
            for t in ENTITY_TYPES:
                c = rep.per_type[t]
                acc = totals[t]
                acc.hits += c.hits
                acc.substitutions += c.substitutions
                acc.insertions += c.insertions
                acc.deletions += c.deletions
            overall.hits += rep.overall.hits
            overall.substitutions += rep.overall.substitutions
            overall.insertions += rep.overall.insertions
            overall.deletions += rep.overall.deletions
        results[asr.name] = (word_errors / word_total, overall, totals)

    names = list(results)
    print("%-24s %12s %12s" % ("metric", names[0], names[1]))
    print("-" * 52)
    print("%-24s %12.3f %12.3f" % (
        "word error rate", results[names[0]][0], results[names[1]][0]))
    print("%-24s %12.3f %12.3f" % (
        "entity error rate",
        results[names[0]][1].rate, results[names[1]][1].rate))
    print()
    print("per-entity-type error rate (the average above hides these)")
    print("%-24s %12s %12s" % ("entity type", names[0], names[1]))
    print("-" * 52)
    for t in ENTITY_TYPES:
        a = results[names[0]][2][t]
        b = results[names[1]][2][t]
        print("%-24s %12.3f %12.3f" % (t, a.rate, b.rate))
    print()
    print("Ranked by WER      : %s" % _rank(results, 0))
    print("Ranked by entity ER: %s" % _rank(results, 1))


def constrained_section() -> None:
    from .decoding import (IDENTIFIER_TYPES, build_catalogue,
                           constrained_decode, naive_nearest_decode,
                           score_identifiers, score_raw_identifiers)
    rule("CONSTRAINED DECODING: A-general output vs parts/fault catalogue")
    corpus = make_corpus(CORPUS_SIZE)
    cat, off = build_catalogue(corpus)
    print("catalogue: %d parts, %d fault codes; %d utterances carry a planted"
          % (len(cat.parts), len(cat.faults), len(off)))
    print("part number that is NOT in the catalogue (must be routed).")
    asr = SimulatedAsr(NARRATIVE_STRONG, seed=SEED)
    raw, con, nai = {}, {}, {}
    for k, u in enumerate(corpus):
        h = asr.transcribe(u.reference)
        score_raw_identifiers(u.gold, h, cat, raw, k in off)
        score_identifiers(u.gold, constrained_decode(h, cat), cat, k in off, con)
        score_identifiers(u.gold, naive_nearest_decode(h, cat), cat, k in off,
                          nai)
    print()
    print("%-11s %-12s %9s %9s %8s %8s %11s" % (
        "type", "decoder", "wrong", "wrong%", "routed", "missing",
        "unvalidated"))
    print("-" * 74)
    for t in IDENTIFIER_TYPES:
        for name, acc in (("none", raw), ("naive-snap", nai),
                          ("constrained", con)):
            o = acc[t]
            print("%-11s %-12s %9d %9.3f %8d %8d %11d" % (
                t, name, o.committed_wrong, o.committed_error_rate, o.routed,
                o.missing, o.unvalidated))
    print()
    print("'wrong' = an incorrect ID written into the maintenance record.")


def _rank(results, index) -> str:
    def key(item):
        value = item[1][index]
        return value if index == 0 else value.rate
    ordered = sorted(results.items(), key=key)
    return " < ".join(name for name, _ in ordered)


def main() -> None:
    print(BAR)
    print("fieldvoice - entity error rate for spoken field reports")
    print(BAR)
    print("NOTE: there is no speech recogniser here. No audio, no weights, no")
    print("network. Hypotheses come from a seeded SIMULATED ASR backend behind")
    print("the AsrBackend interface. What is real, and what is tested, is the")
    print("evaluation: WER by Levenshtein alignment, and entity error rate over")
    print("typed fields extracted from the hypothesis.")
    headline()
    corpus_evaluation()
    constrained_section()
    print()
    print("Bottom line: WER says A. The maintenance system needs B.")


if __name__ == "__main__":
    main()
