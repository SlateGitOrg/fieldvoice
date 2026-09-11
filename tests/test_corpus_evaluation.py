"""Corpus-level evaluation: the flip is not a single cherry-picked sentence,
and the honest boundary of the claim.
"""

import unittest
from dataclasses import replace

from src.asr import DOMAIN_CONSTRAINED, NARRATIVE_STRONG, SimulatedAsr
from src.entities import ENTITY_TYPES, PART_ID, extract_entities
from src.generator import make_corpus
from src.metrics import ErrorCounts, align_words, entity_error_report

CORPUS = make_corpus(200)
SEED = 4


def evaluate(profile, corpus=CORPUS, seed=SEED):
    asr = SimulatedAsr(profile, seed=seed)
    word_errors = word_total = 0
    per_type = {t: ErrorCounts() for t in ENTITY_TYPES}
    overall = ErrorCounts()
    for utt in corpus:
        hyp = asr.transcribe(utt.reference)
        counts = align_words(utt.reference, hyp).counts
        word_errors += counts.errors
        word_total += counts.reference_count
        report = entity_error_report(utt.gold, extract_entities(hyp))
        for t in ENTITY_TYPES:
            acc, c = per_type[t], report.per_type[t]
            acc.hits += c.hits
            acc.substitutions += c.substitutions
            acc.insertions += c.insertions
            acc.deletions += c.deletions
        overall.hits += report.overall.hits
        overall.substitutions += report.overall.substitutions
        overall.insertions += report.overall.insertions
        overall.deletions += report.overall.deletions
    return word_errors / word_total, overall, per_type


class CorpusFlipTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wer_a, cls.eer_a, cls.types_a = evaluate(NARRATIVE_STRONG)
        cls.wer_b, cls.eer_b, cls.types_b = evaluate(DOMAIN_CONSTRAINED)

    def test_wer_ranks_the_general_recogniser_first(self):
        self.assertLess(self.wer_a, self.wer_b)

    def test_entity_error_rate_ranks_the_domain_recogniser_first(self):
        self.assertLess(self.eer_b.rate, self.eer_a.rate)

    def test_the_entity_gap_is_an_order_of_magnitude(self):
        # Not a chosen threshold: the two profiles differ by roughly 10x in
        # per-class field error propensity by construction, so anything less
        # than a 5x measured gap would mean the metric is not tracking them.
        self.assertGreater(self.eer_a.rate, 5 * self.eer_b.rate)

    def test_per_type_rates_are_reported_and_differ(self):
        rates = {t: self.types_a[t].rate for t in ENTITY_TYPES}
        self.assertGreater(max(rates.values()) - min(rates.values()), 0.1)
        self.assertGreater(rates[PART_ID], self.eer_a.rate * 0.9)

    def test_every_type_has_gold_instances(self):
        for t in ENTITY_TYPES:
            self.assertGreater(self.types_a[t].reference_count, 0)

    def test_evaluation_is_reproducible(self):
        again = evaluate(NARRATIVE_STRONG)
        self.assertEqual(again[0], self.wer_a)
        self.assertEqual(again[1].errors, self.eer_a.errors)


class ClaimBoundaryTest(unittest.TestCase):
    """Where the corpus-level claim stops being true.

    The flip at corpus level depends on the domain system being noisy enough
    on narrative.  Quieten it and it simply wins on both metrics -- which is a
    better outcome operationally and a weaker argument rhetorically.  Saying so
    here is cheaper than being caught claiming a universal result.
    """

    def test_flip_needs_narrative_noise(self):
        quiet = replace(DOMAIN_CONSTRAINED, narrative_error_rate=0.22)
        wer_a, eer_a, _ = evaluate(NARRATIVE_STRONG)
        wer_quiet, eer_quiet, _ = evaluate(quiet)
        self.assertLess(wer_quiet, wer_a)
        self.assertLess(eer_quiet.rate, eer_a.rate)

    def test_entity_ranking_is_invariant_to_narrative_noise(self):
        # The whole point: the metric that matters does not move when the
        # narrative channel does.
        _, eer_a, _ = evaluate(NARRATIVE_STRONG)
        rates = []
        for noise in (0.05, 0.22, 0.30, 0.45):
            _, eer, _ = evaluate(replace(DOMAIN_CONSTRAINED,
                                         narrative_error_rate=noise))
            rates.append(eer.rate)
            self.assertLess(eer.rate, eer_a.rate)
        self.assertLess(max(rates) - min(rates), 0.02)


if __name__ == "__main__":
    unittest.main()
