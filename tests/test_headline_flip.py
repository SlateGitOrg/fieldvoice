"""THE HEADLINE TEST: WER prefers A, entity error rate prefers B.

Two transcripts of the same utterance.  System A makes fewer word errors and
every one of them lands on a field that gets written into the maintenance
record.  System B makes more word errors and every one of them lands on
narrative.  WER ranks A first.  The report is only fileable from B.
"""

import unittest

from src.entities import QUANTITY, extract_entities
from src.generator import headline_case
from src.metrics import classify_word_errors, entity_error_report, word_error_rate


class HeadlineFlipTest(unittest.TestCase):
    def setUp(self):
        self.reference, self.gold, self.hyp_a, self.hyp_b = headline_case()
        self.wer_a = word_error_rate(self.reference, self.hyp_a)
        self.wer_b = word_error_rate(self.reference, self.hyp_b)
        self.eer_a = entity_error_report(self.gold, extract_entities(self.hyp_a))
        self.eer_b = entity_error_report(self.gold, extract_entities(self.hyp_b))

    def test_word_error_rate_prefers_system_a(self):
        self.assertLess(self.wer_a, self.wer_b)

    def test_entity_error_rate_prefers_system_b(self):
        self.assertLess(self.eer_b.rate, self.eer_a.rate)

    def test_the_ranking_actually_flips(self):
        wer_winner = "A" if self.wer_a < self.wer_b else "B"
        eer_winner = "A" if self.eer_a.rate < self.eer_b.rate else "B"
        self.assertNotEqual(wer_winner, eer_winner)

    def test_system_b_files_a_perfect_report(self):
        self.assertEqual(self.eer_b.rate, 0.0)

    def test_system_a_loses_most_of_the_fields(self):
        # Five of six fields wrong: ID, date, both quantities, fault code.
        self.assertEqual(self.eer_a.overall.errors, 5)
        self.assertEqual(self.eer_a.overall.reference_count, 6)

    def test_divergence_is_attributable_to_specific_words(self):
        ent_a, narr_a = classify_word_errors(self.reference, self.hyp_a, self.gold)
        ent_b, narr_b = classify_word_errors(self.reference, self.hyp_b, self.gold)
        self.assertEqual(narr_a, [])
        self.assertEqual(ent_b, [])
        self.assertEqual(
            [(r, h) for _, r, h in ent_a],
            [
                ("gb-4471", "gb-4771"),
                ("2024-03-12", "2024-03-21"),
                ("nm", "mm"),
                ("ml", "mg"),
                ("e-212", "e-272"),
            ],
        )
        self.assertEqual(
            [r for _, r, _ in narr_b],
            ["inspected", "was", "and", "we", "of", "the"],
        )

    def test_a_has_strictly_fewer_word_errors(self):
        ent_a, narr_a = classify_word_errors(self.reference, self.hyp_a, self.gold)
        ent_b, narr_b = classify_word_errors(self.reference, self.hyp_b, self.gold)
        self.assertEqual(len(ent_a) + len(narr_a), 5)
        self.assertEqual(len(ent_b) + len(narr_b), 6)

    def test_unit_slips_are_the_quiet_half_of_the_damage(self):
        # Two of A's five field errors are unit-only: same magnitude, wrong
        # dimension.  A magnitude-only check would call them correct.
        self.assertEqual(self.eer_a.per_type[QUANTITY].substitutions, 2)
        hyp = extract_entities(self.hyp_a)
        quantities = [e for e in hyp if e.etype == QUANTITY]
        gold_q = [e for e in self.gold if e.etype == QUANTITY]
        self.assertEqual(
            [g.magnitude for g in gold_q], [h.magnitude for h in quantities]
        )
        self.assertNotEqual([g.unit for g in gold_q], [h.unit for h in quantities])


if __name__ == "__main__":
    unittest.main()
