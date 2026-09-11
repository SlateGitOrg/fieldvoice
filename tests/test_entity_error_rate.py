"""Entity error rate: alignment, shift tolerance, and per-type reporting."""

import unittest

from src.entities import (
    DATE,
    FAULT_CODE,
    LOCATION,
    PART_ID,
    QUANTITY,
    Entity,
    extract_entities,
)
from src.metrics import entity_error_report, word_error_rate

REFERENCE = (
    "inspected gearbox GB-4471 at north ridge on 2024-03-12 torque was 47 nm "
    "fault code E-212"
)


def gold():
    return extract_entities(REFERENCE)


class AlignmentTest(unittest.TestCase):
    def test_perfect_transcript_scores_zero(self):
        self.assertEqual(entity_error_report(gold(), gold()).rate, 0.0)

    def test_shifted_entity_still_matches(self):
        # Narrative words inserted ahead of the fields push every entity to a
        # later token index.  Position-based matching would score this as a
        # total loss; order-based alignment scores it as what it is: perfect.
        shifted = "well so er " + REFERENCE
        hyp_entities = extract_entities(shifted)
        report = entity_error_report(gold(), hyp_entities)
        self.assertGreater(word_error_rate(REFERENCE, shifted), 0.0)
        self.assertEqual(report.rate, 0.0)
        self.assertNotEqual(
            [e.pos for e in hyp_entities], [e.pos for e in gold()]
        )

    def test_missing_entity_counts_as_deletion(self):
        hyp = extract_entities(REFERENCE.replace("GB-4471", "gb forty four seventy one"))
        report = entity_error_report(gold(), hyp)
        self.assertEqual(report.per_type[PART_ID].deletions, 1)
        self.assertEqual(report.per_type[PART_ID].substitutions, 0)
        self.assertAlmostEqual(report.type_rate(PART_ID), 1.0)

    def test_hallucinated_entity_counts_as_insertion(self):
        hyp = extract_entities(REFERENCE + " and spare YD-9001")
        report = entity_error_report(gold(), hyp)
        self.assertEqual(report.per_type[PART_ID].insertions, 1)
        self.assertGreater(report.rate, 0.0)

    def test_types_are_scored_separately(self):
        # A date must never be able to "substitute" for a part ID; if the
        # alignment ran over a single mixed list it could, and a system that
        # dropped every ID would be credited with matching something.
        hyp = extract_entities(REFERENCE.replace("GB-4471", "2024-01-01"))
        report = entity_error_report(gold(), hyp)
        self.assertEqual(report.per_type[PART_ID].deletions, 1)
        self.assertEqual(report.per_type[DATE].insertions, 1)
        self.assertEqual(report.per_type[DATE].substitutions, 0)

    def test_swapped_same_type_entities_are_two_errors(self):
        ref = "applied 250 ml and 47 nm"
        hyp = "applied 47 nm and 250 ml"
        report = entity_error_report(extract_entities(ref), extract_entities(hyp))
        self.assertEqual(report.per_type[QUANTITY].substitutions, 2)


class PerTypeReportingTest(unittest.TestCase):
    def test_average_hides_a_catastrophic_class(self):
        # Every ID wrong, everything else perfect.  The headline number looks
        # like one field in six; the part-number channel is unusable.
        hyp = extract_entities(REFERENCE.replace("GB-4471", "GB-4999"))
        report = entity_error_report(gold(), hyp)
        self.assertAlmostEqual(report.type_rate(PART_ID), 1.0)
        self.assertAlmostEqual(report.type_rate(DATE), 0.0)
        self.assertLess(report.rate, 0.35)
        self.assertEqual(report.worst_type(), PART_ID)

    def test_every_type_is_reported_even_when_absent(self):
        report = entity_error_report(gold(), gold())
        for etype in (PART_ID, FAULT_CODE, QUANTITY, DATE, LOCATION):
            self.assertIn(etype, report.per_type)

    def test_absent_type_has_zero_reference_count(self):
        ref = extract_entities("applied 250 ml")
        report = entity_error_report(ref, extract_entities("applied 250 ml"))
        self.assertEqual(report.per_type[PART_ID].reference_count, 0)
        self.assertEqual(report.per_type[PART_ID].rate, 0.0)

    def test_insertions_against_an_empty_reference_are_not_free(self):
        # Scoring 0.0 here would let a hallucinating system look flawless.
        report = entity_error_report([], extract_entities("part GB-4471"))
        self.assertGreater(report.per_type[PART_ID].rate, 0.0)

    def test_worst_type_ignores_types_with_no_gold(self):
        report = entity_error_report(
            extract_entities("applied 250 ml"), extract_entities("applied 250 mg")
        )
        self.assertEqual(report.worst_type(), QUANTITY)


if __name__ == "__main__":
    unittest.main()
