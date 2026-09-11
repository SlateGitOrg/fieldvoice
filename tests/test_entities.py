"""Typed field extraction, and the quantity semantics that make it useful."""

import unittest

from src.entities import (
    DATE,
    FAULT_CODE,
    LOCATION,
    PART_ID,
    QUANTITY,
    extract_entities,
    parse_number,
)
from src.generator import make_corpus
from src.metrics import entity_error_report


def only(entities, etype):
    return [e for e in entities if e.etype == etype]


class ExtractionTest(unittest.TestCase):
    def test_recovers_every_planted_entity_on_a_corpus(self):
        # Gold comes from the structured record, not from the extractor, so
        # this is a real round-trip and not a tautology.
        for utt in make_corpus(120, seed=5):
            got = extract_entities(utt.reference)
            self.assertEqual(
                [(e.etype, e.value) for e in got],
                [(e.etype, e.value) for e in utt.gold],
                msg=utt.reference,
            )

    def test_entity_positions_are_recovered_too(self):
        for utt in make_corpus(40, seed=9):
            got = extract_entities(utt.reference)
            self.assertEqual([e.pos for e in got], [e.pos for e in utt.gold])

    def test_part_and_fault_shapes_are_distinguished(self):
        ents = extract_entities("part GB-4471 raised fault E-212")
        self.assertEqual([e.value for e in only(ents, PART_ID)], ["GB-4471"])
        self.assertEqual([e.value for e in only(ents, FAULT_CODE)], ["E-212"])

    def test_multiword_location_is_not_split(self):
        ents = only(extract_entities("we met at north ridge"), LOCATION)
        self.assertEqual([e.value for e in ents], ["north ridge"])

    def test_unknown_place_name_is_not_a_location(self):
        # A closed gazetteer is the point: inventing locations would put
        # unvalidated place names into a maintenance record.
        self.assertEqual(only(extract_entities("we met at bracken hill"), LOCATION), [])

    def test_number_without_unit_is_not_a_quantity(self):
        self.assertEqual(only(extract_entities("crew of 47 arrived"), QUANTITY), [])

    def test_date_is_normalised(self):
        ents = only(extract_entities("logged on 2024-03-05"), DATE)
        self.assertEqual([e.value for e in ents], ["2024-03-05"])


class QuantitySemanticsTest(unittest.TestCase):
    """A quantity is correct only if magnitude AND unit are both right."""

    def gold(self, text):
        return extract_entities(text)

    def rate(self, ref_text, hyp_text):
        return entity_error_report(
            self.gold(ref_text), extract_entities(hyp_text)
        ).type_rate(QUANTITY)

    def test_spoken_and_digit_forms_are_the_same_reading(self):
        self.assertEqual(parse_number("fifteen"), 15.0)
        self.assertAlmostEqual(self.rate("applied fifteen mg", "applied 15 mg"), 0.0)

    def test_fifteen_heard_as_fifty_is_an_error(self):
        self.assertAlmostEqual(self.rate("applied fifteen mg", "applied fifty mg"), 1.0)

    def test_right_magnitude_wrong_unit_is_an_error(self):
        self.assertAlmostEqual(self.rate("applied 15 mg", "applied 15 ml"), 1.0)

    def test_both_wrong_is_one_error_not_two(self):
        report = entity_error_report(
            self.gold("applied 15 mg"), extract_entities("applied 50 ml")
        )
        counts = report.per_type[QUANTITY]
        self.assertEqual(counts.substitutions, 1)
        self.assertEqual(counts.errors, 1)

    def test_orthographic_unit_variants_are_the_same_unit(self):
        self.assertAlmostEqual(
            self.rate("applied 250 millilitres", "applied 250 ml"), 0.0
        )

    def test_magnitude_only_comparison_would_miss_the_unit_slip(self):
        # The naive alternative: score the number and ignore the unit.  It
        # calls 15 mg -> 15 ml a perfect transcription.  This is the half of
        # the argument that shows the obvious implementation is wrong.
        gold = self.gold("applied 15 mg")
        hyp = extract_entities("applied 15 ml")
        naive_errors = sum(
            1 for g, h in zip(gold, hyp) if g.magnitude != h.magnitude
        )
        self.assertEqual(naive_errors, 0)
        self.assertAlmostEqual(self.rate("applied 15 mg", "applied 15 ml"), 1.0)


if __name__ == "__main__":
    unittest.main()
