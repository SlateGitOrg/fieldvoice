"""The simulated ASR backend: interface contract and error characteristics.

There is no recogniser to test.  What is testable, and what a real backend
would have to honour, is the interface: deterministic given a seed, order
independent, and producing the specific failure classes the evaluation is
designed to expose.
"""

import unittest
from dataclasses import replace

from src.asr import (
    CLEAN,
    DOMAIN_CONSTRAINED,
    NARRATIVE_STRONG,
    AsrBackend,
    NoiseProfile,
    SimulatedAsr,
)
from src.entities import PART_ID, QUANTITY, extract_entities, tokenize
from src.generator import make_corpus

REFERENCE = (
    "crew alpha inspected the gearbox GB-4471 at north ridge on 2024-03-12 "
    "torque was 47 nm and we applied fifteen mg of grease fault code E-212"
)


class InterfaceTest(unittest.TestCase):
    def test_abstract_backend_refuses_to_pretend(self):
        with self.assertRaises(NotImplementedError):
            AsrBackend().transcribe(REFERENCE)

    def test_clean_profile_is_a_perfect_recogniser(self):
        asr = SimulatedAsr(CLEAN, seed=1)
        self.assertEqual(asr.transcribe(REFERENCE).split(), tokenize(REFERENCE))

    def test_same_seed_gives_the_same_hypothesis(self):
        a = SimulatedAsr(NARRATIVE_STRONG, seed=3).transcribe(REFERENCE)
        b = SimulatedAsr(NARRATIVE_STRONG, seed=3).transcribe(REFERENCE)
        self.assertEqual(a, b)

    def test_different_seeds_give_different_hypotheses(self):
        corpus = [u.reference for u in make_corpus(40, seed=2)]
        first = SimulatedAsr(NARRATIVE_STRONG, seed=1).transcribe_batch(corpus)
        second = SimulatedAsr(NARRATIVE_STRONG, seed=2).transcribe_batch(corpus)
        self.assertNotEqual(first, second)

    def test_transcription_is_independent_of_batch_order(self):
        # Carrying rng state across utterances would make an utterance
        # hypothesis depend on what was transcribed before it, and every
        # measured number would silently depend on corpus ordering.
        corpus = [u.reference for u in make_corpus(20, seed=6)]
        asr = SimulatedAsr(NARRATIVE_STRONG, seed=5)
        forward = asr.transcribe_batch(corpus)
        backward = asr.transcribe_batch(list(reversed(corpus)))
        self.assertEqual(forward, list(reversed(backward)))


class ErrorCharacteristicsTest(unittest.TestCase):
    def test_spellout_destroys_the_identifier_field(self):
        profile = NoiseProfile("spellout", identifier_error_rate=1.0,
                               spellout_share=1.0)
        hyp = SimulatedAsr(profile, seed=0).transcribe(REFERENCE)
        self.assertIn("forty four", hyp)
        self.assertEqual([e for e in extract_entities(hyp) if e.etype == PART_ID], [])

    def test_unit_confusion_stays_acoustic_not_dimensional(self):
        profile = NoiseProfile("units", unit_error_rate=1.0)
        hyp = SimulatedAsr(profile, seed=0).transcribe("applied fifteen mg of grease")
        quantities = [e for e in extract_entities(hyp) if e.etype == QUANTITY]
        self.assertEqual(len(quantities), 1)
        self.assertEqual(quantities[0].unit, "ml")
        self.assertEqual(quantities[0].magnitude, 15.0)

    def test_magnitude_confusion_is_the_teen_tens_trap(self):
        profile = NoiseProfile("numbers", number_error_rate=1.0)
        hyp = SimulatedAsr(profile, seed=0).transcribe("applied fifteen mg")
        self.assertIn("fifty", hyp)

    def test_narrative_noise_leaves_every_field_intact(self):
        profile = NoiseProfile("narrative-only", narrative_error_rate=1.0)
        for utt in make_corpus(30, seed=8):
            hyp = SimulatedAsr(profile, seed=0).transcribe(utt.reference)
            got = [(e.etype, e.value) for e in extract_entities(hyp)]
            self.assertEqual(got, [(e.etype, e.value) for e in utt.gold])

    def test_profiles_differ_in_the_way_the_scenario_claims(self):
        corpus = make_corpus(60, seed=4)
        general = SimulatedAsr(NARRATIVE_STRONG, seed=4)
        domain = SimulatedAsr(DOMAIN_CONSTRAINED, seed=4)
        general_fields = _field_errors(corpus, general)
        domain_fields = _field_errors(corpus, domain)
        self.assertGreater(general_fields, 5 * domain_fields)

    def test_zero_rate_profile_never_touches_its_class(self):
        profile = replace(NARRATIVE_STRONG, identifier_error_rate=0.0,
                          number_error_rate=0.0, unit_error_rate=0.0,
                          date_error_rate=0.0, location_error_rate=0.0)
        asr = SimulatedAsr(profile, seed=0)
        for utt in make_corpus(20, seed=12):
            hyp = asr.transcribe(utt.reference)
            self.assertEqual(
                [(e.etype, e.value) for e in extract_entities(hyp)],
                [(e.etype, e.value) for e in utt.gold],
            )


def _field_errors(corpus, asr):
    from src.metrics import entity_error_report

    total = 0
    for utt in corpus:
        hyp = asr.transcribe(utt.reference)
        total += entity_error_report(utt.gold, extract_entities(hyp)).overall.errors
    return total


if __name__ == "__main__":
    unittest.main()
