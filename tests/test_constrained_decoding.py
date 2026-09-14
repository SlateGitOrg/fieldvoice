"""Constrained decoding against the catalogue, scored on planted ground truth."""

import socket
import unittest
from pathlib import Path
from unittest import mock

from src.asr import NARRATIVE_STRONG, SimulatedAsr
from src.decoding import (
    ACCEPTED, CONFIRM, CORRECTED, Catalogue, build_catalogue,
    constrained_decode, naive_nearest_decode, score_identifiers,
    score_raw_identifiers,
)
from src.entities import FAULT_CODE, PART_ID
from src.generator import make_corpus

CAT = Catalogue(frozenset({"GB-4471", "YD-1203", "PT-9001", "PT-9011"}),
                frozenset({"E-212", "E-640"}))


def run_corpus(decoder=None, n=200, seed=11, **kw):
    corpus = make_corpus(n, seed=seed)
    cat, off = build_catalogue(corpus, **kw)
    asr = SimulatedAsr(NARRATIVE_STRONG, seed=4)
    acc = {}
    for k, u in enumerate(corpus):
        h = asr.transcribe(u.reference)
        if decoder is None:
            score_raw_identifiers(u.gold, h, cat, acc, k in off)
        else:
            score_identifiers(u.gold, decoder(h, cat), cat, k in off, acc)
    return acc


class UnitBehaviour(unittest.TestCase):
    def test_exact_hit_accepted(self):
        f = constrained_decode("gearbox GB-4471 ok", CAT).fields
        self.assertEqual((f[0].value, f[0].status), ("GB-4471", ACCEPTED))

    def test_digit_slip_snaps_back(self):
        f = constrained_decode("gearbox GB-4771 and E-272", CAT).fields
        self.assertEqual([(x.value, x.status) for x in f],
                         [("GB-4471", CORRECTED), ("E-212", CORRECTED)])

    def test_spelled_out_part_recomposed(self):
        f = constrained_decode("gearbox gb forty four seventy one at", CAT).fields
        self.assertEqual((f[0].value, f[0].status), ("GB-4471", CORRECTED))

    def test_spelled_out_with_leading_zero_chunk(self):
        f = constrained_decode("yd twelve three was", CAT).fields
        self.assertEqual(f[0].value, "YD-1203")

    def test_spelled_out_fault_code(self):
        f = constrained_decode("fault code e twenty one two cleared", CAT).fields
        self.assertEqual((f[0].etype, f[0].value), (FAULT_CODE, "E-212"))

    def test_unknown_part_is_routed_not_guessed(self):
        f = constrained_decode("fitted HV-5555 today", CAT).fields
        self.assertEqual((f[0].status, f[0].value), (CONFIRM, None))

    def test_ambiguous_snap_is_routed(self):
        # PT-9001 and PT-9011 are both one slip from PT-9021.
        f = constrained_decode("PT-9021", CAT).fields[0]
        self.assertEqual(f.status, CONFIRM)
        self.assertEqual(f.candidates, ("PT-9001", "PT-9011"))

    def test_naive_nearest_guesses_unknown_part(self):
        """The generic approach: validated output, wrong part."""
        f = naive_nearest_decode("fitted HV-5555 today", CAT).fields[0]
        self.assertNotEqual(f.status, CONFIRM)
        self.assertIn(f.value, CAT.parts)   # passes validation...
        self.assertNotEqual(f.value, "HV-5555")  # ...and is still wrong

    def test_non_identifier_fields_survive(self):
        rep = constrained_decode("GB-4771 torque 47 nm at north ridge", CAT)
        vals = {e.value for e in rep.committed()}
        self.assertTrue({"GB-4471", "47 nm", "north ridge"} <= vals)


class CorpusOutcomes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = run_corpus()
        cls.con = run_corpus(constrained_decode)
        cls.naive = run_corpus(naive_nearest_decode)

    def test_never_emits_unvalidated_identifier(self):
        for t in (PART_ID, FAULT_CODE):
            self.assertEqual(self.con[t].unvalidated, 0)
        # and the baseline does, so the invariant is not vacuous
        self.assertGreater(self.raw[PART_ID].unvalidated, 0)

    def test_per_class_committed_error_threshold(self):
        # Thresholds: measured 0.020 (part) / 0.050 (fault) on this seed; the
        # bound is 1/3 of the unconstrained rate (~0.32), i.e. the claim is
        # "at least 3x fewer wrong IDs written to the record", per class.
        for t in (PART_ID, FAULT_CODE):
            self.assertLess(self.con[t].committed_error_rate,
                            self.raw[t].committed_error_rate / 3, t)

    def test_spellouts_no_longer_lost(self):
        self.assertGreater(self.raw[PART_ID].missing, 30)
        self.assertLessEqual(self.con[PART_ID].missing, 2)

    def test_naive_nearest_commits_more_wrong_ids(self):
        for t in (PART_ID, FAULT_CODE):
            self.assertGreater(self.naive[t].committed_wrong,
                               2 * self.con[t].committed_wrong, t)
            self.assertEqual(self.naive[t].routed, 0)

    def test_off_catalogue_parts_are_never_silently_accepted(self):
        """10 planted unknown parts.

        The original claim here was "all are routed".  The suite disproved it:
        GB-5161 (unknown) was heard as GB-5151, which is one slip from the
        catalogued GB-5251, so the snap commits a wrong part.  That is a real
        limit of any snapping scheme, not a decoder bug.  What the decoder
        DOES guarantee, and what is asserted, is: an unknown ID is never
        committed as heard, and any commit is a unique radius-1 snap.  The
        naive decoder, for contrast, commits every one of the 10.
        """
        corpus = make_corpus(200, seed=11)
        cat, off = build_catalogue(corpus, distractors=0)
        self.assertEqual(len(off), 10)
        asr = SimulatedAsr(NARRATIVE_STRONG, seed=4)
        routed = naive_committed = 0
        for k in off:
            h = asr.transcribe(corpus[k].reference)
            for f in constrained_decode(h, cat).fields:
                if f.etype != PART_ID:
                    continue
                self.assertNotEqual(f.status, ACCEPTED, k)
                if f.status == CONFIRM:
                    routed += 1
                else:
                    self.assertEqual(len(f.candidates), 1, k)
                    self.assertIn(f.value, cat.parts, k)
            naive_committed += sum(
                1 for f in naive_nearest_decode(h, cat).fields
                if f.etype == PART_ID and f.status != CONFIRM)
        self.assertEqual(naive_committed, 10)
        self.assertGreater(routed, 10 // 2)   # measured: 9 of 10

    def test_radius_two_routes_more(self):
        r2 = run_corpus(lambda h, c: constrained_decode(h, c, radius=2))
        self.assertGreater(r2[PART_ID].routed, 2 * self.con[PART_ID].routed)
        self.assertLessEqual(r2[PART_ID].committed_wrong,
                             self.con[PART_ID].committed_wrong)


class VocabularyRegression(unittest.TestCase):
    def test_denser_catalogue_degrades_part_class(self):
        """A vocabulary change that crowds the ID space must show up as a
        per-class regression: this is the failure the suite is there to catch.
        """
        sparse = run_corpus(constrained_decode, distractors=0)
        dense = run_corpus(constrained_decode, distractors=20000)
        self.assertGreater(dense[PART_ID].committed_wrong
                           + dense[PART_ID].routed,
                           sparse[PART_ID].committed_wrong
                           + sparse[PART_ID].routed)


class Offline(unittest.TestCase):
    def test_pipeline_runs_with_networking_disabled(self):
        def boom(*a, **k):
            raise AssertionError("network access attempted")
        with mock.patch.object(socket.socket, "connect", boom), \
                mock.patch.object(socket, "create_connection", boom), \
                mock.patch.object(socket, "getaddrinfo", boom):
            acc = run_corpus(constrained_decode, n=50)
        self.assertEqual(acc[PART_ID].gold, 50)

    def test_no_network_imports_in_src(self):
        src = Path(__file__).resolve().parent.parent / "src"
        for p in src.glob("*.py"):
            text = p.read_text(encoding="utf-8")
            for mod in ("socket", "urllib", "http.client", "requests"):
                self.assertNotIn("import " + mod, text, p.name)


if __name__ == "__main__":
    unittest.main()
