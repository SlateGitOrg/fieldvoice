"""Word error rate: does the Levenshtein alignment actually do the arithmetic."""

import random
import unittest

from src.metrics import DEL, INS, MATCH, SUB, align_words, word_error_rate


def brute_force_distance(a, b):
    """Independent reference implementation, recursive with memoisation.

    Deliberately not the same algorithm shape as the DP table under test, so
    an off-by-one in the table cannot be mirrored here.
    """
    memo = {}

    def go(i, j):
        if i == 0:
            return j
        if j == 0:
            return i
        key = (i, j)
        if key in memo:
            return memo[key]
        cost = 0 if a[i - 1] == b[j - 1] else 1
        best = min(go(i - 1, j - 1) + cost, go(i - 1, j) + 1, go(i, j - 1) + 1)
        memo[key] = best
        return best

    return go(len(a), len(b))


class WordErrorRateTest(unittest.TestCase):
    def test_identical_transcripts_score_zero(self):
        self.assertEqual(word_error_rate("a b c d", "a b c d"), 0.0)

    def test_single_substitution(self):
        self.assertAlmostEqual(word_error_rate("a b c d", "a x c d"), 0.25)

    def test_single_deletion(self):
        self.assertAlmostEqual(word_error_rate("a b c d", "a c d"), 0.25)

    def test_single_insertion(self):
        self.assertAlmostEqual(word_error_rate("a b c d", "a b z c d"), 0.25)

    def test_empty_hypothesis_is_total_loss(self):
        self.assertAlmostEqual(word_error_rate("a b c d", ""), 1.0)

    def test_operation_counts_are_itemised(self):
        counts = align_words("a b c d", "a x c d z").counts
        self.assertEqual(counts.substitutions, 1)
        self.assertEqual(counts.insertions, 1)
        self.assertEqual(counts.deletions, 0)
        self.assertEqual(counts.hits, 3)
        self.assertEqual(counts.reference_count, 4)

    def test_substitution_preferred_over_delete_plus_insert(self):
        # Same edit distance either way; reporting it as del+ins would count
        # one mistake twice and inflate WER above the edit distance.
        al = align_words("a b c", "a x c")
        ops = [op for op, _, _ in al.ops]
        self.assertEqual(ops, [MATCH, SUB, MATCH])
        self.assertNotIn(DEL, ops)
        self.assertNotIn(INS, ops)

    def test_wer_can_exceed_one_with_many_insertions(self):
        # A real property of WER that a clamped implementation would hide.
        self.assertGreater(word_error_rate("a", "a b c d e"), 1.0)

    def test_alignment_matches_independent_edit_distance(self):
        rng = random.Random(3)
        vocab = "abcdefg"
        for _ in range(200):
            ref = [rng.choice(vocab) for _ in range(rng.randint(0, 9))]
            hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 9))]
            counts = align_words(ref, hyp).counts
            self.assertEqual(
                counts.errors,
                brute_force_distance(ref, hyp),
                msg="ref=%r hyp=%r" % (ref, hyp),
            )

    def test_backtrace_reconstructs_the_hypothesis(self):
        # If the op list did not actually describe the edit, the demo would be
        # naming the wrong words as the cause of the divergence.
        rng = random.Random(7)
        vocab = "abcd"
        for _ in range(100):
            ref = [rng.choice(vocab) for _ in range(rng.randint(0, 8))]
            hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 8))]
            al = align_words(ref, hyp)
            rebuilt = [h for op, _, h in al.ops if op != DEL]
            self.assertEqual(rebuilt, hyp)

    def test_tokenizer_ignores_case_and_edge_punctuation(self):
        self.assertEqual(word_error_rate("Torque was 47 nm.", "torque was 47 nm"), 0.0)


if __name__ == "__main__":
    unittest.main()
