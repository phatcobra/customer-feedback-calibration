"""Tests for the rule-based sentiment baseline."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment import classify


class TestSentiment(unittest.TestCase):
    def test_positive(self):
        r = classify("Alex was great and answered all my questions.")
        self.assertEqual(r.sentiment, "positive")
        self.assertGreater(r.positive_hits, 0)

    def test_strongly_positive(self):
        r = classify("Excellent, perfect and amazing service throughout.")
        self.assertEqual(r.sentiment, "strongly_positive")

    def test_negative(self):
        r = classify("Terrible experience, rude and completely unhelpful.")
        self.assertEqual(r.sentiment, "negative")
        self.assertGreater(r.negative_hits, 0)

    def test_mixed(self):
        r = classify(
            "He was polite, but I had to wait too long and my issue was not resolved."
        )
        self.assertEqual(r.sentiment, "mixed")

    def test_neutral(self):
        r = classify("Called about my account balance.")
        self.assertEqual(r.sentiment, "neutral")
        self.assertEqual(r.confidence, 0.0)

    def test_raw_counts_preserved(self):
        # The winning label must never hide the raw evidence.
        r = classify("Alex was great and answered all my questions.")
        self.assertEqual(r.positive_hits, 2)
        self.assertEqual(r.negative_hits, 0)
        self.assertAlmostEqual(r.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
