"""Opt-in integration test against the real Comprehend API.

Run with: RUN_AWS_INTEGRATION_TESTS=1 python3 -m unittest tests.test_comprehend_integration
Requires AWS credentials with comprehend:BatchDetectSentiment permission.
Never runs as part of the default suite: AWS availability must not be
required to test the policy engine.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment import ComprehendProvider

RUN = os.environ.get("RUN_AWS_INTEGRATION_TESTS") == "1"


@unittest.skipUnless(RUN, "opt-in: needs AWS credentials (RUN_AWS_INTEGRATION_TESTS=1)")
class TestComprehendIntegration(unittest.TestCase):
    def test_real_api_returns_full_probability_vector(self):
        provider = ComprehendProvider(
            region=os.environ.get("AWS_REGION", "us-east-1"))
        texts = ["The service was great and quick.",
                 "Terrible experience, never again.",
                 "It was fine, nothing special."]
        evs = provider.analyze_batch(texts)
        self.assertEqual(len(evs), 3)
        for ev in evs:
            self.assertEqual(ev.status, "available")
            self.assertIn(ev.label, ("POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"))
            total = ev.positive + ev.negative + ev.neutral + ev.mixed
            self.assertAlmostEqual(total, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
