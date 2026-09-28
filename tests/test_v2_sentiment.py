"""V2 tests: Comprehend provider integration without AWS.

Acceptance gates under test:
- All four confidence scores are stored; the probability vector is kept.
- Any failure -> sentiment_status="unavailable"; never a default label.
- Flag classification is identical regardless of Comprehend's label.
- insufficient_history and zero_variance_baseline behavior is preserved.
- Batching follows the API contract (<=25 docs per call, 5KB per doc).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alignment import calibrate
from src.schemas import Review
from src.sentiment import ComprehendProvider

POSITIVE = "Alex was great and answered all my questions."
NEUTRAL = "The service was fine, nothing special."


def R(reviewer_id, employee, score, text):
    return Review(reviewer_id=reviewer_id, employee=employee,
                  customer_score=score, review_text=text)


class FakeComprehendClient:
    """Scripted stand-in for boto3's comprehend client."""

    def __init__(self, handler):
        self._handler = handler
        self.calls: list[list[str]] = []

    def batch_detect_sentiment(self, TextList, LanguageCode):
        self.calls.append(list(TextList))
        return self._handler(list(TextList), LanguageCode)


def _result(i, label, p=0.1, n=0.1, neu=0.7, m=0.1):
    return {"Index": i, "Sentiment": label,
            "SentimentScore": {"Positive": p, "Negative": n,
                               "Neutral": neu, "Mixed": m}}


def _fixed_label_client(label):
    def handler(texts, lang):
        return {"ResultList": [_result(i, label) for i in range(len(texts))],
                "ErrorList": []}
    return FakeComprehendClient(handler)


def _planted_anomaly_reviews():
    # 11 history reviews ~9-10, then a neutral-text 4.
    reviews = [R("r", "alex", s, POSITIVE)
               for s in (9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9)]
    reviews.append(R("r", "alex", 4, NEUTRAL))
    return reviews


class TestComprehendProvider(unittest.TestCase):
    def test_maps_all_four_confidence_scores(self):
        client = _fixed_label_client("NEUTRAL")
        evs = ComprehendProvider(client=client).analyze_batch([POSITIVE, NEUTRAL])
        self.assertEqual(len(evs), 2)
        for ev in evs:
            self.assertEqual(ev.status, "available")
            self.assertEqual(ev.label, "NEUTRAL")
            self.assertEqual((ev.positive, ev.negative, ev.neutral, ev.mixed),
                             (0.1, 0.1, 0.7, 0.1))

    def test_partial_batch_failure_marks_only_failed_docs(self):
        def handler(texts, lang):
            return {"ResultList": [_result(0, "POSITIVE"), _result(2, "NEGATIVE")],
                    "ErrorList": [{"Index": 1, "ErrorCode": "InternalServerException",
                                   "ErrorMessage": "boom"}]}
        evs = ComprehendProvider(client=FakeComprehendClient(handler)).analyze_batch(
            [POSITIVE, NEUTRAL, POSITIVE])
        self.assertEqual(evs[0].status, "available")
        self.assertEqual(evs[1].status, "unavailable")
        self.assertIsNone(evs[1].label)
        self.assertEqual(evs[2].status, "available")

    def test_whole_batch_exception_never_defaults_to_neutral(self):
        def handler(texts, lang):
            raise RuntimeError("throttling")
        evs = ComprehendProvider(client=FakeComprehendClient(handler)).analyze_batch(
            [POSITIVE, NEUTRAL])
        for ev in evs:
            self.assertEqual(ev.status, "unavailable")
            self.assertIsNone(ev.label)  # no silent NEUTRAL default
            self.assertIn("RuntimeError", ev.detail)

    def test_oversize_input_is_unavailable(self):
        big = "x" * 5001
        evs = ComprehendProvider(client=_fixed_label_client("NEUTRAL")).analyze_batch([big])
        self.assertEqual(evs[0].status, "unavailable")
        self.assertIn("5KB", evs[0].detail)

    def test_missing_from_response_is_unavailable(self):
        def handler(texts, lang):
            return {"ResultList": [], "ErrorList": []}
        evs = ComprehendProvider(client=FakeComprehendClient(handler)).analyze_batch([POSITIVE])
        self.assertEqual(evs[0].status, "unavailable")

    def test_batches_of_at_most_25(self):
        client = _fixed_label_client("NEUTRAL")
        ComprehendProvider(client=client).analyze_batch([POSITIVE] * 60)
        self.assertEqual([len(c) for c in client.calls], [25, 25, 10])


class TestFlagIndependence(unittest.TestCase):
    def test_anomaly_flag_identical_across_comprehend_labels(self):
        expected_z = None
        for label in ["NEUTRAL", "POSITIVE", "NEGATIVE", "MIXED"]:
            with self.subTest(label=label):
                results = calibrate(_planted_anomaly_reviews(),
                                    provider=ComprehendProvider(
                                        client=_fixed_label_client(label)))
                anomalous = [r for r in results if r.score == 4][0]
                self.assertEqual(anomalous.sentiment_label, label)
                self.assertEqual(anomalous.calibration_status, "calibration_anomaly")
                self.assertTrue(anomalous.flag)
                if expected_z is None:
                    expected_z = anomalous.score_z
                self.assertEqual(anomalous.score_z, expected_z)

    def test_anomaly_flag_identical_when_sentiment_unavailable(self):
        def handler(texts, lang):
            raise RuntimeError("outage")
        results = calibrate(_planted_anomaly_reviews(),
                            provider=ComprehendProvider(
                                client=FakeComprehendClient(handler)))
        anomalous = [r for r in results if r.score == 4][0]
        self.assertEqual(anomalous.sentiment_status, "unavailable")
        self.assertIsNone(anomalous.sentiment_label)
        self.assertEqual(anomalous.calibration_status, "calibration_anomaly")
        self.assertTrue(anomalous.flag)
        self.assertIn("reviewer history only", anomalous.reason)

    def test_insufficient_history_preserved_with_comprehend(self):
        reviews = [R("r_new", "alex", 9, POSITIVE) for _ in range(5)]
        results = calibrate(reviews, provider=ComprehendProvider(
            client=_fixed_label_client("POSITIVE")))
        for r in results:
            self.assertEqual(r.baseline_status, "insufficient_history")
            self.assertEqual(r.calibration_status, "insufficient_history")
            self.assertFalse(r.flag)

    def test_zero_variance_preserved_with_comprehend(self):
        reviews = [R("r", "alex", 9, POSITIVE) for _ in range(11)]
        reviews.append(R("r", "maria", 5, POSITIVE))
        results = calibrate(reviews, provider=ComprehendProvider(
            client=_fixed_label_client("POSITIVE")))
        anomalous = [r for r in results if r.score == 5][0]
        self.assertEqual(anomalous.baseline_status, "zero_variance_baseline")
        self.assertEqual(anomalous.calibration_status, "calibration_anomaly")
        self.assertTrue(anomalous.flag)


if __name__ == "__main__":
    unittest.main()
