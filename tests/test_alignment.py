"""Tests for the calibration-anomaly engine (V1.1).

Core invariants:
- "calibration anomaly", never "mismatch": the engine measures
  unusualness relative to reviewer history, nothing more.
- Leave-one-out: the current observation never feeds its own baseline.
- Sentiment is context, never a flag gate.
- Low history -> explicit insufficient_history, never an invented baseline.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import alignment
from src.alignment import calibrate
from src.schemas import Review


def R(reviewer_id, employee, score, text):
    return Review(reviewer_id=reviewer_id, employee=employee,
                  customer_score=score, review_text=text)


POSITIVE = "Alex was great and answered all my questions."
NEUTRAL = "The service was fine, nothing special."
NEGATIVE = "Terrible experience, rude and completely unhelpful."


def constant_reviewer(rid, score, n, text=POSITIVE):
    return [R(rid, "alex", score, text) for _ in range(n)]


class TestCalibration(unittest.TestCase):
    def test_policy_constants_are_explicit(self):
        self.assertEqual(alignment.MIN_HISTORY, 10)
        self.assertEqual(alignment.ANOMALY_THRESHOLD_ABS_Z, 2.0)

    def test_leave_one_out_excludes_current_observation(self):
        # 10 prior 9s + one 4: the 4's baseline must be exactly the ten 9s.
        reviews = constant_reviewer("r", 9, 10) + [R("r", "alex", 4, NEUTRAL)]
        by_score = {}
        for r in calibrate(reviews):
            by_score.setdefault(r.score, []).append(r)
        anomaly = by_score[4][0]
        self.assertEqual(anomaly.history_count, 10)
        self.assertEqual(anomaly.reviewer_historical_mean, 9.0)
        self.assertEqual(anomaly.baseline_status, "zero_variance_baseline")
        self.assertIsNone(anomaly.score_z)

    def test_insufficient_history_is_explicit_not_invented(self):
        reviews = constant_reviewer("r_new", 9, 5)
        for r in calibrate(reviews):
            self.assertEqual(r.baseline_status, "insufficient_history")
            self.assertEqual(r.calibration_status, "insufficient_history")
            self.assertFalse(r.flag)
            self.assertIsNone(r.score_z)
            self.assertIsNone(r.reviewer_historical_mean)

    def test_zero_variance_consistent(self):
        reviews = constant_reviewer("r", 9, 11)
        for r in calibrate(reviews):
            self.assertEqual(r.baseline_status, "zero_variance_baseline")
            self.assertEqual(r.calibration_status, "consistent_with_reviewer_history")
            self.assertFalse(r.flag)

    def test_zero_variance_deviation_is_anomaly(self):
        reviews = constant_reviewer("r", 9, 11) + [R("r", "maria", 5, POSITIVE)]
        anomalous = [r for r in calibrate(reviews) if r.score == 5][0]
        self.assertEqual(anomalous.baseline_status, "zero_variance_baseline")
        self.assertEqual(anomalous.calibration_status, "calibration_anomaly")
        self.assertTrue(anomalous.flag)
        self.assertIsNone(anomalous.score_z)

    def test_large_z_flags_even_with_neutral_text(self):
        # Sentiment must not gate the flag: neutral text + big deviation flags.
        reviews = [R("r", "alex", s, POSITIVE)
                   for s in (9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9)] + [
            R("r", "alex", 4, NEUTRAL)]
        anomalous = [r for r in calibrate(reviews) if r.score == 4][0]
        self.assertEqual(anomalous.sentiment_label, "neutral")
        self.assertTrue(abs(anomalous.score_z) >= 2.0)
        self.assertEqual(anomalous.calibration_status, "calibration_anomaly")
        self.assertTrue(anomalous.flag)

    def test_consistent_reviewer_not_flagged(self):
        reviews = [R("r", "alex", s, POSITIVE) for s in
                   (9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9)]
        for r in calibrate(reviews):
            self.assertEqual(r.calibration_status, "consistent_with_reviewer_history")
            self.assertFalse(r.flag)

    def test_no_mismatch_terminology(self):
        reviews = (constant_reviewer("a", 9, 11) + [R("a", "alex", 4, NEUTRAL)]
                   + constant_reviewer("b", 5, 3))
        for r in calibrate(reviews):
            for field in (r.baseline_status, r.calibration_status, r.reason):
                self.assertNotIn("mismatch", field.lower())

    def test_reason_never_prescribes_score(self):
        reviews = constant_reviewer("r", 9, 11) + [R("r", "alex", 4, NEUTRAL)]
        anomalous = [r for r in calibrate(reviews) if r.score == 4][0]
        self.assertIn("do not auto-adjust", anomalous.reason.lower())
        self.assertNotIn("should have", anomalous.reason.lower())


if __name__ == "__main__":
    unittest.main()
