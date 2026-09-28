"""Production-contract tests: explicit calibration_method and require_timestamps.

The missing-timestamp fallback is a documented legacy path, not a silent
degradation. Every result carries calibration_method explicitly, and a real
deployment can require timestamps: timestamp-less records are then refused
(missing_timestamp) instead of falling back to leave-one-out -- the fallback
would silently reintroduce the temporal-leakage limitation V3 fixed.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alignment import calibrate  # noqa: E402
from src.schemas import Review  # noqa: E402

POSITIVE = "Alex was great and answered all my questions."
NEGATIVE = "Terrible experience, nobody helped and nothing was resolved."
T0 = datetime(2026, 1, 1)


def R(reviewer_id, employee, score, text, timestamp):
    return Review(reviewer_id=reviewer_id, employee=employee,
                  customer_score=score, review_text=text, timestamp=timestamp)


def ts(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _history_10(reviewer="r"):
    return [R(reviewer, "alex", 9, POSITIVE, ts(T0 + timedelta(days=d)))
            for d in range(10)]


class TestCalibrationMethodExplicit(unittest.TestCase):
    def test_timestamped_uses_point_in_time(self):
        reviews = _history_10() + [R("r", "alex", 4, NEGATIVE, ts(T0 + timedelta(days=14)))]
        target = calibrate(reviews)[-1]
        self.assertEqual(target.calibration_method, "point_in_time")
        self.assertEqual(target.calibration_status, "calibration_anomaly")
        self.assertNotIn("leave_one_out", target.reason)

    def test_timestamp_less_uses_legacy_and_says_so(self):
        reviews = [R("r", "alex", 9, POSITIVE, None) for _ in range(10)]
        reviews.append(R("r", "alex", 4, NEGATIVE, None))
        target = calibrate(reviews)[-1]
        self.assertEqual(target.calibration_method, "leave_one_out_legacy")
        self.assertEqual(target.calibration_status, "calibration_anomaly")
        # The legacy mode must be explicit in the human-readable reason, not silent.
        self.assertIn("leave_one_out_legacy", target.reason)

    def test_method_never_missing(self):
        reviews = _history_10() + [R("r", "alex", 9, POSITIVE, None)]
        for r in calibrate(reviews):
            self.assertIn(r.calibration_method, ("point_in_time", "leave_one_out_legacy"))


class TestRequireTimestamps(unittest.TestCase):
    def test_timestamp_less_refused_not_degraded(self):
        reviews = _history_10() + [R("r", "alex", 4, NEGATIVE, None)]
        target = calibrate(reviews, require_timestamps=True)[-1]
        self.assertEqual(target.calibration_status, "missing_timestamp")
        self.assertEqual(target.baseline_status, "missing_timestamp")
        self.assertFalse(target.flag)
        # No baseline was computed: refusal, not a quiet fallback.
        self.assertIsNone(target.reviewer_historical_mean)
        self.assertIsNone(target.reviewer_historical_stddev)
        self.assertIsNone(target.score_z)
        self.assertEqual(target.history_count, 0)
        # The refusal must name what it refused to do: no silent degradation.
        self.assertIn("excluded rather than degraded to the leave-one-out",
                      target.reason)
        self.assertIn("missing_timestamp", target.reason)

    def test_timestamped_unaffected_by_require_timestamps(self):
        reviews = _history_10() + [R("r", "alex", 4, NEGATIVE, ts(T0 + timedelta(days=14)))]
        default = calibrate(reviews)[-1]
        strict = calibrate(reviews, require_timestamps=True)[-1]
        self.assertEqual(strict.calibration_method, "point_in_time")
        self.assertEqual(strict.calibration_status, "calibration_anomaly")
        for field in ("reviewer_historical_mean", "reviewer_historical_stddev",
                      "score_z", "score_percentile", "history_count", "flag"):
            self.assertEqual(getattr(default, field), getattr(strict, field))

    def test_require_timestamps_never_flags_missing(self):
        # Even an extreme score with no timestamp must not be flagged in
        # production mode: no temporal ordering is possible, so no claim.
        reviews = [R("r", "alex", 1, NEGATIVE, None)]
        target = calibrate(reviews, require_timestamps=True)[-1]
        self.assertEqual(target.calibration_status, "missing_timestamp")
        self.assertFalse(target.flag)


if __name__ == "__main__":
    unittest.main()
