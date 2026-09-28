"""V3 tests: strict point-in-time calibration.

The core guarantee: a review at time T is calibrated ONLY on the same
reviewer's observations with timestamp strictly < T. Future data cannot
leak into the baseline, and input row order cannot change results.
"""

import dataclasses
import os
import random
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alignment import calibrate
from src.schemas import Review

POSITIVE = "Alex was great and answered all my questions."
T0 = datetime(2026, 1, 1)


def R(reviewer_id, employee, score, text, timestamp):
    return Review(reviewer_id=reviewer_id, employee=employee,
                  customer_score=score, review_text=text, timestamp=timestamp)


def ts(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _key(r):
    return (r.reviewer_id, r.timestamp or "", r.score, r.review_text, r.employee)


class TestPointInTime(unittest.TestCase):
    def test_future_data_cannot_affect_target(self):
        # 10 history 9s, target 4 on Jan 15, 5 future 1s in Feb.
        reviews = [R("r", "alex", 9, POSITIVE, ts(T0 + timedelta(days=d)))
                   for d in range(10)]
        reviews.append(R("r", "alex", 4, POSITIVE, ts(T0 + timedelta(days=14))))
        reviews += [R("r", "alex", 1, POSITIVE, ts(T0 + timedelta(days=31 + d)))
                    for d in range(5)]
        before = { _key(r): r for r in calibrate(reviews) }
        target_key = ("r", ts(T0 + timedelta(days=14)), 4, POSITIVE, "alex")
        snap = before[target_key]
        self.assertEqual(snap.calibration_status, "calibration_anomaly")

        # Mutate the future aggressively: different scores, more reviews.
        reviews2 = [R("r", "alex", 9, POSITIVE, ts(T0 + timedelta(days=d)))
                    for d in range(10)]
        reviews2.append(R("r", "alex", 4, POSITIVE, ts(T0 + timedelta(days=14))))
        reviews2 += [R("r", "alex", 10, POSITIVE, ts(T0 + timedelta(days=31 + d)))
                     for d in range(12)]
        after = { _key(r): r for r in calibrate(reviews2) }
        snap2 = after[target_key]
        for field in ("reviewer_historical_mean", "reviewer_historical_stddev",
                      "score_z", "score_percentile", "history_count",
                      "baseline_status", "calibration_status", "flag"):
            self.assertEqual(getattr(snap, field), getattr(snap2, field),
                             f"future data leaked into {field}")

    def test_same_timestamp_records_cannot_see_each_other(self):
        reviews = [R("r", "alex", 9, POSITIVE, ts(T0 + timedelta(days=d)))
                   for d in range(10)]
        same_t = ts(datetime(2026, 3, 1, 12, 0, 0))
        reviews += [R("r", "alex", 9, POSITIVE, same_t) for _ in range(3)]
        results = calibrate(reviews)
        same = [r for r in results if r.timestamp == same_t]
        self.assertEqual(len(same), 3)
        for r in same:
            self.assertEqual(r.history_count, 10)  # the 10 Jan reviews only
            self.assertEqual(r.reviewer_historical_mean, 9.0)
            self.assertEqual(r.calibration_status, "consistent_with_reviewer_history")
        # Identical shared history -> identical statistics.
        self.assertEqual(len({(r.score_z, r.score_percentile) for r in same}), 1)

    def test_out_of_order_csv_gives_identical_results(self):
        reviews = []
        for rid, scores in (("a", [9, 10, 9, 4, 9, 10, 9, 10, 9, 10, 9, 9]),
                            ("b", [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2])):
            for d, s in enumerate(scores):
                reviews.append(R(rid, "alex", s, POSITIVE,
                                 ts(T0 + timedelta(days=d, hours=d))))
        # astuple order: 0 reviewer_id, 1 employee, 2 score, 3 review_text,
        # ..., 24 timestamp
        key = lambda t: (t[0], t[24] or "", t[3], t[2], t[1])
        ref = sorted((dataclasses.astuple(r) for r in calibrate(reviews)), key=key)
        shuffled = reviews[:]
        random.Random(7).shuffle(shuffled)
        got = sorted((dataclasses.astuple(r) for r in calibrate(shuffled)), key=key)
        self.assertEqual(ref, got)

    def test_threshold_crossing_tenth_insufficient_eleventh_sufficient(self):
        reviews = [R("r", "alex", s, POSITIVE,
                     ts(datetime(2026, 1, d, 10, 0, 0)))
                   for d, s in enumerate([9, 10] * 6, start=1)][:11]
        by_ts = {r.timestamp: r for r in calibrate(reviews)}
        tenth = by_ts[ts(datetime(2026, 1, 10, 10, 0, 0))]
        eleventh = by_ts[ts(datetime(2026, 1, 11, 10, 0, 0))]
        self.assertEqual(tenth.history_count, 9)
        self.assertEqual(tenth.calibration_status, "insufficient_history")
        self.assertFalse(tenth.flag)
        self.assertEqual(eleventh.history_count, 10)
        self.assertEqual(eleventh.baseline_status, "sufficient_history")

    def test_missing_timestamp_uses_legacy_leave_one_out(self):
        reviews = [Review(reviewer_id="r", employee="alex", customer_score=s,
                          review_text=POSITIVE) for s in [9, 10] * 6]
        for r in calibrate(reviews):
            self.assertEqual(r.history_count, 11)  # sees all others
            self.assertEqual(r.baseline_status, "sufficient_history")


if __name__ == "__main__":
    unittest.main()
