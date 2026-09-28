"""Forced-failure demo: Comprehend goes down, calibration survives.

Runs the full pipeline twice over the synthetic dataset:
  1. with a working (stubbed) Comprehend client,
  2. with a Comprehend client that raises on every call (outage/throttling).

Prints both anomaly decisions side by side to show the deterministic
policy layer is unaffected by auxiliary-inference failure.
No AWS credentials needed; no network calls.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alignment import calibrate
from src.ingest import load_reviews
from src.sentiment import ComprehendProvider


class WorkingClient:
    def batch_detect_sentiment(self, TextList, LanguageCode):
        return {"ResultList": [
            {"Index": i, "Sentiment": "NEUTRAL",
             "SentimentScore": {"Positive": 0.08, "Negative": 0.06,
                                "Neutral": 0.83, "Mixed": 0.03}}
            for i in range(len(TextList))], "ErrorList": []}


class DeadClient:
    def batch_detect_sentiment(self, TextList, LanguageCode):
        raise RuntimeError("simulated Comprehend outage (throttling)")


def main() -> None:
    reviews = load_reviews(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "mock_reviews.csv"))
    ok = calibrate(reviews, provider=ComprehendProvider(client=WorkingClient()))
    dead = calibrate(reviews, provider=ComprehendProvider(client=DeadClient()))

    ok_flags = [(r.reviewer_id, r.score, r.timestamp) for r in ok if r.flag]
    dead_flags = [(r.reviewer_id, r.score, r.timestamp) for r in dead if r.flag]
    unavailable = sum(1 for r in dead if r.sentiment_status == "unavailable")

    print(f"Reviews: {len(reviews)}")
    print(f"Comprehend working: {len(ok_flags)} anomalies flagged")
    print(f"Comprehend dead:    {len(dead_flags)} anomalies flagged, "
          f"{unavailable} sentiments unavailable")
    print(f"Identical anomaly decisions: {ok_flags == dead_flags}")
    for rid, score, tstamp in dead_flags:
        print(f"  - {rid} score={score} at {tstamp}: still flagged with "
              "sentiment_status=unavailable")


if __name__ == "__main__":
    main()
