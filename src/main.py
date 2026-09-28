"""Pipeline entry point: ingest -> sentiment -> calibration -> report.

Usage:
  python3 src/main.py                          # rule-based sentiment (default)
  python3 src/main.py --sentiment comprehend    # Amazon Comprehend (needs AWS creds)
  python3 src/main.py --sentiment comprehend --region us-west-2
  python3 src/main.py --require-timestamps     # production contract: refuse
                                               # timestamp-less records instead
                                               # of degrading to leave-one-out
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alignment import calibrate  # noqa: E402
from src.ingest import load_reviews  # noqa: E402
from src.report import render_text, summarize  # noqa: E402
from src.sentiment import ComprehendProvider, RuleBasedProvider  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLUMNS = [
    "reviewer_id", "employee", "score", "review_text", "timestamp",
    "history_count", "baseline_status",
    "reviewer_historical_mean", "reviewer_historical_stddev",
    "score_z", "score_percentile", "anomaly_threshold_abs_z",
    "calibration_status", "calibration_method", "flag",
    "sentiment_status", "sentiment_label",
    "sentiment_positive", "sentiment_negative",
    "sentiment_neutral", "sentiment_mixed",
    "positive_hits", "negative_hits",
    "positive_themes", "negative_themes", "reason",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", choices=["rules", "comprehend"],
                        default="rules")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--require-timestamps", action="store_true",
                        help="production contract: refuse timestamp-less records "
                             "(missing_timestamp) instead of degrading to the "
                             "V2 leave-one-out approximation")
    args = parser.parse_args()

    if args.sentiment == "comprehend":
        try:
            provider = ComprehendProvider(region=args.region)
        except ImportError:
            sys.exit("boto3 is required for --sentiment comprehend: pip install boto3")
        print(f"Sentiment provider: Amazon Comprehend ({args.region})")
    else:
        provider = RuleBasedProvider()
        print("Sentiment provider: rule-based lexicon (V1 baseline)")

    if args.require_timestamps:
        print("Calibration contract: point-in-time REQUIRED "
              "(timestamp-less records refused as missing_timestamp)")
    else:
        print("Calibration contract: point-in-time; legacy leave-one-out "
              "fallback for timestamp-less records (explicit per row)")

    input_path = os.path.join(BASE, "data", "mock_reviews.csv")
    output_path = os.path.join(BASE, "data", "analyzed_reviews.csv")
    report_path = os.path.join(BASE, "data", "report.txt")

    reviews = load_reviews(input_path)
    results = calibrate(reviews, provider=provider,
                        require_timestamps=args.require_timestamps)
    summary = summarize(results)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for r in results:
            writer.writerow([
                r.reviewer_id, r.employee, r.score, r.review_text, r.timestamp,
                r.history_count, r.baseline_status,
                r.reviewer_historical_mean, r.reviewer_historical_stddev,
                r.score_z, r.score_percentile, r.anomaly_threshold_abs_z,
                r.calibration_status, r.calibration_method, r.flag,
                r.sentiment_status, r.sentiment_label,
                r.sentiment_positive, r.sentiment_negative,
                r.sentiment_neutral, r.sentiment_mixed,
                r.positive_hits, r.negative_hits,
                ";".join(r.positive_themes), ";".join(r.negative_themes),
                r.reason,
            ])

    report = render_text(summary)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    print(report)
    print(f"\nWrote {output_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
