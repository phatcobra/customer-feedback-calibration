"""Deterministic synthetic review generator (seed=42).

Every review carries an ISO-8601 timestamp. Rows are written in
deterministic shuffled order to prove the pipeline is order-invariant.
Planted reviewers exercise every baseline path:
  r_planted    - 12 consistent 9-10s in Jan, neutral-text 4 on 2026-03-01
                 (calibration anomaly; sentiment is not a gate)
  r_zerovar    - 11 constant 9s in Jan, 5 on 2026-03-01 (zero-variance anomaly)
  r_zerovar_ok - 12 constant 9s Jan-Feb (zero variance, consistent)
  r_new        - 2 reviews (insufficient_history)
  r_thresh     - 11 daily reviews 2026-01-01..11: the 10th chronological
                 observation is still insufficient_history (9 priors);
                 the 11th calibrates on the preceding 10
  r_same       - 10 reviews spread over Jan, then 3 reviews at exactly the
                 same timestamp: same-T records never see each other
All data is fictional.
"""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POSITIVE_TEXTS = [
    "Alex was great and answered all my questions.",
    "Very professional, quick, and helpful.",
    "Everything was handled correctly. Thank you.",
    "Maria was friendly, patient, and knowledgeable.",
    "James resolved my issue quickly and explained everything clearly.",
    "Excellent service, very professional and efficient.",
    "She was courteous and made the whole process easy.",
    "Fantastic help, I was impressed with how thorough he was.",
    "Quick, clear answers. Very satisfied with the support.",
    "Wonderful experience, he was helpful and friendly throughout.",
]

MIXED_TEXTS = [
    "He was polite, but I had to wait too long and my issue was not resolved.",
    "Friendly service but the wait was frustrating.",
    "She was helpful, though the process was confusing at first.",
    "Quick to answer but my problem is still unresolved.",
    "Professional and courteous, but I waited a long time to be seen.",
]

NEGATIVE_TEXTS = [
    "I still cannot access my account and nobody called me back.",
    "Terrible experience, rude and completely unhelpful.",
    "Worst service ever. Long wait, confusing answers, nothing resolved.",
    "Nobody called me back and the issue is still broken.",
    "Awful. I was ignored and my problem remains unresolved.",
]

NEUTRAL_TEXTS = [
    "Called about my account balance.",
    "Visited the branch on Tuesday.",
    "Spoke with a representative about a form.",
    "The service was fine, nothing special.",
]

REVIEWERS = {
    "r_cons": {"score": lambda r: r.choice([6, 7, 7, 8, 8])},
    "r_gen": {"score": lambda r: r.choice([9, 9, 10, 10])},
    "r_avg": {"score": lambda r: r.choice([4, 5, 6, 7, 8, 9])},
    "r_harsh": {"score": lambda r: r.choice([3, 4, 5, 5, 6])},
}
EMPLOYEES = ["alex", "maria", "james"]
T0 = datetime(2026, 1, 1)


def _ts(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _rand_ts(rng: random.Random, lo: datetime, hi: datetime) -> str:
    delta = (hi - lo).total_seconds()
    return _ts(lo + timedelta(seconds=rng.randint(0, int(delta))))


def _row(reviewer_id, employee, score, text, timestamp):
    return {"reviewer_id": reviewer_id, "employee": employee,
            "customer_score": score, "review_text": text,
            "timestamp": timestamp}


def _text_for(score: int, rng: random.Random) -> str:
    if score >= 8:
        pool = POSITIVE_TEXTS + MIXED_TEXTS[:1]
    elif score >= 6:
        pool = POSITIVE_TEXTS[:4] + MIXED_TEXTS
    elif score >= 4:
        pool = MIXED_TEXTS + NEGATIVE_TEXTS[:2] + NEUTRAL_TEXTS
    else:
        pool = NEGATIVE_TEXTS + MIXED_TEXTS[:1]
    return rng.choice(pool)


def generate(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    jan, mar31 = datetime(2026, 1, 1), datetime(2026, 3, 31)

    for reviewer_id, cfg in REVIEWERS.items():
        for _ in range(60):
            score = cfg["score"](rng)
            rows.append(_row(reviewer_id, rng.choice(EMPLOYEES), score,
                             _text_for(score, rng), _rand_ts(rng, jan, mar31)))

    # r_planted: 12 consistent high scores in Jan, anomaly on 2026-03-01.
    for _ in range(12):
        score = rng.choice([9, 9, 10, 10])
        rows.append(_row("r_planted", rng.choice(EMPLOYEES), score,
                         rng.choice(POSITIVE_TEXTS), _rand_ts(rng, jan, datetime(2026, 1, 31))))
    rows.append(_row("r_planted", "alex", 4,
                     "The service was fine, nothing special.",
                     "2026-03-01T12:00:00"))

    # r_zerovar: 11 constant 9s in Jan, then a 5 on 2026-03-01.
    for _ in range(11):
        rows.append(_row("r_zerovar", rng.choice(EMPLOYEES), 9,
                         rng.choice(POSITIVE_TEXTS), _rand_ts(rng, jan, datetime(2026, 1, 31))))
    rows.append(_row("r_zerovar", "maria", 5,
                     "Alex was great and answered all my questions.",
                     "2026-03-01T12:00:00"))

    # r_zerovar_ok: 12 constant 9s, no anomaly.
    for _ in range(12):
        rows.append(_row("r_zerovar_ok", rng.choice(EMPLOYEES), 9,
                         rng.choice(POSITIVE_TEXTS), _rand_ts(rng, jan, mar31)))

    # r_new: too little history for calibration.
    rows.append(_row("r_new", "alex", 9, rng.choice(POSITIVE_TEXTS),
                     _rand_ts(rng, jan, mar31)))
    rows.append(_row("r_new", "james", 8, rng.choice(POSITIVE_TEXTS),
                     _rand_ts(rng, jan, mar31)))

    # r_thresh: 11 daily reviews; chronological #10 has 9 priors
    # (insufficient), #11 has 10 priors (sufficient).
    for day in range(1, 12):
        rows.append(_row("r_thresh", "alex", 9, rng.choice(POSITIVE_TEXTS),
                         _ts(datetime(2026, 1, day, 10, 0, 0))))

    # r_same: 10 reviews spread over Jan, then 3 at exactly the same timestamp.
    for _ in range(10):
        rows.append(_row("r_same", rng.choice(EMPLOYEES), 9,
                         rng.choice(POSITIVE_TEXTS), _rand_ts(rng, jan, datetime(2026, 1, 31))))
    for _ in range(3):
        rows.append(_row("r_same", rng.choice(EMPLOYEES), 9,
                         rng.choice(POSITIVE_TEXTS), "2026-03-01T12:00:00"))

    # Deterministic shuffle: the CSV is deliberately out of order.
    rng.shuffle(rows)
    return rows


def main() -> None:
    path = os.path.join(BASE, "data", "mock_reviews.csv")
    rows = generate()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["reviewer_id", "employee", "customer_score",
                            "review_text", "timestamp"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic reviews to {path}")


if __name__ == "__main__":
    main()
