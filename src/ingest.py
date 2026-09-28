"""CSV ingestion for synthetic reviews (V3: timestamps)."""

from __future__ import annotations

import csv
from datetime import datetime

from .schemas import Review


def _parse_timestamp(raw: str | None) -> str | None:
    if raw is None or not raw.strip():
        return None
    try:
        datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        raise ValueError(f"timestamp is not ISO-8601: {raw!r}") from exc
    return raw.strip()


def load_reviews(path: str) -> list[Review]:
    reviews: list[Review] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        has_ts = "timestamp" in (reader.fieldnames or [])
        for row in reader:
            score = int(row["customer_score"])
            if not 1 <= score <= 10:
                raise ValueError(f"customer_score out of range 1-10: {row!r}")
            reviews.append(
                Review(
                    reviewer_id=row["reviewer_id"].strip(),
                    employee=row["employee"].strip(),
                    customer_score=score,
                    review_text=row["review_text"].strip(),
                    timestamp=_parse_timestamp(row.get("timestamp")) if has_ts else None,
                )
            )
    return reviews
