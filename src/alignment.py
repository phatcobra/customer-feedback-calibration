"""Calibration-anomaly engine (V1.1 logic, V2 provider wiring).

Terminology is deliberate: "calibration anomaly", never "mismatch".
A mismatch implies one side is correct; all we measure is that an
observation is unusual relative to a reviewer's own history.

Four locked constraints (2026-09-28):
1. Leave-one-out: the current observation never contributes to its own
   baseline statistics. (V1 violated this; the observation was shrinking
   its own deviation.)
2. Zero/near-zero variance is explicit: z is undefined. A score equal to
   the constant pattern is consistent; any deviation is an anomaly.
3. Scores are bounded and discrete (1-10): z is an interpretable
   deviation metric, NOT proof of statistical abnormality. Empirical
   percentile is reported alongside because reviewer distributions are
   rarely normal.
4. The flag threshold is a deterministic, configurable project policy
   (|z| >= 2.0), not a universal statistical truth.

V2: sentiment arrives via a SentimentProvider (rule-based default,
Comprehend opt-in). It is contextual evidence for the rationale only --
it never gates the flag, and an unavailable sentiment degrades the
rationale without touching the calibration analysis.

V3: strict point-in-time calibration. A review at time T sees only the
same reviewer's observations with timestamp strictly < T. Same-timestamp
records never contribute to one another (no implicit ordering). Reviews
with no timestamp use the V2 leave-one-out approximation as a documented
legacy path, and timestamp-less reviews are excluded from timestamped
targets' histories (they cannot be ordered).
"""

from __future__ import annotations

import math
from datetime import datetime

from .schemas import CalibrationResult, Review
from .sentiment import RuleBasedProvider, SentimentEvidence, SentimentProvider

# --- Project policy (explicit, configurable, not statistical truth) ---
MIN_HISTORY = 10
ANOMALY_THRESHOLD_ABS_Z = 2.0
ZERO_VAR_TOL = 1e-9
# ----------------------------------------------------------------------

THEME_KEYWORDS = {
    "wait time": {"wait", "waiting", "waited", "long", "slow"},
    "professionalism": {"professional", "courteous", "polite", "friendly", "rude"},
    "knowledge": {"knowledgeable", "answered", "clear", "confusing"},
    "responsiveness": {"quick", "fast", "efficient", "callback", "never called back", "nobody"},
    "resolution": {"resolved", "unresolved", "access", "error", "broken"},
}

_SENTIMENT_POLARITY = {
    "strongly_positive": 2, "positive": 1, "neutral": 0, "mixed": 0, "negative": -2,
}


def extract_themes(text: str, label: str | None) -> tuple[tuple, tuple]:
    # Unknown/unavailable sentiment -> polarity 0 (themes listed both sides,
    # same treatment as neutral): we have no polarity information.
    polarity = _SENTIMENT_POLARITY.get(label.lower(), 0) if label else 0
    lowered = text.lower()
    pos_themes, neg_themes = [], []
    for theme, keywords in THEME_KEYWORDS.items():
        if any(k in lowered for k in keywords):
            if polarity >= 0:
                pos_themes.append(theme)
            if polarity <= 0:
                neg_themes.append(theme)
    return tuple(sorted(set(pos_themes))), tuple(sorted(set(neg_themes)))


def _mean_stddev(scores: list[int]) -> tuple[float, float]:
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var)


def _percentile(history: list[int], score: int) -> float:
    """Empirical percentile: % of history strictly below this score (0-100)."""
    return 100.0 * sum(1 for s in history if s < score) / len(history)


def _history_for(review: Review, i: int, reviews: list[Review],
                 positions: dict[str, list[int]]) -> list[int]:
    """Point-in-time history for one review.

    timestamp set   -> same reviewer's scores with timestamp strictly < T.
    timestamp unset -> V2 legacy: all other reviews by this reviewer
                       (leave-one-out approximation; documented, tested).
    """
    if review.timestamp is None:
        return [reviews[j].customer_score
                for j in positions[review.reviewer_id] if j != i]
    cutoff = datetime.fromisoformat(review.timestamp)
    out = []
    for r in reviews:
        if r.reviewer_id != review.reviewer_id or r.timestamp is None:
            continue
        if datetime.fromisoformat(r.timestamp) < cutoff:
            out.append(r.customer_score)
    return out


def _sentiment_note(ev: SentimentEvidence) -> str:
    if ev.status == "available":
        return f"The written feedback was classified as {ev.label}; sentiment was not used to determine the anomaly."
    return (f"Written-feedback sentiment unavailable ({ev.detail}); "
            "the anomaly determination used reviewer history only.")
    if ev.status == "available":
        return f"The written feedback was classified as {ev.label}; sentiment was not used to determine the anomaly."
    return (f"Written-feedback sentiment unavailable ({ev.detail}); "
            "the anomaly determination used reviewer history only.")


def _base_kwargs(review: Review, ev: SentimentEvidence, pos_themes, neg_themes) -> dict:
    return dict(
        reviewer_id=review.reviewer_id, employee=review.employee,
        score=review.customer_score, review_text=review.review_text,
        timestamp=review.timestamp,
        sentiment_status=ev.status, sentiment_label=ev.label,
        sentiment_positive=ev.positive, sentiment_negative=ev.negative,
        sentiment_neutral=ev.neutral, sentiment_mixed=ev.mixed,
        positive_hits=ev.positive_hits, negative_hits=ev.negative_hits,
        anomaly_threshold_abs_z=ANOMALY_THRESHOLD_ABS_Z,
        positive_themes=pos_themes, negative_themes=neg_themes,
    )


def calibrate(reviews: list[Review],
              provider: SentimentProvider | None = None) -> list[CalibrationResult]:
    provider = provider or RuleBasedProvider()
    evidence = provider.analyze_batch([r.review_text for r in reviews])
    assert len(evidence) == len(reviews), "provider must return one evidence per review"

    # Index positions per reviewer so leave-one-out is exact.
    positions: dict[str, list[int]] = {}
    for i, r in enumerate(reviews):
        positions.setdefault(r.reviewer_id, []).append(i)

    results: list[CalibrationResult] = []
    for i, review in enumerate(reviews):
        ev = evidence[i]
        # V3: strict point-in-time history (Constraint 1, temporal form).
        history = _history_for(review, i, reviews, positions)
        n = len(history)
        pos_themes, neg_themes = extract_themes(review.review_text, ev.label)
        base = _base_kwargs(review, ev, pos_themes, neg_themes)

        if n < MIN_HISTORY:
            results.append(CalibrationResult(
                **base,
                history_count=n, baseline_status="insufficient_history",
                reviewer_historical_mean=None, reviewer_historical_stddev=None,
                score_z=None, score_percentile=None,
                calibration_status="insufficient_history",
                flag=False,
                reason=(f"Insufficient reviewer history ({n} < {MIN_HISTORY}): "
                        "no calibration performed. No inference about this score."),
            ))
            continue

        mean, std = _mean_stddev(history)
        percentile = _percentile(history, review.customer_score)

        if std < ZERO_VAR_TOL:
            # Constraint 2: zero variance -- z undefined; deviation from a
            # constant pattern is itself the anomaly.
            consistent = review.customer_score == history[0]
            status = ("consistent_with_reviewer_history" if consistent
                      else "calibration_anomaly")
            reason = (
                f"Reviewer has a constant historical pattern (all {n} prior "
                f"scores = {history[0]}); this score "
                f"{'matches' if consistent else 'differs from'} it. "
                "z-score undefined (zero variance)."
            )
            if not consistent:
                reason += " Human review recommended; do not auto-adjust."
            results.append(CalibrationResult(
                **base,
                history_count=n, baseline_status="zero_variance_baseline",
                reviewer_historical_mean=round(mean, 2),
                reviewer_historical_stddev=0.0,
                score_z=None, score_percentile=round(percentile, 1),
                calibration_status=status,
                flag=not consistent, reason=reason,
            ))
            continue

        z = (review.customer_score - mean) / std
        anomaly = abs(z) >= ANOMALY_THRESHOLD_ABS_Z
        direction = "low" if z < 0 else "high"
        if anomaly:
            status = "calibration_anomaly"
            reason = (
                f"Calibration anomaly: score {review.customer_score} is unusually "
                f"{direction} relative to this reviewer's prior pattern "
                f"(mean {mean:.2f}, std {std:.2f}, z={z:.2f}, "
                f"percentile {percentile:.0f}; project threshold |z| >= "
                f"{ANOMALY_THRESHOLD_ABS_Z}). {_sentiment_note(ev)} "
                "Human review recommended; do not auto-adjust."
            )
        else:
            status = "consistent_with_reviewer_history"
            reason = (
                f"The score is consistent with this reviewer's established "
                f"scoring pattern (mean {mean:.2f}, std {std:.2f}, z={z:.2f}, "
                f"percentile {percentile:.0f}). {_sentiment_note(ev)}"
            )
        results.append(CalibrationResult(
            **base,
            history_count=n, baseline_status="sufficient_history",
            reviewer_historical_mean=round(mean, 2),
            reviewer_historical_stddev=round(std, 2),
            score_z=round(z, 2), score_percentile=round(percentile, 1),
            calibration_status=status,
            flag=anomaly, reason=reason,
        ))
    return results
