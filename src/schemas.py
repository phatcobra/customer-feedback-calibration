"""Data contracts for the feedback-calibration pipeline (V1.1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Review:
    reviewer_id: str
    employee: str
    customer_score: int  # 1..10
    review_text: str
    timestamp: str | None = None  # ISO-8601; None = no temporal info (V2 legacy path)


@dataclass(frozen=True)
class SentimentResult:
    sentiment: str  # strongly_positive | positive | neutral | mixed | negative
    positive_hits: int
    negative_hits: int
    matched_positive: tuple = ()
    matched_negative: tuple = ()
    # Raw, auditable inputs: never reduce to the winning label alone.
    confidence: float = 0.0  # |pos-neg| / (pos+neg), 0 when no hits


@dataclass(frozen=True)
class CalibrationResult:
    reviewer_id: str
    employee: str
    score: int
    review_text: str
    history_count: int  # prior observations EXCLUDING this one (leave-one-out)
    baseline_status: str  # sufficient_history | insufficient_history | zero_variance_baseline
    reviewer_historical_mean: float | None
    reviewer_historical_stddev: float | None  # sample stddev (n-1)
    score_z: float | None  # undefined when variance is zero
    score_percentile: float | None  # % of history strictly below this score
    anomaly_threshold_abs_z: float  # project policy, not a statistical truth
    calibration_status: str  # consistent_with_reviewer_history | calibration_anomaly | insufficient_history | missing_timestamp
    calibration_method: str  # "point_in_time" | "leave_one_out_legacy"
    # point_in_time: strict V3 baseline (timestamp strictly < T).
    # leave_one_out_legacy: V2 approximation for timestamp-less records;
    #   kept for backward compatibility, never used when timestamps exist.
    #   (A missing_timestamp record in require_timestamps mode reports
    #   calibration_method="point_in_time": the mode in force refused it.)
    # --- sentiment: contextual evidence only; never a flag gate (V2) ---
    sentiment_status: str  # "available" | "unavailable"
    sentiment_label: str | None  # e.g. "NEUTRAL" (Comprehend) / "neutral" (rules)
    sentiment_positive: float | None
    sentiment_negative: float | None
    sentiment_neutral: float | None
    sentiment_mixed: float | None
    # -------------------------------------------------------------------
    positive_hits: int
    negative_hits: int
    flag: bool
    reason: str
    positive_themes: tuple = field(default=())
    negative_themes: tuple = field(default=())
    timestamp: str | None = None  # ISO-8601 of the reviewed observation
