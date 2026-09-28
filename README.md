# Customer Feedback Calibration Analyzer — V3

Detects **calibration anomalies**: customer scores that are unusual
relative to *that reviewer's own* historical scoring pattern, using
**strict point-in-time calibration** — a review at time T sees only
observations with timestamp strictly earlier than T.
**Synthetic data only. Advisory output only.**

![architecture](docs/architecture.svg)

*Decision authority: deterministic statistical policy only. Amazon
Comprehend authority: contextual evidence only — its output provably
cannot move a flag.*

## What it does

1. Loads `data/mock_reviews.csv` (deterministic, seed=42, all fictional,
   deliberately written out of chronological order).
2. Classifies each review's sentiment via a swappable `SentimentProvider`:
   rule-based lexicon (default, zero cost) or Amazon Comprehend
   (`--sentiment comprehend`, opt-in). All four Comprehend confidence
   scores are stored; the probability vector is never discarded.
3. For each review, computes baseline statistics from the reviewer's
   **strictly prior** observations: count, mean, sample stddev, empirical
   percentile, z-score. Same-timestamp records never see each other.
4. Flags `calibration_anomaly` when |z| >= 2.0 (configurable project
   policy in `src/alignment.py`, not a statistical truth). **The flag can
   never depend on sentiment output** — regression-tested.
5. Every result reports `calibration_method` explicitly:
   `point_in_time` or `leave_one_out_legacy`. The legacy fallback is never
   silent.
6. Writes `data/analyzed_reviews.csv` and `data/report.txt`.

## Run

Setup (deterministic — dependencies pinned in `requirements.txt`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py   # dashboard
```

Pipeline (stdlib only, except boto3 for Comprehend mode):

```bash
python3 scripts/gen_mock.py                  # regenerate synthetic data (optional)
python3 src/main.py                          # rule-based sentiment (default)
python3 src/main.py --sentiment comprehend   # Amazon Comprehend (needs AWS creds)
python3 src/main.py --require-timestamps     # production contract: refuse
                                             # timestamp-less records instead of
                                             # degrading to leave-one-out
python3 -m unittest discover -s tests -v     # 37 tests, no AWS needed
python3 scripts/demo_forced_failure.py       # Comprehend outage demo (no AWS needed)
RUN_AWS_INTEGRATION_TESTS=1 python3 -m unittest tests.test_comprehend_integration  # opt-in live test
```

## Results (synthetic data, seed=42)

- 303 observations processed; **3 anomalies detected**.
- 2 of the 3 are planted fixtures (`r_planted`, `r_zerovar`); the third is
  an **unplanted emergent anomaly in the synthetic dataset** (`r_avg`:
  score 4 against a personal mean of 7.0, z=-2.22) — the detector is not
  just recognizing hand-authored test fixtures. Synthetic data cannot
  establish real-world validity; that is not the claim.
- 208 observations consistent with reviewer history.
- 92 correctly withheld from calibration: insufficient prior history is an
  explicit status, never an invented baseline.
- Point-in-time baselines proven invariant to future observations;
  input-order invariance proven; same-timestamp leakage prevented;
  threshold transition proven exactly at 10 prior observations.
- Auxiliary AWS inference proven unable to move the deterministic decision
  (forced-failure demo below).
- 37/37 automated tests green, 0 AWS calls in the default run.

## Architecture

Timestamped reviewer history → strict point-in-time baseline →
deterministic calibration policy → flag / no flag. Review text →
Amazon Comprehend → contextual evidence only. (Diagram above.)

## Graceful degradation of auxiliary inference (V2)

"Fail-open/fail-closed" is the wrong frame — nothing is accepted that
should have been rejected. The precise behavior: Comprehend is an
auxiliary inference service, and its failure degrades the rationale
without touching the deterministic calibration engine.

- Comprehend outage, throttling, per-document errors, oversize input, or
  malformed responses → `sentiment_status = "unavailable"`, label `None`.
- The calibration analysis still completes; the rationale notes that
  sentiment was unavailable and the determination used history only.
- **No error path converts unavailable into a default NEUTRAL.** Unit-tested.
- Comprehend output is recorded as evidence, not ground truth. No LLM.

Forced-failure demo (`scripts/demo_forced_failure.py`) — the same anomaly
decisions with Comprehend working vs. fully dead:

```
Reviews: 303
Comprehend working: 3 anomalies flagged
Comprehend dead:    3 anomalies flagged, 303 sentiments unavailable
Identical anomaly decisions: True
  - r_avg score=4 at 2026-01-21T09:45:40: still flagged with sentiment_status=unavailable
  - r_planted score=4 at 2026-03-01T12:00:00: still flagged with sentiment_status=unavailable
  - r_zerovar score=5 at 2026-03-01T12:00:00: still flagged with sentiment_status=unavailable
```

## Locked design constraints

1. **Point-in-time.** History for a review at T is the same reviewer's
   observations with timestamp strictly < T. Future data cannot leak:
   adding or changing reviews dated after the target provably cannot
   change its baseline, z-score, percentile, status, or flag (tested).
   Same-timestamp records never contribute to one another. Reviews without
   a timestamp use the V2 leave-one-out approximation as a documented
   legacy path — and every result reports `calibration_method`
   (`point_in_time` | `leave_one_out_legacy`) so the mode is explicit in
   output, never silent.
2. **Production contract for missing timestamps.** `--require-timestamps`
   (or `REQUIRE_TIMESTAMPS = True`, or `calibrate(..., require_timestamps=True)`)
   makes timestamp-less records an explicit refusal:
   `calibration_status = missing_timestamp`, no baseline, no flag — instead
   of silently degrading to leave-one-out, which would reintroduce the exact
   temporal-leakage limitation V3 fixed. Tested.
2. **Zero variance is explicit.** `baseline_status = zero_variance_baseline`,
   z-score undefined (`None`). A score matching the constant pattern is
   consistent; any deviation is an anomaly.
3. **z is a deviation metric, not proof.** Scores are bounded and discrete
   (1–10); empirical percentile is reported alongside because reviewer
   distributions are rarely normal.
4. **Deterministic, configurable threshold.** `MIN_HISTORY = 10`,
   `ANOMALY_THRESHOLD_ABS_Z = 2.0` live as named constants. Reviewers
   below the history minimum get `insufficient_history` — an explicit
   status, never an invented baseline. Early-history observations are
   therefore uncalibrated by design, not by accident.
5. **Terminology.** "Calibration anomaly", never "mismatch". A mismatch
   implies one side is correct; the engine only measures unusualness
   relative to reviewer history.

## Known limitation

Leave-one-out (the no-timestamp legacy path) is a documented
approximation, now explicit per row via `calibration_method`; the
timestamped path is strict. Within the timestamped path, the method is
exact. Real deployments can eliminate the approximation entirely with
`--require-timestamps`.

## Test suite

37 tests, 0 AWS calls in the default run:

- `test_sentiment.py` — rule-based classifier incl. raw-count preservation
- `test_alignment.py` — calibration policy (leave-one-out legacy path)
- `test_v2_sentiment.py` — Comprehend provider: four-score mapping, partial
  batch failure, throttling → unavailable, oversize input, 25-doc batching,
  flag identical across all Comprehend labels and under total outage
- `test_v3_temporal.py` — future-data isolation, same-timestamp exclusion,
  out-of-order input invariance, threshold crossing (10th insufficient /
  11th sufficient), legacy path
- `test_production_mode.py` — `calibration_method` explicit on every row,
  legacy mode named in reasons, `require_timestamps` refuses
  timestamp-less records (`missing_timestamp`, no baseline, no flag) and
  leaves timestamped results byte-identical
- `test_comprehend_integration.py` — opt-in live AWS test (skipped by default)

## Boundaries

- Synthetic reviews only. Never feed real client comments, employee
  evaluations, or PII into this project.
- Advisory only: every anomaly rationale ends with "Human review
  recommended; do not auto-adjust." The tool never changes a rating or
  evaluates an employee.

## Resume bullet

Built a point-in-time customer-feedback calibration system that detects
reviewer-specific rating anomalies using deterministic statistical rules
while isolating Amazon Comprehend sentiment as non-decision-making
contextual evidence. Implemented strict temporal leakage prevention,
insufficient-history and zero-variance handling, graceful degradation of
auxiliary inference, batch sentiment processing, and regression tests
proving probabilistic model output cannot alter anomaly classifications.
Validated on 303 synthetic reviews with 37/37 automated tests passing.
