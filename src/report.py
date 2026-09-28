"""Summary report for calibrated reviews (V2)."""

from __future__ import annotations

from collections import Counter

from .schemas import CalibrationResult


def summarize(results: list[CalibrationResult]) -> dict:
    by_employee: dict[str, list[int]] = {}
    for r in results:
        by_employee.setdefault(r.employee, []).append(r.score)

    return {
        "n_reviews": len(results),
        "avg_score_by_employee": {
            emp: round(sum(scores) / len(scores), 2)
            for emp, scores in sorted(by_employee.items())
        },
        "sentiment_status_counts": dict(Counter(r.sentiment_status for r in results)),
        "sentiment_label_counts": dict(Counter(
            r.sentiment_label for r in results if r.sentiment_label)),
        "baseline_status_counts": dict(Counter(r.baseline_status for r in results)),
        "calibration_status_counts": dict(Counter(r.calibration_status for r in results)),
        "calibration_method_counts": dict(Counter(r.calibration_method for r in results)),
        "positive_theme_counts": dict(
            Counter(t for r in results for t in r.positive_themes)
        ),
        "negative_theme_counts": dict(
            Counter(t for r in results for t in r.negative_themes)
        ),
        "flagged": [r for r in results if r.flag],
    }


def render_text(summary: dict) -> str:
    lines = [
        "FEEDBACK CALIBRATION REPORT (V3, synthetic data)",
        "=" * 60,
        f"Reviews analyzed: {summary['n_reviews']}",
        "",
        "Average score by employee:",
    ]
    for emp, avg in summary["avg_score_by_employee"].items():
        lines.append(f"  {emp}: {avg}")
    lines += ["", "Sentiment status (evidence only, never a flag gate):"]
    for s, c in sorted(summary["sentiment_status_counts"].items()):
        lines.append(f"  {s}: {c}")
    lines += ["", "Sentiment labels:"]
    for s, c in sorted(summary["sentiment_label_counts"].items()):
        lines.append(f"  {s}: {c}")
    lines += ["", "Baseline status:"]
    for s, c in sorted(summary["baseline_status_counts"].items()):
        lines.append(f"  {s}: {c}")
    lines += ["", "Calibration status:"]
    for s, c in sorted(summary["calibration_status_counts"].items()):
        lines.append(f"  {s}: {c}")
    lines += ["", "Calibration method (explicit per row; never silent):"]
    for s, c in sorted(summary["calibration_method_counts"].items()):
        lines.append(f"  {s}: {c}")
    lines += ["", "Positive themes:"]
    for t, c in sorted(summary["positive_theme_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"  {t}: {c}")
    lines += ["", "Negative themes:"]
    for t, c in sorted(summary["negative_theme_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"  {t}: {c}")
    lines += ["", f"Flagged for human review: {len(summary['flagged'])}"]
    for r in summary["flagged"]:
        z = f"z={r.score_z}" if r.score_z is not None else "z=undefined (zero variance)"
        lines.append(
            f"  [{r.calibration_status}] reviewer={r.reviewer_id} "
            f"employee={r.employee} score={r.score} "
            f"(history n={r.history_count}, mean={r.reviewer_historical_mean}, {z})"
        )
        lines.append(f"    text: {r.review_text[:90]}")
        lines.append(f"    why: {r.reason}")
    return "\n".join(lines)
