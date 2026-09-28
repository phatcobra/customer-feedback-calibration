"""Streamlit dashboard for the calibration analyzer (V3 presentation layer).

Run from the project root:
    .venv/bin/streamlit run app/streamlit_app.py

Reads data/analyzed_reviews.csv produced by `python3 src/main.py`.
"""

from __future__ import annotations

import csv
import os
from collections import Counter

import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE, "data", "analyzed_reviews.csv")


@st.cache_data
def load_rows() -> list[dict]:
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    st.set_page_config(page_title="Feedback Calibration Analyzer", layout="wide")
    st.title("Customer Feedback Calibration Analyzer")
    st.caption(
        "Point-in-time reviewer calibration. The anomaly decision comes from "
        "deterministic statistics only; sentiment is contextual evidence."
    )

    rows = load_rows()

    reviewers = sorted({r["reviewer_id"] for r in rows})
    statuses = sorted({r["calibration_status"] for r in rows})
    col1, col2 = st.columns(2)
    with col1:
        sel_reviewers = st.multiselect("Reviewer", reviewers, default=reviewers)
    with col2:
        sel_statuses = st.multiselect("Calibration status", statuses, default=statuses)
    rows = [r for r in rows
            if r["reviewer_id"] in sel_reviewers and r["calibration_status"] in sel_statuses]

    flagged = [r for r in rows if r["flag"] == "True"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", len(rows))
    c2.metric("Anomalies flagged", len(flagged))
    c3.metric("Insufficient history",
              sum(1 for r in rows if r["calibration_status"] == "insufficient_history"))
    c4.metric("Sentiment unavailable",
              sum(1 for r in rows if r["sentiment_status"] == "unavailable"))

    st.subheader("Flagged for human review")
    if not flagged:
        st.write("No anomalies in the current selection.")
    for r in flagged:
        z = r["score_z"] or "undefined (zero variance)"
        with st.expander(
            f"{r['reviewer_id']} · score {r['score']} · "
            f"history n={r['history_count']}, mean={r['reviewer_historical_mean']}, z={z}"
        ):
            st.write(f"**Text:** {r['review_text']}")
            st.write(f"**Sentiment:** {r['sentiment_status']} / {r['sentiment_label']}")
            st.write(f"**Why:** {r['reason']}")

    st.subheader("Distributions")
    d1, d2 = st.columns(2)
    with d1:
        st.caption("Calibration status")
        st.bar_chart(Counter(r["calibration_status"] for r in rows))
    with d2:
        st.caption("Sentiment labels (evidence only)")
        st.bar_chart(Counter(r["sentiment_label"] for r in rows if r["sentiment_label"]))


if __name__ == "__main__":
    main()
