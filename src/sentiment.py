"""Rule-based sentiment baseline (V1) + provider abstraction (V2).

V2 adds a SentimentProvider interface so the calibration engine can
swap implementations without touching the deterministic policy layer.
The rule-based classifier remains the zero-cost default and the
regression baseline; ComprehendProvider is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import SentimentResult

POSITIVE_WORDS = frozenset({
    "great", "excellent", "helpful", "professional", "quick", "fast",
    "resolved", "knowledgeable", "friendly", "courteous", "patient",
    "thorough", "efficient", "amazing", "wonderful", "fantastic",
    "perfect", "answered", "clear", "easy", "smooth", "satisfied",
    "impressed", "polite",
})

NEGATIVE_WORDS = frozenset({
    "wait", "waiting", "waited", "rude", "unresolved", "confusing",
    "never", "terrible", "awful", "bad", "slow", "long", "frustrating",
    "frustrated", "angry", "disappointed", "ignored", "nobody",
    "access", "denied", "error", "broken", "worst", "horrible",
    "incompetent", "useless", "callback",
})

# Multi-word phrases checked before single words.
NEGATIVE_PHRASES = frozenset({
    "never called back", "no callback", "too long",
})


def _hits(text: str, words: frozenset, phrases: frozenset = frozenset()) -> list:
    lowered = text.lower()
    found = [p for p in phrases if p in lowered]
    tokens = set()
    for token in lowered.replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split():
        tokens.add(token.strip("'\""))
    found.extend(sorted(tokens & words))
    return found


def classify(text: str) -> SentimentResult:
    """Classify review text into one of five sentiment buckets."""
    pos = _hits(text, POSITIVE_WORDS)
    neg = _hits(text, NEGATIVE_WORDS, NEGATIVE_PHRASES)
    p, n = len(pos), len(neg)

    if p == 0 and n == 0:
        sentiment = "neutral"
    elif p > 0 and n > 0:
        sentiment = "mixed"
    elif p >= 3:
        sentiment = "strongly_positive"
    elif p > 0:
        sentiment = "positive"
    elif n >= 3:
        sentiment = "negative"
    else:
        sentiment = "negative"

    confidence = abs(p - n) / (p + n) if (p + n) else 0.0
    return SentimentResult(
        sentiment=sentiment,
        positive_hits=p,
        negative_hits=n,
        matched_positive=tuple(pos),
        matched_negative=tuple(neg),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# V2: provider abstraction. Model output is evidence, never ground truth,
# and never a flag gate. Any failure -> status "unavailable", never a
# default label.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SentimentEvidence:
    status: str  # "available" | "unavailable"
    label: str | None  # e.g. "NEUTRAL" (Comprehend) or "neutral" (rules)
    positive: float | None
    negative: float | None
    neutral: float | None
    mixed: float | None
    detail: str = ""
    positive_hits: int | None = None  # rule-based evidence only
    negative_hits: int | None = None


class SentimentProvider:
    """Interface: batch of texts in, one SentimentEvidence per text out."""

    def analyze_batch(self, texts: list[str]) -> list[SentimentEvidence]:
        raise NotImplementedError


class RuleBasedProvider(SentimentProvider):
    """V1 lexicon classifier, wrapped as a provider. Zero cost, no network."""

    def analyze_batch(self, texts: list[str]) -> list[SentimentEvidence]:
        out = []
        for text in texts:
            r = classify(text)
            out.append(SentimentEvidence(
                status="available", label=r.sentiment,
                positive=None, negative=None, neutral=None, mixed=None,
                detail="rule-based lexicon",
                positive_hits=r.positive_hits, negative_hits=r.negative_hits,
            ))
        return out


class ComprehendProvider(SentimentProvider):
    """Amazon Comprehend BatchDetectSentiment. Opt-in; failures degrade to
    sentiment_status="unavailable" and the calibration analysis proceeds."""

    BATCH_SIZE = 25
    MAX_BYTES = 5000  # per-document limit enforced by the API

    def __init__(self, client=None, region: str = "us-east-1"):
        if client is None:
            import boto3
            client = boto3.client("comprehend", region_name=region)
        self._client = client

    def analyze_batch(self, texts: list[str]) -> list[SentimentEvidence]:
        out: list[SentimentEvidence] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            out.extend(self._analyze_chunk(texts[start:start + self.BATCH_SIZE]))
        return out

    def _analyze_chunk(self, texts: list[str]) -> list[SentimentEvidence]:
        out: list[SentimentEvidence] = [
            SentimentEvidence("unavailable", None, None, None, None, None,
                              "input exceeds 5KB")
            for _ in texts
        ]
        valid = [i for i, t in enumerate(texts)
                 if len(t.encode("utf-8")) <= self.MAX_BYTES]
        if not valid:
            return out
        try:
            resp = self._client.batch_detect_sentiment(
                TextList=[texts[i] for i in valid], LanguageCode="en")
        except Exception as exc:  # outage, throttling, auth: all -> unavailable
            for i in valid:
                out[i] = SentimentEvidence(
                    "unavailable", None, None, None, None, None,
                    f"aws-error: {type(exc).__name__}")
            return out
        # Response Index values are positions within the submitted TextList.
        by_pos = {r["Index"]: r for r in resp.get("ResultList", [])}
        errors = {e["Index"]: e for e in resp.get("ErrorList", [])}
        for pos, i in enumerate(valid):
            if pos in errors:
                out[i] = SentimentEvidence(
                    "unavailable", None, None, None, None, None,
                    f"aws-error: {errors[pos].get('ErrorCode', 'unknown')}")
            elif pos in by_pos:
                r = by_pos[pos]
                s = r["SentimentScore"]
                out[i] = SentimentEvidence(
                    "available", r["Sentiment"],
                    s["Positive"], s["Negative"], s["Neutral"], s["Mixed"],
                    "comprehend")
            else:
                out[i] = SentimentEvidence(
                    "unavailable", None, None, None, None, None,
                    "missing from response")
        return out
