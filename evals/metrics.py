"""Metrics for the eval harness.

Three families, each answering a different question.

**Recall at k** asks whether retrieval put the right answer anywhere the next
stage can still reach it. It is the ceiling on everything downstream: a code
that never makes the shortlist cannot be chosen, no matter how good the model
is. Reporting it per stage is what turns "we use hybrid retrieval with
reranking" from a claim into a measurement.

**Accuracy at 1** asks whether the final answer agreed with the recorded code.
Agreement, not correctness. The labels come from contracting officers and
contracting officers miscode, so this is a floor on real accuracy rather than an
estimate of it.

**Calibration** asks whether the confidence number means anything. A system that
routes low confidence cases to a human is only as good as its confidence, and an
overconfident model is worse than a diffident one because it does not ask for
help. Expected calibration error summarises the gap between claimed and actual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of queries whose true code appeared in the top k.

    ``None`` means the code was not found at any depth searched.
    """
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Rewards putting the answer near the top, not merely in the list.

    Reported alongside recall because two pipelines can have identical recall at
    20 while one of them puts the answer at position 2 and the other at 19, and
    only one of those is useful to a reranker with a depth budget.
    """
    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest rank percentile. No interpolation, so the result is a real
    observation rather than an average of two that never happened."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


@dataclass
class LatencySummary:
    count: int
    p50_ms: float
    p95_ms: float
    mean_ms: float

    @classmethod
    def of(cls, samples: list[float]) -> LatencySummary:
        if not samples:
            return cls(0, 0.0, 0.0, 0.0)
        return cls(
            count=len(samples),
            p50_ms=median(samples),
            p95_ms=percentile(samples, 0.95),
            mean_ms=sum(samples) / len(samples),
        )


@dataclass
class CalibrationBucket:
    lower: float
    upper: float
    count: int = 0
    correct: int = 0
    confidence_sum: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.count if self.count else 0.0

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.count if self.count else 0.0

    @property
    def label(self) -> str:
        return f"{self.lower:.1f} to {self.upper:.1f}"


@dataclass
class Calibration:
    buckets: list[CalibrationBucket] = field(default_factory=list)

    @classmethod
    def build(cls, pairs: list[tuple[float, bool]], *, bins: int = 5) -> Calibration:
        width = 1.0 / bins
        buckets = [CalibrationBucket(i * width, (i + 1) * width) for i in range(bins)]
        for confidence, correct in pairs:
            index = min(bins - 1, int(confidence / width))
            bucket = buckets[index]
            bucket.count += 1
            bucket.confidence_sum += confidence
            bucket.correct += int(correct)
        return cls(buckets=buckets)

    @property
    def expected_calibration_error(self) -> float:
        """Average gap between claimed confidence and observed accuracy,
        weighted by how many predictions fall in each bucket."""
        total = sum(bucket.count for bucket in self.buckets)
        if not total:
            return 0.0
        return sum(
            (bucket.count / total) * abs(bucket.mean_confidence - bucket.accuracy)
            for bucket in self.buckets
            if bucket.count
        )


@dataclass
class RoutingOutcome:
    """What a confidence threshold actually buys.

    The number that matters operationally is not overall accuracy, it is: how
    much work can be trusted without a human, and how good is that slice. A
    system with 70 percent overall accuracy that correctly identifies which 40
    percent it is sure about is far more useful than one with 75 percent
    accuracy and no idea which answers to doubt.
    """

    threshold: float
    auto_count: int
    auto_correct: int
    review_count: int
    review_correct: int

    @property
    def total(self) -> int:
        return self.auto_count + self.review_count

    @property
    def automation_rate(self) -> float:
        return self.auto_count / self.total if self.total else 0.0

    @property
    def auto_accuracy(self) -> float:
        return self.auto_correct / self.auto_count if self.auto_count else 0.0

    @property
    def review_accuracy(self) -> float:
        return self.review_correct / self.review_count if self.review_count else 0.0

    @classmethod
    def at(cls, pairs: list[tuple[float, bool]], threshold: float) -> RoutingOutcome:
        auto = [(c, ok) for c, ok in pairs if c > threshold]
        review = [(c, ok) for c, ok in pairs if c <= threshold]
        return cls(
            threshold=threshold,
            auto_count=len(auto),
            auto_correct=sum(1 for _, ok in auto if ok),
            review_count=len(review),
            review_correct=sum(1 for _, ok in review if ok),
        )
