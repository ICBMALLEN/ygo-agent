"""Pure-Python supervised nearest-neighbor model for masked deck completion."""

from __future__ import annotations

from collections import Counter
import heapq
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


MODEL_FORMAT = "ygoai.deck-mask-knn.v1"


def _feature_counts(observed_main: Sequence[int], observed_extra: Sequence[int]) -> Counter[str]:
    counts: Counter[str] = Counter()
    counts.update(f"main:{int(code)}" for code in observed_main)
    counts.update(f"extra:{int(code)}" for code in observed_extra)
    return counts


def _zone_counts(value: Mapping[Any, Any] | None) -> dict[int, float]:
    return {int(code): float(count) for code, count in (value or {}).items() if float(count) > 0.0}


class DeckMaskPredictor:
    """Predict residual Main/Extra counts from 1-10 visible card instances.

    The fit step stores supervised masked examples. Inference uses an IDF-weighted
    multiset similarity over zone-tagged visible cards and averages the hidden-count
    labels of the nearest examples. It uses only the Python standard library.
    """

    def __init__(self, examples: list[dict[str, Any]], idf: dict[str, float], neighbors: int = 8):
        if not examples:
            raise ValueError("predictor must contain at least one training sample")
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        self.examples = examples
        self.idf = idf
        self.neighbors = neighbors
        self._features = [
            _feature_counts(example["observed_main"], example["observed_extra"])
            for example in self.examples
        ]
        self._feature_norms = [
            sum(self.idf.get(feature, 0.0) * count for feature, count in features.items())
            for features in self._features
        ]
        self.prior_main: dict[int, float] = {}
        self.prior_extra: dict[int, float] = {}
        self._build_priors()

    @classmethod
    def fit(cls, samples: Sequence[Mapping[str, Any]], neighbors: int = 8) -> "DeckMaskPredictor":
        if not samples:
            raise ValueError("cannot fit a predictor without training samples")
        examples = []
        document_frequency: Counter[str] = Counter()
        for sample in samples:
            features = _feature_counts(sample.get("observed_main", []), sample.get("observed_extra", []))
            if not features:
                raise ValueError("training examples must have at least one observed card")
            document_frequency.update(features.keys())
            examples.append(
                {
                    "deck_id": str(sample.get("deck_id", "")),
                    "observed_main": [int(code) for code in sample.get("observed_main", [])],
                    "observed_extra": [int(code) for code in sample.get("observed_extra", [])],
                    "target_main": {str(code): int(count) for code, count in sample.get("target_main", {}).items() if int(count) > 0},
                    "target_extra": {str(code): int(count) for code, count in sample.get("target_extra", {}).items() if int(count) > 0},
                }
            )
        sample_count = len(examples)
        idf = {
            feature: math.log(1.0 + (sample_count - frequency + 0.5) / (frequency + 0.5))
            for feature, frequency in document_frequency.items()
        }
        return cls(examples=examples, idf=idf, neighbors=neighbors)

    def _build_priors(self) -> None:
        total = 0
        sums_main: Counter[int] = Counter()
        sums_extra: Counter[int] = Counter()
        for sample in self.examples:
            sums_main.update({int(code): int(count) for code, count in sample["target_main"].items()})
            sums_extra.update({int(code): int(count) for code, count in sample["target_extra"].items()})
            total += 1
        if total:
            self.prior_main = {code: count / total for code, count in sums_main.items()}
            self.prior_extra = {code: count / total for code, count in sums_extra.items()}

    def _similarity(
        self,
        query: Counter[str],
        query_norm: float,
        candidate: Counter[str],
        candidate_norm: float,
    ) -> float:
        if query_norm + candidate_norm == 0.0:
            return 0.0
        overlap = sum(
            self.idf.get(feature, 0.0) * min(count, candidate.get(feature, 0))
            for feature, count in query.items()
        )
        return 2.0 * overlap / (query_norm + candidate_norm)

    def predict(
        self,
        observed_main: Sequence[int],
        observed_extra: Sequence[int],
        *,
        neighbors: int | None = None,
    ) -> dict[str, dict[int, float]]:
        """Return expected hidden counts by zone; observed cards stay zone-tagged."""
        query = _feature_counts(observed_main, observed_extra)
        if not query:
            return {"main": dict(self.prior_main), "extra": dict(self.prior_extra)}
        query_norm = sum(self.idf.get(feature, 0.0) * count for feature, count in query.items())
        limit = self.neighbors if neighbors is None else neighbors
        if limit < 1:
            raise ValueError("neighbors must be positive")
        selected_indexes = heapq.nsmallest(
            limit,
            (
                (-similarity, index)
                for index, (candidate, candidate_norm) in enumerate(zip(self._features, self._feature_norms))
                if (similarity := self._similarity(query, query_norm, candidate, candidate_norm)) > 0.0
            ),
        )
        selected = [(-negative_similarity, index, self.examples[index]) for negative_similarity, index in selected_indexes]
        if not selected:
            return {"main": dict(self.prior_main), "extra": dict(self.prior_extra)}
        weights = [similarity * similarity for similarity, _index, _example in selected]
        weight_sum = sum(weights)
        prediction_main: Counter[int] = Counter()
        prediction_extra: Counter[int] = Counter()
        for weight, (_similarity, _index, example) in zip(weights, selected):
            for code, count in example["target_main"].items():
                prediction_main[int(code)] += weight * int(count) / weight_sum
            for code, count in example["target_extra"].items():
                prediction_extra[int(code)] += weight * int(count) / weight_sum
        return {
            "main": {code: count for code, count in prediction_main.items() if count > 0.0},
            "extra": {code: count for code, count in prediction_extra.items() if count > 0.0},
        }

    def predict_top(
        self,
        observed_main: Sequence[int],
        observed_extra: Sequence[int],
        limit: int = 20,
    ) -> dict[str, list[dict[str, float | int]]]:
        if limit < 0:
            raise ValueError("limit cannot be negative")
        prediction = self.predict(observed_main, observed_extra)
        return {
            zone: [
                {"code": code, "expected_count": round(count, 6)}
                for code, count in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]
            ]
            for zone, values in prediction.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MODEL_FORMAT,
            "neighbors": self.neighbors,
            "idf": self.idf,
            "examples": self.examples,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary_name = stream.name
                stream.write(payload)
            Path(temporary_name).replace(path)
        finally:
            if temporary_name and Path(temporary_name).exists():
                Path(temporary_name).unlink()

    @classmethod
    def load(cls, path: str | Path) -> "DeckMaskPredictor":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("format") != MODEL_FORMAT:
            raise ValueError(f"unsupported model format: {value.get('format')!r}")
        return cls(
            examples=value["examples"],
            idf={str(feature): float(weight) for feature, weight in value["idf"].items()},
            neighbors=int(value["neighbors"]),
        )
