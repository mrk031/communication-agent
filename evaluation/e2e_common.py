"""Shared, dependency-light helpers for end-to-end evaluation scripts."""

from __future__ import annotations

import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def estimate_tokens(text: str) -> int:
    """Transparent approximation for providers that do not expose usage."""
    return math.ceil(len(str(text or "")) / 2)


def normalize_units(text: str) -> list[str]:
    value = str(text or "").lower()
    return [char for char in value if "\u3400" <= char <= "\u9fff" or char.isalnum()]


def lexical_scores(prediction: str, reference: str) -> dict[str, float]:
    predicted = collections.Counter(normalize_units(prediction))
    expected = collections.Counter(normalize_units(reference))
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values()) if predicted else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_char_precision": precision,
        "reference_char_recall": recall,
        "reference_char_f1": f1,
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate_numeric(rows: Sequence[dict[str, Any]], path: Sequence[str]) -> dict[str, float | None]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)):
            values.append(float(value))
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }
