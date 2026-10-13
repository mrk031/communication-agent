#!/usr/bin/env python3
"""Dependency-free, offline-first evaluation harness for the local project copy."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "data" / "sample_cases.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "sample_corpus.jsonl"
DEFAULT_OUTPUT = ROOT / "output" / "baseline"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: each line must be a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def require_fields(rows: Sequence[Dict[str, Any]], fields: Sequence[str], label: str) -> None:
    seen = set()
    id_field = fields[0]
    for index, row in enumerate(rows, 1):
        missing = [field for field in fields if field not in row]
        if missing:
            raise ValueError(f"{label} row {index} missing fields: {', '.join(missing)}")
        row_id = str(row[id_field]).strip()
        if not row_id:
            raise ValueError(f"{label} row {index} has empty {id_field}")
        if row_id in seen:
            raise ValueError(f"duplicate {id_field}: {row_id}")
        seen.add(row_id)


def validate(cases: Sequence[Dict[str, Any]], corpus: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    require_fields(cases, ("case_id", "question", "reference_answer"), "case")
    require_fields(corpus, ("doc_id", "content"), "corpus")
    doc_ids = {str(row["doc_id"]) for row in corpus}
    warnings = []
    for case in cases:
        unknown = sorted(set(map(str, case.get("gold_doc_ids", []))) - doc_ids)
        if unknown and case.get("split") != "no_answer":
            warnings.append({"case_id": case["case_id"], "unknown_gold_doc_ids": unknown})
    return {"case_count": len(cases), "document_count": len(corpus), "warnings": warnings}


_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[\u3400-\u9fff]")


def tokenize(text: str) -> List[str]:
    """Simple deterministic tokenizer suitable for an offline baseline."""
    normalized = str(text or "").lower()
    ascii_words = _ASCII_WORD.findall(normalized)
    cjk_chars = _CJK.findall(normalized)
    cjk_bigrams = [cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)]
    return ascii_words + cjk_chars + cjk_bigrams


@dataclass
class RankedDocument:
    doc_id: str
    score: float
    content: str
    source_type: str


class BM25Retriever:
    """Small local baseline; it never mutates the supplied corpus."""

    def __init__(self, corpus: Sequence[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.corpus = list(corpus)
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(row.get("content", "")) for row in self.corpus]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_len = statistics.fmean(self.lengths) if self.lengths else 1.0
        self.doc_freq: collections.Counter[str] = collections.Counter()
        for tokens in self.tokens:
            self.doc_freq.update(set(tokens))

    def search(self, query: str, top_k: int, excluded_doc_ids: Sequence[str]) -> List[RankedDocument]:
        query_terms = collections.Counter(tokenize(query))
        excluded = set(map(str, excluded_doc_ids))
        total = max(len(self.corpus), 1)
        ranked: List[RankedDocument] = []
        for row, tokens, doc_len in zip(self.corpus, self.tokens, self.lengths):
            doc_id = str(row["doc_id"])
            if doc_id in excluded:
                continue
            frequencies = collections.Counter(tokens)
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
                score += query_frequency * idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                ranked.append(RankedDocument(
                    doc_id=doc_id,
                    score=score,
                    content=str(row.get("content", "")),
                    source_type=str(row.get("source_type", "unknown")),
                ))
        ranked.sort(key=lambda item: (-item.score, item.doc_id))
        return ranked[:top_k]


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def text_metrics(prediction: str, reference: str) -> Dict[str, float]:
    predicted_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    predicted_counts = collections.Counter(predicted_tokens)
    reference_counts = collections.Counter(reference_tokens)
    overlap = sum((predicted_counts & reference_counts).values())
    precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    compact = lambda value: re.sub(r"\s+", "", str(value or "")).lower()
    return {
        "exact_match": float(bool(reference) and compact(prediction) == compact(reference)),
        "token_precision": precision,
        "reference_coverage": recall,
        "token_f1": f1,
    }


def evaluate_case(case: Dict[str, Any], retriever: BM25Retriever, top_k: int) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    excluded = list(map(str, case.get("holdout_doc_ids", [])))
    retrieval_started = time.perf_counter()
    ranked = retriever.search(str(case["question"]), top_k, excluded)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    returned_ids = [item.doc_id for item in ranked]
    gold = set(map(str, case.get("gold_doc_ids", [])))
    hits = [index + 1 for index, doc_id in enumerate(returned_ids) if doc_id in gold]
    no_answer = case.get("split") == "no_answer" or not gold
    predicted_answer = str(case.get("predicted_answer", ""))
    context_chars = sum(len(item.content) for item in ranked)
    total_ms = (time.perf_counter() - started) * 1000

    result: Dict[str, Any] = {
        "request_id": request_id,
        "case_id": str(case["case_id"]),
        "split": str(case.get("split", "test")),
        "question": str(case["question"]),
        "excluded_doc_ids": excluded,
        "returned": [
            {"rank": index + 1, "doc_id": item.doc_id, "score": round(item.score, 6),
             "source_type": item.source_type}
            for index, item in enumerate(ranked)
        ],
        "retrieval": {
            "recall_at_k": (len(set(returned_ids) & gold) / len(gold)) if gold else None,
            "precision_at_k": (len(set(returned_ids) & gold) / len(returned_ids)) if returned_ids and gold else (0.0 if gold else None),
            "mrr": (1.0 / min(hits)) if hits else (0.0 if gold else None),
            "no_answer_false_retrieval": bool(no_answer and ranked),
        },
        "answer": text_metrics(predicted_answer, str(case["reference_answer"])) if predicted_answer else None,
        "measurements": {
            "retrieval_ms": retrieval_ms,
            "total_ms": total_ms,
            "retrieved_context_chars": context_chars,
            "retrieved_context_tokens_estimate": math.ceil(context_chars / 2),
        },
    }
    return result


def mean_available(results: Sequence[Dict[str, Any]], path: Sequence[str]) -> float | None:
    values = []
    for result in results:
        value: Any = result
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return statistics.fmean(values) if values else None


def build_summary(results: Sequence[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    latencies = [float(row["measurements"]["total_ms"]) for row in results]
    summary = {
        "case_count": len(results),
        "top_k": top_k,
        "retrieval": {
            "mean_recall_at_k": mean_available(results, ("retrieval", "recall_at_k")),
            "mean_precision_at_k": mean_available(results, ("retrieval", "precision_at_k")),
            "mean_mrr": mean_available(results, ("retrieval", "mrr")),
            "no_answer_false_retrieval_rate": statistics.fmean(
                float(row["retrieval"]["no_answer_false_retrieval"])
                for row in results if row["retrieval"]["recall_at_k"] is None
            ) if any(row["retrieval"]["recall_at_k"] is None for row in results) else None,
        },
        "answer": {
            "mean_exact_match": mean_available(results, ("answer", "exact_match")),
            "mean_token_f1": mean_available(results, ("answer", "token_f1")),
            "mean_reference_coverage": mean_available(results, ("answer", "reference_coverage")),
        },
        "performance": {
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "mean_context_chars": mean_available(results, ("measurements", "retrieved_context_chars")),
            "mean_context_tokens_estimate": mean_available(results, ("measurements", "retrieved_context_tokens_estimate")),
        },
        "safety": {
            "network_used": False,
            "source_files_modified": False,
            "holdout_mode": "in_memory_doc_id_filter",
        },
    }
    return summary


def trace_rows(results: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for row in results:
        yield {
            "request_id": row["request_id"],
            "case_id": row["case_id"],
            "events": [
                {"phase": "retrieve", "duration_ms": row["measurements"]["retrieval_ms"],
                 "returned_count": len(row["returned"]),
                 "context_chars": row["measurements"]["retrieved_context_chars"]},
                {"phase": "complete", "duration_ms": row["measurements"]["total_ms"]},
            ],
        }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation harness (no network, no service startup).")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--cases", type=Path, default=DEFAULT_CASES)
        sub.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
        if name == "run":
            sub.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
            sub.add_argument("--top-k", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cases = read_jsonl(args.cases.resolve())
    corpus = read_jsonl(args.corpus.resolve())
    report = validate(cases, corpus)
    if args.command == "validate":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    retriever = BM25Retriever(corpus)
    results = [evaluate_case(case, retriever, args.top_k) for case in cases]
    output = args.output.resolve()
    write_json(output / "summary.json", build_summary(results, args.top_k))
    write_jsonl(output / "predictions.jsonl", results)
    write_jsonl(output / "traces.jsonl", trace_rows(results))
    print(json.dumps({"validation": report, "summary": build_summary(results, args.top_k),
                      "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

