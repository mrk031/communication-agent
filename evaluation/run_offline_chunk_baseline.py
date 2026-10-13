#!/usr/bin/env python3
"""Repeatable BM25 proxy baseline over the exact exported RAGFlow chunks."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evaluate import BM25Retriever, percentile, read_jsonl, tokenize


ROOT = Path(__file__).resolve().parent


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def coverage(source: str, target: str) -> float:
    source_tokens, target_tokens = set(tokenize(source)), tokenize(target)
    return sum(token in source_tokens for token in target_tokens) / len(target_tokens) if target_tokens else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "prepared" / "teacher_qa_cases_unlabeled.jsonl")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "prepared" / "ragflow_chunks_corpus.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "offline_chunk_baseline")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    cases, corpus = read_jsonl(args.cases), read_jsonl(args.corpus)
    retriever = BM25Retriever(corpus)
    results: List[Dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        ranked = retriever.search(case["question"], args.top_k, [])
        latency = (time.perf_counter() - started) * 1000
        context = "\n".join(item.content for item in ranked)
        results.append({
            "case_id": case["case_id"], "question": case["question"],
            "reference_answer": case["reference_answer"],
            "chunks": [{"rank": i + 1, "chunk_id": item.doc_id, "score": item.score,
                        "content": item.content} for i, item in enumerate(ranked)],
            "proxy_metrics": {
                "question_coverage_by_context": coverage(context, case["question"]),
                "answer_coverage_by_context": coverage(context, case["reference_answer"]),
            },
            "measurements": {"retrieval_ms": latency, "context_chars": len(context),
                             "context_tokens_estimate": math.ceil(len(context) / 2)},
            "label_status": "unlabeled_proxy_only",
        })
    latencies = [row["measurements"]["retrieval_ms"] for row in results]
    summary = {
        "baseline": "offline_bm25_over_exported_ragflow_chunks",
        "cases": len(results), "chunks": len(corpus), "top_k": args.top_k,
        "proxy_metrics": {
            "mean_question_coverage_by_context": statistics.fmean(row["proxy_metrics"]["question_coverage_by_context"] for row in results),
            "mean_answer_coverage_by_context": statistics.fmean(row["proxy_metrics"]["answer_coverage_by_context"] for row in results),
        },
        "performance": {
            "retrieval_p50_ms": percentile(latencies, .5), "retrieval_p95_ms": percentile(latencies, .95),
            "mean_context_chars": statistics.fmean(row["measurements"]["context_chars"] for row in results),
            "mean_context_tokens_estimate": statistics.fmean(row["measurements"]["context_tokens_estimate"] for row in results),
        },
        "formal_metrics": {"recall_at_k": None, "mrr": None, "reason": "gold chunk labels not yet annotated"},
        "warning": "This is a lexical proxy baseline, not the deployed RAGFlow hybrid retrieval result.",
    }
    output = args.output.resolve()
    write_jsonl(output / "results.jsonl", results)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

