#!/usr/bin/env python3
"""Measure the currently deployed RAGFlow retrieval path without changing server data."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import requests

from export_ragflow import load_env_file


ROOT = Path(__file__).resolve().parent
_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK = re.compile(r"[\u3400-\u9fff]")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokens(text: str) -> List[str]:
    value = str(text or "").lower()
    chars = _CJK.findall(value)
    return _ASCII_WORD.findall(value) + chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def coverage(source: str, target: str) -> float:
    """Fraction of target tokens that occur in source, with multiplicity."""
    target_counts = collections.Counter(tokens(target))
    if not target_counts:
        return 0.0
    source_counts = collections.Counter(tokens(source))
    return sum((source_counts & target_counts).values()) / sum(target_counts.values())


def jaccard(a: str, b: str) -> float:
    left, right = set(tokens(a)), set(tokens(b))
    return len(left & right) / len(left | right) if left | right else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def retrieval_metrics(retrieved_ids: Sequence[str], gold_ids: Sequence[str], ks: Sequence[int]) -> Dict[str, float]:
    """Binary-relevance metrics for one query."""
    gold = {str(value) for value in gold_ids if str(value)}
    retrieved = [str(value) for value in retrieved_ids]
    if not gold:
        return {}
    metrics: Dict[str, float] = {}
    first_rank = next((rank for rank, value in enumerate(retrieved, 1) if value in gold), None)
    metrics["reciprocal_rank"] = 1.0 / first_rank if first_rank else 0.0
    for k in sorted(set(ks)):
        top = retrieved[:k]
        relevant = sum(1 for value in top if value in gold)
        metrics[f"hit_at_{k}"] = float(relevant > 0)
        metrics[f"recall_at_{k}"] = relevant / len(gold)
        metrics[f"precision_at_{k}"] = relevant / k
        dcg = sum(1.0 / math.log2(rank + 1) for rank, value in enumerate(top, 1) if value in gold)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
        metrics[f"ndcg_at_{k}"] = dcg / ideal if ideal else 0.0
    return metrics


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", type=Path, default=ROOT / ".env")
    pre_args, _ = pre.parse_known_args()
    load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(description="Read-only online RAGFlow retrieval baseline")
    parser.add_argument("--env-file", type=Path, default=pre_args.env_file)
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "adjudicated" / "eval_v0" / "gold_v0.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "ragflow_real_baseline")
    parser.add_argument("--base-url", default=os.getenv("RAGFLOW_BASE_URL", "http://127.0.0.1:9380"))
    parser.add_argument("--api-key", default=os.getenv("RAGFLOW_API_KEY"))
    parser.add_argument("--dataset-id", default=os.getenv("RAGFLOW_DATASET_ID", "cf7e12082f8411f1be839bd662726c89"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--embedding-top-k", type=int, default=1024,
                        help="RAGFlow internal vector candidate count; not the returned page size")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--vector-weight", type=float, default=0.3)
    parser.add_argument("--split", choices=("all", "dev", "test"), default="dev")
    parser.add_argument("--eval-k", type=int, action="append", dest="eval_ks",
                        help="Metric cutoff; repeat as needed. Defaults to 1, 4 and 10")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("RAGFLOW_API_KEY is required")

    cases = read_jsonl(args.cases.resolve())
    if args.split != "all":
        cases = [case for case in cases if case.get("split") == args.split]
    if args.limit:
        cases = cases[:args.limit]
    eval_ks = sorted(set(args.eval_ks or [1, 4, 10]))
    if args.candidate_k < max(eval_ks):
        parser.error(f"--candidate-k must be >= largest --eval-k ({max(eval_ks)})")
    endpoint = args.base_url.rstrip("/") + "/api/v1/retrieval"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    output = args.output.resolve()
    results = []
    review_rows = []

    for index, case in enumerate(cases, 1):
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        try:
            response = requests.post(endpoint, headers=headers, json={
                "dataset_ids": [args.dataset_id],
                "question": case["question"],
                "page": 1,
                "page_size": args.candidate_k,
                "similarity_threshold": args.threshold,
                "vector_similarity_weight": args.vector_weight,
                "top_k": args.embedding_top_k,
            }, timeout=args.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            results.append({
                "request_id": request_id, "case_id": case["case_id"],
                "question": case["question"], "reference_answer": case["reference_answer"],
                "status": "error", "error_type": "http_error", "error_message": str(exc),
                "measurements": {"retrieval_ms": latency_ms},
            })
            print(f"[{index}/{len(cases)}] {case['case_id']}: HTTP ERROR")
            continue
        latency_ms = (time.perf_counter() - started) * 1000
        payload = response.json()
        if payload.get("code") not in (0, None):
            results.append({
                "request_id": request_id, "case_id": case["case_id"],
                "question": case["question"], "reference_answer": case["reference_answer"],
                "status": "error", "error_type": "ragflow_retrieval_error",
                "error_message": str(payload.get("message", "unknown error")),
                "measurements": {"retrieval_ms": latency_ms},
            })
            print(f"[{index}/{len(cases)}] {case['case_id']}: ERROR")
            continue
        data = payload.get("data", {})
        chunks = data.get("chunks", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        accepted = chunks[:args.top_k]
        context = "\n".join(str(chunk.get("content_with_weight") or chunk.get("content") or "") for chunk in accepted)
        pairwise = [
            jaccard(str(accepted[i].get("content", "")), str(accepted[j].get("content", "")))
            for i in range(len(accepted)) for j in range(i + 1, len(accepted))
        ]
        scores = [float(chunk.get("similarity", 0) or 0) for chunk in accepted]
        candidate_ids = [str(chunk.get("id", chunk.get("chunk_id", ""))) for chunk in chunks]
        gold_ids = [str(value) for value in case.get("gold_chunk_ids", case.get("gold_doc_ids", []))]
        formal = retrieval_metrics(candidate_ids, gold_ids, eval_ks)
        result = {
            "request_id": request_id,
            "case_id": case["case_id"],
            "question": case["question"],
            "reference_answer": case["reference_answer"],
            "split": case.get("split"),
            "chapter": case.get("chapter"),
            "question_type": case.get("question_type"),
            "difficulty": case.get("difficulty"),
            "gold_chunk_ids": gold_ids,
            "retrieved_count": len(accepted),
            "chunks": [{
                "rank": rank,
                "chunk_id": str(chunk.get("id", chunk.get("chunk_id", ""))),
                "document_id": str(chunk.get("document_id", chunk.get("doc_id", ""))),
                "document_name": str(chunk.get("document_name", chunk.get("docnm_kwd", ""))),
                "similarity": float(chunk.get("similarity", 0) or 0),
                "content": str(chunk.get("content_with_weight") or chunk.get("content") or ""),
            } for rank, chunk in enumerate(accepted, 1)],
            "candidates": [{
                "rank": rank,
                "chunk_id": str(chunk.get("id", chunk.get("chunk_id", ""))),
                "document_id": str(chunk.get("document_id", chunk.get("doc_id", ""))),
                "document_name": str(chunk.get("document_name", chunk.get("docnm_kwd", ""))),
                "similarity": float(chunk.get("similarity", 0) or 0),
                "vector_similarity": float(chunk.get("vector_similarity", 0) or 0),
                "term_similarity": float(chunk.get("term_similarity", 0) or 0),
                "content": str(chunk.get("content_with_weight") or chunk.get("content") or ""),
            } for rank, chunk in enumerate(chunks, 1)],
            "proxy_metrics": {
                "question_coverage_by_context": coverage(context, case["question"]),
                "answer_coverage_by_context": coverage(context, case["reference_answer"]),
                "mean_similarity": statistics.fmean(scores) if scores else 0.0,
                "min_similarity": min(scores) if scores else 0.0,
                "mean_pairwise_chunk_jaccard": statistics.fmean(pairwise) if pairwise else 0.0,
            },
            "retrieval_metrics": formal,
            "measurements": {
                "retrieval_ms": latency_ms,
                "context_chars": len(context),
                "context_tokens_estimate": math.ceil(len(context) / 2),
            },
            "label_status": "gold_labeled" if gold_ids else "unlabeled_proxy_only",
        }
        results.append(result)
        for rank, chunk in enumerate(chunks, 1):
            review_rows.append({
                "case_id": case["case_id"], "question": case["question"],
                "reference_answer": case["reference_answer"], "rank": rank,
                "chunk_id": str(chunk.get("id", chunk.get("chunk_id", ""))),
                "similarity": float(chunk.get("similarity", 0) or 0),
                "content": str(chunk.get("content_with_weight") or chunk.get("content") or ""),
                "relevance_label": None,
                "label_note": "",
            })
        print(f"[{index}/{len(cases)}] {case['case_id']}: {len(accepted)} chunks, {latency_ms:.1f} ms")

    successful = [row for row in results if row.get("status", "success") != "error"]
    latencies = [row["measurements"]["retrieval_ms"] for row in results]
    def mean_metric(name: str) -> float:
        return statistics.fmean(row["proxy_metrics"][name] for row in successful) if successful else 0.0
    labeled = [row for row in successful if row.get("retrieval_metrics")]
    formal_names = ["reciprocal_rank"] + [
        f"{prefix}_at_{k}" for k in eval_ks for prefix in ("hit", "recall", "precision", "ndcg")
    ]
    formal_summary = {
        name: statistics.fmean(row["retrieval_metrics"].get(name, 0.0) for row in labeled)
        for name in formal_names
    } if labeled else {}
    summary = {
        "baseline": "deployed_ragflow_retrieval",
        "cases": len(results), "successful_cases": len(successful),
        "retrieval_success_rate": len(successful) / len(results) if results else 0.0,
        "dataset_id": args.dataset_id,
        "split": args.split,
        "similarity_threshold": args.threshold, "top_k": args.top_k,
        "candidate_k_for_annotation": args.candidate_k,
        "embedding_top_k": args.embedding_top_k,
        "vector_similarity_weight": args.vector_weight,
        "proxy_metrics": {
            "mean_question_coverage_by_context": mean_metric("question_coverage_by_context"),
            "mean_answer_coverage_by_context": mean_metric("answer_coverage_by_context"),
            "mean_similarity": mean_metric("mean_similarity"),
            "mean_min_similarity": mean_metric("min_similarity"),
            "mean_chunk_redundancy_jaccard": mean_metric("mean_pairwise_chunk_jaccard"),
        },
        "performance": {
            "retrieval_p50_ms": percentile(latencies, .5),
            "retrieval_p95_ms": percentile(latencies, .95),
            "mean_context_chars": statistics.fmean(row["measurements"]["context_chars"] for row in successful) if successful else 0,
            "mean_context_tokens_estimate": statistics.fmean(row["measurements"]["context_tokens_estimate"] for row in successful) if successful else 0,
        },
        "formal_metrics": {
            "labeled_cases": len(labeled),
            **formal_summary,
        } if labeled else {"reason": "no gold_chunk_ids present"},
        "safety": {"read_only_retrieval": True, "server_data_modified": False},
    }
    write_jsonl(output / "results.jsonl", results)
    write_jsonl(output / "annotation_candidates.jsonl", review_rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
