#!/usr/bin/env python3
"""Analyze one RAGFlow retrieval experiment and export failure cases."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(rows: Iterable[dict[str, Any]], metric: str) -> float | None:
    values = [row.get("retrieval_metrics", {}).get(metric) for row in rows]
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def breakdown(rows: list[dict[str, Any]], field: str, cutoff: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "unknown")].append(row)
    return {
        key: {
            "cases": len(items),
            f"hit_at_{cutoff}": mean(items, f"hit_at_{cutoff}"),
            f"recall_at_{cutoff}": mean(items, f"recall_at_{cutoff}"),
            "mrr": mean(items, "reciprocal_rank"),
            "retrieval_mean_ms": statistics.fmean(item["measurements"]["retrieval_ms"] for item in items),
        }
        for key, items in sorted(groups.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Experiment output directory or results.jsonl")
    parser.add_argument("--cutoff", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    results_path = source / "results.jsonl" if source.is_dir() else source
    output = args.output.resolve() if args.output else results_path.parent / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_jsonl(results_path) if row.get("status", "success") != "error"]
    labeled = [row for row in rows if row.get("retrieval_metrics")]
    metric = f"hit_at_{args.cutoff}"
    misses = [row for row in labeled if row["retrieval_metrics"].get(metric, 0) == 0]

    analysis = {
        "cases": len(rows), "labeled_cases": len(labeled), "cutoff": args.cutoff,
        "overall": {
            metric: mean(labeled, metric),
            f"recall_at_{args.cutoff}": mean(labeled, f"recall_at_{args.cutoff}"),
            f"ndcg_at_{args.cutoff}": mean(labeled, f"ndcg_at_{args.cutoff}"),
            "mrr": mean(labeled, "reciprocal_rank"),
            "misses": len(misses),
        },
        "by_chapter": breakdown(labeled, "chapter", args.cutoff),
        "by_question_type": breakdown(labeled, "question_type", args.cutoff),
        "by_difficulty": breakdown(labeled, "difficulty", args.cutoff),
    }
    (output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "misses.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["case_id", "question", "chapter", "question_type", "difficulty", "gold_chunk_ids",
                  "top_retrieved_ids", "top_scores", "top_contents"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in misses:
            candidates = row.get("candidates", row.get("chunks", []))[:args.cutoff]
            writer.writerow({
                "case_id": row["case_id"], "question": row["question"], "chapter": row.get("chapter", ""),
                "question_type": row.get("question_type", ""), "difficulty": row.get("difficulty", ""),
                "gold_chunk_ids": "|".join(row.get("gold_chunk_ids", [])),
                "top_retrieved_ids": "|".join(item.get("chunk_id", "") for item in candidates),
                "top_scores": "|".join(str(item.get("similarity", "")) for item in candidates),
                "top_contents": " || ".join(str(item.get("content", "")).replace("\n", " ")[:250] for item in candidates),
            })
    print(json.dumps({**analysis, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
