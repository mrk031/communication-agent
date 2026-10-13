#!/usr/bin/env python3
"""Create paired comparisons for controlled end-to-end answer experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from e2e_common import read_jsonl


def value(row: dict[str, Any], *path: str) -> float | None:
    current: Any = row
    for key in path:
        current = current.get(key) if isinstance(current, dict) else None
    return float(current) if isinstance(current, (int, float)) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Controlled E2E output directory or results.jsonl")
    parser.add_argument("--baseline", default="no_rag")
    parser.add_argument("--challenger", action="append", dest="challengers")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    results_path = source / "results.jsonl" if source.is_dir() else source
    output = args.output.resolve() if args.output else results_path.parent / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_jsonl(results_path) if row.get("status") == "success"]
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["variant"]] = row
    challengers = args.challengers or [name for name in ("e00", "e02") if name != args.baseline]
    metrics = {
        "reference_char_f1": ("answer_proxy_metrics", "reference_char_f1"),
        "ttft_ms": ("measurements", "ttft_ms"),
        "end_to_end_ms": ("measurements", "end_to_end_ms"),
        "input_tokens_estimate": ("measurements", "input_tokens_estimate"),
        "estimated_cost": ("measurements", "estimated_cost"),
    }
    comparisons: dict[str, Any] = {}
    review_rows: list[dict[str, Any]] = []
    for challenger in challengers:
        pairs = [(variants[args.baseline], variants[challenger]) for variants in by_case.values()
                 if args.baseline in variants and challenger in variants]
        differences: dict[str, Any] = {}
        for metric, path in metrics.items():
            deltas = []
            for baseline, candidate in pairs:
                left, right = value(baseline, *path), value(candidate, *path)
                if left is not None and right is not None:
                    deltas.append(right - left)
            differences[metric] = {
                "paired_cases": len(deltas), "mean_delta_challenger_minus_baseline": statistics.fmean(deltas) if deltas else None,
                "challenger_better": sum(delta > 0 for delta in deltas) if metric == "reference_char_f1" else None,
            }
        comparisons[f"{challenger}_vs_{args.baseline}"] = {"paired_cases": len(pairs), "deltas": differences}
        for baseline, candidate in pairs:
            review_rows.append({
                "case_id": baseline["case_id"], "comparison": f"{challenger}_vs_{args.baseline}",
                "question": baseline["question"], "reference_answer": baseline.get("reference_answer", ""),
                "baseline_answer": baseline.get("answer", ""), "challenger_answer": candidate.get("answer", ""),
                "baseline_f1": value(baseline, "answer_proxy_metrics", "reference_char_f1"),
                "challenger_f1": value(candidate, "answer_proxy_metrics", "reference_char_f1"),
                "human_winner": "", "correctness_baseline_0_2": "", "correctness_challenger_0_2": "",
                "faithfulness_challenger_0_2": "", "human_note": "",
            })
    (output / "paired_summary.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")
    if review_rows:
        with (output / "paired_answer_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
            writer.writeheader()
            writer.writerows(review_rows)
    print(json.dumps({"comparisons": comparisons, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
