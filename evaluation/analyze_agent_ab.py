#!/usr/bin/env python3
"""Paired comparison of two run_agent_e2e.py result directories."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from e2e_common import percentile, read_jsonl


METRICS = (
    "ttft_ms",
    "answer_complete_ms",
    "full_stream_ms",
    "workflow_tail_ms",
    "answer_tokens_estimate",
    "reasoning_tokens_estimate",
)


def bootstrap_mean_ci(values: list[float], rounds: int = 10000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(20261010)
    size = len(values)
    means = [statistics.fmean(rng.choices(values, k=size)) for _ in range(rounds)]
    return [float(percentile(means, 0.025)), float(percentile(means, 0.975))]


def summarize_delta(values: list[float], lower_is_better: bool) -> dict[str, Any]:
    return {
        "mean_delta": statistics.fmean(values),
        "median_delta": statistics.median(values),
        "p95_delta": percentile(values, 0.95),
        "mean_delta_bootstrap_95ci": bootstrap_mean_ci(values),
        "candidate_better": sum(value < 0 if lower_is_better else value > 0 for value in values),
        "baseline_better": sum(value > 0 if lower_is_better else value < 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "count": len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired A/B analysis for full Agent black-box runs")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = {row["case_id"]: row for row in read_jsonl(args.baseline / "results.jsonl")}
    candidate = {row["case_id"]: row for row in read_jsonl(args.candidate / "results.jsonl")}
    case_ids = sorted(set(baseline) & set(candidate))
    report: dict[str, Any] = {
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "paired_cases": len(case_ids),
        "only_baseline": sorted(set(baseline) - set(candidate)),
        "only_candidate": sorted(set(candidate) - set(baseline)),
        "metrics": {},
    }
    for metric in METRICS:
        deltas = [
            float(candidate[case_id]["measurements"][metric])
            - float(baseline[case_id]["measurements"][metric])
            for case_id in case_ids
        ]
        report["metrics"][metric] = summarize_delta(deltas, lower_is_better=True)

    f1_deltas = [
        float(candidate[case_id]["answer_proxy_metrics"]["reference_char_f1"])
        - float(baseline[case_id]["answer_proxy_metrics"]["reference_char_f1"])
        for case_id in case_ids
    ]
    report["metrics"]["reference_char_f1"] = summarize_delta(
        f1_deltas, lower_is_better=False
    )
    ranked = sorted(zip(f1_deltas, case_ids))
    report["largest_f1_decreases"] = [
        {
            "case_id": case_id,
            "delta": delta,
            "question": candidate[case_id]["question"],
            "baseline_answer": baseline[case_id]["answer"],
            "candidate_answer": candidate[case_id]["answer"],
        }
        for delta, case_id in ranked[:5]
    ]
    report["largest_f1_increases"] = [
        {
            "case_id": case_id,
            "delta": delta,
            "question": candidate[case_id]["question"],
            "baseline_answer": baseline[case_id]["answer"],
            "candidate_answer": candidate[case_id]["answer"],
        }
        for delta, case_id in reversed(ranked[-5:])
    ]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
