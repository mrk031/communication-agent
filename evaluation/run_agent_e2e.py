#!/usr/bin/env python3
"""Measure the deployed /tasks_distribute SSE request from a user's perspective."""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from e2e_common import aggregate_numeric, estimate_tokens, lexical_scores, percentile, read_jsonl, write_jsonl
from export_ragflow import load_env_file


ROOT = Path(__file__).resolve().parent


def run_case(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    first_event_ms = None
    first_user_visible_event_ms = None
    ttft_ms = None
    answer_complete_ms = None
    first_section_ms = None
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    event_counts: collections.Counter[str] = collections.Counter()
    section_keys: list[str] = []
    selected_business_types: list[str] = []
    trace_events: list[dict[str, Any]] = []
    config = {"content": case["question"], "model": args.model, "is_reasoning": False}
    try:
        with requests.post(
            args.agent_url,
            data={"config": json.dumps(config, ensure_ascii=False),
                  "entrance": json.dumps({"mode": "NORMAL"}),
                  "locale": json.dumps({"language": "ZH"})},
            timeout=args.timeout, stream=True,
        ) as response:
            response.raise_for_status()
            response.encoding = "utf-8"
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                elapsed = (time.perf_counter() - started) * 1000
                first_event_ms = first_event_ms or elapsed
                data = line[6:]
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                event_type = str(payload.get("type", "unknown"))
                event_counts[event_type] += 1
                if event_type != "trace":
                    first_user_visible_event_ms = first_user_visible_event_ms or elapsed
                if event_type == "token":
                    content = str(payload.get("content", ""))
                    if content:
                        ttft_ms = ttft_ms or elapsed
                        answer_parts.append(content)
                elif event_type == "reasoning_token":
                    reasoning_parts.append(str(payload.get("content", "")))
                elif event_type in ("answer_complete", "next_questions_start"):
                    answer_complete_ms = answer_complete_ms or elapsed
                elif event_type == "section":
                    first_section_ms = first_section_ms or elapsed
                    section_keys.append(str(payload.get("key", "")))
                elif event_type == "config":
                    config_data = payload.get("data", {})
                    if isinstance(config_data, dict):
                        selected_business_types = [
                            str(key) for key, enabled in config_data.items()
                            if key != "问题内容" and enabled == 1
                        ]
                elif event_type == "trace":
                    trace_events.append(payload)
        total_ms = (time.perf_counter() - started) * 1000
        answer_complete_ms = answer_complete_ms or total_ms
        answer = "".join(answer_parts)
        return {
            "case_id": case["case_id"], "question": case["question"],
            "reference_answer": case.get("reference_answer", ""), "answer": answer,
            "reasoning": "".join(reasoning_parts), "status": "success" if answer else "error",
            "event_counts": dict(event_counts), "section_keys": section_keys,
            "selected_business_types": selected_business_types,
            "trace_events": trace_events,
            "answer_proxy_metrics": lexical_scores(answer, case.get("reference_answer", "")),
            "measurements": {
                "first_sse_event_ms": first_event_ms, "ttft_ms": ttft_ms,
                "first_user_visible_event_ms": first_user_visible_event_ms,
                "first_section_ms": first_section_ms,
                "answer_complete_ms": answer_complete_ms, "full_stream_ms": total_ms,
                "answer_tokens_estimate": estimate_tokens(answer),
                "reasoning_tokens_estimate": estimate_tokens("".join(reasoning_parts)),
                "workflow_tail_ms": max(total_ms - answer_complete_ms, 0.0),
            },
            "trace_scope": "deployed_agent_black_box",
        }
    except Exception as exc:
        return {"case_id": case["case_id"], "question": case["question"], "status": "error",
                "error": str(exc), "measurements": {"full_stream_ms": (time.perf_counter() - started) * 1000}}


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", type=Path, default=ROOT / ".env")
    pre_args, _ = pre.parse_known_args()
    load_env_file(pre_args.env_file)
    parser = argparse.ArgumentParser(description="Black-box deployed agent end-to-end evaluation")
    parser.add_argument("--env-file", type=Path, default=pre_args.env_file)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/adjudicated/eval_v0/gold_v0.jsonl")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--agent-url", default=os.getenv("AGENT_BASE_URL", "http://127.0.0.1:9450") + "/tasks_distribute")
    parser.add_argument("--model", default=os.getenv("E2E_MODEL", "GPT"))
    parser.add_argument("--output", type=Path, default=ROOT / "output/e2e/agent_dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--resume", action="store_true", help="Reuse successful rows already in results.jsonl")
    args = parser.parse_args()
    cases = read_jsonl(args.cases.resolve())
    if args.split != "all":
        cases = [case for case in cases if case.get("split") == args.split]
    if args.limit:
        cases = cases[:args.limit]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and result_path.exists():
        existing = {
            row["case_id"]: row for row in read_jsonl(result_path)
            if row.get("status") == "success"
        }
    rows = []
    for index, case in enumerate(cases, 1):
        if case["case_id"] in existing:
            row = existing[case["case_id"]]
            rows.append(row)
            print(f"[{index}/{len(cases)}] {case['case_id']}: reused")
            continue
        row = run_case(args, case)
        rows.append(row)
        print(f"[{index}/{len(cases)}] {case['case_id']}: {row['status']}")
        write_jsonl(result_path, rows)
    write_jsonl(result_path, rows)
    successful = [row for row in rows if row["status"] == "success"]
    event_totals: collections.Counter[str] = collections.Counter()
    section_totals: collections.Counter[str] = collections.Counter()
    selected_business_totals: collections.Counter[str] = collections.Counter()
    trace_durations: dict[str, list[float]] = collections.defaultdict(list)
    for row in successful:
        event_totals.update(row.get("event_counts", {}))
        section_totals.update(row.get("section_keys", []))
        selected_business_totals.update(row.get("selected_business_types", []))
        for event in row.get("trace_events", []):
            duration = event.get("duration_ms")
            if event.get("status") == "end" and isinstance(duration, (int, float)):
                stage = str(event.get("stage", "unknown"))
                metadata = event.get("metadata", {})
                if stage == "business" and isinstance(metadata, dict) and metadata.get("key"):
                    stage = f"business:{metadata['key']}"
                trace_durations[stage].append(float(duration))
    summary = {
        "mode": "deployed_agent_black_box", "split": args.split, "model": args.model,
        "agent_url": args.agent_url,
        "cases": len(rows), "successful": len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "first_sse_event_ms": aggregate_numeric(successful, ("measurements", "first_sse_event_ms")),
        "first_user_visible_event_ms": aggregate_numeric(successful, ("measurements", "first_user_visible_event_ms")),
        "ttft_ms": aggregate_numeric(successful, ("measurements", "ttft_ms")),
        "first_section_ms": aggregate_numeric(successful, ("measurements", "first_section_ms")),
        "answer_complete_ms": aggregate_numeric(successful, ("measurements", "answer_complete_ms")),
        "full_stream_ms": aggregate_numeric(successful, ("measurements", "full_stream_ms")),
        "workflow_tail_ms": aggregate_numeric(successful, ("measurements", "workflow_tail_ms")),
        "answer_tokens_estimate": aggregate_numeric(successful, ("measurements", "answer_tokens_estimate")),
        "reasoning_tokens_estimate": aggregate_numeric(successful, ("measurements", "reasoning_tokens_estimate")),
        "reference_char_f1": aggregate_numeric(successful, ("answer_proxy_metrics", "reference_char_f1")),
        "event_totals": dict(event_totals),
        "section_totals": dict(section_totals),
        "selected_business_totals": dict(selected_business_totals),
        "trace_stage_ms": {
            stage: {
                "count": len(values),
                "mean": sum(values) / len(values),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
            }
            for stage, values in sorted(trace_durations.items())
        },
        "scope_caveat": "Black-box agent metrics include intent/business/next-question work but cannot isolate internal RAG or exact provider token usage.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
