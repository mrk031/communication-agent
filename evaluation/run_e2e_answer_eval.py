#!/usr/bin/env python3
"""Controlled no-RAG/e00/e02 answer-path evaluation.

This script calls the same deployed RAGFlow and streaming LLM services while
keeping the model and prompt fixed. It does not invoke the intent/business
branches; use run_agent_e2e.py for the deployed orchestrator's full latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from e2e_common import aggregate_numeric, estimate_tokens, lexical_scores, read_jsonl, write_jsonl
from export_ragflow import load_env_file
from run_ragflow_baseline import retrieval_metrics


ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "no_rag": None,
    "e00": {"threshold": 0.2, "vector_weight": 0.3, "candidate_k": 10, "context_k": 4},
    "e02": {"threshold": 0.2, "vector_weight": 0.6, "candidate_k": 10, "context_k": 4},
}
SYSTEM_PROMPT = (
    "你是一个专业的通信原理助手。请严格遵守以下规则回答：\n"
    "1. 上下文优先原则：如果用户要求'比较'或指代'上面提到的'，你必须优先从历史对话中提取最近的通信概念进行回答。\n"
    "2. 简洁性：在保证准确的前提下，用较少的文字回答。\n"
    "3. 领域锁定：所有回答必须锁定在通信领域内。"
)
RAG_INSTRUCTIONS = {
    "legacy": "请先结合参考知识库进行思考分析；思考结束后，请在独立的段落中给出专业、准确且精简的最终解答。",
    "concise": "请结合参考知识库在独立的段落中给出专业、准确且精简的最终解答。",
}


def build_messages(question: str, context: str, rag_instruction: str = RAG_INSTRUCTIONS["legacy"]) -> list[dict[str, str]]:
    user_content = question
    if context:
        user_content = (
            f"【参考知识库】\n{context}\n\n【用户提问】\n{question}\n\n"
            f"要求：{rag_instruction}"
        )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]


def reusable_result(row: dict[str, Any], rag_prompt_mode: str) -> bool:
    """Only resume a successful row produced with the requested prompt mode."""
    if row.get("status") != "success":
        return False
    if row.get("variant") == "no_rag":
        return True
    return row.get("prompt", {}).get("rag_mode") == rag_prompt_mode


def retrieve(args: argparse.Namespace, question: str, config: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    response = requests.post(
        args.ragflow_url.rstrip("/") + "/api/v1/retrieval",
        headers={"Authorization": f"Bearer {args.ragflow_key}"},
        json={
            "dataset_ids": [args.dataset_id], "question": question, "page": 1,
            "page_size": config["candidate_k"], "top_k": args.embedding_top_k,
            "similarity_threshold": config["threshold"],
            "vector_similarity_weight": config["vector_weight"],
        },
        timeout=args.timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") not in (0, None):
        raise RuntimeError(str(payload.get("message", "RAGFlow retrieval error")))
    data = payload.get("data", {})
    chunks = data.get("chunks", []) if isinstance(data, dict) else data
    return list(chunks or []), (time.perf_counter() - started) * 1000


def stream_answer(args: argparse.Namespace, messages: list[dict[str, str]]) -> tuple[str, str, dict[str, float]]:
    started = time.perf_counter()
    first_event_ms = None
    first_answer_ms = None
    answer_parts: list[str] = []
    reasoning_parts: list[str] = []
    with requests.post(
        args.llm_url, json={"messages": messages, "model": args.model, "is_reasoning": False},
        timeout=args.timeout, stream=True,
    ) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000
            first_event_ms = first_event_ms or elapsed_ms
            data = line[6:]
            if data == "[DONE]":
                break
            payload = json.loads(data)
            if payload.get("type") == "error":
                raise RuntimeError(str(payload.get("content", "LLM stream error")))
            delta = payload.get("choices", [{}])[0].get("delta", {})
            reasoning = str(delta.get("reasoning_content") or "")
            content = str(delta.get("content") or "")
            if reasoning:
                reasoning_parts.append(reasoning)
            if content:
                first_answer_ms = first_answer_ms or elapsed_ms
                answer_parts.append(content)
    return "".join(answer_parts), "".join(reasoning_parts), {
        "llm_first_event_ms": first_event_ms,
        "ttft_ms": first_answer_ms,
        "llm_total_ms": (time.perf_counter() - started) * 1000,
    }


def estimated_cost(args: argparse.Namespace, input_tokens: int, output_tokens: int, embedding_tokens: int) -> float | None:
    prices = (args.input_price, args.output_price, args.embedding_price)
    if not any(prices):
        return None
    return (
        input_tokens * args.input_price + output_tokens * args.output_price
        + embedding_tokens * args.embedding_price
    ) / 1_000_000


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", type=Path, default=ROOT / ".env")
    pre_args, _ = pre.parse_known_args()
    load_env_file(pre_args.env_file)
    parser = argparse.ArgumentParser(description="Controlled end-to-end answer evaluation")
    parser.add_argument("--env-file", type=Path, default=pre_args.env_file)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/adjudicated/eval_v0/gold_v0.jsonl")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--variant", action="append", choices=tuple(VARIANTS), dest="variants")
    parser.add_argument("--output", type=Path, default=ROOT / "output/e2e/answer_dev")
    parser.add_argument("--ragflow-url", default=os.getenv("RAGFLOW_BASE_URL", ""))
    parser.add_argument("--ragflow-key", default=os.getenv("RAGFLOW_API_KEY", ""))
    parser.add_argument("--dataset-id", default=os.getenv("RAGFLOW_DATASET_ID", "cf7e12082f8411f1be839bd662726c89"))
    parser.add_argument("--llm-url", default=os.getenv("E2E_LLM_URL", "http://127.0.0.1:9453/chat"))
    parser.add_argument("--model", default=os.getenv("E2E_MODEL", "GPT"))
    parser.add_argument("--rag-prompt-mode", choices=tuple(RAG_INSTRUCTIONS), default="legacy",
                        help="RAG-only instruction. Use concise to test the shortened prompt.")
    parser.add_argument("--embedding-top-k", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Reuse successful case/variant rows already in results.jsonl")
    parser.add_argument("--input-price", type=float, default=float(os.getenv("E2E_INPUT_PRICE_PER_1M", "0") or 0))
    parser.add_argument("--output-price", type=float, default=float(os.getenv("E2E_OUTPUT_PRICE_PER_1M", "0") or 0))
    parser.add_argument("--embedding-price", type=float, default=float(os.getenv("E2E_EMBEDDING_PRICE_PER_1M", "0") or 0))
    args = parser.parse_args()
    variants = args.variants or ["no_rag", "e00", "e02"]
    if any(VARIANTS[name] for name in variants) and (not args.ragflow_url or not args.ragflow_key):
        parser.error("RAGFLOW_BASE_URL and RAGFLOW_API_KEY are required for RAG variants")

    cases = read_jsonl(args.cases.resolve())
    if args.split != "all":
        cases = [case for case in cases if case.get("split") == args.split]
    if args.limit:
        cases = cases[:args.limit]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.jsonl"
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and result_path.exists():
        existing = {
            (row["case_id"], row["variant"]): row for row in read_jsonl(result_path)
            if reusable_result(row, args.rag_prompt_mode)
        }
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        # Rotate order to reduce systematic cache/time bias among variants.
        ordered = variants[case_index % len(variants):] + variants[:case_index % len(variants)]
        for variant in ordered:
            if (case["case_id"], variant) in existing:
                rows.append(existing[(case["case_id"], variant)])
                print(f"[{case_index + 1}/{len(cases)}] {case['case_id']} {variant}: reused")
                continue
            total_started = time.perf_counter()
            config = VARIANTS[variant]
            try:
                chunks: list[dict[str, Any]] = []
                retrieval_ms = 0.0
                if config:
                    chunks, retrieval_ms = retrieve(args, case["question"], config)
                accepted = chunks[: config["context_k"]] if config else []
                context = "\n".join(str(c.get("content_with_weight") or c.get("content") or "") for c in accepted)
                rag_instruction = RAG_INSTRUCTIONS[args.rag_prompt_mode]
                messages = build_messages(case["question"], context, rag_instruction)
                answer, reasoning, llm_times = stream_answer(args, messages)
                input_tokens = estimate_tokens("\n".join(message["content"] for message in messages))
                answer_tokens = estimate_tokens(answer)
                reasoning_tokens = estimate_tokens(reasoning)
                output_tokens = answer_tokens + reasoning_tokens
                embedding_tokens = estimate_tokens(case["question"]) if config else 0
                candidate_ids = [str(c.get("id", c.get("chunk_id", ""))) for c in chunks]
                gold_ids = [str(value) for value in case.get("gold_chunk_ids", [])]
                row = {
                    "case_id": case["case_id"], "variant": variant, "question": case["question"],
                    "reference_answer": case.get("reference_answer", ""), "answer": answer,
                    "reasoning": reasoning, "split": case.get("split"), "gold_chunk_ids": gold_ids,
                    "retrieved_chunk_ids": candidate_ids, "context": context,
                    "retrieval_metrics": retrieval_metrics(candidate_ids, gold_ids, [1, 4, 10]) if config else {},
                    "answer_proxy_metrics": lexical_scores(answer, case.get("reference_answer", "")),
                    "measurements": {
                        "retrieval_ms": retrieval_ms, **llm_times,
                        "end_to_end_ms": (time.perf_counter() - total_started) * 1000,
                        "context_chars": len(context), "context_tokens_estimate": estimate_tokens(context),
                        "input_tokens_estimate": input_tokens, "output_tokens_estimate": output_tokens,
                        "answer_tokens_estimate": answer_tokens,
                        "reasoning_tokens_estimate": reasoning_tokens,
                        "embedding_tokens_estimate": embedding_tokens,
                        "estimated_cost": estimated_cost(args, input_tokens, output_tokens, embedding_tokens),
                    },
                    "status": "success", "usage_source": "character_estimate_not_provider_usage",
                    "prompt": {
                        "rag_mode": args.rag_prompt_mode if config else "not_applicable",
                        "rag_instruction_sha256": hashlib.sha256(rag_instruction.encode("utf-8")).hexdigest() if config else None,
                    },
                }
            except Exception as exc:
                row = {"case_id": case["case_id"], "variant": variant, "question": case["question"],
                       "status": "error", "error": str(exc),
                       "measurements": {"end_to_end_ms": (time.perf_counter() - total_started) * 1000}}
            rows.append(row)
            print(f"[{case_index + 1}/{len(cases)}] {case['case_id']} {variant}: {row['status']}")

    write_jsonl(result_path, rows)
    summaries: dict[str, Any] = {}
    for variant in variants:
        successful = [row for row in rows if row["variant"] == variant and row["status"] == "success"]
        summaries[variant] = {
            "cases": sum(row["variant"] == variant for row in rows), "successful": len(successful),
            "ttft_ms": aggregate_numeric(successful, ("measurements", "ttft_ms")),
            "llm_first_event_ms": aggregate_numeric(successful, ("measurements", "llm_first_event_ms")),
            "end_to_end_ms": aggregate_numeric(successful, ("measurements", "end_to_end_ms")),
            "retrieval_ms": aggregate_numeric(successful, ("measurements", "retrieval_ms")),
            "input_tokens_estimate": aggregate_numeric(successful, ("measurements", "input_tokens_estimate")),
            "output_tokens_estimate": aggregate_numeric(successful, ("measurements", "output_tokens_estimate")),
            "answer_tokens_estimate": aggregate_numeric(successful, ("measurements", "answer_tokens_estimate")),
            "reasoning_tokens_estimate": aggregate_numeric(successful, ("measurements", "reasoning_tokens_estimate")),
            "estimated_cost": aggregate_numeric(successful, ("measurements", "estimated_cost")),
            "reference_char_f1": aggregate_numeric(successful, ("answer_proxy_metrics", "reference_char_f1")),
        }
    summary = {
        "mode": "controlled_answer_path", "split": args.split, "model": args.model,
        "rag_prompt_mode": args.rag_prompt_mode,
        "rag_instruction": RAG_INSTRUCTIONS[args.rag_prompt_mode],
        "rag_instruction_sha256": hashlib.sha256(RAG_INSTRUCTIONS[args.rag_prompt_mode].encode("utf-8")).hexdigest(),
        "variants": summaries,
        "quality_caveat": "Lexical F1 is a proxy. Formal correctness/faithfulness requires calibrated judging or human review.",
        "usage_caveat": "Token and cost values are estimates unless the provider exposes usage through the wrapper.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
