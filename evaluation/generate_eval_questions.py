#!/usr/bin/env python3
"""Generate reviewable RAG evaluation questions from exported RAGFlow chunks.

The script reads only the local chunk export. It never calls or changes
RAGFlow. Model calls use an OpenAI-compatible chat-completions endpoint.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from export_ragflow import load_env_file


ROOT = Path(__file__).resolve().parent
CJK_RE = re.compile(r"[\u3400-\u9fff]")
CHAPTER_RE = re.compile(r"第\s*([0-9一二三四五六七八九十百]+)\s*章")
QUESTION_TYPES = {"definition", "mechanism", "comparison", "calculation", "application"}
DIFFICULTIES = {"easy", "medium", "hard"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_chapter(text: str, previous: str = "unknown") -> str:
    match = CHAPTER_RE.search(text[:200])
    return f"第{match.group(1)}章" if match else previous


def useful_chunk(text: str, min_chars: int = 20) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    cjk = CJK_RE.findall(compact)
    citation_marks = len(re.findall(r"[\[［][0-9]{1,3}[\]］]", text))
    bibliography_terms = sum(text.count(term) for term in ("参考文献", "出版社", "Press", "IEEE Trans"))
    looks_like_bibliography = (
        citation_marks >= 4
        or (citation_marks >= 2 and bibliography_terms >= 2)
        or (citation_marks >= 1 and bibliography_terms >= 1)
    )
    looks_like_toc = text.count("……") + text.count("····") >= 3
    return (
        len(compact) >= min_chars
        and len(cjk) >= 12
        and len(set(cjk)) >= 8
        and not looks_like_bibliography
        and not looks_like_toc
    )


def build_passages(
    chunks: Sequence[dict[str, Any]], target_chars: int = 450, max_chars: int = 900
) -> list[dict[str, Any]]:
    """Combine consecutive short chunks without crossing documents/chapters."""
    passages: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    current_doc = ""
    current_chapter = "unknown"
    chapter_by_doc: dict[str, str] = {}

    def flush() -> None:
        nonlocal current, current_chars
        if current and current_chars >= 100:
            chunk_ids = [str(item["chunk_id"]) for item in current]
            passages.append({
                "passage_id": hashlib.sha1("|".join(chunk_ids).encode()).hexdigest()[:16],
                "document_name": current[0].get("document_name", ""),
                "chapter": current_chapter,
                "chunk_ids": chunk_ids,
                "chunks": [
                    {"chunk_id": str(item["chunk_id"]), "content": str(item.get("content", ""))}
                    for item in current
                ],
                "content_chars": current_chars,
            })
        current, current_chars = [], 0

    for item in chunks:
        text = str(item.get("content", ""))
        if not useful_chunk(text):
            continue
        doc = str(item.get("document_name", item.get("doc_id", "unknown")))
        previous_chapter = chapter_by_doc.get(doc, "unknown")
        chapter = infer_chapter(text, previous_chapter)
        chapter_by_doc[doc] = chapter
        compact_len = len(re.sub(r"\s+", "", text))
        if compact_len > max_chars:
            # Do not silently truncate evidence: the exported chunk ID must refer
            # to exactly the text shown to the generator and reviewer.
            continue
        boundary = bool(current) and (doc != current_doc or chapter != current_chapter)
        overflow = bool(current) and current_chars + compact_len > max_chars
        if boundary or overflow:
            flush()
        current_doc, current_chapter = doc, chapter
        current.append(item)
        current_chars += compact_len
        if current_chars >= target_chars:
            flush()
    flush()
    return passages


def stratified_sample(passages: Sequence[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for passage in passages:
        if passage["chapter"] == "unknown":
            continue
        strata[(passage["document_name"], passage["chapter"])].append(passage)
    for rows in strata.values():
        rng.shuffle(rows)
    keys = list(strata)
    rng.shuffle(keys)
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < min(size, len(passages)):
        next_keys: list[tuple[str, str]] = []
        for key in keys:
            if strata[key] and len(selected) < size:
                selected.append(strata[key].pop())
            if strata[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def passage_prompt(passage: dict[str, Any], questions_per_passage: int) -> str:
    segments = "\n\n".join(
        f"<chunk id=\"{item['chunk_id']}\">\n{item['content']}\n</chunk>"
        for item in passage["chunks"]
    )
    return f"""你正在为《通信原理》RAG系统构建检索评测集。只能依据下方证据出题。

生成 {questions_per_passage} 道彼此不同、学生可能自然提出的问题。每道题需满足：
1. source_question贴近教材表述；paraphrased_question采用自然口语和同义表达，不能照抄原句。
2. reference_answer必须完全由证据推出，不补充外部知识。
3. evidence_quote必须逐字摘自证据，且足以支撑答案。
4. supporting_chunk_ids只列真正包含证据的chunk id，可为多个。
5. 避免目录、页码、残缺公式、纯记忆章节号，以及“根据上文”一类依赖提示。
6. question_type只能是 definition/mechanism/comparison/calculation/application。
7. difficulty只能是 easy/medium/hard。

仅输出一个JSON对象，不要Markdown：
{{"items":[{{"source_question":"...","paraphrased_question":"...","reference_answer":"...","evidence_quote":"...","supporting_chunk_ids":["..."],"question_type":"definition","difficulty":"medium"}}]}}

证据：
{segments}"""


def call_model(
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: float,
    json_mode: bool,
    max_retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "你是严谨的数据集构建助手，输出必须忠于给定证据。"},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            started = time.perf_counter()
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            elapsed_ms = (time.perf_counter() - started) * 1000
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            parsed = json.loads(str(content).strip().removeprefix("```json").removesuffix("```").strip())
            meta = {"latency_ms": elapsed_ms, "usage": payload.get("usage", {}), "model": payload.get("model", model)}
            return parsed, meta
        except (requests.RequestException, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"generation failed after {max_retries + 1} attempts: {last_error}")


def normalized(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9\u3400-\u9fff]", "", str(text or "")).lower()


def char_overlap(left: str, right: str) -> float:
    left_norm, right_norm = normalized(left), normalized(right)
    a = {left_norm[i:i + 2] for i in range(max(0, len(left_norm) - 1))}
    b = {right_norm[i:i + 2] for i in range(max(0, len(right_norm) - 1))}
    return len(a & b) / len(a | b) if a | b else 0.0


def validate_item(item: dict[str, Any], passage: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    required = ("source_question", "paraphrased_question", "reference_answer", "evidence_quote")
    for field in required:
        if not str(item.get(field, "")).strip():
            warnings.append(f"missing_{field}")
    if len(normalized(item.get("paraphrased_question", ""))) < 8:
        warnings.append("question_too_short")
    if len(normalized(item.get("reference_answer", ""))) < 10:
        warnings.append("answer_too_short")
    if normalized(item.get("source_question", "")) == normalized(item.get("paraphrased_question", "")):
        warnings.append("paraphrase_unchanged")
    evidence = normalized(item.get("evidence_quote", ""))
    corpus = normalized("".join(chunk["content"] for chunk in passage["chunks"]))
    if len(evidence) < 8 or evidence not in corpus:
        warnings.append("evidence_not_verbatim")
    valid_ids = set(passage["chunk_ids"])
    support = {str(value) for value in item.get("supporting_chunk_ids", [])}
    if not support or not support.issubset(valid_ids):
        warnings.append("invalid_supporting_chunk_ids")
    if item.get("question_type") not in QUESTION_TYPES:
        warnings.append("invalid_question_type")
    if item.get("difficulty") not in DIFFICULTIES:
        warnings.append("invalid_difficulty")
    if char_overlap(item.get("paraphrased_question", ""), item.get("evidence_quote", "")) > 0.85:
        warnings.append("question_may_copy_evidence")
    return warnings


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", type=Path, default=ROOT / ".env")
    pre_args, _ = pre.parse_known_args()
    load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(description="Generate a review queue from local RAGFlow chunks")
    parser.add_argument("--env-file", type=Path, default=pre_args.env_file)
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/local/ragflow/chunks.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/generated/eval_v0")
    parser.add_argument("--base-url", default=os.getenv("EVAL_LLM_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("EVAL_LLM_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("EVAL_LLM_MODEL", ""))
    parser.add_argument("--passages", type=int, default=150)
    parser.add_argument("--questions-per-passage", type=int, default=2)
    parser.add_argument("--target-passage-chars", type=int, default=450)
    parser.add_argument("--max-passage-chars", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only select and save evidence passages")
    parser.add_argument("--resume", action="store_true", help="Skip passages already present in raw_responses.jsonl")
    args = parser.parse_args()

    chunks = read_jsonl(args.chunks.resolve())
    passages = build_passages(chunks, args.target_passage_chars, args.max_passage_chars)
    selected = stratified_sample(passages, args.passages, args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "selected_passages.jsonl", selected)
    if args.dry_run:
        summary = {"chunks": len(chunks), "passages": len(passages), "selected": len(selected), "seed": args.seed, "output": str(output)}
        (output / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.base_url or not args.api_key or not args.model:
        parser.error("EVAL_LLM_BASE_URL, EVAL_LLM_API_KEY and EVAL_LLM_MODEL are required unless --dry-run is used")

    raw_path = output / "raw_responses.jsonl"
    completed: set[str] = set()
    if args.resume and raw_path.exists():
        completed = {row["passage_id"] for row in read_jsonl(raw_path) if row.get("status") == "success"}
    elif raw_path.exists():
        raw_path.unlink()

    generated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    api_url = endpoint(args.base_url)
    for index, passage in enumerate(selected, 1):
        if passage["passage_id"] in completed:
            continue
        try:
            payload, meta = call_model(
                api_url, args.api_key, args.model,
                passage_prompt(passage, args.questions_per_passage),
                args.timeout, not args.no_json_mode, args.max_retries,
            )
            items = payload.get("items", []) if isinstance(payload, dict) else []
            append_jsonl(raw_path, {"passage_id": passage["passage_id"], "status": "success", "response": payload, "meta": meta})
            for item_index, item in enumerate(items, 1):
                warnings = validate_item(item, passage)
                question_key = normalized(item.get("paraphrased_question", ""))
                if question_key in seen_questions:
                    warnings.append("duplicate_question")
                seen_questions.add(question_key)
                support = [str(value) for value in item.get("supporting_chunk_ids", [])]
                case = {
                    "case_id": f"auto-{passage['passage_id']}-{item_index}",
                    "question": item.get("paraphrased_question", ""),
                    "source_question": item.get("source_question", ""),
                    "reference_answer": item.get("reference_answer", ""),
                    "evidence_quote": item.get("evidence_quote", ""),
                    "gold_chunk_ids": support,
                    "document_name": passage["document_name"],
                    "chapter": passage["chapter"],
                    "question_type": item.get("question_type", ""),
                    "difficulty": item.get("difficulty", ""),
                    "generation_passage_id": passage["passage_id"],
                    "validation_warnings": warnings,
                    "review_status": None,
                    "review_note": "",
                    "split": "unassigned",
                }
                (rejected if warnings else generated).append(case)
            print(f"[{index}/{len(selected)}] {passage['passage_id']}: {len(items)} items")
        except Exception as exc:
            error = {"passage_id": passage["passage_id"], "status": "error", "error": str(exc)}
            errors.append(error)
            append_jsonl(raw_path, error)
            print(f"[{index}/{len(selected)}] {passage['passage_id']}: ERROR {exc}")
        if args.delay:
            time.sleep(args.delay)

    # When resuming, rebuild all cases deterministically from successful raw responses.
    if args.resume and raw_path.exists():
        generated, rejected, seen_questions = [], [], set()
        passage_map = {row["passage_id"]: row for row in selected}
        latest = {row["passage_id"]: row for row in read_jsonl(raw_path) if row.get("status") == "success"}
        for passage_id, raw in latest.items():
            passage = passage_map.get(passage_id)
            if not passage:
                continue
            for item_index, item in enumerate(raw.get("response", {}).get("items", []), 1):
                warnings = validate_item(item, passage)
                key = normalized(item.get("paraphrased_question", ""))
                if key in seen_questions:
                    warnings.append("duplicate_question")
                seen_questions.add(key)
                case = {
                    "case_id": f"auto-{passage_id}-{item_index}", "question": item.get("paraphrased_question", ""),
                    "source_question": item.get("source_question", ""), "reference_answer": item.get("reference_answer", ""),
                    "evidence_quote": item.get("evidence_quote", ""),
                    "gold_chunk_ids": [str(value) for value in item.get("supporting_chunk_ids", [])],
                    "document_name": passage["document_name"], "chapter": passage["chapter"],
                    "question_type": item.get("question_type", ""), "difficulty": item.get("difficulty", ""),
                    "generation_passage_id": passage_id, "validation_warnings": warnings,
                    "review_status": None, "review_note": "", "split": "unassigned",
                }
                (rejected if warnings else generated).append(case)

    write_jsonl(output / "review_queue.jsonl", generated)
    write_jsonl(output / "rejected_cases.jsonl", rejected)
    write_jsonl(output / "generation_errors.jsonl", errors)
    successful_raw = [row for row in read_jsonl(raw_path) if row.get("status") == "success"] if raw_path.exists() else []
    usage = collections.Counter()
    for row in successful_raw:
        for key, value in row.get("meta", {}).get("usage", {}).items():
            if isinstance(value, (int, float)):
                usage[key] += value
    summary = {
        "source_chunks": len(chunks), "candidate_passages": len(passages), "selected_passages": len(selected),
        "valid_review_cases": len(generated), "rejected_cases": len(rejected), "request_errors": len(errors),
        "seed": args.seed, "model": args.model, "json_mode": not args.no_json_mode,
        "api_usage": dict(usage),
        "note": "Only review_status=true cases should enter the Gold set.",
    }
    (output / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
