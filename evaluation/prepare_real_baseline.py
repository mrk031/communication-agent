#!/usr/bin/env python3
"""Prepare clean teacher-QA cases and the exported RAGFlow chunk corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_nodes(value: Any) -> List[str]:
    """Extract only explicit text nodes; image bytes are never converted to strings."""
    pieces: List[str] = []
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("content"), str):
            content = value["content"].strip()
            if content:
                pieces.append(content)
        else:
            for child in value.values():
                pieces.extend(text_nodes(child))
    elif isinstance(value, list):
        for child in value:
            pieces.extend(text_nodes(child))
    return pieces


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-qa", type=Path, default=ROOT / "data" / "local" / "teacher_qa.jsonl")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "local" / "ragflow" / "chunks.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "prepared")
    args = parser.parse_args()

    cases = []
    skipped = []
    for row in read_jsonl(args.teacher_qa):
        question = str(row.get("question_pure_text", "")).strip()
        if not question:
            question = "\n".join(text_nodes(row.get("question_whole"))).strip()
        answer = "\n".join(text_nodes(row.get("answer"))).strip()
        reason = None
        if not question:
            reason = "missing_text_question"
        elif not answer:
            reason = "missing_text_answer"
        if reason:
            skipped.append({"record_id": row.get("record_id"), "reason": reason})
            continue
        cases.append({
            "case_id": str(row["record_id"]),
            "question": question,
            "reference_answer": answer,
            "gold_doc_ids": [],
            "holdout_doc_ids": [],
            "intent_labels": ["大模型概念回答"],
            "split": "unlabeled",
            "source": "teacher_qa",
        })

    corpus = []
    for row in read_jsonl(args.chunks):
        content = str(row.get("content", "")).strip()
        if not content:
            continue
        corpus.append({
            "doc_id": str(row["chunk_id"]),
            "content": content,
            "source_type": "ragflow_chunk",
            "parent_document_id": str(row.get("doc_id", "")),
            "document_name": str(row.get("document_name", "")),
        })

    output = args.output.resolve()
    write_jsonl(output / "teacher_qa_cases_unlabeled.jsonl", cases)
    write_jsonl(output / "ragflow_chunks_corpus.jsonl", corpus)
    write_jsonl(output / "skipped_teacher_qa.jsonl", skipped)
    summary = {
        "teacher_records_used": len(cases),
        "teacher_records_skipped": len(skipped),
        "ragflow_chunks": len(corpus),
        "image_binary_policy": "ignored; only nodes with type=text are retained",
        "gold_status": "unlabeled",
    }
    (output / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**summary, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

