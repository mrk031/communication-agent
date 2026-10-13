#!/usr/bin/env python3
"""Read-only export of one RAGFlow dataset's originals, metadata and chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def safe_name(name: str, fallback: str) -> str:
    name = Path(str(name or "")).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return name or fallback


def unique_path(directory: Path, name: str, document_id: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    return directory / f"{stem}__{document_id[:12]}{suffix}"


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return result if isinstance(result, dict) else {}
    return dict(vars(value)) if hasattr(value, "__dict__") else {}


def list_all_documents(dataset: Any, page_size: int) -> List[Any]:
    documents: List[Any] = []
    page = 1
    while True:
        batch = dataset.list_documents(page=page, page_size=page_size, orderby="create_time", desc=False)
        if not batch:
            break
        documents.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return documents


def list_all_chunks(document: Any, page_size: int) -> List[Any]:
    chunks: List[Any] = []
    page = 1
    while True:
        batch = document.list_chunks(page=page, page_size=page_size)
        if not batch:
            break
        chunks.extend(batch)
        if len(batch) < page_size:
            break
        page += 1
    return chunks


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", type=Path, default=SCRIPT_DIR / ".env")
    pre_args, _ = pre_parser.parse_known_args()
    load_env_file(pre_args.env_file)
    parser = argparse.ArgumentParser(description="Read-only RAGFlow dataset exporter")
    parser.add_argument("--env-file", type=Path, default=pre_args.env_file)
    parser.add_argument("--base-url", default=os.getenv("RAGFLOW_BASE_URL", "http://127.0.0.1:9380"))
    parser.add_argument("--api-key", default=os.getenv("RAGFLOW_API_KEY"))
    parser.add_argument("--dataset-name", default=os.getenv("RAGFLOW_DATASET_NAME", "通原"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "data" / "local" / "ragflow")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--skip-originals", action="store_true")
    parser.add_argument("--skip-chunks", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("provide RAGFLOW_API_KEY or --api-key")
    try:
        from ragflow_sdk import RAGFlow
    except ImportError as exc:
        raise SystemExit("ragflow-sdk is required: pip install ragflow-sdk") from exc

    client = RAGFlow(api_key=args.api_key, base_url=args.base_url)
    datasets = client.list_datasets(id=args.dataset_id) if args.dataset_id else client.list_datasets(name=args.dataset_name)
    if not datasets:
        raise SystemExit("dataset not found")
    if len(datasets) != 1:
        choices = [{"id": item.id, "name": item.name} for item in datasets]
        print(json.dumps({"error": "multiple datasets matched", "datasets": choices}, ensure_ascii=False, indent=2))
        return 2
    dataset = datasets[0]
    documents = list_all_documents(dataset, args.page_size)
    print(json.dumps({"dataset_id": dataset.id, "dataset_name": dataset.name,
                      "document_count": len(documents)}, ensure_ascii=False, indent=2))
    if args.list_only:
        for document in documents:
            print(json.dumps({"document_id": document.id, "name": document.name,
                              "size": document.size, "chunk_count": document.chunk_count}, ensure_ascii=False))
        return 0

    output = args.output.resolve()
    originals = output / "originals"
    originals.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    chunks_out: List[Dict[str, Any]] = []

    for index, document in enumerate(documents, 1):
        doc_id = str(document.id)
        filename = safe_name(document.name, doc_id)
        local_path = unique_path(originals, filename, doc_id)
        sha256 = None
        download_status = "skipped"
        if not args.skip_originals:
            if local_path.exists() and not args.overwrite:
                blob = local_path.read_bytes()
                download_status = "existing"
            else:
                blob = document.download()
                local_path.write_bytes(blob)
                download_status = "downloaded"
            sha256 = hashlib.sha256(blob).hexdigest()

        chunk_count = 0
        if not args.skip_chunks:
            for chunk in list_all_chunks(document, args.page_size):
                raw = as_dict(chunk)
                content = raw.get("content") or raw.get("content_with_weight") or ""
                chunks_out.append({
                    "chunk_id": str(raw.get("id", "")),
                    "doc_id": doc_id,
                    "document_name": str(document.name),
                    "content": str(content),
                    "important_keywords": raw.get("important_keywords", []),
                    "questions": raw.get("questions", []),
                    "available": raw.get("available", True),
                })
                chunk_count += 1

        manifest.append({
            "dataset_id": str(dataset.id),
            "dataset_name": str(dataset.name),
            "document_id": doc_id,
            "name": str(document.name),
            "source_type": str(getattr(document, "source_type", "")),
            "size": int(getattr(document, "size", 0) or 0),
            "reported_chunk_count": int(getattr(document, "chunk_count", 0) or 0),
            "exported_chunk_count": chunk_count,
            "local_path": str(local_path.relative_to(output)) if not args.skip_originals else None,
            "sha256": sha256,
            "download_status": download_status,
        })
        print(f"[{index}/{len(documents)}] {document.name}: {download_status}, chunks={chunk_count}")

    write_jsonl(output / "manifest.jsonl", manifest)
    if not args.skip_chunks:
        write_jsonl(output / "chunks.jsonl", chunks_out)
    (output / "export_summary.json").write_text(json.dumps({
        "dataset_id": str(dataset.id), "dataset_name": str(dataset.name),
        "documents": len(manifest), "chunks": len(chunks_out),
        "base_url": args.base_url, "read_only": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
