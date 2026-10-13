#!/usr/bin/env python3
"""Read-only inspection of RAGFlow dataset/document chunk parser settings."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

from export_ragflow import list_all_chunks, list_all_documents, load_env_file


SCRIPT_DIR = Path(__file__).resolve().parent


def plain(value: Any) -> Any:
    if hasattr(value, "to_json"):
        return value.to_json()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def exact_boundary_overlap(left: str, right: str, cap: int = 500) -> int:
    """Return exact suffix/prefix overlap; this does not detect semantic overlap."""
    maximum = min(len(left), len(right), cap)
    for size in range(maximum, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def chunk_stats(chunks: list[Any]) -> dict[str, Any]:
    lengths: list[int] = []
    contents: list[str] = []
    for chunk in chunks:
        raw = plain(chunk)
        content = raw.get("content") or raw.get("content_with_weight") or ""
        text = str(content)
        contents.append(text)
        lengths.append(len(text))
    overlaps = [
        exact_boundary_overlap(left, right)
        for left, right in zip(contents, contents[1:])
    ]
    return {
        "count": len(lengths),
        "characters": {
            "min": min(lengths) if lengths else None,
            "p25": percentile(lengths, 0.25),
            "median": statistics.median(lengths) if lengths else None,
            "p75": percentile(lengths, 0.75),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths) if lengths else None,
            "mean": statistics.fmean(lengths) if lengths else None,
        },
        "adjacent_exact_boundary_overlap": {
            "pairs": len(overlaps),
            "pairs_ge_1_char": sum(value >= 1 for value in overlaps),
            "pairs_ge_10_chars": sum(value >= 10 for value in overlaps),
            "pairs_ge_20_chars": sum(value >= 20 for value in overlaps),
            "max_chars": max(overlaps) if overlaps else None,
            "note": (
                "Only exact suffix-to-prefix repetition is measured. Repeated headings may "
                "look like overlap, while semantic or parent-context overlap is not detected."
            ),
        },
    }


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", type=Path, default=SCRIPT_DIR / ".env")
    pre_args, _ = pre.parse_known_args()
    load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(
        description="Read-only inspection of RAGFlow parser_config and observed chunk lengths"
    )
    parser.add_argument("--env-file", type=Path, default=pre_args.env_file)
    parser.add_argument("--base-url", default=os.getenv("RAGFLOW_BASE_URL", "http://127.0.0.1:9380"))
    parser.add_argument("--api-key", default=os.getenv("RAGFLOW_API_KEY"))
    parser.add_argument("--dataset-name", default=os.getenv("RAGFLOW_DATASET_NAME", "通原"))
    parser.add_argument("--dataset-id")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--skip-chunk-stats", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "data" / "local" / "ragflow" / "parser_config.json",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("provide RAGFLOW_API_KEY or --api-key")

    try:
        from ragflow_sdk import RAGFlow
    except ImportError as exc:
        raise SystemExit("ragflow-sdk is required: pip install ragflow-sdk") from exc

    client = RAGFlow(api_key=args.api_key, base_url=args.base_url)
    datasets = (
        client.list_datasets(id=args.dataset_id)
        if args.dataset_id
        else client.list_datasets(name=args.dataset_name)
    )
    if len(datasets) != 1:
        raise SystemExit(f"expected exactly one dataset, found {len(datasets)}")

    dataset = datasets[0]
    documents = list_all_documents(dataset, args.page_size)
    report: dict[str, Any] = {
        "read_only": True,
        "base_url": args.base_url,
        "dataset": {
            "id": str(dataset.id),
            "name": str(dataset.name),
            "chunk_method": getattr(dataset, "chunk_method", None),
            "parser_config": plain(getattr(dataset, "parser_config", None)),
            "embedding_model": getattr(dataset, "embedding_model", None),
            "document_count": getattr(dataset, "document_count", None),
            "chunk_count": getattr(dataset, "chunk_count", None),
        },
        "documents": [],
        "interpretation_note": (
            "parser_config is authoritative for exposed settings. Observed length statistics "
            "describe results only and cannot prove overlap by themselves."
        ),
    }
    for document in documents:
        item: dict[str, Any] = {
            "id": str(document.id),
            "name": str(document.name),
            "chunk_method": getattr(document, "chunk_method", None),
            "parser_config": plain(getattr(document, "parser_config", None)),
            "reported_chunk_count": getattr(document, "chunk_count", None),
            "token_count": getattr(document, "token_count", None),
        }
        if not args.skip_chunk_stats:
            item["observed_chunks"] = chunk_stats(
                list_all_chunks(document, args.page_size)
            )
        report["documents"].append(item)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
