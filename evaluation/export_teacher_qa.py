#!/usr/bin/env python3
"""Read-only Elasticsearch export for teacher Q&A records."""

from __future__ import annotations

import argparse
import json
import os
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def flatten_text(value: Any) -> str:
    value = maybe_json(value)
    pieces: List[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            pieces.append(item)
        elif isinstance(item, dict):
            if item.get("type") == "text" and item.get("content"):
                pieces.append(str(item["content"]))
            else:
                for child in item.values():
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return "\n".join(piece.strip() for piece in pieces if piece.strip())


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", type=Path, default=SCRIPT_DIR / ".env")
    pre_args, _ = pre_parser.parse_known_args()
    load_env_file(pre_args.env_file)
    parser = argparse.ArgumentParser(description="Read-only teacher Q&A Elasticsearch exporter")
    parser.add_argument("--env-file", type=Path, default=pre_args.env_file)
    parser.add_argument("--url", default=os.getenv("TEACHER_QA_ES_URL", "http://127.0.0.1:9201"))
    parser.add_argument("--index", default=os.getenv("TEACHER_QA_ES_INDEX", "ctc_discussion1.5"))
    parser.add_argument("--username", default=os.getenv("TEACHER_QA_ES_USERNAME"))
    parser.add_argument("--password", default=os.getenv("TEACHER_QA_ES_PASSWORD"))
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "data" / "local" / "teacher_qa.jsonl")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification only for an HTTPS endpoint")
    args = parser.parse_args()

    try:
        import requests
    except ImportError as exc:
        raise SystemExit("requests is required: pip install requests") from exc

    auth = (args.username, args.password) if args.username else None
    verify = not args.insecure
    base = args.url.rstrip("/")

    # Read-only preflight. This intentionally never uses PUT/POST/DELETE against the index.
    info = requests.get(f"{base}/{args.index}", auth=auth, verify=verify, timeout=15)
    info.raise_for_status()

    rows: List[Dict[str, Any]] = []
    scroll_id = None
    try:
        body: Dict[str, Any] = {
            "size": min(args.page_size, args.limit) if args.limit else args.page_size,
            "query": {"match_all": {}},
            "sort": ["_doc"],
            "_source": ["question_pure_text", "question_whole", "answer"],
        }
        response = requests.post(f"{base}/{args.index}/_search", params={"scroll": "2m"},
                                 json=body, auth=auth, verify=verify, timeout=60)
        response.raise_for_status()
        payload = response.json()
        scroll_id = payload.get("_scroll_id")
        while True:
            hits = payload.get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                source = hit.get("_source", {})
                rows.append({
                    "record_id": str(hit.get("_id", "")),
                    "index": args.index,
                    "question_pure_text": str(source.get("question_pure_text", "")),
                    "question_whole": maybe_json(source.get("question_whole")),
                    "answer": maybe_json(source.get("answer")),
                    "question_text": flatten_text(source.get("question_whole")) or str(source.get("question_pure_text", "")),
                    "answer_text": flatten_text(source.get("answer")),
                })
                if args.limit and len(rows) >= args.limit:
                    break
            if (args.limit and len(rows) >= args.limit) or not scroll_id:
                break
            response = requests.post(f"{base}/_search/scroll", json={"scroll": "2m", "scroll_id": scroll_id},
                                     auth=auth, verify=verify, timeout=60)
            response.raise_for_status()
            payload = response.json()
            scroll_id = payload.get("_scroll_id", scroll_id)
    finally:
        # Clears only the temporary search cursor; it never deletes index data.
        if scroll_id:
            try:
                requests.delete(f"{base}/_search/scroll", json={"scroll_id": [scroll_id]},
                                auth=auth, verify=verify, timeout=15)
            except requests.RequestException:
                pass

    output = args.output.resolve()
    write_jsonl(output, rows)
    summary = output.with_suffix(".summary.json")
    summary.write_text(json.dumps({"index": args.index, "records": len(rows), "url": args.url,
                                   "read_only": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(rows), "summary": str(summary)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
