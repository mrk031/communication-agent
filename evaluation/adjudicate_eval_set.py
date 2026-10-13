#!/usr/bin/env python3
"""Create high-confidence Gold, broader Silver, and a small manual-review queue.

This is a conservative, fully offline adjudicator. Original generated files are
never edited. Every decision contains machine-readable reasons and scores.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEICTIC_RE = re.compile(
    r"(这道题|那道题|题目里|题里|题中|本题|上[文面]|下[文面]|如图|如下|上述|"
    r"课本里|教材中|文中提到|这本书|新版教材|第\s*\d+\s*章里.{0,8}题|"
    r"例\s*\d+(?:\.\d+)*|这个卷积码|那个理论|它给出的|要我们求|先要求)"
)
BROKEN_RE = re.compile(r"[�]|\.{4,}|·{4,}|…{3,}")
FORMULA_RE = re.compile(r"[=<>±×÷∑∫√]|[A-Za-z][_(（(]|\^|[α-ωΑ-Ω]")
META_RE = re.compile(
    r"(这本.{0,12}(书|通信原理)|新版教材|这一版改动|作者.{0,12}(改动|修订)|"
    r"适合哪些.{0,8}(学生|专业)|从第.{0,20}章|应该看哪些章|主要讲了哪些知识板块)"
)
TYPE_QUOTAS = {"definition": 24, "mechanism": 20, "comparison": 14, "application": 12, "calculation": 10}
CHAPTER_NUMBERS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
                   "十": 10, "十一": 11, "十二": 12, "十三": 13}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9\u3400-\u9fff]", "", str(text or "")).lower()


def ngrams(text: str, n: int = 2) -> set[str]:
    value = normalize(text)
    return {value[i:i + n] for i in range(max(0, len(value) - n + 1))}


def coverage(source: str, target: str) -> float:
    source_set, target_set = ngrams(source), ngrams(target)
    return len(source_set & target_set) / len(target_set) if target_set else 0.0


def normalize_chapter(value: str) -> str:
    match = re.fullmatch(r"第\s*([一二三四五六七八九十]+)\s*章", str(value or ""))
    return f"第{CHAPTER_NUMBERS[match.group(1)]}章" if match and match.group(1) in CHAPTER_NUMBERS else str(value)


def load_passages(path: Path) -> dict[str, dict[str, Any]]:
    return {row["passage_id"]: row for row in read_jsonl(path)}


def assess(row: dict[str, Any], passage: dict[str, Any] | None) -> dict[str, Any]:
    question = str(row.get("question", ""))
    answer = str(row.get("reference_answer", ""))
    evidence = str(row.get("evidence_quote", ""))
    gold_ids = {str(value) for value in row.get("gold_chunk_ids", [])}
    warnings = list(row.get("validation_warnings", []))
    reasons: list[str] = []
    hard_flags: list[str] = []

    chunks = passage.get("chunks", []) if passage else []
    valid_ids = {str(chunk.get("chunk_id")) for chunk in chunks}
    passage_text = "\n".join(str(chunk.get("content", "")) for chunk in chunks)
    supported_text = "\n".join(
        str(chunk.get("content", "")) for chunk in chunks if str(chunk.get("chunk_id")) in gold_ids
    )
    evidence_exact = bool(normalize(evidence)) and normalize(evidence) in normalize(passage_text)
    evidence_in_gold = bool(normalize(evidence)) and normalize(evidence) in normalize(supported_text)
    valid_gold_ids = bool(gold_ids) and gold_ids.issubset(valid_ids)
    answer_evidence_coverage = coverage(evidence, answer)
    answer_passage_coverage = coverage(passage_text, answer)
    question_evidence_overlap = coverage(evidence, question)
    deictic = bool(DEICTIC_RE.search(question))
    broken = bool(BROKEN_RE.search(question + answer))
    formula_heavy = len(FORMULA_RE.findall(question + answer)) >= 5
    book_metadata = bool(META_RE.search(question))

    score = 100.0
    if not evidence_exact:
        score -= 25
        reasons.append("evidence_not_exact_in_passage")
    if not evidence_in_gold:
        score -= 25
        hard_flags.append("evidence_not_in_gold_chunks")
    if not valid_gold_ids:
        score -= 40
        hard_flags.append("invalid_gold_chunk_ids")
    if answer_evidence_coverage < 0.55:
        score -= 18
        reasons.append("answer_weakly_supported_by_quote")
    elif answer_evidence_coverage < 0.72:
        score -= 8
        reasons.append("answer_partially_supported_by_quote")
    if answer_passage_coverage < 0.60:
        score -= 15
        hard_flags.append("answer_weakly_supported_by_passage")
    if deictic:
        score -= 15
        reasons.append("question_not_standalone")
    if book_metadata:
        score -= 20
        reasons.append("book_metadata_not_domain_knowledge")
    if broken:
        score -= 10
        reasons.append("possible_ocr_or_truncation")
    if formula_heavy:
        score -= 5
        reasons.append("formula_requires_visual_check")
    if not 12 <= len(normalize(question)) <= 100:
        score -= 8
        reasons.append("question_length_suspicious")
    if len(normalize(answer)) < 12:
        score -= 10
        reasons.append("answer_too_short")
    if question_evidence_overlap > 0.90:
        score -= 8
        reasons.append("question_too_close_to_evidence")
    if warnings:
        score -= min(20, 5 * len(warnings))
        reasons.extend(f"generator_warning:{warning}" for warning in warnings)

    if hard_flags or score < 60:
        tier = "needs_review"
    elif score >= 82 and not warnings and not deictic and not broken and not book_metadata:
        tier = "gold_candidate"
    else:
        tier = "silver"
    return {
        **row,
        "chapter": normalize_chapter(row.get("chapter", "")),
        "adjudication": {
            "tier": tier,
            "score": round(max(0, score), 2),
            "reasons": list(dict.fromkeys(reasons)),
            "hard_flags": list(dict.fromkeys(hard_flags)),
            "signals": {
                "evidence_exact": evidence_exact,
                "evidence_in_gold_chunks": evidence_in_gold,
                "valid_gold_chunk_ids": valid_gold_ids,
                "answer_evidence_coverage": round(answer_evidence_coverage, 4),
                "answer_passage_coverage": round(answer_passage_coverage, 4),
                "question_evidence_overlap": round(question_evidence_overlap, 4),
            },
            "reviewer": "offline_rule_assisted_v1",
        },
    }


def select_gold(candidates: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    """Select by type quota while round-robining chapters to limit concentration."""
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    quotas = dict(TYPE_QUOTAS)
    if target != sum(quotas.values()):
        scale = target / sum(quotas.values())
        quotas = {key: max(1, round(value * scale)) for key, value in quotas.items()}

    def too_similar(row: dict[str, Any]) -> bool:
        question = ngrams(row.get("question", ""))
        evidence = ngrams(row.get("evidence_quote", ""))
        for prior in selected:
            prior_q = ngrams(prior.get("question", ""))
            prior_e = ngrams(prior.get("evidence_quote", ""))
            q_similarity = len(question & prior_q) / len(question | prior_q) if question | prior_q else 0
            e_similarity = len(evidence & prior_e) / len(evidence | prior_e) if evidence | prior_e else 0
            if q_similarity >= 0.72 or e_similarity >= 0.86:
                return True
        return False

    def take_for_type(question_type: str, quota: int) -> None:
        pool = [row for row in candidates if row.get("question_type") == question_type and row["case_id"] not in used]
        by_chapter: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in pool:
            by_chapter[str(row.get("chapter", "unknown"))].append(row)
        for rows in by_chapter.values():
            rows.sort(key=lambda item: (-item["adjudication"]["score"], item["case_id"]))
        chapters = sorted(by_chapter)
        while chapters and quota > 0:
            active: list[str] = []
            for chapter in chapters:
                if by_chapter[chapter] and quota > 0:
                    row = by_chapter[chapter].pop(0)
                    if not too_similar(row):
                        selected.append(row)
                        used.add(row["case_id"])
                        quota -= 1
                if by_chapter[chapter]:
                    active.append(chapter)
            chapters = active

    for question_type, quota in quotas.items():
        take_for_type(question_type, quota)
    if len(selected) < target:
        remaining = sorted(
            (row for row in candidates if row["case_id"] not in used),
            key=lambda item: (-item["adjudication"]["score"], item["case_id"]),
        )
        for row in remaining:
            if len(selected) >= target:
                break
            if not too_similar(row):
                selected.append(row)
    return selected[:target]


def assign_splits(gold: list[dict[str, Any]], test_ratio: float = 0.25) -> None:
    by_type: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in gold:
        by_type[str(row.get("question_type", "unknown"))].append(row)
    for rows in by_type.values():
        rows.sort(key=lambda row: row["case_id"])
        test_count = round(len(rows) * test_ratio)
        # Stable hashing prevents file/order changes from moving cases randomly.
        ranked = sorted(rows, key=lambda row: __import__("hashlib").sha1(row["case_id"].encode()).hexdigest())
        test_ids = {row["case_id"] for row in ranked[:test_count]}
        for row in rows:
            row["split"] = "test" if row["case_id"] in test_ids else "dev"


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["case_id", "question", "reference_answer", "evidence_quote", "gold_chunk_ids", "chapter",
              "question_type", "difficulty", "score", "hard_flags", "reasons", "human_decision", "human_note"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            adj = row["adjudication"]
            writer.writerow({
                "case_id": row["case_id"], "question": row["question"],
                "reference_answer": row["reference_answer"], "evidence_quote": row["evidence_quote"],
                "gold_chunk_ids": "|".join(row.get("gold_chunk_ids", [])), "chapter": row.get("chapter", ""),
                "question_type": row.get("question_type", ""), "difficulty": row.get("difficulty", ""),
                "score": adj["score"], "hard_flags": "|".join(adj["hard_flags"]),
                "reasons": "|".join(adj["reasons"]), "human_decision": "", "human_note": "",
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/generated/eval_v0")
    parser.add_argument("--output", type=Path, default=ROOT / "data/adjudicated/eval_v0")
    parser.add_argument("--gold-size", type=int, default=80)
    args = parser.parse_args()

    source = args.input.resolve()
    passages = load_passages(source / "selected_passages.jsonl")
    rows = read_jsonl(source / "review_queue.jsonl") + read_jsonl(source / "rejected_cases.jsonl")
    assessed = [assess(row, passages.get(row.get("generation_passage_id", ""))) for row in rows]
    candidates = [row for row in assessed if row["adjudication"]["tier"] == "gold_candidate"]
    gold = select_gold(candidates, min(args.gold_size, len(candidates)))
    gold_ids = {row["case_id"] for row in gold}
    for row in gold:
        row["review_status"] = True
        row["adjudication"]["tier"] = "gold"
    assign_splits(gold)
    silver = [row for row in assessed if row["case_id"] not in gold_ids and row["adjudication"]["tier"] != "needs_review"]
    needs_review = [row for row in assessed if row["adjudication"]["tier"] == "needs_review"]

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "gold_v0.jsonl", gold)
    write_jsonl(output / "silver_v0.jsonl", silver)
    write_jsonl(output / "needs_review.jsonl", needs_review)
    write_review_csv(output / "needs_review.csv", needs_review)
    summary = {
        "source_cases": len(rows), "gold": len(gold), "silver": len(silver), "needs_review": len(needs_review),
        "gold_splits": dict(collections.Counter(row.get("split") for row in gold)),
        "gold_reviewer": "offline_rule_assisted_v1",
        "gold_by_type": dict(collections.Counter(row.get("question_type") for row in gold)),
        "gold_by_chapter": dict(collections.Counter(row.get("chapter") for row in gold)),
        "review_hard_flags": dict(collections.Counter(flag for row in needs_review for flag in row["adjudication"]["hard_flags"])),
        "caveat": "Gold V0 is autonomously adjudicated. Do not tune on the frozen test subset; unresolved cases remain separate.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
