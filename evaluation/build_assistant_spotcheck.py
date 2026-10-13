#!/usr/bin/env python3
"""Build the evidence-based e02 vs no-RAG spot-check without altering source CSV."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "output/e2e/answer_dev/analysis/paired_answer_review.csv"
OUTPUT = ROOT / "output/e2e/answer_dev/analysis/assistant_spotcheck_e02_vs_no_rag.csv"
SUMMARY = ROOT / "output/e2e/answer_dev/analysis/assistant_spotcheck_e02_vs_no_rag_summary.json"

# winner, baseline correctness, challenger correctness, challenger faithfulness,
# sampling stratum, evidence-based note
JUDGMENTS = {
    "auto-1a13a4392b96f3cf-2": ("challenger", 1, 2, 2, "large_gain", "RAG答案严格复现教材结论；no-RAG混入教材之外的后续运营信息，与本题参考口径冲突。"),
    "auto-025ba16e44ba47c2-1": ("challenger", 1, 2, 2, "large_gain", "no-RAG虽答对硬件简单，但给出另一套搜索模型和时间公式；RAG准确给出教材的(N-1)T。"),
    "auto-f62e785193420eb9-1": ("challenger", 0, 2, 2, "large_gain", "no-RAG误答成MAP/ML判决准则；RAG正确列出教材中的三种误码度量准则及适用范围。"),
    "auto-79f4115013041035-1": ("challenger", 1, 2, 2, "large_gain", "no-RAG主结论为3至5 dB，与参考的正负10 dB不符；RAG有直接证据支持。"),
    "auto-25a0abe20be29dcb-2": ("challenger", 0, 2, 2, "large_gain", "no-RAG按现代IP路由器解释并断言路由器不重发，和教材所述逐跳确认重发机制相反。"),
    "auto-3bdad26c71f11fbe-2": ("challenger", 1, 2, 2, "new_e02_recall", "两者都提到压缩冗余；no-RAG额外把A/D数字化列为主要任务，RAG对统计冗余的解释更完整、更贴合证据。"),
    "auto-ce49ef53fe9a1137-1": ("challenger", 1, 2, 2, "new_e02_recall", "no-RAG未回答改造信源/适应信源两条思路，且把哈夫曼一类误称为非前缀码；RAG正确。"),
    "auto-4ab9a343297fa2a8-2": ("challenger", 0, 2, 2, "large_loss_by_f1", "尽管challenger字符F1低于baseline，no-RAG的3.5和16QAM均错误；RAG的4.64和64QAM与教材一致。"),
    "auto-998b394dc92895bf-1": ("tie", 2, 2, 2, "new_e02_recall", "两者都准确说明端到端专用通路在通话期间持续独占；RAG更贴近教材但没有实质正确性优势。"),
    "auto-10d7e162c4142560-2": ("tie", 2, 2, 2, "new_e02_recall", "两者均正确回答正交MFSK信号矢量等距；F1差异主要来自表述长度。"),
    "auto-6f5f4411b60bf49e-1": ("tie", 2, 2, 2, "representative_gain", "两者都正确解释OFDM、子载波间隔和频谱重叠；RAG更精简，但不构成实质质量差异。"),
    "auto-18d8111183dffb1c-1": ("tie", 2, 2, 2, "near_zero_delta", "两份答案完全相同且正确。"),
    "auto-01417ff58965d891-1": ("tie", 2, 2, 2, "negative_delta", "两者均正确解释R和R(D)；RAG增加的最小信息率解释有上下文支持。"),
    "auto-21c243fcb4c9f8ce-1": ("tie", 2, 2, 2, "gold_miss", "两者核心答案均为数字基带传输系统；RAG虽未命中标注Gold ID，但检索上下文仍包含充分证据。"),
    "auto-07dbf2d7a67641fe-1": ("tie", 2, 2, 2, "negative_delta", "两者均正确回答QPSK主瓣宽度为一半；字符F1下降来自多余的分析段落。"),
    "auto-9fb90297dac68619-2": ("tie", 2, 2, 2, "negative_delta", "两者都正确给出卡松公式；RAG增加的98%说明可由检索上下文直接支持。"),
    "auto-f5f6089d508eae8c-1": ("tie", 2, 2, 2, "largest_loss", "两者均正确得到10 kHz；RAG因解释更长而字符F1更低，并非语义退化。"),
    "auto-a5eb75a4a526ad6f-2": ("challenger", 1, 2, 2, "large_loss_by_f1", "no-RAG漏掉两个信源各自的信息熵H(X)、H(Y)；RAG完整覆盖各自熵、联合熵和条件熵。"),
    "auto-125e39d8f1f83580-2": ("tie", 2, 2, 2, "gold_miss", "两者都正确给出正频率范围和B=fH-fL；Gold ID未命中没有造成答案错误。"),
    "auto-bb19a9fbf27229ce-1": ("baseline", 2, 1, 1, "negative_delta", "两者推导基本正确；RAG末句称均值也随t变化，但本题均值恒为0。常数可视为周期函数，不过该措辞不严谨。"),
}


def main() -> int:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    indexed = {
        row["case_id"]: row for row in source_rows if row["comparison"] == "e02_vs_no_rag"
    }
    missing = sorted(set(JUDGMENTS) - set(indexed))
    if missing:
        raise RuntimeError(f"Missing reviewed cases: {missing}")

    output_rows = []
    for case_id, judgment in JUDGMENTS.items():
        winner, baseline_score, challenger_score, faithfulness, stratum, note = judgment
        row = dict(indexed[case_id])
        row.update({
            "human_winner": winner,
            "correctness_baseline_0_2": baseline_score,
            "correctness_challenger_0_2": challenger_score,
            "faithfulness_challenger_0_2": faithfulness,
            "human_note": note,
            "review_type": "assistant_manual_evidence_based",
            "sampling_stratum": stratum,
        })
        output_rows.append(row)

    fieldnames = list(output_rows[0])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    winners = Counter(row["human_winner"] for row in output_rows)
    summary = {
        "comparison": "e02_vs_no_rag",
        "review_type": "assistant_manual_evidence_based",
        "sample_size": len(output_rows),
        "sampling": "purposive stratified spot-check; not an unbiased estimate of all 60 cases",
        "winner_counts": dict(winners),
        "mean_correctness_baseline_0_2": sum(int(row["correctness_baseline_0_2"]) for row in output_rows) / len(output_rows),
        "mean_correctness_challenger_0_2": sum(int(row["correctness_challenger_0_2"]) for row in output_rows) / len(output_rows),
        "mean_faithfulness_challenger_0_2": sum(int(row["faithfulness_challenger_0_2"]) for row in output_rows) / len(output_rows),
        "observations": [
            "RAG clearly corrected several textbook-specific factual errors made by no-RAG.",
            "Several retrieval/F1 improvements were semantic ties because no-RAG was already correct.",
            "Character F1 penalized correct longer answers and cannot serve as the final quality judge.",
            "Legacy RAG prompt frequently caused explicit analysis/final-answer headings and unnecessary verbosity.",
            "One sampled RAG answer had a minor precision issue despite grounded derivation.",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
