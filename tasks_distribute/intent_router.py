"""Fast, deterministic routing for optional business sections.

The main LLM answer is always enabled by the caller.  These rules only decide
whether a user explicitly asked for an auxiliary resource panel.  Keeping the
rules conservative avoids both an extra LLM round trip and noisy panels on
ordinary textbook questions.
"""

from __future__ import annotations

from typing import Dict


QUESTION_BANK_CUES = (
    "题库", "找题", "搜题", "推荐题", "类似题", "相关题", "练习题",
    "习题", "例题", "试题", "考题", "出题", "刷题", "作业题",
)

TEACHER_QA_CUES = (
    "教师答疑", "老师答疑", "课堂答疑", "答疑记录", "历史答疑",
    "老师以前怎么回答", "老师讲过", "同学问过",
)

INTERACTIVE_CUES = (
    "交互式案例", "交互案例", "H5案例", "H5演示", "软件仿真",
    "仿真案例", "动画演示", "可视化演示",
)

HARDWARE_STRONG_CUES = (
    "在线硬件实验", "硬件实验", "在线实验", "实验入口", "实验链接",
    "实验平台", "开始实验", "启动实验", "操作实验", "进入实验",
    "打开实验", "实训",
)

EXPERIMENT_ACTION_CUES = (
    "做实验", "找实验", "搜实验", "实验演示", "实验操作", "实验实操",
    "帮我做", "帮我找", "给我找", "我要做", "我想做", "有没有实验",
)

OTHER_COURSE_CUES = (
    "其他课程实验", "非通信原理实验", "数字电路实验", "模拟电路实验",
    "信号与系统实验", "微机原理实验", "计算机网络实验",
)

IDEOLOGY_STRONG_CUES = (
    "思政", "爱国", "先烈", "榜样", "科学家精神", "技术突破故事",
    "人物故事", "科学家故事", "发展史", "通信史", "历史故事",
    "北邮名师",
)

IDEOLOGY_NAMES = ("张平", "马可尼", "香农")
STORY_CUES = ("故事", "生平", "历史", "精神", "贡献", "事迹")


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue.lower() in text for cue in cues)


def classify_business_intents(question: str) -> Dict[str, object]:
    """Route explicit requests for auxiliary panels without calling an LLM."""

    original = (question or "").strip()
    q = original.lower()
    result: Dict[str, object] = {
        "问题内容": original,
        "通信原理题目库": 0,
        "通信原理课堂教师答疑记录": 0,
        "大模型概念回答": 1,
        "通信原理课堂交互式案例": 0,
        "通信原理课堂在线硬件实验": 0,
        "其他课程实验案例": 0,
        "思政教育案例": 0,
    }

    result["通信原理题目库"] = int(_contains_any(q, QUESTION_BANK_CUES))
    result["通信原理课堂教师答疑记录"] = int(_contains_any(q, TEACHER_QA_CUES))
    result["通信原理课堂交互式案例"] = int(_contains_any(q, INTERACTIVE_CUES))

    is_other_course = _contains_any(q, OTHER_COURSE_CUES)
    result["其他课程实验案例"] = int(is_other_course)

    explicit_hardware = _contains_any(q, HARDWARE_STRONG_CUES)
    experiment_action = _contains_any(q, EXPERIMENT_ACTION_CUES)
    # A bare short experiment name (for example “星座图实验”) is treated as
    # navigation.  Long explanatory questions containing “实验” are not.
    bare_experiment_name = (
        "实验" in q
        and len(original) <= 20
        and not any(cue in q for cue in ("为什么", "如何", "怎么", "原理", "结果", "测量"))
    )
    result["通信原理课堂在线硬件实验"] = int(
        not is_other_course and (explicit_hardware or experiment_action or bare_experiment_name)
    )

    ideology = _contains_any(q, IDEOLOGY_STRONG_CUES)
    if not ideology and _contains_any(q, IDEOLOGY_NAMES):
        ideology = _contains_any(q, STORY_CUES)
    result["思政教育案例"] = int(ideology)

    return result

