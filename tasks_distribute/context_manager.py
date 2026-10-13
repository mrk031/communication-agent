"""Dependency-free short-term context budgeting for the Agent request path."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


ALLOWED_ROLES = {"user", "assistant"}


def estimate_tokens(text: str) -> int:
    """Conservative, transparent estimate for mixed Chinese/English content."""
    return max(1, (len(str(text or "")) + 1) // 2)


def truncate_message(text: str, max_chars: int) -> str:
    """Retain both the beginning and conclusion of an oversized message."""
    value = str(text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    marker = "\n…[历史消息已截断]…\n"
    available = max(max_chars - len(marker), 2)
    head = available * 2 // 3
    tail = available - head
    return value[:head] + marker + value[-tail:]


def normalize_history(
    raw_messages: Iterable[Dict[str, Any]], max_message_chars: int = 2000
) -> List[Dict[str, str]]:
    """Normalize roles/content and discard empty or unsupported history rows."""
    normalized: List[Dict[str, str]] = []
    for item in raw_messages:
        raw_role = str(item.get("role") or item.get("sender_role") or "").lower()
        role = raw_role if raw_role in ALLOWED_ROLES else "assistant"
        content = truncate_message(str(item.get("content") or ""), max_message_chars)
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def _group_turns(messages: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    """Group a user message with following assistant messages as one complete turn."""
    turns: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    for message in messages:
        if message["role"] == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def select_recent_context(
    raw_messages: Iterable[Dict[str, Any]],
    token_budget: int = 1600,
    max_messages: int = 10,
    max_message_chars: int = 2000,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Select recent complete turns under both message and estimated-token budgets.

    The newest turn is always retained. If it alone exceeds the remaining budget,
    its messages are shortened proportionally instead of silently dropping the
    user's latest context.
    """
    normalized = normalize_history(raw_messages, max_message_chars=max_message_chars)
    turns = _group_turns(normalized)
    selected_reversed: List[List[Dict[str, str]]] = []
    used_tokens = 0
    used_messages = 0

    for turn in reversed(turns):
        turn_tokens = sum(estimate_tokens(message["content"]) for message in turn)
        would_exceed = (
            used_messages + len(turn) > max_messages
            or used_tokens + turn_tokens > token_budget
        )
        if would_exceed and selected_reversed:
            break
        selected_reversed.append(turn)
        used_tokens += turn_tokens
        used_messages += len(turn)
        if used_messages >= max_messages or used_tokens >= token_budget:
            break

    selected = [
        message
        for turn in reversed(selected_reversed)
        for message in turn
    ]
    if len(selected) > max_messages:
        selected = selected[-max_messages:]

    # Enforce the final budget for an exceptionally large newest turn while
    # retaining both sides of that turn for reference resolution.
    selected_tokens = sum(estimate_tokens(item["content"]) for item in selected)
    if selected and selected_tokens > token_budget:
        per_message_tokens = max(token_budget // len(selected), 1)
        final = [
            {
                "role": message["role"],
                "content": truncate_message(message["content"], per_message_tokens * 2),
            }
            for message in selected
        ]
    else:
        final = selected

    stats = {
        "input_messages": len(normalized),
        "kept_messages": len(final),
        "dropped_messages": max(len(normalized) - len(final), 0),
        "estimated_tokens": sum(estimate_tokens(item["content"]) for item in final),
        "token_budget": token_budget,
    }
    return final, stats

