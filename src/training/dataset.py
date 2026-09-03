"""Claude ReAct transcripts -> OpenAI tool-calling chat records.

Only the ``transcript`` field of ``data/eval/*_runs.jsonl`` is faithful: the flat
``messages`` / ``prompt`` / ``completion`` collapse the whole trajectory into one
assistant turn and ``tool_calls`` keeps only the first of each parallel batch.
This module reads ``transcript`` and nothing else.

Findings baked in here:
  * the transcript has no system message — it is prepended from
    ``prompts/restaurant_agent.md``;
  * the user turn is only the raw query — preferences arrive through the
    ``read_food_preferences`` tool result;
  * every Claude ``thinking`` block in this dataset is empty (redacted signature
    only) — they are stripped, and the model is trained non-thinking.

No torch. No tokenizer. Pure dict-shuffling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"
SYSTEM_PROMPT_NAME = "restaurant_agent"


class DropRow(Exception):
    """A transcript that cannot be converted to a clean training record."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class ConversionStats:
    kept: int = 0
    dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped += 1
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1

    def report(self) -> str:
        lines = [f"kept {self.kept}, dropped {self.dropped}"]
        for reason, n in sorted(self.drop_reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {n:>3}  {reason}")
        return "\n".join(lines)


def load_system_prompt() -> str:
    from src.utils.prompts import load_prompt

    return load_prompt(PROMPTS_DIR, SYSTEM_PROMPT_NAME)


def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _convert_ai_message(msg: dict[str, Any]) -> dict[str, Any]:
    """One transcript ``ai`` turn -> one OpenAI assistant message."""
    content = msg.get("content")
    if isinstance(content, str):
        text, tool_calls = content, []
    else:
        text_parts: list[str] = []
        tool_calls = []
        for block in content or []:
            btype = block.get("type")
            if btype == "thinking":
                continue  # all empty in this dataset; train non-thinking
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(
                                block.get("input", {}), ensure_ascii=False
                            ),
                        },
                    }
                )
            # unknown block keys (`caller`, `toolset_name`) are ignored
        text = "".join(text_parts)

    out: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def transcript_to_messages(
    transcript: list[dict[str, Any]], system_prompt: str
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for msg in transcript:
        role = msg.get("role")
        content = msg.get("content")
        if role == "human":
            messages.append({"role": "user", "content": _content_to_str(content)})
        elif role == "ai":
            messages.append(_convert_ai_message(msg))
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id"),
                    "name": msg.get("name"),
                    "content": content if isinstance(content, str) else json.dumps(
                        content, ensure_ascii=False
                    ),
                }
            )
        else:
            raise DropRow(f"unknown transcript role: {role!r}")
    return messages


def _validate(messages: list[dict[str, Any]]) -> None:
    open_calls: dict[str, str] = {}  # id -> tool name
    answered: set[str] = set()
    for m in messages:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls", []):
                open_calls[tc["id"]] = tc["function"]["name"]
        elif m["role"] == "tool":
            cid = m.get("tool_call_id")
            if cid not in open_calls:
                raise DropRow("tool message with unresolved tool_call_id")
            answered.add(cid)
    unanswered = set(open_calls) - answered
    if unanswered:
        raise DropRow("assistant tool call left unanswered")

    last = messages[-1]
    if last["role"] != "assistant":
        raise DropRow("last message is not an assistant turn")
    if last.get("tool_calls"):
        raise DropRow("final assistant turn still has tool calls")
    if not (last.get("content") or "").strip():
        raise DropRow("final assistant turn has no text")


def row_to_record(row: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    if row.get("error"):
        raise DropRow("row carries an error field")
    if not _content_to_str(row.get("completion") or "").strip():
        raise DropRow("row has empty completion")
    transcript = row.get("transcript")
    if not transcript:
        raise DropRow("row has no transcript")

    messages = transcript_to_messages(transcript, system_prompt)
    _validate(messages)
    return {"id": row.get("id"), "messages": messages}


def build_records(
    rows: Iterable[dict[str, Any]],
    system_prompt: str | None = None,
    *,
    dedup_by: str | None = "id",
    stats: ConversionStats | None = None,
) -> list[dict[str, Any]]:
    system_prompt = system_prompt or load_system_prompt()
    stats = stats if stats is not None else ConversionStats()
    records: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for row in rows:
        try:
            rec = row_to_record(row, system_prompt)
        except DropRow as exc:
            stats.drop(exc.reason)
            continue
        if dedup_by and rec.get(dedup_by) in seen:
            stats.drop(f"duplicate {dedup_by}")
            continue
        if dedup_by:
            seen.add(rec.get(dedup_by))
        records.append(rec)
        stats.kept += 1
    return records


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def split_records(
    records: list[dict[str, Any]], *, val_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import random

    ordered = sorted(records, key=lambda r: str(r.get("id")))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_val = round(len(ordered) * val_frac)
    return ordered[n_val:], ordered[:n_val]
