"""Shared fixtures for the hermetic training tests — no torch, no model download."""

from __future__ import annotations

import json

import pytest


class FakeTokenizer:
    """Deterministic, append-only ``apply_chat_template``.

    Renders each message as ``<role>…</role>`` (assistant turns start with the
    exact ``<assistant>`` header the generation prompt emits, so
    ``messages[:i]`` + generation-prompt is a strict prefix of ``messages[:i+1]``).
    ``tokenize=True`` returns one int per character.
    """

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"

    def _render_message(self, m: dict) -> str:
        role = m["role"]
        if role == "assistant":
            body = m.get("content") or ""
            for tc in m.get("tool_calls", []):
                body += f"<tool_call>{json.dumps(tc['function'], sort_keys=True)}</tool_call>"
            return f"<assistant>{body}</assistant>"
        if role == "tool":
            return f"<tool name={m.get('name')}>{m.get('content')}</tool>"
        return f"<{role}>{m.get('content')}</{role}>"

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=False, tokenize=False, **kw
    ):
        text = ""
        if tools:
            text += f"<tools>{json.dumps([t['function']['name'] for t in tools])}</tools>"
        for m in messages:
            text += self._render_message(m)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(c) for c in text] if tokenize else text

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()
