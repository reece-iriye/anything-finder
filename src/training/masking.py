"""Assistant-only label masking by incremental chat-template rendering.

Qwen chat templates carry no ``{% generation %}`` markers, so TRL's
``assistant_only_loss`` cannot be used. Instead, for each assistant turn we render
``messages[:i]`` (``add_generation_prompt=True``) and ``messages[:i+1]``
(``add_generation_prompt=False``) and unmask exactly the token span between the
two lengths. This is template-agnostic and handles parallel tool calls for free.

The generation-prompt header (``<|im_start|>assistant\\n``) sits in the prefix and
stays masked — matching inference, where the server supplies that header.

Depends only on a tokenizer with ``apply_chat_template`` — no torch.
"""

from __future__ import annotations

from typing import Any

IGNORE_INDEX = -100


class TemplateInvariantError(RuntimeError):
    """A chat template broke the strict-prefix invariant between successive renders."""


def _render(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
    template_kwargs: dict[str, Any],
) -> list[int]:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        **template_kwargs,
    )


def build_masked_example(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    max_seq_len: int = 4096,
    on_overlong: str = "drop",
    template_kwargs: dict[str, Any] | None = None,
) -> dict[str, list[int]] | None:
    """Return ``{input_ids, labels, attention_mask}`` or ``None`` when dropped.

    ``labels`` is ``IGNORE_INDEX`` everywhere except the tokens the assistant
    generated (tool-call JSON included).
    """
    template_kwargs = dict(template_kwargs or {})
    template_kwargs.setdefault("enable_thinking", False)

    assistant_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not assistant_idxs:
        return None

    full_ids = _render(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=False,
        template_kwargs=template_kwargs,
    )
    labels = [IGNORE_INDEX] * len(full_ids)

    prev_len = 0
    for i in assistant_idxs:
        prefix = _render(
            tokenizer,
            messages[:i],
            tools=tools,
            add_generation_prompt=True,
            template_kwargs=template_kwargs,
        )
        upto = _render(
            tokenizer,
            messages[: i + 1],
            tools=tools,
            add_generation_prompt=False,
            template_kwargs=template_kwargs,
        )
        if full_ids[: len(prefix)] != prefix or full_ids[: len(upto)] != upto:
            raise TemplateInvariantError(
                f"render of messages[:{i}] / messages[:{i + 1}] is not a prefix of "
                "the full render; template is not append-only"
            )
        if len(prefix) < prev_len or len(upto) <= len(prefix):
            raise TemplateInvariantError(
                f"non-monotonic render lengths at assistant index {i}"
            )
        for pos in range(len(prefix), len(upto)):
            labels[pos] = full_ids[pos]
        prev_len = len(upto)

    if len(full_ids) > max_seq_len:
        if on_overlong == "drop":
            return None
        if on_overlong == "truncate":
            full_ids = full_ids[:max_seq_len]
            labels = labels[:max_seq_len]
        else:
            raise ValueError(f"unknown on_overlong: {on_overlong!r}")

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def render_debug(
    tokenizer: Any,
    example: dict[str, list[int]],
    *,
    masked_char: str = "·",
) -> str:
    """Human-readable view: trained tokens verbatim, masked tokens as ``·``."""
    out: list[str] = []
    for tok, lab in zip(example["input_ids"], example["labels"]):
        piece = tokenizer.decode([tok])
        out.append(piece if lab != IGNORE_INDEX else masked_char * max(len(piece), 1))
    return "".join(out)
