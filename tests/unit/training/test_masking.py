from __future__ import annotations

import pytest

from src.training.masking import (
    IGNORE_INDEX,
    TemplateInvariantError,
    build_masked_example,
)


def _trained_text(tok, ex) -> str:
    return "".join(
        chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGNORE_INDEX
    )


def _masked_text(tok, ex) -> str:
    return "".join(
        chr(t) for t, l in zip(ex["input_ids"], ex["labels"]) if l == IGNORE_INDEX
    )


SYSTEM = {"role": "system", "content": "SYSPROMPT"}
USER = {"role": "user", "content": "ramen in Deep Ellum"}


def test_single_assistant_turn(fake_tokenizer):
    messages = [SYSTEM, USER, {"role": "assistant", "content": "Try Ramen Hakata."}]
    ex = build_masked_example(fake_tokenizer, messages, tools=None)
    assert "Try Ramen Hakata." in _trained_text(fake_tokenizer, ex)
    assert "SYSPROMPT" in _masked_text(fake_tokenizer, ex)
    assert "ramen in Deep Ellum" in _masked_text(fake_tokenizer, ex)
    assert "SYSPROMPT" not in _trained_text(fake_tokenizer, ex)


def test_parallel_tool_calls_and_results_masked(fake_tokenizer):
    messages = [
        SYSTEM,
        USER,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function",
                 "function": {"name": "read_food_preferences", "arguments": "{}"}},
                {"id": "b", "type": "function",
                 "function": {"name": "geocode_location",
                              "arguments": '{"place": "Deep Ellum"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "name": "read_food_preferences",
         "content": "SECRET_PREFS"},
        {"role": "tool", "tool_call_id": "b", "name": "geocode_location",
         "content": '{"lat": 1, "lon": 2}'},
        {"role": "assistant", "content": "Ramen Hakata it is."},
    ]
    ex = build_masked_example(fake_tokenizer, messages, tools=None)
    trained = _trained_text(fake_tokenizer, ex)
    masked = _masked_text(fake_tokenizer, ex)

    # both parallel calls are trained
    assert "read_food_preferences" in trained and "geocode_location" in trained
    assert "Ramen Hakata it is." in trained
    # tool RESULTS are never trained
    assert "SECRET_PREFS" in masked and "SECRET_PREFS" not in trained


def test_tools_passed_through_to_template(fake_tokenizer):
    schemas = [{"type": "function", "function": {"name": "search_restaurants"}}]
    messages = [USER, {"role": "assistant", "content": "ok"}]
    ex = build_masked_example(fake_tokenizer, messages, tools=schemas)
    # the <tools> header lands in the masked prefix
    assert "search_restaurants" in _masked_text(fake_tokenizer, ex)


def test_overlong_drop_vs_truncate(fake_tokenizer):
    messages = [USER, {"role": "assistant", "content": "x" * 200}]
    assert build_masked_example(fake_tokenizer, messages, max_seq_len=20) is None
    ex = build_masked_example(
        fake_tokenizer, messages, max_seq_len=20, on_overlong="truncate"
    )
    assert len(ex["input_ids"]) == 20 == len(ex["labels"])


@pytest.mark.live
def test_real_qwen_tokenizer_masks_assistant_only():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    messages = [
        SYSTEM,
        USER,
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "a", "type": "function",
              "function": {"name": "geocode_location",
                           "arguments": '{"place": "Deep Ellum"}'}}]},
        {"role": "tool", "tool_call_id": "a", "name": "geocode_location",
         "content": '{"lat": 1, "lon": 2}'},
        {"role": "assistant", "content": "Ramen Hakata."},
    ]
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "geocode_location",
                "description": "geocode",
                "parameters": {
                    "type": "object",
                    "properties": {"place": {"type": "string"}},
                    "required": ["place"],
                },
            },
        }
    ]
    ex = build_masked_example(tok, messages, tools=schemas)
    trained = tok.decode(
        [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != IGNORE_INDEX]
    )
    assert "Ramen Hakata." in trained
    assert "Deep Ellum" in trained          # the tool-call args are trained
    assert "SYSPROMPT" not in trained
    assert '"lat": 1' not in trained         # the tool result is not


def test_broken_template_raises():
    class BadTokenizer:
        def apply_chat_template(self, messages, tools=None, add_generation_prompt=False,
                                tokenize=False, **kw):
            # not append-only: returns a constant regardless of history
            return [1, 2, 3] if tokenize else "abc"

    messages = [USER, {"role": "assistant", "content": "hi"}, USER,
                {"role": "assistant", "content": "bye"}]
    with pytest.raises(TemplateInvariantError):
        build_masked_example(BadTokenizer(), messages)
