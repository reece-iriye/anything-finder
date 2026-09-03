from __future__ import annotations

import pytest

from src.training.dataset import (
    ConversionStats,
    DropRow,
    build_records,
    row_to_record,
    split_records,
    transcript_to_messages,
)

SYS = "SYSTEM PROMPT"


def _tool_use(tid, name, inp):
    return {"id": tid, "caller": {"type": "direct"}, "input": inp,
            "name": name, "type": "tool_use", "toolset_name": None}


def _thinking():
    return {"type": "thinking", "thinking": "", "signature": "redacted=="}


BASE_TRANSCRIPT = [
    {"role": "human", "content": "Indian near Fair Park, upscale"},
    {"role": "ai", "content": [
        _thinking(),
        _tool_use("t1", "read_food_preferences", {}),
        _tool_use("t2", "geocode_location", {"place": "Fair Park"}),
    ]},
    {"role": "tool", "tool_call_id": "t1", "name": "read_food_preferences",
     "content": "# prefs\nLoves Indian"},
    {"role": "tool", "tool_call_id": "t2", "name": "geocode_location",
     "content": {"lat": 32.7, "lon": -96.7}},
    {"role": "ai", "content": [_thinking(), _tool_use(
        "t3", "search_restaurants",
        {"lat": 32.7, "lon": -96.7, "radius_m": 2000, "cuisine": "Indian"})]},
    {"role": "tool", "tool_call_id": "t3", "name": "search_restaurants", "content": []},
    {"role": "ai", "content": "Try Gymkhana or Namak."},
]


def _row(**over):
    row = {"id": "q-1", "completion": "Try Gymkhana or Namak.",
           "transcript": [dict(m) for m in BASE_TRANSCRIPT]}
    row.update(over)
    return row


def test_happy_path_shape():
    rec = row_to_record(_row(), SYS)
    roles = [m["role"] for m in rec["messages"]]
    assert roles[0] == "system" and rec["messages"][0]["content"] == SYS
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant",
                     "tool", "assistant"]


def test_parallel_tool_calls_preserved():
    rec = row_to_record(_row(), SYS)
    first_ai = rec["messages"][2]
    assert [tc["function"]["name"] for tc in first_ai["tool_calls"]] == [
        "read_food_preferences", "geocode_location"]
    assert first_ai["tool_calls"][1]["function"]["arguments"] == '{"place": "Fair Park"}'


def test_thinking_blocks_stripped():
    rec = row_to_record(_row(), SYS)
    for m in rec["messages"]:
        assert "thinking" not in (m.get("content") or "")
        assert m["content"] == "" or "signature" not in m["content"]


def test_empty_list_tool_content_json_encoded():
    rec = row_to_record(_row(), SYS)
    search_result = rec["messages"][6]
    assert search_result["role"] == "tool" and search_result["content"] == "[]"


def test_error_row_dropped():
    with pytest.raises(DropRow):
        row_to_record(_row(error="Nominatim miss"), SYS)


def test_empty_completion_dropped():
    with pytest.raises(DropRow):
        row_to_record(_row(completion="  "), SYS)


def test_unanswered_tool_call_dropped():
    t = [dict(m) for m in BASE_TRANSCRIPT]
    del t[5]  # drop the search_restaurants tool result
    with pytest.raises(DropRow):
        row_to_record(_row(transcript=t), SYS)


def test_dangling_tool_result_dropped():
    t = [dict(m) for m in BASE_TRANSCRIPT]
    t.insert(6, {"role": "tool", "tool_call_id": "ghost", "name": "x", "content": "y"})
    with pytest.raises(DropRow):
        row_to_record(_row(transcript=t), SYS)


def test_last_message_must_be_text_assistant():
    t = [dict(m) for m in BASE_TRANSCRIPT][:-1]
    with pytest.raises(DropRow):
        row_to_record(_row(transcript=t), SYS)


def test_system_prompt_prepended_via_transcript_to_messages():
    msgs = transcript_to_messages(BASE_TRANSCRIPT, SYS)
    assert msgs[0] == {"role": "system", "content": SYS}


def test_build_records_dedup_and_stats():
    stats = ConversionStats()
    recs = build_records([_row(), _row(), _row(error="x")], SYS, stats=stats)
    assert len(recs) == 1
    assert stats.kept == 1 and stats.dropped == 2
    assert "row carries an error field" in stats.drop_reasons


def test_split_deterministic():
    recs = [{"id": f"q-{i}", "messages": []} for i in range(20)]
    a1, b1 = split_records(recs, val_frac=0.1, seed=7)
    a2, b2 = split_records(list(reversed(recs)), val_frac=0.1, seed=7)
    assert [r["id"] for r in b1] == [r["id"] for r in b2]
    assert len(b1) == 2 and len(a1) == 18
