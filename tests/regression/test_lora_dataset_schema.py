"""Stability guards for the LoRA training dataset + exported tool schemas.

Pins the prepared-record key set and the tool-schema names / required-arg sets so
an accidental change to the conversion or the production tool signatures fails
loudly. Update intentionally when the contract changes.
"""

from __future__ import annotations

from src.training.dataset import row_to_record
from src.training.tool_schemas import build_tool_schemas

_TRANSCRIPT = [
    {"role": "human", "content": "sushi in Deep Ellum"},
    {"role": "ai", "content": [
        {"type": "tool_use", "id": "t1", "name": "read_food_preferences", "input": {}},
    ]},
    {"role": "tool", "tool_call_id": "t1", "name": "read_food_preferences",
     "content": "prefs"},
    {"role": "ai", "content": "Try Deep Sushi."},
]


def test_prepared_record_keys():
    rec = row_to_record(
        {"id": "q-1", "completion": "Try Deep Sushi.", "transcript": _TRANSCRIPT},
        "SYS",
    )
    assert set(rec) == {"id", "messages"}
    assert set(rec["messages"][2]) == {"role", "content", "tool_calls"}
    assert set(rec["messages"][3]) == {"role", "tool_call_id", "name", "content"}


def test_exported_tool_schema_contract():
    fns = {s["function"]["name"]: s["function"] for s in build_tool_schemas()}
    assert set(fns) == {
        "read_food_preferences",
        "get_current_location",
        "geocode_location",
        "search_restaurants",
    }
    required = {name: set(fn["parameters"].get("required", [])) for name, fn in fns.items()}
    assert required == {
        "read_food_preferences": set(),
        "get_current_location": set(),
        "geocode_location": {"place"},
        "search_restaurants": {"lat", "lon", "radius_m"},
    }
