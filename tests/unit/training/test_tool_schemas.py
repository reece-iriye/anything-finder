from __future__ import annotations

from src.training.tool_schemas import (
    DOMAIN_TOOL_NAMES,
    build_tool_schemas,
    export_tool_schemas,
    load_tool_schemas,
)


def _by_name(schemas):
    return {s["function"]["name"]: s["function"] for s in schemas}


def test_all_four_domain_tools_exported():
    fns = _by_name(build_tool_schemas())
    assert set(fns) == set(DOMAIN_TOOL_NAMES)


def test_search_restaurants_arg_contract():
    fn = _by_name(build_tool_schemas())["search_restaurants"]
    params = fn["parameters"]
    assert set(params["required"]) == {"lat", "lon", "radius_m"}
    assert "cuisine" in params["properties"] and "include_casual" in params["properties"]
    assert "cuisine" not in params["required"]
    assert "include_casual" not in params["required"]


def test_read_food_preferences_hides_injected_config():
    fn = _by_name(build_tool_schemas())["read_food_preferences"]
    assert "config" not in fn["parameters"].get("properties", {})
    assert not fn["parameters"].get("required")


def test_export_round_trips(tmp_path):
    path = tmp_path / "tool_schemas.json"
    export_tool_schemas(path)
    assert {s["function"]["name"] for s in load_tool_schemas(path)} == set(DOMAIN_TOOL_NAMES)
