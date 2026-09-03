"""Real tool signatures -> OpenAI JSON tool schemas.

The model must be trained against the *same* tool contract production uses, so we
build the actual tool objects from the factories in ``src.agents.geo_search.tools``
(the ``@tool`` decorator infers the schema at decoration time — no network, no
client calls) and run them through langchain's ``convert_to_openai_tool``.

``read_food_preferences``'s ``InjectedToolArg`` config parameter is dropped by
langchain automatically, which is correct: the model never sees or supplies it.

Schemas are exported once to ``data/lora/tool_schemas.json`` so ``train_lora.py``
never needs langchain. A regression test guards the exported shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The four domain tools, in the order the system prompt walks through them.
DOMAIN_TOOL_NAMES = (
    "read_food_preferences",
    "get_current_location",
    "geocode_location",
    "search_restaurants",
)


def _domain_tools() -> list[Any]:
    import httpx
    from src.agents.geo_search.tools import (
        make_geocode_location,
        make_get_current_location,
        make_read_food_preferences,
        make_search_restaurants,
    )

    # Dummy deps: never called, only closure-bound. The schema is already fixed
    # by the time these factories return.
    dummy_client = httpx.AsyncClient()
    dummy_prefs = Path("/nonexistent/prefs")
    return [
        make_read_food_preferences(dummy_prefs),
        make_get_current_location(dummy_client, "Dallas", "TX"),
        make_geocode_location(dummy_client),
        make_search_restaurants(dummy_client),
    ]


def build_tool_schemas(*, include_builtin_tools: bool = False) -> list[dict[str, Any]]:
    """Return OpenAI-format tool schemas for the restaurant agent.

    ``include_builtin_tools`` additionally introspects a compiled deep agent for
    the deepagents built-ins (``ls``, ``read_file``, ``task``, …). Off by default:
    the captured dataset shows they were never called.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    tools = _domain_tools()
    schemas = [convert_to_openai_tool(t) for t in tools]

    if include_builtin_tools:
        try:
            schemas.extend(
                _builtin_tool_schemas({t["function"]["name"] for t in schemas})
            )
        except Exception as exc:  # noqa: BLE001 — built-ins are best-effort
            import warnings

            warnings.warn(f"could not introspect deepagents built-ins: {exc}")

    return schemas


def _builtin_tool_schemas(already: set[str]) -> list[dict[str, Any]]:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from deepagents import create_deep_agent

    compiled = create_deep_agent(model=None, tools=[], system_prompt="x")
    out: list[dict[str, Any]] = []
    seen = set(already)
    for tool in getattr(compiled, "tools", []) or []:
        try:
            schema = convert_to_openai_tool(tool)
        except Exception:  # noqa: BLE001 — best effort for optional built-ins
            continue
        name = schema["function"]["name"]
        if name in seen:
            continue
        seen.add(name)
        out.append(schema)
    return out


def export_tool_schemas(
    out_path: str | Path, *, include_builtin_tools: bool = False
) -> list[dict[str, Any]]:
    schemas = build_tool_schemas(include_builtin_tools=include_builtin_tools)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(schemas, indent=2) + "\n", encoding="utf-8")
    return schemas


def load_tool_schemas(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
