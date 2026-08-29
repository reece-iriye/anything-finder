from pathlib import Path
from typing import Any, Awaitable, Callable

from deepagents import create_deep_agent
import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

from src.agents.geo_search.state import GeoSearchState
from src.schemas.geo_search.intent import CravingIntent
from src.utils.nominatim import Coordinates, geocode_query, haversine_miles
from src.utils.overpass import query_restaurants
from src.utils.prompts import load_prompt

# A node is an async function mapping the current state to a partial state update.
GeoNode = Callable[[GeoSearchState], Awaitable[dict[str, Any]]]

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

_DEFAULT_RADIUS_M = 2_000
_MAX_RADIUS_M = 16_000
_MIN_GOOD_RESULTS = 5


# --------------------------------------------------------------------------- #
# parse_intent
# --------------------------------------------------------------------------- #


def make_parse_intent_node(llm: BaseChatModel) -> GeoNode:
    # json_schema (vLLM guided decoding) is the robust structured-output path; it does
    # not rely on the served model's tool-calling support.
    structured = llm.with_structured_output(CravingIntent, method="json_schema")
    system_prompt = load_prompt(_PROMPTS_DIR, "intent")

    async def parse_intent(state: GeoSearchState) -> dict[str, Any]:
        raw = state.get("raw_query")
        if not raw:
            return {"errors": ["parse_intent: raw_query is required"]}
        try:
            intent = await structured.ainvoke([("system", system_prompt), ("human", raw)])
        except Exception as exc:  # surface, don't crash the graph
            return {"errors": [f"parse_intent failed: {exc}"]}
        return {"intent": intent}

    return parse_intent


# --------------------------------------------------------------------------- #
# resolve_location
# --------------------------------------------------------------------------- #


def make_resolve_location_node(nominatim: httpx.AsyncClient) -> GeoNode:
    async def resolve_location(state: GeoSearchState) -> dict[str, Any]:
        # Caller-supplied coordinates win; otherwise geocode the intent's location phrase.
        loc = state.get("location")
        if loc is not None:
            return {"resolved_location": loc}

        intent = state.get("intent")
        phrase = intent.location_phrase if intent else None
        if not phrase:
            return {
                "errors": ["resolve_location: no coordinates supplied and no location " "phrase found in the query"]
            }
        try:
            lat, lon = await geocode_query(phrase, nominatim)
        except Exception as exc:
            return {"errors": [f"resolve_location failed for {phrase!r}: {exc}"]}
        return {"resolved_location": Coordinates(lat=lat, lon=lon)}

    return resolve_location


# --------------------------------------------------------------------------- #
# search (bounded deepagents tool-loop)
# --------------------------------------------------------------------------- #


def make_search_node(
    llm: BaseChatModel,
    overpass: httpx.AsyncClient,
    *,
    recursion_limit: int = 12,
) -> GeoNode:
    system_prompt = load_prompt(
        _PROMPTS_DIR,
        "search",
        default_radius=_DEFAULT_RADIUS_M,
        min_results=_MIN_GOOD_RESULTS,
        max_radius=_MAX_RADIUS_M,
    )

    async def search(state: GeoSearchState) -> dict[str, Any]:
        loc = state.get("resolved_location")
        if loc is None:
            return {"errors": ["search: no resolved location to search around"]}

        intent = state.get("intent")
        categories = intent.craving if intent else None
        base_radius = state.get("radius_m") or _DEFAULT_RADIUS_M
        include_casual = state.get("include_casual", False)

        # The tool closes over per-request coords/client; the deep agent only chooses
        # the radius. We capture full results here and hand the LLM a compact view.
        progress: dict[str, Any] = {
            "attempts": 0,
            "widened": False,
            "candidates": [],
        }

        @tool
        async def search_restaurants(radius_m: int) -> list[dict[str, Any]]:
            """Search nearby eateries within radius_m meters of the user.

            Increase radius_m and call again to widen the search when too few
            results are returned.
            """
            radius = min(radius_m, _MAX_RADIUS_M)
            progress["attempts"] += 1
            if radius > base_radius:
                progress["widened"] = True
            results = await query_restaurants(
                overpass,
                lat=loc.lat,
                lon=loc.lon,
                radius_m=radius,
                categories=categories,
                include_casual=include_casual,
            )
            progress["candidates"] = results
            # Compact projection keeps the agent's context small.
            return [{"name": r["name"], "amenity": r["amenity"]} for r in results]

        agent = create_deep_agent(
            model=llm,
            tools=[search_restaurants],
            system_prompt=system_prompt,
        )
        craving = categories or "a place to eat"
        try:
            await agent.ainvoke(
                {
                    "messages": [
                        (
                            "human",
                            f"Find {craving} near me. Start at {base_radius} meters.",
                        )
                    ]
                },
                config={"recursion_limit": recursion_limit},
            )
        except Exception as exc:
            return {
                "candidates": progress["candidates"],
                "widened_geo_search": progress["widened"],
                "search_attempts": progress["attempts"],
                "errors": [f"search loop error: {exc}"],
            }

        return {
            "candidates": progress["candidates"],
            "widened_geo_search": progress["widened"],
            "search_attempts": progress["attempts"],
        }

    return search


# --------------------------------------------------------------------------- #
# rank (pure, deterministic)
# --------------------------------------------------------------------------- #


def rank_node(state: GeoSearchState) -> dict[str, Any]:
    loc = state.get("resolved_location")
    candidates = state.get("candidates", [])
    intent = state.get("intent")
    craving = intent.craving.lower() if intent and intent.craving else ""

    def score(c: dict[str, Any]) -> tuple[float, float]:
        cuisine = str(c.get("tags", {}).get("cuisine", "")).lower()
        name = str(c.get("name", "")).lower()
        match = 1.0 if craving and (craving in cuisine or craving in name) else 0.0
        dist = haversine_miles(loc.lat, loc.lon, c["lat"], c["lon"]) if loc is not None else 0.0
        # Category matches first, then nearer venues (negate distance so higher
        # tuples sort first under reverse=True).
        return (match, -dist)

    ranked = sorted(candidates, key=score, reverse=True)
    return {"ranked": ranked}


# --------------------------------------------------------------------------- #
# synthesize
# --------------------------------------------------------------------------- #


def make_synthesize_node(llm: BaseChatModel) -> GeoNode:
    system_prompt = load_prompt(_PROMPTS_DIR, "synthesize")

    async def synthesize(state: GeoSearchState) -> dict[str, Any]:
        intent = state.get("intent")
        ranked = state.get("ranked", [])
        top = ranked[:5]
        venue_lines = (
            "\n".join(
                f"- {c['name']}" + (f" (cuisine: {c['tags']['cuisine']})" if c.get("tags", {}).get("cuisine") else "")
                for c in top
            )
            or "(no venues found)"
        )
        craving = intent.craving if intent else "food"
        vibe = ", ".join(intent.vibe) if intent and intent.vibe else "no specific vibe"
        human = f"Craving: {craving}\nVibe: {vibe}\nNearby venues:\n{venue_lines}"
        try:
            resp = await llm.ainvoke([("system", system_prompt), ("human", human)])
        except Exception as exc:
            return {"errors": [f"synthesize failed: {exc}"]}
        return {"response": resp.content}

    return synthesize
