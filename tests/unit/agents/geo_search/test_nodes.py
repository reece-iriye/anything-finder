import re

from src.agents.geo_search import nodes
from src.schemas.geo_search.intent import CravingIntent
from src.utils.nominatim import Coordinates
from tests.conftest import NOMINATIM_BASE, OVERPASS_BASE, overpass_payload


# --------------------------- parse_intent --------------------------- #


async def test_parse_intent_extracts_structured(fake_llms):
    node = nodes.make_parse_intent_node(fake_llms["intent"])
    out = await node({"raw_query": "quiet sushi in The Colony"})
    assert out["intent"].craving == "sushi"


async def test_parse_intent_requires_query(fake_llms):
    node = nodes.make_parse_intent_node(fake_llms["intent"])
    out = await node({})
    assert out["errors"]


# --------------------------- resolve_location --------------------------- #


async def test_resolve_prefers_supplied_coords(nominatim_client):
    node = nodes.make_resolve_location_node(nominatim_client)
    out = await node({"location": Coordinates(lat=1.0, lon=2.0)})
    assert out["resolved_location"] == Coordinates(lat=1.0, lon=2.0)


async def test_resolve_geocodes_phrase(nominatim_client, httpx_mock, craving_intent):
    httpx_mock.add_response(
        method="GET",
        url=re.compile(rf"{NOMINATIM_BASE}/search"),
        json=[{"lat": "33.0", "lon": "-96.0"}],
    )
    node = nodes.make_resolve_location_node(nominatim_client)
    out = await node({"intent": craving_intent})
    assert out["resolved_location"] == Coordinates(lat=33.0, lon=-96.0)


async def test_resolve_errors_without_location(nominatim_client):
    node = nodes.make_resolve_location_node(nominatim_client)
    out = await node({"intent": CravingIntent(craving="sushi")})  # no phrase/coords
    assert out["errors"]


# --------------------------- search (deepagents loop) --------------------------- #


class _FakeSearchAgent:
    """Drives the bound search tool over a fixed sequence of radii."""

    def __init__(self, tool, radii):
        self._tool = tool
        self._radii = radii

    async def ainvoke(self, _inp, config=None):
        for r in self._radii:
            await self._tool.ainvoke({"radius_m": r})
        return {"messages": []}


async def test_search_widens_when_sparse(
    monkeypatch, overpass_client, httpx_mock, fake_llms, craving_intent
):
    # First (base radius) returns 1 result; second (widened) returns 6.
    httpx_mock.add_response(
        method="POST",
        url=re.compile(rf"{OVERPASS_BASE}/api/interpreter"),
        json=overpass_payload(("Lonely Sushi", "sushi", 33.0, -96.0)),
    )
    httpx_mock.add_response(
        method="POST",
        url=re.compile(rf"{OVERPASS_BASE}/api/interpreter"),
        json=overpass_payload(
            *[(f"Sushi {i}", "sushi", 33.0, -96.0) for i in range(6)]
        ),
    )
    monkeypatch.setattr(
        nodes,
        "create_deep_agent",
        lambda model, tools, system_prompt: _FakeSearchAgent(tools[0], [2000, 4000]),
    )

    node = nodes.make_search_node(fake_llms["search"], overpass_client)
    out = await node(
        {"intent": craving_intent, "resolved_location": Coordinates(33.0, -96.0)}
    )
    assert out["search_attempts"] == 2
    assert out["widened_geo_search"] is True
    assert len(out["candidates"]) == 6  # captures the last (widened) result set


async def test_search_requires_resolved_location(fake_llms, overpass_client):
    node = nodes.make_search_node(fake_llms["search"], overpass_client)
    out = await node({"intent": CravingIntent(craving="sushi")})
    assert out["errors"]


# --------------------------- rank --------------------------- #


def test_rank_prefers_category_match_then_proximity():
    state = {
        "resolved_location": Coordinates(lat=33.0, lon=-96.0),
        "intent": CravingIntent(craving="sushi"),
        "candidates": [
            {"name": "Far Pizza", "lat": 33.2, "lon": -96.0, "tags": {"cuisine": "pizza"}},
            {"name": "Near Sushi", "lat": 33.01, "lon": -96.0, "tags": {"cuisine": "sushi"}},
            {"name": "Far Sushi", "lat": 33.3, "lon": -96.0, "tags": {"cuisine": "sushi"}},
        ],
    }
    ranked = nodes.rank_node(state)["ranked"]
    assert [c["name"] for c in ranked] == ["Near Sushi", "Far Sushi", "Far Pizza"]
