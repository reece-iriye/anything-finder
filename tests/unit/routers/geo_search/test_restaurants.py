import httpx
from langgraph.checkpoint.memory import InMemorySaver
import pytest

from src.agents.geo_search import nodes
from src.agents.geo_search.graph import compile_geo_graph
from src.main import app


class _SingleCallAgent:
    def __init__(self, tool):
        self._tool = tool

    async def ainvoke(self, _inp, config=None):
        await self._tool.ainvoke({"radius_m": 2000})
        return {"messages": []}


@pytest.fixture
async def client(monkeypatch, fake_llms, nominatim_client, overpass_client):
    # Stub Overpass at the tool boundary so the API test needs no live HTTP and avoids
    # any transport conflict with the ASGI test client.
    async def fake_query_restaurants(_client, **kwargs):
        return [
            {"name": "Near Sushi", "amenity": "restaurant", "lat": 33.0, "lon": -96.0,
             "tags": {"cuisine": "sushi"}}
        ]

    monkeypatch.setattr(nodes, "query_restaurants", fake_query_restaurants)
    monkeypatch.setattr(
        nodes,
        "create_deep_agent",
        lambda model, tools, system_prompt: _SingleCallAgent(tools[0]),
    )

    # ASGITransport does not run the lifespan, so wire app.state by hand.
    app.state.nominatim = nominatim_client
    app.state.overpass = overpass_client
    app.state.geo_graph = compile_geo_graph(
        fake_llms, nominatim_client, overpass_client, InMemorySaver()
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_search_endpoint_returns_response(client):
    resp = await client.post(
        "/api/geo-search/restaurants/user-123",
        json={"query": "quiet sushi", "latitude": 33.0, "longitude": -96.0},
    )
    assert resp.status_code == 200
    assert resp.json()["response"].startswith("Try Near Sushi")


async def test_rejects_place_and_coords_together(client):
    resp = await client.post(
        "/api/geo-search/restaurants/user-123",
        json={
            "query": "sushi",
            "city": "The Colony",
            "state": "TX",
            "latitude": 33.0,
            "longitude": -96.0,
        },
    )
    assert resp.status_code == 400


async def test_rejects_partial_coordinates(client):
    resp = await client.post(
        "/api/geo-search/restaurants/user-123",
        json={"query": "sushi", "latitude": 33.0},  # missing longitude
    )
    assert resp.status_code == 400


async def test_accepts_zero_coordinates(client):
    # 0.0 lat/lon (Gulf of Guinea) is valid and must not be treated as "missing".
    resp = await client.post(
        "/api/geo-search/restaurants/user-123",
        json={"query": "sushi", "latitude": 0.0, "longitude": 0.0},
    )
    assert resp.status_code == 200
