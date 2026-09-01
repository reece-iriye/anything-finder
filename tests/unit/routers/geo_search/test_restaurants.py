import httpx
from langchain_core.messages import AIMessage
import pytest

from src.agents.geo_search import agent as agent_module
from src.agents.geo_search.agent import build_restaurant_agent
from src.main import app


class _FakeAgent:
    def __init__(self, text: str = "Try Near Sushi for a quiet bite."):
        self._text = text

    async def ainvoke(self, inp, config=None):
        return {"messages": [AIMessage(content=self._text)]}


@pytest.fixture
async def client(monkeypatch, fake_llm, nominatim_client, overpass_client, tmp_path):
    monkeypatch.setattr(
        agent_module,
        "create_deep_agent",
        lambda model, tools, system_prompt, checkpointer=None: _FakeAgent(),
    )

    app.state.nominatim = nominatim_client
    app.state.overpass = overpass_client
    app.state.restaurant_agent = build_restaurant_agent(
        fake_llm,
        nominatim_client,
        overpass_client,
        prefs_dir=tmp_path / "prefs",
        home_city="Dallas",
        home_state="TX",
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
        json={"query": "sushi", "latitude": 33.0},
    )
    assert resp.status_code == 400


async def test_accepts_zero_coordinates(client):
    # 0.0 lat/lon (Gulf of Guinea) is valid and must not be treated as "missing".
    resp = await client.post(
        "/api/geo-search/restaurants/user-123",
        json={"query": "sushi", "latitude": 0.0, "longitude": 0.0},
    )
    assert resp.status_code == 200
