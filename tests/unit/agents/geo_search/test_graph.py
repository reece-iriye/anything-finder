import re

from langgraph.checkpoint.memory import InMemorySaver

from src.agents.geo_search import nodes
from src.agents.geo_search.graph import compile_geo_graph
from src.utils.nominatim import Coordinates
from tests.conftest import NOMINATIM_BASE, OVERPASS_BASE, overpass_payload


class _SingleCallAgent:
    def __init__(self, tool):
        self._tool = tool

    async def ainvoke(self, _inp, config=None):
        await self._tool.ainvoke({"radius_m": 2000})
        return {"messages": []}


async def test_full_pipeline_and_persistence(
    monkeypatch, fake_llms, nominatim_client, overpass_client, httpx_mock
):
    httpx_mock.add_response(
        method="GET",
        url=re.compile(rf"{NOMINATIM_BASE}/search"),
        json=[{"lat": "33.0", "lon": "-96.0"}],
    )
    httpx_mock.add_response(
        method="POST",
        url=re.compile(rf"{OVERPASS_BASE}/api/interpreter"),
        json=overpass_payload(("Near Sushi", "sushi", 33.01, -96.0)),
    )
    monkeypatch.setattr(
        nodes,
        "create_deep_agent",
        lambda model, tools, system_prompt: _SingleCallAgent(tools[0]),
    )

    checkpointer = InMemorySaver()
    graph = compile_geo_graph(fake_llms, nominatim_client, overpass_client, checkpointer)

    # Node order is the declared linear pipeline.
    node_names = {n for n in graph.get_graph().nodes} - {"__start__", "__end__"}
    assert node_names == {
        "parse_intent",
        "resolve_location",
        "search",
        "rank",
        "synthesize",
    }

    cfg = {"configurable": {"thread_id": "t1"}}
    result = await graph.ainvoke(
        {"raw_query": "quiet sushi in The Colony", "errors": []}, config=cfg
    )
    assert result["response"].startswith("Try Near Sushi")
    assert not result["errors"]

    # Checkpointer persisted intermediate state under the thread id.
    snap = await graph.aget_state(cfg)
    assert snap.values["intent"].craving == "sushi"
    assert snap.values["resolved_location"] == Coordinates(lat=33.0, lon=-96.0)
    assert snap.values["ranked"][0]["name"] == "Near Sushi"
