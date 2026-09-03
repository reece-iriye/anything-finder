from langchain_core.messages import AIMessage
import pytest

from src.agents.geo_search import agent as agent_module
from src.agents.geo_search.agent import build_restaurant_agent


class _FakeAgent:
    """Returns a fixed AI message."""

    def __init__(self, text: str = "Try Near Sushi for a quiet bite."):
        self._text = text

    async def ainvoke(self, inp, config=None):
        return {"messages": [AIMessage(content=self._text)]}


async def test_build_agent_wires_four_tools(
    monkeypatch, fake_llm, nominatim_client, overpass_client, tmp_path
):
    built_tools = []

    def fake_create(model, tools, system_prompt, checkpointer=None):
        built_tools.extend(tools)
        return _FakeAgent()

    monkeypatch.setattr(agent_module, "create_deep_agent", fake_create)

    agent = build_restaurant_agent(
        fake_llm,
        nominatim_client,
        overpass_client,
        prefs_dir=tmp_path / "prefs",
        home_city="Dallas",
        home_state="TX",
    )
    assert len(built_tools) == 4
    result = await agent.ainvoke({"messages": [("human", "sushi")]})
    assert result["messages"][-1].content == "Try Near Sushi for a quiet bite."


async def test_service_extracts_final_ai_message(
    monkeypatch, fake_llm, nominatim_client, overpass_client, tmp_path
):
    monkeypatch.setattr(
        agent_module,
        "create_deep_agent",
        lambda model, tools, system_prompt, checkpointer=None: _FakeAgent(),
    )

    from src.services.geo_search.restaurants import RestaurantSearch

    agent = build_restaurant_agent(
        fake_llm,
        nominatim_client,
        overpass_client,
        prefs_dir=tmp_path / "prefs",
        home_city="Dallas",
        home_state="TX",
    )
    search = await RestaurantSearch.create(
        user_id="user-123",
        raw_query="quiet sushi tonight",
        restaurant_agent=agent,
    )
    response, err, code = await search.ainvoke_search_workflow()
    assert err is None
    assert response == "Try Near Sushi for a quiet bite."
