from pathlib import Path

from fastapi import Request
import httpx
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from deepagents import create_deep_agent

from src.agents.geo_search.tools import (
    make_geocode_location,
    make_get_current_location,
    make_read_food_preferences,
    make_search_restaurants,
)
from src.utils.prompts import load_prompt

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


def build_restaurant_agent(
    llm: BaseChatModel,
    nominatim: httpx.AsyncClient,
    overpass: httpx.AsyncClient,
    *,
    prefs_dir: Path,
    home_city: str,
    home_state: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the restaurant-search deep agent once at startup.

    HTTP clients and static config are closure-bound into tool factories;
    per-request state (user_id, thread_id) flows through RunnableConfig.
    """
    tools = [
        make_read_food_preferences(prefs_dir),
        make_get_current_location(nominatim, home_city, home_state),
        make_geocode_location(nominatim),
        make_search_restaurants(overpass),
    ]
    return create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=load_prompt(_PROMPTS_DIR, "restaurant_agent"),
        checkpointer=checkpointer,
    )


def get_restaurant_agent(request: Request) -> CompiledStateGraph:
    return request.app.state.restaurant_agent
