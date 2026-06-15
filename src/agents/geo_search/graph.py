from fastapi import Request
import httpx
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from src.agents.geo_search.nodes import (
    make_parse_intent_node,
    make_resolve_location_node,
    make_search_node,
    make_synthesize_node,
    rank_node,
)
from src.agents.geo_search.state import GeoSearchState


def compile_geo_graph(
    llms: dict[str, BaseChatModel],
    nominatim: httpx.AsyncClient,
    overpass: httpx.AsyncClient,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the geo-search workflow once at startup.

    Dependencies (per-role LLMs + singleton HTTP clients) are closure-bound into the
    node factories here, so they never live in the serialized graph state.

    ``llms`` is keyed by role: "intent", "search", "synthesize".
    """
    g = StateGraph(GeoSearchState)
    g.add_node("parse_intent", make_parse_intent_node(llms["intent"]))
    g.add_node("resolve_location", make_resolve_location_node(nominatim))
    g.add_node("search", make_search_node(llms["search"], overpass))
    g.add_node("rank", rank_node)
    g.add_node("synthesize", make_synthesize_node(llms["synthesize"]))

    g.add_edge(START, "parse_intent")
    g.add_edge("parse_intent", "resolve_location")
    g.add_edge("resolve_location", "search")
    g.add_edge("search", "rank")
    g.add_edge("rank", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile(checkpointer=checkpointer)


def get_geo_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.geo_graph
