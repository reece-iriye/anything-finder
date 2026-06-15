from fastapi import Request
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI

from src.agents.geo_search.state import GeoSearchState


def compile_geo_graph(llm: ChatOpenAI) -> CompiledStateGraph:
    g = StateGraph(GeoSearchState)
    g.add_node("parse_intent")
    g.add_edge(START, "parse_intent")
    g.add_edge("parse_intent", END)
    return g.compile()


def get_geo_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.geo_search
