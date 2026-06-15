from fastapi import FastAPI
import httpx
from langgraph.graph.state import CompiledStateGraph
import uvicorn

from contextlib import asynccontextmanager
import os

import src.routers
from src.agents.geo_search.graph import compile_geo_graph
from src.utils.nominatim import make_nominatim_client
from src.utils.overpass import make_overpass_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    nominatim: httpx.AsyncClient = make_nominatim_client()
    overpass: httpx.AsyncClient = make_overpass_client()
    geo_graph: CompiledStateGraph = compile_geo_graph()

    app.state.nominatim = nominatim
    app.state.overpass = overpass
    app.state.geo_graph = geo_graph

    yield

    await app.state.nominatim.aclose()
    await app.state.overpass.aclose()
    await app.state.geo_graph.aclear_cache()


app = FastAPI(
    title="Anything Finder",
    description="Application for conveniently finding things to do.",
    version="v0.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "healthy"}


app.include_router(src.routers.geo_search.restaurants_router)


if __name__ == "__main__":
    config = uvicorn.Config(
        "src.main:app",
        host=os.getenv("APP_IP", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "9022")),
        log_level=os.getenv("APP_LOG_LEVEL", "info"),
        reload=False,
    )
    server = uvicorn.Server(config)
    server.run()
