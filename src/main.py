from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import uvicorn

from contextlib import asynccontextmanager
import os

import src.routers
from src.agents.geo_search.graph import compile_geo_graph
from src.utils.llm import make_llm
from src.utils.nominatim import make_nominatim_client
from src.utils.overpass import make_overpass_client

# Each graph node can use a different served model; all share one vLLM endpoint.
_LLM_ROLES = ("intent", "search", "synthesize")


@asynccontextmanager
async def lifespan(app: FastAPI):
    nominatim = make_nominatim_client()
    overpass = make_overpass_client()
    llms: dict[str, BaseChatModel] = {role: make_llm(role) for role in _LLM_ROLES}

    # The Postgres checkpointer shares conversation state across k8s replicas. Its
    # connection pool lives for the duration of the context manager.
    dsn = os.environ["POSTGRES_DSN"]
    async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
        await checkpointer.setup()

        app.state.nominatim = nominatim
        app.state.overpass = overpass
        app.state.geo_graph = compile_geo_graph(
            llms, nominatim, overpass, checkpointer
        )

        try:
            yield
        finally:
            await nominatim.aclose()
            await overpass.aclose()


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
