from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import uvicorn

import src.routers
from src.agents.geo_search.agent import build_restaurant_agent
from src.utils.llm import make_llm
from src.utils.nominatim import make_nominatim_client
from src.utils.overpass import make_overpass_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    nominatim = make_nominatim_client()
    overpass = make_overpass_client()
    llm = make_llm("agent")

    prefs_dir = Path(os.environ.get("FOOD_PREFERENCES_DIR", "data/preferences"))
    home_city = os.environ.get("HOME_CITY", "Dallas")
    home_state = os.environ.get("HOME_STATE", "TX")

    dsn = os.environ["POSTGRES_DSN"]
    async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
        await checkpointer.setup()

        app.state.nominatim = nominatim
        app.state.overpass = overpass
        app.state.restaurant_agent = build_restaurant_agent(
            llm,
            nominatim,
            overpass,
            prefs_dir=prefs_dir,
            home_city=home_city,
            home_state=home_state,
            checkpointer=checkpointer,
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
