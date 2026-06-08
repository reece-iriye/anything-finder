from fastapi import FastAPI
import uvicorn

from contextlib import asynccontextmanager
import os

import src.routers
from src.utils.location_utils import make_nominatim_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.nominatim = make_nominatim_client()
    yield
    await app.state.nominatim.aclose()


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
