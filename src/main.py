from fastapi import FastAPI
import uvicorn

import os

app = FastAPI(
    title="Anything Finder",
    description="Application for conveniently finding things to do around me.",
)


@app.get("/health")
def health():
    return {"status": "healthy"}


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
