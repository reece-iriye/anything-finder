from fastapi import Request
import httpx

import os

_OVERPASS_USE_EXTERNAL: bool = (
    os.getenv("OVERPASS_USE_EXTERNAL_API", "false").lower() == "true"
)
_OVERPASS_USE_KUBERNETES_WITH_GEOSEARCH_NAMESPACE: bool = (
    os.getenv("OVERPASS_USE_KUBERNETES_WITH_GEOSEARCH_NAMESPACE", "false").lower()
    == "true"
)
_OVERPASS_BASE_URL: str = os.getenv(
    "OVERPASS_BASE_URL",
    (
        "https://overpass-api.de"
        if _OVERPASS_USE_EXTERNAL
        else (
            "http://overpass-service.geosearch.svc.cluster.local:8080"
            if _OVERPASS_USE_KUBERNETES_WITH_GEOSEARCH_NAMESPACE
            else "http://overpass-service:8080"
        )
    ),
)
_OVERPASS_TIMEOUT = float(os.getenv("OVERPASS_TIMEOUT_SECONDS", "3.0"))


def make_overpass_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=_OVERPASS_BASE_URL,
        timeout=httpx.Timeout(_OVERPASS_TIMEOUT, connect=2.0),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30.0,
        ),
    )


def get_overpass_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.overpass
