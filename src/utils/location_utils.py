import httpx
from fastapi import Request

import math
import os
from dataclasses import dataclass

_NOMINATIM_USE_EXTERNAL = (
    os.getenv("NOMINATIM_USE_EXTERNAL_API", "false").lower() == "true"
)
_NOMINATIM_BASE_URL = os.getenv(
    "NOMINATIM_BASE_URL",
    (
        "https://nominatim.openstreetmap.org"
        if _NOMINATIM_USE_EXTERNAL
        else "http://nominatim-service:8080"
    ),
)
_NOMINATIM_TIMEOUT = float(os.getenv("NOMINATIM_TIMEOUT_SECONDS", "2.0"))


@dataclass
class ApproximateLocation:
    city: str
    state: str
    country_code: str
    lat: float
    lon: float


@dataclass
class Coordinates:
    lat: float
    lon: float


async def reverse_geocode_with_nominatim(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
) -> ApproximateLocation:
    response = await client.get(
        "/reverse",
        params={
            "lat": lat,
            "lon": lon,
            "format": "jsonv2",
            "zoom": 10,  # city-level; 18=building, 10=city, 6=state
            "addressdetails": 1,
        },
    )
    response.raise_for_status()
    data = response.json()
    address = data.get("address", {})
    return ApproximateLocation(
        city=address.get("city") or address.get("town") or address.get("village") or "",
        state=address.get("state", ""),
        country_code=address.get("country_code", "").upper(),
        lat=float(data["lat"]),
        lon=float(data["lon"]),
    )


async def forward_geocode_with_nominatim(
    city: str,
    state: str,
    client: httpx.AsyncClient,
) -> tuple[float, float]:
    response = await client.get(
        "/search",
        params={
            "city": city,
            "state": state,
            "format": "jsonv2",
            "limit": 1,
        },
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError(f"No coordinates found for {city}, {state}")
    return float(results[0]["lat"]), float(results[0]["lon"])


_EARTH_RADIUS_MILES = 3958.8


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return _EARTH_RADIUS_MILES * 2 * math.asin(math.sqrt(a))


def make_nominatim_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=_NOMINATIM_BASE_URL,
        timeout=_NOMINATIM_TIMEOUT,
        headers={"Accept-Language": "en"},
    )


def get_nominatim_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.nominatim
