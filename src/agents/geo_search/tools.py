from pathlib import Path
from typing import Annotated, Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from src.utils.nominatim import forward_geocode_with_nominatim, geocode_query, haversine_miles
from src.utils.overpass import query_restaurants

_MAX_RESULTS = 10
_MAX_RADIUS_M = 16_000


def make_read_food_preferences(prefs_dir: Path):
    @tool
    async def read_food_preferences(
        config: Annotated[RunnableConfig, InjectedToolArg],
    ) -> str:
        """Read the current user's food preferences from their profile."""
        user_id = config["configurable"]["user_id"]
        path = prefs_dir / f"{user_id}.md"
        if not path.exists():
            return "No food preferences on file for this user."
        return path.read_text(encoding="utf-8").strip()

    return read_food_preferences


def make_get_current_location(
    nominatim: httpx.AsyncClient, home_city: str, home_state: str
):
    @tool
    async def get_current_location() -> dict[str, Any]:
        """Get the user's home location as a coordinate fallback when no location is mentioned."""
        lat, lon = await forward_geocode_with_nominatim(
            city=home_city, state=home_state, client=nominatim
        )
        return {"lat": lat, "lon": lon, "label": f"{home_city}, {home_state}"}

    return get_current_location


def make_geocode_location(nominatim: httpx.AsyncClient):
    @tool
    async def geocode_location(place: str) -> dict[str, Any]:
        """Geocode a place name or phrase mentioned in the user's query to coordinates."""
        lat, lon = await geocode_query(place, nominatim)
        return {"lat": lat, "lon": lon}

    return geocode_location


def make_search_restaurants(overpass: httpx.AsyncClient):
    @tool
    async def search_restaurants(
        lat: float,
        lon: float,
        radius_m: int,
        cuisine: str | None = None,
        include_casual: bool = False,
    ) -> list[dict[str, Any]]:
        """Search for restaurants near (lat, lon) within radius_m meters.

        Increase radius_m and call again to widen the search when too few results
        are returned.
        """
        radius = min(radius_m, _MAX_RADIUS_M)
        results = await query_restaurants(
            overpass,
            lat=lat,
            lon=lon,
            radius_m=radius,
            categories=cuisine,
            include_casual=include_casual,
        )
        projected = [
            {
                "name": r["name"],
                "amenity": r["amenity"],
                "cuisine": r.get("tags", {}).get("cuisine"),
                "distance_mi": round(haversine_miles(lat, lon, r["lat"], r["lon"]), 2),
            }
            for r in results
        ]
        return projected[:_MAX_RESULTS]

    return search_restaurants
