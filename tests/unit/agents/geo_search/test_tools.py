import re

import pytest

from src.agents.geo_search.tools import (
    make_geocode_location,
    make_get_current_location,
    make_read_food_preferences,
    make_search_restaurants,
)
from tests.conftest import NOMINATIM_BASE, OVERPASS_BASE, overpass_payload


# ---------- read_food_preferences ----------


async def test_read_preferences_returns_text(tmp_path):
    prefs_dir = tmp_path / "prefs"
    prefs_dir.mkdir()
    (prefs_dir / "user-abc.md").write_text("Loves sushi.", encoding="utf-8")
    tool_fn = make_read_food_preferences(prefs_dir)
    result = await tool_fn.ainvoke(
        {}, config={"configurable": {"user_id": "user-abc"}}
    )
    assert "sushi" in result.lower()


async def test_read_preferences_missing_file(tmp_path):
    tool_fn = make_read_food_preferences(tmp_path / "prefs")
    result = await tool_fn.ainvoke(
        {}, config={"configurable": {"user_id": "no-such-user"}}
    )
    assert "no food preferences" in result.lower()


# ---------- get_current_location ----------


async def test_get_current_location_returns_home(nominatim_client, httpx_mock):
    httpx_mock.add_response(
        method="GET",
        url=re.compile(rf"{NOMINATIM_BASE}/search"),
        json=[{"lat": "33.0", "lon": "-96.0"}],
    )
    tool_fn = make_get_current_location(nominatim_client, "Dallas", "TX")
    result = await tool_fn.ainvoke({})
    assert result["lat"] == 33.0
    assert result["lon"] == -96.0
    assert "Dallas" in result["label"]


# ---------- geocode_location ----------


async def test_geocode_location_returns_coords(nominatim_client, httpx_mock):
    httpx_mock.add_response(
        method="GET",
        url=re.compile(rf"{NOMINATIM_BASE}/search"),
        json=[{"lat": "32.5", "lon": "-97.3"}],
    )
    tool_fn = make_geocode_location(nominatim_client)
    result = await tool_fn.ainvoke({"place": "The Colony, TX"})
    assert result["lat"] == 32.5
    assert result["lon"] == -97.3


# ---------- search_restaurants ----------


async def test_search_restaurants_annotates_distance(overpass_client, httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url=re.compile(rf"{OVERPASS_BASE}/api/interpreter"),
        json=overpass_payload(("Sushi Bar", "sushi", 33.01, -96.0)),
    )
    tool_fn = make_search_restaurants(overpass_client)
    results = await tool_fn.ainvoke({"lat": 33.0, "lon": -96.0, "radius_m": 2000})
    assert len(results) == 1
    assert results[0]["name"] == "Sushi Bar"
    assert results[0]["cuisine"] == "sushi"
    assert results[0]["distance_mi"] > 0


async def test_search_restaurants_caps_at_max_radius(overpass_client, httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url=re.compile(rf"{OVERPASS_BASE}/api/interpreter"),
        json=overpass_payload(("Big Burger", "burger", 33.0, -96.0)),
    )
    tool_fn = make_search_restaurants(overpass_client)
    results = await tool_fn.ainvoke({"lat": 33.0, "lon": -96.0, "radius_m": 999_999})
    assert results[0]["name"] == "Big Burger"
