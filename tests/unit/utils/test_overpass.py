import re

from src.utils.overpass import build_restaurant_query, query_restaurants
from tests.conftest import OVERPASS_BASE, overpass_payload


def test_query_restaurant_only_by_default():
    q = build_restaurant_query(lat=33.1, lon=-96.8, radius_m=2000)
    assert '["amenity"="restaurant"]' in q
    assert "fast_food" not in q
    assert "around:2000,33.1,-96.8" in q


def test_query_casual_broadens_amenity():
    q = build_restaurant_query(lat=33.1, lon=-96.8, radius_m=2000, include_casual=True)
    for amenity in ("restaurant", "fast_food", "cafe", "bar", "pub"):
        assert amenity in q
    assert '["amenity"~"^(restaurant|fast_food|cafe|bar|pub)$"]' in q


def test_query_adds_cuisine_filter():
    q = build_restaurant_query(
        lat=33.1, lon=-96.8, radius_m=2000, categories=["sushi", "ramen"]
    )
    assert '["cuisine"~"sushi|ramen",i]' in q


def test_query_includes_node_way_relation():
    q = build_restaurant_query(lat=33.1, lon=-96.8, radius_m=2000)
    assert q.count("around:2000,33.1,-96.8") == 3  # node + way + relation


async def test_query_restaurants_normalizes_and_skips_unnamed(
    overpass_client, httpx_mock
):
    payload = overpass_payload(("Near Sushi", "sushi", 33.01, -96.0))
    payload["elements"].append({"type": "node", "lat": 1, "lon": 2, "tags": {}})
    payload["elements"].append(
        {
            "type": "way",
            "center": {"lat": 33.5, "lon": -96.5},
            "tags": {"name": "Way Diner", "amenity": "restaurant"},
        }
    )
    httpx_mock.add_response(
        method="POST",
        url=re.compile(rf"{OVERPASS_BASE}/api/interpreter"),
        json=payload,
    )

    results = await query_restaurants(
        overpass_client, lat=33.0, lon=-96.0, radius_m=2000, categories="sushi"
    )
    names = {r["name"] for r in results}
    assert names == {"Near Sushi", "Way Diner"}  # unnamed skipped
    way = next(r for r in results if r["name"] == "Way Diner")
    assert (way["lat"], way["lon"]) == (33.5, -96.5)  # center coords used
