from fastapi import Request
import httpx

import os
from typing import Any

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


# A "casual eateries" search broadens beyond sit-down restaurants to everyday food/drink
# spots. Kept as a tuple so the amenity regex and any future filtering stay in sync.
_CASUAL_AMENITIES: tuple[str, ...] = ("restaurant", "fast_food", "cafe", "bar", "pub")


def _escape_overpass_regex(values: list[str]) -> str:
    """Join cuisine/category values into an Overpass case-insensitive regex alternation."""
    # Overpass tag regexes are POSIX ERE; escape characters with regex meaning.
    cleaned = [
        v.strip().replace("\\", "\\\\").replace("|", "\\|").replace(".", "\\.")
        for v in values
        if v and v.strip()
    ]
    return "|".join(cleaned)


def build_restaurant_query(
    *,
    lat: float,
    lon: float,
    radius_m: int,
    categories: str | list[str] | None = None,
    include_casual: bool = False,
) -> str:
    """Build Overpass QL selecting eateries within ``radius_m`` of (lat, lon).

    By default only ``amenity=restaurant`` is matched. When ``include_casual`` is
    True the search broadens to casual spots (fast food, cafes, bars, pubs) via an
    ``amenity`` regex. When ``categories`` is provided it additionally filters on the
    ``cuisine`` tag with a case-insensitive regex.
    """
    if isinstance(categories, str):
        categories = [categories]
    if include_casual:
        amenity_filter = f'["amenity"~"^({"|".join(_CASUAL_AMENITIES)})$"]'
    else:
        amenity_filter = '["amenity"="restaurant"]'
    cuisine_filter = ""
    if categories:
        pattern = _escape_overpass_regex(categories)
        if pattern:
            cuisine_filter = f'["cuisine"~"{pattern}",i]'
    around = f"(around:{radius_m},{lat},{lon})"
    # node + way + relation so we catch venues mapped as areas, not just points.
    # `out center tags` yields a single representative coordinate per element
    # (Overpass adds `center` to ways/relations), keeping normalization simple.
    return (
        "[out:json][timeout:25];"
        "("
        f"node{amenity_filter}{cuisine_filter}{around};"
        f"way{amenity_filter}{cuisine_filter}{around};"
        f"relation{amenity_filter}{cuisine_filter}{around};"
        ");"
        "out center tags;"
    )


async def query_restaurants(
    client: httpx.AsyncClient,
    *,
    lat: float,
    lon: float,
    radius_m: int,
    categories: str | list[str] | None = None,
    include_casual: bool = False,
) -> list[dict[str, Any]]:
    """Run an Overpass eatery search and normalize results.

    Returns a list of ``{name, amenity, lat, lon, tags}`` dicts (unnamed elements
    skipped). Set ``include_casual`` to widen beyond sit-down restaurants.
    """
    query = build_restaurant_query(
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        categories=categories,
        include_casual=include_casual,
    )
    response = await client.post("/api/interpreter", content=f"data={query}")
    response.raise_for_status()
    elements = response.json().get("elements", [])

    results: list[dict[str, Any]] = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        # ways/relations carry coords under "center"; nodes carry them inline.
        center = el.get("center", {})
        el_lat = el.get("lat", center.get("lat"))
        el_lon = el.get("lon", center.get("lon"))
        if el_lat is None or el_lon is None:
            continue
        results.append(
            {
                "name": name,
                "amenity": tags.get("amenity", "restaurant"),
                "lat": float(el_lat),
                "lon": float(el_lon),
                "tags": tags,
            }
        )
    return results


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
