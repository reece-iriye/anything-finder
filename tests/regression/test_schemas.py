"""Schema stability guards.

These pin the public shape of the request/response/intent models so accidental
field renames or type changes (which would silently break the API contract or the
graph state) fail loudly. Update intentionally when the contract changes.
"""

from src.schemas.geo_search.intent import CravingIntent
from src.schemas.geo_search.request import GeoLocationRestaurantSearchRequest
from src.schemas.geo_search.response import GeoLocationRestaurantSearchResponse


def _fields(model) -> dict[str, object]:
    return {name: f.annotation for name, f in model.model_fields.items()}


def test_request_contract():
    fields = _fields(GeoLocationRestaurantSearchRequest)
    assert set(fields) == {
        "query",
        "session_id",
        "city",
        "state",
        "latitude",
        "longitude",
        "radius_m",
        "include_casual",
    }
    # query is the one required field; everything else has a default.
    required = {
        n
        for n, f in GeoLocationRestaurantSearchRequest.model_fields.items()
        if f.is_required()
    }
    assert required == {"query"}


def test_response_contract():
    assert set(_fields(GeoLocationRestaurantSearchResponse)) == {"response"}


def test_intent_contract():
    assert set(_fields(CravingIntent)) == {"craving", "vibe", "location_phrase"}
    # craving required; vibe defaults to empty list; location_phrase optional.
    intent = CravingIntent(craving="sushi")
    assert intent.vibe == []
    assert intent.location_phrase is None


def test_request_defaults_are_safe():
    req = GeoLocationRestaurantSearchRequest(query="sushi")
    assert req.include_casual is False
    assert req.latitude is None and req.longitude is None
