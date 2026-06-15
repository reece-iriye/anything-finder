from annotated_doc import Doc
from pydantic import BaseModel

from typing import Annotated


class GeoLocationRestaurantSearchRequest(BaseModel):
    query: Annotated[
        str,
        Doc(
            "Natural-language description of what the user wants, e.g. "
            "'quiet sushi near Grandscape in The Colony, TX'. Craving, vibe, and "
            "any location phrase are extracted from this text."
        ),
    ]
    session_id: Annotated[
        str | None,
        Doc(
            "Conversation/session identifier used as the checkpointer thread id so "
            "multi-turn state is shared across replicas. Falls back to the user id."
        ),
    ] = None
    city: Annotated[
        str | None,
        Doc(
            "Optional location override. When city and state are provided, "
            "coordinates are derived from them. Cannot be combined with "
            "latitude/longitude."
        ),
    ] = None
    state: Annotated[
        str | None,
        Doc(
            "Optional location override. When city and state are provided, "
            "coordinates are derived from them. Cannot be combined with "
            "latitude/longitude."
        ),
    ] = None
    latitude: Annotated[
        float | None,
        Doc(
            "Optional explicit latitude. When latitude and longitude are provided "
            "they are used directly and geocoding is skipped. Cannot be combined "
            "with city/state."
        ),
    ] = None
    longitude: Annotated[
        float | None,
        Doc(
            "Optional explicit longitude. When latitude and longitude are provided "
            "they are used directly and geocoding is skipped. Cannot be combined "
            "with city/state."
        ),
    ] = None
    radius_m: Annotated[
        int | None,
        Doc("Optional starting search radius in meters; a default is applied if omitted."),
    ] = None
    include_casual: Annotated[
        bool,
        Doc(
            "Widen the search beyond sit-down restaurants to casual eateries "
            "(fast food, cafes, bars, pubs)."
        ),
    ] = False
