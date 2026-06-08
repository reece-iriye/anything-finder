from annotated_doc import Doc
from pydantic import BaseModel

from typing import Annotated


class GeoLocationRestaurantSearchRequest(BaseModel):
    restaurant_categories: Annotated[
        str | list[str], Doc("Restaurant category/categories to search.")
    ]
    city: Annotated[
        str | None,
        Doc(
            "City where user is located or will be located."
            "When city and state are provided, latitude and longitude are "
            "derived from city and state, and request fails if both city "
            "and state *and* latitude and longitude are provided."
        ),
    ]
    state: Annotated[
        str | None,
        Doc(
            "State where user is located or will be located."
            "When city and state are provided, latitude and longitude are "
            "derived from city and state, and request fails if both city "
            "and state *and* latitude and longitude are provided."
        ),
    ]
    latitude: Annotated[
        float | None,
        Doc(
            "Latitude where user is located or will be located."
            "When latitude and longitude are provided, these are *directly* "
            "used for nearby restaurant lookups. Request fails if both city "
            "and state *and* latitude and longitude are provided."
        ),
    ]
    longitude: Annotated[
        float | None,
        Doc(
            "Longitude where user is located or will be located."
            "When latitude and longitude are provided, these are *directly* "
            "used for nearby restaurant lookups. Request fails if both city "
            "and state *and* latitude and longitude are provided."
        ),
    ]
