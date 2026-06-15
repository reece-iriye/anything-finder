from typing import TypedDict

from src.utils.nominatim import Coordinates
from src.schemas.geo_search.intent import CravingIntent


class GeoSearchState(TypedDict, total=False):
    # ========= INPUT =========
    session_id: str | None
    raw_query: str | None
    location: Coordinates | None
    radius_m: int
    # ========= ACCUMULATED =========
    intent: CravingIntent | None
    widened_geo_search: bool
    # ======= OUTPUT ========
    response: str | None
    errors: list[str]
