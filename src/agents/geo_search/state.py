import operator
from typing import Annotated, Any, TypedDict

from src.utils.nominatim import Coordinates
from src.schemas.geo_search.intent import CravingIntent


class GeoSearchState(TypedDict, total=False):
    # NOTE: every field is persisted by the Postgres checkpointer, so values must be
    # serializable. Never put httpx clients or the LLM here. Those are closure-bound
    # into the node factories at graph-compile time.

    # ========= INPUT =========
    session_id: str | None
    raw_query: str | None
    location: Coordinates | None  # caller-supplied coordinates, if any
    radius_m: int
    include_casual: bool
    # ========= ACCUMULATED =========
    intent: CravingIntent | None
    resolved_location: Coordinates | None  # `location`, or geocoded from the query
    candidates: list[dict[str, Any]]
    ranked: list[dict[str, Any]]
    widened_geo_search: bool
    search_attempts: int
    # ======= OUTPUT ========
    response: str | None
    # Accumulate across nodes instead of overwriting, so every failure is captured.
    errors: Annotated[list[str], operator.add]
