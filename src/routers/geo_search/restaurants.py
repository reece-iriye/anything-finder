from annotated_doc import Doc
from fastapi import APIRouter, Depends, HTTPException, status
import httpx
from langgraph.graph.state import CompiledStateGraph

import logging
from typing import Annotated

import src.schemas.geo_search
import src.services.geo_search
from src.agents.geo_search.agent import get_restaurant_agent
from src.utils.nominatim import (
    Coordinates,
    get_nominatim_client,
    forward_geocode_with_nominatim,
)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/geo-search/restaurants")


@router.post(
    path="/{user_id}",
    response_model=src.schemas.geo_search.GeoLocationRestaurantSearchResponse,
)
async def search_restaurants_for_user(
    user_id: Annotated[
        str,
        Doc("UUID for knowledge-based filtering to provide stronger recommendations."),
    ],
    payload: Annotated[
        src.schemas.geo_search.GeoLocationRestaurantSearchRequest,
        Doc(
            "Natural-language restaurant search. Location is taken from the query "
            "text, or overridden by city/state or latitude/longitude."
        ),
    ],
    nominatim_client: Annotated[
        httpx.AsyncClient,
        Depends(get_nominatim_client),
    ],
    restaurant_agent: Annotated[
        CompiledStateGraph,
        Depends(get_restaurant_agent),
    ],
):
    city, state = payload.city, payload.state
    lat, lon = payload.latitude, payload.longitude

    has_place = bool(city or state)
    has_coord = lat is not None or lon is not None
    if has_place and has_coord:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="INVALID INPUT: provide city/state OR latitude/longitude, not both.",
        )
    if (lat is None) != (lon is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="INVALID INPUT: latitude and longitude must be provided together.",
        )

    # Resolve an optional coordinate override; otherwise the agent geocodes from the query.
    location: Coordinates | None = None
    if city and state:
        lat, lon = await forward_geocode_with_nominatim(
            city=city, state=state, client=nominatim_client
        )
    if lat is not None and lon is not None:
        location = Coordinates(lat=lat, lon=lon)

    search = await src.services.geo_search.RestaurantSearch.create(
        user_id=user_id,
        raw_query=payload.query,
        session_id=payload.session_id,
        location=location,
        radius_m=payload.radius_m,
        include_casual=payload.include_casual,
        restaurant_agent=restaurant_agent,
    )

    response, err, err_code = await search.ainvoke_search_workflow()
    if err is not None:
        raise HTTPException(
            status_code=err_code or status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"INTERNAL SERVER ERROR: {err}",
        )

    return src.schemas.geo_search.GeoLocationRestaurantSearchResponse(
        response=response,
    )
