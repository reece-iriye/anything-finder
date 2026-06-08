import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from annotated_doc import Doc

import logging
from typing import Annotated

import src.schemas.geo_search
import src.services.geo_search
from src.utils.location_utils import (
    get_nominatim_client,
    forward_geocode_with_nominatim,
    reverse_geocode_with_nominatim,
)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/geo-search/restaurants")


@router.get(
    path="{user_id}",
    response_class=src.schemas.geo_search.response.GeoLocationRestaurantSearchResponse,
)
async def get_human_readable_search_results(
    user_id: Annotated[
        str,
        Doc("UUID for knowledge-based filtering to provide stronger recommendations."),
    ],
    restaurant_search_request_payload: Annotated[
        src.schemas.geo_search.request.GeoLocationRestaurantSearchRequest,
        Doc(
            "Geolocation data for searching restaurants by city/state."
            "Data is translated to coordinates for bounding-box search to be conducted"
            "as proximate the the coordinates as possible."
        ),
    ],
    nominatim_client: Annotated[httpx.AsyncClient, Depends(get_nominatim_client)],
):
    req_city: str | None = restaurant_search_request_payload.city
    req_state: str | None = restaurant_search_request_payload.state
    req_lat: float | None = restaurant_search_request_payload.latitude
    req_long: float | None = restaurant_search_request_payload.longitude

    if (req_city or req_state) and (req_lat or req_long):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="INVALID INPUT: city & state *xor* latitude & longitude must be supplied in request body.",
        )

    if req_city and req_state:
        req_lat, req_long = await forward_geocode_with_nominatim(
            city=req_city,
            state=req_state,
            client=nominatim_client,
        )

    if not req_lat or not req_long:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="INVALID INPUT: city & state *xor* latitude & longitude must be supplied in request body.",
        )

    search_client = await src.services.geo_search.RestaurantSearch.create(
        latitude=req_lat,
        longitude=req_long,
        user_id=user_id,
        nominatim_client=nominatim_client,
    )

    resp_str, err, err_code = await search_client.invoke_search_workflow()
    if err is not None:
        raise HTTPException(
            status_code=err_code,
            detail=f"INTERNAL SERVER ERROR: {err}",
        )

    return src.schemas.geo_search.GeoLocationRestaurantSearchResponse(
        response=resp_str,
        error=None,
    )
