from annotated_doc import Doc
from fastapi import APIRouter, Depends, HTTPException, status
import httpx
from langgraph.graph.state import CompiledStateGraph

import logging
from typing import Annotated

import src.schemas.geo_search
import src.services.geo_search
from src.agents.geo_search.graph import get_geo_graph
from src.utils.nominatim import (
    get_nominatim_client,
    forward_geocode_with_nominatim,
)
from src.utils.overpass import (
    get_overpass_client,
)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/geo-search/restaurants")


@router.get(
    path="{user_id}",
    response_class=src.schemas.geo_search.response.GeoLocationRestaurantSearchResponse,
)
async def get_human_readable_search_results_with_user_context(
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
    nominatim_client: Annotated[
        Annotated[httpx.AsyncClient, Depends(get_nominatim_client)],
        Doc(
            "Already instantiated client for sending requests to Nominatim "
            "service, which handles location resolution when no coordinates "
            "are passed in."
        ),
    ],
    overpass_client: Annotated[
        Annotated[httpx.AsyncClient, Depends(get_overpass_client)],
        Doc(
            "Already instantiated client for sending requests to Overpass "
            "service, which handles finding establishments and businesses "
            "based on a relative provided location."
        ),
    ],
    geo_search_workflow: Annotated[
        Annotated[CompiledStateGraph, Depends(get_geo_graph)],
        Doc("Compiled LangGraph StateGraph for restaurant search AI Workflow."),
    ],
):
    req_city: str | None = restaurant_search_request_payload.city
    req_state: str | None = restaurant_search_request_payload.state
    req_lat: float | None = restaurant_search_request_payload.latitude
    req_long: float | None = restaurant_search_request_payload.longitude

    if (req_city or req_state) and (req_lat or req_long):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="INVALID INPUT: city & state *xor* latitude & longitude must be supplied in request body. Bits of both combinations were provided.",
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
            detail="INVALID INPUT: city & state *xor* latitude & longitude must be supplied in request body. Either combination was NOT provided.",
        )

    search_client = await src.services.geo_search.RestaurantSearch.create(
        latitude=req_lat,
        longitude=req_long,
        user_id=user_id,
        nominatim_client=nominatim_client,
        overpass_client=overpass_client,
        geo_search_workflow=geo_search_workflow,
    )

    resp_str, err, err_code = await search_client.ainvoke_search_workflow()
    if err is not None:
        raise HTTPException(
            status_code=err_code,
            detail=f"INTERNAL SERVER ERROR: {err}",
        )

    return src.schemas.geo_search.GeoLocationRestaurantSearchResponse(
        response=resp_str,
        error=None,
    )
