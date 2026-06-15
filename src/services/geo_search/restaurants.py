import os
from typing import Self, Any
from uuid import UUID

from annotated_doc import Doc
import httpx
from langgraph.graph.state import CompiledStateGraph

from src.agents.geo_search.state import GeoSearchState


class RestaurantSearch:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        context_data: dict[str, Any],
        nominatim_client: httpx.AsyncClient,
        overpass_client: httpx.AsyncClient,
        geo_search_workflow: CompiledStateGraph,
    ):
        self.latitude: float = latitude
        self.longitude: float = longitude
        self.preferences: dict[str, Any] = context_data
        self.nominatim_client: httpx.AsyncClient = nominatim_client
        self.overpass_client: httpx.AsyncClient = overpass_client
        self.geo_search_workflow: CompiledStateGraph = geo_search_workflow

    @classmethod
    async def create(
        cls,
        *,
        latitude: float,
        longitude: float,
        user_id: str | UUID,
        nominatim_client: httpx.AsyncClient,
        overpass_client: httpx.AsyncClient,
        geo_search_workflow: CompiledStateGraph,
    ) -> Self:
        # TODO: Add user_id lookup logic to get relevant context data
        # from the user.
        #
        # user_location:
        val = cls._fetch_user_context_data(user_id)

        return cls(
            latitude=latitude,
            longitude=longitude,
            context_data=val,
            nominatim_client=nominatim_client,
            overpass_client=overpass_client,
            geo_search_workflow=geo_search_workflow,
        )

    @staticmethod
    async def _fetch_user_context_data(user_id: str | UUID) -> dict[str, Any]:
        # TODO: Implement this function
        raise NotImplementedError(f"_fetch_user_context_data() not implemented yet")

    async def ainvoke_search_workflow(
        self,
    ) -> tuple[str, str | None, int | None]:
        try:
            resp: GeoSearchState = await self.geo_search_workflow.ainvoke()
        except:
            resp: str = "Fake response"
        return resp, None, None
