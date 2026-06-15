from typing import Any, Self
from uuid import UUID

from langgraph.graph.state import CompiledStateGraph

from src.agents.geo_search.state import GeoSearchState
from src.utils.nominatim import Coordinates


class RestaurantSearch:
    """Per-request wrapper that builds graph input and invokes the workflow.

    The compiled graph already has its LLMs and HTTP clients closure-bound, so this
    service only carries the per-request inputs and the resolved user context.
    """

    def __init__(
        self,
        *,
        user_id: str | UUID,
        raw_query: str,
        session_id: str | None,
        location: Coordinates | None,
        radius_m: int | None,
        include_casual: bool,
        context_data: dict[str, Any],
        geo_search_workflow: CompiledStateGraph,
    ):
        self.user_id = user_id
        self.raw_query = raw_query
        self.session_id = session_id
        self.location = location
        self.radius_m = radius_m
        self.include_casual = include_casual
        self.preferences = context_data
        self.geo_search_workflow = geo_search_workflow

    @classmethod
    async def create(
        cls,
        *,
        user_id: str | UUID,
        raw_query: str,
        geo_search_workflow: CompiledStateGraph,
        session_id: str | None = None,
        location: Coordinates | None = None,
        radius_m: int | None = None,
        include_casual: bool = False,
    ) -> Self:
        context_data = await cls._fetch_user_context_data(user_id)
        return cls(
            user_id=user_id,
            raw_query=raw_query,
            session_id=session_id,
            location=location,
            radius_m=radius_m,
            include_casual=include_casual,
            context_data=context_data,
            geo_search_workflow=geo_search_workflow,
        )

    @staticmethod
    async def _fetch_user_context_data(user_id: str | UUID) -> dict[str, Any]:
        # TODO: look up stored user preferences for knowledge-based filtering.
        # Returns empty context for now so the workflow runs end-to-end.
        return {}

    async def ainvoke_search_workflow(self) -> tuple[str | None, str | None, int | None]:
        """Run the graph. Returns (response, error_detail, error_status_code)."""
        initial: GeoSearchState = {
            "session_id": self.session_id,
            "raw_query": self.raw_query,
            "location": self.location,
            "include_casual": self.include_casual,
            "errors": [],
        }
        if self.radius_m is not None:
            initial["radius_m"] = self.radius_m

        # thread_id keys the checkpointer; a session shares state, else fall back to user.
        thread_id = self.session_id or str(self.user_id)
        result: GeoSearchState = await self.geo_search_workflow.ainvoke(
            initial,
            config={"configurable": {"thread_id": thread_id}},
        )

        response = result.get("response")
        errors = result.get("errors") or []
        if response is None and errors:
            return None, "; ".join(errors), 500
        return response, None, None
