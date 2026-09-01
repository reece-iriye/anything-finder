from typing import Self
from uuid import UUID

from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from src.utils.nominatim import Coordinates


class RestaurantSearch:
    """Per-request wrapper that builds the agent input and invokes the workflow.

    The compiled agent already has its LLM, HTTP clients, and config closure-bound,
    so this service only carries per-request inputs.
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
        restaurant_agent: CompiledStateGraph,
    ):
        self.user_id = user_id
        self.raw_query = raw_query
        self.session_id = session_id
        self.location = location
        self.radius_m = radius_m
        self.include_casual = include_casual
        self.restaurant_agent = restaurant_agent

    @classmethod
    async def create(
        cls,
        *,
        user_id: str | UUID,
        raw_query: str,
        restaurant_agent: CompiledStateGraph,
        session_id: str | None = None,
        location: Coordinates | None = None,
        radius_m: int | None = None,
        include_casual: bool = False,
    ) -> Self:
        return cls(
            user_id=user_id,
            raw_query=raw_query,
            session_id=session_id,
            location=location,
            radius_m=radius_m,
            include_casual=include_casual,
            restaurant_agent=restaurant_agent,
        )

    async def ainvoke_search_workflow(self) -> tuple[str | None, str | None, int | None]:
        """Run the agent. Returns (response, error_detail, error_status_code)."""
        parts = [self.raw_query]
        if self.location:
            parts.append(
                f"My coordinates: lat={self.location.lat}, lon={self.location.lon}"
            )
        if self.radius_m is not None:
            parts.append(f"Preferred search radius: {self.radius_m}m")
        if self.include_casual:
            parts.append("Include casual spots (cafes, fast food, bars).")
        human_msg = "\n".join(parts)

        thread_id = self.session_id or str(self.user_id)
        config = {
            "configurable": {"thread_id": thread_id, "user_id": str(self.user_id)}
        }
        try:
            result = await self.restaurant_agent.ainvoke(
                {"messages": [("human", human_msg)]},
                config=config,
            )
        except Exception as exc:
            return None, str(exc), 500

        messages = result.get("messages", [])
        response = next(
            (
                m.content
                for m in reversed(messages)
                if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
            ),
            None,
        )
        if response is None:
            return None, "Agent returned no response", 500
        return response, None, None
