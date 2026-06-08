from typing import Self
from uuid import UUID

from annotated_doc import Doc
import httpx


class RestaurantSearch:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        preferences: list[str],
        nominatim_client: httpx.AsyncClient,
    ):
        self.latitude: float = latitude
        self.longitude: float = longitude
        self.preferences: list[str] = preferences
        self.nominatim_client: httpx.AsyncClient = nominatim_client

    @classmethod
    async def create(
        cls,
        latitude: float,
        longitude: float,
        user_id: str | UUID,
        nominatim_client: httpx.AsyncClient,
    ) -> Self:
        # TODO: Add user_id lookup logic to get relevant data.
        #
        # if preferences is list:
        #     prefs = preferences
        # elif preferences is str:
        #     prefs = [preferences]
        prefs = []

        return cls(
            latitude=latitude,
            longitude=longitude,
            preferences=prefs,
            nominatim_client=nominatim_client,
        )

    async def invoke_search_workflow(self) -> tuple[str, str | None, int | None]:
        return "Fake response", None, None
