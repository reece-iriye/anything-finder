from annotated_doc import Doc
from pydantic import BaseModel

from typing import Annotated


class GeoLocationRestaurantSearchResponse(BaseModel):
    response: Annotated[
        str | None,
        Doc(
            "String response from agentic workflow. Returns hard-coded "
            "string if non-raised error occurs."
        ),
    ]
