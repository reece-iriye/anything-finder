from pydantic import BaseModel, Field


class CravingIntent(BaseModel):
    """Structured extraction of what the user wants right now."""

    craving: str = Field(description="The food/drink being craved, e.g. 'sushi'")
    vibe: list[str] = Field(
        default_factory=list,
        description="Atmosphere descriptors, e.g. ['quiet', 'date-night']",
    )
    location_phrase: str | None = Field(
        default=None,
        description="Location mentioned in utterance, e.g. 'near The Colony Grandscape Mall'",
    )
