from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_prompt(directory: Path | str, name: str, **fmt: object) -> str:
    """Load a markdown prompt ``<directory>/<name>.md``.

    Prompts live in markdown files (never inline in source). Pass keyword args to
    interpolate ``str.format`` placeholders in the template. Raw file reads are cached.
    """
    text = _read(str(Path(directory) / f"{name}.md"))
    return text.format(**fmt) if fmt else text
