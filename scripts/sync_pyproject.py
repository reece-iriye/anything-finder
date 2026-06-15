import re
import pathlib


def _normalize_requirement(name: str, raw_ver: str) -> str:
    """Build a PEP 508 requirement string from a Pipfile name/value pair.

    Handles both the simple form (``ver = ">=1.0"``) and the inline-table form
    with extras (``{extras = ["binary", "pool"], version = ">=3.3.0"}``).
    """
    raw_ver = raw_ver.strip()
    if raw_ver.startswith("{"):
        extras_match = re.search(r"extras\s*=\s*\[([^\]]*)\]", raw_ver)
        version_match = re.search(r'version\s*=\s*"([^"]*)"', raw_ver)
        extras = ""
        if extras_match:
            items = [e.strip().strip('"').strip("'") for e in extras_match.group(1).split(",")]
            items = [e for e in items if e]
            if items:
                extras = "[" + ",".join(items) + "]"
        version = version_match.group(1) if version_match else "*"
        return f"{name}{extras}" if version == "*" else f"{name}{extras}{version}"
    ver = raw_ver.strip('"')
    return name if ver == "*" else f"{name}{ver}"


def parse_pipfile_packages(pipfile_path: str = "Pipfile") -> dict[str, str]:
    packages: dict[str, str] = {}
    in_packages = False
    for line in pathlib.Path(pipfile_path).read_text().splitlines():
        stripped = line.strip()
        if stripped == "[packages]":
            in_packages = True
            continue
        if stripped.startswith("["):
            in_packages = False
            continue
        if in_packages and "=" in stripped and not stripped.startswith("#"):
            name, _, ver = stripped.partition("=")
            packages[name.strip()] = ver.strip()
    return packages


def sync(pipfile_path: str = "Pipfile", pyproject_path: str = "pyproject.toml") -> None:
    packages = parse_pipfile_packages(pipfile_path)
    deps = [
        f'    "{_normalize_requirement(name, ver)}"'
        for name, ver in packages.items()
    ]
    pyproject = pathlib.Path(pyproject_path).read_text()
    # Anchor on the closing bracket at the start of a line so dependency specifiers
    # that themselves contain brackets (e.g. extras like psycopg[binary,pool]) don't
    # terminate the match early.
    updated = re.sub(
        r"dependencies = \[.*?\n\]",
        "dependencies = [\n" + ",\n".join(deps) + "\n]",
        pyproject,
        flags=re.DOTALL,
    )
    pathlib.Path(pyproject_path).write_text(updated)
    print(f"synced {len(packages)} deps → pyproject.toml")


if __name__ == "__main__":
    sync()
