import re
import pathlib


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
            packages[name.strip()] = ver.strip().strip('"')
    return packages


def sync(pipfile_path: str = "Pipfile", pyproject_path: str = "pyproject.toml") -> None:
    packages = parse_pipfile_packages(pipfile_path)
    deps = [
        f'    "{name}"' if ver == "*" else f'    "{name}{ver}"'
        for name, ver in packages.items()
    ]
    pyproject = pathlib.Path(pyproject_path).read_text()
    updated = re.sub(
        r"dependencies = \[.*?\]",
        "dependencies = [\n" + ",\n".join(deps) + "\n]",
        pyproject,
        flags=re.DOTALL,
    )
    pathlib.Path(pyproject_path).write_text(updated)
    print(f"synced {len(packages)} deps → pyproject.toml")


if __name__ == "__main__":
    sync()
