#!/usr/bin/env python3
"""Serve pre-prepared OSM data files over HTTP for Nominatim and Overpass.

Run `make osm-convert` once before starting docker compose to produce
data/Dallas.osm.bz2 from data/Dallas.osm.gz.
"""

import http.server
import os
import socketserver
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(os.getenv("OSM_DATA_DIR", "/data"))
_LOCAL_PBF = os.getenv("OSM_PBF_FILENAME")
_LOCAL_GZ  = os.getenv("OSM_GZ_FILENAME")
_PBF_URL   = os.getenv("OSM_PBF_URL", "")


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url}...", flush=True)
    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"{dest.name} downloaded.", flush=True)


def serve(directory: Path, port: int = 8080) -> None:
    os.chdir(directory)
    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"OSM server listening on :{port}", flush=True)
        httpd.serve_forever()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resolve PBF ────────────────────────────────────────────────────────────
    if _LOCAL_PBF:
        pbf = DATA_DIR / _LOCAL_PBF
        if not pbf.exists():
            print(f"Fatal: {pbf} not found — place the file there and restart.", flush=True)
            sys.exit(1)
        print(f"{pbf.name} found.", flush=True)
    elif _PBF_URL:
        pbf_name = _PBF_URL.rsplit("/", 1)[-1]
        pbf = DATA_DIR / pbf_name
        if not pbf.exists():
            download(_PBF_URL, pbf)
        else:
            print(f"{pbf.name} already cached.", flush=True)
    else:
        print("Fatal: set OSM_PBF_FILENAME or OSM_PBF_URL", flush=True)
        sys.exit(1)

    # ── Check bz2 ──────────────────────────────────────────────────────────────
    bz2_out = DATA_DIR / pbf.name.replace(".osm.pbf", ".osm.bz2")
    if not bz2_out.exists():
        print(
            f"Fatal: {bz2_out.name} not found — run `make osm-convert` first.",
            flush=True,
        )
        sys.exit(1)
    print(f"{bz2_out.name} found.", flush=True)

    serve(DATA_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal: {e}", flush=True)
        sys.exit(1)
