#!/usr/bin/env python3
"""Download a Geofabrik OSM PBF extract and convert it to osm.bz2 for Overpass."""

import bz2
import http.server
import os
import socketserver
import sys
import urllib.request
from pathlib import Path

import osmium

PBF_URL = os.getenv(
    "OSM_PBF_URL",
    "https://download.geofabrik.de/north-america/us/texas-latest.osm.pbf",
)
DATA_DIR = Path(os.getenv("OSM_DATA_DIR", "/data"))


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


class _Converter(osmium.SimpleHandler):
    def __init__(self, out: Path) -> None:
        super().__init__()
        self._w = bz2.open(out, "wt", encoding="utf-8")
        self._w.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        self._w.write('<osm version="0.6" generator="PyOsmium">\n')

    def node(self, n) -> None:
        tags = "".join(f'<tag k="{t.k}" v="{t.v}"/>' for t in n.tags)
        self._w.write(
            f'  <node id="{n.id}" lat="{n.location.lat:.7f}"'
            f' lon="{n.location.lon:.7f}" version="{n.version}">\n'
        )
        if tags:
            self._w.write(f"    {tags}\n")
        self._w.write("  </node>\n")

    def way(self, w) -> None:
        refs = "".join(f'<nd ref="{nd.ref}"/>' for nd in w.nodes)
        tags = "".join(f'<tag k="{t.k}" v="{t.v}"/>' for t in w.tags)
        self._w.write(f'  <way id="{w.id}" version="{w.version}">\n')
        if refs:
            self._w.write(f"    {refs}\n")
        if tags:
            self._w.write(f"    {tags}\n")
        self._w.write("  </way>\n")

    def relation(self, r) -> None:
        members = "".join(
            f'<member type="{m.type}" ref="{m.ref}" role="{m.role}"/>'
            for m in r.members
        )
        tags = "".join(f'<tag k="{t.k}" v="{t.v}"/>' for t in r.tags)
        self._w.write(f'  <relation id="{r.id}" version="{r.version}">\n')
        if members:
            self._w.write(f"    {members}\n")
        if tags:
            self._w.write(f"    {tags}\n")
        self._w.write("  </relation>\n")

    def close(self) -> None:
        self._w.write("</osm>\n")
        self._w.close()


def convert(pbf: Path, bz2_out: Path) -> None:
    print("Converting PBF to osm.bz2...", flush=True)
    tmp = bz2_out.with_suffix(".tmp")
    c = _Converter(tmp)
    try:
        c.apply_file(str(pbf))
        c.close()
        tmp.rename(bz2_out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print("Conversion complete.", flush=True)


def serve(directory: Path, port: int = 8080) -> None:
    os.chdir(directory)
    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"OSM server listening on :{port}", flush=True)
        httpd.serve_forever()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pbf_name = PBF_URL.rsplit("/", 1)[-1]
    pbf = DATA_DIR / pbf_name
    bz2_out = DATA_DIR / pbf_name.replace(".osm.pbf", ".osm.bz2")

    if not pbf.exists():
        download(PBF_URL, pbf)
    else:
        print(f"{pbf.name} already cached.", flush=True)

    if not bz2_out.exists():
        convert(pbf, bz2_out)
    else:
        print(f"{bz2_out.name} already cached.", flush=True)

    serve(DATA_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal: {e}", flush=True)
        sys.exit(1)
