"""Build weather_core/data/basemap.json, the coastlines and borders under forecast maps.

Input: Natural Earth 1:50m (public domain) coastline and land boundary lines as GeoJSON, from
https://github.com/nvkelso/natural-earth-vector/tree/master/geojson:

    python3 scripts/make_basemap.py ne_50m_coastline.geojson ne_50m_admin_0_boundary_lines_land.geojson

Lines are clipped to the forecast region (weather_core.grid.EUROPE plus a margin) and stored as flat lists of
lat, lon in hundredths of a degree.
"""

import json
import sys
from pathlib import Path

SOUTH, WEST, NORTH, EAST = 30.0 - 3, -30.0 - 3, 72.0 + 3, 45.0 + 3
OUT = Path(__file__).parent.parent / "packages/weather_core/src/weather_core/data/basemap.json"


def inside(lon: float, lat: float) -> bool:
    return SOUTH <= lat <= NORTH and WEST <= lon <= EAST


def lines(path: str):
    for feature in json.load(open(path))["features"]:
        geometry = feature["geometry"]
        parts = geometry["coordinates"] if geometry["type"] == "MultiLineString" else [geometry["coordinates"]]
        for part in parts:
            run = []
            for lon, lat in part:
                if inside(lon, lat):
                    point = (round(lat * 100), round(lon * 100))
                    if not run or run[-1] != point:
                        run.append(point)
                elif run:
                    yield run
                    run = []
            if run:
                yield run


def flat(path: str) -> list[list[int]]:
    return [[x for point in run for x in point] for run in lines(path) if len(run) > 1]


def main(coastline: str, borders: str) -> None:
    OUT.parent.mkdir(exist_ok=True)
    data = {
        "source": "Made with Natural Earth (1:50m, public domain)",
        "bounds": [SOUTH, WEST, NORTH, EAST],
        "coastline": flat(coastline),
        "borders": flat(borders),
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
