"""Collect the Google Places review counts of the sites (Exposure Index, Section 2.5).

For each part of each site polygon, the Places API (legacy Nearby Search) is
queried at the centroid of the part, with a radius equal to the distance from
the centroid to the farthest vertex (minimum 50 m), once for each of 10
categories. Places are deduplicated by place_id across parts and categories;
the reviews of a site are the sum of ``user_ratings_total`` of its places.
Only the first page of results (at most 20 places) is read for each query.

The API key is read from the environment variable GOOGLE_API_KEY (for example
from the .env file); it is never written in the code.

The counts change over time: the article uses the counts collected in July 2026
(data/sites/exposure_reviews.csv). Running this script again gives new counts
and writes them to results/exposure/exposure_reviews_<date>.csv. The Google
Maps Platform terms limit storing and sharing Places content: check them before
publishing new counts.

Requires: pip install googlemaps
Usage   : python scripts/collect_google_reviews.py
"""
import datetime as dt
import logging
import os
import time

import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon

from coastline import setup_logging
from coastline.config import load_config
from coastline.io import read_sites

log = logging.getLogger("collect_google_reviews")

PLACE_TYPES = ["tourist_attraction", "museum", "church", "cemetery", "place_of_worship",
               "art_gallery", "natural_feature", "park", "point_of_interest", "establishment"]
MIN_RADIUS_M = 50
DELAY_S = 0.25


def parts(geometry):
    """Polygon parts of a geometry (a point or line gives its convex hull)."""
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, Polygon):
        return [geometry]
    return [geometry.convex_hull]


def circumscribed_radius_m(polygon_utm) -> float:
    """Largest distance (m) from the centroid to a vertex of the exterior ring."""
    c = polygon_utm.centroid
    return max(c.distance(Point(x, y)) for x, y in polygon_utm.exterior.coords)


def query_part(client, lat: float, lon: float, radius_m: float) -> dict:
    """Unique places around one point, all categories. Returns {place_id: info}."""
    found = {}
    for place_type in PLACE_TYPES:
        try:
            resp = client.places_nearby(location=(lat, lon), radius=int(max(MIN_RADIUS_M, radius_m)),
                                        type=place_type)
        except Exception as exc:  # network or quota error: log and continue with the next category
            log.warning("query failed (%s, %s): %s", place_type, (lat, lon), exc)
            resp = {}
        time.sleep(DELAY_S)
        if resp.get("status") != "OK":
            continue
        for place in resp.get("results", []):
            pid = place.get("place_id")
            if pid and pid not in found:
                found[pid] = {"name": place.get("name", ""), "type": place_type,
                              "reviews": place.get("user_ratings_total", 0) or 0}
    return found


def main() -> None:
    import googlemaps  # optional dependency, only for this script

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set the environment variable GOOGLE_API_KEY (see .env.example).")
    client = googlemaps.Client(key=key)

    cfg = load_config()
    polys_utm = read_sites(cfg, "polygons")
    polys_wgs = polys_utm.to_crs("EPSG:4326")
    rows = []
    for (_, utm), (_, wgs) in zip(polys_utm.iterrows(), polys_wgs.iterrows()):
        places = {}
        for part_wgs, part_utm in zip(parts(wgs.geometry), parts(utm.geometry)):
            c = part_wgs.centroid
            for pid, info in query_part(client, c.y, c.x, circumscribed_radius_m(part_utm)).items():
                places.setdefault(pid, info)
        rows.append({"id": utm["id"], "name": utm["name"], "reviews": sum(p["reviews"] for p in places.values()),
                     "n_places": len(places)})
        log.info("%s: %d places, %d reviews", utm["name"], rows[-1]["n_places"], rows[-1]["reviews"])

    out = cfg.output("exposure", f"exposure_reviews_{dt.date.today():%Y%m%d}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("written %s", out)


if __name__ == "__main__":
    setup_logging()
    main()
