"""Export the PostGIS layers used by the preprocessing step to GeoPackage files.

Run once on the computer that has the COAST database. The connection uses the
environment variables SQL_USER, SQL_PASSWORD and, optionally, SQL_HOST
(default localhost), SQL_PORT (default 5432) and SQL_DB (default COAST),
for example from the .env file. No password is written in the code.

Output: <data_dir>/raw/postgis/<table>.gpkg
"""
import logging
import os

import geopandas as gpd
from sqlalchemy import create_engine, text

from coastline import setup_logging
from coastline.config import load_config

log = logging.getLogger("export_postgis_layers")

QUERIES = {
    "world_countries_cyprus": "SELECT * FROM world_countries WHERE country = 'Cyprus'",
    "Coastline_Cyprus": 'SELECT * FROM "Coastline_Cyprus"',
    "Roads_cyprus": 'SELECT * FROM "Roads_cyprus" WHERE geom IS NOT NULL',
    "Drain_cyprus": 'SELECT * FROM "Drain_cyprus"',
    "Fires_2018_buffered_cyprus": 'SELECT * FROM "Fires_2018_buffered_cyprus"',
}


def main() -> None:
    cfg = load_config()
    url = "postgresql://{u}:{p}@{h}:{port}/{db}".format(
        u=os.environ["SQL_USER"], p=os.environ["SQL_PASSWORD"], h=os.environ.get("SQL_HOST", "localhost"),
        port=os.environ.get("SQL_PORT", "5432"), db=os.environ.get("SQL_DB", "COAST"))
    engine = create_engine(url)
    out_dir = cfg.data_dir / "raw" / "postgis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with engine.connect() as conn:
        for name, sql in QUERIES.items():
            gdf = gpd.read_postgis(text(sql), conn, geom_col="geom")
            gdf.to_file(out_dir / f"{name}.gpkg", driver="GPKG")
            log.info("%s: %d features", name, len(gdf))


if __name__ == "__main__":
    setup_logging()
    main()
