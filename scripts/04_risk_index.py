"""Step 4 - Risk Index (Section 2.6, Table 7, Figures 5 and 6).

Input : results/hazard/hazard_index.tif, results of steps 2 and 3, the site points
Output: results/risk/risk_index.csv (one row per site), risk_index.geojson, thresholds.csv
"""
import logging

import geopandas as gpd
import pandas as pd

from coastline import setup_logging
from coastline.config import load_config
from coastline.io import read_sites, read_table, site_buffer_max
from coastline.risk import risk_index

log = logging.getLogger("04_risk_index")


def main() -> None:
    cfg = load_config()
    r = cfg["risk"]
    sites = read_sites(cfg)

    # Hazard Index at each site: maximum in a 125 m buffer (doubled up to 1000 m if empty)
    hb = site_buffer_max(cfg.output("hazard", "hazard_index.tif"), sites, r["buffer_m"], r["buffer_max_m"])
    hazard = pd.DataFrame({"id": sites["id"], "name": sites["name"],
                           "hi": hb["value"].values, "hi_radius_m": hb["radius_m"].values})
    if (hazard["hi_radius_m"] > r["buffer_m"]).any():
        log.warning("buffer enlarged for: %s", hazard.loc[hazard.hi_radius_m > r["buffer_m"], "name"].tolist())

    table, thr = risk_index(hazard,
                            read_table(cfg.output("vulnerability", "vulnerability_index.csv")),
                            read_table(cfg.output("exposure", "exposure_index.csv")),
                            {int(k): v for k, v in r["vi_rescale"].items()})
    table.to_csv(cfg.output("risk", "risk_index.csv"), index=False, float_format="%.4f")
    gpd.GeoDataFrame(table, geometry=sites.geometry.values, crs=sites.crs).to_file(
        cfg.output("risk", "risk_index.geojson"), driver="GeoJSON")
    pd.DataFrame([(k, *v.round(4)) for k, v in thr.items()]).to_csv(
        cfg.output("risk", "thresholds.csv"), index=False, header=False)

    log.info("Risk classes Low/Medium/High: %s", table["r_class"].value_counts().sort_index().tolist())
    log.info("High Risk sites: %s", table.loc[table.r_class == 3].sort_values("r", ascending=False)["name"].tolist())


if __name__ == "__main__":
    setup_logging()
    main()
