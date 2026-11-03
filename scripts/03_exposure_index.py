"""Step 3 - Exposure Index (Section 2.5).

Input : data/sites/exposure_reviews.csv   review counts from Google Places (July 2026)
Output: results/exposure/exposure_index.csv, thresholds.csv

The review counts are collected with scripts/collect_google_reviews.py. They
change over time, so the article uses the archived counts of July 2026.
"""
import logging

import pandas as pd

from coastline import setup_logging
from coastline.config import load_config
from coastline.exposure import exposure_index
from coastline.io import read_table

log = logging.getLogger("03_exposure_index")


def main() -> None:
    cfg = load_config()
    table, thresholds = exposure_index(read_table(cfg.input("exposure_reviews")), cfg["exposure"]["n_classes"])
    table.to_csv(cfg.output("exposure", "exposure_index.csv"), index=False, float_format="%.4f")
    pd.DataFrame({"threshold": thresholds}).to_csv(cfg.output("exposure", "thresholds.csv"), index=False)
    log.info("Exposure classes 1-5: %s", table["e_class"].value_counts().sort_index().tolist())


if __name__ == "__main__":
    setup_logging()
    main()
