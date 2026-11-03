"""Step 0 - Build the 8 normalized Hazard factors from the raw datasets (Section 2.3).

Input : raw datasets listed under `inputs` in config/default.yaml (see data/README.md)
Output: data/factors/<CODE>.tif on the common 30 m grid (EPSG:32636)

This step is NOT needed to reproduce the article: the 8 factors used in the
article are archived as underlying data. Run it to rebuild the factors from
newer source data or for another region.

Usage: python scripts/00_preprocess_factors.py [--factors SAI BUI ...]
"""
import argparse
import logging

from coastline import setup_logging
from coastline.config import load_config
from coastline.preprocess import BUILDERS, boundary, build_factor, land_mask, reference_grid

log = logging.getLogger("00_preprocess_factors")


def main(codes: list[str]) -> None:
    cfg = load_config()
    grid = reference_grid(cfg)
    mask = land_mask(boundary(cfg, cfg.crs).geometry, grid["transform"], (grid["height"], grid["width"]))
    log.info("common grid %d x %d, %d land pixels", grid["width"], grid["height"], int(mask.sum()))
    for code in codes:
        build_factor(cfg, code, grid, mask, cfg.input("factors", code))


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--factors", nargs="+", choices=list(BUILDERS), default=list(BUILDERS))
    main(parser.parse_args().factors)
