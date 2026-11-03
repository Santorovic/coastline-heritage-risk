"""Step 5 - Sensitivity analyses (Section 2.7, results in Section 3.5).

Needs the outputs of steps 1-4. Each part can be run alone with --parts.

    vulnerability  site polygon support, without Enclosure and Shelter, 3 sets of fixed
                   thresholds                        -> vulnerability_tests*.csv
    aggregation    product of the continuous indices, mean rank -> aggregation*.csv
    pixels         OAT of the weights at pixel level (Table 8) -> oat_pixel_*.csv
    coastal        without CPI, Distance to Coast, Elevation  -> coastal_tests*.csv
    stability      share of OAT scenarios with a class change per pixel (Figure 7)
                                                     -> oat_stability.tif, oat_stability_summary.csv
    sites          OAT of the weights at the 34 sites (Figure 8) -> oat_sites_*.csv

Usage
    python scripts/05_sensitivity.py                      # every part
    python scripts/05_sensitivity.py --parts pixels sites # some parts
    python scripts/05_sensitivity.py --skip-polygon       # without DEM, slope and WorldCover
"""
import argparse
import logging

import numpy as np
import pandas as pd

from coastline import setup_logging
from coastline.config import load_config
from coastline.constants import SCORE_COLUMNS
from coastline.hazard import cascade_weighting, pixel_sample, to_grid
from coastline.io import load_factors, read_sites, read_table, site_pixel_index, valid_mask, write_raster
from coastline.sensitivity import aggregation, coastal, vulnerability as vt, weights as wt

log = logging.getLogger("05_sensitivity")
PARTS = ["vulnerability", "aggregation", "pixels", "coastal", "stability", "sites"]


class Run:
    """Inputs shared by the parts, loaded only when a part needs them."""

    def __init__(self):
        self.cfg = load_config()
        self.sites = read_sites(self.cfg)
        self.risk = read_table(self.cfg.output("risk", "risk_index.csv"))
        self.vi = read_table(self.cfg.output("vulnerability", "vulnerability_index.csv"))
        self._factors = None

    def out(self, name):
        return self.cfg.output("sensitivity", name)

    def factors(self):
        """(arrays, profile, valid) of the 8 factors."""
        if self._factors is None:
            arrays, profile = load_factors(self.cfg)
            self._factors = (arrays, profile, valid_mask(arrays))
        return self._factors

    def pixel_matrix(self):
        arrays, profile, valid = self.factors()
        X = np.column_stack([arrays[c][valid] for c in self.cfg.factor_codes])
        return X, profile, valid

    def cascade(self):
        X, profile, valid = self.pixel_matrix()
        self._factors = None  # free memory: the full rasters are not needed any more
        h = self.cfg["hazard"]
        c = cascade_weighting(X, pixel_sample(X.shape[0], h["sample_fraction"], h["seed"]),
                              self.cfg.factor_codes, h["cond_prob_percentile"])
        return c, profile, valid


def run_vulnerability(run: Run, skip_polygon: bool) -> None:
    s = run.cfg["sensitivity"]
    tests = {}
    if not skip_polygon:
        tests["site_polygon"] = vt.polygon_support(run.cfg, run.vi, run.sites, read_sites(run.cfg, "polygons"))
    tests["no_enclosure_shelter"] = vt.without_attributes(run.vi[SCORE_COLUMNS], ["s_enclosure", "s_shelter"])
    for key, (elev, dist) in s["fixed_thresholds"].items():
        tests[f"fixed_thresholds_{key[-1]}"] = vt.fixed_thresholds(run.vi, tuple(elev), tuple(dist))
    rows, classes = [], {"id": run.risk["id"], "name": run.risk["name"], "main": run.risk["r_class"]}
    for key, scores in tests.items():
        row, classes[key] = vt.compare(run.risk, scores, key)
        rows.append(row)
    pd.DataFrame(rows).to_csv(run.out("vulnerability_tests.csv"), index=False, float_format="%.3f")
    pd.DataFrame(classes).to_csv(run.out("vulnerability_tests_sites.csv"), index=False)


def run_aggregation(run: Run) -> None:
    per_site, summary = aggregation.alternative_aggregations(run.risk)
    per_site.to_csv(run.out("aggregation_sites.csv"), index=False, float_format="%.4f")
    summary.to_csv(run.out("aggregation.csv"), index=False, float_format="%.3f")


def run_pixels(run: Run) -> None:
    cfg, s, h = run.cfg, run.cfg["sensitivity"], run.cfg["hazard"]
    arrays, _, valid = run.factors()
    # The weights of Table 5 (3 decimals) are perturbed, as in the article.
    table5 = pd.read_csv(cfg.output("hazard", "weights.csv")).set_index("factor")["weight_rounded"]
    scenarios, table8 = wt.oat_pixels(arrays, valid, {c: float(table5[c]) for c in cfg.factor_codes},
                                      h["n_classes"], s["oat_range"], s["oat_step"],
                                      s["oat_pixel_sample_fraction"], h["seed"])
    scenarios.to_csv(run.out("oat_pixel_scenarios.csv"), index=False, float_format="%.6f")
    table8.to_csv(run.out("oat_pixel_table8.csv"), index=False, float_format="%.4f")


def run_coastal(run: Run) -> None:
    X, profile, valid = run.pixel_matrix()
    per_site, summary = coastal.coastal_tests(run.cfg, X, run.cfg.factor_codes, valid, profile, run.sites,
                                              run.risk, run.vi[SCORE_COLUMNS])
    per_site.to_csv(run.out("coastal_tests_sites.csv"), index=False, float_format="%.4f")
    summary.to_csv(run.out("coastal_tests.csv"), index=False, float_format="%.3f")


def run_stability(run: Run) -> None:
    s, h = run.cfg["sensitivity"], run.cfg["hazard"]
    cascade, profile, valid = run.cascade()
    count, n, _ = wt.oat_stability(cascade.propagated, cascade.weights, run.cfg.factor_codes,
                                   h["n_classes"], s["oat_range"], s["oat_step"])
    write_raster(run.out("oat_stability.tif"), to_grid(100 * count / n, valid), profile)
    wt.stability_summary(count, n).to_csv(run.out("oat_stability_summary.csv"), index=False, float_format="%.2f")


def run_sites(run: Run) -> None:
    r, s = run.cfg["risk"], run.cfg["sensitivity"]
    cascade, profile, valid = run.cascade()
    site_pos = site_pixel_index(valid, profile, run.sites, r["buffer_m"], r["buffer_max_m"])
    scenarios, per_site = wt.oat_sites(cascade.propagated, cascade.weights, run.cfg.factor_codes, site_pos,
                                       run.risk[["id", "name", "vi_resc", "e_class"]],
                                       tuple(s["oat_site_deltas"]))
    scenarios.to_csv(run.out("oat_sites_scenarios.csv"), index=False, float_format="%.4f")
    per_site.to_csv(run.out("oat_sites_per_site.csv"), index=False, float_format="%.4f")


def main(parts: list[str], skip_polygon: bool) -> None:
    run = Run()
    for part in parts:
        log.info("--- %s", part)
        if part == "vulnerability":
            run_vulnerability(run, skip_polygon)
        else:
            globals()[f"run_{part}"](run)
    log.info("written to %s", run.out("x").parent)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=PARTS)
    parser.add_argument("--skip-polygon", action="store_true", help="skip the site polygon test (needs rasters)")
    args = parser.parse_args()
    main(args.parts, args.skip_polygon)
