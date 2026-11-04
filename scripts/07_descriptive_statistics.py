"""Step 7 - Descriptive statistics of the Results section.

Input : results/risk/risk_index.csv, results/vulnerability/thresholds.csv, data/sites (attributes, raster values)
Output: results/descriptive/statistics.csv  (statistic, value, p_value, definition)

One definition per statistic, no variants: Spearman correlation on the index values at the
34 sites (HI, VI, EI as in risk_index.csv); Distance to Coast in metres; Land Use as the
1 to 3 score; UNESCO status as 0/1; VI without Distance to Coast from the 6 other scores.
"""
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from coastline import setup_logging
from coastline.config import load_config
from coastline.constants import SCORE_COLUMNS
from coastline.io import read_table
from coastline.vulnerability import score_attributes, vulnerability_from_scores

log = logging.getLogger("07_descriptive_statistics")


def main() -> None:
    cfg = load_config()
    risk = read_table(cfg.output("risk", "risk_index.csv"))
    scores, _ = score_attributes(read_table(cfg.input("site_attributes")), read_table(cfg.input("site_raster_values")),
                                 cfg["vulnerability"]["n_classes"])
    scores["vi6"] = vulnerability_from_scores(scores[[c for c in SCORE_COLUMNS if c != "s_dist_coast_m"]])[1]
    d = risk[["id", "hi", "vi_norm", "e_norm", "reviews", "r"]].merge(
        scores[["id", "dist_coast_m", "slope_deg", "s_land_use", "s_heritage", "vi6"]], on="id")
    d["unesco"] = (d["s_heritage"] == 1).astype(int)
    thr = pd.read_csv(cfg.output("vulnerability", "thresholds.csv")).set_index("variable")["threshold_1"]["dist_coast_m"]

    rows = []

    def spearman(name, x, y, definition):
        r, p = spearmanr(d[x], d[y])
        rows.append((name, r, p, f"Spearman, {definition}, n = {len(d)}"))

    def single(name, value, definition):
        rows.append((name, value, np.nan, definition))

    spearman("HI-VI", "hi", "vi_norm", "HI value and VI value")
    spearman("HI-EI", "hi", "e_norm", "HI value and EI value")
    spearman("VI-EI", "vi_norm", "e_norm", "VI value and EI value")
    spearman("Distance to Coast-HI", "dist_coast_m", "hi", "Distance to Coast (m) and HI value")
    spearman("Distance to Coast-VI", "dist_coast_m", "vi_norm", "Distance to Coast (m) and VI value")
    spearman("Distance to Coast-EI", "dist_coast_m", "e_norm", "Distance to Coast (m) and EI value")
    spearman("Slope-HI", "slope_deg", "hi", "Slope (degrees) and HI value")
    spearman("Land Use score-HI", "s_land_use", "hi", "Land Use score (1 to 3) and HI value")
    spearman("UNESCO status-EI", "unesco", "e_norm", "UNESCO status (0/1) and EI value")
    spearman("VI without Distance to Coast-VI", "vi6", "vi_norm", "VI value from the 6 other attributes and VI value")

    u = d["unesco"] == 1
    single("UNESCO share of the reviews (%)", 100 * d.loc[u, "reviews"].sum() / d["reviews"].sum(),
           f"{int(d.loc[u, 'reviews'].sum()):,} of {int(d['reviews'].sum()):,} reviews at the {int(u.sum())} UNESCO sites")
    for name, f in (("minimum", np.min), ("maximum", np.max), ("mean", np.mean), ("median", np.median),
                    ("standard deviation", lambda v: np.std(v, ddof=1))):
        single(f"VI {name}", f(d["vi_norm"]), "VI value (0 to 1) at the 34 sites" + (", n - 1" if "dev" in name else ""))
    near = d["dist_coast_m"] <= thr
    for label, m in ((f"within {thr / 1000:.1f} km", near), (f"beyond {thr / 1000:.1f} km", ~near)):
        for idx, col in (("HI", "hi"), ("VI", "vi_norm"), ("EI", "e_norm")):
            single(f"Median {idx} {label}", d.loc[m, col].median(),
                   f"{idx} value, {int(m.sum())} sites, cut at the first Multi-Otsu threshold of Distance to Coast ({thr:.1f} m)")
    counts = d["r"].value_counts()
    single("Distinct Risk values", d["r"].nunique(), "values of the Risk Index at the 34 sites")
    single("Sites sharing their Risk value", int(counts[counts > 1].sum()), "sites whose Risk value is shared with at least one other site")

    out = pd.DataFrame({
        "statistic": [r[0] for r in rows],
        "value": pd.Series([round(r[1], 4) if isinstance(r[1], float) else r[1] for r in rows], dtype=object),  # counts stay integers
        "p_value": [float(f"{r[2]:.3g}") if np.isfinite(r[2]) else np.nan for r in rows],
        "definition": [r[3] for r in rows]})
    out.to_csv(cfg.output("descriptive", "statistics.csv"), index=False)
    log.info("%d statistics written to %s", len(out), cfg.output("descriptive", "statistics.csv"))


if __name__ == "__main__":
    setup_logging()
    main()
