"""Step 1 - Hazard Index (Section 2.3, Tables 5, Figure 3, Figure 4).

Input : the 8 normalized factors (config: inputs.factors)
Output: results/hazard/
    correlation.csv, redundant_pairs.csv, pca_eigenvalues.csv, pca_loadings.csv   pre-screening
    edge_weights.csv, conditional_probability.csv, weights.csv                    cascade weighting
    hazard_index.tif (continuous, [0, 1]) and hazard_class.tif (5 Multi-Otsu classes)
"""
import logging

import numpy as np
import pandas as pd
from skimage.filters import threshold_multiotsu

from coastline import setup_logging
from coastline.classify import apply_thresholds
from coastline.config import load_config
from coastline.hazard import cascade_weighting, hazard_index, pixel_sample, redundancy_screening, to_grid
from coastline.io import load_factors, valid_mask, write_raster

log = logging.getLogger("01_hazard_index")


def main() -> None:
    cfg = load_config()
    p = cfg["hazard"]
    codes = cfg.factor_codes

    arrays, profile = load_factors(cfg)
    valid = valid_mask(arrays)
    X = np.column_stack([arrays[c][valid] for c in codes])
    del arrays
    log.info("valid land pixels: %d", X.shape[0])

    # The same random pixel sample is used by the pre-screening and by the cascade weighting.
    idx = pixel_sample(X.shape[0], p["sample_fraction"], p["seed"])

    # 1. Pre-screening for redundancy (Pearson + PCA)
    s = redundancy_screening(X[idx], codes, p["redundancy_r"], p["kaiser_eigenvalue"], p["cumulative_variance"])
    s.correlation.to_csv(cfg.output("hazard", "correlation.csv"), float_format="%.4f")
    s.redundant_pairs.to_csv(cfg.output("hazard", "redundant_pairs.csv"), index=False)
    s.eigen.to_csv(cfg.output("hazard", "pca_eigenvalues.csv"), index=False, float_format="%.4f")
    s.loadings.to_csv(cfg.output("hazard", "pca_loadings.csv"), float_format="%.4f")
    log.info("redundant pairs (|r| > %.2f): %d; PCs retained: %d (Kaiser %d, cumulative %d)",
             p["redundancy_r"], len(s.redundant_pairs), s.n_retained, s.n_kaiser, s.n_cumulative)

    # 2. Cascade weighting
    cascade = cascade_weighting(X, idx, codes, p["cond_prob_percentile"])
    pd.DataFrame(cascade.edges, index=codes, columns=codes).to_csv(
        cfg.output("hazard", "edge_weights.csv"), float_format="%.4f")
    pd.DataFrame(cascade.cond_prob, index=codes, columns=codes).to_csv(
        cfg.output("hazard", "conditional_probability.csv"), float_format="%.4f")
    cascade.table().to_csv(cfg.output("hazard", "weights.csv"), index=False, float_format="%.6f")

    # 3. Hazard Index and its 5 classes over the whole island
    hi = hazard_index(cascade.propagated, cascade.weights)
    thresholds = threshold_multiotsu(hi, classes=p["n_classes"])
    write_raster(cfg.output("hazard", "hazard_index.tif"), to_grid(hi, valid), profile)
    write_raster(cfg.output("hazard", "hazard_class.tif"),
                 to_grid(apply_thresholds(hi, thresholds).astype(np.float32), valid), profile, nodata=0)
    pd.DataFrame({"threshold": thresholds}).to_csv(cfg.output("hazard", "hazard_class_thresholds.csv"), index=False)
    log.info("Hazard Index: mean %.3f, class thresholds %s", hi.mean(), np.round(thresholds, 3))


if __name__ == "__main__":
    setup_logging()
    main()
