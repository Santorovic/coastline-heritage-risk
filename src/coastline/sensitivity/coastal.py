"""Coastal inputs (Section 2.7): Risk without CPI, Distance to Coast and Elevation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..constants import SCORE_COLUMNS
from ..hazard import cascade_weighting, hazard_index, pixel_sample, to_grid
from ..io import site_buffer_max
from ..risk import rescale_vulnerability, risk_from_hazard
from ..vulnerability import vulnerability_from_scores


def coastal_tests(cfg, X: np.ndarray, codes: list[str], valid: np.ndarray, profile: dict, sites,
                  main: pd.DataFrame, scores: pd.DataFrame,
                  drop_factor: str = "CPI") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three tests: (A) Hazard without CPI; (B) A + Vulnerability without Distance to
    Coast; (C) B + Vulnerability without Elevation.

    ``X`` is the matrix of the valid pixels (columns in the order of ``codes``).
    The cascade weighting is repeated with the remaining factors on the same
    valid pixels and the same pixel sample rule.
    Returns (one row per site, summary).
    """
    h, r = cfg["hazard"], cfg["risk"]
    keep = [i for i, c in enumerate(codes) if c != drop_factor]
    kept_codes = [codes[i] for i in keep]
    X7 = X[:, keep]
    idx = pixel_sample(X7.shape[0], h["sample_fraction"], h["seed"])
    cascade = cascade_weighting(X7, idx, kept_codes, h["cond_prob_percentile"])
    hi = to_grid(hazard_index(cascade.propagated, cascade.weights), valid)
    weights = dict(zip(kept_codes, np.round(cascade.weights, 3).tolist()))
    del X7, cascade
    hi_site = site_buffer_max(hi, sites, r["buffer_m"], r["buffer_max_m"], profile)["value"].values

    def vi_without(drop):
        _, _, cls, _ = vulnerability_from_scores(scores[[c for c in SCORE_COLUMNS if c not in drop]])
        return rescale_vulnerability(cls)

    e = main["e_class"].values
    scenarios = {
        "main": (main["hi"].values, main["vi_resc"].values),
        "no_cpi": (hi_site, main["vi_resc"].values),
        "no_cpi_dtc": (hi_site, vi_without(["s_dist_coast_m"])),
        "no_cpi_dtc_elevation": (hi_site, vi_without(["s_dist_coast_m", "s_elevation_m"])),
    }
    out = main[["id", "name"]].copy()
    summary = []
    for key, (hs, vr) in scenarios.items():
        hc, rv, rc, _, _ = risk_from_hazard(hs, vr, e)
        out[f"hi_{key}"], out[f"hi_class_{key}"], out[f"r_{key}"], out[f"r_class_{key}"] = hs, hc, rv, rc
        summary.append({
            "test": key,
            "rho_hi": spearmanr(hs, main["hi"])[0],
            "hi_class_changes": int((hc != main["hi_class"].values).sum()),
            "rho_r": spearmanr(rv, main["r"])[0],
            "r_class_changes": int((rc != main["r_class"].values).sum()),
            "n_low_medium_high": np.bincount(rc, minlength=4)[1:].tolist(),
            "high_sites": ";".join(out.loc[rc == 3, "name"]),
        })
    summary = pd.DataFrame(summary)
    summary["weights_without_cpi"] = str(weights)
    return out, summary
