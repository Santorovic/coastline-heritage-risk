"""Weights of the Hazard Index: one-at-a-time (OAT) perturbation (Eqs. 22-28).

Three readings of the same perturbation, as in the article:
* :func:`oat_pixels`    pixel level, Table 8 (weighted sum of the normalized factors, Eq. 24)
* :func:`oat_stability` pixel level, Figure 7 (share of scenarios with a class change)
* :func:`oat_sites`     the 34 sites, Hazard and Risk classes (Figure 8, Figure 9)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from skimage.filters import threshold_multiotsu

from ..classify import apply_thresholds, multiotsu_classes
from ..hazard import minmax, pixel_sample

log = logging.getLogger(__name__)


def perturbed_weights(weights: np.ndarray, j: int, delta: float) -> np.ndarray:
    """Weight j changed by ``delta`` (e.g. -0.20); the other weights rescaled
    in proportion so that the sum stays 1 (Eqs. 22-23)."""
    w = np.asarray(weights, dtype=float).copy()
    wj = w[j] * (1 + delta)
    others = np.arange(len(w)) != j
    w[others] = w[others] * (1 - wj) / (1 - w[j])
    w[j] = wj
    return w


def oat_steps(max_delta: float = 0.20, step: float = 0.01) -> np.ndarray:
    """-20 %, -19 %, ... +20 % (41 steps, including 0)."""
    n = int(round(max_delta / step))
    return np.round(np.arange(-n, n + 1) * step, 2)


# ============================================================ pixel level
def oat_pixels(arrays: dict, valid: np.ndarray, weights: dict, n_classes: int = 5,
               max_delta: float = 0.20, step: float = 0.01, sample_fraction: float = 0.15,
               seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OAT at pixel level on the weighted sum of the normalized factors (Eq. 24).

    ``weights`` are the cascade weights as reported in Table 5 (3 decimals).
    The class thresholds of the base surface (Multi-Otsu) stay fixed.
    Change of the mean, share of pixels changing class and Pearson correlation
    are computed on a random pixel sample, as in the article.

    Returns (one row per scenario, Table 8).
    """
    codes = list(weights)
    w0 = np.array([weights[c] for c in codes])
    rows, cols = np.where(valid)
    idx = pixel_sample(rows.size, sample_fraction, seed)
    sample = {c: np.nan_to_num(arrays[c], nan=0.0)[rows[idx], cols[idx]] for c in codes}

    h_base = np.zeros(valid.shape, dtype=np.float32)
    for c, w in zip(codes, w0):
        h_base += w * np.nan_to_num(arrays[c], nan=0.0)
    h_base[~valid] = np.nan
    base_mean = float(np.nanmean(h_base))
    thresholds = threshold_multiotsu(h_base[valid], classes=n_classes)
    base_sample = h_base[rows[idx], cols[idx]]
    base_class = apply_thresholds(base_sample, thresholds)

    records = []
    for j, code in enumerate(codes):
        for delta in oat_steps(max_delta, step):
            w = perturbed_weights(w0, j, delta)
            h = np.zeros_like(base_sample)
            for c, wc in zip(codes, w):
                h = h + wc * sample[c]
            d_mean = float(np.mean(h) - np.mean(base_sample))
            pct_out = 100 * d_mean / base_mean
            records.append({
                "factor": code, "delta": delta, "weight": w[j],
                "delta_mean_hi": d_mean,
                "pct_pixels_class_change": 100 * float(np.mean(apply_thresholds(h, thresholds) != base_class)),
                "pearson_r": pearsonr(base_sample, h)[0],
                "si": pct_out / (100 * delta) if delta != 0 else np.nan,
            })
    scenarios = pd.DataFrame(records)

    # Table 8 (the closed-form sensitivity index, Eq. 26, equals the mean of the SI)
    means = {c: float(np.nanmean(np.where(valid, arrays[c], np.nan))) for c in codes}
    table = []
    for j, code in enumerate(codes):
        others = [c for c in codes if c != code]
        w_others = np.array([weights[c] for c in others]) / (1 - weights[code])
        mean_others = float(np.dot(w_others, [means[c] for c in others]))
        s = scenarios[(scenarios.factor == code) & (scenarios.delta != 0)]
        table.append({
            "factor": code, "weight": weights[code],
            "mean_normalized_value": means[code],
            "weighted_mean_other_factors": mean_others,
            "sensitivity_index": weights[code] * (means[code] - mean_others) / base_mean,
            "mean_si_empirical": s["si"].mean(),
            "mean_abs_si": s["si"].abs().mean(),
            "largest_change_mean_hi": s["delta_mean_hi"].abs().max(),
            "largest_pct_pixels_class_change": s["pct_pixels_class_change"].max(),
            "min_pearson_r": s["pearson_r"].min(),
        })
    return scenarios, pd.DataFrame(table)


def oat_stability(propagated: np.ndarray, weights: np.ndarray, codes: list[str], n_classes: int = 5,
                  max_delta: float = 0.20, step: float = 0.01) -> tuple[np.ndarray, int, np.ndarray]:
    """Number of perturbed scenarios in which each pixel changes Hazard class (Figure 7).

    Uses the propagated factors and the min-max rescaling of the Hazard Index.
    Returns (count per valid pixel, number of scenarios, base thresholds).
    """
    h0 = minmax(propagated @ weights).astype("float64")
    cls0, thr = multiotsu_classes(h0, n_classes)
    count = np.zeros(h0.shape[0], dtype=np.uint16)
    n = 0
    for j, code in enumerate(codes):
        for delta in oat_steps(max_delta, step):
            if delta == 0:
                continue
            h = minmax(propagated @ perturbed_weights(weights, j, delta))
            count += (apply_thresholds(h, thr) != cls0)
            n += 1
        log.info("OAT stability: %s done (%d scenarios)", code, n)
    return count, n, thr


def stability_summary(count: np.ndarray, n_scenarios: int) -> pd.DataFrame:
    """Share of pixels by share of scenarios with a class change (Figure 7b)."""
    pct = 100 * count / n_scenarios
    cats = pd.cut(pct, [-np.inf, 0, 5, 20, 50, np.inf],
                  labels=["Never", "Up to 5%", "5% to 20%", "20% to 50%", "More than 50%"])
    share = pd.Series(cats).value_counts(normalize=True, sort=False) * 100
    return share.rename_axis("scenarios_with_change").reset_index(name="pct_pixels")


# ============================================================== site level
def oat_sites(propagated: np.ndarray, weights: np.ndarray, codes: list[str], site_pos: list,
              sites: pd.DataFrame, deltas=(-0.20, -0.10, 0.10, 0.20)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OAT read at the 34 sites (32 weight vectors).

    The Hazard class and the Risk class are assigned with the thresholds of the
    main analysis kept FIXED, and with Multi-Otsu RECOMPUTED in each scenario.
    ``sites`` has ``id``, ``name``, ``vi_resc``, ``e_class`` (sorted like ``site_pos``).

    Returns (one row per scenario, one row per site).
    """
    def site_values(h):
        return np.array([h[p].max() if len(p) else np.nan for p in site_pos])

    vi_resc, e_cls = sites["vi_resc"].values, sites["e_class"].values
    s0 = site_values(minmax(propagated @ weights))
    c0, t_hi0 = multiotsu_classes(s0, 5)
    r0 = c0 * vi_resc * e_cls
    rc0, t_r0 = multiotsu_classes(r0.astype(float), 3)
    rank0 = rankdata(-s0, method="average")
    high0 = set(sites.loc[rc0 == 3, "name"])

    n = len(sites)
    per = {k: np.zeros(n, int) for k in ("hi_fixed", "r_fixed", "hi_mot", "r_mot")}
    rank_min, rank_max, shift_max = rank0.copy(), rank0.copy(), np.zeros(n)
    rows = []
    for j, code in enumerate(codes):
        for delta in deltas:
            w = perturbed_weights(weights, j, delta)
            s = site_values(minmax(propagated @ w))
            cf = apply_thresholds(s, t_hi0)
            rcf = apply_thresholds(cf * vi_resc * e_cls, t_r0)
            cm, _ = multiotsu_classes(s, 5)
            rm = cm * vi_resc * e_cls
            rcm, _ = multiotsu_classes(rm.astype(float), 3)
            rk = rankdata(-s, method="average")
            shift = np.abs(rk - rank0)
            high_f, high_m = set(sites.loc[rcf == 3, "name"]), set(sites.loc[rcm == 3, "name"])
            rows.append({
                "factor": code, "delta": delta, "weight": w[j],
                "rho_hi": spearmanr(s, s0)[0], "rho_r_fixed": spearmanr(cf * vi_resc * e_cls, r0)[0],
                "rho_r_mot": spearmanr(rm, r0)[0],
                "hi_class_changes_fixed": int((cf != c0).sum()), "r_class_changes_fixed": int((rcf != rc0).sum()),
                "hi_class_changes_mot": int((cm != c0).sum()), "r_class_changes_mot": int((rcm != rc0).sum()),
                "n_high_fixed": len(high_f), "n_high_mot": len(high_m),
                "high_lost_mot": ";".join(sorted(high0 - high_m)),
                "high_gained_mot": ";".join(sorted(high_m - high0)),
                "max_rank_shift": shift.max(), "mean_rank_shift": shift.mean(),
            })
            per["hi_fixed"] += cf != c0
            per["r_fixed"] += rcf != rc0
            per["hi_mot"] += cm != c0
            per["r_mot"] += rcm != rc0
            rank_min, rank_max = np.minimum(rank_min, rk), np.maximum(rank_max, rk)
            shift_max = np.maximum(shift_max, shift)

    per_site = sites[["id", "name"]].copy()
    per_site["hi_base"], per_site["hi_class_base"] = s0, c0
    per_site["r_base"], per_site["r_class_base"] = r0, rc0
    per_site["hi_rank_base"], per_site["hi_rank_min"], per_site["hi_rank_max"] = rank0, rank_min, rank_max
    per_site["max_rank_shift"] = shift_max
    for k, v in per.items():
        per_site[f"n_{k}_changes"] = v
    per_site["n_scenarios"] = len(rows)
    return pd.DataFrame(rows), per_site
