"""Risk Index (Section 2.6): R = HI_class x VI_resc x E_class, classified with Multi-Otsu."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .classify import multiotsu_classes
from .constants import LABELS_3, LABELS_5

DEFAULT_VI_RESCALE = {1: 1, 2: 3, 3: 5}


def rescale_vulnerability(vi_class, mapping: dict | None = None) -> np.ndarray:
    """Stretch the 3 Vulnerability classes to 1, 3, 5 (same range as Hazard and Exposure)."""
    mapping = mapping or DEFAULT_VI_RESCALE
    return np.array([mapping[int(c)] for c in vi_class])


def risk_from_classes(hi_class, vi_resc, e_class, n_classes: int = 3):
    """Risk value (Eq. 21, range 1-125) and Risk class (1 Low, 2 Medium, 3 High)."""
    r = np.asarray(hi_class) * np.asarray(vi_resc) * np.asarray(e_class)
    r_class, thr = multiotsu_classes(r.astype(float), n_classes)
    return r.astype(int), r_class, thr


def risk_from_hazard(hi_site, vi_resc, e_class, n_hazard: int = 5, n_risk: int = 3):
    """Classify the site Hazard values, then compute the Risk value and class.

    Returns (hi_class, R, r_class, hazard thresholds, risk thresholds).
    """
    hi_class, t_hi = multiotsu_classes(hi_site, n_hazard)
    r, r_class, t_r = risk_from_classes(hi_class, vi_resc, e_class, n_risk)
    return hi_class, r, r_class, t_hi, t_r


def risk_index(hazard: pd.DataFrame, vulnerability: pd.DataFrame, exposure: pd.DataFrame,
               vi_rescale: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Join the three indices by site id and compute the Risk Index.

    ``hazard`` has ``id``, ``name``, ``hi`` (buffer maximum), ``hi_radius_m``;
    ``vulnerability`` has ``id``, ``vi_norm``, ``vi_class``;
    ``exposure`` has ``id``, ``e_norm``, ``e_class``.
    """
    df = (hazard.merge(vulnerability[["id", "vi_norm", "vi_class"]], on="id", how="inner")
                .merge(exposure[["id", "reviews", "e_norm", "e_class"]], on="id", how="inner"))
    if not (len(df) == len(hazard) == len(vulnerability) == len(exposure)):
        raise ValueError("the three indices do not cover the same sites")
    df["vi_resc"] = rescale_vulnerability(df["vi_class"], vi_rescale)
    df["hi_class"], df["r"], df["r_class"], t_hi, t_r = risk_from_hazard(
        df["hi"].values, df["vi_resc"].values, df["e_class"].values)
    df["hi_label"] = df["hi_class"].map(LABELS_5)
    df["vi_label"] = df["vi_class"].map(LABELS_3)
    df["e_label"] = df["e_class"].map(LABELS_5)
    df["r_label"] = df["r_class"].map(LABELS_3)
    cols = ["id", "name", "hi", "hi_radius_m", "hi_class", "hi_label", "vi_norm", "vi_class", "vi_label",
            "vi_resc", "reviews", "e_norm", "e_class", "e_label", "r", "r_class", "r_label"]
    return df[cols], {"hazard": t_hi, "risk": t_r}
