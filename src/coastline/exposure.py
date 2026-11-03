"""Exposure Index (Section 2.5): visitor presence from Google Places review counts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.filters import threshold_multiotsu

from .constants import LABELS_5


def exposure_norm(reviews) -> np.ndarray:
    """E_norm = log(R + 1) / log(R_max + 1) (Eq. 20). A site with no review gives 0."""
    r = np.asarray(reviews, dtype=float)
    return np.log(r + 1) / np.log(r.max() + 1)


def exposure_index(reviews: pd.DataFrame, n_classes: int = 5) -> tuple[pd.DataFrame, np.ndarray]:
    """Exposure value and class (1 Very Low ... 5 Very High) of each site.

    ``reviews`` has the columns ``id``, ``name``, ``reviews``.
    """
    df = reviews[["id", "name", "reviews"]].copy()
    df["reviews"] = df["reviews"].fillna(0).astype(float)
    df["e_norm"] = exposure_norm(df["reviews"])
    values = df["e_norm"].values
    if len(np.unique(values)) >= n_classes:
        thresholds = threshold_multiotsu(values, classes=n_classes)
    else:  # too few distinct values for Multi-Otsu (never the case in the article)
        thresholds = np.quantile(values, np.linspace(0, 1, n_classes + 1)[1:-1])
    df["e_class"] = np.digitize(values, bins=thresholds) + 1
    df["e_label"] = df["e_class"].map(LABELS_5)
    return df, thresholds
