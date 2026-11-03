"""Multi-Otsu classification (Eq. 1 of the article).

Every class in the framework is computed "live" from the data with
Multi-Otsu thresholding: no threshold is written by hand in the code.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import threshold_multiotsu


def multiotsu_classes(values, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Split ``values`` into ``n_classes`` ordinal classes (1 = lowest).

    The thresholds are computed on the finite values only. A value equal to a
    threshold goes to the upper class (``np.digitize`` with ``right=False``).

    Returns
    -------
    classes : int array (same shape as ``values``; 0 where the value is not finite)
    thresholds : the ``n_classes - 1`` thresholds
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    thresholds = threshold_multiotsu(v[finite], classes=n_classes)
    classes = np.digitize(v, bins=thresholds) + 1
    classes[~finite] = 0
    return classes.astype(int), thresholds


def apply_thresholds(values, thresholds) -> np.ndarray:
    """Classes of ``values`` with thresholds computed elsewhere (fixed thresholds)."""
    return np.digitize(np.asarray(values, dtype=float), bins=thresholds) + 1


def multiotsu_score(values, inverted: bool, n_classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Score a continuous attribute from 1 to ``n_classes`` with Multi-Otsu.

    ``inverted=True`` gives the highest score to the LOWEST values
    (elevation and distance to coast). Missing values keep a missing score.
    """
    v = np.asarray(values, dtype=float)
    classes, thresholds = multiotsu_classes(v, n_classes)
    score = classes.astype(float)
    if inverted:
        score = n_classes + 1 - score
    score[~np.isfinite(v)] = np.nan
    return score, thresholds


def fixed_score(values, t1: float, t2: float) -> np.ndarray:
    """Inverted 3-class score with fixed limits: 3 if x <= t1, 2 if x <= t2, else 1."""
    x = np.asarray(values, dtype=float)
    return np.where(x <= t1, 3, np.where(x <= t2, 2, 1))
