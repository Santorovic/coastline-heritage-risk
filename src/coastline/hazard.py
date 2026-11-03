"""Hazard Index (Section 2.3): redundancy pre-screening, cascade weighting, index.

The functions work on the matrix ``X`` of the valid land pixels:
one row per pixel, one column per factor (order of ``codes``).

    X = np.column_stack([arrays[c][valid] for c in codes])
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# --------------------------------------------------------------- sampling
def pixel_sample(n_valid: int, fraction: float, seed: int) -> np.ndarray:
    """Random sample (without replacement) of the valid pixels.

    Uses the legacy NumPy generator so that the sample of the article
    (``np.random.seed(42)`` then ``np.random.choice``) is reproduced exactly.
    """
    rng = np.random.RandomState(seed)
    return rng.choice(n_valid, size=int(n_valid * fraction), replace=False)


def correlation_matrix(X_sample: np.ndarray, codes: list[str]) -> pd.DataFrame:
    """Pearson correlation between every pair of factors (Eq. 10)."""
    return pd.DataFrame(X_sample, columns=codes).corr(method="pearson")


# ------------------------------------------------- pre-screening (PCA)
@dataclass
class Screening:
    correlation: pd.DataFrame
    redundant_pairs: pd.DataFrame      # pairs with |r| above the threshold
    eigen: pd.DataFrame                # eigenvalue, explained and cumulative variance per PC
    loadings: pd.DataFrame             # factor x PC
    n_kaiser: int
    n_cumulative: int
    n_retained: int


def redundancy_screening(X_sample: np.ndarray, codes: list[str], r_max: float = 0.70,
                         kaiser: float = 1.0, cum_var: float = 0.75) -> Screening:
    """Pearson correlation + Principal Component Analysis (Eqs. 10-12)."""
    corr = correlation_matrix(X_sample, codes)
    pairs = [(codes[i], codes[j], corr.iloc[i, j])
             for i in range(len(codes)) for j in range(i + 1, len(codes))
             if abs(corr.iloc[i, j]) > r_max]
    redundant = pd.DataFrame(pairs, columns=["factor_a", "factor_b", "r"])

    scaled = StandardScaler().fit_transform(X_sample)
    pca = PCA(n_components=len(codes), random_state=42).fit(scaled)
    eigenvalues = pca.explained_variance_
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pcs = [f"PC{i + 1}" for i in range(len(codes))]
    eigen = pd.DataFrame({
        "pc": pcs,
        "eigenvalue": eigenvalues,
        "variance_pct": 100 * pca.explained_variance_ratio_,
        "cumulative_pct": 100 * cumulative,
    })
    loadings = pd.DataFrame(pca.components_.T, index=codes, columns=pcs)

    n_kaiser = int((eigenvalues > kaiser).sum())
    n_cum = int(np.searchsorted(cumulative, cum_var) + 1)
    return Screening(corr, redundant, eigen, loadings, n_kaiser, n_cum, max(n_kaiser, n_cum))


# ---------------------------------------------------- cascade weighting
@dataclass
class Cascade:
    codes: list[str]
    edges: np.ndarray            # |r_ij| with zero diagonal (Eq. 13)
    weights: np.ndarray          # network centrality, sums to 1 (Eq. 16)
    cond_prob: np.ndarray        # P(H_j | H_i), row = i (Eq. 14), zero diagonal
    high_thresholds: np.ndarray  # 75th percentile of each factor
    propagated: np.ndarray       # n_valid x k, rescaled to [0, 1] per factor (Eq. 15)

    def table(self, decimals: int = 3) -> pd.DataFrame:
        raw = self.edges.sum(axis=1)
        return pd.DataFrame({"factor": self.codes, "centrality": raw, "weight": self.weights,
                             "weight_rounded": np.round(self.weights, decimals)})


def cascade_weighting(X: np.ndarray, sample_idx: np.ndarray, codes: list[str],
                      percentile: float = 75) -> Cascade:
    """Cascade weighting of the factors (Capolupo and Tarantino, 2026; Tang et al., 2023).

    1. edge weights w_ij = |Pearson r| on the pixel sample, diagonal = 0
    2. weight of a factor = its weighted degree centrality, normalized to sum 1
    3. P(H_j | H_i) = share of the pixels "high" in i (above the percentile)
       that are also high in j
    4. propagation H_j' = H_j + sum_i w_ij P(H_j|H_i) H_i, then min-max per factor
    """
    k = len(codes)
    corr = correlation_matrix(X[sample_idx], codes).values
    edges = np.abs(corr)
    np.fill_diagonal(edges, 0)
    centrality = edges.sum(axis=1)
    weights = centrality / centrality.sum()

    thresholds = np.percentile(X, percentile, axis=0)
    high = X > thresholds
    cond = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        n_high = high[:, i].sum()
        if n_high == 0:          # empty "high" set (RNI: P75 = max): no propagation from i
            continue
        for j in range(k):
            if i != j:
                cond[i, j] = (high[:, i] & high[:, j]).sum() / n_high
    del high

    propagated = X + X @ (edges * cond)   # diagonal of edges is 0: no self-reinforcement
    for j in range(k):
        lo, hi = propagated[:, j].min(), propagated[:, j].max()
        if hi > lo:
            propagated[:, j] = (propagated[:, j] - lo) / (hi - lo)
    log.info("cascade weights: %s", dict(zip(codes, np.round(weights, 3))))
    return Cascade(codes, edges, weights, cond, thresholds, propagated)


def minmax(v: np.ndarray) -> np.ndarray:
    return (v - v.min()) / (v.max() - v.min())


def hazard_index(propagated: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted sum of the propagated factors (Eq. 17), rescaled to [0, 1]."""
    return minmax((propagated * weights).sum(axis=1))


def to_grid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Put the values of the valid pixels back on the raster grid (NaN elsewhere)."""
    grid = np.full(valid.shape, np.nan, dtype=np.float32)
    grid[valid] = values
    return grid
