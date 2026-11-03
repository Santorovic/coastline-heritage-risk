"""Aggregation of the Risk Index (Section 2.7): two alternatives to the product of classes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def top(values, n: int = 5) -> np.ndarray:
    """Boolean mask of the ``n`` highest values (ties broken by position, stable)."""
    mask = np.zeros(len(values), bool)
    mask[np.argsort(-np.asarray(values), kind="stable")[:n]] = True
    return mask


def alternative_aggregations(risk: pd.DataFrame, n_top: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Product of the continuous indices and mean rank, compared with the Risk Index.

    ``risk`` is the Risk table (``hi``, ``vi_norm``, ``e_norm``, ``r``, ``name``).
    Returns (one row per site, summary).
    """
    df = risk[["id", "name", "r"]].copy()
    df["product"] = (risk["hi"] * risk["vi_norm"] * risk["e_norm"]).values
    df["mean_rank"] = (rankdata(risk["hi"]) + rankdata(risk["vi_norm"]) + rankdata(risk["e_norm"])) / 3
    top_r = top(risk["r"], n_top)
    rows = []
    for col in ("product", "mean_rank"):
        df[f"top_{col}"] = top(df[col], n_top)
        rows.append({
            "aggregation": col,
            "rho_with_risk": spearmanr(df[col], df["r"])[0],
            "shared_top": int((df[f"top_{col}"] & top_r).sum()),
            "top_sites": ";".join(df.loc[df[f"top_{col}"], "name"]),
            "risk_top_sites_missing": ";".join(df.loc[top_r & ~df[f"top_{col}"], "name"]),
        })
    return df, pd.DataFrame(rows)
