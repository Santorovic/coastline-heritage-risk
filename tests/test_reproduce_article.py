"""Regression tests: the results of scripts 01-05 must equal the article.

The reference values are in tests/reference/ (results of the verified run of the
article, Draft v3). Each test is skipped if the corresponding output has not
been produced yet, so `pytest` also runs on a computer without the rasters.

Run the pipeline first (see README), then:  pytest -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coastline.config import load_config

REF = Path(__file__).parent / "reference"
CFG = load_config()


def load(*parts):
    path = CFG.results_dir.joinpath(*parts)
    if not path.exists():
        pytest.skip(f"{path} not found: run the pipeline first")
    return pd.read_csv(path)


def test_cascade_weights_table5():
    out = load("hazard", "weights.csv").set_index("factor")["weight_rounded"]
    ref = pd.read_csv(REF / "weights.csv").set_index("factor")["weight_rounded"]
    pd.testing.assert_series_equal(out, ref)


@pytest.mark.parametrize("column", ["hi", "hi_class", "vi_norm", "vi_class", "e_norm", "e_class", "r", "r_class"])
def test_risk_index_of_the_34_sites(column):
    out = load("risk", "risk_index.csv").sort_values("id")
    ref = pd.read_csv(REF / "risk_index.csv").sort_values("id")
    np.testing.assert_allclose(out[column].values, ref[column].values, atol=1e-4)


def test_article_summary_numbers():
    r = load("risk", "risk_index.csv")
    assert list(np.bincount(r.r_class)[1:]) == [17, 12, 5]          # Risk classes Low/Medium/High
    assert list(np.bincount(r.hi_class)[1:]) == [6, 8, 9, 9, 2]     # Hazard classes
    assert list(np.bincount(r.vi_class)[1:]) == [10, 7, 17]         # Vulnerability classes
    assert list(np.bincount(r.e_class)[1:]) == [3, 10, 10, 8, 3]    # Exposure classes
    assert r.r.median() == 23.5 and round(r.r.mean(), 1) == 28.3


@pytest.mark.parametrize("name, key", [
    ("oat_pixel_table8.csv", "factor"),
    ("vulnerability_tests.csv", "test"),
    ("coastal_tests.csv", "test"),
    ("aggregation.csv", "aggregation"),
    ("oat_stability_summary.csv", "scenarios_with_change"),
    ("oat_sites_per_site.csv", "id"),
])
def test_sensitivity_tables(name, key):
    out = load("sensitivity", name).set_index(key).sort_index()
    ref = pd.read_csv(REF / name).set_index(key).sort_index()
    num = ref.select_dtypes("number").columns
    np.testing.assert_allclose(out[num].values, ref[num].values, atol=1e-3)
    txt = [c for c in ref.columns if c not in num]
    pd.testing.assert_frame_equal(out[txt], ref[txt])
