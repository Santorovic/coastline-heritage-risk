"""Unit tests on small synthetic data: they run anywhere, without the rasters.

Run with:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from coastline.classify import fixed_score, multiotsu_classes, multiotsu_score
from coastline.config import load_config
from coastline.exposure import exposure_index, exposure_norm
from coastline.hazard import cascade_weighting, hazard_index, pixel_sample
from coastline.risk import rescale_vulnerability, risk_from_classes
from coastline.sensitivity.weights import oat_steps, perturbed_weights
from coastline.vulnerability import score_enclosure, score_heritage, score_land_use, score_shelter


def test_config_has_the_eight_factors_in_the_article_order():
    cfg = load_config()
    assert cfg.factor_codes == ["SAI", "BUI", "LSI", "CPI", "FFI", "RNI", "SLRI", "FHI"]


def test_multiotsu_separates_three_clear_groups():
    values = np.r_[np.full(10, 1.0), np.full(10, 5.0), np.full(10, 9.0)]
    classes, thresholds = multiotsu_classes(values, 3)
    assert list(np.unique(classes[:10])) == [1] and list(np.unique(classes[-10:])) == [3]
    assert len(thresholds) == 2


def test_inverted_score_gives_3_to_the_lowest_values():
    score, _ = multiotsu_score(np.r_[np.full(5, 2.0), np.full(5, 50.0), np.full(5, 500.0)], inverted=True)
    assert score[0] == 3 and score[-1] == 1


def test_fixed_score_limits_are_inclusive():
    assert list(fixed_score([10, 11, 50, 51], 10, 50)) == [3, 2, 2, 1]


@pytest.mark.parametrize("delta", [-0.2, -0.05, 0.1, 0.2])
def test_perturbed_weights_sum_to_one(delta):
    w = np.array([0.131, 0.132, 0.262, 0.101, 0.049, 0.093, 0.183, 0.049])
    p = perturbed_weights(w, 2, delta)
    assert p.sum() == pytest.approx(1.0)
    assert p[2] == pytest.approx(w[2] * (1 + delta))


def test_oat_has_41_steps_per_factor():
    steps = oat_steps(0.20, 0.01)
    assert len(steps) == 41 and steps[0] == -0.20 and steps[-1] == 0.20


def test_cascade_weights_and_propagated_values():
    rng = np.random.RandomState(0)
    a = rng.rand(5000)
    X = np.column_stack([a, a * 0.5 + rng.rand(5000) * 0.5, rng.rand(5000)]).astype(np.float32)
    c = cascade_weighting(X, pixel_sample(len(X), 0.2, 42), ["A", "B", "C"])
    assert c.weights.sum() == pytest.approx(1.0)
    assert c.weights[2] < c.weights[0]                 # C is unrelated to A and B
    assert np.all(np.diag(c.cond_prob) == 0)
    assert c.propagated.min() >= 0 and c.propagated.max() <= 1
    hi = hazard_index(c.propagated, c.weights)
    assert hi.min() == pytest.approx(0) and hi.max() == pytest.approx(1)


def test_vulnerability_scores():
    assert score_heritage("UNESCO World Heritage Site") == 1
    assert score_heritage("No UNESCO") == 3
    assert [score_enclosure(v) for v in ["Full Fence", "Partial Fence", "No Fence", "?"]] == [1, 2, 3, 2]
    assert [score_shelter(v) for v in ["Yes", "Partial", "No"]] == [1, 2, 3]
    assert [score_land_use(c) for c in [10, 30, 40, 50, 60, 20]] == [1, 2, 2, 3, 3, 2]


def test_exposure_norm_limits():
    e = exposure_norm([0, 9, 99])
    assert e[0] == 0 and e[-1] == pytest.approx(1)


def test_exposure_classes_are_1_to_5():
    df = pd.DataFrame({"id": range(20), "name": "x", "reviews": np.logspace(0, 4, 20)})
    table, thr = exposure_index(df, 5)
    assert set(table["e_class"]) <= {1, 2, 3, 4, 5} and len(thr) == 4


def test_risk_range_and_rescaling():
    assert list(rescale_vulnerability([1, 2, 3])) == [1, 3, 5]
    r, r_class, _ = risk_from_classes([1, 5, 3, 2, 4, 5], [1, 5, 3, 1, 3, 5], [1, 5, 2, 1, 4, 4])
    assert r.min() >= 1 and r.max() <= 125 and set(r_class) <= {1, 2, 3}
