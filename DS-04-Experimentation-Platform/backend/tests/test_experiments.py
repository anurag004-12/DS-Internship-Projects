import numpy as np
import pytest
from statsmodels.stats.proportion import proportions_ztest

from app.experiments import adjust_pvalues, compare_proportions


def test_proportions_matches_statsmodels():
    r = compare_proportions(100, 1000, 130, 1000)
    _, p_ref = proportions_ztest([130, 100], [1000, 1000])
    assert r["p_value"] == pytest.approx(p_ref, rel=1e-6)
    assert r["abs_lift"] == pytest.approx(0.03)
    assert r["rel_lift"] == pytest.approx(0.30)
    assert r["significant"] is True


def test_proportions_no_difference_is_not_significant():
    r = compare_proportions(100, 1000, 101, 1000)
    assert r["significant"] is False
    assert r["ci"][0] < 0 < r["ci"][1]


def test_proportions_rejects_bad_input():
    with pytest.raises(ValueError):
        compare_proportions(1500, 1000, 100, 1000)


def test_bonferroni_and_bh_hand_calculated():
    p = [0.01, 0.02, 0.04]
    bonf = [x["adjusted_p"] for x in adjust_pvalues(p, "bonferroni")]
    bh = [x["adjusted_p"] for x in adjust_pvalues(p, "benjamini-hochberg")]
    assert bonf == pytest.approx([0.03, 0.06, 0.12])
    assert bh == pytest.approx([0.03, 0.03, 0.04])


def test_correction_controls_false_alarms_across_20_metrics():
    """20 metrics with NO real effect: how often do we get at least one false win?"""
    rng = np.random.default_rng(0)
    runs = [rng.uniform(size=20) for _ in range(1000)]
    raw = np.mean([(r < 0.05).any() for r in runs])
    bonf = np.mean([any(x["significant"] for x in adjust_pvalues(list(r), "bonferroni"))
                    for r in runs])
    assert raw > 0.5      # theory: 1 - 0.95**20 = 64%
    assert bonf < 0.08    # theory: about 5%
