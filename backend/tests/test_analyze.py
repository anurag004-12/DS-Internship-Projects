import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.analyze import analyze_ab, load_csv, preview
from app.experiments import compare_proportions, simulate_experiment
from app.main import app


def _binary_df(n=20_000, p_a=0.10, p_b=0.12, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.random(n) < 0.5
    return pd.DataFrame({"variant": np.where(b, "B", "A"),
                         "converted": (rng.random(n) < np.where(b, p_b, p_a)).astype(int)})


def _cont_df(**kw):
    d = simulate_experiment(**kw)
    d["arm"] = np.where(d.group == 1, "treatment", "control")
    return d


# ------------------------------------------------------------ results
def test_binary_matches_the_underlying_test_and_assumes_labels():
    df = _binary_df()
    r = analyze_ab(df, "variant", "converted")
    k = df.groupby("variant")["converted"].agg(["sum", "count"])
    ref = compare_proportions(int(k.loc["A", "sum"]), int(k.loc["A", "count"]),
                              int(k.loc["B", "sum"]), int(k.loc["B", "count"]))
    assert r["result"]["p_value"] == pytest.approx(ref["p_value"])
    assert r["verdict"]["status"] == "significant_win"
    assert r["setup"]["metric_type"] == "binary"
    assert r["result"]["prob_treatment_better"] > 0.95
    assert any("assumed" in w["message"] for w in r["warnings"])


def test_swapping_labels_flips_the_sign():
    df = _binary_df()
    fwd = analyze_ab(df, "variant", "converted", "A", "B")["result"]["effect"]
    rev = analyze_ab(df, "variant", "converted", "B", "A")["result"]["effect"]
    assert fwd == pytest.approx(-rev)


def test_continuous_with_cuped_recovers_effect_and_cuts_noise():
    r = analyze_ab(_cont_df(n_users=20_000, true_effect=0.5, seed=1), "arm", "metric",
                   "control", "treatment", pre_metric_col="pre_metric")
    assert r["result"]["effect"] == pytest.approx(0.5, abs=0.15)
    assert 40 < r["cuped"]["variance_reduction_pct"] < 60
    assert r["verdict"]["status"] == "significant_win"


def test_no_real_effect_is_inconclusive_and_explains_power():
    r = analyze_ab(_cont_df(n_users=2_000, true_effect=0.0, seed=11), "arm", "metric", "control", "treatment")
    assert r["verdict"]["status"] == "inconclusive"
    assert r["doctor"]["min_detectable_change_80pct_power"] > 0
    assert any("not 'no effect'" in d for d in r["verdict"]["details"])


# ------------------------------------------------------------ doctor checks
def test_sample_ratio_mismatch_blocks_the_verdict():
    r = analyze_ab(_cont_df(n_users=20_000, true_effect=0.5, seed=1, srm_bug=0.15),
                   "arm", "metric", "control", "treatment")
    assert r["verdict"]["status"] == "do_not_trust"
    assert any(w["level"] == "error" for w in r["warnings"])


def test_extreme_users_make_the_verdict_fragile():
    d = _cont_df(n_users=20_000, true_effect=0.0, seed=2)
    d.loc[d.index[d.group == 1][:5], "metric"] = 60_000
    r = analyze_ab(d, "arm", "metric", "control", "treatment")
    assert r["result"]["significant"] is True            # raw numbers look like a win...
    assert r["robustness"]["significant"] is False       # ...but not once whales are capped
    assert r["verdict"]["status"] == "fragile"


def test_differential_missing_data_and_imbalanced_start_are_flagged():
    d = _cont_df(n_users=10_000, true_effect=0.3, seed=7)
    drop = (d.group == 1) & (np.random.default_rng(0).random(len(d)) < 0.15)
    d.loc[drop, "metric"] = np.nan
    d.loc[d.group == 1, "pre_metric"] += 1.0
    r = analyze_ab(d, "arm", "metric", "control", "treatment", pre_metric_col="pre_metric")
    msgs = " ".join(w["message"] for w in r["warnings"])
    assert "Missing metric values" in msgs and "BEFORE the experiment" in msgs
    assert r["doctor"]["pre_experiment_balance"]["balanced"] is False


def test_estimates_are_unbiased_with_good_coverage():
    covered, effects = 0, []
    for seed in range(60):
        r = analyze_ab(_cont_df(n_users=4_000, true_effect=0.1, seed=seed), "arm", "metric",
                       "control", "treatment", pre_metric_col="pre_metric")
        lo, hi = r["result"]["ci_95"]
        covered += lo <= 0.1 <= hi
        effects.append(r["result"]["effect"])
    assert abs(np.mean(effects) - 0.1) < 0.05
    assert covered >= 52                                  # about 95% of 60


# ------------------------------------------------------------ input handling
def test_bad_inputs_give_clear_errors():
    df = _binary_df(n=500)
    with pytest.raises(ValueError, match="not found"):
        analyze_ab(df, "nope", "converted")
    with pytest.raises(ValueError, match="different"):
        analyze_ab(df, "variant", "variant")
    three = df.assign(variant=np.random.default_rng(0).choice(["A", "B", "C"], len(df)))
    with pytest.raises(ValueError, match="control and which is treatment"):
        analyze_ab(three, "variant", "converted")
    assert analyze_ab(three, "variant", "converted", "A", "B")["setup"]["treatment"] == "B"
    with pytest.raises(ValueError, match="not numeric"):
        analyze_ab(df.assign(converted=np.where(df.converted == 1, "yes", "no")), "variant", "converted")
    with pytest.raises(ValueError, match="at least 30"):
        analyze_ab(df.head(40), "variant", "converted")
    with pytest.raises(ValueError, match="0/1"):
        analyze_ab(df.assign(converted=df.converted * 5), "variant", "converted", metric_type="binary")
    with pytest.raises(ValueError, match="empty"):
        load_csv(b"")


def test_preview_describes_columns():
    p = preview(_binary_df(n=100))
    assert p["n_rows"] == 100 and {c["name"] for c in p["columns"]} == {"variant", "converted"}
    assert p["columns"][0]["n_unique"] == 2


# ------------------------------------------------------------ HTTP layer
def _csv(df):
    return ("data.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")


def test_upload_endpoints_end_to_end():
    c = TestClient(app)
    df = _binary_df(n=5_000)
    r = c.post("/analyze/preview", files={"file": _csv(df)})
    assert r.status_code == 200 and r.json()["n_rows"] == 5_000

    r = c.post("/analyze/ab", files={"file": _csv(_binary_df())},
               data={"group_col": "variant", "metric_col": "converted", "control_label": "A", "treatment_label": "B"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"]["status"] == "significant_win"
    assert len(body["charts"]["group_estimates"]) == 2

    r = c.post("/analyze/ab", files={"file": _csv(df)}, data={"group_col": "nope", "metric_col": "converted"})
    assert r.status_code == 422 and "not found" in r.json()["detail"]

    r = c.post("/analyze/ab", files={"file": ("e.csv", io.BytesIO(b""), "text/csv")},
               data={"group_col": "a", "metric_col": "b"})
    assert r.status_code == 422