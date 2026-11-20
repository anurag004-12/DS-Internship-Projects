from fastapi.testclient import TestClient

from app.experiments import compare_means, cuped_adjust, simulate_experiment, srm_check
from app.main import app
from app.power import sample_size_proportions


def test_known_sample_size():
    # Textbook case: 10% -> 11% needs roughly 14,700 users per group.
    r = sample_size_proportions(0.10, 0.01)
    assert 14_000 < r["n_control"] < 15_500


def test_srm_detects_bug_and_passes_clean():
    clean = simulate_experiment(seed=1)
    buggy = simulate_experiment(seed=1, srm_bug=0.15)
    c = srm_check((clean.group == 0).sum(), (clean.group == 1).sum())
    b = srm_check((buggy.group == 0).sum(), (buggy.group == 1).sum())
    assert not c["srm_detected"] and b["srm_detected"]


def test_cuped_shrinks_variance_and_keeps_estimate_unbiased():
    df = simulate_experiment(n_users=50_000, true_effect=0.5, seed=7)
    raw = compare_means(df)
    df["adj"] = cuped_adjust(df)
    adj = compare_means(df, metric="adj")
    assert adj["se"] < raw["se"] * 0.8
    assert abs(adj["effect"] - 0.5) < 0.15


def test_simulator_correlation_matches_knob():
    df = simulate_experiment(n_users=100_000, true_effect=0.0, pre_corr=0.7, seed=3)
    assert abs(df["pre_metric"].corr(df["metric"]) - 0.7) < 0.02


def test_simulator_recovers_planted_effect():
    df = simulate_experiment(n_users=200_000, true_effect=0.5, seed=11)
    assert abs(compare_means(df)["effect"] - 0.5) < 0.1


def test_api_endpoints():
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/power/proportions", json={"p_baseline": 0.1, "mde_abs": 0.01})
    assert r.status_code == 200 and r.json()["n_control"] > 10_000
    assert c.post("/power/proportions", json={"p_baseline": 0.99, "mde_abs": 0.05}).status_code == 422
    assert c.post("/doctor/srm", json={"n_control": 5000, "n_treatment": 4000}).json()["srm_detected"]


def test_new_endpoints():
    c = TestClient(app)
    r = c.post("/experiments/proportions",
               json={"conv_control": 100, "n_control": 1000, "conv_treatment": 130, "n_treatment": 1000})
    assert r.status_code == 200 and r.json()["significant"] is True
    r = c.post("/experiments/adjust-pvalues", json={"p_values": [0.01, 0.02, 0.04], "method": "bonferroni"})
    assert r.status_code == 200 and abs(r.json()[0]["adjusted_p"] - 0.03) < 1e-9
    assert c.post("/experiments/adjust-pvalues", json={"p_values": [0.01], "method": "nope"}).status_code == 422

def test_sequential_endpoints():
    c = TestClient(app)
    r = c.post("/sequential/check", json={"mean_diff": 0.1, "n_per_group": 3000, "sd": 1.0})
    assert r.status_code == 200 and r.json()["decision"] == "stop: significant"
    r = c.post("/sequential/peeking-demo", json={"n_sims": 2000, "n_looks": 10})
    assert r.status_code == 200 and r.json()["naive_peeking"] > r.json()["fixed_horizon"]
    assert c.post("/sequential/peeking-demo", json={"n_looks": 1}).status_code == 422


def test_bayesian_endpoint():
    c = TestClient(app)
    r = c.post("/bayesian/ab", json={"conv_a": 100, "n_a": 1000, "conv_b": 130, "n_b": 1000})
    assert r.status_code == 200 and 0.97 < r.json()["prob_b_beats_a"] < 0.99
    assert c.post("/bayesian/ab", json={"conv_a": 1500, "n_a": 1000, "conv_b": 1, "n_b": 10}).status_code == 422


def test_causal_endpoint():
    c = TestClient(app)
    r = c.post("/causal/compare", json={"n": 2000})
    assert r.status_code == 200
    names = [x["method"] for x in r.json()["results"]]
    assert "naive difference" in names and "propensity matching" in names
    assert c.post("/causal/compare", json={"n": 10}).status_code == 422


def test_panel_endpoint():
    c = TestClient(app)
    r = c.post("/causal/panel-compare", json={"loading_spread": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["did"]["parallel_trends_ok"] is False
    assert abs(body["synthetic_control"]["error_vs_truth"]) < abs(body["did"]["error_vs_truth"])
    assert c.post("/causal/panel-compare", json={"n_donors": 1}).status_code == 422