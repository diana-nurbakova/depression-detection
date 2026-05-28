"""Tests for tom_act.wasserstein — W1 properties and temporal variants."""

from mentalriskes.tom_act import wasserstein as W


def test_w1_identical_is_zero():
    assert W.w1_between([1, 2, 3, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0], "GAD-7") == 0.0


def test_w1_zero_vector_safe():
    assert W.w1_between([0] * 9, [0] * 9, "PHQ-9") == 0.0
    assert W.w1_between([1, 0, 0, 0, 0, 0, 0, 0, 0], [0] * 9, "PHQ-9") == 0.0


def test_w1_symmetric_and_positive():
    a = [3, 0, 0, 0, 0, 0, 0, 0, 0]
    b = [0, 0, 0, 0, 0, 0, 0, 0, 3]
    ab = W.w1_between(a, b, "PHQ-9")
    ba = W.w1_between(b, a, "PHQ-9")
    assert ab > 0
    assert abs(ab - ba) < 1e-9


def test_temporal_consecutive_warmup_and_fire():
    # Flat then a big jump at the last round.
    vecs = [(r, [1, 1, 1, 1, 1, 1, 1]) for r in range(1, 6)]
    vecs.append((6, [3, 0, 0, 0, 0, 0, 0]))
    rows = W.temporal_traces(vecs, "GAD-7", "consecutive", alert_sigma=2.0)
    assert rows[0]["w1"] == 0.0                 # round 1 degenerate
    assert all(not r["fired"] for r in rows[:3])  # k<3 warmup never fires
    assert rows[-1]["fired"] is True            # the jump fires
    assert all(r["variant"] == "consecutive" for r in rows)


def test_temporal_barycenter_variant_runs():
    vecs = [(r, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) for r in range(1, 5)]
    rows = W.temporal_traces(vecs, "CompACT-10", "barycenter")
    assert len(rows) == 4
    assert all(r["variant"] == "barycenter" for r in rows)


def test_cross_perspective_round_pairs():
    views = {
        "self_a": [1, 0, 0, 0, 0, 0, 0],
        "self_b": [1, 1, 0, 0, 0, 0, 0],
        "observer_p": [0, 0, 1, 0, 0, 0, 0],
        "observer_pt": [0, 0, 0, 1, 0, 0, 0],
    }
    gaps = W.cross_perspective_round(views, "GAD-7")
    assert "self_a__observer_p" in gaps
    assert len(gaps) == 6
    assert all(v >= 0 for v in gaps.values())


def test_aggregate_gap_weighting():
    g = W.aggregate_gap({"PHQ-9": 1.0, "GAD-7": 1.0, "CompACT-10": 1.0})
    assert abs(g - 1.0) < 1e-9
