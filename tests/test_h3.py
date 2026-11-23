"""Tests for analysis/h3_test.py on synthetic hidden states (no real data, no text)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("sklearn")
from analysis import h3_test as h3  # noqa: E402


def synthetic(style, n=600, d=20, seed=0, noise=0.05):
    """Vanilla prompts: harmful ones shifted by 3 along axis 0. Adversarial = vanilla + the style vector + noise."""
    rng = np.random.default_rng(seed)
    harm = np.arange(n) % 2 == 0
    Hv = rng.normal(size=(n, d))
    Hv[harm, 0] += 3.0
    Ha = Hv + np.asarray(style) + rng.normal(scale=noise, size=(n, d))
    split = np.array(["train"] * int(n * 0.7) + ["val"] * int(n * 0.15) + ["test"] * (n - int(n * 0.7) - int(n * 0.15)))
    meta = pd.DataFrame({"intent": np.where(harm, "harmful", "benign"), "split": split})
    return Hv, Ha, meta


def unit(i, scale, d=20):
    v = np.zeros(d)
    v[i] = scale
    return v


def test_style_off_the_harm_axis_is_detected_by_g2_but_barely_moves_r_hat_and_does_not_fool_the_probe():
    r = h3.h3_parts(*synthetic(unit(5, 2.0)), n_boot=300)
    assert r["b"]["holds"] and r["b"]["auroc"][0] > 0.9   # a 2-sd shift: AUROC about 0.92
    assert r["c"]["holds"]
    assert abs(r["a"]["style"][0]) < 0.1 * 3.0             # only the small error in the fitted r_hat picks it up


def test_no_move_at_all_fails_part_a():
    Hv, _, meta = synthetic(np.zeros(20))
    r = h3.h3_parts(Hv, Hv + np.random.default_rng(1).normal(scale=0.05, size=Hv.shape), meta, n_boot=300)
    assert not r["a"]["holds"] and "a" in r["failed_parts"] and r["verdict"] == "not supported"


def test_style_along_the_harm_axis_moves_r_hat_and_fools_the_probe():
    r = h3.h3_parts(*synthetic(unit(0, 1.5)), n_boot=300)
    assert r["a"]["holds"] and r["a"]["style"][0] > 1
    assert not r["c"]["holds"] and r["c"]["score_difference"][0] > r["c"]["margin"]
    assert r["verdict"] == "not supported" and "c" in r["failed_parts"]


def test_no_style_means_g2_is_near_chance():
    r = h3.h3_parts(*synthetic(np.zeros(20)), n_boot=300)
    assert not r["b"]["holds"] and r["b"]["auroc"][0] < 0.7


def test_all_three_parts_can_hold_together():
    # a small push along the harm axis is visible to r_hat but too small to move the probe's score past the margin
    r = h3.h3_parts(*synthetic(unit(5, 2.0) + unit(0, 0.12), n=4000), n_boot=300)
    assert r["a"]["holds"] and r["b"]["holds"] and r["c"]["holds"] and r["verdict"] == "supported"


def test_paired_auroc_is_one_for_perfect_separation_and_counts_test_pairs():
    est, lo, hi = h3.paired_auroc_ci(np.zeros(10), np.ones(10), n_boot=50)
    assert (est, lo, hi) == (1.0, 1.0, 1.0)
    r = h3.h3_parts(*synthetic(unit(5, 2.0)), n_boot=50)
    assert r["n_test_pairs"] == 90 and r["a"]["n_benign_test"] == 45 and r["n_train_pairs"] == 420


def test_probes_never_see_test_rows():
    Hv, Ha, meta = synthetic(unit(5, 2.0))
    base = h3.h3_parts(Hv, Ha, meta, n_boot=50)
    Hv2, Ha2 = Hv.copy(), Ha.copy()
    te = (meta["split"] != "train").to_numpy()
    Hv2[te] += 50.0                                  # wreck every held-out row
    Ha2[te] += 50.0
    moved = h3.h3_parts(Hv2, Ha2, meta, n_boot=50)
    assert moved["a"]["style"][0] == pytest.approx(base["a"]["style"][0], rel=1e-4)   # dp of a shifted pair is unchanged
