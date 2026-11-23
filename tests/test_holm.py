"""Tests for analysis/holm.py: bootstrap p-values agree with the CI rules, and the Holm step-down is right."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis import holm, lib  # noqa: E402


def test_resamples_are_the_ones_behind_lib_bootstrap_ci():
    x = np.random.default_rng(3).normal(size=57)
    est, lo, hi = lib.bootstrap_ci(x, np.mean, n=2000, seed=0)
    b = holm.boot_means(x, 2000, 0)
    assert (lo, hi) == pytest.approx((np.quantile(b, 0.025), np.quantile(b, 0.975)))


@pytest.mark.parametrize("shift", [-0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5])
def test_p_below_0_05_exactly_when_the_95_ci_excludes_the_null(shift):
    x = np.random.default_rng(1).normal(loc=shift, size=80)
    b = holm.boot_means(x, 2000, 0)
    lo, hi = np.quantile(b, [0.025, 0.975])
    assert (holm.p_excludes(b, 0.0) < 0.05) == (lo > 0 or hi < 0)
    assert (holm.p_above(b, 0.0) < 0.05) == (lo > 0)
    assert (holm.p_below(b, 0.0) < 0.05) == (hi < 0)
    assert (holm.p_inside(b, 0.3) < 0.05) == (-0.3 < lo and hi < 0.3)


def test_p_is_never_zero_and_never_above_one():
    assert holm.p_excludes(np.ones(2000), 0.0) == pytest.approx(2 / 2001)
    assert holm.p_above(np.zeros(10), 5.0) == 1.0


def test_holm_step_down_on_a_textbook_example():
    p = dict(H1=0.01, H2=0.04, H3=0.03, H4=0.005, H5=0.2, H6=0.5, H7=0.001)
    r = holm.holm(p)
    # sorted: H7 .001 x7=.007, H4 .005 x6=.03, H1 .01 x5=.05, H3 .03 x4=.12, H2 .04 x3=.12, H5 .2 x2=.4, H6 .5
    assert [r[t]["adjusted_p"] for t in ("H7", "H4", "H1", "H3", "H2", "H5", "H6")] == pytest.approx(
        [0.007, 0.03, 0.05, 0.12, 0.12, 0.4, 0.5])
    assert {t for t, v in r.items() if v["decision"] == "rejected"} == {"H7", "H4", "H1"}


def test_holm_with_missing_tests_is_provisional_and_safe():
    r = holm.holm(dict(H1=0.001, H2=0.02, H3=0.3))
    assert r["H1"]["decision"] == "rejected whatever the missing tests give"      # 0.001 <= 0.05 / 7
    assert r["H2"]["decision"].startswith("waits for H4")
    assert r["H3"]["decision"] == "not rejected whatever the missing tests give"
    assert all(v["adjusted_p"] is None for v in r.values())


def test_h1_and_h2_rules_combine_their_parts():
    pa = np.r_[np.zeros(80), np.ones(20)]          # 80% of adversarial prompts below a midpoint of 0.5
    pv = np.ones(100)                                # no vanilla twin below it
    r = holm.h1_p(pa, pv, 0.5)
    assert r["p"] < 0.05 and r["p"] == max(r["parts"].values())
    r = holm.h1_p(np.r_[np.zeros(52), np.ones(48)], pv, 0.5)   # 52%: majority, but the CI reaches 0.5
    assert r["p"] > 0.05
    no_control = holm.h2_p(np.full(10, -5.0), np.zeros(10), np.zeros(20), np.zeros(1))
    assert no_control["p"] == 1.0 and not no_control["computable"]
