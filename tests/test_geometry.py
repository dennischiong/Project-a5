"""Tests for analysis/geometry.py: synthetic data with a KNOWN planted answer (no real prompts, no GPU)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis import geometry as g  # noqa: E402

D = 32


def two_tactics(angle_deg=60, n=200, noise=0.02, seed=0):
    """Tactic A moves along a, tactic B along b, with exactly angle_deg between a and b."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=D)
    a /= np.linalg.norm(a)
    z = rng.normal(size=D)
    z -= (z @ a) * a
    z /= np.linalg.norm(z)
    th = np.radians(angle_deg)
    b = np.cos(th) * a + np.sin(th) * z
    scale = rng.uniform(0.5, 2.0, size=(n, 1))
    dA = scale * a + noise * rng.normal(size=(n, D))
    dB = scale * b + noise * rng.normal(size=(n, D))
    return a, b, dA, dB


def test_principal_angle_recovers_planted_angle():
    for angle in (20, 60, 85):
        _, _, dA, dB = two_tactics(angle)
        got = g.principal_angles(g.tactic_subspace(dA, 1), g.tactic_subspace(dB, 1))
        assert got.shape == (1,) and abs(got[0] - angle) < 2.0


def test_principal_angles_of_identical_and_orthogonal_subspaces():
    I = np.eye(D)
    np.testing.assert_allclose(g.principal_angles(I[:, :2], I[:, :2]), [0, 0], atol=1e-6)
    np.testing.assert_allclose(g.principal_angles(I[:, :2], I[:, 2:4]), [90, 90], atol=1e-6)


def test_mean_displacements_and_cosine_matrix():
    a, b, dA, dB = two_tactics(60)
    delta = np.vstack([dA, dB])
    tac = ["A"] * len(dA) + ["B"] * len(dB)
    m = g.mean_displacements(delta, tac)
    C = g.cosine_matrix(np.array([m["A"], m["B"]]))
    assert abs(C[0, 1] - 0.5) < 0.02 and np.allclose(np.diag(C), 1)
    assert g.mean_displacements(delta, tac, mask=np.array(tac) == "A").keys() == {"A"}


def test_cluster_order_groups_similar_rows():
    v = np.array([[1, 0, 0], [0, 1, 0], [0.99, 0.1, 0], [0, 0.98, 0.1]], float)
    order = g.cluster_order(g.cosine_matrix(v))
    pos = {k: order.index(k) for k in range(4)}
    assert abs(pos[0] - pos[2]) == 1 and abs(pos[1] - pos[3]) == 1


def test_pca_2d_finds_the_spread():
    rng = np.random.default_rng(0)
    pts = np.outer(rng.normal(size=50), np.eye(D)[0]) * 10 + rng.normal(size=(50, D)) * 0.1
    coords, share = g.pca_2d(pts)
    assert coords.shape == (50, 2) and share[0] > 0.9


def test_knn_composition_same_tactic_neighbours():
    _, _, dA, dB = two_tactics(80)
    comp = g.knn_composition(np.vstack([dA, dB]), ["A"] * len(dA) + ["B"] * len(dB), k=10)
    assert comp.loc["A", "A"] > 0.95 and comp.loc["B", "B"] > 0.95
    np.testing.assert_allclose(comp.sum(axis=1), 1.0)


def test_effective_rank():
    assert g.effective_rank([1, 1]) == pytest.approx(2.0)
    assert g.effective_rank([5, 0, 0]) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    # displacements living in a 2-D plane with equal spread -> effective rank about 2
    plane = np.linalg.qr(rng.normal(size=(D, 2)))[0]
    coef = rng.normal(size=(4000, 2))
    s = g.singular_values(coef @ plane.T)
    assert abs(g.effective_rank(s) - 2.0) < 0.1
    assert s[0] >= s[1] >= s[2] and s[2] < 1e-8 * s[0]
    assert g.energy_rank(s, 0.9) == 2
    assert g.energy_rank([3, 1, 0], 0.9) == 1 and g.energy_rank([1, 1, 1, 1], 0.9) == 4


def test_style_content_split_recovers_planted_values():
    syn = g.synthetic_ws3(n_pairs=800, style=-0.8, content=-1.2, seed=1)
    a2 = g.a2_projection(syn["Hv"], syn["Ha"], syn["meta"])
    sc = g.style_content(a2["dp"], syn["meta"]["tactic"], syn["meta"]["intent"])
    pooled = sc["_pooled"]
    # r_hat is fitted from noisy data, so allow 10%; the CI must cover the truth
    assert abs(pooled["style"][0] - (-0.8)) < 0.1 and abs(pooled["content"][0] - (-1.2)) < 0.12
    for t in syn["W"]:
        st, lo, hi = sc[t]["style"]
        assert lo < st < hi and abs(st + 0.8) < 0.25


def test_style_content_skips_tactics_without_both_intents():
    dp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = g.style_content(dp, ["t"] * 5, ["harmful"] * 5)
    assert out == {}


def test_angle_between_and_refusal_direction():
    assert g.angle_between([1, 0], [0, 1]) == pytest.approx(90)
    assert g.angle_between([1, 0], [-1, 0]) == pytest.approx(180)
    rng = np.random.default_rng(0)
    v = rng.normal(size=D); v /= np.linalg.norm(v)
    refused = np.arange(400) % 2 == 0
    H = rng.normal(size=(400, D)) + np.where(refused[:, None], 1.5, -1.5) * v
    assert g.angle_between(g.refusal_direction(H, refused), v) < 10


def test_direction_stability_recovers_rotation():
    syn = g.synthetic_ws3(n_pairs=600, configs={"R": 0, "fp16": 3, "nf4": 30}, seed=2)
    out = g.direction_stability(syn["by_config"], syn["meta"], reference="R")
    assert out["R"]["cos_to_reference"] == pytest.approx(1.0)
    assert out["fp16"]["cos_to_reference"] > 0.99
    assert abs(out["nf4"]["cos_to_reference"] - np.cos(np.radians(30))) < 0.05
    # dp is rotation-invariant here (the whole space rotates), so the masking effect keeps its size
    assert abs(out["nf4"]["dp"]["harmful"][0] - out["R"]["dp"]["harmful"][0]) < 1e-6


def test_a2_projection_sign_and_split_discipline():
    syn = g.synthetic_ws3(n_pairs=400, style=-0.5, content=-1.0, seed=3)
    a2 = g.a2_projection(syn["Hv"], syn["Ha"], syn["meta"])
    assert a2["r_hat"] @ syn["u"] > 0.95
    assert a2["overall"]["harmful"][0] < a2["overall"]["benign"][0] < 0
    assert set(a2["per_tactic"]) == set(syn["W"])


def test_ws3_figures_draw(tmp_path):
    from analysis import figures as F
    paths = F.render_all_synthetic(tmp_path)
    assert len(paths) >= 18
    for p in paths:
        assert p.exists() and p.stat().st_size > 5_000, p
