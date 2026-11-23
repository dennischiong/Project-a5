"""Tests for analysis/lib.py using synthetic data only (no GPU, no real prompts)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis import lib  # noqa: E402

D = 64


def make_synthetic(n_pairs=2000, shift=-3.0, seed=0):
    """Vanilla harmful/benign differ along a planted unit direction u (+/-2 on u); the
    adversarial copy of each pair is the vanilla vector moved by ``shift`` along u plus tiny noise."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=D)
    u /= np.linalg.norm(u)
    rows, h_van, h_adv = [], [], []
    for i in range(n_pairs):
        harmful = i % 2 == 0
        base = rng.normal(size=D) + (2.0 if harmful else -2.0) * u
        pid = f"p{i:04d}"
        rows.append(dict(pair_id=pid, intent="harmful" if harmful else "benign", split=lib.assign_split(pid)))
        h_van.append(base)
        h_adv.append(base + shift * u + 0.01 * rng.normal(size=D))
    return u, pd.DataFrame(rows), np.array(h_van), np.array(h_adv)


def test_direction_recovered_with_right_sign_and_dp():
    u, meta, hv, ha = make_synthetic(shift=-3.0)
    tr = (meta["split"] == "train").to_numpy()
    m = meta[tr].assign(role="vanilla")
    r_hat = lib.fit_direction_train(hv[tr], m)
    assert abs(np.linalg.norm(r_hat) - 1) < 1e-9
    assert r_hat @ u > 0.95  # cosine, and positive = harmful side
    dp = lib.project(ha - hv, r_hat)
    # planted shift -3.0 along u; recovered direction is slightly off u, so allow 10%
    assert np.all(dp < 0)
    assert abs(dp.mean() - (-3.0)) < 0.3
    # vanilla harmful must project higher than vanilla benign (sign check)
    p = lib.project(hv, r_hat)
    assert p[(meta.intent == "harmful").to_numpy()].mean() > p[(meta.intent == "benign").to_numpy()].mean()


def test_positive_shift_gives_positive_dp():
    u, meta, hv, ha = make_synthetic(shift=+2.0)
    r_hat = lib.fit_direction(hv[(meta.intent == "harmful").to_numpy()], hv[(meta.intent == "benign").to_numpy()])
    assert lib.project(ha - hv, r_hat).mean() > 1.5


def test_fit_refuses_non_train_rows():
    _, meta, hv, _ = make_synthetic(n_pairs=100)
    meta = meta.assign(role="vanilla")
    with pytest.raises(ValueError):
        lib.fit_direction_train(hv, meta)  # contains validation/test rows
    train_only = meta[meta.split == "train"]
    lib.fit_direction_train(hv[(meta.split == "train").to_numpy()], train_only)  # ok
    adv_rows = train_only.assign(role="adversarial")
    with pytest.raises(ValueError):
        lib.fit_direction_train(hv[(meta.split == "train").to_numpy()], adv_rows)


def test_split_stable_twins_and_proportions():
    assert lib.assign_split("abc") == lib.assign_split("abc")
    # pinned values guard against an accidental change of the hash recipe (sha256, first 8 bytes, 70/15/15)
    import hashlib
    for pid in ["abc", "wj0000001", "p0042"]:
        u = int.from_bytes(hashlib.sha256(pid.encode()).digest()[:8], "big") / 2**64
        expected = "train" if u < 0.70 else ("validation" if u < 0.85 else "test")
        assert lib.assign_split(pid) == expected
    # and it must not depend on Python's per-process hash() salt: same answer in a fresh interpreter
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); from analysis import lib; print(lib.assign_split('wj0000001'))"
            % os.path.join(os.path.dirname(__file__), ".."))
    fresh = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": "123"}).stdout.strip()
    assert fresh == lib.assign_split("wj0000001")
    ids = [f"id{i}" for i in range(20000)]
    splits = pd.Series([lib.assign_split(i) for i in ids])
    frac = splits.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.02
    assert abs(frac["validation"] - 0.15) < 0.02
    assert abs(frac["test"] - 0.15) < 0.02
    # twins: both rows of a pair are built from the same pair_id
    df = pd.DataFrame({"pair_id": ids[:50] * 2, "role": ["vanilla"] * 50 + ["adversarial"] * 50})
    df["split"] = df.pair_id.map(lib.assign_split)
    assert (df.groupby("pair_id")["split"].nunique() == 1).all()


def test_distances_known_values():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(lib.cosine_distance(a, b), [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(lib.relative_l2(a, b), [np.sqrt(2) / 1.0, 1.0 / 2.0])


def test_bootstrap_ci_covers_mean_and_is_seeded():
    rng = np.random.default_rng(1)
    x = rng.normal(5.0, 1.0, size=200)
    est, lo, hi = lib.bootstrap_ci(x, np.mean, n=2000, seed=0)
    assert lo < est < hi and lo < 5.0 < hi
    assert (hi - lo) < 0.6  # ~ 2*1.96*1/sqrt(200)=0.28
    assert lib.bootstrap_ci(x, np.mean, n=2000, seed=0) == (est, lo, hi)


def test_bootstrap_resamples_whole_pairs():
    """values hold ONE number per pair; every resample must be m whole entries of that list, drawn with
    replacement (so a pair's value is never split or invented)."""
    vals = np.arange(10, dtype=float)  # distinct per-pair values
    seen = []
    est, lo, hi = lib.bootstrap_ci(vals, lambda v: (seen.append(v.copy()), v.mean())[1], n=200, seed=0)
    # one call is the point estimate on the original array, the other n are resamples; with 10 distinct values a
    # resample equal to the original in the same order has probability 1e-10, so this split is safe
    boots = [s for s in seen if not np.array_equal(s, vals)]
    assert len(boots) == 200 and all(len(b) == len(vals) for b in boots)
    assert all(set(b) <= set(vals) for b in boots)
    assert any(len(set(b)) < len(vals) for b in boots)  # with replacement: some values repeat
    assert est == 4.5 and lo <= 4.5 <= hi


def test_cohens_d():
    rng = np.random.default_rng(2)
    a, b = rng.normal(1.0, 1.0, 5000), rng.normal(0.0, 1.0, 5000)
    assert abs(lib.cohens_d(a, b) - 1.0) < 0.1
    assert abs(lib.cohens_d(np.array([1.0, 2.0, 3.0]))) == pytest.approx(2.0)  # mean 2 / sd 1


def test_auroc_per_group_na_rule():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.5, 0.5])
    g = np.array(["a"] * 6 + ["b"] * 2)
    out = lib.auroc_per_group(y, s, g, min_per_class=3)
    assert out["a"] == 1.0
    assert np.isnan(out["b"])


def good_pairs():
    rows = []
    for i in range(6):
        pid = f"q{i}"
        intent = "harmful" if i % 2 == 0 else "benign"
        for role in ("vanilla", "adversarial"):
            rows.append(dict(pair_id=pid, role=role, intent=intent, tactic="roleplay" if role == "adversarial" else "",
                             token_count=20, text="some text", split="train"))
    return pd.DataFrame(rows)


def test_validate_pairs_accepts_good_table():
    lib.validate_pairs(good_pairs())


@pytest.mark.parametrize("mutate", [
    lambda d: d.__setitem__("text", d["text"].where(~((d.role == "adversarial") & (d.pair_id == "q0")), "")),
    lambda d: d.__setitem__("text", d["text"].where(~((d.role == "adversarial") & (d.pair_id == "q0")), "None")),
    lambda d: d.__setitem__("text", d["text"].where(~((d.role == "adversarial") & (d.pair_id == "q0")), np.nan)),
    lambda d: d.__setitem__("token_count", d["token_count"].where(~((d.role == "adversarial") & (d.pair_id == "q1")), 1)),
])
def test_validate_pairs_row_failures(mutate):
    df = good_pairs()
    mutate(df)
    with pytest.raises(ValueError):
        lib.validate_pairs(df)


def test_validate_pairs_error_never_contains_text():
    secret = "SENTINEL-PROMPT-TEXT-123"
    df = good_pairs()
    df["text"] = secret
    df.loc[(df.role == "adversarial") & (df.pair_id == "q0"), "token_count"] = 1  # force a failure
    with pytest.raises(ValueError) as exc:
        lib.validate_pairs(df)
    assert secret not in str(exc.value)


def test_validate_pairs_missing_twin():
    df = good_pairs()
    df = df[~((df.pair_id == "q2") & (df.role == "vanilla"))]
    with pytest.raises(ValueError, match="exactly one"):
        lib.validate_pairs(df)


def test_validate_pairs_duplicate_role():
    df = good_pairs()
    df = pd.concat([df, df[(df.pair_id == "q3") & (df.role == "vanilla")]])
    with pytest.raises(ValueError, match="duplicate"):
        lib.validate_pairs(df)


def test_validate_pairs_all_harmful_unknown_tactic():
    df = good_pairs()
    df.loc[(df.role == "adversarial") & (df.intent == "harmful"), "tactic"] = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        lib.validate_pairs(df)
    # benign rows being 'unknown' alone must NOT trigger it
    df2 = good_pairs()
    df2.loc[(df2.role == "adversarial") & (df2.intent == "benign"), "tactic"] = "unknown"
    lib.validate_pairs(df2)


def test_archive_name_matches_work_plan_format():
    """Work plan Appendix A: model__dataset__config__spechash[:8].tpxz (harvest/drive_archive.py, the team copy)."""
    from harvest import drive_archive as da

    name = da.archive_name("llama31-8b", "sorry-bench", "vllm-bf16-c64", "1a2b3c4d5e6f")
    assert name == "llama31-8b__sorry-bench__vllm-bf16-c64__1a2b3c4d.tpxz"


def test_load_activations_roundtrip_and_nan_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    good = {"a|vanilla": torch.randn(5, 8).bfloat16(), "a|adversarial": torch.randn(5, 8).bfloat16()}
    p = str(tmp_path / "ok.safetensors")
    save_file(good, p)
    out = lib.load_activations(p, n_layers_plus_1=5, hidden=8)
    assert out["a|vanilla"].shape == (5, 8) and out["a|vanilla"].dtype == np.float32
    with pytest.raises(AssertionError):
        lib.load_activations(p, n_layers_plus_1=6)

    bad = {"a|vanilla": torch.full((5, 8), float("nan"))}
    p2 = str(tmp_path / "bad.safetensors")
    save_file(bad, p2)
    with pytest.raises(AssertionError, match="NaN"):
        lib.load_activations(p2)
