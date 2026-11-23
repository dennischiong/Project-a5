"""Tests for the pilot tactics fallback, the tactics map hook, the noise floor summary and the Task 5 figures.
Tiny synthetic data only: ids, labels and random numbers, never real prompts. No GPU, no download."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis import lib  # noqa: E402
from harvest import pilot_extract as pe  # noqa: E402

WJ_COLS = ["vanilla", "adversarial", "completion", "data_type"]  # WildJailbreak's real header: no 'tactics'
STOP_MSG = ("STOP: required column(s) missing: ['tactics']. "
            "WildJailbreak gives no per-row tactic labels; ask the mentor/Dennis where tactics come from.")


def fake_rows(n=40, tactics=None):
    """Synthetic dataset rows: placeholder strings, never real prompts."""
    df = pd.DataFrame({"source_row_id": range(n), "vanilla": [f"v{i}" for i in range(n)],
                       "adversarial": [f"a{i}" for i in range(n)], "data_type": "adversarial_harmful"})
    df["tactic"] = tactics if tactics is not None else "unknown"
    return df


# ------------------------------------------------------------------------------- where tactics come from
def test_missing_tactics_without_flag_stops_with_the_old_message(tmp_path):
    with pytest.raises(SystemExit) as exc:
        pe.tactics_source_for(WJ_COLS, allow_missing_tactics=False, map_path=tmp_path / "none.parquet")
    assert str(exc.value) == STOP_MSG


def test_missing_tactics_with_flag_is_the_pilot_override(tmp_path, capsys):
    src = pe.tactics_source_for(WJ_COLS, allow_missing_tactics=True, map_path=tmp_path / "none.parquet")
    assert src == pe.SRC_OVERRIDE == "missing (pilot override)"
    assert capsys.readouterr().out.count("WARNING") == 1


def test_missing_base_column_always_stops_even_with_flag(tmp_path):
    with pytest.raises(SystemExit, match="vanilla"):
        pe.tactics_source_for(["adversarial", "data_type"], allow_missing_tactics=True, map_path=tmp_path / "x.parquet")


def test_tactics_column_is_used_when_present(tmp_path):
    assert pe.tactics_source_for(WJ_COLS + ["tactics"], map_path=tmp_path / "none.parquet") == pe.SRC_COLUMN


def test_map_file_wins_when_it_exists(tmp_path):
    p = tmp_path / "tactics_map.parquet"
    pd.DataFrame({"source_row_id": [0], "tactic": ["roleplay"]}).to_parquet(p)
    assert pe.tactics_source_for(WJ_COLS, allow_missing_tactics=False, map_path=p) == pe.SRC_MAP


# ------------------------------------------------------------------------------- the tactics map hook
def test_tactics_map_joins_on_the_key_and_fills_unknown(tmp_path):
    p = tmp_path / "tactics_map.parquet"
    pd.DataFrame({"source_row_id": [1, 3, 5], "tactic": ["roleplay", "['persona', 'fiction']", "obfuscation"]}).to_parquet(p)
    df = fake_rows(8).drop(columns="tactic")
    out = pe.assign_tactics(df, pe.SRC_MAP, map_path=p)
    assert out.loc[out.source_row_id == 1, "tactic"].item() == "roleplay"
    assert out.loc[out.source_row_id == 3, "tactic"].item() == "persona"  # first listed tactic
    assert out.loc[out.source_row_id == 5, "tactic"].item() == "obfuscation"
    assert (out.loc[~out.source_row_id.isin([1, 3, 5]), "tactic"] == "unknown").all()
    assert "tactic" not in df.columns  # input not modified


def test_tactics_map_accepts_list_values_and_another_key(tmp_path):
    p = tmp_path / "m.parquet"
    pd.DataFrame({"row_key": ["k0", "k1"], "tactic": [["a", "b"], []]}).to_parquet(p)
    m = pe.load_tactics_map(p, key="row_key")
    assert m == {"k0": "a", "k1": "unknown"}


@pytest.mark.parametrize("table, match", [
    (pd.DataFrame({"source_row_id": [0], "label": ["x"]}), "missing column"),
    (pd.DataFrame({"source_row_id": [0, 0], "tactic": ["x", "y"]}), "duplicated"),
])
def test_tactics_map_bad_files_stop(tmp_path, table, match):
    p = tmp_path / "bad.parquet"
    table.to_parquet(p)
    with pytest.raises(SystemExit, match=match):
        pe.load_tactics_map(p)


def test_override_assigns_unknown_everywhere():
    out = pe.assign_tactics(fake_rows(5).drop(columns="tactic"), pe.SRC_OVERRIDE)
    assert (out["tactic"] == "unknown").all()


# ------------------------------------------------------------------------------- sampling
def test_fallback_sampling_is_plain_seeded_random():
    sub = fake_rows(40)
    a = pe.sample_spread(sub, 10, np.random.default_rng(0))
    b = pe.sample_spread(sub, 10, np.random.default_rng(0))
    c = pe.sample_spread(sub, 10, np.random.default_rng(1))
    assert len(a) == 10 and a["source_row_id"].is_unique
    assert list(a["source_row_id"]) == list(b["source_row_id"])  # same seed, same sample
    assert list(a["source_row_id"]) != list(c["source_row_id"])  # different seed, different sample
    expected = sub.sample(n=10, random_state=int(np.random.default_rng(0).integers(1 << 31)))
    assert list(a["source_row_id"]) == list(expected["source_row_id"])  # exactly pandas' seeded sample


def test_fallback_sampling_caps_at_available_rows():
    assert len(pe.sample_spread(fake_rows(6), 10, np.random.default_rng(0))) == 6


def test_round_robin_still_spreads_over_real_tactics():
    tactics = ["t1"] * 30 + ["t2"] * 5 + ["t3"] * 5
    out = pe.sample_spread(fake_rows(40, tactics), 9, np.random.default_rng(0))
    assert out["tactic"].value_counts().to_dict() == {"t1": 3, "t2": 3, "t3": 3}


# ------------------------------------------------------------------------------- validate_pairs switch
def unknown_pairs():
    rows = []
    for i in range(4):
        for role in ("vanilla", "adversarial"):
            rows.append(dict(pair_id=f"q{i}", role=role, intent="harmful" if i % 2 == 0 else "benign",
                             tactic="unknown", token_count=20, text="placeholder", split="train"))
    return pd.DataFrame(rows)


def test_validate_pairs_unknown_tactics_only_allowed_with_switch():
    with pytest.raises(ValueError, match="unknown"):
        lib.validate_pairs(unknown_pairs())
    lib.validate_pairs(unknown_pairs(), allow_unknown_tactics=True)  # pilot override


# ------------------------------------------------------------------------------- noise floor summary
def test_noise_floor_summary_per_view():
    rng = np.random.default_rng(0)
    n, d = 20, 8
    base = {v: rng.normal(size=(n + 12, d)) for v in pe.NOISE_VIEWS}
    singles = [{v: base[v][i:i + 1] for v in base} for i in range(n)]
    # batch copy: postnorm identical, layer_minus2 slightly perturbed, prenorm more perturbed
    noise = {"postnorm_last": 0.0, "layer_minus2": 1e-3, "prenorm_last": 1e-1}
    full = {v: base[v] + noise[v] * rng.normal(size=base[v].shape) for v in base}
    batches = [{v: full[v][i:i + 16] for v in full} for i in range(0, 32, 16)]
    out = pe.noise_floor_summary(singles, batches, n)
    assert set(out) == set(pe.NOISE_VIEWS)
    assert out["postnorm_last"]["relative_l2"]["max"] == 0.0
    assert 0 < out["layer_minus2"]["relative_l2"]["mean"] < out["prenorm_last"]["relative_l2"]["mean"]
    lo, hi = out["prenorm_last"]["cosine_distance"]["ci95"]
    assert lo <= out["prenorm_last"]["cosine_distance"]["mean"] <= hi


# ------------------------------------------------------------------------------- pilot_analysis tactic check
def test_has_tactic_labels():
    from analysis import pilot_analysis as pa
    assert not pa.has_tactic_labels(["unknown", "unknown"])
    assert pa.has_tactic_labels(["unknown", "roleplay"])


def test_floor_for_old_and_new_files():
    from analysis import pilot_analysis as pa
    old = {"cosine_distance": {"mean": 1}, "relative_l2": {"mean": 2}}
    assert pa.floor_for(old, "postnorm_last") is old and pa.floor_for(old, "prenorm_last") is None
    new = dict(old, per_view={"prenorm_last": {"cosine_distance": {"mean": 3}}})
    assert pa.floor_for(new, "prenorm_last")["cosine_distance"]["mean"] == 3


# ------------------------------------------------------------------------------- Task 5 figures draw
def test_task5_figures_draw(tmp_path):
    from analysis import pilot_figures as pf
    for p in (pf.draw_extraction_flow(tmp_path / "flow.png"),
              pf.draw_pilot_glance(pf.synthetic_pairs(), tmp_path / "glance.png", kind="SYNTHETIC"),
              pf.draw_noise_floor(pf.synthetic_noise_floor(), tmp_path / "noise.png", kind="SYNTHETIC"),
              pf.draw_noise_floor({"n": 5, "cosine_distance": {"mean": 1e-5, "max": 3e-5},   # old file format
                                   "relative_l2": {"mean": 1e-3, "max": 2e-3}}, tmp_path / "old.png"),
              pf.draw_noise_floor({"n": 5, "per_view": {                                    # bit-identical runs: zeros
                  "postnorm_last": {k: {"mean": 0.0, "max": 0.0, "ci95": [0.0, 0.0]} for k in ("cosine_distance", "relative_l2")},
                  "prenorm_last": {k: {"mean": 1e-4, "max": 3e-4, "ci95": [0.0, 2e-4]} for k in ("cosine_distance", "relative_l2")}}},
                  tmp_path / "zeros.png"),
              pf.draw_archive_sizes({"name": "x.tpxz", "n_files": 8, "raw_bytes": 40e6, "archive_bytes": 30e6,
                                     "sha256": "ab" * 32, "pixz_level": 1}, tmp_path / "sizes.png", kind="SYNTHETIC")):
        assert p.exists() and p.stat().st_size > 10_000
