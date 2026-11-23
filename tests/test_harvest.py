"""Tests for lib.load_harvest / check_harvest / join_pairs / build_2x2 on SYNTHETIC shards (random numbers, no text)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from analysis import lib  # noqa: E402
from fake_shards import make_fake_harvest  # noqa: E402

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_load_harvest_shapes_and_alignment(tmp_path):
    ids = make_fake_harvest(tmp_path, n_shards=3, prompts_per_shard=4, n_layers=6, hidden=8)
    acts, index = lib.load_harvest(tmp_path, "last_prefill")
    assert acts.shape == (12, 6, 8) and acts.dtype == np.float32
    assert index["prompt_id"].tolist() == ids  # row i of acts is prompt i, across shards in order
    assert index.attrs["layers"] == list(range(6)) and index.attrs["point"] == "last_prefill"
    acts2, _ = lib.load_harvest(tmp_path, "last_instruction")
    assert acts2.shape == (12, 3, 8)  # every 2nd layer: 1, 3, 5


def test_load_harvest_rows_really_match_the_file(tmp_path):
    """The value read for (prompt, layer) must be the one written for that prompt, not a shuffled row."""
    from safetensors import safe_open
    make_fake_harvest(tmp_path, n_shards=2, prompts_per_shard=3, n_layers=4, hidden=5)
    acts, index = lib.load_harvest(tmp_path, "last_prefill", layers=[2])
    fw = lib._tensor_reader()
    with safe_open(str(tmp_path / "shard_00001.safetensors"), framework=fw) as f:
        t = f.get_tensor("last_prefill/layer_2")
        t = t.float().numpy() if fw == "pt" else np.asarray(t, dtype=np.float32)
    np.testing.assert_array_equal(acts[3:, 0, :], t)


def test_load_harvest_missing_layer_and_point(tmp_path):
    make_fake_harvest(tmp_path, n_layers=4)
    with pytest.raises(KeyError):
        lib.load_harvest(tmp_path, "last_prefill", layers=[99])
    with pytest.raises(ValueError, match="no tensors for point"):
        lib.load_harvest(tmp_path, "not_a_point")
    with pytest.raises(FileNotFoundError):
        lib.load_harvest(tmp_path / "empty")


def test_check_harvest_clean(tmp_path, capsys):
    make_fake_harvest(tmp_path, n_shards=2, prompts_per_shard=5, n_layers=8, hidden=16)
    rep = lib.check_harvest(tmp_path)
    out = capsys.readouterr().out
    assert "shards found: 2" in out and "prompts: 10" in out
    assert rep["n_prompts"] == 10 and rep["hidden"] == {16}
    assert sum(rep["nan_inf"].values()) == 0
    assert set(rep["points"]) == set(lib.EXPECTED_SCHEMA["points"])
    failing = [f for f, ok, _ in rep["schema"] if not ok]
    assert failing == ([] if HAS_TORCH else ["dtype bfloat16"])  # without torch the fake shards are float32


def test_check_harvest_reports_problems_without_crashing(tmp_path, capsys):
    make_fake_harvest(tmp_path, nan_at=("last_prefill", 3), truncate_rows=(1,))
    (tmp_path / "shard_00001.parquet").unlink()                  # an index is missing
    rep = lib.check_harvest(tmp_path)
    out = capsys.readouterr().out
    assert rep["nan_inf"][("last_prefill", 3)] == 1
    assert sum(rep["truncated"]) == 1
    assert "MISMATCH one parquet index per shard" in out
    assert "MISMATCH no NaN / inf values" in out


def test_check_harvest_unexpected_format(tmp_path, capsys):
    from safetensors.numpy import save_file
    save_file({"model.layers.3.weird": np.zeros((2, 4), np.float32)}, str(tmp_path / "x.safetensors"))
    pd.DataFrame({"id": ["a", "b"]}).to_parquet(tmp_path / "x.parquet")
    rep = lib.check_harvest(tmp_path)            # must not raise
    out = capsys.readouterr().out
    assert "tensor_key_regex" in out and "model.layers.3.weird" in out
    rep2 = lib.check_harvest(tmp_path / "does-not-exist")
    assert rep2["problems"] and rep2["n_prompts"] == 0
    rep3 = lib.check_harvest("")                 # empty path is a problem, not the current folder
    assert rep3["problems"] and rep3["shards"] == []


def test_join_and_2x2(tmp_path):
    pair_ids = [f"wj{i:07d}" for i in range(6)]
    make_fake_harvest(tmp_path, n_shards=2, prompts_per_shard=6, pair_ids=pair_ids)
    acts, index = lib.load_harvest(tmp_path, "last_prefill")
    pairs = pd.DataFrame([dict(pair_id=p, role=r, intent="harmful" if k % 2 == 0 else "benign", tactic="t",
                               split=lib.assign_split(p)) for k, p in enumerate(pair_ids[:5]) for r in ("vanilla", "adversarial")])
    joined = lib.join_pairs(index, pairs)
    assert len(joined) == len(index) and joined["pair_id"].notna().sum() == 10  # last pair not in the table
    Hv, Ha, meta = lib.build_2x2(acts, joined)
    assert Hv.shape == (5, acts.shape[1], acts.shape[2]) and list(meta["pair_id"]) == sorted(pair_ids[:5])
    k = index["prompt_id"].tolist().index(f"{pair_ids[2]}|adversarial")
    np.testing.assert_array_equal(Ha[2], acts[k])
    assert set(meta["intent"]) == {"harmful", "benign"}


def test_ws3_pairs_from_harvest_and_json(tmp_path):
    from analysis import ws3_data
    pair_ids = [f"wj{i:07d}" for i in range(8)]
    make_fake_harvest(tmp_path / "h", n_shards=2, prompts_per_shard=8, n_layers=6, hidden=10, pair_ids=pair_ids)
    pairs = pd.DataFrame([dict(pair_id=p, role=r, intent="harmful" if k < 4 else "benign", split=lib.assign_split(p))
                          for k, p in enumerate(pair_ids) for r in ("vanilla", "adversarial")])
    Hv, Ha, meta, info = ws3_data.pairs_from_harvest(tmp_path / "h", pairs, point="last_prefill", layer=None)
    assert Hv.shape == (8, 10) and Ha.shape == (8, 10) and info["layer"] == 5 and info["n_pairs"] == 8
    assert (meta["tactic"] == "unknown").all()          # no tactic column in this pairs table
    with pytest.raises(KeyError):
        ws3_data.pairs_from_harvest(tmp_path / "h", pairs, point="last_instruction", layer=2)  # only odd layers stored
    p = ws3_data.save_json({"a": np.float32(1.5), "b": [np.nan, 2], "c": np.arange(3)}, tmp_path / "r.json")
    import json
    assert json.loads(p.read_text()) == {"a": 1.5, "b": [None, 2], "c": [0, 1, 2]}


def test_harvest_figures_draw(tmp_path):
    from analysis import harvest_figures as hf
    make_fake_harvest(tmp_path / "h", n_shards=3, prompts_per_shard=8, truncate_rows=(2,))
    rep = lib.check_harvest(tmp_path / "h", verbose=False)
    for p in (hf.draw_data_shape(tmp_path / "shape.png"), hf.draw_schema_check(rep, tmp_path / "schema.png", "SYNTHETIC"),
              hf.draw_health(rep, tmp_path / "health.png", "SYNTHETIC")):
        assert p.exists() and p.stat().st_size > 10_000
