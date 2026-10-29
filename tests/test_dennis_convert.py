"""Tests for harvest/dennis_harvest_convert.py on SYNTHETIC shards with a planted answer (no model, no GPU, no pixz,
no real prompts): fake index.parquet + activations.safetensors in Dennis's layout, a fake pair map and pairs table."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis import lib  # noqa: E402
from analysis import ws3_pilot as wp  # noqa: E402
from harvest import dennis_harvest_convert as dc  # noqa: E402

D = 24


def write_shard(folder, prompt_ids, X, skipped=None):
    """One restored shard in Dennis's layout: tensor row i = index row i, fp16, the keys the real shards use."""
    from safetensors.numpy import save_file
    folder.mkdir(parents=True)
    X16 = np.asarray(X, np.float16)
    tensors = {f"last_prefill.layer_{k}": X16 for k in (7, 27)}
    tensors["last_instruction.layer_27"] = X16 * 0
    save_file(tensors, str(folder / "activations.safetensors"),
              metadata={"residual_stream": "decoder-layer output, pre final norm", "model": "synthetic"})
    pd.DataFrame({"prompt_id": prompt_ids, "token_count": 10, "dataset": "x", "backend": "hf", "concurrency": 1,
                  "dtype": "fp16", "quantization": "nf4", "spec_hash": "e7f5ea94",
                  "skipped_too_long": skipped if skipped is not None else False}).to_parquet(folder / "index.parquet")
    return folder


def fake_world(tmp_path, seed=0):
    """u = harmfulness axis. WildJailbreak: harmful bases +2u, benign -2u, rewrites move -1u. SORRY-Bench (harmful,
    one shared base per 3 rows): 'towards_benign' moves -3u, 'sideways' moves along w (orthogonal to u), 'base' is
    the same prompt on both sides (no move)."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=D)
    u /= np.linalg.norm(u)
    w = rng.normal(size=D)
    w -= (w @ u) * u
    w /= np.linalg.norm(w)
    vec, table, pm = {}, [], []

    def prompt(v):
        pid = f"p{len(vec):05d}"
        vec[pid] = v
        return pid

    for i in range(160):
        intent = "harmful" if i % 2 == 0 else "benign"
        pid = f"wj_{i:04d}"
        base = (2.0 if intent == "harmful" else -2.0) * u + 0.4 * rng.normal(size=D)
        pv, pa = prompt(base), prompt(base - u + 0.2 * rng.normal(size=D))
        table.append(dict(pair_id=pid, source="WildJailbreak", intent=intent, tactic="wildjailbreak",
                          split=lib.assign_split(pid), holdout_group="default"))
        pm += [dict(pair_id=pid, role="vanilla", prompt_id=pv), dict(pair_id=pid, role="adv", prompt_id=pa)]
    for b in range(40):
        pv = prompt(2.0 * u + 0.4 * rng.normal(size=D))
        for t, shift in (("towards_benign", -3.0 * u), ("sideways", 3.0 * w), ("base", None)):
            pid = f"sb_{b:03d}_{t}"
            pa = pv if shift is None else prompt(vec[pv] + shift + 0.2 * rng.normal(size=D))
            table.append(dict(pair_id=pid, source="SORRY-Bench", intent="harmful", tactic=t,
                              split=lib.assign_split(pid), holdout_group="loto_heldout"))
            pm += [dict(pair_id=pid, role="vanilla", prompt_id=pv), dict(pair_id=pid, role="adv", prompt_id=pa)]
    table.append(dict(pair_id="xs_0001", source="XSTest", intent="benign", tactic="x", split="test",
                      holdout_group="default"))                                   # other source: ignored
    table.append(dict(pair_id="wj_lost", source="WildJailbreak", intent="harmful", tactic="wildjailbreak",
                      split="test", holdout_group="default"))                     # in the table, not in the map
    pm += [dict(pair_id="wj_nohv", role="vanilla", prompt_id="p_never_harvested"),
           dict(pair_id="wj_nohv", role="adv", prompt_id="p00001")]
    table.append(dict(pair_id="wj_nohv", source="WildJailbreak", intent="harmful", tactic="wildjailbreak",
                      split="test", holdout_group="default"))                     # a prompt no shard holds
    ids = sorted(vec)
    X = np.array([vec[i] for i in ids])
    half = len(ids) // 2
    shard_dirs = [write_shard(tmp_path / "s0", ids[:half], X[:half]),
                  write_shard(tmp_path / "s1", ids[half:], X[half:])]
    return u, vec, pd.DataFrame(table), pd.DataFrame(pm), shard_dirs


def test_archive_dataset_parses_real_names():
    assert dc.archive_dataset("qwen25-7b__sorry-bench-s00018__hf-nf4-c6__e7f5ea94.tpxz") == "sorry-bench"
    assert dc.archive_dataset("qwen25-7b__wildjailbreak-s00000__hf-nf4-c6__e7f5ea94.tpxz") == "wildjailbreak"


def test_plan_pairs_drops_unmapped_and_identical(tmp_path):
    _, _, table, pm, _ = fake_world(tmp_path)
    plan, counts = dc.plan_pairs(pm, table, ["WildJailbreak", "SORRY-Bench"])
    assert counts == dict(table_pairs=162 + 120, not_in_map=1, identical_prompts=40)
    assert set(plan["source"]) == {"WildJailbreak", "SORRY-Bench"} and "base" not in set(plan["tactic"])
    assert not any("prompt" in c and c not in ("prompt_v", "prompt_a") for c in plan.columns)   # no text column


def test_read_shard_matches_rows_and_flags_skipped(tmp_path):
    X = np.arange(3 * D, dtype=float).reshape(3, D)
    folder = write_shard(tmp_path / "s", ["a", "b", "c"], X, skipped=[False, True, False])
    rows, V, serving = dc.read_shard(folder)
    assert list(rows["prompt_id"]) == ["a", "b", "c"] and rows["finite"].tolist() == [True, False, True]
    assert V.dtype == np.float32 and np.allclose(V, X.astype(np.float16).astype(np.float32))
    assert serving["quantization"] == "nf4" and serving["residual_stream"].startswith("decoder-layer output")


def test_build_folder_rows_line_up_and_load_works(tmp_path):
    u, vec, table, pm, shard_dirs = fake_world(tmp_path)
    plan, _ = dc.plan_pairs(pm, table, ["WildJailbreak", "SORRY-Bench"])
    out = tmp_path / "out"
    meta = dc.build_folder(plan, [dc.read_shard(s)[:2] for s in shard_dirs], out,
                           dict(model_tag="qwen25-7b", config="c6", config_name="hf-nf4-c6", num_hidden_layers=28,
                                sweep_layers=[], views=["prenorm"]))
    assert meta["dropped_prompt_not_harvested"] == 1 and meta["n_pairs"] == {"WildJailbreak": 160, "SORRY-Bench": 80}
    assert not (out / "postnorm_last.npy").exists()
    d = wp.load(out, "prenorm")
    Hv, Ha, m = wp.pair_arrays(d, "WildJailbreak")
    p = pd.read_parquet(out / "pairs.parquet").set_index("pair_id")
    keys = d["prompts"]["prompt_key"].to_numpy()
    for i in (0, 17, len(m) - 1):                         # every row is the vector of the prompt the map names
        pid = m["pair_id"].iloc[i]
        pv, pa = pm[(pm.pair_id == pid) & (pm.role == "vanilla")].prompt_id.item(), \
            pm[(pm.pair_id == pid) & (pm.role == "adv")].prompt_id.item()
        assert keys[p.loc[pid, "row_v"]] == pv and keys[p.loc[pid, "row_a"]] == pa
        assert np.allclose(Hv[i], np.float16(vec[pv]), atol=1e-2) and np.allclose(Ha[i], np.float16(vec[pa]), atol=1e-2)
    assert wp.run_labels(json.loads((out / "meta.json").read_text()))[:2] == ("qwen25-7b [hf-nf4-c6]", "RESULT")


def test_run_all_on_converted_folder_recovers_planted_moves(tmp_path):
    u, _, table, pm, shard_dirs = fake_world(tmp_path)
    plan, _ = dc.plan_pairs(pm, table, ["WildJailbreak", "SORRY-Bench"])
    out = tmp_path / "out"
    dc.build_folder(plan, [dc.read_shard(s)[:2] for s in shard_dirs], out,
                    dict(model_tag="qwen25-7b", config="c6", config_name="hf-nf4-c6", num_hidden_layers=28,
                         sweep_layers=[], views=["prenorm"]))
    res, figs = wp.run_all(out, "prenorm", fig_dir=tmp_path / "figs", n_boot=200)
    w = res["A2_wildjailbreak"]["mean_dp_heldout"]
    assert w["harmful"][2] < 0 and w["benign"][2] < 0                       # planted -1u for both intents
    rows = res["A2_sorry_per_tactic"]["per_tactic"]
    assert rows["towards_benign"]["mean_dp"][2] < -2.0                     # planted -3u
    assert abs(rows["sideways"]["mean_dp"][0]) < 0.5                        # planted move orthogonal to u
    assert abs(res["A7_wildjailbreak"]["pooled"]["content"][0]) < 0.4        # same move for both intents
    assert res["A8"]["angle_harm_vs_refusal_proxy_deg"] is None             # no next-token stats: skipped, not 0
    assert res["A8"]["refusal_proxy"].startswith("not in this extraction")
    assert res["label"].startswith("RESULT") and "A8_angle" not in figs


def test_restore_refuses_an_archive_that_differs_from_its_manifest(tmp_path):
    a = tmp_path / "qwen25-7b__wildjailbreak-s00000__hf-nf4-c6__e7f5ea94.tpxz"
    a.write_bytes(b"not the archive the manifest describes")
    (tmp_path / f"{a.name}.json").write_text(json.dumps({"sha256": "0" * 64}))
    (tmp_path / "work").mkdir()
    with pytest.raises(ValueError, match="sha256 mismatch"):
        dc.restore_archive(a, tmp_path / "work")
