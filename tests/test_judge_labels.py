"""Tests for analysis/judge_labels.py and its use in run_all and regenerate_all, on SYNTHETIC data with planted answers."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from analysis import judge_labels as jl  # noqa: E402
from analysis import regenerate_all as ra  # noqa: E402
from analysis import ws3_pilot as wp  # noqa: E402
from test_ws3_pilot import make_folder  # noqa: E402


def verdict(key, status="ok", refusal="no", harmful="no"):
    return dict(prompt_key=key, status=status, harmful_request="yes", refusal=refusal, harmful_response=harmful, reply_truncated=False)


def planted_arrays(harm_gap=-2.0, benign_gap=0.0, n=200, seed=0):
    """Test pairs: harmful ones split success / fail, benign ones split answered / refused, with planted dp gaps."""
    rng = np.random.default_rng(seed)
    noise = lambda: rng.normal(0, 0.1, n)  # noqa: E731
    dp = np.r_[harm_gap + noise(), noise(), benign_gap + noise(), noise()]
    intent = np.r_[["harmful"] * 2 * n, ["benign"] * 2 * n]
    success = np.r_[np.ones(n, bool), np.zeros(n, bool), np.zeros(2 * n, bool)]
    refused = np.r_[np.zeros(n, bool), np.ones(n, bool), np.zeros(n, bool), np.ones(n, bool)]
    return dp, intent, np.full(4 * n, "test"), np.ones(4 * n, bool), success, refused


# ------------------------------------------------------------------------------------------------ joining
def test_labels_join_by_prompt_key_not_by_row_order():
    prompts = pd.DataFrame(dict(prompt_key=["a", "b", "c"]))
    labels = pd.DataFrame([verdict("c", harmful="yes"), verdict("a", refusal="yes")])
    al = jl.align(prompts, labels)
    assert al["valid"].tolist() == [True, False, True]         # b has no label
    assert al["refused"].tolist() == [True, False, False]
    assert al["harmful_response"].tolist() == [False, False, True]


def test_rows_without_a_clean_verdict_are_not_valid():
    prompts = pd.DataFrame(dict(prompt_key=["a", "b"]))
    labels = pd.DataFrame([verdict("a", status="unparsed", refusal=""), verdict("b", status="empty_reply", refusal="")])
    assert jl.align(prompts, labels)["valid"].tolist() == [False, False]


def test_success_rules_differ_only_on_harmful_replies_that_also_refuse():
    al = dict(harmful_response=np.array([True, True, False]), refused=np.array([False, True, False]))
    rows = np.arange(3)
    assert jl.SUCCESS_RULES["harmful_response"](al, rows).tolist() == [True, True, False]
    assert jl.SUCCESS_RULES["harmful_not_refused"](al, rows).tolist() == [True, False, False]
    assert jl.SUCCESS_RULES["complied"](al, rows).tolist() == [True, False, True]
    assert jl.H2_SUCCESS_RULE == "harmful_response"


def test_align_reads_the_truncation_flag_and_treats_missing_rows_as_not_truncated():
    prompts = pd.DataFrame(dict(prompt_key=["a", "b", "c"]))
    labels = pd.DataFrame([dict(verdict("a"), reply_truncated=True), verdict("b")])
    assert jl.align(prompts, labels)["truncated"].tolist() == [True, False, False]


def test_find_labels_matches_model_and_subset_and_ignores_timing_tests(tmp_path):
    def add(name, **meta):
        (tmp_path / "ws3_judges" / name).mkdir(parents=True)
        (tmp_path / "ws3_judges" / name / "meta.json").write_text(json.dumps(meta))
    add("m__sub4000__wildguard", model_tag="m", subset=4000)
    add("m__smoke10__wildguard", model_tag="m", subset=None, limit=10)
    add("n__wildguard", model_tag="n", subset=None)
    assert jl.find_labels(tmp_path, "m", 4000).name == "m__sub4000__wildguard"
    assert jl.find_labels(tmp_path, "m", None) is None         # the timing test is not a full-run label set
    assert jl.find_labels(tmp_path, "n", 0).name == "n__wildguard"
    assert jl.find_labels(tmp_path / "nowhere", "m", 4000) is None


# ------------------------------------------------------------------------------------------------ H2
def test_contrast_recovers_a_planted_difference_with_a_ci_around_it():
    rng = np.random.default_rng(1)
    est, lo, hi = jl.contrast_ci([rng.normal(-2, .1, 500), rng.normal(0, .1, 500)], [1, -1], n_boot=500)
    assert abs(est + 2) < 0.05 and lo < est < hi and hi < -1.9 + 0.1
    assert jl.excludes_zero((est, lo, hi))
    assert not jl.excludes_zero((0.1, -0.2, 0.4))


def test_h2_supported_when_success_moves_further_and_benign_prompts_do_not():
    r = jl.h2_result(*planted_arrays(harm_gap=-2.0, benign_gap=0.0), n_boot=300)
    assert r["verdict"] == "supported"
    assert abs(r["difference"][0] + 2) < 0.1 and abs(r["difference_in_differences"][0] + 2) < 0.1
    assert (r["n_success"], r["n_fail"], r["n_benign_answered"], r["n_benign_refused"]) == (200, 200, 200, 200)


def test_h2_not_specific_when_benign_prompts_show_the_same_gap():
    r = jl.h2_result(*planted_arrays(harm_gap=-2.0, benign_gap=-2.0), n_boot=300)
    assert r["difference"][2] < 0 and not jl.excludes_zero(r["difference_in_differences"])
    assert r["verdict"] == "harmful_effect_not_specific"


def test_h2_not_supported_when_there_is_no_gap_or_the_gap_is_the_wrong_way():
    assert jl.h2_result(*planted_arrays(harm_gap=0.0), n_boot=300)["verdict"] == "not_supported"
    assert jl.h2_result(*planted_arrays(harm_gap=2.0), n_boot=300)["verdict"] == "not_supported"


def test_h2_is_not_computable_with_too_few_successes_or_no_benign_split():
    dp, intent, split, valid, success, refused = planted_arrays(n=20)
    none_won = jl.h2_result(dp, intent, split, valid, np.zeros(80, bool), refused, n_boot=50)
    assert none_won["verdict"] == "not_computable" and none_won["difference"] is None and none_won["n_success"] == 0
    all_answered = jl.h2_result(dp, intent, split, valid, success, np.zeros(80, bool), n_boot=50)
    assert all_answered["verdict"] == "not_computable_no_benign_control" and all_answered["difference_in_differences"] is None


def test_h2_uses_test_pairs_only_and_leaves_out_rows_without_a_verdict():
    dp, intent, split, valid, success, refused = planted_arrays(n=20)
    split = split.copy()
    split[:5] = "train"           # 5 successes are not test pairs
    valid = valid.copy()
    valid[20:23] = False          # 3 fails have no verdict
    r = jl.h2_result(dp, intent, split, valid, success, refused, n_boot=50)
    assert (r["n_success"], r["n_fail"], r["n_harmful_no_verdict"]) == (15, 17, 3)


# ------------------------------------------------------------------------------------------------ real folder layout
def labels_for(folder, refused=lambda i: False, harmful=lambda i: False):
    keys = pd.read_parquet(folder / "prompts.parquet")["prompt_key"]
    return pd.DataFrame([verdict(k, refusal="yes" if refused(i) else "no", harmful="yes" if harmful(i) else "no")
                         for i, k in enumerate(keys)])


def test_refusal_angle_recovers_a_planted_direction_on_train_rows(tmp_path):
    u, _ = make_folder(tmp_path)
    H = np.load(tmp_path / "prenorm_last.npy")
    prompts = pd.read_parquet(tmp_path / "prompts.parquet")
    side = H @ u > 0              # refused = the +u side, so the refusal direction lies along u
    labels = pd.DataFrame([verdict(k, refusal="yes" if s else "no") for k, s in zip(prompts["prompt_key"], side)])
    r = jl.refusal_angle(H, np.arange(len(H)), u, jl.align(prompts, labels))
    assert r["angle_deg"] < 15 and r["n_train_rows"] == len(H) and 0 < r["n_refused"] < len(H)
    none = pd.DataFrame([verdict(k) for k in prompts["prompt_key"]])
    assert jl.refusal_angle(H, np.arange(len(H)), u, jl.align(prompts, none))["angle_deg"] is None


def test_run_all_adds_wildguard_next_to_the_proxy_and_changes_nothing_without_labels(tmp_path):
    make_folder(tmp_path)
    base, base_figs = wp.run_all(tmp_path, "prenorm", fig_dir=tmp_path / "f0", n_boot=50)
    assert "refusal_wildguard" not in base["A8"] and "A8_angle_wildguard" not in base_figs
    labels = labels_for(tmp_path, refused=lambda i: i % 3 == 0)
    res, figs = wp.run_all(tmp_path, "prenorm", fig_dir=tmp_path / "f1", n_boot=50, labels=labels)
    assert res["A8"]["refusal_wildguard"]["angle_deg"] is not None and "A8_angle_wildguard" in figs
    assert figs["A8_angle_wildguard"].exists()
    assert res["A8"]["angle_harm_vs_refusal_proxy_deg"] == base["A8"]["angle_harm_vs_refusal_proxy_deg"]
    assert res["A2_wildjailbreak"]["h1_test"] == base["A2_wildjailbreak"]["h1_test"]


def test_h2_for_folder_labels_the_adversarial_reply_of_each_pair(tmp_path):
    make_folder(tmp_path)
    pairs = pd.read_parquet(tmp_path / "pairs.parquet")
    wj = pairs[(pairs["source"] == "WildJailbreak") & (pairs["intent"] == "harmful") & (pairs["split"] == "test")]
    won = set(wj["row_a"].iloc[: len(wj) // 2])
    res = jl.h2_for_folder(tmp_path, labels_for(tmp_path, harmful=lambda i: i in won), n_boot=50)
    row = res["wildjailbreak"]["harmful_response"]
    assert row["n_success"] == len(won) and row["n_fail"] == len(wj) - len(won)
    assert set(res["wildjailbreak"]) == set(jl.SUCCESS_RULES)
    assert "wildjailbreak" in res and "sorry_exploratory" in res
    assert res["wildjailbreak_excluding_truncated"]["n_success"] == len(won)    # nothing truncated in these labels


def test_the_truncation_check_drops_only_cut_off_replies(tmp_path):
    make_folder(tmp_path)
    pairs = pd.read_parquet(tmp_path / "pairs.parquet")
    wj = pairs[(pairs["source"] == "WildJailbreak") & (pairs["intent"] == "harmful") & (pairs["split"] == "test")]
    won = list(wj["row_a"].iloc[: len(wj) // 2])
    cut = set(won[:3])
    labels = labels_for(tmp_path, harmful=lambda i: i in set(won))
    labels.loc[list(cut), "reply_truncated"] = True
    res = jl.h2_for_folder(tmp_path, labels, n_boot=50)
    assert res["wildjailbreak"]["harmful_response"]["n_success"] == len(won)
    assert res["wildjailbreak_excluding_truncated"]["n_success"] == len(won) - 3
    assert res["wildjailbreak_excluding_truncated"]["n_harmful_no_verdict"] == 3


def test_step_h2_writes_one_row_per_labelled_model_and_prefers_c1(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "REPO", tmp_path / "repo")
    (tmp_path / "repo" / "results").mkdir(parents=True)
    for name, config in (("mod__hf-bf16-bs1__sub4000", "c1"), ("mod__hf-nf4-bs1__sub4000", "c6")):
        folder = tmp_path / "ws3_harvest" / name
        folder.mkdir(parents=True)
        make_folder(folder)
        meta = json.loads((folder / "meta.json").read_text())
        meta.update(model_tag="mod", config=config, config_name=name.split("__")[1], subset=4000, gpu="g")
        (folder / "meta.json").write_text(json.dumps(meta))
    judged = tmp_path / "ws3_judges" / "mod__wildguard"
    judged.mkdir(parents=True)
    labels_for(tmp_path / "ws3_harvest" / "mod__hf-bf16-bs1__sub4000").to_parquet(judged / "labels.parquet", index=False)
    (judged / "meta.json").write_text(json.dumps(dict(model_tag="mod", subset=4000, replies_backend="vLLM x", replies_dtype="bfloat16")))
    folders = ra.find_folders(tmp_path)
    ra.step_h2(folders, tmp_path, n_boot=50)
    out = json.loads((tmp_path / "repo" / "results" / "WS3_H2.json").read_text())
    assert out["mod"]["activations"]["folder"] == "mod__hf-bf16-bs1__sub4000"
    assert set(out["mod"]["views"]) == {"prenorm", "postnorm"}
    assert out["mod"]["deciding_rule"] == jl.H2_SUCCESS_RULE and "fixed 10 Nov" in out["mod"]["note"]
    assert "wildjailbreak_excluding_truncated" in out["mod"]["views"]["prenorm"]


def test_step_h2_says_so_when_there_are_no_labels(tmp_path, capsys):
    ra.step_h2([], tmp_path, n_boot=10)
    assert "skipped" in capsys.readouterr().out
