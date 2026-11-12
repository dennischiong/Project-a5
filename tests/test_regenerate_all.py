"""Tests for the folder choice behind A2 across models in analysis/regenerate_all.py (meta.json files only, no arrays)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis import regenerate_all as ra  # noqa: E402


def folder(root, sub, name, **meta):
    f = root / sub / name
    f.mkdir(parents=True)
    (f / "meta.json").write_text(json.dumps(meta))
    return f


def test_full_c1_beats_full_c6_beats_subsets(tmp_path):
    a = folder(tmp_path, "ws3_harvest", "m__c6_sub", model_tag="m", config="c6", subset=4000)
    b = folder(tmp_path, "ws3_harvest", "m__c1_sub", model_tag="m", config="c1", subset=4000)
    c = folder(tmp_path, "ws3_harvest", "m__c6", model_tag="m", config="c6")
    assert ra.pick_model_folders([a, b])[0] == {"m": b}
    assert ra.pick_model_folders([a, b, c])[0] == {"m": c}
    d = folder(tmp_path, "ws3_harvest", "m__c1", model_tag="m", config="c1")
    assert ra.pick_model_folders([d, a, b, c])[0] == {"m": d}


def test_bf16_beats_fp16_when_runs_are_otherwise_equal(tmp_path):
    fp = folder(tmp_path, "ws3_harvest", "q__hf-fp16-bs1", model_tag="q", config="c1", dtype="float16")
    bf = folder(tmp_path, "ws3_harvest", "q__hf-bf16-bs1", model_tag="q", config="c1", dtype="bfloat16")
    assert ra.pick_model_folders([bf, fp])[0] == {"q": bf}     # fp16 sorts later but still loses
    sub = folder(tmp_path, "ws3_harvest", "q__bf16_sub", model_tag="q", config="c1", dtype="bfloat16", subset=4000)
    assert ra.pick_model_folders([sub, fp])[0] == {"q": fp}    # a full run still beats a subset run


def test_same_config_only_when_every_model_is_a_full_c1_run_in_one_precision(tmp_path):
    q1 = folder(tmp_path, "ws3_harvest", "q1", model_tag="qwen25-1p5b", config="c1", config_name="hf-fp16-bs1")
    q7 = folder(tmp_path, "ws3_harvest", "q7", model_tag="qwen25-7b", config="c6", config_name="hf-nf4-c6")
    mi = folder(tmp_path, "ws3_harvest", "mi", model_tag="mistral-7b", config="c1", config_name="hf-bf16-bs1")
    chosen, same = ra.pick_model_folders([q1, mi])
    assert not same and chosen == {"qwen25-1p5b": q1, "mistral-7b": mi}        # fp16 and bf16 are not the same config
    q2 = folder(tmp_path, "ws3_harvest", "q2", model_tag="qwen25-1p5b", config="c1", config_name="hf-bf16-bs1")
    assert ra.pick_model_folders([q2, mi])[1]
    chosen, same = ra.pick_model_folders([q1, q7, mi])
    assert not same and set(chosen) == {"qwen25-1p5b", "qwen25-7b", "mistral-7b"}


def test_later_folder_wins_a_tie_so_the_final_table_replaces_the_pilot(tmp_path):
    old = folder(tmp_path, "ws3_pilot", "qwen25-1p5b")                     # no tag/config: the 9 Oct pilot meta
    new = folder(tmp_path, "ws3_harvest", "qwen25-1p5b__wj-sorry__hf-fp16-bs1", model_tag="qwen25-1p5b", config="c1")
    assert ra.pick_model_folders([old, new])[0] == {"qwen25-1p5b": new}


def test_figure_kind_is_a_valid_title_kind_and_pilot_wins():
    from analysis import plotstyle as ps
    assert ra.figure_kind(True) == "PILOT" and ra.figure_kind(False) == "RESULT"
    assert {ra.figure_kind(True), ra.figure_kind(False)} <= set(ps.KINDS)   # ps.tag accepts only the listed kinds


def test_mixed_config_labels_name_the_config_and_subset(tmp_path):
    f = folder(tmp_path, "ws3_harvest", "mi", model_tag="mistral-7b", config="c6", config_name="hf-nf4-bs1",
               subset=4000)
    info = ra.folder_info(f)
    assert ra.model_label(info, same_config=True) == "mistral-7b"
    assert ra.model_label(info, same_config=False) == "mistral-7b\nhf-nf4-bs1, sub4000"
    p = folder(tmp_path, "ws3_harvest", "q1", model_tag="qwen25-1p5b", config="c1", config_name="hf-fp16-bs1")
    assert ra.model_label(ra.folder_info(p), same_config=False) == "qwen25-1p5b (pilot)\nhf-fp16-bs1"


def test_sweep_views_add_the_last_layer_and_skip_short_sweeps():
    assert ra.sweep_views(dict(num_hidden_layers=28, sweep_layers=[16, 0, 8, 24])) == [
        (8, "8"), (16, "16"), (24, "24"), ("prenorm", "28 (last, pre-norm)")]             # layer 0: same vector in every prompt
    assert ra.sweep_views(dict(num_hidden_layers=28, sweep_layers=[0])) == []       # subset runs keep layer 0 only
    assert ra.sweep_views(dict(num_hidden_layers=28, sweep_layers=[])) == []        # converted harvest, no sweep
    assert ra.sweep_views(dict(num_hidden_layers=28, sweep_layers=[4, 28])) == []   # 28 = post-norm, not a sweep point
