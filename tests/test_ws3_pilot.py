"""Tests for analysis/ws3_pilot.py on SYNTHETIC data with planted answers (no model, no GPU, no real prompts)."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis import lib  # noqa: E402
from analysis import ws3_pilot as wp  # noqa: E402

D = 32


def fake_pairs_table():
    """Text-bearing pairs table in the v1.1 shape: 2 WildJailbreak pairs + 2 SORRY-Bench mutations of ONE base."""
    rows = [dict(pair_id="wj1", source="WildJailbreak", intent="harmful", tactic="unknown", split="train",
                 vanilla_prompt="plain a", adversarial_prompt="wrapped a"),
            dict(pair_id="wj2", source="WildJailbreak", intent="benign", tactic="unknown", split="test",
                 vanilla_prompt="plain b", adversarial_prompt="wrapped b"),
            dict(pair_id="s1", source="SORRY-Bench", intent="harmful", tactic="slang", split="train",
                 vanilla_prompt="base x", adversarial_prompt="slang x"),
            dict(pair_id="s2", source="SORRY-Bench", intent="harmful", tactic="morse", split="val",
                 vanilla_prompt="base x", adversarial_prompt="morse x"),
            dict(pair_id="j1", source="JailbreakBench", intent="benign", tactic="jbb_benign_twin", split="train",
                 vanilla_prompt="jbb v", adversarial_prompt="jbb a")]
    return pd.DataFrame(rows)


def test_prompt_plan_dedups_and_keeps_no_text():
    prompts, plan = wp.prompt_plan(fake_pairs_table())
    keys = [k for k, _ in prompts]
    assert len(keys) == 7                        # 4 WJ + 1 shared SORRY base + 2 mutations; JBB left out
    assert keys == sorted(keys)                  # deterministic order
    s = plan[plan["source"] == "SORRY-Bench"]
    assert s["key_v"].nunique() == 1             # the shared base instruction is run once
    assert not any("prompt" in c for c in plan.columns)     # no text column survives
    assert all(len(k) == 20 for k in keys) and wp.prompt_key("base x") == s["key_v"].iloc[0]
    assert wp.keys_hash(keys) == wp.keys_hash(list(keys)) != wp.keys_hash(keys[::-1])


def make_folder(tmp_path, seed=0, wj_shift=(-1.0, -1.0)):
    """Fake extraction folder with planted geometry:
    u = harmfulness axis. WildJailbreak: harmful bases at +2u, benign at -2u; rewrites move wj_shift = (harmful,
    benign) times u, by default -1.0u for both intents (style -1, content 0).
    SORRY-Bench (all harmful, bases at +2u, shared by tactics): 'towards_benign' moves -3u, 'sideways' moves along
    w (orthogonal to u), 'mixed' moves -1.5u + w2."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=D)
    u /= np.linalg.norm(u)

    def orth():
        w = rng.normal(size=D)
        w -= (w @ u) * u
        return w / np.linalg.norm(w)

    w, w2 = orth(), orth()
    shifts = {"towards_benign": -3.0 * u, "sideways": 3.0 * w, "mixed": -1.5 * u + 2.0 * w2}
    H, prows, pairs = [], [], []

    def add(vec, refused=0.0, finite=True):
        H.append(vec)
        prows.append(dict(prompt_key=f"k{len(prows):05d}", n_tokens=10, refusal_mass=refused, compliance_mass=0.1,
                          finite=finite))
        return len(H) - 1

    for i in range(240):
        intent = "harmful" if i % 2 == 0 else "benign"
        pid = f"wj{i:04d}"
        base = (2.0 if intent == "harmful" else -2.0) * u + 0.5 * rng.normal(size=D)
        rv = add(base, refused=0.9 if intent == "harmful" else 0.05)
        ra = add(base + wj_shift[intent != "harmful"] * u + 0.3 * rng.normal(size=D), refused=0.3)
        pairs.append(dict(pair_id=pid, source="WildJailbreak", intent=intent, tactic="unknown",
                          split=lib.assign_split(pid), holdout_group="", row_v=rv, row_a=ra))
    for b in range(60):
        rv = add(2.0 * u + 0.5 * rng.normal(size=D), refused=0.9)
        for t, sh in shifts.items():
            pid = f"s{b:03d}_{t}"
            ra = add(H[rv] + sh + 0.3 * rng.normal(size=D), refused=0.2)
            pairs.append(dict(pair_id=pid, source="SORRY-Bench", intent="harmful", tactic=t, split=lib.assign_split(pid),
                              holdout_group=f"b{b}", row_v=rv, row_a=ra))
    prows[-1]["finite"] = False                           # one broken prompt: its pair must be dropped
    H = np.array(H, np.float32)
    np.save(tmp_path / "prenorm_last.npy", H)
    np.save(tmp_path / "postnorm_last.npy", H * 0.5)
    np.save(tmp_path / "sweep.npy", np.stack([H, H], axis=1).astype(np.float16))
    pd.DataFrame(prows).to_parquet(tmp_path / "prompts.parquet", index=False)
    pd.DataFrame(pairs).to_parquet(tmp_path / "pairs.parquet", index=False)
    (tmp_path / "meta.json").write_text(json.dumps(dict(num_hidden_layers=28, sweep_layers=[0, 4], model="synthetic")))
    return u, w


def test_load_and_pair_arrays(tmp_path):
    make_folder(tmp_path)
    d = wp.load(tmp_path, "prenorm")
    Hv, Ha, meta = wp.pair_arrays(d, "SORRY-Bench")
    assert meta.attrs["dropped_nonfinite"] == 1 and len(meta) == 179
    assert np.allclose(Hv, d["H"][meta["row_v"]]) and np.allclose(Ha, d["H"][meta["row_a"]])
    assert wp.load(tmp_path, 4)["H"].shape == d["H"].shape
    assert np.allclose(wp.load(tmp_path, "postnorm")["H"], d["H"] * 0.5)
    assert wp.layer_label(d) == "28 (last, pre-norm)"


def test_per_tactic_masking_and_alignment_recover_planted_moves(tmp_path):
    u, _ = make_folder(tmp_path)
    d = wp.load(tmp_path)
    Sv, Sa, ms = wp.pair_arrays(d, "SORRY-Bench")
    mid = 0.0                                             # bases at +2u, benign side at -2u: midpoint ~ 0
    rows = wp.per_tactic_masking(Sv, Sa, ms, u, mid, n_boot=300)
    tb, sw = rows["towards_benign"], rows["sideways"]
    assert tb["mean_dp"][2] < -2.5 and abs(sw["mean_dp"][0]) < 0.3       # planted -3u vs 0
    assert tb["share_adv_below"][0] > 0.6 and sw["share_adv_below"][0] < 0.1
    assert tb["share_van_below"] < 0.1 and tb["family"] == "other"
    al = wp.tactic_alignment(Sa - Sv, ms["tactic"], u)
    assert al["towards_benign"]["cos_to_r"] < -0.95 and abs(al["sideways"]["cos_to_r"]) < 0.15
    assert -0.95 < al["mixed"]["cos_to_r"] < -0.3


def test_geometry_helpers(tmp_path):
    make_folder(tmp_path)
    d = wp.load(tmp_path)
    Sv, Sa, ms = wp.pair_arrays(d, "SORRY-Bench")
    names, A = wp.angle_matrix(Sa - Sv, ms["tactic"], k=1)
    i, j = names.index("towards_benign"), names.index("sideways")
    assert A[i, j] > 75 and np.allclose(A, A.T) and np.all(np.diag(A) == 0)   # orthogonal planted moves
    ks = wp.knn_self_share(Sa - Sv, ms["tactic"], k=5)
    assert all(v["self_share"] > 0.8 for v in ks.values())              # distinct moves -> own clusters
    assert abs(sum(v["chance"] for v in ks.values()) - 1) < 1e-9
    rk = wp.rank_by_tactic(Sa - Sv, ms["tactic"])
    assert set(rk) == {"towards_benign", "sideways", "mixed"} and all(v["effective_rank"] >= 1 for v in rk.values())
    assert wp.refusal_proxy(d["prompts"]).sum() == (d["prompts"]["refusal_mass"] >= 0.5).sum()


def test_run_all_end_to_end(tmp_path):
    (tmp_path / "x").mkdir()
    make_folder(tmp_path / "x")
    res, figs = wp.run_all(tmp_path / "x", "prenorm", fig_dir=tmp_path / "figs", results_path=tmp_path / "r.json",
                           n_boot=200)
    w = res["A2_wildjailbreak"]
    assert w["mean_dp_heldout"]["harmful"][2] < 0 and w["mean_dp_heldout"]["benign"][2] < 0   # planted -1u for both
    assert res["A2_sorry_per_tactic"]["sorry_vanilla_share_on_harmful_side"] > 0.9
    assert res["A8"]["angle_harm_vs_refusal_proxy_deg"] is not None
    a7 = res["A7_wildjailbreak"]["pooled"]                # planted: both intents move -1u -> style -1, content 0
    assert a7["style"][2] < 0 and abs(a7["content"][0]) < 0.3
    assert all(p.exists() and p.stat().st_size > 5_000 for p in figs.values()) and len(figs) == 11
    saved = json.loads((tmp_path / "r.json").read_text())
    assert "per_tactic" in saved["A2_sorry_per_tactic"] and saved["label"].startswith("PILOT")
    assert "pooled" in saved["A7_wildjailbreak"]
    assert len(wp.summary_lines(res)) > 5 and any(s.startswith("A7") for s in wp.summary_lines(res))


def test_a7_style_content_recovers_planted_split(tmp_path):
    """Harmful rewrites move -3u, benign ones -1u: style (benign) = -1, content (harmful - benign) = -2, on held-out
    pairs only; WildJailbreak's single tactic label gives one row, so the figure shows the pooled bar alone."""
    make_folder(tmp_path, wj_shift=(-3.0, -1.0))
    res, figs = wp.run_all(tmp_path, "prenorm", fig_dir=tmp_path / "figs", n_boot=300)
    a7 = res["A7_wildjailbreak"]
    st, ct = a7["pooled"]["style"], a7["pooled"]["content"]
    assert abs(st[0] + 1.0) < 0.3 and st[1] <= st[0] <= st[2] and st[2] < 0
    assert abs(ct[0] + 2.0) < 0.4 and ct[1] <= ct[0] <= ct[2] and ct[2] < 0
    assert set(a7["per_tactic"]) == {"unknown"} and a7["per_tactic"]["unknown"]["style"] == a7["pooled"]["style"]
    held_out = sum(lib.assign_split(f"wj{i:04d}") != "train" for i in range(240))
    assert a7["pooled"]["n_harmful"] + a7["pooled"]["n_benign"] == held_out    # train pairs never enter A7
    assert figs["A7_wildjailbreak"].exists()
