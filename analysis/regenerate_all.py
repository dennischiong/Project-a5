"""Regenerate every WS3 figure and numbers file from one script (work plan s.8: every figure regenerates from one
script). CPU only; reads the extraction folders on Drive, never prompt text.

What it does, in order (each step is skipped with a printed reason when its data is missing):
  1  diagrams + synthetic previews of every figure function    -> figures/_previews/, figures/_contact_sheet.png
  2  Task 6 pilot (harvest/out, 100 pairs)                      -> analysis/pilot_analysis.main()
  3  every real extraction folder (ws3_pilot/*, ws3_harvest/*; smoke tests skipped), views prenorm + postnorm
                                                                -> figures/WS3/<folder>/, results/WS3_<folder>_<view>.json
  4  A2 across models (one folder per model: full C1 if any, else the best other config, labelled; Cohen's d)
                                                                -> figures/WS3/A2_models.png, figures/WS3/F2.png (paper F2), results/WS3_A2_models.json
  5  X5 per model: C6 (NF4) against C1 (HF bf16) on the same pairs -> figures/WS3/X5_*.png, results/WS3_X5.json
  5b A2 across layers (exploratory), every folder with a layer sweep: sweep layers + the last layer pre-norm
                                                                -> figures/WS3/<folder>/A2_layers.png, results/WS3_layer_sweep.json
  6  contact sheet of the real figures                          -> figures/WS3/_real_contact_sheet.png

    python analysis/regenerate_all.py [--data-root $A5_WORK_ROOT] [--skip-previews] [--n-boot 2000]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from seeds import set_seed  # noqa: E402

DATA_ROOT = os.environ.get("A5_WORK_ROOT", "/content/drive/MyDrive/a5-work")   # holds ws3_pilot/ and ws3_harvest/
REFERENCE = "C1 HF bf16 bs1"
CONFIG_LABEL = {"c1": REFERENCE, "c6": "C6 NF4 4-bit"}
F2_MODEL = "qwen25-7b"  # panel (a) of the paper figure F2; else the first non-pilot model


def find_folders(data_root):
    """Finished extraction folders (meta.json present), smoke tests excluded, sorted by name."""
    root = Path(data_root)
    out = []
    for sub in ("ws3_pilot", "ws3_harvest"):
        for f in sorted((root / sub).glob("*")):
            if (f / "meta.json").exists() and "smoke" not in f.name:
                out.append(f)
    return out


def folder_info(folder):
    m = json.loads((Path(folder) / "meta.json").read_text())
    tag = m.get("model_tag", "qwen25-1p5b")
    return dict(tag=tag, config=m.get("config", "c1"), subset=m.get("subset"), pilot=tag == "qwen25-1p5b", meta=m)


def aligned_wj(folders, view="prenorm"):
    """WildJailbreak (Hv, Ha) per folder on the pairs present and finite in ALL folders, rows in one order.
    Returns ({folder: (Hv, Ha)}, meta of the common pairs)."""
    from analysis import ws3_pilot as wp
    got = {}
    for f in folders:
        Hv, Ha, m = wp.pair_arrays(wp.load(f, view), "WildJailbreak")
        got[f] = (Hv, Ha, m.reset_index(drop=True))
    common = set.intersection(*(set(m["pair_id"]) for _, _, m in got.values()))
    ref = next(iter(got.values()))[2]
    meta = ref[ref["pair_id"].isin(common)].sort_values("pair_id").reset_index(drop=True)
    out = {}
    for f, (Hv, Ha, m) in got.items():
        pos = dict(zip(m["pair_id"], range(len(m))))
        idx = np.array([pos[p] for p in meta["pair_id"]])
        out[f] = (Hv[idx], Ha[idx])
    return out, meta


def step_previews():
    from analysis import contact_sheet as cs
    prev = REPO / "figures" / "_previews"
    prev.mkdir(parents=True, exist_ok=True)
    pngs = cs.render_everything(prev)
    p = cs.contact_sheet(pngs, REPO / "figures" / "_contact_sheet.png")
    print(f"[1] {len(pngs)} diagrams and synthetic previews -> {prev}; contact sheet {p}")


def step_task6():
    if not (REPO / "harvest" / "out" / "pilot_meta.json").exists():
        print("[2] skipped: harvest/out/pilot_meta.json not found (Task 5 pilot extraction)")
        return
    from analysis import pilot_analysis
    pilot_analysis.main()
    print("[2] Task 6 pilot figures and analysis/pilot_results.json regenerated")


def step_folders(folders, n_boot):
    from analysis import ws3_pilot as wp
    made = []
    for f in folders:
        # a converted harvest (harvest/dennis_harvest_convert.py) has the pre-norm view only
        for view in [v for v in ("prenorm", "postnorm") if (f / f"{v}_last.npy").exists()]:
            res, figs = wp.run_all(f, view, fig_dir=REPO / "figures" / "WS3" / f.name,
                                   results_path=REPO / "results" / f"WS3_{f.name}_{view}.json", n_boot=n_boot)
            made += [p for p in figs.values() if p]
            print(f"[3] {f.name} {view}: {len(figs)} figures")
            for line in wp.summary_lines(res)[:2]:
                print("     " + line)
    return made


def folder_rank(info):
    """Which folder speaks for a model in A2 across models: full C1 first, then full C6, then subset runs."""
    return (bool(info["subset"]), info["config"] != "c1")


def pick_model_folders(folders):
    """One folder per model tag. Returns ({tag: folder}, same_config): same_config is True when every model is a
    full C1 run in the same precision (the clean comparison); otherwise the configs are mixed and every label must name its config.
    Ties go to the later folder in the list (find_folders puts ws3_harvest, the final pairs table, after ws3_pilot)."""
    best = {}
    for f in folders:
        info = folder_info(f)
        if info["tag"] not in best or folder_rank(info) <= folder_rank(best[info["tag"]][1]):
            best[info["tag"]] = (f, info)
    same = (all(folder_rank(i) == (False, False) for _, i in best.values())
            and len({i["meta"].get("config_name", i["config"]) for _, i in best.values()}) == 1)
    return {t: f for t, (f, _) in best.items()}, same


def figure_kind(any_pilot):
    """Title kind of a figure that pools folders: PILOT as soon as one of them is the plumbing-scale pilot model
    (ws3_pilot.run_labels labels every Qwen2.5-1.5B run PILOT), else RESULT."""
    return "PILOT" if any_pilot else "RESULT"


def model_label(info, same_config):
    label = info["tag"] + (" (pilot)" if info["pilot"] else "")
    if not same_config:
        label += f"\n{info['meta'].get('config_name', info['config'])}" + (f", sub{info['subset']}" if info["subset"] else "")
    return label


def step_models(folders, n_boot):
    from analysis import figures as F
    from analysis import geometry as g
    from analysis import lib
    from analysis import ws3_data
    from analysis import ws3_pilot as wp
    chosen, same = pick_model_folders(folders)
    if len(chosen) < 2:
        print(f"[4] skipped: A2 across models needs 2+ models, found {len(chosen)}")
        return []
    by_model, out, pilot, f2 = {}, {}, False, None
    for tag, f in chosen.items():
        info = folder_info(f)
        Hv, Ha, m = wp.pair_arrays(wp.load(f, "prenorm"), "WildJailbreak")
        a2 = g.a2_projection(Hv, Ha, m, n_boot=n_boot)
        ev = (m["split"] != "train").to_numpy()
        harm = (m["intent"] == "harmful").to_numpy()
        # dp is in each model's own units: Cohen's d (mean / SD of held-out dp) is what compares across models
        a2["d"] = {i: lib.bootstrap_ci(a2["dp"][ev & (harm if i == "harmful" else ~harm)], lib.cohens_d, n=n_boot)
                   for i in ("harmful", "benign")}
        pilot |= info["pilot"]
        by_model[model_label(info, same)] = a2
        if tag == F2_MODEL or (f2 is None and not info["pilot"]):
            f2 = (a2, m, model_label(info, same))
        out[tag] = dict(folder=f.name, config=info["meta"].get("config_name", info["config"]), subset=info["subset"],
                        n_pairs=len(m), overall=a2["overall"], cohens_d=a2["d"], midpoint=a2["midpoint"])
    kind = figure_kind(pilot)
    notes = ([] if same else ["MIXED SERVING CONFIGS: compare by Cohen's d"]) + (["includes the Qwen2.5-1.5B pilot"] if pilot else [])
    F.TITLE_NOTE = "; ".join(notes) or None
    made = [F.a2_models_line(by_model, REPO / "figures" / "WS3" / "A2_models.png", kind=kind, layer="last, pre-norm")]
    if f2 is not None:
        made.append(F.f2_combined(*f2[:2], by_model, REPO / "figures" / "WS3" / "F2.png", kind=kind, model=f2[2],
                                  layer="last, pre-norm"))
    F.TITLE_NOTE = None
    out["same_config"] = same
    ws3_data.save_json(out, REPO / "results" / "WS3_A2_models.json")
    print(f"[4] A2 across {len(by_model)} models ({'all full C1' if same else 'mixed configs'}) -> {', '.join(map(str, made))}")
    return made


def step_x5(folders, n_boot):
    from analysis import figures as F
    from analysis import geometry as g
    from analysis import ws3_data
    by_tag = {}
    for f in folders:
        info = folder_info(f)
        by_tag.setdefault(info["tag"], {}).setdefault(info["config"], []).append((f, info))
    table, out, made = {}, {}, []
    for tag, cfgs in by_tag.items():
        if "c1" not in cfgs or "c6" not in cfgs:
            continue
        # prefer the C1 run on the same subset as C6; otherwise the full C1 run (pairs are intersected either way)
        c6f, c6i = cfgs["c6"][0]
        c1f = next((f for f, i in cfgs["c1"] if i["subset"] == c6i["subset"]), cfgs["c1"][0][0])
        arrs, meta = aligned_wj([c1f, c6f], "prenorm")
        stab = g.direction_stability({REFERENCE: arrs[c1f], CONFIG_LABEL["c6"]: arrs[c6f]}, meta, REFERENCE,
                                     n_boot=n_boot)
        table[tag] = {c: v["cos_to_reference"] for c, v in stab.items()}
        out[tag] = {c: dict(cos_to_reference=v["cos_to_reference"], dp=v["dp"]) for c, v in stab.items()}
        out[tag]["n_pairs"] = len(meta)
        made.append(F.x5_dp_dotplot(stab, REPO / "figures" / "WS3" / f"X5_dp_{tag}.png", kind=figure_kind(c6i["pilot"]),
                                    model=tag, layer="last, pre-norm"))
    if not table:
        print("[5] skipped: X5 needs a c1 and a c6 folder for the same model")
        return made
    made.append(F.x5_cosine_heatmap(table, REPO / "figures" / "WS3" / "X5_cosine.png",
                                    kind=figure_kind(any(folder_info(f)["pilot"] for f in folders
                                                         if folder_info(f)["tag"] in table)),
                                    layer="last, pre-norm"))
    out["reference"] = f"{REFERENCE} (no vLLM R in the WS3 pipeline)"
    ws3_data.save_json(out, REPO / "results" / "WS3_X5.json")
    print(f"[5] X5 for {sorted(table)}: " + ", ".join(f"{t} cos {v[CONFIG_LABEL['c6']]:.4f}" for t, v in table.items()))
    return made


def sweep_views(meta):
    """Views of the layer sweep, shallow to deep, as (view, x label): every sweep layer (hidden_states index), then
    the last layer pre-norm. Empty when the folder has fewer than 2 sweep layers (nothing to sweep over)."""
    n = meta.get("num_hidden_layers")
    # hidden_states[n] is the post-norm output, not a sweep point. hidden_states[0] at prompt[-1] is the embedding of
    # the chat template's last token, the same vector in every prompt, so no r_hat can be fitted there.
    layers = [l for l in sorted(int(l) for l in meta.get("sweep_layers") or []) if l > 0 and (n is None or l < n)]
    if len(layers) < 2:
        return []
    return [(l, str(l)) for l in layers] + [("prenorm", f"{n} (last, pre-norm)")]


def step_sweep(folders, n_boot):
    from analysis import figures as F
    from analysis import geometry as g
    from analysis import ws3_data
    from analysis import ws3_pilot as wp
    out, made = {}, []
    for f in folders:
        info = folder_info(f)
        views = sweep_views(info["meta"])
        if not views:
            continue
        rows = {}
        for view, label in views:
            Hv, Ha, m = wp.pair_arrays(wp.load(f, view), "WildJailbreak")
            rows[label] = g.a2_layer_row(Hv, Ha, m, n_boot=n_boot)
        F.TITLE_NOTE = "EXPLORATORY (layer sweep, not pre-registered)"
        model = f"{info['tag']} [{info['meta'].get('config_name', info['config'])}]"
        made.append(F.a2_layer_sweep(rows, REPO / "figures" / "WS3" / f.name / "A2_layers.png",
                                     kind=figure_kind(info["pilot"]), model=model))
        F.TITLE_NOTE = None
        out[f.name] = dict(n_pairs=len(m), layers=rows)
        print(f"[5b] {f.name}: " + ", ".join(f"{l} d {r['d']['harmful'][0]:+.2f}/{r['d']['benign'][0]:+.2f}"
                                             for l, r in rows.items()))
    if not out:
        print("[5b] skipped: no folder has a layer sweep of 2+ layers")
        return made
    ws3_data.save_json(out, REPO / "results" / "WS3_layer_sweep.json")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--skip-previews", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    set_seed(0)
    from analysis import plotstyle as ps
    from analysis import prereg_tools
    ps.apply_style()
    print(f"pre-registration: {prereg_tools.apply_labels(REPO)['state']}")
    (REPO / "figures" / "WS3").mkdir(parents=True, exist_ok=True)
    (REPO / "results").mkdir(exist_ok=True)

    if a.skip_previews:
        print("[1] skipped (--skip-previews)")
    else:
        step_previews()
    step_task6()
    folders = find_folders(a.data_root)
    print(f"real extraction folders: {[f.name for f in folders] or 'none yet'}")
    made = step_folders(folders, a.n_boot)
    made += step_models(folders, a.n_boot)
    made += step_x5(folders, a.n_boot)
    made += step_sweep(folders, a.n_boot)
    if made:
        from analysis import contact_sheet as cs
        p = cs.contact_sheet(made, REPO / "figures" / "WS3" / "_real_contact_sheet.png", what="real data")
        print(f"[6] {len(made)} real figures; contact sheet {p}")
    else:
        print("[6] skipped: no real figures yet")


if __name__ == "__main__":
    main()
