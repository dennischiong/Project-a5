"""Task 6: pilot analyses on harvest/out (CPU). One script regenerates every pilot figure and the numbers file:

  figures/D0_pilot_data.png           pilot data at a glance: token counts, pairs per tactic, pairs per split
  figures/F2_pilot_<view>.png         A2: 2x2 projection histograms (harmful/benign x vanilla/adversarial), per layer view
  figures/F1_pilot_<view>.png         A1: displacement size per intent against the rough noise floor
  figures/A3_pilot.png                A3: vanilla-trained probe AUROC on held-out vanilla vs adversarial prompts
  figures/A2_layers_pilot.png         EXPLORATORY: separation and mean dp across layers, both token positions
  analysis/pilot_results.json         all numbers (no text)

Prints/saves only ids, counts and numbers (no text). PILOT NUMBERS, NOT RESULTS.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import lib  # noqa: E402
from analysis import plotstyle as ps  # noqa: E402
from analysis import prereg_tools  # noqa: E402
from seeds import set_seed  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (backend set by plotstyle)

OUT = REPO / "harvest" / "out"
FIG = REPO / "figures"
NA = "n/a"
# HF hidden_states = [embeddings, out layer 0, ..., out layer N-2, final_norm(out layer N-1)], so index -2 is one layer
# EARLIER than index -1, not the same layer before the norm. The true pre-norm last layer comes from a separate file
# (captured with a hook on the final norm in harvest/pilot_extract.py).


SKIPPED_TACTICS = "per-tactic results skipped: no tactic labels"


def has_tactic_labels(tactics) -> bool:
    """False when every tactic is 'unknown' (the pilot's --allow-missing-tactics override): per-tactic outputs are
    then skipped, while the overall A1/A2/A3 results still run."""
    return bool((pd.Series(tactics).astype(str) != "unknown").any())


def f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def fmt(x, spec=".3f"):
    """Format a number for printing; None/NaN/'n/a' print as n/a instead of crashing."""
    try:
        return NA if x is None or np.isnan(float(x)) else format(float(x), spec)
    except (TypeError, ValueError):
        return NA


def auroc_ci(y, s, n=2000, seed=0):
    """AUROC with a 95% bootstrap CI over pairs (resamples with a single class are skipped)."""
    y, s = np.asarray(y), np.asarray(s)
    if not 0 < y.sum() < len(y):
        return (NA, NA, NA)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(roc_auc_score(y[i], s[i]))
    return float(roc_auc_score(y, s)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def title(meta, layer, position, text):
    """Figure title per the VISUALS RULE: 'PILOT - not results | model | layer | position | what it shows'."""
    tag = ps.tag("PILOT", str(meta["model"]).split("/")[-1], layer, position)
    return tag.replace("PILOT", "PILOT - not results", 1) + f" | {text}"


def config_line(meta, extra=""):
    return (f"model {meta['model']} | backend {meta['backend']} | concurrency {meta.get('concurrency', '?')} | "
            f"dtype {meta['dtype']} | quantization {meta.get('quantization', '?')} | GPU {meta['gpu']}"
            + (f" | {extra}" if extra else "") + " | PILOT (~100 pairs), not results")


# ------------------------------------------------------------------------------------------------ figures
def fig_data_overview(pairs, meta):
    """D0: what the pilot data looks like (counts only)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.2, 1.3, 0.8]})
    ax = axes[0]
    bins = np.logspace(np.log10(max(1, pairs.token_count.min())), np.log10(pairs.token_count.max() + 1), 25)
    for form, color, ls in (("vanilla", ps.INK2, "-"), ("adversarial", ps.AQUA, "--")):
        ax.hist(pairs.loc[pairs.role == form, "token_count"], bins=bins, histtype="step", color=color, ls=ls, lw=2, label=form)
    ax.set_xscale("log")
    ax.set(title="Prompt length after the chat template", xlabel="tokens (log scale)", ylabel="prompts")
    ax.legend()

    ax = axes[1]
    adv = pairs[pairs.role == "adversarial"]
    counts = adv.groupby(["tactic", "intent"]).size().unstack(fill_value=0).reindex(columns=["harmful", "benign"], fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    if len(counts) > 12:  # fold the tail into Other instead of inventing more colours or rows
        counts = pd.concat([counts.iloc[:11], counts.iloc[11:].sum().to_frame("Other").T])
    y = np.arange(len(counts))[::-1]
    left = np.zeros(len(counts))
    for intent in ("harmful", "benign"):
        ax.barh(y, counts[intent], left=left, height=0.7, color=ps.INTENT_COLOR[intent], label=intent,
                edgecolor=ps.SURFACE, linewidth=1)
        left += counts[intent].to_numpy()
    ax.set_yticks(y, [t if len(t) <= 28 else t[:27] + "..." for t in counts.index], fontsize=7)
    if has_tactic_labels(adv["tactic"]):
        ax.set(title=f"Pairs per tactic (first listed tactic, {pairs.tactic.nunique()} distinct)", xlabel="pairs")
    else:
        ax.set(title="Pairs per tactic: NO tactic labels (all 'unknown', pilot override)", xlabel="pairs")
    ax.legend(loc="lower right")

    ax = axes[2]
    split = pairs[pairs.role == "vanilla"].groupby(["split", "intent"]).size().unstack(fill_value=0)
    split = split.reindex(["train", "validation", "test"], fill_value=0).reindex(columns=["harmful", "benign"], fill_value=0)
    x = np.arange(3)
    for k, intent in enumerate(("harmful", "benign")):
        bars = ax.bar(x + (k - 0.5) * 0.38, split[intent], width=0.36, color=ps.INTENT_COLOR[intent], label=intent)
        ax.bar_label(bars, fontsize=7, color=ps.INK2, padding=1)
    ax.set_xticks(x, ["train", "validation", "test"])
    ax.set(title="Pairs per split (70/15/15 by pair_id hash)", ylabel="pairs")
    ax.legend()
    ps.caption(fig, title(meta, "none (counts only)", "n/a", "D0 - pilot data at a glance (counts only, no text)"),
               config_line(meta))
    ps.save(fig, FIG / "D0_pilot_data.png")


def fig_f2(tag, lname, pv, pa, harm, tr, ev, meta):
    """A2 / F2: 2x2 projection histograms with shared bins; dashed line = H1 midpoint of the train means."""
    cells = {("harmful", "vanilla"): pv[ev & harm], ("harmful", "adversarial"): pa[ev & harm],
             ("benign", "vanilla"): pv[ev & ~harm], ("benign", "adversarial"): pa[ev & ~harm]}
    allv = np.concatenate(list(cells.values()))
    bins = np.linspace(allv.min(), allv.max(), 16)
    mid = (pv[tr & harm].mean() + pv[tr & ~harm].mean()) / 2
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True, sharey=True)
    for i, intent in enumerate(("harmful", "benign")):
        for j, form in enumerate(("vanilla", "adversarial")):
            ax, vals = axes[i, j], cells[(intent, form)]
            ax.hist(vals, bins=bins, color=ps.INTENT_COLOR[intent], hatch=ps.FORM_HATCH[form],
                    alpha=1.0 if form == "vanilla" else 0.55, edgecolor=ps.SURFACE, linewidth=1)
            ax.axvline(mid, color=ps.INK2, ls="--", lw=1)
            ax.set_title(f"{form} {intent}  (n={len(vals)}, mean {vals.mean():.2f})" if len(vals) else f"{form} {intent} (n=0)")
    axes[0, 0].annotate("H1 midpoint", xy=(mid, axes[0, 0].get_ylim()[1] * 0.92), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=ps.INK2)
    for ax in axes[1]:
        ax.set_xlabel("projection onto r_hat  (higher = looks more harmful)")
    for ax in axes[:, 0]:
        ax.set_ylabel("prompts")
    ps.caption(fig, title(meta, lname, "prompt[-1]", "F2 (A2) - projection onto r_hat, validation+test pairs"),
               config_line(meta, "position prompt[-1]; r_hat fit on vanilla TRAIN only"))
    ps.save(fig, FIG / f"F2_pilot_{tag}.png")


def floor_for(nfloor, tag):
    """The noise floor measured on this layer view, or None. Older noise_floor.json files (before per-view floors)
    only hold the post-norm last layer at the top level."""
    per_view = nfloor.get("per_view") or {}
    if tag in per_view:
        return per_view[tag]
    return nfloor if tag == "postnorm_last" else None


def fig_f1(tag, lname, cd, rl, harm, ev, floor, meta):
    """A1 / F1: displacement size per intent, log scale, with the rough noise floor of the same layer view
    (``floor`` from floor_for; None = not measured on this view, so not drawn)."""
    floor_applies = floor is not None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    rng = np.random.default_rng(0)
    for ax, vals, key, label in ((axes[0], cd, "cosine_distance", "cosine distance (1 - cos)"),
                                 (axes[1], rl, "relative_l2", "relative L2  ||h_adv - h_van|| / ||h_van||")):
        data = [vals[ev & harm], vals[ev & ~harm]]
        ax.boxplot(data, positions=[0, 1], widths=0.45, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=ps.SURFACE, edgecolor=ps.AXIS), medianprops=dict(color=ps.INK, lw=1.5),
                   whiskerprops=dict(color=ps.AXIS), capprops=dict(color=ps.AXIS))
        for k, (d, intent) in enumerate(zip(data, ("harmful", "benign"))):
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(d)), d, s=14, color=ps.INTENT_COLOR[intent], alpha=0.8, zorder=3)
        if floor_applies:
            fl = floor[key]["mean"]
            ax.axhline(fl, color=ps.INK2, ls="--", lw=1)
            ax.annotate("rough noise floor (bs 1 vs 16)", xy=(1.45, fl), fontsize=7, color=ps.INK2, va="bottom", ha="right")
        ax.set_yscale("log")
        ax.set_xticks([0, 1], [f"harmful pairs (n={len(data[0])})", f"benign pairs (n={len(data[1])})"])
        ax.set(title=label, ylabel="per-pair distance (log scale)")
    note = "noise floor of this layer view drawn" if floor_applies else "noise floor NOT drawn: it was not measured on this layer view"
    ps.caption(fig, title(meta, lname, "prompt[-1]", "F1 (A1) - how far each adversarial rewrite moves prompt[-1]"),
               config_line(meta, f"validation+test pairs; {note}; paraphrase baseline not available yet"))
    ps.save(fig, FIG / f"F1_pilot_{tag}.png")


def fig_a3(a3_by_view, meta):
    """A3: AUROC with bootstrap CIs, held-out vanilla vs held-out adversarial, per layer view."""
    views = list(a3_by_view)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(views))
    for k, (form, key) in enumerate((("vanilla", "auroc_vanilla_ci"), ("adversarial", "auroc_adversarial_ci"))):
        est = [a3_by_view[v][key][0] if isinstance(a3_by_view[v][key][0], float) else np.nan for v in views]
        lo = [a3_by_view[v][key][1] if isinstance(a3_by_view[v][key][1], float) else np.nan for v in views]
        hi = [a3_by_view[v][key][2] if isinstance(a3_by_view[v][key][2], float) else np.nan for v in views]
        err = np.array([np.subtract(est, lo), np.subtract(hi, est)])
        bars = ax.bar(x + (k - 0.5) * 0.36, est, width=0.34, color=ps.BLUE, alpha=1.0 if form == "vanilla" else 0.55,
                      hatch=ps.FORM_HATCH[form], edgecolor=ps.SURFACE, label=f"held-out {form} prompts")
        ax.errorbar(x + (k - 0.5) * 0.36, est, yerr=err, fmt="none", ecolor=ps.INK2, capsize=3, lw=1)
        ax.bar_label(bars, labels=[fmt(e, ".2f") for e in est], fontsize=7, color=ps.INK2, padding=8)
    ax.axhline(0.5, color=ps.MUTED, ls=":", lw=1)
    ax.annotate("chance", xy=(len(views) - 0.5, 0.5), fontsize=7, color=ps.MUTED, va="bottom", ha="right")
    ax.set_xticks(x, [v.split(" (")[0] for v in views], fontsize=7.5)
    ax.set(ylim=(0, 1.08), ylabel="AUROC, harmful vs benign")
    ax.legend(loc="lower left")
    ps.caption(fig, title(meta, "3 views (x-axis)", "prompt[-1]",
                          "A3 - harmfulness probe trained on vanilla prompts only, tested on held-out prompts"),
               config_line(meta, "prompt[-1]; StandardScaler + LogisticRegression(C=1) on vanilla TRAIN; 95% bootstrap CIs over pairs"))
    ps.save(fig, FIG / "A3_pilot.png")


def fig_layer_sweep(positions, pid, meta):
    """EXPLORATORY: per hidden_states index, refit r_hat on vanilla TRAIN; separation (AUROC of vanilla eval
    projections) and mean dp for harmful / benign pairs. Both token positions."""
    harm = (pid["intent"] == "harmful").to_numpy()
    tr, ev = (pid["split"] == "train").to_numpy(), (pid["split"] != "train").to_numpy()
    tmeta = pd.DataFrame({"intent": pid["intent"][tr].to_numpy(), "split": "train", "role": "vanilla"})
    sweep = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for pos, (Hv_all, Ha_all) in positions.items():
        n_hs = Hv_all.shape[1]
        sep, dph, dpb = [], [], []
        for layer in range(1, n_hs):  # skip the embedding output
            Hv, Ha = Hv_all[:, layer], Ha_all[:, layer]
            r_hat = lib.fit_direction_train(Hv[tr], tmeta)
            pv, dp = lib.project(Hv, r_hat), lib.project(Ha - Hv, r_hat)
            sep.append(roc_auc_score(harm[ev].astype(int), pv[ev]) if 0 < harm[ev].sum() < ev.sum() else np.nan)
            dph.append(dp[ev & harm].mean())
            dpb.append(dp[ev & ~harm].mean())
        layers = np.arange(1, n_hs)
        sweep[pos] = dict(layers=layers.tolist(), separation_auroc=sep, mean_dp_harmful=dph, mean_dp_benign=dpb)
        axes[0].plot(layers, sep, color=ps.POSITION_COLOR[pos], label=pos)
        style = "-" if pos == "last prefill" else "--"
        axes[1].plot(layers, dph, color=ps.ORANGE, ls=style, label=f"harmful pairs, {pos}")
        axes[1].plot(layers, dpb, color=ps.BLUE, ls=style, label=f"benign pairs, {pos}")
    axes[0].axhline(0.5, color=ps.MUTED, ls=":", lw=1)
    axes[0].set(title="Vanilla harmful vs benign separation along r_hat", xlabel="hidden_states index (last = after final norm)",
                ylabel="AUROC (validation+test)", ylim=(0.4, 1.02))
    axes[1].axhline(0, color=ps.AXIS, lw=1)
    axes[1].set(title="Mean projection shift dp = (h_adv - h_van) . r_hat", xlabel="hidden_states index (last = after final norm)",
                ylabel="mean dp  (negative = looks less harmful)")
    for ax in axes:
        ax.legend(fontsize=7)
    ps.caption(fig, title(meta, "every layer (x-axis)", "prompt[-1] + last instruction",
                          "EXPLORATORY layer sweep (not pre-registered; A6 is Kayson's) - r_hat refit per layer on vanilla TRAIN"),
               config_line(meta, "two token positions"))
    ps.save(fig, FIG / "A2_layers_pilot.png")
    return sweep


# ------------------------------------------------------------------------------------------------ main
def main():
    set_seed(0)
    ps.apply_style()
    FIG.mkdir(exist_ok=True)
    # pre-registration lock: if prereg.md changed after the lock, every title and output says EXPLORATORY
    prereg = prereg_tools.apply_labels(REPO)
    print(f"pre-registration: {prereg['state']}")
    print("WARNING: ~100 pairs => very wide CIs. These are PILOT numbers, not results.\n")
    pairs = lib.load_pairs(str(OUT / "pairs.parquet"))
    meta = json.loads((OUT / "pilot_meta.json").read_text())
    acts = lib.load_activations(str(OUT / "pilot_acts.safetensors"))
    nfloor = json.loads((OUT / "noise_floor.json").read_text())
    pid = pairs.drop_duplicates("pair_id").set_index("pair_id")[["intent", "tactic", "split"]]
    ids = list(pid.index)
    n_dec = next(iter(acts.values())).shape[0] - 1
    fig_data_overview(pairs, meta)

    views = [
        ("postnorm_last", f"post-final-norm, last layer {n_dec} (hidden_states[-1])", acts, -1),
        ("layer_minus2", f"no final norm, layer {n_dec - 1} (hidden_states[-2], one layer earlier)", acts, -2),
    ]
    pre_path = OUT / "pilot_prenorm_last.safetensors"
    if pre_path.exists():
        views.append(("prenorm_last", f"pre-final-norm, last layer {n_dec} (input to the final norm)",
                      lib.load_activations(str(pre_path)), 0))
    else:
        print("note: pilot_prenorm_last.safetensors not found; true pre-norm last layer skipped")
    label = "PILOT - not results" + (f" | {prereg_tools.EXPLORATORY}" if prereg["state"] == "CHANGED" else "")
    results = {"warning": "pilot numbers, not results", "label": label, "prereg": prereg, "meta": meta,
               "noise_floor": nfloor, "layers": {}}
    harm = (pid["intent"] == "harmful").to_numpy()
    tr = (pid["split"] == "train").to_numpy()
    ev = (pid["split"] != "train").to_numpy()  # validation + test
    tac = pid["tactic"].to_numpy()
    per_tactic_ok = has_tactic_labels(tac)
    if not per_tactic_ok:
        print(SKIPPED_TACTICS + " (every tactic is 'unknown'); overall A1/A2/A3 still run.\n")
    a3_by_view = {}

    for tag, lname, src, li in views:
        Hv = np.array([src[lib.make_key(p, "vanilla")][li] for p in ids])
        Ha = np.array([src[lib.make_key(p, "adversarial")][li] for p in ids])

        # ---- fit on vanilla TRAIN only
        tmeta = pd.DataFrame({"intent": pid["intent"][tr].to_numpy(), "split": "train", "role": "vanilla"})
        r_hat = lib.fit_direction_train(Hv[tr], tmeta)
        pv, pa = lib.project(Hv, r_hat), lib.project(Ha, r_hat)
        dp = lib.project(Ha - Hv, r_hat)

        # ---- SIGN CHECK (must pass or stop)
        m_h_tr, m_b_tr = pv[tr & harm].mean(), pv[tr & ~harm].mean()
        eval_gap = pv[ev & harm].mean() - pv[ev & ~harm].mean()
        if not (m_h_tr > m_b_tr and eval_gap > 0):
            sys.exit(f"STOP: sign check FAILED at {lname}: vanilla harmful does not project higher than vanilla benign "
                     f"(train gap {m_h_tr - m_b_tr:.3f}, eval gap {eval_gap:.3f}).")
        print(f"[{lname}] sign check PASS: train gap {m_h_tr - m_b_tr:.3f}, eval gap {eval_gap:.3f}")
        fig_f2(tag, lname, pv, pa, harm, tr, ev, meta)

        # ---- A2: H1 share and dp per tactic
        mid = (m_h_tr + m_b_tr) / 2  # midpoint of TRAIN vanilla means (fitted quantity; no leakage)
        h1 = (pa[ev & harm] < mid)
        h1_est = lib.bootstrap_ci(h1.astype(float), np.mean) if h1.size else (None,) * 3
        h1_van = (pv[ev & harm] < mid).mean() if (ev & harm).any() else None
        per_tactic = {} if per_tactic_ok else {"skipped": SKIPPED_TACTICS}
        for t in (sorted(set(tac[ev])) if per_tactic_ok else []):
            row = {}
            for intent_name, mask in (("harmful", harm), ("benign", ~harm)):
                v = dp[ev & mask & (tac == t)]
                if len(v) >= 2:
                    est, lo, hi = lib.bootstrap_ci(v, np.mean)
                    row[intent_name] = dict(n=int(len(v)), mean_dp=est, ci95=[lo, hi], cohens_d=f(lib.cohens_d(v)))
                else:
                    row[intent_name] = dict(n=int(len(v)), mean_dp=f(v.mean()) if len(v) else NA, ci95=NA, cohens_d=NA)
            per_tactic[t] = row
        overall = {}
        for intent_name, mask in (("harmful", harm), ("benign", ~harm)):
            v = dp[ev & mask]
            est, lo, hi = lib.bootstrap_ci(v, np.mean)
            overall[intent_name] = dict(n=int(len(v)), mean_dp=est, ci95=[lo, hi], cohens_d=f(lib.cohens_d(v)))

        # ---- A1: distances per tactic vs the noise floor of the same layer view (if it was measured there)
        cd, rl = lib.cosine_distance(Hv, Ha), lib.relative_l2(Hv, Ha)
        a1 = {} if per_tactic_ok else {"skipped": SKIPPED_TACTICS}
        for t in (sorted(set(tac[ev])) if per_tactic_ok else []):
            sel = ev & (tac == t)
            a1[t] = dict(n=int(sel.sum()), cosine_distance_mean=float(cd[sel].mean()), relative_l2_mean=float(rl[sel].mean()))
        a1_all = dict(cosine_distance=lib.bootstrap_ci(cd[ev], np.mean), relative_l2=lib.bootstrap_ci(rl[ev], np.mean))
        floor = floor_for(nfloor, tag)
        fig_f1(tag, lname, cd, rl, harm, ev, floor, meta)

        # ---- A3: logistic probe on vanilla train, evaluate on held-out
        sc = StandardScaler().fit(Hv[tr])
        clf = LogisticRegression(max_iter=2000, random_state=0).fit(sc.transform(Hv[tr]), harm[tr].astype(int))
        s_adv, s_van = clf.decision_function(sc.transform(Ha[ev])), clf.decision_function(sc.transform(Hv[ev]))
        y_ev = harm[ev].astype(int)
        van_ci, adv_ci = auroc_ci(y_ev, s_van), auroc_ci(y_ev, s_adv)
        a3 = dict(
            auroc_vanilla_heldout=van_ci[0], auroc_adversarial_heldout=adv_ci[0],
            auroc_vanilla_ci=van_ci, auroc_adversarial_ci=adv_ci,
            probe="StandardScaler + LogisticRegression(C=1.0, sklearn defaults) fitted on vanilla TRAIN only",
            auroc_adversarial_per_tactic=(
                {k: (NA if np.isnan(v) else v) for k, v in lib.auroc_per_group(y_ev, s_adv, tac[ev], 3).items()}
                if per_tactic_ok else {"skipped": SKIPPED_TACTICS}),
        )
        a3_by_view[lname] = a3

        results["layers"][lname] = dict(
            r_hat_train_pairs=int(tr.sum()), eval_pairs=int(ev.sum()),
            sign_check=dict(train_gap=float(m_h_tr - m_b_tr), eval_gap=float(eval_gap)),
            h1=dict(share_adv_harmful_below_midpoint=h1_est, share_vanilla_harmful_below_midpoint=f(h1_van), midpoint=float(mid), n=int(h1.size)),
            dp_overall=overall, dp_per_tactic=per_tactic, a1_overall=a1_all, a1_per_tactic=a1, a3=a3,
        )
        oh, ob = overall["harmful"], overall["benign"]
        print(f"  H1 share(adv harmful below midpoint)={fmt(h1_est[0], '.2f')} CI[{fmt(h1_est[1], '.2f')},{fmt(h1_est[2], '.2f')}] "
              f"n={h1.size}; vanilla-harmful share={fmt(h1_van, '.2f')}")
        print(f"  mean dp harmful={fmt(oh['mean_dp'])} CI[{fmt(oh['ci95'][0])},{fmt(oh['ci95'][1])}] d={fmt(oh['cohens_d'], '.2f')}; "
              f"benign={fmt(ob['mean_dp'])} CI[{fmt(ob['ci95'][0])},{fmt(ob['ci95'][1])}] d={fmt(ob['cohens_d'], '.2f')}")
        if floor is not None:
            print(f"  A1 cosine dist={fmt(a1_all['cosine_distance'][0], '.4g')} vs pilot-only rough floor "
                  f"{fmt(floor['cosine_distance']['mean'], '.4g')} (same layer view, bs1 vs bs16); "
                  f"rel L2={fmt(a1_all['relative_l2'][0], '.4g')} vs {fmt(floor['relative_l2']['mean'], '.4g')}. "
                  "Paraphrase baselines do not exist yet.")
        else:
            print(f"  A1 cosine dist={fmt(a1_all['cosine_distance'][0], '.4g')}; rel L2={fmt(a1_all['relative_l2'][0], '.4g')} "
                  "(no noise floor measured on this layer view). Paraphrase baselines do not exist yet.")
        print(f"  A3 AUROC vanilla={fmt(van_ci[0])} [{fmt(van_ci[1])},{fmt(van_ci[2])}] "
              f"adversarial={fmt(adv_ci[0])} [{fmt(adv_ci[1])},{fmt(adv_ci[2])}]\n")

    fig_a3(a3_by_view, meta)

    # ---- EXPLORATORY layer sweep at both positions
    positions = {"last prefill": (np.array([acts[lib.make_key(p, "vanilla")] for p in ids]),
                                  np.array([acts[lib.make_key(p, "adversarial")] for p in ids]))}
    instr_path = OUT / "pilot_acts_instr.safetensors"
    if instr_path.exists():
        acts_i = lib.load_activations(str(instr_path))
        positions["last instruction"] = (np.array([acts_i[lib.make_key(p, "vanilla")] for p in ids]),
                                         np.array([acts_i[lib.make_key(p, "adversarial")] for p in ids]))
    results["exploratory_layer_sweep"] = fig_layer_sweep(positions, pid, meta)

    def clean(o):  # NaN/inf -> null so the file is valid JSON
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.integer):
            return int(o)
        return o

    (REPO / "analysis" / "pilot_results.json").write_text(json.dumps(clean(results), indent=2, allow_nan=False))
    print("saved figures: D0_pilot_data, F2_pilot_<view>, F1_pilot_<view>, A3_pilot, A2_layers_pilot (exploratory); "
          "and analysis/pilot_results.json")


if __name__ == "__main__":
    main()
