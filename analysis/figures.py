"""WS3 figures (A2, A5, A7, A8, X5), one function per picture. All use plotstyle; every title starts with
'KIND | model | layer | position' (KIND = SYNTHETIC / PILOT / RESULT). Numbers only, never text.

Concept diagrams (the maths as a picture): concept_a2, concept_a5, concept_a7, concept_a8, concept_x5
A2 (= F2): a2_f2_histograms, a2_tactic_bars, a2_models_line; f2_combined = paper figure F2 (histograms + models)
A5: a5_cosine_heatmap, a5_pca_scatter, a5_principal_angles, a5_knn_stacked
A7: a7_style_content
A8: a8_spectrum, a8_angle_arc
X5: x5_cosine_heatmap, x5_dp_dotplot
Layer sweep (exploratory): a2_layer_sweep
ws3_summary: one page tiling the main figure of each analysis
render_all_synthetic(folder): draws every figure above on SYNTHETIC data (tests, previews, contact sheet)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

from analysis import geometry as g
from analysis import plotstyle as ps

TACTIC_COLORS = [ps.BLUE, ps.ORANGE, ps.AQUA, "#7a5195", "#b8860b", "#52514e", "#c94f7c", "#2f4b7c"]


def _ci_err(est, lo, hi):
    est, lo, hi = (np.asarray(x, float) for x in (est, lo, hi))
    return np.vstack([np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)])


TITLE_NOTE = None  # e.g. EXPLORATORY (not pre-registered): set by the notebook for A5 / A8 results


def _finish(fig, kind, model, layer, position, what, footer, path):
    note = f" | {TITLE_NOTE}" if TITLE_NOTE and kind != "DIAGRAM" else ""
    ps.caption(fig, ps.tag(kind, model, layer, position) + note + f" | {what}", footer)
    if kind == "SYNTHETIC":
        for ax in fig.axes:
            if ax.get_label() != "<colorbar>":
                ps.watermark(ax)
    return ps.savefig(fig, path)


def _arrow(ax, p0, p1, color, lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, ls=ls, zorder=4))


# ------------------------------------------------------------------------------------------------ concept diagrams
def concept_a2(path):
    """Two clouds (harmful, benign), the r_hat arrow between their means, and one pair's dp as the shadow of its
    displacement on r_hat."""
    ps.apply_style()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    H = rng.normal([2.4, 1.6], 0.45, (40, 2))
    B = rng.normal([-0.6, 0.2], 0.45, (40, 2))
    ax.scatter(*H.T, s=12, color=ps.ORANGE, alpha=0.6, label="vanilla harmful prompts")
    ax.scatter(*B.T, s=12, color=ps.BLUE, alpha=0.6, label="vanilla benign prompts")
    mh, mb = H.mean(0), B.mean(0)
    r = (mh - mb) / np.linalg.norm(mh - mb)
    # the r_hat line through the benign mean, extended both ways (the ruler every prompt is projected onto)
    ax.plot(*np.array([mb - 1.5 * r, mh + 1.2 * r]).T, color=ps.MUTED, lw=1, ls=":")
    _arrow(ax, mb, mh, ps.INK, lw=2.4)
    ax.annotate("r_hat: benign mean -> harmful mean", xy=mb + 0.3 * (mh - mb), xytext=(-3.9, 2.4), fontsize=7.5,
                color=ps.INK, arrowprops=dict(arrowstyle="->", color=ps.INK2, lw=0.8))
    hv = np.array([1.9, 3.3])
    ha = hv + np.array([-1.4, -0.6])
    _arrow(ax, hv, ha, ps.AQUA, lw=2)
    ax.scatter(*hv, s=40, color=ps.ORANGE, edgecolor=ps.INK, zorder=5)
    ax.scatter(*ha, s=40, color="white", edgecolor=ps.ORANGE, lw=1.5, zorder=5)
    ax.text(*(hv + [0.1, 0.1]), "h_van (one harmful pair)", fontsize=7)
    ax.text(*(ha + [-0.2, 0.18]), "h_adv", fontsize=7)
    # shadows: drop both points straight onto the r_hat line; dp is the distance between the two feet
    fv = mb + ((hv - mb) @ r) * r
    fa = mb + ((ha - mb) @ r) * r
    for p, f in ((hv, fv), (ha, fa)):
        ax.plot(*np.array([p, f]).T, color=ps.AQUA, lw=0.9, ls="--")
    _arrow(ax, fv, fa, ps.AQUA, lw=3.2)
    dp = (ha - hv) @ r
    ax.annotate(f"dp = (h_adv - h_van) . r_hat = {dp:.2f}\n(the shadow of the move on r_hat;\nnegative: the rewrite "
                "looks less harmful)", xy=(fv + fa) / 2, xytext=(1.0, -1.1), fontsize=7.2, color=ps.INK,
                arrowprops=dict(arrowstyle="->", color=ps.AQUA, lw=0.9))
    ax.set(xlabel="hidden-state direction 1 (made-up 2-D sketch)", ylabel="hidden-state direction 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left", fontsize=7)
    return _finish(fig, "DIAGRAM", "sketch", "any", "prompt[-1]", "A2 concept: r_hat and dp", "a picture of the maths, not data", path)


def concept_a5(path):
    """Tactic arrows from a common origin and the angle between two of them."""
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    angs = [20, 70, 100, 160]
    names = ["roleplay", "persona", "obfuscation", "hypothetical"]
    for k, (a, n) in enumerate(zip(angs, names)):
        v = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) * (1.6 + 0.2 * k)
        _arrow(ax, (0, 0), v, TACTIC_COLORS[k], lw=2.4)
        ax.text(*(v * 1.08), n, fontsize=8, color=TACTIC_COLORS[k], ha="center")
    ax.add_patch(Arc((0, 0), 1.2, 1.2, theta1=20, theta2=70, color=ps.INK, lw=1.4))
    ax.text(0.75, 0.55, "angle = 50 deg\n(cosine 0.64)", fontsize=7.5)
    ax.set(xlim=(-2.3, 2.3), ylim=(-0.4, 2.4), xlabel="direction 1 (sketch)", ylabel="direction 2 (sketch)")
    ax.set_aspect("equal")
    ax.text(-2.2, -0.3, "each arrow = one tactic's mean displacement; small angle = tactics move the state the same way",
            fontsize=6.8, color=ps.INK2)
    return _finish(fig, "DIAGRAM", "sketch", "any", "prompt[-1]", "A5 concept: tactic directions and the angle between them",
                   "a picture of the maths, not data", path)


def concept_a7(path):
    """One harmful tactic's move along r_hat split into a style part and a content part."""
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    style, content = -0.8, -1.2
    ax.barh([1], [style], color=ps.BLUE, height=0.5, label="style part = benign pairs' dp (same tactic)")
    ax.barh([1], [content], left=[style], color=ps.ORANGE, height=0.5, hatch="///", edgecolor=ps.SURFACE,
            label="content part = harmful dp - benign dp")
    ax.barh([0], [style + content], color=ps.INK2, height=0.5, label="harmful pairs' total dp")
    ax.axvline(0, color=ps.INK, lw=1)
    ax.text(style / 2, 1, "style", ha="center", va="center", color="white", fontsize=8)
    ax.text(style + content / 2, 1, "content", ha="center", va="center", color="white", fontsize=8)
    ax.set_yticks([0, 1], ["harmful total", "split"])
    ax.set(xlabel="move along r_hat (dp, made-up units)", xlim=(-2.4, 0.4), ylim=(-0.5, 1.5))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.8, ncol=2)
    return _finish(fig, "DIAGRAM", "sketch", "any", "prompt[-1]", "A7 concept: style part + content part = total move",
                   "a picture of the maths, not data", path)


def concept_a8(path):
    """Singular values falling off: a few big directions = low-rank movement."""
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    k = np.arange(1, 21)
    low = 10 * np.exp(-0.9 * (k - 1)) + 0.05
    flat = np.full(20, 2.5)
    ax.plot(k, low, marker="o", color=ps.ORANGE, label="low rank: movement uses ~2 directions")
    ax.plot(k, flat, marker="s", ls="--", color=ps.BLUE, label="full rank: movement spread over all directions")
    ax.set_yscale("log")
    ax.set(xlabel="direction number k (sorted, biggest first)", ylabel="singular value (log scale)")
    ax.legend(fontsize=7)
    return _finish(fig, "DIAGRAM", "sketch", "any", "prompt[-1]", "A8 concept: how fast singular values fall",
                   "a picture of the maths, not data", path)


def concept_x5(path):
    """r_hat under the reference config and under another config, and the angle between them."""
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    _arrow(ax, (0, 0), (2, 0.3), ps.INK, lw=2.4)
    _arrow(ax, (0, 0), (1.85, 0.85), ps.ORANGE, lw=2.4, ls="--")
    ax.text(2.05, 0.25, "r_hat (reference R:\nbf16, vLLM, batch 64)", fontsize=7.5)
    ax.text(1.9, 0.9, "r_hat refitted on\n4-bit weights", fontsize=7.5, color=ps.ORANGE)
    ax.add_patch(Arc((0, 0), 1.6, 1.6, theta1=8.5, theta2=24.7, color=ps.INK2))
    ax.text(0.95, 0.25, "cosine 0.96", fontsize=7.5)
    ax.set(xlim=(-0.2, 3.3), ylim=(-0.3, 1.4), xlabel="direction 1 (sketch)", ylabel="direction 2 (sketch)")
    ax.set_aspect("equal")
    return _finish(fig, "DIAGRAM", "sketch", "any", "prompt[-1]", "X5 concept: does the harmfulness direction move "
                   "under another serving config?", "a picture of the maths, not data", path)


# ------------------------------------------------------------------------------------------------ A2
def a2_f2_histograms(a2, meta, path, kind, model, layer, position="prompt[-1]"):
    """F2: 2x2 projection histograms on held-out pairs, dashed line = train midpoint."""
    ps.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    _f2_hists(axes, a2, meta)
    return _finish(fig, kind, model, layer, position, "F2 (A2): projection onto r_hat, held-out pairs",
                   "r_hat fitted on vanilla TRAIN pairs only; dashed line = midpoint of the train means", path)


def _f2_hists(axes, a2, meta):
    """Draw the F2 2x2 histograms into a 2x2 array of axes."""
    harm = (meta["intent"] == "harmful").to_numpy()
    ev = (meta["split"] != "train").to_numpy()
    pv, pa, mid = a2["pv"], a2["pa"], a2["midpoint"]
    cells = {("harmful", "vanilla"): pv[ev & harm], ("harmful", "adversarial"): pa[ev & harm],
             ("benign", "vanilla"): pv[ev & ~harm], ("benign", "adversarial"): pa[ev & ~harm]}
    allv = np.concatenate([v for v in cells.values() if len(v)])
    bins = np.linspace(allv.min(), allv.max(), 22)
    for i, intent in enumerate(("harmful", "benign")):
        for j, form in enumerate(("vanilla", "adversarial")):
            ax, v = axes[i, j], cells[(intent, form)]
            ax.hist(v, bins=bins, color=ps.INTENT_COLOR[intent], hatch=ps.FORM_HATCH[form], alpha=1 if form == "vanilla" else 0.6,
                    edgecolor=ps.SURFACE)
            ax.axvline(mid, color=ps.INK2, ls="--", lw=1)
            ax.set_title(f"{form} {intent} (n={len(v)}, mean {v.mean():.2f})" if len(v) else f"{form} {intent} (n=0)")
    for ax in axes[1]:
        ax.set_xlabel("projection onto r_hat (higher = looks more harmful)")
    for ax in axes[:, 0]:
        ax.set_ylabel("prompts (count)")
    axes[0, 0].text(mid, axes[0, 0].get_ylim()[1] * 0.9, " train midpoint (H1 line)", fontsize=7, color=ps.INK2)


def a2_tactic_bars(a2, path, kind, model, layer, position="prompt[-1]"):
    """Mean dp per tactic with 95% CI error bars, harmful and benign side by side."""
    ps.apply_style()
    tacs = list(a2["per_tactic"])
    x = np.arange(len(tacs))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(tacs) + 3), 4))
    for k, intent in enumerate(("harmful", "benign")):
        est = [a2["per_tactic"][t][intent][0] for t in tacs]
        lo = [a2["per_tactic"][t][intent][1] for t in tacs]
        hi = [a2["per_tactic"][t][intent][2] for t in tacs]
        ax.bar(x + (k - 0.5) * 0.38, est, 0.36, color=ps.INTENT_COLOR[intent], label=f"{intent} pairs", hatch="///",
               edgecolor=ps.SURFACE)
        ax.errorbar(x + (k - 0.5) * 0.38, est, yerr=_ci_err(est, lo, hi), fmt="none", ecolor=ps.INK2, capsize=3, lw=1,
                    label="95% CI" if k == 0 else None)
    ax.axhline(0, color=ps.INK, lw=1)
    ax.set_xticks(x, tacs, rotation=20, ha="right", fontsize=7.5)
    ax.set(ylabel="mean dp per pair (negative = looks less harmful)")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    return _finish(fig, kind, model, layer, position, "A2: mean dp per tactic with 95% CIs (held-out pairs)",
                   "bootstrap over pairs, 2000 resamples, seed 0", path)


def a2_models_line(by_model, path, kind, layer, position="prompt[-1]"):
    """Mean dp per model (harmful and benign) with 95% CIs. by_model: {model: a2 dict}. When every a2 dict also has
    d ({intent: (d, lo, hi)}), a second panel shows Cohen's d, the scale-free number to compare models by."""
    ps.apply_style()
    panels = _model_panels(by_model)
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 if len(panels) == 1 else 11, 3.8), squeeze=False)
    _models_lines(axes[0], by_model, panels)
    return _finish(fig, kind, "several (x-axis)", layer, position, "A2: mean dp per model", "held-out pairs", path)


def _model_panels(by_model):
    panels = [("overall", "mean dp per pair (model's own units)")]
    if all("d" in a2 for a2 in by_model.values()):
        panels.append(("d", "Cohen's d of dp (scale-free)"))
    return panels


def _models_lines(axes, by_model, panels):
    """Draw one A2-across-models panel per (key, ylabel) in panels."""
    models = list(by_model)
    x = np.arange(len(models))
    for ax, (key, ylabel) in zip(axes, panels):
        for intent, ls in (("harmful", "-"), ("benign", "--")):
            est = [by_model[m][key][intent][0] for m in models]
            lo = [by_model[m][key][intent][1] for m in models]
            hi = [by_model[m][key][intent][2] for m in models]
            ax.errorbar(x, est, yerr=_ci_err(est, lo, hi), color=ps.INTENT_COLOR[intent], ls=ls, marker="o",
                        capsize=4, label=f"{intent} pairs (95% CI)")
        ax.axhline(0, color=ps.INK, lw=1)
        ax.set_xticks(x, models, fontsize=7.5)
        ax.set(ylabel=ylabel, xlabel="model")
        ax.legend(fontsize=7)


def f2_combined(a2, meta, by_model, path, kind, model, layer, position="prompt[-1]"):
    """Paper figure F2 in one file: (a) the 2x2 histograms of one model, (b) mean dp and Cohen's d per model."""
    ps.apply_style()
    panels = _model_panels(by_model)
    fig = plt.figure(figsize=(11, 9.5))
    gs_a = fig.add_gridspec(2, 2, top=0.88, bottom=0.47, hspace=0.35, wspace=0.08)
    gs_b = fig.add_gridspec(1, len(panels), top=0.36, bottom=0.08, wspace=0.25)
    ax_a = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            ax_a[i, j] = fig.add_subplot(gs_a[i, j], sharex=ax_a[0, 0] if (i or j) else None,
                                         sharey=ax_a[0, 0] if (i or j) else None)
    for ax in ax_a[0]:
        ax.tick_params(labelbottom=False)
    for ax in ax_a[:, 1]:
        ax.tick_params(labelleft=False)
    _f2_hists(ax_a, a2, meta)
    ax_b = [fig.add_subplot(gs_b[0, k]) for k in range(len(panels))]
    _models_lines(ax_b, by_model, panels)
    ax_a[0, 0].text(-0.12, 1.15, f"(a) {model}", transform=ax_a[0, 0].transAxes, fontsize=11, fontweight="bold")
    ax_b[0].text(-0.12, 1.08, "(b) all models", transform=ax_b[0].transAxes, fontsize=11, fontweight="bold")
    return _finish(fig, kind, f"(a) {model}, (b) several", layer, position, "F2 (A2): projection onto r_hat",
                   "held-out pairs; r_hat fitted on vanilla TRAIN pairs; dashed line = midpoint of the train means; "
                   "95% bootstrap CIs over pairs", path)


# ------------------------------------------------------------------------------------------------ A5
def a5_cosine_heatmap(means, path, kind, model, layer, position="prompt[-1]"):
    """Cosine similarity between tactics' mean displacements, ordered by clustering, values printed in cells."""
    ps.apply_style()
    tacs = list(means)
    C = g.cosine_matrix(np.array([means[t] for t in tacs]))
    order = g.cluster_order(C)
    C = C[np.ix_(order, order)]
    tacs = [tacs[i] for i in order]
    fig, ax = plt.subplots(figsize=(1.0 * len(tacs) + 3, 0.8 * len(tacs) + 2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(tacs)):
        for j in range(len(tacs)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(C[i, j]) > 0.6 else ps.INK)
    ax.set_xticks(range(len(tacs)), tacs, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(len(tacs)), tacs, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="cosine similarity (1 = same direction)")
    return _finish(fig, kind, model, layer, position, "A5: cosine similarity between tactics' mean displacements",
                   "rows/columns ordered by average-linkage clustering on 1 - cosine", path)


def a5_pca_scatter(means, path, kind, model, layer, position="prompt[-1]"):
    """2-D PCA of the tactics' mean displacements, one labelled dot per tactic."""
    ps.apply_style()
    tacs = list(means)
    coords, share = g.pca_2d(np.array([means[t] for t in tacs]))
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for k, t in enumerate(tacs):
        ax.scatter(*coords[k], s=60, color=TACTIC_COLORS[k % len(TACTIC_COLORS)], zorder=3)
        ax.annotate(t, coords[k], xytext=(5, 4), textcoords="offset points", fontsize=7.5)
    ax.axhline(0, color=ps.AXIS, lw=0.8)
    ax.axvline(0, color=ps.AXIS, lw=0.8)
    ax.set(xlabel=f"PC1 ({share[0]:.0%} of the spread between tactics)", ylabel=f"PC2 ({share[1]:.0%})")
    return _finish(fig, kind, model, layer, position, "A5: tactics' mean displacements, flattened to 2-D (PCA)",
                   "close dots = tactics that move the hidden state in similar ways", path)


def a5_principal_angles(angles, path, kind, model, layer, position="prompt[-1]"):
    """Bar chart of the smallest principal angle between pairs of tactic subspaces. angles: {A vs B: degrees}."""
    ps.apply_style()
    names = list(angles)
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(names) + 3), 3.8))
    bars = ax.bar(range(len(names)), [angles[n] for n in names], color=ps.BLUE)
    ax.bar_label(bars, fmt="%.0f", fontsize=7, color=ps.INK2)
    ax.axhline(90, color=ps.MUTED, ls=":", lw=1)
    ax.text(len(names) - 0.5, 90, "90 = unrelated", ha="right", va="bottom", fontsize=7, color=ps.MUTED)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7)
    ax.set(ylabel="smallest principal angle (degrees)", ylim=(0, 100))
    return _finish(fig, kind, model, layer, position, "A5: principal angles between tactic subspaces (k = 1)",
                   "0 degrees = two tactics share a direction", path)


def a5_knn_stacked(comp, path, kind, model, layer, position="prompt[-1]", k=10):
    """Stacked bar: for adversarial harmful prompts of each tactic, which tactics their k nearest neighbours have."""
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(comp) + 3), 4))
    left = np.zeros(len(comp))
    for j, t in enumerate(comp.columns):
        ax.barh(range(len(comp)), comp[t].to_numpy(), left=left, color=TACTIC_COLORS[j % len(TACTIC_COLORS)], label=t,
                edgecolor=ps.SURFACE)
        left += comp[t].to_numpy()
    ax.set_yticks(range(len(comp)), comp.index, fontsize=7.5)
    ax.set(xlabel=f"share of the {k} nearest neighbours (cosine), per query tactic", xlim=(0, 1))
    ax.legend(fontsize=6.5, ncol=min(4, len(comp.columns)), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return _finish(fig, kind, model, layer, position, "A5: k-NN composition of adversarial harmful prompts",
                   "a bar mostly in its own colour = that tactic forms its own cluster", path)


# ------------------------------------------------------------------------------------------------ A7
def a7_style_content(sc, path, kind, model, layer, position="prompt[-1]"):
    """Grouped bars per tactic: style part vs content part, with 95% CIs and a zero line."""
    ps.apply_style()
    tacs = [t for t in sc if t != "_pooled"] + (["_pooled"] if "_pooled" in sc else [])
    x = np.arange(len(tacs))
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(tacs) + 3), 4))
    for k, (part, color, hatch) in enumerate((("style", ps.BLUE, None), ("content", ps.ORANGE, "///"))):
        est = [sc[t][part][0] for t in tacs]
        lo = [sc[t][part][1] for t in tacs]
        hi = [sc[t][part][2] for t in tacs]
        ax.bar(x + (k - 0.5) * 0.38, est, 0.36, color=color, hatch=hatch, edgecolor=ps.SURFACE, label=f"{part} part")
        ax.errorbar(x + (k - 0.5) * 0.38, est, yerr=_ci_err(est, lo, hi), fmt="none", ecolor=ps.INK2, capsize=3, lw=1,
                    label="95% CI" if k == 0 else None)
    ax.axhline(0, color=ps.INK, lw=1)
    ax.set_xticks(x, [t.replace("_pooled", "ALL (pooled)") for t in tacs], rotation=20, ha="right", fontsize=7.5)
    ax.set(ylabel="move along r_hat (dp units)")
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    return _finish(fig, kind, model, layer, position, "A7: style part vs content part per tactic",
                   "style = benign pairs' mean dp; content = harmful minus benign; bootstrap within tactic", path)


# ------------------------------------------------------------------------------------------------ A8
def a8_spectrum(s, path, kind, model, layer, position="prompt[-1]", top=40):
    """Singular-value spectrum of the stacked displacements (log y), effective rank marked."""
    ps.apply_style()
    s_all = np.asarray(s)
    er, k90 = g.effective_rank(s_all), g.energy_rank(s_all, 0.9)
    s = s_all[:top]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(np.arange(1, len(s) + 1), s, marker="o", ms=3, color=ps.BLUE, label="singular value")
    ax.axvline(er, color=ps.ORANGE, ls="--", lw=1.5, label=f"effective rank (entropy) = {er:.1f}")
    ax.axvline(k90, color=ps.AQUA, ls=":", lw=1.8, label=f"directions holding 90% of the movement = {k90}")
    ax.set_yscale("log")
    ax.set(xlabel="direction number k (biggest first)", ylabel="singular value (log scale)")
    ax.legend(fontsize=7)
    return _finish(fig, kind, model, layer, position, "A8: singular-value spectrum of the displacements",
                   f"first {len(s)} values; effective rank = exp(entropy of normalised singular values)", path)


def a8_angle_arc(angle_deg, path, kind, model, layer, position="prompt[-1]"):
    """Annotated arc: the angle between the harmfulness direction r_hat and the refusal direction."""
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    th = np.radians(angle_deg)
    _arrow(ax, (0, 0), (1.8, 0), ps.ORANGE, lw=2.4)
    _arrow(ax, (0, 0), (1.8 * np.cos(th), 1.8 * np.sin(th)), ps.BLUE, lw=2.4, ls="--")
    ax.add_patch(Arc((0, 0), 1.0, 1.0, theta1=0, theta2=angle_deg, color=ps.INK, lw=1.4))
    ax.text(0.62 * np.cos(th / 2), 0.62 * np.sin(th / 2), f"{angle_deg:.0f} deg", fontsize=9, color=ps.INK)
    ax.text(1.85, -0.05, "harmfulness\nr_hat", fontsize=7.5, color=ps.ORANGE, va="top")
    ax.text(1.85 * np.cos(th), 1.85 * np.sin(th) + 0.05, "refusal direction", fontsize=7.5, color=ps.BLUE)
    ax.set(xlim=(-1.9, 2.6), ylim=(-0.5, 2.1))
    ax.set_aspect("equal")
    ax.axis("off")
    return _finish(fig, kind, model, layer, position, f"A8: angle between harmfulness and refusal directions = {angle_deg:.0f} deg",
                   "0 = the same direction; 90 = unrelated (Zhao et al.: encoded separately)", path)


# ------------------------------------------------------------------------------------------------ X5
def x5_cosine_heatmap(table, path, kind, layer, position="prompt[-1]"):
    """Heat map of cosine(r_hat under config, r_hat under reference) for config x model.
    table: {model: {config: cosine}}."""
    ps.apply_style()
    models = list(table)
    cfgs = sorted({c for m in models for c in table[m]})
    M = np.array([[table[m].get(c, np.nan) for c in cfgs] for m in models])
    fig, ax = plt.subplots(figsize=(1.3 * len(cfgs) + 3, 0.7 * len(models) + 2.2))
    cmap = plt.get_cmap("viridis").with_extremes(bad=ps.GRID)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=min(0.8, np.nanmin(M)), vmax=1)
    for i in range(len(models)):
        for j in range(len(cfgs)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] < 0.95 else ps.INK)
    ax.set_xticks(range(len(cfgs)), cfgs, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)), models, fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="cosine to the reference r_hat")
    return _finish(fig, kind, "several (rows)", layer, position, "X5: does r_hat stay put across serving configs?",
                   "grey = config not run for that model", path)


def x5_dp_dotplot(stab, path, kind, model, layer, position="prompt[-1]"):
    """Dot plot of mean dp (harmful, benign) under each config with 95% CI whiskers."""
    ps.apply_style()
    cfgs = list(stab)
    y = np.arange(len(cfgs))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(cfgs) + 1.8))
    for k, intent in enumerate(("harmful", "benign")):
        est = [stab[c]["dp"][intent][0] for c in cfgs]
        lo = [stab[c]["dp"][intent][1] for c in cfgs]
        hi = [stab[c]["dp"][intent][2] for c in cfgs]
        ax.errorbar(est, y + (k - 0.5) * 0.25, xerr=_ci_err(est, lo, hi), fmt="o", color=ps.INTENT_COLOR[intent],
                    capsize=3, label=f"{intent} pairs (95% CI)")
    ax.axvline(0, color=ps.INK, lw=1)
    ax.set_yticks(y, cfgs, fontsize=7.5)
    ax.set(xlabel="mean dp per pair (negative = looks less harmful)")
    ax.legend(fontsize=7)
    return _finish(fig, kind, model, layer, position, "X5: dp under each serving config",
                   "r_hat refitted on each config's vanilla TRAIN pairs", path)


# ------------------------------------------------------------------------------------------------ layer sweep
def a2_layer_sweep(rows, path, kind, model, position="prompt[-1]"):
    """A2 across layers (exploratory). rows: {layer label: geometry.a2_layer_row dict}, shallow to deep. Left: Cohen's
    d of held-out dp (harmful, benign); right: share of test harmful prompts below the midpoint, adversarial vs vanilla."""
    ps.apply_style()
    labels = list(rows)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for intent, ls in (("harmful", "-"), ("benign", "--")):
        est, lo, hi = (np.array([rows[l]["d"][intent][k] for l in labels]) for k in range(3))
        axes[0].errorbar(x, est, yerr=_ci_err(est, lo, hi), color=ps.INTENT_COLOR[intent], ls=ls, marker="o",
                         capsize=4, label=f"{intent} pairs (95% CI)")
    axes[0].axhline(0, color=ps.INK, lw=1)
    axes[0].set(ylabel="Cohen's d of dp (scale-free)")
    for key, color, ls, name in (("share_adv", ps.ORANGE, "-", "adversarial"), ("share_van", ps.MUTED, "--", "vanilla")):
        est, lo, hi = (np.array([rows[l]["h1"][key][k] for l in labels]) for k in range(3))
        axes[1].errorbar(x, est, yerr=_ci_err(est, lo, hi), color=color, ls=ls, marker="o", capsize=4,
                         label=f"{name} harmful (95% CI)")
    axes[1].axhline(0.5, color=ps.INK, lw=1, ls=":")
    axes[1].set(ylabel="share below the midpoint (H1 line)", ylim=(0, 1))
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set(xlabel="layer (hidden_states index)")
        ax.legend(fontsize=7)
    return _finish(fig, kind, model, "sweep (x-axis)", position, "A2 across layers",
                   "r_hat refitted per layer on vanilla TRAIN pairs; held-out pairs (d), test pairs (shares)", path)


# ------------------------------------------------------------------------------------------------ summary + synthetic
def ws3_summary(pngs, path, kind):
    """One page tiling the main figure of each analysis. pngs: {label: path}."""
    import matplotlib.image as mpimg

    ps.apply_style()
    items = [(k, v) for k, v in pngs.items() if v and Path(v).exists()]
    n = len(items)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (label, p) in zip(axes, items):
        ax.imshow(mpimg.imread(p))
        ax.set_title(label, fontsize=9, loc="left")
    ps.caption(fig, ps.tag(kind, "see panels", "see panels", "see panels") + " | WS3 summary: main figure of A2, A5, A7, A8, X5",
               "each panel is the saved PNG of that analysis; read the panel titles for model, layer and label")
    return ps.savefig(fig, path)


def render_all_synthetic(folder):
    """Draw every WS3 figure on SYNTHETIC data with a planted answer. Returns the list of PNG paths."""
    folder = Path(folder)
    syn = g.synthetic_ws3(configs={"R (bf16, vLLM, c64)": 0, "C1 HF bf16 bs1": 1, "C3 fp16": 2, "C6 NF4 4-bit": 20})
    Hv, Ha, meta = syn["Hv"], syn["Ha"], syn["meta"]
    a2 = g.a2_projection(Hv, Ha, meta)
    kw = dict(kind="SYNTHETIC", model="synthetic", layer="last")
    harm = (meta["intent"] == "harmful").to_numpy()
    delta = Ha - Hv
    means = g.mean_displacements(delta, meta["tactic"], mask=harm)
    tacs = list(means)
    angles = {f"{a} vs {b}": float(g.principal_angles(g.tactic_subspace(delta[harm & (meta.tactic == a).to_numpy()]),
                                                      g.tactic_subspace(delta[harm & (meta.tactic == b).to_numpy()]))[0])
              for i, a in enumerate(tacs) for b in tacs[i + 1:]}
    comp = g.knn_composition(delta[harm], meta["tactic"][harm], k=10)
    sc = g.style_content(a2["dp"], meta["tactic"], meta["intent"])
    s = g.singular_values(delta)
    stab = g.direction_stability(syn["by_config"], meta, reference="R (bf16, vLLM, c64)")
    other = g.synthetic_ws3(seed=5, style=-0.4, content=-0.9)
    by_model = {"llama31-8b": a2, "qwen25-7b": g.a2_projection(other["Hv"], other["Ha"], other["meta"])}
    table = {"llama31-8b": {c: v["cos_to_reference"] for c, v in stab.items()},
             "qwen25-7b": {c: v["cos_to_reference"] ** 1.5 for c, v in list(stab.items())[:3]}}
    out = [concept_a2(folder / "WS3_A2_concept.png"), concept_a5(folder / "WS3_A5_concept.png"),
           concept_a7(folder / "WS3_A7_concept.png"), concept_a8(folder / "WS3_A8_concept.png"),
           concept_x5(folder / "WS3_X5_concept.png"),
           a2_f2_histograms(a2, meta, folder / "WS3_A2_F2_SYNTHETIC.png", **kw),
           a2_tactic_bars(a2, folder / "WS3_A2_tactics_SYNTHETIC.png", **kw),
           a2_models_line(by_model, folder / "WS3_A2_models_SYNTHETIC.png", kind="SYNTHETIC", layer="last"),
           a5_cosine_heatmap(means, folder / "WS3_A5_cosine_SYNTHETIC.png", **kw),
           a5_pca_scatter(means, folder / "WS3_A5_pca_SYNTHETIC.png", **kw),
           a5_principal_angles(angles, folder / "WS3_A5_angles_SYNTHETIC.png", **kw),
           a5_knn_stacked(comp, folder / "WS3_A5_knn_SYNTHETIC.png", **kw),
           a7_style_content(sc, folder / "WS3_A7_style_content_SYNTHETIC.png", **kw),
           a8_spectrum(s, folder / "WS3_A8_spectrum_SYNTHETIC.png", **kw),
           a8_angle_arc(63.0, folder / "WS3_A8_angle_SYNTHETIC.png", **kw),
           x5_cosine_heatmap(table, folder / "WS3_X5_cosine_SYNTHETIC.png", kind="SYNTHETIC", layer="last"),
           x5_dp_dotplot(stab, folder / "WS3_X5_dp_SYNTHETIC.png", **kw),
           f2_combined(a2, meta, by_model, folder / "WS3_F2_SYNTHETIC.png", **kw),
           a2_layer_sweep({f"{l}": g.a2_layer_row(s_["Hv"], s_["Ha"], s_["meta"], n_boot=200)
                           for l, s_ in ((0, g.synthetic_ws3(seed=7, style=0.0, content=-0.1)),
                                         (8, g.synthetic_ws3(seed=7, style=-0.4, content=-0.6)),
                                         (16, syn))},
                          folder / "WS3_A2_layers_SYNTHETIC.png", kind="SYNTHETIC", model="synthetic")]
    out.append(ws3_summary({"A2 (F2)": out[5], "A5 cosine": out[8], "A7 style vs content": out[12],
                            "A8 spectrum": out[13], "X5 cosine": out[15]}, folder / "WS3_summary_SYNTHETIC.png", "SYNTHETIC"))
    return out


# ------------------------------------------------------------------------------------------------ H3 and Holm
MODEL_NAME = {"qwen25-7b": "Qwen2.5-7B", "qwen25-1p5b": "Qwen2.5-1.5B", "mistral-7b": "Mistral-7B", "gemma2-9b": "Gemma-2-9B"}


def h3_probe_shift(h3, path, kind="RESULT", view="prenorm"):
    """H3 (c): the vanilla-trained harmfulness probe's mean P(harmful) on benign test prompts, plain vs written as a
    jailbreak, per model; grey band = the +/- m margin around the plain score; orange tick = plain harmful prompts.
    h3: results/WS3_H3.json as a dict."""
    ps.apply_style()
    tags = list(h3["models"])
    fig, ax = plt.subplots(figsize=(7.5, 0.75 * len(tags) + 1.9))
    for i, t in enumerate(tags[::-1]):
        r = h3["models"][t]["views"][view]
        c, ms = r["c"], r["c"]["mean_score"]
        vb, ab, vh = ms["vanilla_benign"], ms["adversarial_benign"], ms["vanilla_harmful"]
        ax.add_patch(plt.Rectangle((vb - c["margin"], i - 0.28), 2 * c["margin"], 0.56, color=ps.GRID, zorder=1))
        ax.plot([vb, ab], [i, i], color=ps.BLUE, lw=2, zorder=2)
        ax.scatter([vb], [i], s=55, color=ps.BLUE, zorder=3, label="benign, plain" if i == 0 else None)
        ax.scatter([ab], [i], s=55, facecolor="white", edgecolor=ps.BLUE, lw=2, hatch="///", zorder=3,
                   label="benign, written as a jailbreak" if i == 0 else None)
        ax.plot([vh, vh], [i - 0.25, i + 0.25], color=ps.ORANGE, lw=2.5, zorder=3, label="harmful, plain" if i == 0 else None)
        verdict = " ".join(f"({k}) {'yes' if r[k]['holds'] else 'no'}" for k in ("a", "b", "c"))
        ax.text(1.02, i, verdict, va="center", fontsize=7, color=ps.INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(tags)), [MODEL_NAME.get(t, t) for t in tags[::-1]], fontsize=8)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.6, len(tags) - 0.4), xlabel="harmfulness probe, mean P(harmful) on test prompts")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=7, frameon=False)
    return _finish(fig, kind, "four models", "last, " + view.replace("norm", "-norm"), "prompt[-1]",
                   "H3: the jailbreak form alone raises the probe's harm score",
                   "grey band = H3 (c) margin, +/-10% of the plain harmful-benign gap; right: H3 parts (a) (b) (c) met", path)


def holm_pvalues(res, path, kind="RESULT"):
    """Holm over H1-H7: each WS3 test's bootstrap p-value per model (log scale) against 0.05/7 (survives whatever H4-H7
    give) and 0.05 (never survives). res: results/WS3_Holm.json as a dict."""
    ps.apply_style()
    tags = list(res["models"])
    tests, cols = ("H1", "H2", "H3"), (ps.BLUE, ps.ORANGE, ps.AQUA)
    fig, ax = plt.subplots(figsize=(7.5, 0.75 * len(tags) + 1.9))
    m, alpha = len(res["family"]), res["alpha"]
    ax.axvspan(1e-4, alpha / m, color=ps.GRID, alpha=0.6, zorder=0)
    for x, lab in ((alpha / m, f"0.05/{m}"), (alpha, "0.05")):
        ax.axvline(x, color=ps.INK2, lw=1, ls="--", zorder=1)
        ax.text(x, len(tags) - 0.45, lab, ha="center", va="bottom", fontsize=7, color=ps.INK2)
    for i, t in enumerate(tags[::-1]):
        pv = res["models"][t]["pvalues"]
        for k, (h, col) in enumerate(zip(tests, cols)):
            y = i + (1 - k) * 0.2
            if pv.get(h) is None:
                ax.text(1.5, y, f"{h} not run ", ha="right", va="center", fontsize=6.5, color=ps.MUTED)
                continue
            ax.scatter([max(pv[h], 1.2e-4)], [y], s=40, color=col, zorder=3)
    ax.set_xscale("log")
    ax.set(xlim=(1e-4, 1.5), ylim=(-0.6, len(tags) - 0.1), xlabel="bootstrap p-value (pre-registered vector)")
    ax.set_yticks(range(len(tags)), [MODEL_NAME.get(t, t) for t in tags[::-1]], fontsize=8)
    for h, col in zip(tests, cols):
        ax.scatter([], [], s=40, color=col, label=h)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=7, frameon=False)
    return _finish(fig, kind, "four models", "last, pre-norm", "prompt[-1]",
                   "Holm over H1-H7: which WS3 tests survive",
                   "shaded = survives whatever H4-H7 give; right of 0.05 = never survives; H4-H7 from the probe work", path)
