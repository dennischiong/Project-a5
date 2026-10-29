"""Task 5 pictures (numbers only, never text):

  draw_extraction_flow   figures/diagrams/T5_extraction_flow.png   how the pilot extraction runs, incl. the tactics decision
  draw_pilot_glance      figures/T5_pilot_glance.png               (a) pairs per 2x2 cell, (b) token counts, (c) tactic counts
  draw_noise_floor       figures/T5_noise_floor.png                batch 1 vs 16 differences per layer view, log scale

Before Task 5 has run, the notebook calls the same functions on synthetic_pairs() / synthetic_noise_floor(), which
are clearly labelled SYNTHETIC (title, watermark and a _SYNTHETIC file name).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import diagrams as dg
from analysis import plotstyle as ps

MODEL = "Qwen2.5-1.5B-Instruct"
VIEW_LABEL = {"postnorm_last": "post-norm\nlast layer", "layer_minus2": "hidden_states[-2]\n(no norm)",
              "prenorm_last": "pre-norm\nlast layer"}


# ------------------------------------------------------------------------------------------------ diagram
def draw_extraction_flow(path):
    """DIAGRAM of the Task 5 extraction: dataset -> column check -> tactics decision (diamond) -> sample -> pairs
    table -> forward pass -> hidden states -> harvest/out/. The PILOT ONLY override path is orange and dashed."""
    fig, ax = dg.canvas(12, 5.6, ps.tag("DIAGRAM") + " | Task 5b pilot extraction flow | "
                        + f"{MODEL}, every layer, prompt[-1] + last-instruction token")
    O = ps.ORANGE
    wj = dg.node(ax, 8, 80, "WildJailbreak\ntrain.tsv", w=13, h=12)
    cc = dg.node(ax, 25, 80, "5a column check\n(names + counts only)", w=15, h=12)
    d1 = dg.node(ax, 44, 80, "tactic labels?\n(map file, then\n'tactics' column)", w=19, h=20, shape="diamond")
    sm = dg.node(ax, 66, 80, "sample 50 harmful +\n50 benign pairs", w=17, h=12)
    pt = dg.node(ax, 88, 80, "pairs table\nharvest/out/pairs.parquet\n(text only in data/)", w=20, h=14)
    d2 = dg.node(ax, 44, 48, "ALLOW_MISSING\n_TACTICS\nticked?", w=19, h=20, shape="diamond", ec=O)
    ov = dg.node(ax, 66, 48, "PILOT ONLY\ntactic = 'unknown'\nplain seeded random sample", w=19, h=14, ec=O, ls="--")
    st = dg.node(ax, 20, 48, "STOP\n(same message as before:\nask Dennis / mentor)", w=19, h=13, fc="#ece9e1", bold=False)
    fw = dg.node(ax, 88, 16, f"{MODEL}\nbf16, batch 1\none forward pass", w=20, h=14)
    hs = dg.node(ax, 62, 16, "hidden states at\nprompt[-1] and the\nlast-instruction token", w=20, h=14)
    fo = dg.node(ax, 34, 16, "files in harvest/out/\n8 files, numbers only\n+ noise floor", w=20, h=14)

    dg.arrow(ax, wj, cc)
    dg.arrow(ax, cc, d1)
    dg.arrow(ax, d1, sm, "yes")
    dg.arrow(ax, sm, pt)
    dg.arrow(ax, d1, d2, "no", sides=("bottom", "top"))
    dg.arrow(ax, d2, ov, "yes", color=O, ls="--")
    dg.arrow(ax, ov, sm, color=O, ls="--", sides=("top", "bottom"))
    dg.arrow(ax, d2, st, "no")
    dg.arrow(ax, pt, fw, sides=("bottom", "top"))
    dg.arrow(ax, fw, hs)
    dg.arrow(ax, hs, fo)
    dg.legend(ax, [("line", ps.INK2, "normal path", "-"), ("line", O, "PILOT ONLY override path (--allow-missing-tactics)", "--"),
                   ("patch", "edge", "diamond = decision", "-")], loc="lower left")
    return dg.savefig(fig, path)


# ------------------------------------------------------------------------------------------------ data at a glance
def synthetic_pairs(n_pairs=100, seed=0):
    """SYNTHETIC pairs table (ids, counts, 'unknown' tactics; no text) to preview draw_pilot_glance."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        intent = "harmful" if i < n_pairs // 2 else "benign"
        for role, mu in (("vanilla", 3.6), ("adversarial", 5.4)):
            rows.append(dict(pair_id=f"syn{i:04d}", role=role, intent=intent, tactic="unknown",
                             token_count=int(np.exp(rng.normal(mu, 0.45)))))
    return pd.DataFrame(rows)


def draw_pilot_glance(pairs, path, kind="PILOT", model=MODEL):
    """Three panels from the pairs table: (a) prompts per 2x2 cell, (b) token counts vanilla vs adversarial,
    (c) adversarial prompts per tactic (one 'unknown' bar while tactic labels are missing)."""
    assert kind in ("PILOT", "SYNTHETIC")
    ps.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw={"width_ratios": [1, 1.25, 1.1]})

    ax = axes[0]
    cells = [(i, f) for i in ("harmful", "benign") for f in ("vanilla", "adversarial")]
    counts = [int(((pairs.intent == i) & (pairs.role == f)).sum()) for i, f in cells]
    bars = ax.bar(range(4), counts, color=[ps.INTENT_COLOR[i] for i, _ in cells],
                  hatch=None, edgecolor=ps.SURFACE, linewidth=1)
    for b, (_, f) in zip(bars, cells):
        b.set_hatch(ps.FORM_HATCH[f])
        b.set_alpha(1.0 if f == "vanilla" else 0.6)
    ax.bar_label(bars, fontsize=7, color=ps.INK2, padding=1)
    ax.set_xticks(range(4), [f"{i}\n{f}" for i, f in cells], fontsize=7)
    ax.set(title="(a) prompts per cell of the 2x2 design", ylabel="prompts (count)")

    ax = axes[1]
    tc = pairs["token_count"].astype(float)
    bins = np.logspace(np.log10(max(1, tc.min())), np.log10(tc.max() + 1), 25)
    for form, color, ls in (("vanilla", ps.INK2, "-"), ("adversarial", ps.AQUA, "--")):
        ax.hist(pairs.loc[pairs.role == form, "token_count"], bins=bins, histtype="step", color=color, ls=ls, lw=2,
                label=f"{form} (median {int(pairs.loc[pairs.role == form, 'token_count'].median())})")
    ax.set_xscale("log")
    ax.set(title="(b) prompt length after the chat template", xlabel="tokens per prompt (log scale)", ylabel="prompts (count)")
    ax.legend()

    ax = axes[2]
    adv = pairs[pairs.role == "adversarial"]
    tcount = adv.groupby(["tactic", "intent"]).size().unstack(fill_value=0).reindex(columns=["harmful", "benign"], fill_value=0)
    tcount = tcount.loc[tcount.sum(axis=1).sort_values(ascending=False).index]
    if len(tcount) > 12:  # fold the tail into "Other"
        tcount = pd.concat([tcount.iloc[:11], tcount.iloc[11:].sum().to_frame("Other").T])
    y = np.arange(len(tcount))[::-1]
    left = np.zeros(len(tcount))
    for intent in ("harmful", "benign"):
        ax.barh(y, tcount[intent], left=left, height=0.6, color=ps.INTENT_COLOR[intent], label=intent,
                edgecolor=ps.SURFACE, linewidth=1)
        left += tcount[intent].to_numpy()
    ax.set_yticks(y, [t if len(t) <= 26 else t[:25] + "..." for t in tcount.index], fontsize=7)
    only_unknown = list(tcount.index) == ["unknown"]
    ax.set(title="(c) adversarial prompts per tactic" + ("\nNO tactic labels: all 'unknown'" if only_unknown else ""),
           xlabel="adversarial prompts (count)")
    ax.set_ylim(-0.9, len(tcount) - 0.1)  # a free strip on top for the legend
    ax.legend(loc="upper right", ncol=2)
    if kind == "SYNTHETIC":
        for a in axes:
            ps.watermark(a)
    ps.caption(fig, ps.tag(kind, model, "none (counts only)", "n/a") + " | Task 5b pilot data at a glance (numbers only, no text)",
               "counts from harvest/out/pairs.parquet" if kind == "PILOT" else "SYNTHETIC preview: made-up counts, not data")
    return ps.savefig(fig, path)


# ------------------------------------------------------------------------------------------------ archive (Task 7)
def draw_archive_sizes(manifest, path, kind="PILOT", model=MODEL):
    """Raw size of the pilot outputs vs the pixz archive on Drive (from the archive's manifest)."""
    assert kind in ("PILOT", "SYNTHETIC")
    ps.apply_style()
    raw, arc = manifest["raw_bytes"] / 1e6, manifest["archive_bytes"] / 1e6
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    bars = ax.barh([1, 0], [raw, arc], color=[ps.MUTED, ps.BLUE], height=0.55)
    ax.bar_label(bars, labels=[f"{raw:.1f} MB ({manifest['n_files']} files)", f"{arc:.1f} MB (one .tpxz file)"],
                 fontsize=7.5, padding=3)
    ax.set_yticks([1, 0], ["harvest/out/ before", "archive on Drive"])
    ax.set_xlim(0, max(raw, arc) * 1.45)
    ax.set_xlabel("size (MB)")
    saved = 1 - arc / raw if raw else 0
    ps.caption(fig, ps.tag(kind, model, "all stored", "prompt[-1] + last instruction") +
               f" | Task 7 archive size: pixz -{manifest.get('pixz_level', '?')} saved {saved:.0%}",
               f"from {manifest['name']}.json (sha256 {manifest['sha256'][:12]}); work plan expects about 25% on bf16 data")
    if kind == "SYNTHETIC":
        ps.watermark(ax)
    return ps.savefig(fig, path)


# ------------------------------------------------------------------------------------------------ noise floor
def synthetic_noise_floor(seed=0):
    """SYNTHETIC noise_floor.json-shaped dict (tiny made-up values) to preview draw_noise_floor."""
    rng = np.random.default_rng(seed)
    per_view = {}
    for k, view in enumerate(VIEW_LABEL):
        d = {}
        for name, base in (("cosine_distance", 2e-5), ("relative_l2", 6e-3)):
            m = base * (1 + 0.6 * k) * float(np.exp(rng.normal(0, 0.2)))
            d[name] = dict(mean=m, median=0.8 * m, max=4 * m, ci95=[0.75 * m, 1.3 * m])
        per_view[view] = d
    return dict(label="SYNTHETIC", n=50, per_view=per_view)


def noise_views(nfloor):
    """{view: {cosine_distance, relative_l2}} from a noise_floor.json dict; old files hold only the post-norm view."""
    if nfloor.get("per_view"):
        return {v: nfloor["per_view"][v] for v in VIEW_LABEL if v in nfloor["per_view"]}
    return {"postnorm_last": {k: nfloor[k] for k in ("cosine_distance", "relative_l2")}}


def draw_noise_floor(nfloor, path, kind="PILOT", model=MODEL, gpu="n/a"):
    """Bar per layer view = mean batch-1 vs batch-16 difference over the measured prompts, whiskers = 95% bootstrap
    CI (when the file has one), dot = the largest single prompt. Log scale, so tiny values stay visible."""
    assert kind in ("PILOT", "SYNTHETIC")
    views = noise_views(nfloor)
    ps.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, (key, label) in zip(axes, (("cosine_distance", "cosine distance (1 - cos), no unit"),
                                       ("relative_l2", "relative L2 = ||h_bs16 - h_bs1|| / ||h_bs1||, no unit"))):
        names = list(views)
        mean = np.array([views[v][key]["mean"] for v in names])
        ci = [views[v][key].get("ci95") for v in names]
        x = np.arange(len(names))
        mx = np.array([views[v][key]["max"] for v in names], dtype=float)
        has_ci = all(isinstance(c, (list, tuple)) and len(c) == 2 for c in ci)
        lo = np.array([c[0] for c in ci], dtype=float) if has_ci else mean
        hi = np.array([c[1] for c in ci], dtype=float) if has_ci else mean
        positive = np.concatenate([mean, lo, hi, mx])
        positive = positive[positive > 0]
        # a log axis cannot show 0: values that are exactly 0 (bit-identical runs) are written as text instead
        floor_ = positive.min() / 3 if positive.size else 1e-12
        shown = np.where(mean > 0, mean, floor_)
        bars = ax.bar(x, shown - floor_, bottom=floor_, width=0.55, color=ps.BLUE, label="mean over prompts")
        if has_ci:
            ax.errorbar(x, shown, yerr=[np.clip(shown - np.maximum(lo, floor_), 0, None), np.clip(hi - shown, 0, None)],
                        fmt="none", ecolor=ps.INK2, capsize=4, lw=1, label="95% CI")
        ax.scatter(x, np.where(mx > 0, mx, floor_), marker="D", s=22, color=ps.ORANGE, zorder=4, label="largest single prompt")
        for xi, m in zip(x, mean):
            ax.text(xi, floor_ * 1.15, f"mean {m:.2g}" if m > 0 else "exactly 0\n(identical)", ha="center", va="bottom",
                    fontsize=6.5, color=ps.SURFACE if m > 0 else ps.INK2, zorder=5)
        ax.set_yscale("log")
        top = positive.max() if positive.size else 1e-11
        ax.set_ylim(floor_, top * 12)  # head room for the legend above the markers
        ax.set_xticks(x, [VIEW_LABEL[v] for v in names], fontsize=7)
        ax.set(title=label.split(",")[0], ylabel=label.split(" = ")[0].split(",")[0] + ", log scale (no unit)")
        ax.legend(loc="upper left", ncol=3, fontsize=6.5)
        if kind == "SYNTHETIC":
            ps.watermark(ax)
    n = nfloor.get("n", "?")
    ps.caption(fig, ps.tag(kind, model, "3 views (see x-axis)", "prompt[-1]") + " | numerical noise floor: batch size 1 vs 16",
               f"{n} vanilla prompts, same prompt run alone and inside a batch of 16 (left padding) | GPU {gpu} | "
               + ("pilot-only rough floor" if kind == "PILOT" else "SYNTHETIC preview: made-up values, not data"))
    return ps.savefig(fig, path)
