"""Task 8 pictures for Kayson's harvest shards (numbers only):

  draw_data_shape     DIAGRAM: one shard as a block (prompts x layers x hidden, per extraction point), the parquet index
                      beside it, and the prompt-id join to the pairs table (sizes for Llama-3.1-8B as the example)
  draw_schema_check   one row per expected field: OK (tick) or MISMATCH (cross), from lib.check_harvest
  draw_health         (a) prompts per shard (b) token counts, truncated highlighted (c) NaN/inf heat map by layer x
                      point (d) mean activation norm per layer
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from analysis import diagrams as dg
from analysis import plotstyle as ps

RED = ps.STATUS_COLOR["PROBLEM"]
GREEN = ps.STATUS_COLOR["READY"]


def _block(ax, x, y, w, h, d, fc):
    """Pseudo-3D block: front face w x h at (x, y), depth d drawn up-right."""
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True, fc=fc, ec=ps.INK2, lw=1.1, zorder=2))
    ax.add_patch(Polygon([(x, y + h), (x + d, y + h + d * 0.6), (x + w + d, y + h + d * 0.6), (x + w, y + h)],
                         closed=True, fc=fc, ec=ps.INK2, lw=1.1, alpha=0.75, zorder=2))
    ax.add_patch(Polygon([(x + w, y), (x + w + d, y + d * 0.6), (x + w + d, y + h + d * 0.6), (x + w, y + h)],
                         closed=True, fc=fc, ec=ps.INK2, lw=1.1, alpha=0.55, zorder=2))


def draw_data_shape(path, n_prompts=500, n_layers=32, hidden=4096):
    fig, ax = dg.canvas(13, 6.2, ps.tag("DIAGRAM") + " | one Undercurrent shard: prompts x extraction points x layers x hidden "
                        "(Llama-3.1-8B sizes as the example)")
    pts = [("last_prefill", n_layers, "\n(all layers)"), ("last_instruction", n_layers // 2, "\n(every 2nd)"),
           ("gen_4", 4, ""), ("gen_9", 4, ""), ("gen_19", 4, "")]
    x, y0, h, d, gap = 6, 30, 34, 4, 5
    for k, (name, L, note) in enumerate(pts):
        w = 2 + 12 * L / n_layers
        _block(ax, x, y0, w, h, d, "#dce9f8" if k == 0 else "#eef3fa")
        ax.text(x + w / 2, y0 - 2, f"{name}\n{L} layers{note}", ha="center", va="top", fontsize=6.2, color=ps.INK)
        x += w + gap
    right = x - gap + d
    # dimension labels drawn once, on the first block
    ax.text(4.5, y0 + h / 2, f"{n_prompts} prompts", ha="right", va="center", fontsize=6.5, color=ps.INK2, rotation=90)
    ax.text(6 + 8, y0 + h + d * 0.6 + 2, f"depth = {hidden:,} hidden units (every block)", ha="left", va="bottom",
            fontsize=6.5, color=ps.INK2)
    ax.text(6, 88, "shard_00000.safetensors  (bf16; one tensor per point and layer, shape (prompts, hidden))",
            fontsize=7.5, color=ps.INK, fontweight="bold")
    idx = dg.node(ax, 86, 66, "shard_00000.parquet (index)\none row per prompt:\nprompt_id, token_count, truncated,\n"
                  "backend, concurrency, dtype,\nquantization, undercurrent_version,\ntemplate_hash", w=24, h=26, fs=6.0, fc="white")
    prs = dg.node(ax, 86, 18, "pairs table (Dennis)\npair_id, role, intent,\ntactic, split, source_row_id", w=24, h=16,
                  fs=6.2, fc="#f6f5f0")
    dg.arrow(ax, idx, prs, "join on prompt_id\n(= pair_id|role)", sides=("bottom", "top"), lw=1.6)
    ax.annotate("", xy=(73.5, 60), xytext=(right + 1, 48), arrowprops=dict(arrowstyle="<->", color=ps.INK2, lw=1.2))
    ax.text((right + 73.5) / 2, 45, "tensor row i =\nindex row i", fontsize=6.3, color=ps.INK2, ha="center", va="top")
    ax.text(4, 8, f"sizes per shard (bf16, 2 bytes): last_prefill {n_prompts} x {n_layers} x {hidden:,} = "
            f"{n_prompts * n_layers * hidden * 2 / 1e6:.0f} MB; the assumed key names live in lib.EXPECTED_SCHEMA",
            fontsize=6.5, color=ps.MUTED)
    return dg.savefig(fig, path)


def draw_schema_check(rep, path, kind="SYNTHETIC", model="as named in the archive"):
    """One row per expected field: green tick = matches EXPECTED_SCHEMA, red cross = mismatch."""
    ps.apply_style()
    rows = rep["schema"]
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(rows) + 1.3))
    y = np.arange(len(rows))[::-1]
    for yi, (field, ok, hint) in zip(y, rows):
        ax.scatter([0], [yi], marker="o", s=110, color=GREEN if ok else RED, zorder=3)
        ax.text(0, yi, "✓" if ok else "✗", ha="center", va="center", fontsize=8, color="white",
                fontweight="bold", zorder=4)
        ax.text(0.08, yi, ("OK" if ok else "MISMATCH") + "   " + field, va="center", fontsize=7.2,
                color=ps.INK if ok else RED)
        if not ok:
            ax.text(0.62, yi, hint[:90], va="center", fontsize=6, color=ps.INK2)
    ax.set_xlim(-0.05, 1.6)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.axis("off")
    n_ok = sum(ok for _, ok, _ in rows)
    ps.caption(fig, ps.tag(kind, model, "all", "all points") + f" | schema check: {n_ok}/{len(rows)} fields match "
               "EXPECTED_SCHEMA", f"folder: {rep['shard_dir']}")
    return ps.savefig(fig, path)


def draw_health(rep, path, kind="SYNTHETIC", model="as named in the archive"):
    """Four-panel harvest health check."""
    ps.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    ax = axes[0, 0]
    names = list(rep["prompts_per_shard"])
    vals = [rep["prompts_per_shard"][n] for n in names]
    bars = ax.bar(range(len(vals)), vals, color=ps.BLUE)
    ax.bar_label(bars, fontsize=6.5, color=ps.INK2)
    ax.set_xticks(range(len(vals)), [n.replace(".safetensors", "") for n in names], fontsize=6, rotation=30, ha="right")
    ax.set(title="(a) prompts per shard (should be 500, last shard may be fewer)", ylabel="prompts (count)")

    ax = axes[0, 1]
    tc = np.array(rep["token_count"], dtype=float)
    tr = np.array(rep["truncated"], dtype=bool)
    if tc.size:
        bins = np.linspace(tc.min(), tc.max() + 1, 25)
        ax.hist(tc[~tr], bins=bins, color=ps.BLUE, label=f"not truncated (n={int((~tr).sum())})")
        ax.hist(tc[tr], bins=bins, color=RED, hatch="xx", edgecolor=ps.SURFACE, label=f"truncated (n={int(tr.sum())})")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no token_count column", transform=ax.transAxes, ha="center")
    ax.set(title="(b) prompt length; truncated prompts highlighted (should be none)", xlabel="tokens per prompt",
           ylabel="prompts (count)")

    ax = axes[1, 0]
    points = sorted(rep["points"])
    layers = sorted({L for p in points for L in rep["points"][p]})
    if points and layers:
        M = np.full((len(points), len(layers)), np.nan)
        for i, p in enumerate(points):
            for j, L in enumerate(layers):
                if (p, L) in rep["nan_inf"]:
                    M[i, j] = rep["nan_inf"][(p, L)]
        cmap = plt.get_cmap("Reds").with_extremes(bad=ps.GRID)  # grey = layers a point does not store
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=0, vmax=max(1, np.nanmax(M)))
        ax.set_yticks(range(len(points)), points, fontsize=7)
        step = max(1, len(layers) // 16)
        ax.set_xticks(range(0, len(layers), step), [layers[j] for j in range(0, len(layers), step)], fontsize=6.5)
        fig.colorbar(im, ax=ax, label="NaN/inf values (count)")
        total = int(np.nansum(M))
        ax.set_title(f"(c) NaN/inf count by point x layer (total {total}; should be all 0; grey = not stored)")
    else:
        ax.text(0.5, 0.5, "no readable tensors", transform=ax.transAxes, ha="center")
        ax.set_title("(c) NaN/inf count by point x layer")
    ax.set_xlabel("layer")
    ax.grid(False)

    ax = axes[1, 1]
    for k, p in enumerate(points):
        ls = sorted(rep["points"][p])
        ys = [rep["mean_norm"].get((p, L), np.nan) for L in ls]
        ax.plot(ls, ys, marker="o", ms=3, lw=1.5, ls=["-", "--", ":", "-.", (0, (1, 1))][k % 5], label=p)
    ax.set(title="(d) mean activation norm per layer (should rise smoothly with depth)", xlabel="layer",
           ylabel="mean L2 norm over prompts")
    if points:
        ax.legend(fontsize=6.5)
    if kind == "SYNTHETIC":
        for a in axes.flat:
            ps.watermark(a)
    ps.caption(fig, ps.tag(kind, model, "all stored", "all points") + " | Task 8 harvest health",
               f"folder: {rep['shard_dir']} | {rep['n_prompts']} prompts in {len(rep['shards'])} shard(s)")
    return ps.savefig(fig, path)
