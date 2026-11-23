"""Shared figure style for every A5 figure (work plan: Hasif owns the shared figure style).

Colours come from the validated reference palette's first three categorical slots (blue, orange, aqua), which pass
the colour-blind (CVD) and normal-vision separation checks for every pair in light mode. Identity is never colour
alone: adversarial marks also carry a hatch texture, and every figure has a legend or direct labels.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

# Fixed meaning, never cycled: intent -> hue; form -> texture
INTENT_COLOR = {"harmful": ORANGE, "benign": BLUE}
FORM_HATCH = {"vanilla": None, "adversarial": "///"}
POSITION_COLOR = {"last prefill": BLUE, "last instruction": AQUA}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 9.5,
        "axes.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.frameon": False, "legend.fontsize": 7.5, "font.family": "sans-serif",
        "lines.linewidth": 2, "patch.linewidth": 0,
    })


def caption(fig, title, config_line):
    """Every figure states what it shows plus the serving configuration (layer, norm, model, backend, dtype, GPU).
    Long titles wrap onto more lines (never cut off); EXPLORATORY_NOTE (prereg changed after lock) is prefixed."""
    import textwrap

    if EXPLORATORY_NOTE and EXPLORATORY_NOTE not in title:
        title = f"{EXPLORATORY_NOTE} | {title}"
    lines = textwrap.wrap(title, width=max(40, int(fig.get_figwidth() * 12)))
    fig.suptitle("\n".join(lines), fontsize=10, color=INK, x=0.01, ha="left", y=0.995, va="top")
    fig._a5_title_lines = len(lines)
    fig.text(0.01, 0.005, config_line, fontsize=6.5, color=MUTED, ha="left", va="bottom", wrap=True)


def _top(fig, default):
    """Top of the plotting area: leaves room for a title of one or more lines."""
    n = getattr(fig, "_a5_title_lines", None)
    return default if n is None else min(default, 1 - (0.08 + 0.17 * n) / fig.get_figheight())


def save(fig, path):
    fig.tight_layout(rect=(0, 0.04, 1, _top(fig, 0.95)))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- labels every figure title carries (VISUALS RULE) -------------------------------------------------------------
# PILOT = real pilot data (plumbing, not results); SYNTHETIC = made-up numbers to preview a chart; RESULT = real
# project data; DIAGRAM = a picture of how things fit together, with no data in it.
KINDS = ("PILOT", "SYNTHETIC", "RESULT", "DIAGRAM")

# Status colours (readiness, task status). Colour is never the only cue: the state word is always written too.
STATUS_COLOR = {"READY": AQUA, "DONE": AQUA, "WAITING": "#e8a33d", "PROBLEM": "#d64545", "NOT RUN": "#c3c2b7"}

# Set by analysis scripts when prereg.md changed after it was locked (see analysis/prereg_tools.py)
EXPLORATORY_NOTE = None


def tag(kind, model="n/a", layer="n/a", position="n/a"):
    """The standard title prefix: 'KIND | model | layer | position'. Every figure title starts with this."""
    assert kind in KINDS, f"kind must be one of {KINDS}"
    parts = [kind]
    if EXPLORATORY_NOTE and kind != "DIAGRAM":
        parts.append(EXPLORATORY_NOTE)
    if kind != "DIAGRAM":
        parts += [f"model {model}", f"layer {layer}", f"position {position}"]
    return " | ".join(parts)


def watermark(ax, text="SYNTHETIC - not data"):
    """Large faint label across a panel so a synthetic preview can never pass for data."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=16, color=MUTED, alpha=0.25,
            rotation=20, zorder=10, fontweight="bold")


def savefig(fig, path, rect=(0, 0.04, 1, 0.93)):
    """Like save(), but creates the folder first and returns the path (used by the newer figure functions)."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(rect[0], rect[1], rect[2], _top(fig, rect[3])))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
