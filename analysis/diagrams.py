"""Box-and-arrow diagrams drawn with matplotlib only (works in Colab with no extra installs).

Coordinates are in a 0-100 x 0-100 canvas, so a layout reads like a sketch on squared paper:
    fig, ax = canvas(10, 5, "DIAGRAM | ...")
    a = node(ax, 15, 50, "first step")
    b = node(ax, 50, 50, "a decision?", shape="diamond")
    arrow(ax, a, b, "label")
    savefig(fig, "figures/diagrams/example.png")
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Polygon

from analysis import plotstyle as ps

LIGHT = "#f1f0ea"  # soft box fill


def canvas(width, height, title, subtitle=None):
    """A blank 0-100 x 0-100 drawing area with the title on top."""
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    import textwrap

    lines = textwrap.wrap(title, width=max(40, int(width * 12)))
    fig.suptitle("\n".join(lines), fontsize=10, color=ps.INK, x=0.01, ha="left", y=0.995, va="top")
    fig._a5_title_lines = len(lines)
    if subtitle:
        fig.text(0.01, 0.005, subtitle, fontsize=6.5, color=ps.MUTED, ha="left", va="bottom")
    return fig, ax


def node(ax, x, y, text, w=16, h=10, shape="box", fc=LIGHT, ec=None, tc=None, fs=7.2, lw=1.2, ls="-", bold=False):
    """Draw a box (or 'diamond' for a decision, 'pill' for a person/badge) centred on (x, y). Returns its geometry."""
    ec = ec or ps.INK2
    tc = tc or ps.INK
    if shape == "diamond":
        ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], closed=True,
                             fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    else:
        style = "round,pad=0,rounding_size=2.5" if shape == "pill" else "round,pad=0,rounding_size=0.8"
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, ls=ls,
                                    zorder=2, mutation_aspect=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.25)
    return dict(x=x, y=y, w=w, h=h, shape=shape)


def _anchor(n, side):
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    return {"left": (x - w / 2, y), "right": (x + w / 2, y), "top": (x, y + h / 2), "bottom": (x, y - h / 2)}[side]


def _auto_sides(a, b):
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    # compare in box units so wide boxes prefer left/right links
    if abs(dx) / max(a["w"], 1) >= abs(dy) / max(a["h"], 1):
        return ("right", "left") if dx > 0 else ("left", "right")
    return ("top", "bottom") if dy > 0 else ("bottom", "top")


def arrow(ax, a, b, label=None, color=None, ls="-", lw=1.3, sides=None, rad=0.0, label_pos=0.5, fs=6.5, label_color=None):
    """Arrow from node a to node b (edge to edge). ``sides`` = ('right', 'left') etc. to choose the edges."""
    color = color or ps.INK2
    sa, sb = sides or _auto_sides(a, b)
    p0, p1 = _anchor(a, sa), _anchor(b, sb)
    ax.annotate("", xy=p1, xytext=p0, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=1, shrinkB=1,
                                mutation_scale=10, connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = p0[0] + (p1[0] - p0[0]) * label_pos, p0[1] + (p1[1] - p0[1]) * label_pos
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs, color=label_color or color, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc=ps.SURFACE, ec="none"))


def legend(ax, items, loc="lower left", ncol=1):
    """items: list of (kind, colour, text, linestyle) with kind 'patch' or 'line'."""
    handles = []
    for kind, color, text, ls in items:
        if kind == "line":
            handles.append(Line2D([0], [0], color=color, lw=1.5, ls=ls, label=text))
        else:
            # explicit lw: plotstyle sets patch.linewidth = 0, and matplotlib refuses a dashed edge of width 0
            handles.append(Patch(fc=color if color != "edge" else LIGHT, ec=ps.INK2, ls=ls or "-", lw=1.0, label=text))
    ax.legend(handles=handles, loc=loc, fontsize=6.5, ncol=ncol, frameon=False)


def savefig(fig, path):
    return ps.savefig(fig, path, rect=(0, 0.03, 1, 0.95))
