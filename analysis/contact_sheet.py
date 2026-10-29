"""Visual audit (Prompt 7): draw every diagram and figure function on synthetic data into one folder, then tile
them all on one contact sheet (figures/_contact_sheet.png) so they can be checked at a glance.

    python analysis/contact_sheet.py            # renders into /tmp-like scratch, writes figures/_contact_sheet.png
"""
import sys
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import plotstyle as ps  # noqa: E402


def render_everything(folder):
    """Every picture function in the repo, on SYNTHETIC data or as a DIAGRAM. Returns the list of PNG paths."""
    import pandas as pd

    from analysis import figures as F
    from analysis import harvest_figures as hf
    from analysis import lib
    from analysis import notebook_maps as nm
    from analysis import pilot_figures as pf
    from analysis import prereg_tools as pt
    from analysis import team_inputs as ti

    folder = Path(folder)
    out = [pf.draw_extraction_flow(folder / "T5_extraction_flow.png"),
           pf.draw_pilot_glance(pf.synthetic_pairs(), folder / "T5_pilot_glance_SYNTHETIC.png", kind="SYNTHETIC"),
           pf.draw_noise_floor(pf.synthetic_noise_floor(), folder / "T5_noise_floor_SYNTHETIC.png", kind="SYNTHETIC"),
           pf.draw_archive_sizes({"name": "example.tpxz", "n_files": 8, "raw_bytes": 40e6, "archive_bytes": 30e6,
                                  "sha256": "0" * 64, "pixz_level": 1}, folder / "T7_archive_sizes_SYNTHETIC.png", kind="SYNTHETIC")]
    for nb in nm.TASKS:
        out += [nm.draw_run_order(nb, folder / f"map_{nb}_run_order.png"),
                nm.draw_data_flow(nb, folder / f"map_{nb}_data_flow.png")]
    out.append(nm.draw_design_2x2(folder / "design_2x2.png"))
    out.append(nm.draw_status([("1 Setup", "DONE", "repo files on Drive"), ("5 Pilot", "WAITING", "needs the GPU run"),
                               ("6 Analyses", "NOT RUN", "needs Task 5"), ("8 Shard", "PROBLEM", "path does not exist")],
                              folder / "status_SYNTHETIC.png"))
    vals = dict(ti.DEFAULTS, SHARED_DRIVE_ROOT=str(folder), COLAB_L4_HOURS_LEFT=24)
    rows = ti.check(vals)
    out += [ti.draw_dependency(rows, folder / "T0_dependency.png"), ti.draw_readiness(rows, folder / "T0_readiness.png"),
            ti.draw_compute(24, folder / "T0_compute.png")]
    sys.path.insert(0, str(REPO / "tests"))
    from fake_shards import make_fake_harvest

    make_fake_harvest(folder / "fake_harvest", n_shards=3, prompts_per_shard=8, truncate_rows=(4,))
    rep = lib.check_harvest(folder / "fake_harvest", verbose=False)
    out += [hf.draw_data_shape(folder / "T8_data_shape.png"), hf.draw_schema_check(rep, folder / "T8_schema_SYNTHETIC.png"),
            hf.draw_health(rep, folder / "T8_health_SYNTHETIC.png")]
    items = pt.find_proposed((REPO / "prereg.md").read_text(encoding="utf-8"))
    out += [pt.draw_hypothesis_map(folder / "T4_hypothesis_map.png"), pt.draw_mentor_chart(items, folder / "T4_mentor.png"),
            pt.draw_decision_rules(folder / "T4_decision_rules.png"),
            pt.draw_lock_badge({"state": "NOT LOCKED"}, folder / "T4_lock_badge.png", ["example: LOCK_PREREG not ticked"])]
    out += F.render_all_synthetic(folder / "ws3")
    return out


def contact_sheet(pngs, path, cols=4, what="synthetic previews"):
    """Tile PNGs (file name under each) on one page."""
    ps.apply_style()
    pngs = [Path(p) for p in pngs]
    rows = int(np.ceil(len(pngs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.3))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, p in zip(np.atleast_1d(axes).ravel(), pngs):
        ax.imshow(mpimg.imread(p))
        ax.set_title(p.name, fontsize=7, color=ps.INK2)
    ps.caption(fig, ps.tag("DIAGRAM") + f" | contact sheet: {len(pngs)} figures and diagrams ({what})",
               "made by analysis/contact_sheet.py; open the single PNGs to check details")
    return ps.savefig(fig, path)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        pngs = render_everything(tmp)
        p = contact_sheet(pngs, REPO / "figures" / "_contact_sheet.png")
        print(f"rendered {len(pngs)} figures; contact sheet: {p}")
