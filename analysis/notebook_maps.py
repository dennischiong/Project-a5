"""Notebook maps: pictures that show how each notebook is put together (no data in them).

  draw_run_order(nb, path)    one box per task, arrows = "must run before", fill = CPU / GPU, file badge under each box
  draw_data_flow(nb, path)    what turns into what, with the file name under each box
  draw_design_2x2(path)       the 2x2 design (harmful/benign x vanilla/adversarial) and what dp measures
  draw_status(rows, path)     one horizontal bar per task, coloured DONE / WAITING / NOT RUN / PROBLEM

``nb`` is one of the keys of TASKS: A5_pipeline_v2, pilot_end_to_end_v2, A5_colab, WS3_analyses.
"""
import matplotlib.pyplot as plt
import numpy as np

from analysis import diagrams as dg
from analysis import plotstyle as ps

RUNTIME_FILL = {"CPU": "#dce9f8", "GPU": "#fbe0d4", "none": "#eeeeea"}

# (id, box label, runtime, files it produces, ids it needs first)
TASKS = {
    "A5_pipeline_v2": [
        ("T0", "Task 0\nTeam inputs", "CPU", "team_inputs.json", []),
        ("T1", "Task 1\nSetup", "CPU", "repo files on Drive", []),
        ("T2", "Task 2\nUnit tests", "CPU", "tests/pytest_last_run.xml", ["T1"]),
        ("T3", "Task 3\nToken positions", "CPU", "figures/offsets.csv", ["T1"]),
        ("T4", "Task 4\nPre-registration", "CPU", "prereg_questions.md\nprereg_lock.json", ["T0", "T1"]),
        ("T5a", "Task 5a\nColumn check", "CPU", "printed columns", ["T1"]),
        ("T8", "Task 8\nKayson's sample shard", "CPU", "figures/T8_*.png", ["T0", "T1"]),
        ("T5b", "Task 5b\nPilot extraction", "GPU", "harvest/out/ (8 files)", ["T0", "T5a"]),
        ("T6", "Task 6\nPilot analyses", "CPU", "pilot_results.json\nfigures/F1, F2, A3", ["T4", "T5b"]),
        ("T7", "Task 7\nArchive test\n(other notebook)", "CPU", "test folder/*.tpxz", ["T0", "T5b"]),
        ("T9", "Task 9\nStatus at a glance", "CPU", "figures/status.png", ["T2", "T6", "T7", "T8"]),
    ],
    "pilot_end_to_end_v2": [
        ("X", "A5_pipeline_v2\nTasks 0, 1, 5b", "GPU", "repo + harvest/out/", []),
        ("S0", "0.1-0.3\nMount, pre-check,\npixz", "CPU", "team_inputs.json read", ["X"]),
        ("S1", "1.1\nArchive to Drive", "CPU", "<name>.tpxz\n<name>.tpxz.json", ["S0"]),
        ("S2", "2.1\nDelete, restore,\ncompare hashes", "CPU", "/content/pilot_tmp/out", ["S1"]),
        ("S3", "3.1\nLoad + overall\nPASS / FAIL", "CPU", "printed PASS/FAIL", ["S2"]),
    ],
    "A5_colab": [
        ("1", "1 GPU check", "GPU", "", []),
        ("2", "2 Install", "GPU", "", ["1"]),
        ("3", "3 HF login", "GPU", "", ["2"]),
        ("4", "4a/4b Gated\naccess, files", "GPU", "", ["3"]),
        ("5", "5 Config,\nseeds", "GPU", "", ["4"]),
        ("6", "6a/6b Build\n10 pairs", "GPU", "pairs.csv (private)", ["5"]),
        ("7", "7 Load model\n(fp16)", "GPU", "", ["6"]),
        ("8", "8 Refusal /\ncomply tokens", "GPU", "", ["7"]),
        ("9", "9 extract()", "GPU", "", ["8"]),
        ("10", "10/10b Run 20\nprompts, reload", "GPU", "week0_acts.pt\nweek0_meta.csv", ["9"]),
        ("11", "11a-d Sanity\nchecks", "GPU", "", ["10"]),
        ("11c", "11c Figure", "GPU", "week0_pair_\nsimilarity.png", ["10"]),
        ("12", "12 Copy to\nDrive", "GPU", "Drive/a5_week0/", ["10", "11c"]),
    ],
    "WS3_analyses": [
        ("W0", "Setup +\nTeam inputs", "CPU", "team_inputs.json read", []),
        ("A2", "A2\nr_hat and dp", "CPU", "figures/WS3_A2_*.png\nresults/A2.json", ["W0"]),
        ("A5", "A5\nTactic geometry", "CPU", "figures/WS3_A5_*.png\nresults/A5.json", ["W0"]),
        ("A7", "A7\nStyle vs content", "CPU", "figures/WS3_A7_*.png\nresults/A7.json", ["W0"]),
        ("A8", "A8\nLow-rank structure", "CPU", "figures/WS3_A8_*.png\nresults/A8.json", ["W0"]),
        ("X5", "X5\nDirection stability", "CPU", "figures/WS3_X5_*.png\nresults/X5.json", ["W0"]),
        ("SUM", "WS3 summary panel", "CPU", "figures/WS3_summary.png", ["A2", "A5", "A7", "A8", "X5"]),
    ],
}

# (box label, file name written under the box)
FLOWS = {
    "A5_pipeline_v2": [
        ("raw dataset\nWildJailbreak", "train/train.tsv (Hugging Face)"),
        ("pairs table", "harvest/out/pairs.parquet"),
        ("hidden states", "harvest/out/pilot_acts.safetensors"),
        ("r_hat direction\n(vanilla TRAIN)", "made in analysis/pilot_analysis.py"),
        ("dp per pair", "analysis/pilot_results.json"),
        ("figures +\nresults JSON", "figures/F2_pilot_*.png, ..."),
    ],
    "pilot_end_to_end_v2": [
        ("pilot outputs", "harvest/out/"),
        ("temp copy", "/content/pilot_tmp/out"),
        ("archive on Drive", "<name>.tpxz + <name>.tpxz.json"),
        ("restored copy", "/content/pilot_tmp/out"),
        ("loaded again", "lib.load_activations / load_pairs"),
        ("PASS / FAIL", "printed, 10 steps"),
    ],
    "A5_colab": [
        ("raw dataset\nWildJailbreak", "train/train.tsv"),
        ("10 pairs", "pairs.csv (text, private)"),
        ("Qwen2.5-1.5B\nforward pass", "fp16, batch 1"),
        ("hidden states", "week0_acts.pt, week0_meta.csv"),
        ("figure", "week0_pair_similarity.png"),
        ("Drive copy", "Drive (week 0 folder)"),
    ],
    "WS3_analyses": [
        ("harvest archives", "<model>__<dataset>__...tpxz"),
        ("restored shards", "/content/restore/..."),
        ("activations +\nindex", "lib.load_harvest"),
        ("directions,\ndisplacements", "analysis/geometry.py"),
        ("numbers", "results/WS3_*.json"),
        ("figures", "figures/WS3_*.png"),
    ],
}


def _levels(tasks):
    """Column of each task = length of its longest chain of prerequisites."""
    need = {t[0]: t[4] for t in tasks}
    lvl = {}

    def depth(i):
        if i not in lvl:
            lvl[i] = 0 if not need[i] else 1 + max(depth(j) for j in need[i])
        return lvl[i]

    for t in tasks:
        depth(t[0])
    return lvl


def _positions(tasks, max_per_col=5):
    lvl = _levels(tasks)
    n_col = max(lvl.values()) + 1
    if n_col > 7:  # long chains: snake through a grid instead
        ncols = 5
        pos = {}
        for k, t in enumerate(tasks):
            r, c = divmod(k, ncols)
            c = c if r % 2 == 0 else ncols - 1 - c
            pos[t[0]] = (10 + c * 20, 84 - r * 27)
        return pos, 15, None
    cols = {}
    for t in tasks:
        cols.setdefault(lvl[t[0]], []).append(t[0])
    pos = {}
    for c, ids in cols.items():
        xs = 8 + c * (84 / max(n_col - 1, 1))
        for r, i in enumerate(ids):
            y = 50 + (len(ids) - 1) / 2 * 18 - r * 18
            pos[i] = (xs, y)
    return pos, min(15, 84 / max(n_col, 1) - 2), ("right", "left")  # columns: arrows always run left to right


def draw_run_order(nb, path):
    """RUN-ORDER DIAGRAM for one notebook."""
    tasks = TASKS[nb]
    pos, w, sides = _positions(tasks)
    fig, ax = dg.canvas(13, 6.8, ps.tag("DIAGRAM") + f" | {nb}.ipynb run order: arrows = must run before")
    nodes = {}
    for i, label, rt, files, _ in tasks:
        x, y = pos[i]
        nodes[i] = dg.node(ax, x, y, label, w=w, h=10, fc=RUNTIME_FILL[rt], fs=6.8)
        if files:
            ax.text(x, y - 6.2, files, ha="center", va="top", fontsize=5.6, color=ps.INK2, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ps.AXIS, lw=0.6))
    for i, *_, needs in tasks:
        for j in needs:
            dg.arrow(ax, nodes[j], nodes[i], lw=1.0, sides=sides)
    dg.legend(ax, [("patch", RUNTIME_FILL["CPU"], "CPU runtime", "-"), ("patch", RUNTIME_FILL["GPU"], "GPU runtime", "-"),
                   ("patch", "white", "badge = files the task produces", "-")], loc="lower right", ncol=3)
    return dg.savefig(fig, path)


def draw_data_flow(nb, path):
    """DATA-FLOW DIAGRAM: boxes in reading order, the file name under each box."""
    steps = FLOWS[nb]
    fig, ax = dg.canvas(13, 2.6, ps.tag("DIAGRAM") + f" | {nb}.ipynb data flow: what turns into what (file name under each box)")
    n = len(steps)
    xs = np.linspace(8, 92, n)
    prev = None
    for x, (label, fname) in zip(xs, steps):
        cur = dg.node(ax, x, 62, label, w=min(14, 84 / n), h=34, fs=7)
        ax.text(x, 36, fname, ha="center", va="top", fontsize=6, color=ps.INK2, family="monospace")
        if prev:
            dg.arrow(ax, prev, cur)
        prev = cur
    return dg.savefig(fig, path)


def draw_design_2x2(path):
    """The 2x2 design and what dp measures."""
    fig, ax = dg.canvas(10, 5.6, ps.tag("DIAGRAM") + " | the 2x2 design: intent (rows) x form (columns), and what dp measures")
    cells = {
        ("harmful", "vanilla"): "harmful request,\nplain wording\n(fits r_hat, TRAIN)",
        ("harmful", "adversarial"): "same request,\njailbreak rewrite\n(H1, H2 test here)",
        ("benign", "vanilla"): "harmless request,\nplain wording\n(fits r_hat, TRAIN)",
        ("benign", "adversarial"): "same request,\njailbreak style\n(control: style only)",
    }
    X = {"vanilla": 28, "adversarial": 66}
    Y = {"harmful": 66, "benign": 26}
    nodes = {}
    for (intent, form), text in cells.items():
        nodes[(intent, form)] = dg.node(ax, X[form], Y[intent], text, w=24, h=24, fc=ps.SURFACE,
                                        ec=ps.INTENT_COLOR[intent], lw=2, ls="-" if form == "vanilla" else "--", fs=7)
    for form, x in X.items():
        ax.text(x, 84, form.upper(), ha="center", fontsize=8.5, color=ps.INK, fontweight="bold")
    for intent, y in Y.items():
        ax.text(9, y, intent.upper(), ha="center", va="center", fontsize=8.5, color=ps.INTENT_COLOR[intent],
                fontweight="bold", rotation=90)
    for intent in ("harmful", "benign"):
        dg.arrow(ax, nodes[(intent, "vanilla")], nodes[(intent, "adversarial")], "delta = h_adv - h_van",
                 color=ps.INTENT_COLOR[intent], lw=1.8)
    dg.arrow(ax, nodes[("benign", "vanilla")], nodes[("harmful", "vanilla")], "r_hat = mean(harmful)\n- mean(benign)",
             color=ps.INK2, sides=("top", "bottom"), lw=1.8)
    ax.text(82, 46, "dp = delta . r_hat\n\nthe part of the rewrite's\nmove that points along\nthe harmfulness direction\n\n"
            "dp < 0: the rewrite\nlooks LESS harmful", ha="left", va="center", fontsize=7.2, color=ps.INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=ps.AXIS))
    return dg.savefig(fig, path)


def draw_status(rows, path, title="task status"):
    """rows: list of (task, state, reason) with state DONE / WAITING / NOT RUN / PROBLEM."""
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.3))
    y = np.arange(len(rows))[::-1]
    for yi, (task, state, reason) in zip(y, rows):
        state = state.upper()
        ax.barh(yi, 1, color=ps.STATUS_COLOR.get(state, ps.MUTED), height=0.7,
                hatch=None if state == "DONE" else ("//" if state == "WAITING" else ("xx" if state == "PROBLEM" else "")),
                edgecolor=ps.SURFACE)
        ax.text(0.02, yi, state, va="center", ha="left", fontsize=7.5, color=ps.INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
        ax.text(1.03, yi, reason, va="center", ha="left", fontsize=7, color=ps.INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 3.2)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("state (bar colour and word) and what is missing (text)")
    ps.caption(fig, ps.tag("DIAGRAM") + f" | {title}", "computed live from the files on Drive each time the cell runs")
    return ps.savefig(fig, path)
