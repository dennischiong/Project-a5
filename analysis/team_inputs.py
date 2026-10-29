"""Task 0: the values teammates give me, one place for all of them.

The Task 0 form cell writes them to team_inputs.json in the repo folder on Drive; every later cell and the archive
notebook read that file instead of hard-coded paths. check() turns each value into READY / WAITING / PROBLEM, with
the exact one-line message to send while it is WAITING.
"""
import json
import os
import time
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import diagrams as dg
from analysis import plotstyle as ps

REPO = Path(__file__).resolve().parents[1]
JSON_NAME = "team_inputs.json"
# private folder for archive tests before write access to the shared a5-activations folder
TEST_FOLDER = os.environ.get("A5_TEST_FOLDER", "/content/drive/MyDrive/a5-test")
L4_HOURS_NEEDED = 22  # work plan s.6.4: Hasif's GPU runs need about 20-22 L4-hours; the check uses the upper end
L4_HOURS_RANGE = (20, 22)

# name: (what it is, who gives it, when it is due, which tasks it unlocks)
INPUTS = {
    "SHARED_DRIVE_ROOT": ("the team's a5-activations folder, with write access", "Kayson / mentor",
                          "before the full harvests (Mon 12 Oct)",
                          ["Task 7 archive test (team folder)", "WS3 restores from Drive", "Gemma + Mistral harvests"]),
    "UNDERCURRENT_SAMPLE_DIR": ("one sample Undercurrent harvest shard folder", "Kayson", "Week 4 (12-18 Oct)",
                                ["Task 8 shard check", "lib.load_harvest on real data"]),
    "PAIRS_TABLE_PATH": ("the frozen pairs table v1 (parquet)", "Dennis", "Fri 9 Oct",
                         ["2x2 join on real data", "WS3 A2 / A5 / A7"]),
    "TACTICS_MAP_PATH": ("tactic labels (source_row_id + tactic)", "Dennis", "once decided",
                         ["Task 5b with real tactics", "per-tactic results (A5, A7)"]),
    "PREREG_CONFIRMED": ("mentor confirmed the [PROPOSED] thresholds", "Mentor", "Mon 12 Oct",
                         ["Task 4.3 pre-registration lock"]),
    "COLAB_L4_HOURS_LEFT": ("L4 hours left on my Colab balance", "Me", "before Mon 12 Oct",
                            ["Gemma + Mistral harvests (GPU)"]),
}
DEFAULTS = {"SHARED_DRIVE_ROOT": TEST_FOLDER, "UNDERCURRENT_SAMPLE_DIR": "", "PAIRS_TABLE_PATH": "",
            "TACTICS_MAP_PATH": "", "PREREG_CONFIRMED": False, "COLAB_L4_HOURS_LEFT": 0.0}
PEOPLE = ["Dennis", "Kayson", "Mentor", "Me"]

# exact messages to send while an input is WAITING
MESSAGES = {
    "SHARED_DRIVE_ROOT": ("Kayson", "Hi Kayson, could you give me write access to the team a5-activations Drive folder "
                                    "and send me its path? I need about 13 GB for the Gemma and Mistral archives."),
    "UNDERCURRENT_SAMPLE_DIR": ("Kayson", "Hi Kayson, could you share one sample Undercurrent shard folder (safetensors + "
                                          "parquet index) on Drive, so I can test my loader before the full harvest?"),
    "PAIRS_TABLE_PATH": ("Dennis", "Hi Dennis, when pairs table v1 is frozen (due Fri 9 Oct), could you send me its "
                                   "Drive path?"),
    "TACTICS_MAP_PATH": ("Dennis", "Hi Dennis, WildJailbreak's train.tsv has no tactics column; where should tactic "
                                   "labels come from, and could you share them as a file with source_row_id and tactic?"),
    "PREREG_CONFIRMED": ("Mentor", "Hi Alizishaan, could you confirm or change the [PROPOSED] thresholds in my "
                                   "pre-registration (list in prereg_questions.md) before the Mon 12 Oct lock?"),
    "COLAB_L4_HOURS_LEFT": ("Me", "Check Colab > Runtime > View resources (compute units), convert to L4 hours and type "
                                  "the number into Task 0."),
}


def json_path(repo=None):
    return Path(repo or REPO) / JSON_NAME


def load(repo=None):
    """Values from team_inputs.json (defaults for anything missing)."""
    values = dict(DEFAULTS)
    p = json_path(repo)
    if p.exists():
        values.update({k: v for k, v in json.loads(p.read_text()).items() if k in DEFAULTS})
    return values


def save(values, repo=None):
    data = {k: values.get(k, DEFAULTS[k]) for k in DEFAULTS}
    data["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = json_path(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    return p


def _writable(folder):
    """Write and delete a tiny test file. Returns (ok, reason)."""
    p = Path(folder) / f".a5_write_test_{uuid.uuid4().hex[:8]}"
    try:
        p.write_text("ok")
        p.unlink()
        return True, ""
    except OSError as e:
        return False, f"cannot write there ({type(e).__name__})"


def check(values):
    """One row per input: dict(name, state READY/WAITING/PROBLEM, detail, who, due, message). Never raises."""
    rows = []
    for name, (what, who, due, unlocks) in INPUTS.items():
        v = values.get(name, DEFAULTS[name])
        state, detail = "WAITING", ""
        try:
            if name == "SHARED_DRIVE_ROOT":
                if not v:
                    detail = "empty"
                elif not Path(v).is_dir():
                    state, detail = "PROBLEM", f"folder does not exist: {v}"
                else:
                    ok, why = _writable(v)
                    if not ok:
                        state, detail = "PROBLEM", why
                    elif str(v).rstrip("/") == TEST_FOLDER:
                        detail = "writable, but it is still the private test folder (A5_TEST_FOLDER)"
                    else:
                        state, detail = "READY", "exists and writable"
            elif name == "UNDERCURRENT_SAMPLE_DIR":
                if not v:
                    detail = "empty"
                elif not Path(v).is_dir():
                    state, detail = "PROBLEM", f"folder does not exist: {v}"
                elif not list(Path(v).rglob("*.safetensors")):
                    state, detail = "PROBLEM", "no .safetensors shard inside"
                else:
                    state, detail = "READY", f"{len(list(Path(v).rglob('*.safetensors')))} shard file(s)"
            elif name in ("PAIRS_TABLE_PATH", "TACTICS_MAP_PATH"):
                if not v:
                    detail = "empty"
                elif not Path(v).is_file():
                    state, detail = "PROBLEM", f"file does not exist: {v}"
                elif not str(v).endswith(".parquet"):
                    state, detail = "PROBLEM", "expected a .parquet file"
                else:
                    cols = set(_parquet_columns(v))
                    need = {"tactic"} if name == "TACTICS_MAP_PATH" else {"pair_id", "role", "intent", "split"}
                    missing = need - cols
                    state, detail = (("PROBLEM", f"missing column(s) {sorted(missing)}") if missing
                                     else ("READY", f"{len(cols)} columns, required ones present"))
            elif name == "PREREG_CONFIRMED":
                state, detail = ("READY", "mentor confirmed") if bool(v) else ("WAITING", "not ticked yet")
            elif name == "COLAB_L4_HOURS_LEFT":
                h = float(v or 0)
                if h <= 0:
                    detail = "not typed in yet"
                elif h < L4_HOURS_NEEDED:
                    state, detail = "PROBLEM", f"only {h:g} h; my runs need about {L4_HOURS_RANGE[0]}-{L4_HOURS_RANGE[1]} h: top up"
                else:
                    state, detail = "READY", f"{h:g} h >= {L4_HOURS_NEEDED} h"
        except Exception as e:  # noqa: BLE001  the check reports, it never crashes
            state, detail = "PROBLEM", f"could not check: {type(e).__name__}"
        to, msg = MESSAGES[name]
        rows.append(dict(name=name, value=v, state=state, detail=detail, who=who, due=due, what=what,
                         unlocks=unlocks, ask=to, message=msg))
    return rows


def _parquet_columns(path):
    """Column names only (reads the parquet footer, not the data)."""
    import pyarrow.parquet as pq

    return pq.read_schema(path).names


def report_lines(rows):
    out = []
    for r in rows:
        if r["state"] == "READY":
            out.append(f"READY    {r['name']}: {r['detail']}")
        elif r["state"] == "WAITING":
            who = "you" if r["ask"] == "Me" else f"ask {r['ask']}"
            note = f" [{r['detail']}]" if r["detail"] else ""
            out.append(f"WAITING  {r['name']}{note} ({who}: \"{r['message']}\")")
        else:
            out.append(f"PROBLEM  {r['name']}: {r['detail']}")
    return out


# ------------------------------------------------------------------------------------------------ pictures
def givers(who):
    """'Kayson / mentor' -> ['Kayson', 'Mentor'] (exact names from PEOPLE)."""
    names = {w.strip().lower() for w in who.split("/")}
    return [p for p in PEOPLE if p.lower() in names]


def draw_dependency(rows, path):
    """People (left) -> the input each gives (middle, coloured by its live state) -> tasks it unlocks (right)."""
    fig, ax = dg.canvas(13, 6.6, ps.tag("DIAGRAM") + " | who gives me what, and what it unlocks (input colours = live state)")
    py = {p: 82 - k * 21 for k, p in enumerate(PEOPLE)}
    people = {p: dg.node(ax, 8, py[p], p, w=11, h=8, shape="pill", fc="white", bold=True) for p in PEOPLE}
    n = len(rows)
    iy = {r["name"]: 90 - k * (80 / max(n - 1, 1)) for k, r in enumerate(rows)}
    unlocks = []
    for r in rows:
        for u in r["unlocks"]:
            if u not in unlocks:
                unlocks.append(u)
    uy = {u: 92 - k * (84 / max(len(unlocks) - 1, 1)) for k, u in enumerate(unlocks)}
    tasks = {u: dg.node(ax, 85, uy[u], u, w=26, h=6.5, fs=6.4, fc="#f6f5f0") for u in unlocks}
    for r in rows:
        fill = ps.STATUS_COLOR[r["state"]]
        box = dg.node(ax, 45, iy[r["name"]], f"{r['name']}\n{r['state']}", w=27, h=9, fc=fill, fs=6.6,
                      ls="-" if r["state"] == "READY" else "--")
        for giver in givers(r["who"]):  # "Kayson / mentor" -> both
            dg.arrow(ax, people[giver], box, lw=0.9, sides=("right", "left"))
        for u in r["unlocks"]:
            dg.arrow(ax, box, tasks[u], lw=0.8, sides=("right", "left"), color=ps.MUTED)
    dg.legend(ax, [("patch", ps.STATUS_COLOR["READY"], "READY", "-"), ("patch", ps.STATUS_COLOR["WAITING"], "WAITING", "--"),
                   ("patch", ps.STATUS_COLOR["PROBLEM"], "PROBLEM", "--")], loc="lower center", ncol=3)
    return dg.savefig(fig, path)


def draw_readiness(rows, path):
    """One bar per input, coloured by state, due date at the end of each bar."""
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.barh(yi, 1, height=0.65, color=ps.STATUS_COLOR[r["state"]], edgecolor=ps.SURFACE,
                hatch=None if r["state"] == "READY" else ("//" if r["state"] == "WAITING" else "xx"))
        ax.text(0.02, yi, r["state"], va="center", fontsize=7.5, fontweight="bold", color=ps.INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
        ax.text(1.03, yi, f"due {r['due']}  |  from {r['who']}", va="center", fontsize=7, color=ps.INK2)
    ax.set_yticks(y, [r["name"] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 2.3)
    ax.set_xticks([])
    ax.grid(False)
    ax.set_xlabel("state of each team input (bar colour and word), due date after the bar")
    ps.caption(fig, ps.tag("DIAGRAM") + " | Task 0 readiness of every team input", "computed live by the readiness check")
    return ps.savefig(fig, path)


def draw_compute(hours, path):
    """L4 hours I have vs the work plan's estimate for my GPU runs (one bar, as the plan gives only a total)."""
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(8, 2.8))
    h = float(hours or 0)
    lo, hi = L4_HOURS_RANGE
    ax.barh(1, h, height=0.5, color=ps.BLUE if h >= L4_HOURS_NEEDED else ps.STATUS_COLOR["PROBLEM"], label="L4 hours I have")
    ax.barh(0, lo, height=0.5, color=ps.MUTED, label="work plan estimate (low end)")
    ax.barh(0, hi - lo, left=lo, height=0.5, color=ps.AXIS, hatch="//", edgecolor=ps.SURFACE, label="estimate range 20-22 h")
    ax.text(h + 0.3, 1, f"{h:g} h" if h > 0 else "not typed in yet", va="center", fontsize=8, color=ps.INK)
    ax.text(hi + 0.3, 0, f"{lo}-{hi} h", va="center", fontsize=8, color=ps.INK)
    ax.axvline(L4_HOURS_NEEDED, color=ps.INK2, ls="--", lw=1)
    ax.set_yticks([0, 1], ["needed (estimate from work plan:\nGemma + Mistral harvests, judges,\nHF-bf16 and NF4 configs)", "I have"],
                  fontsize=7.5)
    ax.set_xlim(0, max(h, hi) * 1.25 + 1)
    ax.set_xlabel("Colab L4 GPU hours")
    ax.legend(loc="upper right", fontsize=6.5)
    ps.caption(fig, ps.tag("DIAGRAM") + " | Task 0 compute: L4 hours I have vs what my runs need",
               "estimate from work plan s.6.4 (a total only, so no per-run split is drawn)")
    return ps.savefig(fig, path)
