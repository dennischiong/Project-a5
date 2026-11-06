"""Pre-registration tools (Task 4): mentor questions, the lock, and the changed after lock check.

  find_proposed(text)     every [PROPOSED - confirm with mentor] item in prereg.md, with its hypotheses
  questions_markdown()    numbered table: hypothesis | what is proposed | why it matters (-> prereg_questions.md)
  try_lock(...)           writes prereg_lock.json (sha256 of prereg.md + date) only when everything is confirmed
  lock_state(repo)        NOT LOCKED / LOCKED / CHANGED (prereg.md edited after the lock)
  record_amendment(...)   records an editorial (wording-only) change to the locked file, keeping the original lock
  apply_labels(repo)      for analysis scripts: loud WARNING + EXPLORATORY figure titles when CHANGED
Pictures: draw_hypothesis_map, draw_mentor_chart, draw_decision_rules, draw_lock_badge.

This module never changes prereg.md. Thresholds are only read, never edited.
"""
import hashlib
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import diagrams as dg
from analysis import plotstyle as ps

REPO = Path(__file__).resolve().parents[1]
TAG = re.compile(r"\[PROPOSED - confirm with mentor(?::\s*(?P<what>[^\]]*))?\]")
HYPS = [f"H{i}" for i in range(1, 8)]
EXPLORATORY = "EXPLORATORY (prereg changed after lock)"

# plain-English reasons, matched on the proposed text (first match wins), then on (hypothesis, value)
WHY_TEXT = [
    ("h1 version", "Which claim H1 makes: that most adversarial-harmful prompts cross the midpoint (A), or that they shift past it more than vanilla-harmful ones (B)."),
    ("norm", "Fixes exactly which vector H1-H3 and H6 use; pre- and post-norm vectors differ in size and direction."),
    ("highest vanilla", "Fixes how the probe's layer is chosen, so it cannot be picked after seeing test results."),
    ("equivalence margin", "How small a style-only score change must be to count as 'no effect'; H3(c) cannot be decided without it."),
    ("holm", "Whether to correct for testing seven hypotheses at once, which lowers the chance of a lucky false alarm."),
]
WHY_VALUE = {
    ("H3", "0.8"): "How well a 'looks like a jailbreak' probe (G2) must work before we say style is detectable.",
    ("H4", "50%"): "How much of the adversarial AUROC drop the augmented probe must win back to count as recovery.",
    ("H5", "0.05"): "How much worse than Llama Guard 3 the probe may be (TPR at 1% FPR) and still count as matching it.",
    ("H5", "5%"): "How cheap the probe must be compared with Llama Guard 3 to claim 'a small fraction of its compute'.",
    ("H7", "0.01"): "The largest AUROC change across serving configs that still counts as 'unchanged'.",
    ("H7", "1%"): "The largest share of prompts whose verdict may flip between configs and still count as robust.",
    ("H7", "0.9"): "The minimum agreement (Cohen's kappa) between configs to call individual verdicts stable.",
    ("H7", "50%"): "How much of the 4-bit loss a cheap threshold refit on ~200 prompts must recover.",
}

# Work plan s.7: which analysis answers which hypothesis, and where it appears in the paper (s.12).
ANALYSES = {  # analysis: (hypotheses it tests, figure)
    "A1": (["H6"], "F1"), "A9": (["H6"], "F1"), "A2": (["H1"], "F2"), "A4": (["H2"], "F4"),
    "A3": (["H4"], "not assigned in plan"), "A7": (["H3"], "appendix"), "G2": (["H3"], "not assigned in plan"),
    "G1": (["H5"], "F3a"), "X1": (["H7"], "F3b"), "X3": (["H7"], "F3b"), "X2": (["H7"], "appendix"),
    "X5": (["H1", "H7"], "appendix"), "A5": (["RQ3"], "appendix"), "A8": (["RQ3"], "appendix"),
}
MINE = {"A2", "A5", "A7", "A8", "X5"}  # Hasif's WS3 analyses

# Decision rules as number lines. MIRROR OF prereg.md: if a threshold there changes, change it here too
# (tests/test_prereg.py checks every value below still appears in prereg.md).
DECISION_RULES = [
    dict(h="H1", quantity="share of adversarial-harmful test prompts below the midpoint (CI lower bound)",
         xlim=(0, 1), thresholds=[0.5], supported=(0.5, 1), proposed=False, example=(0.62, 0.55, 0.69), rule="primary test, part 1 of 2: supported if the CI lower bound is above 0.5"),
    dict(h="H2", quantity="mean dp(success) - mean dp(fail) (whole CI)", xlim=(-1, 1), thresholds=[0], supported=(-1, 0),
         proposed=False, example=(-0.4, -0.7, -0.1), rule="supported if the whole CI is below 0 (plus the benign difference-in-differences check)"),
    dict(h="H3", quantity="(b) G2 AUROC, adversarial vs vanilla form", xlim=(0.5, 1), thresholds=[0.8],
         supported=(0.8, 1), proposed=False, example=(0.86, 0.82, 0.9), rule="part (b) only: supported if G2 AUROC >= 0.8 (parts a and c not drawn)"),
    dict(h="H4", quantity="(a) AUROC(vanilla test) - AUROC(adversarial test) (whole CI)", xlim=(-0.2, 0.4), thresholds=[0],
         supported=(0, 0.4), proposed=False, example=(0.12, 0.05, 0.19), rule="part (a) only: supported if the whole CI is above 0"),
    dict(h="H5", quantity="probe TPR - Llama Guard 3 TPR at 1% FPR (CI lower bound)", xlim=(-0.3, 0.3),
         thresholds=[-0.05], supported=(-0.05, 0.3), proposed=False, example=(-0.01, -0.04, 0.02), rule="supported if the CI lower bound is above -0.05 (plus the compute rule)"),
    dict(h="H6", quantity="median displacement / median noise floor (CI lower bound, log scale)", xlim=(1, 1000),
         thresholds=[10], supported=(10, 1000), proposed=False, example=(60, 35, 90), log=True, rule="supported if the CI lower bound is at least 10"),
    dict(h="H7", quantity="(a) AUROC change vs reference config, 16-bit (whole CI)", xlim=(-0.05, 0.05),
         thresholds=[-0.01, 0.01], supported=(-0.01, 0.01), proposed=False, example=(0.002, -0.004, 0.007), rule="part (a) only: supported if the whole CI lies within +/-0.01"),
]


# ------------------------------------------------------------------------------------------------ reading prereg.md
def _blocks(text):
    """Split into blocks: each table row, each bullet (with its continuation lines), each paragraph line."""
    blocks, cur = [], None
    for no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("|") or s.startswith("- ") or not s or not line.startswith(" "):
            if cur:
                blocks.append(cur)
            cur = [no, line] if s else None
        else:
            cur = [cur[0], cur[1] + " " + s] if cur else [no, line]
    if cur:
        blocks.append(cur)
    return blocks


def _hyps_in(block):
    s = block.strip()
    m = re.match(r"\|\s*(H\d)\s*\|", s)
    if m:
        return [m.group(1)]
    found = []
    for a, b in re.findall(r"H(\d)(?:\s*-\s*H(\d))?", s):
        rng = range(int(a), int(b) + 1) if b else [int(a)]
        found += [f"H{i}" for i in rng if f"H{i}" not in found]
    return found


def find_proposed(text):
    """List of dict(n, line, hypotheses, proposed, general, why). 'general' = the Status line's blanket tag."""
    items = []
    for no, block in _blocks(text):
        for m in TAG.finditer(block):
            what = (m.group("what") or "").strip()
            general = block.strip().startswith("Status:")
            hyps = [] if general else (_hyps_in(block) or ["all"])
            items.append(dict(n=len(items) + 1, line=no, hypotheses=hyps, proposed=what or "(value given just before the tag)",
                              general=general, why=_why(hyps, what)))
    return items


def _why(hyps, what):
    low = what.lower()
    for key, why in WHY_TEXT:
        if key in low:
            return why
    for h in hyps:
        if (h, what) in WHY_VALUE:
            return WHY_VALUE[(h, what)]
    return "A threshold the decision rule depends on; it must be fixed before any results are seen."


def count_tags(text):
    return len(TAG.findall(text))


def questions_markdown(items):
    rows = ["| # | Hypothesis | What is proposed | Why it matters |", "|---|---|---|---|"]
    k = 0
    for it in items:
        if it["general"]:
            continue
        k += 1
        rows.append(f"| {k} | {', '.join(it['hypotheses'])} | {it['proposed']} | {it['why']} |")
    head = ("# Questions for the mentor: pre-registration thresholds\n\nPlease confirm or change each proposal "
            "before the Mon 12 Oct lock. (Generated from prereg.md by Task 4.2.)\n\n")
    tail = ""
    if any(it["general"] for it in items):
        tail = "\n\nNote: prereg.md's Status line also carries a general [PROPOSED] tag; it is removed when locking.\n"
    return head + "\n".join(rows) + tail


def per_hypothesis_counts(items):
    counts = {h: 0 for h in HYPS}
    for it in items:
        for h in (HYPS if it["hypotheses"] == ["all"] else it["hypotheses"]):
            if h in counts:
                counts[h] += 1
    return counts


# ------------------------------------------------------------------------------------------------ the lock
def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_state(repo=None):
    """LOCKED when prereg.md matches the lock's sha256 or a recorded amendment (see record_amendment); the original
    lock date and hash are always reported, and amended names the amendment the current file matches."""
    repo = Path(repo or REPO)
    pp, lp = repo / "prereg.md", repo / "prereg_lock.json"
    cur = sha256_file(pp) if pp.exists() else None
    if not lp.exists():
        return dict(state="NOT LOCKED", current=cur)
    lk = json.loads(lp.read_text())
    amended = next((a for a in reversed(lk.get("amendments", [])) if a.get("sha256") == cur), None)
    locked = lk.get("sha256") == cur or amended is not None
    return dict(state="LOCKED" if locked else "CHANGED", date=lk.get("date"), locked_sha=lk.get("sha256"),
                current=cur, amended=amended)


def record_amendment(repo=None, note=""):
    """Record an editorial change to a locked prereg.md: appends {date, sha256, note} to prereg_lock.json and keeps the
    original lock (date and sha256) untouched, so the change stays visible. Only for wording that changes no
    hypothesis, metric, split or threshold; anything else is a deviation for the final report."""
    repo = Path(repo or REPO)
    st = lock_state(repo)
    if st["state"] != "CHANGED":
        raise ValueError(f"nothing to amend: prereg.md is {st['state']}")
    if not note.strip():
        raise ValueError("an amendment needs a note saying what changed")
    lp = repo / "prereg_lock.json"
    lk = json.loads(lp.read_text())
    lk.setdefault("amendments", []).append(dict(date=time.strftime("%Y-%m-%d", time.gmtime()),
                                                sha256=st["current"], note=note.strip()))
    lp.write_text(json.dumps(lk, indent=2))
    return lock_state(repo)


def try_lock(repo=None, lock_requested=False, confirmed=False):
    """Write prereg_lock.json if LOCK_PREREG is ticked, PREREG_CONFIRMED is ticked and no [PROPOSED] tag remains.
    Returns (locked: bool, list of what is still missing). Never overwrites an existing lock."""
    repo = Path(repo or REPO)
    st = lock_state(repo)
    if st["state"] == "LOCKED":
        return True, []
    missing = []
    if st["state"] == "CHANGED":
        missing.append("prereg_lock.json exists but prereg.md has changed since the lock. Nothing is re-locked "
                       "automatically: list the change as a deviation; only the mentor can approve a new lock.")
    if not lock_requested:
        missing.append("the LOCK_PREREG box is not ticked")
    if not confirmed:
        missing.append("PREREG_CONFIRMED in Task 0 is not ticked (the thresholds are not confirmed yet)")
    text = (repo / "prereg.md").read_text(encoding="utf-8") if (repo / "prereg.md").exists() else None
    if text is None:
        missing.append("prereg.md not found (run Task 1)")
    elif count_tags(text):
        missing.append(f"{count_tags(text)} [PROPOSED - confirm with mentor] tag(s) remain in prereg.md: write the "
                       "confirmed values into prereg.md inside cell 1.2's FILES, then re-run Task 1")
    if missing:
        return False, missing
    lock = dict(file="prereg.md", sha256=sha256_file(repo / "prereg.md"), date=time.strftime("%Y-%m-%d", time.gmtime()),
                locked_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (repo / "prereg_lock.json").write_text(json.dumps(lock, indent=2))
    return True, []


def apply_labels(repo=None):
    """Call at the start of every analysis script. If prereg.md changed after the lock: loud WARNING, and every
    figure title gets EXPLORATORY (via plotstyle). Returns the lock state dict (also stored in results files)."""
    st = lock_state(repo)
    if st["state"] == "CHANGED":
        bar = "!" * 100
        print(f"{bar}\nWARNING: prereg.md changed after it was locked on {st.get('date')}. Every output and figure is "
              f"labelled {EXPLORATORY}.\n{bar}")
        ps.EXPLORATORY_NOTE = EXPLORATORY
    else:
        ps.EXPLORATORY_NOTE = None
    return st


# ------------------------------------------------------------------------------------------------ pictures
def draw_hypothesis_map(path):
    """H1-H7 -> the analyses that test them -> the figure they appear in. Hasif's analyses in the accent colour."""
    fig, ax = dg.canvas(13, 7.4, ps.tag("DIAGRAM") + " | hypothesis map: H1-H7 -> analyses that test them (work plan "
                        "'Answers' column) -> paper figure; my WS3 analyses in orange")
    left = HYPS + ["RQ3"]
    hy = {h: 92 - k * (84 / (len(left) - 1)) for k, h in enumerate(left)}
    hn = {h: dg.node(ax, 8, hy[h], h if h != "RQ3" else "RQ3\n(no hypothesis)", w=10, h=7, fc="white", bold=True,
                     fs=7.5) for h in left}
    names = list(ANALYSES)
    ay = {a: 95 - k * (90 / (len(names) - 1)) for k, a in enumerate(names)}
    figs = ["F1", "F2", "F3a", "F3b", "F4", "appendix", "not assigned in plan"]
    fy = {f: 90 - k * (80 / (len(figs) - 1)) for k, f in enumerate(figs)}
    fn = {f: dg.node(ax, 88, fy[f], f, w=17, h=7, fc="#f6f5f0", fs=7) for f in figs}
    for a in names:
        hyps, f = ANALYSES[a]
        mine = a in MINE
        an = dg.node(ax, 48, ay[a], a + ("  (mine)" if mine else ""), w=13, h=5, fc="#fbe0d4" if mine else "#eef3fa",
                     ec=ps.ORANGE if mine else ps.INK2, lw=1.8 if mine else 1.0, fs=6.8, bold=mine)
        for h in hyps:
            dg.arrow(ax, hn[h], an, lw=0.8, sides=("right", "left"), color=ps.ORANGE if mine else ps.MUTED)
        dg.arrow(ax, an, fn[f], lw=0.8, sides=("right", "left"), color=ps.ORANGE if mine else ps.MUTED)
    ax.text(8, 1, "hypotheses", ha="center", fontsize=7, color=ps.INK2)
    ax.text(48, -1.5, "analyses (A = Track 1, G = Track 2, X = Track 3)", ha="center", fontsize=7, color=ps.INK2)
    ax.text(88, 1, "paper figure", ha="center", fontsize=7, color=ps.INK2)
    return dg.savefig(fig, path)


def draw_mentor_chart(items, path):
    """One bar per hypothesis: how many [PROPOSED] items still need the mentor (drops to 0 as they are confirmed)."""
    ps.apply_style()
    counts = per_hypothesis_counts(items)
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    vals = [counts[h] for h in HYPS]
    bars = ax.bar(HYPS, vals, color=[ps.STATUS_COLOR["WAITING"] if v else ps.STATUS_COLOR["DONE"] for v in vals])
    ax.bar_label(bars, labels=[str(v) if v else "0 (done)" for v in vals], fontsize=7.5, color=ps.INK2)
    ax.set(ylabel="[PROPOSED] items still open (count)", xlabel="hypothesis", ylim=(0, max(vals + [1]) + 1))
    n_general = sum(it["general"] for it in items)
    ps.caption(fig, ps.tag("DIAGRAM") + f" | what still needs the mentor: {sum(not it['general'] for it in items)} "
               "open items in prereg.md", "counted live from prereg.md; an item that covers several hypotheses counts "
               f"for each; {n_general} general tag in the Status line not counted")
    return ps.savefig(fig, path)


def _rule_axes(ax, r):
    lo, hi = r["xlim"]
    log = r.get("log", False)
    s0, s1 = r["supported"]
    ax.axvspan(lo, hi, color=ps.STATUS_COLOR["PROBLEM"], alpha=0.12, lw=0)
    ax.axvspan(s0, s1, color=ps.STATUS_COLOR["READY"], alpha=0.30, lw=0)
    for t in r["thresholds"]:
        ax.axvline(t, color=ps.INK, lw=1.5, ls="--")
        ax.text(t, 0.93, f"{t:g}" + (" (PROPOSED)" if r["proposed"] else ""), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.5, color=ps.INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    est, clo, chi = r["example"]
    ax.errorbar([est], [0.4], xerr=[[est - clo], [chi - est]], fmt="o", color=ps.BLUE, capsize=4, lw=2, ms=5)
    ax.text(chi, 0.4, "  SYNTHETIC EXAMPLE 95% CI", va="center", fontsize=6.2, color=ps.BLUE)
    if log:
        ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel(r["quantity"], fontsize=7)
    ax.set_title(f"{r['h']}: {r['rule']}", fontsize=8, loc="left")


def draw_decision_rules(path, single_dir=None):
    """All seven decision rules as number lines (green = supported zone, red = not supported), each with a
    SYNTHETIC EXAMPLE CI. single_dir: also save one PNG per hypothesis there."""
    ps.apply_style()
    fig, axes = plt.subplots(len(DECISION_RULES), 1, figsize=(9, 1.25 * len(DECISION_RULES) + 1.2))
    for ax, r in zip(axes, DECISION_RULES):
        _rule_axes(ax, r)
    ps.caption(fig, ps.tag("SYNTHETIC", "any", "as pre-registered", "as pre-registered") + " | decision rules H1-H7 "
               "(blue example CIs are SYNTHETIC, not results)", "thresholds from prereg.md (mirror in "
               "analysis/prereg_tools.DECISION_RULES); 'whole CI' = the 95% interval must lie in the green zone")
    out = ps.savefig(fig, path)
    if single_dir:
        for r in DECISION_RULES:
            f1, a1 = plt.subplots(figsize=(8, 1.9))
            _rule_axes(a1, r)
            ps.caption(f1, ps.tag("SYNTHETIC", "any", "as pre-registered", "as pre-registered") + f" | decision rule {r['h']}",
                       "blue CI is a SYNTHETIC EXAMPLE, not a result")
            ps.savefig(f1, Path(single_dir) / f"rule_{r['h']}.png")
    return out


def draw_lock_badge(state, path, missing=()):
    """LOCKED (date, short hash) / NOT LOCKED (what is missing) / CHANGED AFTER LOCK."""
    ps.apply_style()
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis("off")
    s = state["state"]
    color = {"LOCKED": ps.STATUS_COLOR["READY"], "NOT LOCKED": ps.STATUS_COLOR["WAITING"], "CHANGED": ps.STATUS_COLOR["PROBLEM"]}[s]
    head = {"LOCKED": f"LOCKED  {state.get('date')}  sha256 {str(state.get('locked_sha'))[:12]}",
            "NOT LOCKED": "NOT LOCKED", "CHANGED": f"CHANGED AFTER LOCK ({state.get('date')}): results are EXPLORATORY"}[s]
    ax.text(0.02, 0.72, head, fontsize=14, fontweight="bold", color="white", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="none"))
    import textwrap

    detail = ("\n".join(textwrap.fill(f"- {m}", 115, subsequent_indent="  ") for m in missing) if missing else
              ("prereg.md matches the locked hash" if s == "LOCKED" else ""))
    ax.text(0.02, 0.32, detail, fontsize=7, color=ps.INK, va="center", transform=ax.transAxes)
    ps.caption(fig, ps.tag("DIAGRAM") + " | pre-registration lock status", "from prereg_lock.json and prereg.md")
    return ps.savefig(fig, path)
