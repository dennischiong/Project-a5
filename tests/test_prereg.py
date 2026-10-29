"""Tests for analysis/prereg_tools.py: the [PROPOSED] parser, the lock and the hash check (temp folders only)."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analysis import plotstyle as ps  # noqa: E402
from analysis import prereg_tools as pt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CLEAN = "# Pre-registration\n\n| # | Metric |\n|---|---|\n| H1 | share below the midpoint, CI lower bound > 0.5 |\n"


def make_repo(tmp_path, text):
    (tmp_path / "prereg.md").write_text(text, encoding="utf-8")
    return tmp_path


# A draft in the same shape as prereg.md before the lock: the parser tests run on this, because the real
# prereg.md has no [PROPOSED] tags left once it is final.
DRAFT = """# Pre-registration draft

Status: DRAFT. Every numeric threshold below is a proposal: **[PROPOSED - confirm with mentor]**.

- Layer for the representation hypotheses H1, H2, H3, H6: the **last layer**.
  Norm convention: [PROPOSED - confirm with mentor: the last decoder-layer output BEFORE the final norm].

| # | Metric | Supported if |
|---|---|---|
| H3 | G2 AUROC | AUROC >= [PROPOSED - confirm with mentor: 0.8] |
| H5 | TPR at 1% FPR | margin [PROPOSED - confirm with mentor: 0.05] |
| H7 | AUROC change | within +/- [PROPOSED - confirm with mentor: 0.01] |

## Notes
- Multiple comparisons: [PROPOSED - confirm with mentor: Holm over H1-H7].
"""


def test_real_prereg_is_final():
    """prereg.md is final: no [PROPOSED] tag left, so Task 4.3 can lock it."""
    text = (REPO / "prereg.md").read_text(encoding="utf-8")
    assert pt.count_tags(text) == 0 and pt.find_proposed(text) == []
    assert "Status: FINAL" in text


def test_parser_on_a_draft():
    text = DRAFT
    items = pt.find_proposed(text)
    assert len(items) == pt.count_tags(text)
    general = [it for it in items if it["general"]]
    assert len(general) == 1                                   # the Status line
    by = {(tuple(it["hypotheses"]), it["proposed"]) for it in items if not it["general"]}
    assert (("H3",), "0.8") in by and (("H5",), "0.05") in by and (("H7",), "0.01") in by
    holm = [it for it in items if it["proposed"].startswith("Holm")]
    assert holm and holm[0]["hypotheses"] == pt.HYPS        # "H1-H7" is expanded
    norm = [it for it in items if "norm" in it["proposed"].lower()][0]
    assert norm["hypotheses"] == ["H1", "H2", "H3", "H6"]
    assert all(it["why"] for it in items)


def test_questions_markdown_skips_the_general_tag():
    items = pt.find_proposed(DRAFT)
    md = pt.questions_markdown(items)
    assert md.count("\n| ") == len([i for i in items if not i["general"]]) + 1  # header + one row per item
    assert "Status line" in md


def test_decision_rule_thresholds_mirror_prereg():
    """DECISION_RULES copies thresholds from prereg.md; if prereg.md changes, this test fails until both agree."""
    text = (REPO / "prereg.md").read_text(encoding="utf-8")
    for r in pt.DECISION_RULES:
        for t in r["thresholds"]:
            assert f"{abs(t):g}" in text, f"{r['h']} threshold {t} not found in prereg.md"
        if r["proposed"]:
            assert any(f"confirm with mentor: {abs(t):g}]" in text for t in r["thresholds"]), r["h"]


def test_lock_refuses_until_everything_is_confirmed(tmp_path):
    repo = make_repo(tmp_path, "x [PROPOSED - confirm with mentor: 0.8] y")
    ok, missing = pt.try_lock(repo, lock_requested=False, confirmed=False)
    assert not ok and len(missing) == 3 and not (repo / "prereg_lock.json").exists()
    ok, missing = pt.try_lock(repo, lock_requested=True, confirmed=True)
    assert not ok and "1 [PROPOSED" in missing[0]


def test_lock_writes_hash_and_detects_changes(tmp_path):
    repo = make_repo(tmp_path, CLEAN)
    assert pt.lock_state(repo)["state"] == "NOT LOCKED"
    ok, missing = pt.try_lock(repo, lock_requested=True, confirmed=True)
    assert ok and not missing
    lock = json.loads((repo / "prereg_lock.json").read_text())
    assert lock["sha256"] == pt.sha256_file(repo / "prereg.md") and len(lock["sha256"]) == 64
    assert pt.lock_state(repo)["state"] == "LOCKED"
    (repo / "prereg.md").write_text(CLEAN + "\nH8 added later\n", encoding="utf-8")   # edited after the lock
    assert pt.lock_state(repo)["state"] == "CHANGED"
    ok, missing = pt.try_lock(repo, lock_requested=True, confirmed=True)
    assert not ok and "changed since the lock" in missing[0]
    assert json.loads((repo / "prereg_lock.json").read_text()) == lock      # never overwritten


def test_amendment_keeps_the_original_lock_and_only_covers_that_text(tmp_path):
    repo = make_repo(tmp_path, CLEAN)
    pt.try_lock(repo, lock_requested=True, confirmed=True)
    original = json.loads((repo / "prereg_lock.json").read_text())
    with pytest.raises(ValueError):
        pt.record_amendment(repo, "nothing changed yet")               # LOCKED: nothing to amend
    (repo / "prereg.md").write_text(CLEAN.replace("Pre-registration", "Pre-registration (reworded)"), encoding="utf-8")
    with pytest.raises(ValueError):
        pt.record_amendment(repo, "  ")                                 # a note is required
    st = pt.record_amendment(repo, "editorial wording only")
    assert st["state"] == "LOCKED" and st["amended"]["note"] == "editorial wording only"
    assert st["locked_sha"] == original["sha256"] and st["date"] == original["date"]   # original lock kept
    lock = json.loads((repo / "prereg_lock.json").read_text())
    assert {k: lock[k] for k in original} == original and len(lock["amendments"]) == 1
    (repo / "prereg.md").write_text(CLEAN + "\nH8 added later\n", encoding="utf-8")       # a further edit is not covered
    assert pt.lock_state(repo)["state"] == "CHANGED"
    assert pt.try_lock(repo, lock_requested=True, confirmed=True)[0] is False


def test_apply_labels_sets_exploratory_titles(tmp_path, capsys):
    repo = make_repo(tmp_path, CLEAN)
    pt.try_lock(repo, lock_requested=True, confirmed=True)
    assert pt.apply_labels(repo)["state"] == "LOCKED" and ps.EXPLORATORY_NOTE is None
    (repo / "prereg.md").write_text(CLEAN + " ", encoding="utf-8")
    try:
        assert pt.apply_labels(repo)["state"] == "CHANGED"
        assert "WARNING" in capsys.readouterr().out
        assert ps.tag("PILOT", "m", "l", "p").startswith("PILOT | " + pt.EXPLORATORY)
    finally:
        ps.EXPLORATORY_NOTE = None


def test_prereg_figures_draw(tmp_path):
    items = pt.find_proposed(DRAFT)
    outs = [pt.draw_hypothesis_map(tmp_path / "map.png"), pt.draw_mentor_chart(items, tmp_path / "mentor.png"),
            pt.draw_decision_rules(tmp_path / "rules.png", single_dir=tmp_path / "rules"),
            pt.draw_lock_badge({"state": "NOT LOCKED"}, tmp_path / "badge.png", ["LOCK_PREREG is not ticked"])]
    assert all(p.exists() and p.stat().st_size > 5_000 for p in outs)
    assert len(list((tmp_path / "rules").glob("rule_H*.png"))) == 7
