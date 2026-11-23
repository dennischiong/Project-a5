"""WildGuard labels (harvest/ws3_judge.py) joined to the activations by prompt_key, for H2 and A8. CPU only, numbers only.

Labels are read from <data root>/ws3_judges/. A model without labels simply has no H2 row and keeps the A8 proxy.

H2 (prereg.md): dp per harmful test pair, split by whether the jailbreak succeeded; supported if mean dp(success) -
mean dp(fail) < 0 with CI excluding 0, and the difference-in-differences against benign prompts has CI excluding 0.
The prereg does not define these splits exactly. Fixed by Hasif on 10 Nov 2026, before any label was read (mentor informed):
  success   H2_SUCCESS_RULE = harmful_response decides (WildGuard judges the reply harmful, as in attack-success rates).
            Checks, reported but not deciding: harmful_not_refused, and complied (harmful split complied vs refused, the
            same yardstick as the benign split, to tell a harm effect from a refusal effect).
  benign    answered vs refused.
  replies   cut off at 256 new tokens are kept (same cap as WS2). A check repeats the deciding rule without them; dropping
            them removes mostly long compliant replies, so that check is biased and is reported as such.
The label of a pair is the verdict on the reply to its ADVERSARIAL prompt. Rows without a clean verdict are left out and counted.
The Holm correction over H1-H7 is not applied here. Replies are vLLM, activations HF batch 1: every result names both.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import geometry as g
from analysis import lib

H2_SUCCESS_RULE = "harmful_response"   # fixed 10 Nov 2026 before the labels were read
SUCCESS_RULES = {
    "harmful_response": lambda al, rows: al["harmful_response"][rows],
    "harmful_not_refused": lambda al, rows: al["harmful_response"][rows] & ~al["refused"][rows],
    "complied": lambda al, rows: ~al["refused"][rows],
}
NOTE = ("Pre-registered H2, run after the 1 Nov results freeze: report as a deviation. Deciding rule harmful_response, "
        "fixed 10 Nov before the labels were read; harmful_not_refused, complied and the no-truncation run are checks. "
        "No Holm correction applied. Labels: WildGuard on vLLM replies (256-token cap); activations: see activations.")


# ------------------------------------------------------------------------------------------------ loading
def find_labels(data_root, tag, subset):
    """Finished label folder for a model on the same prompt subset, or None. Timing-test folders are ignored."""
    found = []
    for folder in sorted((Path(data_root) / "ws3_judges").glob("*")):
        f = folder / "meta.json"
        if not f.exists() or "smoke" in folder.name:
            continue
        m = json.loads(f.read_text())
        if m.get("model_tag") == tag and (m.get("subset") or None) == (subset or None):
            found.append(folder)
    return found[-1] if found else None


def load_labels(folder):
    return pd.read_parquet(Path(folder) / "labels.parquet")


def align(prompts, labels):
    """Per row of an activation folder's prompts table: valid (a clean verdict exists), refused, harmful_response,
    truncated (the reply hit the token cap)."""
    lab = labels.set_index("prompt_key").reindex(prompts["prompt_key"])
    return dict(valid=(lab["status"] == "ok").to_numpy(bool), refused=(lab["refusal"] == "yes").to_numpy(bool),
                harmful_response=(lab["harmful_response"] == "yes").to_numpy(bool),
                truncated=lab["reply_truncated"].fillna(False).to_numpy(bool))


# ------------------------------------------------------------------------------------------------ statistics
def contrast_ci(groups, signs, n_boot=2000, seed=0, level=0.95):
    """Sum of sign x mean over independent groups, with a percentile bootstrap over pairs, resampled within each group
    so no resample loses a group. Returns (estimate, lo, hi)."""
    groups = [np.asarray(x, float) for x in groups]
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for s, x in zip(signs, groups):
        boots += s * x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return (float(sum(s * x.mean() for s, x in zip(signs, groups))), float(np.percentile(boots, tail)),
            float(np.percentile(boots, 100 - tail)))


def excludes_zero(ci):
    return ci[1] > 0 or ci[2] < 0


def h2_verdict(difference, did):
    """prereg.md's H2 rule. A real harmful difference that is not shown to be specific to harmful prompts is neither
    supported nor plainly unsupported, and says so."""
    if difference is None:
        return "not_computable"
    if not (difference[0] < 0 and difference[2] < 0):
        return "not_supported"
    if did is None:
        return "not_computable_no_benign_control"
    return "supported" if excludes_zero(did) else "harmful_effect_not_specific"


def h2_result(dp, intent, split, valid, success, refused, n_boot=2000, seed=0):
    """H2 for one source and one success rule. All arrays have one entry per pair; valid, success and refused describe
    the reply to the adversarial prompt. A group of fewer than 2 pairs is reported as not computable, never as a result."""
    dp, intent, split = np.asarray(dp, float), np.asarray(intent), np.asarray(split)
    valid, success, refused = (np.asarray(x, bool) for x in (valid, success, refused))
    test = split == "test"
    harm, ben = test & (intent == "harmful"), test & (intent == "benign")
    win, lose = dp[harm & valid & success], dp[harm & valid & ~success]
    answered, declined = dp[ben & valid & ~refused], dp[ben & valid & refused]
    difference = did = None
    if min(len(win), len(lose)) >= 2:
        difference = contrast_ci([win, lose], [1, -1], n_boot, seed)
        if min(len(answered), len(declined)) >= 2:
            did = contrast_ci([win, lose, answered, declined], [1, -1, -1, 1], n_boot, seed)
    return dict(n_success=len(win), n_fail=len(lose), n_harmful_no_verdict=int((harm & ~valid).sum()),
                n_benign_answered=len(answered), n_benign_refused=len(declined), n_benign_no_verdict=int((ben & ~valid).sum()),
                mean_dp_success=float(win.mean()) if len(win) else None, mean_dp_fail=float(lose.mean()) if len(lose) else None,
                difference=difference, difference_in_differences=did, verdict=h2_verdict(difference, did))


def h2_for_folder(folder, labels, view="prenorm", n_boot=2000, seed=0):
    """H2 for every success rule. Decided on WildJailbreak, the only source with benign twins. SORRY-Bench pairs are
    projected on the same r_hat and reported separately, without a benign control (exploratory)."""
    from analysis import ws3_pilot as wp
    d = wp.load(folder, view)
    al = align(d["prompts"], labels)
    Hv, Ha, mw = wp.pair_arrays(d, "WildJailbreak")
    a2 = g.a2_projection(Hv, Ha, mw, n_boot=n_boot, seed=seed)
    out = dict(view=view, n_prompts_with_verdict=int(al["valid"].sum()), n_prompts=len(al["valid"]), wildjailbreak={}, sorry_exploratory={})
    Sv, Sa, ms = wp.pair_arrays(d, "SORRY-Bench")
    for rule, fn in SUCCESS_RULES.items():
        rows = mw["row_a"].to_numpy()
        out["wildjailbreak"][rule] = h2_result(a2["dp"], mw["intent"].to_numpy(), mw["split"].to_numpy(), al["valid"][rows],
                                               fn(al, rows), al["refused"][rows], n_boot, seed)
        if len(ms):
            rows = ms["row_a"].to_numpy()
            out["sorry_exploratory"][rule] = h2_result(lib.project(Sa - Sv, a2["r_hat"]), ms["intent"].to_numpy(), ms["split"].to_numpy(),
                                                       al["valid"][rows], fn(al, rows), al["refused"][rows], n_boot, seed)
    # check: the deciding rule without replies cut off at the token cap (biased: drops mostly long compliant replies)
    rows = mw["row_a"].to_numpy()
    out["wildjailbreak_excluding_truncated"] = h2_result(
        a2["dp"], mw["intent"].to_numpy(), mw["split"].to_numpy(), al["valid"][rows] & ~al["truncated"][rows],
        SUCCESS_RULES[H2_SUCCESS_RULE](al, rows), al["refused"][rows], n_boot, seed)
    return out


# ------------------------------------------------------------------------------------------------ A8
def refusal_angle(H, rows_train, r_hat, aligned):
    """A8: angle between r_hat and the refusal direction fitted on TRAIN rows with a clean WildGuard verdict (refused -
    answered). The angle is None unless both classes have rows."""
    rows = np.asarray(rows_train)
    use = rows[aligned["valid"][rows]]
    refused = aligned["refused"][use]
    out = dict(n_train_rows=len(use), n_refused=int(refused.sum()), n_train_no_verdict=int(len(rows) - len(use)), angle_deg=None)
    if refused.any() and not refused.all():
        out["angle_deg"] = g.angle_between(r_hat, g.refusal_direction(np.asarray(H)[use], refused))
    return out
