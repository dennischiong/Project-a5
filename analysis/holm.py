"""Holm correction over the seven primary tests H1-H7 (prereg.md, Notes). WS3 computes H1, H2 and H3; H4-H7 come
from the probe work and are added through a small JSON file when they exist.

The pre-registration states every rule as a 95% CI condition and gives no p-values, so each rule is turned into a
bootstrap p-value from the same resamples (2,000, seed 0) that give its CI, built so that p < 0.05 when the CI
condition holds (at the boundary the +1 below makes p slightly larger, e.g. Gemma-2-9B's H2 DiD, CI upper bound -0.2, p 0.051):
  CI excludes v   -> p = 2 x min(share of resamples <= v, share >= v)
  CI lower > v    -> p = 2 x share of resamples <= v
  CI upper < v    -> p = 2 x share of resamples >= v
  CI inside +/- m -> p = 2 x max(share <= -m, share >= m)          (equivalence, H3 c)
Shares use (k + 1) / (n + 1), so no p-value is ever 0, and p is capped at 1. A rule that needs several conditions
(H1 primary, H2, H3) is an intersection-union test: its p-value is the largest of its parts. A test that cannot be
computed (Mistral-7B H2: no refused benign prompt) enters the family with p = 1.

Choices the pre-registration left open, fixed on 11 Nov 2026 after H1-H3 had been read (reported as a deviation):
  - family-wise alpha 0.05, matching the 95% CIs;
  - one Holm family of seven tests per model (each model is its own replication), not one family across models;
  - H3 (b) enters with the bootstrap p-value of AUROC > 0.8, stricter than its point-estimate rule.
Until H4-H7 exist the result is provisional: a test with p <= 0.05 / 7 is rejected whatever they turn out to be, a
test with p > 0.05 is never rejected, and anything in between waits for them.

    python analysis/holm.py [--data-root ...] [--probe-pvalues results/probe_pvalues.json]
probe_pvalues.json: {"<model tag>": {"H4": 0.003, "H5": 0.2, ...}, ...}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import lib  # noqa: E402

FAMILY = ("H1", "H2", "H3", "H4", "H5", "H6", "H7")
ALPHA = 0.05
NOTE = __doc__.split("Until H4-H7", 1)[0].strip()


# ------------------------------------------------------------------------------------------------ p-values
def _share(k, n):
    return (k + 1) / (n + 1)


def p_excludes(boots, v=0.0):
    b = np.asarray(boots, float)
    return min(1.0, 2 * min(_share((b <= v).sum(), len(b)), _share((b >= v).sum(), len(b))))


def p_above(boots, v):
    b = np.asarray(boots, float)
    return min(1.0, 2 * _share((b <= v).sum(), len(b)))


def p_below(boots, v):
    b = np.asarray(boots, float)
    return min(1.0, 2 * _share((b >= v).sum(), len(b)))


def p_inside(boots, m):
    b = np.asarray(boots, float)
    return min(1.0, 2 * max(_share((b <= -m).sum(), len(b)), _share((b >= m).sum(), len(b))))


def boot_means(values, n_boot=2000, seed=0):
    """Resampled means, drawn exactly as lib.bootstrap_ci draws them (same rng, same index matrix)."""
    values = np.asarray(values, float)
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def boot_contrast(groups, signs, n_boot=2000, seed=0):
    """Resamples of judge_labels.contrast_ci, drawn in the same order."""
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for s, x in zip(signs, groups):
        x = np.asarray(x, float)
        boots += s * x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return boots


# ------------------------------------------------------------------------------------------------ H1, H2, H3
def h1_p(pa, pv, midpoint, n_boot=2000, seed=0):
    """H1 primary: share of adversarial-harmful test prompts below the midpoint has CI lower bound > 0.5 AND the
    difference from the vanilla twins has a CI excluding 0. Inputs: test harmful pairs only."""
    below_a = (np.asarray(pa) < midpoint).astype(float)
    below_v = (np.asarray(pv) < midpoint).astype(float)
    share, diff = boot_means(below_a, n_boot, seed), boot_means(below_a - below_v, n_boot, seed)
    parts = dict(share_above_half=p_above(share, 0.5), difference_excludes_0=p_excludes(diff))
    return dict(p=max(parts.values()), parts=parts, ci_share=_ci(share), ci_difference=_ci(diff))


def h2_p(win, lose, answered, declined, n_boot=2000, seed=0):
    """H2: difference (success - fail) with CI upper bound < 0 AND difference-in-differences CI excluding 0."""
    if min(len(win), len(lose)) < 2:
        return dict(p=1.0, parts={}, computable=False, why="fewer than 2 successful or failed jailbreaks")
    diff = boot_contrast([win, lose], [1, -1], n_boot, seed)
    if min(len(answered), len(declined)) < 2:
        return dict(p=1.0, parts=dict(difference_below_0=p_below(diff, 0.0)), computable=False,
                    why="fewer than 2 refused or answered benign prompts, so no benign control", ci_difference=_ci(diff))
    did = boot_contrast([win, lose, answered, declined], [1, -1, -1, 1], n_boot, seed)
    parts = dict(difference_below_0=p_below(diff, 0.0), did_excludes_0=p_excludes(did))
    return dict(p=max(parts.values()), parts=parts, computable=True, ci_difference=_ci(diff), ci_did=_ci(did))


def h3_p(Hv, Ha, meta, n_boot=2000, seed=0):
    """H3 (a) AND (b) AND (c), with the probes and resamples of analysis/h3_test.py."""
    from sklearn.metrics import roc_auc_score
    from analysis import h3_test
    Hv, Ha = np.asarray(Hv, np.float32), np.asarray(Ha, np.float32)
    split, harm = meta["split"].to_numpy(), (meta["intent"] == "harmful").to_numpy()
    tr, te = split == "train", split == "test"
    teb, teh = te & ~harm, te & harm
    r_hat = lib.fit_direction_train(Hv[tr], meta[tr].assign(role="vanilla"))
    dp = lib.project(Ha, r_hat) - lib.project(Hv, r_hat)
    style = boot_means(dp[teb], n_boot, seed)

    g2 = h3_test.fit_probe(np.r_[Hv[tr], Ha[tr]], np.r_[np.zeros(tr.sum()), np.ones(tr.sum())])
    neg, pos = g2.predict_proba(Hv[teb])[:, 1], g2.predict_proba(Ha[teb])[:, 1]
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    idx = np.random.default_rng(seed).integers(0, len(neg), (n_boot, len(neg)))
    auroc = np.array([roc_auc_score(y, np.r_[neg[i], pos[i]]) for i in idx])

    hp = h3_test.fit_probe(Hv[tr], harm[tr].astype(int))
    p_v, p_a = hp.predict_proba(Hv)[:, 1], hp.predict_proba(Ha)[:, 1]
    m = h3_test.MARGIN_SHARE * float(p_v[teh].mean() - p_v[teb].mean())
    diff = boot_means(p_a[teb] - p_v[teb], n_boot, seed)
    parts = dict(a_style_excludes_0=p_excludes(style), b_auroc_above_0_8=p_above(auroc, h3_test.AUROC_MIN),
                 c_score_difference_inside_margin=p_inside(diff, m))
    return dict(p=max(parts.values()), parts=parts, margin=m, ci_style=_ci(style), ci_auroc=_ci(auroc), ci_difference=_ci(diff))


def _ci(boots):
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


# ------------------------------------------------------------------------------------------------ Holm
def holm(pvalues, family=FAMILY, alpha=ALPHA):
    """pvalues: {test: p}; tests of the family that are missing make the result provisional.
    Returns {test: dict(p, adjusted_p or None, decision)} with decision one of rejected / not rejected (complete family)
    or rejected whatever the missing tests give / not rejected whatever the missing tests give / waits for <missing>."""
    m = len(family)
    have = {t: float(pvalues[t]) for t in family if pvalues.get(t) is not None}
    missing = [t for t in family if t not in have]
    out = {}
    if not missing:
        order = sorted(have, key=have.get)
        running = 0.0
        for i, t in enumerate(order):
            running = max(running, min(1.0, (m - i) * have[t]))
            out[t] = dict(p=have[t], adjusted_p=running, decision="rejected" if running <= alpha else "not rejected")
        return out
    for t, p in have.items():
        if p <= alpha / m:
            d = "rejected whatever the missing tests give"
        elif p > alpha:
            d = "not rejected whatever the missing tests give"
        else:
            d = f"waits for {', '.join(missing)}"
        out[t] = dict(p=p, adjusted_p=None, decision=d)
    return out


# ------------------------------------------------------------------------------------------------ run
def labelled_folder(folders, tag, data_root):
    """The folder H2 uses for a model: one whose prompt set has WildGuard labels, C1 first (as step_h2 picks it).
    The full Gemma and Mistral runs have no labels; their labels belong to the 4,000-prompt subset runs."""
    from analysis import judge_labels as jl
    from analysis import regenerate_all as ra
    rows = [(f, ra.folder_info(f)) for f in folders]
    rows = [(f, i) for f, i in rows if i["tag"] == tag and jl.find_labels(data_root, tag, i["subset"])]
    return min(rows, key=lambda r: r[1]["config"] != "c1")[0] if rows else None


def model_pvalues(folder, data_root, n_boot=2000, seed=0, h2_folder=None):
    """H1, H2 and H3 p-values for one model, pre-norm (the pre-registered vector). H1 and H3 use `folder`; H2 uses
    `h2_folder` (the folder whose prompts have labels) when given, else `folder`."""
    from analysis import geometry as g
    from analysis import judge_labels as jl
    from analysis import regenerate_all as ra
    from analysis import ws3_pilot as wp
    d = wp.load(folder, "prenorm")
    Hv, Ha, mw = wp.pair_arrays(d, "WildJailbreak")
    a2 = g.a2_projection(Hv, Ha, mw, n_boot=n_boot, seed=seed)
    harm = (mw["intent"] == "harmful").to_numpy()
    te = (mw["split"] == "test").to_numpy()
    out = dict(folder=Path(folder).name, H1=h1_p(a2["pa"][te & harm], a2["pv"][te & harm], a2["midpoint"], n_boot, seed))

    hf = Path(h2_folder or folder)
    info = ra.folder_info(hf)
    judged = jl.find_labels(data_root, info["tag"], info["subset"])
    if judged and hf != Path(folder):
        d = wp.load(hf, "prenorm")
        Hv2, Ha2, mw = wp.pair_arrays(d, "WildJailbreak")
        a2 = g.a2_projection(Hv2, Ha2, mw, n_boot=n_boot, seed=seed)
        harm = (mw["intent"] == "harmful").to_numpy()
        te = (mw["split"] == "test").to_numpy()
    if judged:
        out["H2_folder"] = hf.name
        al = jl.align(d["prompts"], jl.load_labels(judged))
        rows = mw["row_a"].to_numpy()
        valid, refused = al["valid"][rows], al["refused"][rows]
        success = jl.SUCCESS_RULES[jl.H2_SUCCESS_RULE](al, rows)
        dp = a2["dp"]
        out["H2"] = h2_p(dp[te & harm & valid & success], dp[te & harm & valid & ~success],
                         dp[te & ~harm & valid & ~refused], dp[te & ~harm & valid & refused], n_boot, seed)
    else:
        out["H2"] = None                                     # no WildGuard labels for this model: H2 not run
    d = wp.load(folder, "prenorm")
    Hv, Ha, mw = wp.pair_arrays(d, "WildJailbreak")
    out["H3"] = h3_p(Hv, Ha, mw, n_boot, seed)
    return out


def main():
    from analysis import regenerate_all as ra
    from analysis import ws3_data
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=ra.DATA_ROOT)
    ap.add_argument("--probe-pvalues", default=str(REPO / "results" / "probe_pvalues.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    probe = json.loads(Path(a.probe_pvalues).read_text()) if Path(a.probe_pvalues).exists() else {}
    folders = ra.find_folders(a.data_root)
    chosen, _ = ra.pick_model_folders(folders)
    res = dict(note=NOTE, alpha=ALPHA, family=list(FAMILY), probe_pvalues_file=Path(a.probe_pvalues).name if probe else None,
               models={})
    for tag, f in chosen.items():
        mp = model_pvalues(f, a.data_root, a.n_boot, h2_folder=labelled_folder(folders, tag, a.data_root))
        pv = {t: (mp[t]["p"] if mp.get(t) else None) for t in ("H1", "H2", "H3")}
        pv.update({t: probe.get(tag, {}).get(t) for t in ("H4", "H5", "H6", "H7")})
        res["models"][tag] = dict(tests=mp, pvalues=pv, holm=holm(pv))
        print(f"{tag}: " + "; ".join(f"{t} p={v['p']:.4f} -> {v['decision']}" for t, v in res["models"][tag]["holm"].items()))
    ws3_data.save_json(res, REPO / "results" / "WS3_Holm.json")
    from analysis import figures as F
    print("figure", F.holm_pvalues(json.loads((REPO / "results" / "WS3_Holm.json").read_text()), REPO / "figures" / "WS3" / "Holm_pvalues.png"))


if __name__ == "__main__":
    main()
