"""H3 (pre-registered): does the jailbreak's style move the hidden state on its own, without fooling the harmfulness
probe? Three parts, all on WildJailbreak TEST pairs at prompt[-1], last layer (pre-norm deciding, post-norm reported):

  (a) style part: mean dp of benign pairs along r_hat (r_hat fitted on vanilla TRAIN); CI must exclude 0.
  (b) G2 form probe: trained on TRAIN pairs to tell the adversarial prompt from the vanilla one (both intents),
      scored on benign TEST pairs (adversarial-benign vs vanilla-benign); AUROC must be >= 0.8.
  (c) harmfulness probe: trained on vanilla TRAIN prompts (harmful vs benign); its P(harmful) on adversarial-benign
      minus vanilla-benign, paired over benign TEST pairs, must have its CI inside +/- m,
      m = 0.1 x (mean P on vanilla-harmful TEST prompts - mean P on vanilla-benign TEST prompts).
  H3 is supported only if (a), (b) and (c) all hold.

Choices the pre-registration left open, fixed on 11 Nov 2026 before any H3 number was computed (reported as a deviation):
  - both probes are StandardScaler + LogisticRegression(C=1), the probe already used for the A3 pilot, fitted on TRAIN
    pairs only;
  - (b) is decided on the AUROC point estimate, as the rule is worded (AUROC >= 0.8); its CI is reported next to it;
  - (a) is recomputed on TEST pairs, as the pre-registration's data column says; the A7 numbers in the appendix use
    validation + test pairs;
  - the pairs table has one tactic label for all WildJailbreak pairs, so H3 is pooled, not per tactic (deviation D4).
95% CIs: percentile bootstrap over pairs, 2,000 resamples, seed 0.
"""
import numpy as np

from analysis import lib

AUROC_MIN = 0.8
MARGIN_SHARE = 0.1
NOTE = __doc__.split("Choices the pre-registration", 1)[1].split("95% CIs", 1)[0].strip()


def fit_probe(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X, y)


def paired_auroc_ci(neg, pos, n_boot=2000, seed=0):
    """AUROC of pos (label 1) vs neg (label 0), where neg[i] and pos[i] come from the same pair; bootstrap over pairs."""
    from sklearn.metrics import roc_auc_score
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    est = roc_auc_score(y, np.r_[neg, pos])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(neg), (n_boot, len(neg)))
    boots = [roc_auc_score(y, np.r_[neg[i], pos[i]]) for i in idx]
    return float(est), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def h3_parts(Hv, Ha, meta, n_boot=2000, seed=0):
    """All three parts of H3 for one model and one vector. Hv, Ha: (pairs, hidden); meta: intent and split per pair."""
    Hv, Ha = np.asarray(Hv, np.float32), np.asarray(Ha, np.float32)
    split = meta["split"].to_numpy()
    harm = (meta["intent"] == "harmful").to_numpy()
    tr, te = split == "train", split == "test"
    teb, teh = te & ~harm, te & harm

    # (a) style part on TEST benign pairs
    tmeta = meta[tr].assign(role="vanilla")
    r_hat = lib.fit_direction_train(Hv[tr], tmeta)
    dp = lib.project(Ha, r_hat) - lib.project(Hv, r_hat)
    style = lib.bootstrap_ci(dp[teb], np.mean, n=n_boot, seed=seed)
    part_a = dict(style=style, n_benign_test=int(teb.sum()), holds=bool(style[1] > 0 or style[2] < 0))

    # (b) G2 form probe
    g2 = fit_probe(np.r_[Hv[tr], Ha[tr]], np.r_[np.zeros(tr.sum()), np.ones(tr.sum())])
    auroc = paired_auroc_ci(g2.predict_proba(Hv[teb])[:, 1], g2.predict_proba(Ha[teb])[:, 1], n_boot, seed)
    part_b = dict(auroc=auroc, threshold=AUROC_MIN, holds=bool(auroc[0] >= AUROC_MIN),
                  ci_lower_bound_also_above=bool(auroc[1] >= AUROC_MIN))

    # (c) harmfulness probe fooled by style?
    hp = fit_probe(Hv[tr], harm[tr].astype(int))
    p_v, p_a = hp.predict_proba(Hv)[:, 1], hp.predict_proba(Ha)[:, 1]
    gap = float(p_v[teh].mean() - p_v[teb].mean())
    m = MARGIN_SHARE * gap
    diff = lib.bootstrap_ci(p_a[teb] - p_v[teb], np.mean, n=n_boot, seed=seed)
    part_c = dict(score_difference=diff, vanilla_gap=gap, margin=m, holds=bool(-m < diff[1] and diff[2] < m),
                  mean_score=dict(vanilla_harmful=float(p_v[teh].mean()), vanilla_benign=float(p_v[teb].mean()),
                                  adversarial_benign=float(p_a[teb].mean()), adversarial_harmful=float(p_a[teh].mean())))

    supported = part_a["holds"] and part_b["holds"] and part_c["holds"]
    failed = [k for k, p in (("a", part_a), ("b", part_b), ("c", part_c)) if not p["holds"]]
    return dict(a=part_a, b=part_b, c=part_c, verdict="supported" if supported else "not supported",
                failed_parts=failed, n_train_pairs=int(tr.sum()), n_test_pairs=int(te.sum()))
