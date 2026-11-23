"""X6 follow-up on the probe CSV (no archives, no GPU, no text): exact intervals for the
zero/near-zero flip rates, a boundary-aware flip rate, and the size of the probe-score
shift, split by whether the generated tokens had matched. Reads the CSV the X6 job wrote.

    df = x6_followup.load()                       # downloads the CSV from the volume
    x6_followup.report(df, layer=16, config_label="C3")
"""
import subprocess

import numpy as np
import pandas as pd

CSV_REMOTE = "results/x6_probe_flips_llama31-8b.csv"
CSV_LOCAL = "/content/x6_probe_flips.csv"
NEAR_LO, NEAR_HI = 0.10, 0.90      # "near the boundary": reference probability in this range


def load(local=CSV_LOCAL):
    subprocess.run(["modal", "volume", "get", "a5-outputs", CSV_REMOTE, local, "--force"],
                   check=True)
    return pd.read_csv(local)


def exact_ci(k, n, alpha=0.05):
    """Clopper-Pearson interval for k events in n trials."""
    from scipy.stats import beta
    if n == 0:
        return (None, None)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def _row(d):
    n = len(d)
    k = int(d["probe_flip"].sum())
    lo, hi = exact_ci(k, n)
    per_prompt = d.groupby("prompt_id")["probe_flip"].any()
    kp, npr = int(per_prompt.sum()), int(len(per_prompt))
    plo, phi = exact_ci(kp, npr)
    dp = (d["cfg_prob"] - d["ref_prob"]).abs()
    near = d[(d["ref_prob"] >= NEAR_LO) & (d["ref_prob"] <= NEAR_HI)]
    kn = int(near["probe_flip"].sum())
    nlo, nhi = exact_ci(kn, len(near))
    return {"rows": n, "prompts": npr, "flips": k,
            "rate": (k / n if n else None), "row_ci": (lo, hi),
            "prompts_with_flip": kp, "prompt_ci": (plo, phi),
            "near_boundary_rows": int(len(near)), "near_boundary_flips": kn,
            "near_boundary_ci": (nlo, nhi),
            "median_abs_dprob": (float(dp.median()) if n else None),
            "p95_abs_dprob": (float(dp.quantile(0.95)) if n else None),
            "share_dprob_gt_0.05": (float((dp > 0.05).mean()) if n else None)}


def report(df, layer, config_label=None, held_out_only=True):
    d = df.copy()
    if config_label is not None:
        d = d[d["config_label"] == config_label]
    d = d[d["layer"] == layer]
    if held_out_only:
        d = d[d["held_out"]]
    out = {}
    for name, g in (("same_token", d[d["same_token"]]), ("diverged", d[~d["same_token"]])):
        out[name] = _row(g)
    for name, r in out.items():
        f = lambda v: "n/a" if v is None else f"{v:.4f}"
        print(f"\n{name}  ({'held-out' if held_out_only else 'all'} prompts, layer {layer})")
        print(f"  rows {r['rows']}  prompts {r['prompts']}  flips {r['flips']}  "
              f"rate {f(r['rate'])}  exact 95% {tuple(map(f, r['row_ci']))}")
        print(f"  prompts with any flip {r['prompts_with_flip']}/{r['prompts']}  "
              f"exact 95% {tuple(map(f, r['prompt_ci']))}")
        print(f"  near-boundary rows (ref prob {NEAR_LO}-{NEAR_HI}): {r['near_boundary_rows']}  "
              f"flips {r['near_boundary_flips']}  exact 95% {tuple(map(f, r['near_boundary_ci']))}")
        print(f"  |cfg prob - ref prob|: median {f(r['median_abs_dprob'])}  "
              f"p95 {f(r['p95_abs_dprob'])}  share > 0.05: {f(r['share_dprob_gt_0.05'])}")
    by_pos = (d.groupby(["g4_position", "same_token"]).size().unstack(fill_value=0)
              .rename(columns={True: "same_token_rows", False: "diverged_rows"}))
    print("\nrows reaching each G4 position (explains how few diverged rows there are):")
    print(by_pos.to_string())
    return out
