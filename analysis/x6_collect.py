import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

VOLUME = "a5-outputs"
DEST = Path("/content/x6")
FILES = ["results/x6_probe_flips_llama31-8b.csv",
         "results/x6_decode_flips_llama31-8b.csv",
         "results/x6_summary_llama31-8b.json",
         "qa/x6_calibration.json"]
ALPHA = 0.05
SHIFT = 0.05

DEST.mkdir(parents=True, exist_ok=True)
for f in FILES:
    subprocess.run(["modal", "volume", "get", VOLUME, f, str(DEST / Path(f).name), "--force"],
                   check=True)

summary = json.loads((DEST / "x6_summary_llama31-8b.json").read_text())
calib = json.loads((DEST / "x6_calibration.json").read_text())
probe = pd.read_csv(DEST / "x6_probe_flips_llama31-8b.csv")
token = pd.read_csv(DEST / "x6_decode_flips_llama31-8b.csv")


def exact_ci(k, n):
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(ALPHA / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - ALPHA / 2, k + 1, n - k)
    return (float(lo), float(hi))


def describe(d):
    n, k = len(d), int(d["probe_flip"].sum())
    lo, hi = exact_ci(k, n)
    per_prompt = d.groupby("prompt_id")["probe_flip"].any()
    plo, phi = exact_ci(int(per_prompt.sum()), len(per_prompt))
    dp = (d["cfg_prob"] - d["ref_prob"]).abs()
    return {"rows": n, "prompts": int(d["prompt_id"].nunique()), "flips": k,
            "rate": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi,
            "prompts_with_flip": int(per_prompt.sum()), "prompt_ci_lo": plo, "prompt_ci_hi": phi,
            "median_abs_dprob": float(dp.median()) if n else np.nan,
            "p95_abs_dprob": float(dp.quantile(0.95)) if n else np.nan,
            "share_dprob_gt_0.05": float((dp > SHIFT).mean()) if n else np.nan}


never = (token.groupby(["config_label", "prompt_id"])["same_token"].all()
         .rename("prompt_never_diverged").reset_index())
probe = probe.merge(never, on=["config_label", "prompt_id"], how="left")
assert probe["prompt_never_diverged"].notna().all(), "probe rows without a token-level record"

PARTITIONS = {"row-level (plan)": ("same_token", True, False),
              "prompt-level (sensitivity)": ("prompt_never_diverged", True, False)}
records = []
for label, cfg in summary["configs"].items():
    layer = cfg["probe"]["headline_layer"]
    held = probe[(probe["config_label"] == label) & (probe["layer"] == layer) & probe["held_out"]]
    assert not held["role"].eq("train").any(), "a training prompt is in the held-out set"
    for part, (col, same_val, div_val) in PARTITIONS.items():
        for group, val in (("same_token", same_val), ("diverged", div_val)):
            records.append({"config": label, "layer": layer, "partition": part, "group": group,
                            **describe(held[held[col] == val])})
    head = cfg["probe"]["headline"]
    for group in ("same_token", "diverged"):
        mine = next(r for r in records if r["config"] == label
                    and r["partition"] == "row-level (plan)" and r["group"] == group)
        assert mine["rows"] == head[group]["n_rows"], f"{label}/{group}: row count differs from the job summary"
        assert abs(mine["rate"] - head[group]["rate"]) < 1e-9, f"{label}/{group}: rate differs from the job summary"
    for group in ("same_token", "diverged"):
        t = cfg[group]
        records.append({"config": label, "layer": "token-level", "partition": "row-level (plan)",
                        "group": group + " (token flip)", "rows": t["n_rows"], "prompts": t["n_prompts"],
                        "flips": round(t["rate"] * t["n_rows"]) if t["rate"] is not None else 0,
                        "rate": t["rate"], "ci_lo": t["lo"], "ci_hi": t["hi"]})

for label, why in summary.get("missing_configs", {}).items():
    records.append({"config": label, "partition": "not run", "group": str(why)})
if "C2" not in summary["configs"] and "C2" not in summary.get("missing_configs", {}):
    records.append({"config": "C2", "partition": "not run",
                    "group": "not run"})

table = pd.DataFrame(records)
table.to_csv(DEST / "x6_table.csv", index=False)


def cell(r):
    if r is None or r["rows"] == 0:
        return "no rows"
    return (f"{int(r['flips'])}/{int(r['rows'])} ({100 * r['rate']:.2f}%; "
            f"95% CI {100 * r['ci_lo']:.2f}-{100 * r['ci_hi']:.2f}%)")


paper = []
for label in summary["configs"]:
    sub = table[(table["config"] == label) & (table["partition"] == "row-level (plan)")
                & (table["layer"] != "token-level")]
    s = sub[sub["group"] == "same_token"].iloc[0]
    d = sub[sub["group"] == "diverged"].iloc[0]
    paper.append({"Config": label, "Probe layer": s["layer"],
                  "Same-token rows: flip rate": cell(s), "Diverged rows: flip rate": cell(d),
                  "Median |dprob| same / diverged": f"{s['median_abs_dprob']:.4f} / {d['median_abs_dprob']:.4f}"})
paper = pd.DataFrame(paper)

print("calibration:", calib["convention"], "from", calib["convention_source"], "| BOS check:",
      calib["bos_check"].get("status"))
print("\nX6 (held-out prompts, probe-decision flips, validation-selected layer)\n")
print(paper.to_string(index=False))
print("\nfull table ->", DEST / "x6_table.csv")
print("\nsensitivity, prompt-level partition:")
print(table[table["partition"] == "prompt-level (sensitivity)"]
      [["config", "group", "rows", "prompts", "flips", "rate", "ci_lo", "ci_hi"]].to_string(index=False))
for label, cfg in summary["configs"].items():
    layer = cfg["probe"]["headline_layer"]
    held = probe[(probe["config_label"] == label) & (probe["layer"] == layer) & probe["held_out"]]
    print(f"\n{label}: held-out rows reaching each G4 position")
    print(held.groupby(["g4_position", "same_token"]).size().unstack(fill_value=0).to_string())
