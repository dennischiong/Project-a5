# Cell M1 - A1 on Mistral: how far each jailbreak tactic moves the model's last-token state (layer 31, prompt[-1]).
# Same method as the Llama A1: cosine distance and relative L2 between the vanilla and adversarial prompt of every pair,
# median per (source, tactic) with a 95% percentile bootstrap (2,000 resamples, seed 0).
# Needs cell M0 to have run in this Colab session. Reads only what M0 loaded. Writes 3 small files to RESULTS_DIR.
# EXECUTION TIME: about 1-2 min.
# WHAT IS DIFFERENT FROM LLAMA (also written into the output meta file):
#   - The Mistral harvest has no paraphrase/length control pairs and no A9 precision floor, so there are no baselines to compare against.
#   - Only layer 31 at the last prompt token was saved (prenorm_last = the residual stream going into the final norm).
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    pre, pairs, row_of, RESULTS_DIR, MODEL_ALIAS
except NameError:
    raise SystemExit("Run cell M0 first (it loads the harvest and the pairs).")

t0 = time.time()
use = pairs[pairs["vkey"].isin(row_of) & pairs["akey"].isin(row_of)].copy()
use["van_row"] = use["vkey"].map(row_of).astype(int)
use["adv_row"] = use["akey"].map(row_of).astype(int)
print(f"pairs measured: {len(use):,}   tactic groups: {use.groupby(['source', 'tactic']).ngroups}")

cos_d, rel_l2 = displacement(pre, use["van_row"].to_numpy(), use["adv_row"].to_numpy())
bad = int((~np.isfinite(rel_l2)).sum())
print(f"pairs with a non-finite or zero-norm vector (left out of medians): {bad}")

per_pair = use[["pair_id", "source", "tactic", "intent", "split", "holdout_group"]].copy()
per_pair["extraction_point"] = "last_prefill"
per_pair["layer"] = LAST_LAYER
per_pair["cosine_distance"] = cos_d
per_pair["relative_l2"] = rel_l2

rows = []
for (src, tac), part in per_pair.groupby(["source", "tactic"], dropna=False):
    c, l = boot_median(part["cosine_distance"]), boot_median(part["relative_l2"])
    rows.append({"source": src, "tactic": tac, "n_pairs": l["n"],
                 "cos_dist_median": c["median"], "cos_dist_lo": c["lo"], "cos_dist_hi": c["hi"],
                 "rel_l2_median": l["median"], "rel_l2_lo": l["lo"], "rel_l2_hi": l["hi"],
                 "snr_l2_vs_floor": None})          # column kept so the table lines up with the Llama one
summary = pd.DataFrame(rows).sort_values("rel_l2_median", ascending=False)
degenerate = summary["rel_l2_median"].fillna(0) < 1e-9
real = summary[~degenerate]
print(f"tactic groups: {len(summary)}   degenerate (identical text, distance ~0): {int(degenerate.sum())}")
print("\nlargest displacement (relative L2, 95% CI):")
for r in real.head(10).itertuples():
    print(f"  {str(r.source)[:14]:<14} {str(r.tactic)[:28]:<28} n={r.n_pairs:>5}  {r.rel_l2_median:.4f} [{r.rel_l2_lo:.4f}, {r.rel_l2_hi:.4f}]")
print("smallest (non-degenerate):")
for r in real.tail(10).itertuples():
    print(f"  {str(r.source)[:14]:<14} {str(r.tactic)[:28]:<28} n={r.n_pairs:>5}  {r.rel_l2_median:.4f} [{r.rel_l2_lo:.4f}, {r.rel_l2_hi:.4f}]")

# per source, one line each (SORRY-Bench mutations and WildJailbreak tactics are different kinds of change)
print("\nper source (all pairs):")
by_source = {}
for src, part in per_pair.groupby("source", dropna=False):
    l = boot_median(part["relative_l2"])
    by_source[str(src)] = l
    print(f"  {str(src):<16} n={l['n']:>6}  rel L2 {l['median']:.4f} [{l['lo']:.4f}, {l['hi']:.4f}]")

# ---- optional: does Mistral rank the tactics the same way Llama did? (Spearman on the tactic medians)
rank_vs_llama = None
llama_csv = RESULTS_DIR / "a1_summary_by_tactic_llama31-8b.csv"
if llama_csv.is_file():
    from scipy.stats import spearmanr
    ll = pd.read_csv(llama_csv)
    m = real.merge(ll[["source", "tactic", "rel_l2_median"]].rename(columns={"rel_l2_median": "llama_rel_l2_median"}),
                   on=["source", "tactic"], how="inner")
    m = m[m["llama_rel_l2_median"] > 1e-9]
    if len(m) >= 5:
        rho, p = spearmanr(m["rel_l2_median"], m["llama_rel_l2_median"])
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(m), len(m))
            if m["rel_l2_median"].iloc[i].nunique() > 1 and m["llama_rel_l2_median"].iloc[i].nunique() > 1:
                boots.append(spearmanr(m["rel_l2_median"].iloc[i], m["llama_rel_l2_median"].iloc[i])[0])
        rank_vs_llama = {"n_tactic_groups_in_both": int(len(m)), "spearman_rho": float(rho), "p_value": float(p),
                         "rho_ci_lo": float(np.percentile(boots, 2.5)) if boots else None,
                         "rho_ci_hi": float(np.percentile(boots, 97.5)) if boots else None}
        print(f"\nsame ranking as Llama? Spearman rho = {rho:.3f} over {len(m)} tactic groups present in both "
              f"(CI {rank_vs_llama['rho_ci_lo']:.3f} to {rank_vs_llama['rho_ci_hi']:.3f})")
        m.to_csv(RESULTS_DIR / f"a1_tactic_rank_vs_llama_{MODEL_ALIAS}.csv", index=False)
    else:
        print(f"\nfewer than 5 tactic groups shared with the Llama table ({len(m)}): no rank comparison.")
else:
    print(f"\n(Llama tactic table not found at {llama_csv}: skipping the rank comparison)")

# ---- optional: how repeatable is a stored vector? full harvest vs the 4,000-prompt subset harvest (different GPUs)
repro = None
if SUB_DIR is not None and (SUB_DIR / "prenorm_last.npy").is_file():
    try:
        sub_prompts = _read_table(SUB_DIR / "prompts.parquet")
        sub_pre = np.load(SUB_DIR / "prenorm_last.npy")
        skey = sub_prompts[pick(sub_prompts, ["prompt_key", "key"], "prompt_key", regex=r"key")].astype(str).tolist()
        shared = [(i, row_of[k]) for i, k in enumerate(skey) if k in row_of]
        if len(shared) >= 100 and sub_pre.shape[1] == pre.shape[1]:
            si = np.array([a for a, _ in shared]); fi = np.array([b for _, b in shared])
            a, b = sub_pre[si].astype("float32"), pre[fi].astype("float32")
            nb = np.linalg.norm(b, axis=1)
            rel = np.linalg.norm(a - b, axis=1) / np.where(nb > 0, nb, np.nan)
            cosd = 1 - np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * nb)
            repro = {"n_shared_prompts": int(len(shared)), "rel_l2_median": float(np.nanmedian(rel)),
                     "rel_l2_p95": float(np.nanpercentile(rel, 95)), "cos_dist_p95": float(np.nanpercentile(cosd, 95)),
                     "note": "same prompt, same model and dtype, harvested twice (different GPU mix). A rough size of run-to-run "
                             "noise; NOT the A9 precision floor, which was never measured for Mistral."}
            print(f"\nrepeatability (full vs sub4000 harvest, {len(shared):,} shared prompts): relative L2 median "
                  f"{repro['rel_l2_median']:.2e}, 95th percentile {repro['rel_l2_p95']:.2e}")
            ok_tactics = int((real["rel_l2_lo"] > repro["rel_l2_p95"]).sum())
            print(f"tactic groups whose lower CI is above that 95th percentile: {ok_tactics} of {len(real)}")
            repro["tactic_groups_clearing_it"] = ok_tactics
    except Exception as e:          # optional block: never stop A1 because of it
        print(f"\n(repeatability check skipped: {type(e).__name__}: {e})")

# ---- write
tag = MODEL_ALIAS
per_pair.to_csv(RESULTS_DIR / f"a1_per_pair_{tag}.csv", index=False)
summary.to_csv(RESULTS_DIR / f"a1_summary_by_tactic_{tag}.csv", index=False)
write_atomic(RESULTS_DIR / f"a1_{tag}_meta.json", json.dumps({
    "created_at": datetime.now(timezone.utc).isoformat(),
    "analysis": "A1 displacement per tactic", "owner": "WS2 (Kayson)", "model_id": MODEL_ID, "model_alias": tag,
    "harvest": FULL_NAME, "harvest_meta": {k: meta.get(k) for k in ("config_name", "gpu", "n_prompts", "position", "sweep_layers", "pairs_table_sha256")},
    "position": "last_prefill (prompt[-1])", "layer": LAST_LAYER, "stored_as": "prenorm_last (residual stream, float32)",
    "pairs_measured": int(len(use)), "pairs_not_finite": bad, "tactic_groups": int(len(summary)),
    "degenerate_tactic_groups": int(degenerate.sum()), "by_source": by_source,
    "baselines": "NOT AVAILABLE for Mistral: the harvest has no paraphrase / length-filler control pairs",
    "a9_precision_floor": "NOT MEASURED for Mistral",
    "rank_vs_llama": rank_vs_llama, "repeatability_full_vs_sub4000": repro,
    "statistics": {"estimator": "median", "interval": "percentile bootstrap", "n_boot": N_BOOT, "alpha": ALPHA, "seed": SEED},
    "runtime_seconds": round(time.time() - t0, 1)}, indent=2, default=str))
print(f"\nwrote a1_per_pair_{tag}.csv, a1_summary_by_tactic_{tag}.csv, a1_{tag}_meta.json to {RESULTS_DIR}   ({time.time() - t0:.0f}s)")
