# =====================================================================
# A1 + A6 (LAYER SWEEP) FROM THE QWEN2.5-7B MULTILAYER CACHE
# Paste into ONE Colab cell. CPU only - no GPU, no model.
# EXECUTION TIME: 6-10 minutes. Most of it is reading the 929 MB npz from
# Drive (2-5 min). If it sits past 8 minutes on "loading feats", Drive is
# slow - tell me rather than waiting.
# =====================================================================
#
# WHAT THIS COMPUTES, AND WHY
# ---------------------------
# A1 (DOCX section 9): for each vanilla/adversarial pair, the displacement
# delta = h_adv - h_van of the residual stream at the last prompt token. The
# size of that displacement is reported per tactic as cosine distance and as
# relative L2 (||delta|| / ||h_van||), against three baselines the spec names:
#
#   - paraphrase      meaning-preserving rewrites, from Controls
#   - length filler   length-matched padding, from Controls
#   - numerical floor the A9 measurement you took last night
#
# A tactic whose displacement is not clearly above all three has not been
# shown to move the representation at all. That comparison is figure F1,
# which section 12 assigns to you.
#
# A6: the same measurement repeated across the 7 cached layers, so you can
# say where in the network the displacement appears.
#
# WHAT IT HONESTLY IS NOT
# -----------------------
# These activations are Qwen2.5-7B in 4-bit NF4, harvested with plain HF
# hooks rather than Undercurrent, on a T4. All three facts are recorded in
# the output metadata and must appear in the paper. The position split half
# of A6 needs last_instruction, which is not in this cache.
#
# OUTPUT: four files in harvest/results/ and a printed summary table.

import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 72)
print("A1 DISPLACEMENT + A6 LAYER SWEEP")
print("=" * 72)

# -------------------------------------------------------------- paths
try:
    from google.colab import drive

    drive.mount("/content/gdrive")
except Exception as error:
    print("mount reported:", error)

CANDIDATE_ROOTS = [
    Path("/content/gdrive/MyDrive/MYSSP/a5-artifacts"),
    Path("/content/drive/MyDrive/MYSSP/a5-artifacts"),
    Path("/content/gdrive/MyDrive/a5-artifacts"),
    Path("/content/drive/MyDrive/a5-artifacts"),
]
ROOT = next((p for p in CANDIDATE_ROOTS if p.is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("Project root not visible. Mount Drive first.")

HARVEST = ROOT / "harvest"
RESULTS = HARVEST / "results"
QA = HARVEST / "qa"
RESULTS.mkdir(parents=True, exist_ok=True)

DENNIS = ROOT / "Dennis"
NPZ = (DENNIS / "Activations" / "cache_layer16"
       / "qwen25-7b__hf-nf4-c6__e7f5ea94__last_prefill_multilayer.npz")
PAIR_MAP = (DENNIS / "Activations" / "Full harvest Qwen 2.5-7B activations"
            / "pair_maps" / "pair_map__d240ec65.parquet")
PAIRS_CSV = DENNIS / "data table" / "pairs_table_v1_final-2.csv"

for path in (NPZ, PAIR_MAP, PAIRS_CSV):
    if not path.is_file():
        raise FileNotFoundError(path)
print("root:", ROOT)
print(f"cache: {NPZ.name}  ({NPZ.stat().st_size:,} bytes)")

N_BOOT = 2000
ALPHA = 0.05
SEED = 0

# --------------------------------------------------------- the A9 floor
print("\n--- A9 FLOOR ---")
floor_cos = floor_l2 = None
floor_path = QA / "a9" / "a9_summary.csv"
if floor_path.is_file():
    a9 = pd.read_csv(floor_path)
    row = a9[a9["comparison"] == "bf16_vs_fp32"]
    if len(row):
        floor_cos = float(row.iloc[0]["cos_dist_p95"])
        floor_l2 = float(row.iloc[0]["rel_l2_p95"])
        print(f"  precision floor (p95): cosine {floor_cos:.3e}, "
              f"relative L2 {floor_l2:.3e}")

quant_cos = quant_l2 = None
smoke_path = QA / "undercurrent_nf4_smoke.json"
if smoke_path.is_file():
    smoke = json.loads(smoke_path.read_text())
    gap = smoke.get("nf4_vs_bf16") or {}
    quant_cos = gap.get("cos_dist_p95")
    quant_l2 = gap.get("rel_l2_p95")
    if quant_l2:
        print(f"  quantization floor (p95): cosine {quant_cos:.3e}, "
              f"relative L2 {quant_l2:.3e}")
        print("  (upper bound - these activations are NF4, and a paired "
              "difference partly cancels systematic quantization error)")
if floor_l2 is None:
    print("  WARNING: no A9 summary found; SNR columns will be blank.")

# ------------------------------------------------------- load the cache
print("\n--- LOADING THE ACTIVATION CACHE ---")
print("  reading npz headers ...")
with zipfile.ZipFile(NPZ) as zf:
    for name in zf.namelist():
        with zf.open(name) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _f, dtype = np.lib.format.read_array_header_1_0(stream)
            else:
                shape, _f, dtype = np.lib.format.read_array_header_2_0(stream)
        print(f"    {name:<22} {str(shape):<22} {dtype}")

print("  loading ids and layers ...")
with np.load(NPZ) as handle:
    ids = handle["ids"]
    layers = handle["layers"]
    print(f"    ids: {len(ids):,}  layers: {list(layers)}")
    print("  loading feats (929 MB - the slow part) ...")
    feats = handle["feats"]
    print(f"    feats: {feats.shape} {feats.dtype}")

n_prompts, n_layers, hidden = feats.shape
if len(ids) != n_prompts:
    raise RuntimeError(f"ids {len(ids)} != feats rows {n_prompts}")
if n_layers != len(layers):
    raise RuntimeError(f"layers {len(layers)} != feats axis 1 {n_layers}")

row_of = {str(pid): i for i, pid in enumerate(ids)}
if len(row_of) != n_prompts:
    raise RuntimeError("duplicate prompt ids in the cache")

# ------------------------------------------------------- build the pairs
print("\n--- PAIRING ---")
pair_map = pd.read_parquet(PAIR_MAP)
pair_map["pair_id"] = pair_map["pair_id"].astype(str)
pair_map["prompt_id"] = pair_map["prompt_id"].astype(str)
pair_map["role"] = (
    pair_map["role"].astype(str).str.lower().replace({"adv": "adversarial"})
)
print(f"  pair map: {len(pair_map):,} rows, roles "
      f"{pair_map['role'].value_counts().to_dict()}")

wide = pair_map.pivot(index="pair_id", columns="role", values="prompt_id")
if not {"vanilla", "adversarial"} <= set(wide.columns):
    raise RuntimeError(f"pair map roles are {list(wide.columns)}")
wide = wide.dropna(subset=["vanilla", "adversarial"]).reset_index()

meta_cols = ["pair_id", "source", "tactic", "split", "holdout_group",
             "category", "intent", "may_not_be_understood"]
pairs = pd.read_csv(PAIRS_CSV, usecols=lambda c: c in meta_cols,
                    dtype={"pair_id": str})
joined = wide.merge(pairs, on="pair_id", how="inner", validate="one_to_one")
print(f"  pairs with both roles and metadata: {len(joined):,}")

joined["van_row"] = joined["vanilla"].map(row_of)
joined["adv_row"] = joined["adversarial"].map(row_of)
before = len(joined)
joined = joined.dropna(subset=["van_row", "adv_row"]).copy()
joined["van_row"] = joined["van_row"].astype(int)
joined["adv_row"] = joined["adv_row"].astype(int)
dropped = before - len(joined)
print(f"  pairs whose both halves are in the cache: {len(joined):,}"
      f"{f'  ({dropped} dropped - not in cache)' if dropped else ''}")
if len(joined) < 1000:
    raise RuntimeError("Too few usable pairs; the ids do not line up.")

print("\n  pairs per source:")
for source, n in joined["source"].value_counts().items():
    print(f"    {source:<18} {n:>6,}")
print(f"  distinct tactics: {joined['tactic'].nunique()}")

# ------------------------------------------------------------- metrics
def displacement(layer_index, van_rows, adv_rows, block=4096):
    """Cosine distance and relative L2 between paired rows at one layer.

    Done in blocks so peak memory stays modest, and in float32 because the
    cache is float16 and norms of 3584-dim float16 vectors lose precision.
    """
    cos_d = np.empty(len(van_rows), dtype="float64")
    rel_l2 = np.empty(len(van_rows), dtype="float64")

    for start in range(0, len(van_rows), block):
        stop = min(start + block, len(van_rows))
        v = feats[van_rows[start:stop], layer_index, :].astype("float32")
        a = feats[adv_rows[start:stop], layer_index, :].astype("float32")

        nv = np.linalg.norm(v, axis=1)
        na = np.linalg.norm(a, axis=1)
        dot = np.einsum("ij,ij->i", v, a)
        denom = nv * na
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(denom > 0, dot / denom, np.nan)
            cos_d[start:stop] = 1.0 - cos
            rel_l2[start:stop] = np.where(
                nv > 0, np.linalg.norm(a - v, axis=1) / nv, np.nan
            )
    return cos_d, rel_l2


def bootstrap_median(values, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "point": np.nan, "lo": np.nan, "hi": np.nan}
    if arr.size == 1:
        return {"n": 1, "point": float(arr[0]), "lo": float(arr[0]),
                "hi": float(arr[0])}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    draws = np.median(arr[idx], axis=1)
    return {
        "n": int(arr.size),
        "point": float(np.median(arr)),
        "lo": float(np.percentile(draws, 100 * alpha / 2)),
        "hi": float(np.percentile(draws, 100 * (1 - alpha / 2))),
    }


van_rows = joined["van_row"].to_numpy()
adv_rows = joined["adv_row"].to_numpy()

# ------------------------------------------------------- A6 layer sweep
print("\n--- A6: LAYER SWEEP ---")
print("  layer   median cos dist              median relative L2")
sweep_rows = []
per_layer_cos = {}
per_layer_l2 = {}
for i, layer in enumerate(layers):
    cos_d, rel_l2 = displacement(i, van_rows, adv_rows)
    per_layer_cos[int(layer)] = cos_d
    per_layer_l2[int(layer)] = rel_l2

    cos_ci = bootstrap_median(cos_d)
    l2_ci = bootstrap_median(rel_l2)
    sweep_rows.append({
        "layer": int(layer),
        "n_pairs": cos_ci["n"],
        "cos_dist_median": cos_ci["point"],
        "cos_dist_lo": cos_ci["lo"],
        "cos_dist_hi": cos_ci["hi"],
        "rel_l2_median": l2_ci["point"],
        "rel_l2_lo": l2_ci["lo"],
        "rel_l2_hi": l2_ci["hi"],
        "snr_cos_vs_precision_floor": (cos_ci["point"] / floor_cos) if floor_cos else None,
        "snr_l2_vs_precision_floor": (l2_ci["point"] / floor_l2) if floor_l2 else None,
        "snr_l2_vs_quantization_floor": (l2_ci["point"] / quant_l2) if quant_l2 else None,
    })
    print(f"  {int(layer):>5}   {cos_ci['point']:.4f} "
          f"[{cos_ci['lo']:.4f}, {cos_ci['hi']:.4f}]      "
          f"{l2_ci['point']:.4f} [{l2_ci['lo']:.4f}, {l2_ci['hi']:.4f}]")

sweep = pd.DataFrame(sweep_rows)
sweep.to_csv(RESULTS / "a6_layer_sweep_qwen25-7b.csv", index=False)

headline_layer = int(max(layers))
print(f"\n  headline layer for A1: {headline_layer} "
      "(deepest cached layer; Dennis's G1 probe selected layer 20)")

# -------------------------------------------------------------- A1
print("\n--- A1: DISPLACEMENT PER TACTIC ---")
cos_d = per_layer_cos[headline_layer]
rel_l2 = per_layer_l2[headline_layer]

per_pair = joined[["pair_id", "source", "tactic", "split", "holdout_group"]].copy()
per_pair["layer"] = headline_layer
per_pair["cosine_distance"] = cos_d
per_pair["relative_l2"] = rel_l2
per_pair.to_csv(RESULTS / "a1_per_pair_qwen25-7b.csv", index=False)

summary_rows = []
for (source, tactic), part in per_pair.groupby(["source", "tactic"], dropna=False):
    cos_ci = bootstrap_median(part["cosine_distance"])
    l2_ci = bootstrap_median(part["relative_l2"])
    summary_rows.append({
        "source": source,
        "tactic": tactic,
        "n_pairs": cos_ci["n"],
        "cos_dist_median": cos_ci["point"],
        "cos_dist_lo": cos_ci["lo"],
        "cos_dist_hi": cos_ci["hi"],
        "rel_l2_median": l2_ci["point"],
        "rel_l2_lo": l2_ci["lo"],
        "rel_l2_hi": l2_ci["hi"],
        "snr_l2_vs_precision_floor": (l2_ci["point"] / floor_l2) if floor_l2 else None,
        "snr_l2_vs_quantization_floor": (l2_ci["point"] / quant_l2) if quant_l2 else None,
    })

summary = pd.DataFrame(summary_rows).sort_values(
    "rel_l2_median", ascending=False
).reset_index(drop=True)
summary.to_csv(RESULTS / "a1_summary_by_tactic_qwen25-7b.csv", index=False)

# ------------------------------------------------------------ baselines
print("\n--- BASELINES (the comparison figure F1 needs) ---")
controls = per_pair[per_pair["source"] == "Controls"]
baselines = {}
if len(controls):
    print("  Controls tactics available:",
          sorted(controls["tactic"].dropna().unique())[:12])
    for label, pattern in [("paraphrase", "paraphrase"), ("length_filler", "length")]:
        sel = controls[controls["tactic"].astype(str).str.contains(
            pattern, case=False, na=False)]
        if len(sel):
            ci = bootstrap_median(sel["relative_l2"])
            cos_ci = bootstrap_median(sel["cosine_distance"])
            baselines[label] = {
                "n_pairs": ci["n"],
                "rel_l2_median": ci["point"],
                "rel_l2_lo": ci["lo"],
                "rel_l2_hi": ci["hi"],
                "cos_dist_median": cos_ci["point"],
            }
            print(f"  {label:<14} n={ci['n']:>5}  relative L2 "
                  f"{ci['point']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]")
        else:
            print(f"  {label:<14} no matching Controls tactic")
else:
    print("  no Controls pairs in the cache")

if floor_l2:
    baselines["numerical_precision_floor_p95"] = {"rel_l2": floor_l2}
if quant_l2:
    baselines["numerical_quantization_floor_p95"] = {"rel_l2": quant_l2}

print("\n--- A1: LARGEST AND SMALLEST DISPLACEMENTS ---")
print("  (relative L2 at layer", headline_layer, "with 95% bootstrap CI)")
print("\n  largest:")
for row in summary.head(10).itertuples():
    snr = f"{row.snr_l2_vs_precision_floor:6.1f}x" if row.snr_l2_vs_precision_floor else "    -"
    print(f"    {str(row.source)[:14]:<14} {str(row.tactic)[:26]:<26} "
          f"n={row.n_pairs:>5}  {row.rel_l2_median:.4f} "
          f"[{row.rel_l2_lo:.4f}, {row.rel_l2_hi:.4f}]  floor x{snr}")
print("\n  smallest:")
for row in summary.tail(10).itertuples():
    snr = f"{row.snr_l2_vs_precision_floor:6.1f}x" if row.snr_l2_vs_precision_floor else "    -"
    print(f"    {str(row.source)[:14]:<14} {str(row.tactic)[:26]:<26} "
          f"n={row.n_pairs:>5}  {row.rel_l2_median:.4f} "
          f"[{row.rel_l2_lo:.4f}, {row.rel_l2_hi:.4f}]  floor x{snr}")

# how many tactics clear the floors
if floor_l2:
    clears_precision = int((summary["rel_l2_lo"] > floor_l2).sum())
    print(f"\n  tactics whose CI lower bound clears the precision floor "
          f"({floor_l2:.4f}): {clears_precision} of {len(summary)}")
if quant_l2:
    clears_quant = int((summary["rel_l2_lo"] > quant_l2).sum())
    print(f"  tactics whose CI lower bound clears the quantization bound "
          f"({quant_l2:.4f}): {clears_quant} of {len(summary)}")

# -------------------------------------------------------------- metadata
meta = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "analyses": ["A1 displacement per tactic", "A6 layer sweep (last_prefill only)"],
    "owner": "WS2 (Kayson)",
    "source_activations": {
        "file": NPZ.name,
        "harvested_by": "WS1 (Dennis)",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "config": "hf-nf4-c6",
        "spec_hash": "e7f5ea94",
        "quantization": "nf4",
        "inference_dtype": "fp16",
        "gpu": "Tesla T4",
        "harness": "plain HF hooks, NOT Undercurrent",
        "extraction_point": "last_prefill",
        "layers_cached": [int(x) for x in layers],
        "n_prompts_cached": int(n_prompts),
        "hidden_size": int(hidden),
        "storage_dtype": "float16",
    },
    "pairing": {
        "pair_map": PAIR_MAP.name,
        "pairs_table": PAIRS_CSV.name,
        "pairs_measured": int(len(joined)),
        "pairs_dropped_not_in_cache": int(dropped),
        "tactics": int(joined["tactic"].nunique()),
    },
    "headline_layer": headline_layer,
    "statistics": {
        "estimator": "median",
        "interval": "percentile bootstrap",
        "n_boot": N_BOOT,
        "alpha": ALPHA,
        "seed": SEED,
    },
    "floors": baselines,
    "deviations_to_state_in_the_paper": [
        "Activations are Qwen2.5-7B, not Llama-3.1-8B: the Llama harvest did not "
        "run because the available GPU could not hold it in bf16.",
        "Activations are 4-bit NF4, not the bf16 reference configuration. The "
        "measured NF4-vs-bf16 displacement is an upper bound on contamination; a "
        "paired difference partly cancels systematic quantization error, and a "
        "paired-delta floor under NF4 has not been measured.",
        "The harvest used plain HF hooks rather than Undercurrent, so these rows "
        "do not carry an Undercurrent version as section 6 of the spec requires.",
        "A6 is the layer sweep only. The position split needs last_instruction, "
        "which this cache does not contain.",
    ],
    "outputs": [
        "a1_per_pair_qwen25-7b.csv",
        "a1_summary_by_tactic_qwen25-7b.csv",
        "a6_layer_sweep_qwen25-7b.csv",
        "a1_a6_meta.json",
    ],
}
tmp = RESULTS / "a1_a6_meta.json.partial"
tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
os.replace(tmp, RESULTS / "a1_a6_meta.json")

print("\n" + "=" * 72)
print("FILES WRITTEN")
print("=" * 72)
for name in meta["outputs"]:
    path = RESULTS / name
    print(f"  {path}  ({path.stat().st_size:,} bytes)")

print("\n" + "=" * 72)
print("A1 AND A6 COMPLETE")
print("=" * 72)
print(f"  {len(joined):,} pairs, {joined['tactic'].nunique()} tactics, "
      f"{len(layers)} layers")
print("  Paste me the summary above and I will write it into the paper.")
