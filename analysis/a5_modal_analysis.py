# SPDX-License-Identifier: Apache-2.0
"""Project A-5 / WS2 - A1 + A6 (with position split) on the Llama bf16 shards.

Runs on Modal, CPU only, reading the a5-outputs volume in place. No Drive
transfer, no GPU, no model: the activations are already harvested.

It fixes all four deviations at once, because it reads shards that were
produced by Undercurrent, in bf16, on Llama-3.1-8B, at both prompt positions,
and it asserts each of those four facts from the shard provenance before it
measures anything.

    !modal run a5_modal_analysis.py::analyse

Outputs land on a5-outputs under results/ and are printed to the log.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = "a5-ws2-llama-analysis"

MODEL_ALIAS = "llama31-8b"
DATASET_NAME = "a5-pairs-eval-subset"
# Only this config. The 6 older `-subset` archives on the same volume are a
# different generation and must never be mixed into one measurement.
CONFIG_NAME = "hf-bf16-promptpos-full"
EXPECTED_SHARDS = 37
EXPECTED_PROMPTS = 18500
HIDDEN_EXPECTED = 4096

N_BOOT = 2000
ALPHA = 0.05
SEED = 0

DATA = Path("/data")
OUT = Path("/outputs")
RESULTS = OUT / "results"
SCRATCH = Path("/tmp/restore")

# EXACTLY the image the harvest used, byte for byte, so Modal serves it from
# cache and this starts in seconds instead of rebuilding torch for five
# minutes. The A5_* env vars are harvest-specific and harmless here, but they
# must stay to keep the layer hash identical.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("pixz", "tar", "git")
    .pip_install(
        "torch==2.8.0",
        "transformers>=5.5.3,<6",
        "undercurrent==0.1.0",
        "accelerate>=1.0",
        "safetensors>=0.4",
        "pandas>=2.0",
        "pyarrow>=15",
        "PyYAML>=6,<7",
        "numpy>=1.23",
        "huggingface_hub>=0.26",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false",
          "HF_HOME": "/cache/hf",
          "A5_PROFILE": "prompts", "A5_GPU": "A100-40GB"})
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("a5-data", create_if_missing=True)
out_volume = modal.Volume.from_name("a5-outputs", create_if_missing=True)
cache_volume = modal.Volume.from_name("a5-hf-cache", create_if_missing=True)
VOLUMES = {str(DATA): data_volume, str(OUT): out_volume,
           "/cache": cache_volume}


# ---------------------------------------------------------------- helpers

def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_run_module():
    """The project's own run.py, repointed at the volumes, for the pairing."""
    code = DATA / "code"
    _load(code / "collector.py", "a5_collector")
    run = _load(code / "run.py", "a5_run")
    run.ROOT = DATA
    run.HARVEST = DATA
    run.CSV_PATH = DATA / "pairs_table_v1_final-2.csv"
    run.PAIR_MAP_PATH = DATA / "pair_map__d240ec65.parquet"
    run.SPEC_PATH = DATA / "code" / "llama31-8b-promptpos.yaml"
    return run


def _atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _select_archives():
    """The 37 archives of exactly one config, refusing a mixed set."""
    archives = sorted((OUT / "archives").glob(f"*__{CONFIG_NAME}__*shard_*.tpxz"))
    if not archives:
        raise RuntimeError(
            f"no archives matching config {CONFIG_NAME} under {OUT / 'archives'}")
    bases = {p.name.split("__shard_")[0] for p in archives}
    if len(bases) != 1:
        raise RuntimeError(
            "archives span more than one base name, which would mix two "
            f"different harvests: {sorted(bases)}")
    if len(archives) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"expected {EXPECTED_SHARDS} shards for {CONFIG_NAME}, found "
            f"{len(archives)}: {[p.name for p in archives]}")
    sizes = {p.stat().st_size for p in archives}
    base = bases.pop()
    print(f"  base name : {base}")
    print(f"  shards    : {len(archives)}")
    print(f"  size range: {min(sizes) / 2**20:.1f} - {max(sizes) / 2**20:.1f} MiB")
    return base, archives


def _extract(archive: Path, dest: Path):
    """Extract one .tpxz. pixz output is a valid xz stream, so tar auto-detects."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(["tar", "-xf", str(archive), "-C", str(dest)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"tar failed on {archive.name}: {proc.stderr.strip()}")
    found = sorted(dest.rglob("activations.safetensors"))
    if len(found) != 1:
        raise RuntimeError(
            f"{archive.name} contains {len(found)} activations.safetensors, "
            "expected exactly 1")
    shard_dir = found[0].parent
    if not (shard_dir / "index.parquet").is_file():
        raise RuntimeError(f"{archive.name}: no index.parquet beside the tensors")
    return shard_dir


def _provenance(index, label):
    """The four corrections, asserted from the shard's own index columns."""
    def col(name):
        if name not in index.columns:
            return []
        return sorted({str(v) for v in index[name].dropna().unique()})

    checks = {k: col(k) for k in (
        "model_id", "undercurrent_version", "inference_dtype", "storage_dtype",
        "quantization", "backend", "extraction_point")}

    problems = []
    if not any("llama" in v.lower() for v in checks["model_id"]):
        problems.append(f"model_id is {checks['model_id']}, expected Llama-3.1-8B")
    if not checks["undercurrent_version"]:
        problems.append("no undercurrent_version column - cannot confirm the harness")
    if any(v.strip().lower() in ("none", "", "nan")
           for v in checks["undercurrent_version"]):
        problems.append("undercurrent_version says 'none' - not harvested via Undercurrent")
    if any(q.strip().lower() not in ("none", "", "nan") for q in checks["quantization"]):
        problems.append(f"quantization is {checks['quantization']}, expected none (bf16)")
    if not any("bfloat16" in v
               for v in checks["inference_dtype"] + checks["storage_dtype"]):
        problems.append(
            f"dtype is {checks['inference_dtype']}/{checks['storage_dtype']}, "
            "expected bfloat16")
    points = set(checks["extraction_point"])
    if not {"last_prefill", "last_instruction"} <= points:
        problems.append(
            f"extraction points are {sorted(points)}; the position split needs "
            "both last_prefill and last_instruction")
    return checks, problems


# ---------------------------------------------------------------- the job

@app.function(image=image, volumes=VOLUMES, cpu=8.0, memory=65536, timeout=7200)
def _run_analysis():
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.torch import load_file
    from datetime import datetime, timezone

    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("A1 + A6 ON LLAMA bf16 (Undercurrent, both prompt positions)")
    print("=" * 72)

    # --- 1. archives -----------------------------------------------------
    print("\n--- 1. ARCHIVES ---")
    base, archives = _select_archives()

    # --- 2. restore, read, release one shard at a time -------------------
    # Peak disk stays at one shard (~200 MB) instead of 7.3 GB.
    print("\n--- 2. READING (extract -> load -> release, per shard) ---")
    index_parts = []
    store = {}
    hidden = None
    row_of = None
    n_prompts = None
    checks = None
    problems = None
    all_prompts = None

    for n, archive in enumerate(archives, 1):
        tick = time.time()
        shard_dir = _extract(archive, SCRATCH / "one")
        idx = pd.read_parquet(shard_dir / "index.parquet")

        required = {"prompt_id", "tensor_key", "extraction_point", "layer"}
        missing = required - set(idx.columns)
        if missing:
            raise RuntimeError(f"{archive.name}: index.parquet missing {missing}")

        # Fail on the first shard rather than after half an hour of work.
        if n == 1:
            print("\n--- 3. PROVENANCE (the four corrections) ---")
            checks, problems = _provenance(idx, archive.name)
            for key, values in checks.items():
                print(f"  {key:<22} {values}")
            if problems:
                print("\n  DEVIATIONS PRESENT:")
                for p in problems:
                    print("   -", p)
                raise RuntimeError(
                    "provenance does not satisfy the four mandatory corrections; "
                    "refusing to produce numbers that would misreport the method")
            print("\n  ALL FOUR CONFIRMED: Llama, Undercurrent, bf16, both positions.")
            print("\n--- 2. READING (continued) ---")

            # One pass over every shard's index would need all 37 extracted, so
            # the prompt row space comes from run.py instead, which is the same
            # 18,500 the harvest sharded.
            run_mod = _prepare_run_module()
            all_prompts = run_mod.load_prompt_rows()
            prompt_ids = sorted({str(p["prompt_id"]) for p in all_prompts})
            if len(prompt_ids) != EXPECTED_PROMPTS:
                raise RuntimeError(
                    f"run.py returned {len(prompt_ids):,} prompts, expected "
                    f"{EXPECTED_PROMPTS:,}")
            row_of = {pid: i for i, pid in enumerate(prompt_ids)}
            n_prompts = len(prompt_ids)

        tensors = load_file(str(shard_dir / "activations.safetensors"))
        keys = set(tensors.keys())
        sub = idx[idx["tensor_key"].isin(keys)]
        if len(sub) != len(idx):
            raise RuntimeError(
                f"{archive.name}: {len(idx) - len(sub)} index rows have no tensor")

        for row in sub.itertuples():
            key = (str(row.extraction_point), int(row.layer))
            t = tensors[row.tensor_key]
            if hidden is None:
                hidden = int(t.shape[-1]) if t.ndim > 1 else int(t.shape[0])
                if hidden != HIDDEN_EXPECTED:
                    raise RuntimeError(
                        f"hidden size {hidden}, expected {HIDDEN_EXPECTED} for "
                        "Llama-3.1-8B")
                print(f"  hidden size: {hidden}")
            if key not in store:
                store[key] = np.full((n_prompts, hidden), np.nan, dtype="float16")
            pid = str(row.prompt_id)
            if pid not in row_of:
                raise RuntimeError(f"{archive.name}: prompt_id {pid} not in run.py's set")
            store[key][row_of[pid]] = t.reshape(-1).to(torch.float32).numpy()

        # Keep only what the end-of-run provenance re-check and the prompt
        # count need. tensor_key is the biggest string column and is finished
        # with here, so dropping it keeps the retained index small.
        keep_cols = [c for c in (
            "prompt_id", "extraction_point", "layer", "model_id",
            "undercurrent_version", "inference_dtype", "storage_dtype",
            "quantization", "backend") if c in idx.columns]
        index_parts.append(idx[keep_cols].copy())
        n_rows = len(idx)          # read BEFORE the del below; the log line needs it
        del tensors, idx
        shutil.rmtree(SCRATCH / "one", ignore_errors=True)
        print(f"  [{n:2d}/{len(archives)}] {archive.name.split('__shard_')[-1]:<18}"
              f" rows {n_rows:>6,}  {time.time() - tick:5.1f}s", flush=True)

    index = pd.concat(index_parts, ignore_index=True)
    print(f"\n  index rows total : {len(index):,}")
    print(f"  (point, layer)   : {len(store)}")
    print(f"  matrices held    : "
          f"{len(store) * n_prompts * hidden * 2 / 1e9:.2f} GB float16")

    # Re-assert provenance over the whole set, not just shard 0.
    checks, problems = _provenance(index, "all shards")
    if problems:
        raise RuntimeError(f"provenance failed across the full index: {problems}")
    harvested = sorted({str(v) for v in index["prompt_id"].unique()})
    print(f"  distinct prompts : {len(harvested):,}")
    if len(harvested) != EXPECTED_PROMPTS:
        raise RuntimeError(
            f"{len(harvested):,} distinct prompts harvested, expected "
            f"{EXPECTED_PROMPTS:,}")

    # --- 4. pairing ------------------------------------------------------
    print("\n--- 4. PAIRING ---")
    pairs_map = {}
    for prompt in all_prompts:
        for link in json.loads(prompt["pair_links_json"]):
            entry = pairs_map.setdefault(link["pair_id"], {
                "pair_id": link["pair_id"], "source": link.get("source"),
                "tactic": link.get("tactic"), "split": link.get("split"),
                "holdout_group": link.get("holdout_group")})
            entry[link["role"]] = str(prompt["prompt_id"])

    rows = []
    for rec in pairs_map.values():
        v, a = rec.get("vanilla"), rec.get("adversarial")
        if v in row_of and a in row_of:
            rows.append({**rec, "van_row": row_of[v], "adv_row": row_of[a]})
    pairs = pd.DataFrame(rows)
    print(f"  pairs with both halves harvested: {len(pairs):,}")
    if len(pairs) < 1000:
        raise RuntimeError("too few complete pairs; the harvest did not cover the set")
    print("  per source:")
    for s, cnt in pairs["source"].value_counts().items():
        print(f"    {str(s):<18} {cnt:>6,}")
    print(f"  tactics: {pairs['tactic'].nunique()}")

    van_rows = pairs["van_row"].to_numpy()
    adv_rows = pairs["adv_row"].to_numpy()

    # --- metrics ---------------------------------------------------------
    def displacement(matrix, van, adv, block=4096):
        cos_d = np.full(len(van), np.nan, dtype="float64")
        rel = np.full(len(van), np.nan, dtype="float64")
        for start in range(0, len(van), block):
            stop = min(start + block, len(van))
            v = matrix[van[start:stop]].astype("float32")
            a = matrix[adv[start:stop]].astype("float32")
            ok = np.isfinite(v).all(axis=1) & np.isfinite(a).all(axis=1)
            nv = np.linalg.norm(v, axis=1)
            na = np.linalg.norm(a, axis=1)
            dot = np.einsum("ij,ij->i", v, a)
            denom = nv * na
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.where((denom > 0) & ok, dot / denom, np.nan)
                cos_d[start:stop] = 1.0 - c
                rel[start:stop] = np.where(
                    (nv > 0) & ok,
                    np.linalg.norm(a - v, axis=1) / nv, np.nan)
        return cos_d, rel

    def boot(values, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
        arr = np.asarray(values, dtype="float64")
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"n": 0, "point": float("nan"), "lo": float("nan"),
                    "hi": float("nan")}
        if arr.size == 1:
            return {"n": 1, "point": float(arr[0]), "lo": float(arr[0]),
                    "hi": float(arr[0])}
        rng = np.random.default_rng(seed)
        idx_b = rng.integers(0, arr.size, size=(n_boot, arr.size))
        d = np.median(arr[idx_b], axis=1)
        return {"n": int(arr.size), "point": float(np.median(arr)),
                "lo": float(np.percentile(d, 100 * alpha / 2)),
                "hi": float(np.percentile(d, 100 * (1 - alpha / 2)))}

    # The A9 precision floor, only if one has been measured ON LLAMA. No
    # substitute, no carried-over number from another model.
    floor_l2 = floor_cos = None
    floor_source = None
    for cand in (OUT / "qa" / "a9" / "a9_summary.csv",
                 OUT / "results" / "a9_summary_llama31-8b.csv"):
        if cand.is_file():
            a9 = pd.read_csv(cand)
            sel = a9[a9["comparison"] == "bf16_vs_fp32"] if "comparison" in a9.columns \
                else a9
            if len(sel):
                floor_cos = float(sel.iloc[0]["cos_dist_p95"])
                floor_l2 = float(sel.iloc[0]["rel_l2_p95"])
                floor_source = str(cand)
                break
    if floor_l2 is None:
        print("\n  A9 floor on Llama: NOT MEASURED YET. snr_l2_vs_floor is left "
              "empty rather than borrowed from another model.")
    else:
        print(f"\n  A9 floor ({floor_source}): cosine {floor_cos:.3e}, "
              f"relative L2 {floor_l2:.3e}")

    # --- 5. A6 -----------------------------------------------------------
    print("\n--- 5. A6: LAYER SWEEP AND POSITION SPLIT ---")
    sweep = []
    cache = {}
    for (point, layer) in sorted(store):
        cd, rl = displacement(store[(point, layer)], van_rows, adv_rows)
        cache[(point, layer)] = (cd, rl)
        c, l = boot(cd), boot(rl)
        sweep.append({"extraction_point": point, "layer": layer, "n_pairs": l["n"],
                      "cos_dist_median": c["point"], "cos_dist_lo": c["lo"],
                      "cos_dist_hi": c["hi"], "rel_l2_median": l["point"],
                      "rel_l2_lo": l["lo"], "rel_l2_hi": l["hi"],
                      "snr_l2_vs_floor": (l["point"] / floor_l2) if floor_l2 else None})
    sweep_df = pd.DataFrame(sweep).sort_values(["extraction_point", "layer"])
    sweep_df.to_csv(RESULTS / "a6_layer_sweep_llama31-8b.csv", index=False)

    for point in sorted(set(sweep_df["extraction_point"])):
        part = sweep_df[sweep_df["extraction_point"] == point]
        print(f"\n  {point}:")
        for r in part.itertuples():
            print(f"    layer {r.layer:>2}  rel L2 {r.rel_l2_median:.4f} "
                  f"[{r.rel_l2_lo:.4f}, {r.rel_l2_hi:.4f}]  "
                  f"cos {r.cos_dist_median:.4f}")
        peak = part.loc[part["rel_l2_median"].idxmax()]
        print(f"    -> largest rel L2 at layer {int(peak['layer'])} "
              f"({peak['rel_l2_median']:.4f})")

    pre = set(sweep_df[sweep_df.extraction_point == "last_prefill"]["layer"])
    ins = set(sweep_df[sweep_df.extraction_point == "last_instruction"]["layer"])
    shared = sorted(pre & ins)
    if shared:
        print("\n  POSITION SPLIT (layers present at both positions):")
        split_rows = []
        for layer in shared:
            a = sweep_df[(sweep_df.extraction_point == "last_prefill")
                         & (sweep_df.layer == layer)].iloc[0]
            b = sweep_df[(sweep_df.extraction_point == "last_instruction")
                         & (sweep_df.layer == layer)].iloc[0]
            split_rows.append({
                "layer": int(layer),
                "last_prefill_rel_l2": a.rel_l2_median,
                "last_prefill_lo": a.rel_l2_lo, "last_prefill_hi": a.rel_l2_hi,
                "last_instruction_rel_l2": b.rel_l2_median,
                "last_instruction_lo": b.rel_l2_lo, "last_instruction_hi": b.rel_l2_hi,
                "difference": a.rel_l2_median - b.rel_l2_median,
                "intervals_disjoint": bool(a.rel_l2_lo > b.rel_l2_hi
                                           or b.rel_l2_lo > a.rel_l2_hi)})
            print(f"    layer {layer:>2}  prefill {a.rel_l2_median:.4f}  "
                  f"instruction {b.rel_l2_median:.4f}  "
                  f"diff {a.rel_l2_median - b.rel_l2_median:+.4f}"
                  f"{'  (disjoint CIs)' if split_rows[-1]['intervals_disjoint'] else ''}")
        pd.DataFrame(split_rows).to_csv(
            RESULTS / "a6_position_split_llama31-8b.csv", index=False)
    else:
        print("\n  no layer is present at both positions; no split to report")

    # --- 6. A1 -----------------------------------------------------------
    print("\n--- 6. A1: DISPLACEMENT PER TACTIC ---")
    prefill_layers = sorted(l for (p, l) in store if p == "last_prefill")
    headline = max(prefill_layers)
    cd, rl = cache[("last_prefill", headline)]
    print(f"  headline: last_prefill, layer {headline}")

    per_pair = pairs[["pair_id", "source", "tactic", "split", "holdout_group"]].copy()
    per_pair["extraction_point"] = "last_prefill"
    per_pair["layer"] = headline
    per_pair["cosine_distance"] = cd
    per_pair["relative_l2"] = rl
    per_pair.to_csv(RESULTS / "a1_per_pair_llama31-8b.csv", index=False)

    summary = []
    for (src, tac), part in per_pair.groupby(["source", "tactic"], dropna=False):
        c, l = boot(part["cosine_distance"]), boot(part["relative_l2"])
        summary.append({"source": src, "tactic": tac, "n_pairs": l["n"],
                        "cos_dist_median": c["point"], "cos_dist_lo": c["lo"],
                        "cos_dist_hi": c["hi"], "rel_l2_median": l["point"],
                        "rel_l2_lo": l["lo"], "rel_l2_hi": l["hi"],
                        "snr_l2_vs_floor": (l["point"] / floor_l2) if floor_l2 else None})
    summary_df = pd.DataFrame(summary).sort_values("rel_l2_median", ascending=False)
    summary_df.to_csv(RESULTS / "a1_summary_by_tactic_llama31-8b.csv", index=False)

    degenerate = summary_df["rel_l2_median"] < 1e-9
    real = summary_df[~degenerate]
    print(f"  tactic groups: {len(summary_df)}   degenerate (identical text): "
          f"{int(degenerate.sum())}")
    print("\n  largest:")
    for r in real.head(10).itertuples():
        print(f"    {str(r.source)[:14]:<14} {str(r.tactic)[:26]:<26} "
              f"n={r.n_pairs:>5}  {r.rel_l2_median:.4f} "
              f"[{r.rel_l2_lo:.4f}, {r.rel_l2_hi:.4f}]")
    print("\n  smallest (non-degenerate):")
    for r in real.tail(10).itertuples():
        print(f"    {str(r.source)[:14]:<14} {str(r.tactic)[:26]:<26} "
              f"n={r.n_pairs:>5}  {r.rel_l2_median:.4f} "
              f"[{r.rel_l2_lo:.4f}, {r.rel_l2_hi:.4f}]")

    controls = per_pair[per_pair["source"].astype(str) == "Controls"]
    baselines = {}
    for label, pattern in (("paraphrase", "paraphrase"), ("length_filler", "length")):
        sel = controls[controls["tactic"].astype(str)
                       .str.contains(pattern, case=False, na=False)]
        if len(sel):
            ci = boot(sel["relative_l2"])
            baselines[label] = ci
            print(f"\n  baseline {label:<14} n={ci['n']:>5}  {ci['point']:.4f} "
                  f"[{ci['lo']:.4f}, {ci['hi']:.4f}]")
    if floor_l2:
        baselines["precision_floor_p95"] = {"point": floor_l2}
        print(f"\n  clearing the precision floor ({floor_l2:.4f}): "
              f"{int((real['rel_l2_lo'] > floor_l2).sum())} of {len(real)}")

    _atomic_json(RESULTS / "a1_a6_llama31-8b_meta.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analyses": ["A1 displacement per tactic", "A6 layer sweep",
                     "A6 position split"],
        "owner": "WS2 (Kayson)",
        "model_alias": MODEL_ALIAS,
        "dataset": DATASET_NAME,
        "config": CONFIG_NAME,
        "archive_base": base,
        "corrections_confirmed": True,
        "deviations_remaining": [],
        "provenance_from_index": checks,
        "n_shards": len(archives),
        "n_prompts": int(n_prompts),
        "n_prompts_harvested": int(len(harvested)),
        "hidden_size": int(hidden),
        "pairs_measured": int(len(pairs)),
        "tactic_groups": int(len(summary_df)),
        "degenerate_tactic_groups": int(degenerate.sum()),
        "headline_point": "last_prefill",
        "headline_layer": int(headline),
        "a9_floor_rel_l2_p95": floor_l2,
        "a9_floor_cos_p95": floor_cos,
        "a9_floor_source": floor_source,
        "baselines": baselines,
        "statistics": {"estimator": "median", "interval": "percentile bootstrap",
                       "n_boot": N_BOOT, "alpha": ALPHA, "seed": SEED},
        "runtime_seconds": round(time.time() - t0, 1),
    })

    out_volume.commit()

    print("\n" + "=" * 72)
    print("FILES WRITTEN TO a5-outputs/results")
    print("=" * 72)
    written = {}
    for name in ("a1_per_pair_llama31-8b.csv",
                 "a1_summary_by_tactic_llama31-8b.csv",
                 "a6_layer_sweep_llama31-8b.csv",
                 "a6_position_split_llama31-8b.csv",
                 "a1_a6_llama31-8b_meta.json"):
        p = RESULTS / name
        if p.is_file():
            written[name] = p.stat().st_size
            print(f"  {name:<42} {p.stat().st_size:>12,} bytes")
    print(f"\nWall clock: {(time.time() - t0) / 60:.1f} min")
    print("=" * 72)
    return {"files": written, "pairs": int(len(pairs)),
            "runtime_min": round((time.time() - t0) / 60, 1)}


@app.local_entrypoint()
def analyse():
    result = _run_analysis.remote()
    print("\nreturned:", json.dumps(result, indent=2))
