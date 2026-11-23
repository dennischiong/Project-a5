# SPDX-License-Identifier: Apache-2.0
"""Project A-5 / WS2 - X4: representation drift per layer under each serving config
(work plan section 7.3; Kayson). Llama-3.1-8B, CPU only.

WHAT IT MEASURES
  For every prompt in the 500-prompt Llama subset and every layer, the activation under a
  config (fp16 prompt-position, or NF4 when its archive exists) against the bf16 reference
  for the SAME prompt and SAME position, from the full bf16 Llama harvest:
    * relative L2  ||h_cfg - h_ref|| / ||h_ref||          (median over prompts, 95% CI)
    * cosine distance 1 - cos(h_cfg, h_ref)                (median, 95% CI)
    * linear CKA between the two prompt-by-dimension matrices of a layer (centred)
  and the drift as a FRACTION of the adversarial displacement from A1 (the plan's X4
  definition): median relative L2 drift / median A1 displacement at the headline layer.

HONEST LIMITS
  * Only configs whose 500-prompt archive exists are run. Missing configs are reported as
    MISSING, never filled in.
  * The NF4 (C6) configs are run only if their archive is present and single-shard.
  * CKA is computed per layer over the 500 prompts, so it is a comparison of two matrices
    of the same prompts, not a population estimate.

INPUTS (volumes): a5-outputs/archives/*__<config>__*shard_*.tpxz;
  a5-outputs/results/a1_per_pair_llama31-8b.csv (for the A1 denominator).

OUTPUTS on a5-outputs/results: x4_drift_llama31-8b.csv, x4_summary_llama31-8b.json.
No prompt text and no reply text is written.

    modal deploy a5_modal_x4.py
    modal run a5_modal_x4.py::x4_preflight       # which archives exist, ~1 min
    modal run a5_modal_x4.py::x4_run             # ~10-20 min, CPU
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

APP_NAME = "a5-ws2-x4"
MODEL_ALIAS = "llama31-8b"
DATASET = "a5-pairs-eval-subset"
REF_CONFIG = "hf-bf16-promptpos-full"          # C1 reference, full harvest
SUBSET_CONFIGS = {                              # config -> expected archive count
    "fp16": ("hf-fp16-promptpos-sub500", 1),
    "nf4": ("hf-nf4-promptpos-sub500", 1),     # C6, only if it exists
}
REF_SHARDS = 37
POINTS = ("last_prefill", "last_instruction")
HIDDEN = 4096
LAYERS = {"last_prefill": list(range(32)), "last_instruction": list(range(1, 32, 2))}
HEADLINE_LAYER = 31
N_BOOT = 2000
ALPHA = 0.05
SEED = 0

DATA = Path("/data")
OUT = Path("/outputs")
RESULTS = OUT / "results"
SCRATCH = Path("/tmp/x4_restore")

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
          "HF_HOME": "/cache/hf", "A5_PROFILE": "prompts", "A5_GPU": "A100-40GB"})
)
app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("a5-data", create_if_missing=True)
out_volume = modal.Volume.from_name("a5-outputs", create_if_missing=True)
VOLUMES = {str(DATA): data_volume, str(OUT): out_volume}


# ============================================================
# PURE HELPERS (unit-tested locally)
# ============================================================


def rel_l2(h_cfg, h_ref):
    import numpy as np
    h_cfg = np.asarray(h_cfg, dtype="float64")
    h_ref = np.asarray(h_ref, dtype="float64")
    d = np.linalg.norm(h_cfg - h_ref, axis=-1)
    n = np.linalg.norm(h_ref, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, d / n, np.nan)


def cosine_distance(h_cfg, h_ref):
    import numpy as np
    a = np.asarray(h_cfg, dtype="float64")
    b = np.asarray(h_ref, dtype="float64")
    num = np.einsum("ij,ij->i", a, b)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 1.0 - num / den, np.nan)


def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019) between two (n_prompts, dim) matrices of the same
    prompts, after column-centring. Identical representations give 1.0."""
    import numpy as np
    X = np.asarray(X, dtype="float64")
    Y = np.asarray(Y, dtype="float64")
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / den) if den > 0 else float("nan")


def boot_median(values, n_boot=N_BOOT, alpha=ALPHA, seed=SEED) -> dict:
    import numpy as np
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "median": None, "lo": None, "hi": None}
    if arr.size == 1:
        return {"n": 1, "median": float(arr[0]), "lo": float(arr[0]), "hi": float(arr[0])}
    rng = np.random.default_rng(seed)
    meds = np.median(arr[rng.integers(0, arr.size, (n_boot, arr.size))], axis=1)
    return {"n": int(arr.size), "median": float(np.median(arr)),
            "lo": float(np.percentile(meds, 100 * alpha / 2)),
            "hi": float(np.percentile(meds, 100 * (1 - alpha / 2)))}


def drift_fraction(drift_median, a1_median):
    """X4's 'drift as a fraction of the adversarial displacement from A1'."""
    if drift_median is None or a1_median in (None, 0):
        return None
    return float(drift_median / a1_median)


def archives_for(config: str, expected: int):
    """Archive paths for one config, refusing a mixed set. Never accepts a partial set."""
    files = sorted(p for p in (OUT / "archives").glob(f"*__{config}__*shard_*.tpxz"))
    bases = {p.name.split("__shard_")[0] for p in files}
    if not files:
        return None, []
    if len(bases) != 1:
        raise RuntimeError(f"{config}: archives span {len(bases)} bases {sorted(bases)}")
    if len(files) != expected:
        raise RuntimeError(f"{config}: expected {expected} shard(s), found {len(files)}")
    return bases.pop(), files


# ============================================================
# CONTAINER CODE
# ============================================================


def _extract(archive: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    # A transient volume read error ("Cannot read: Invalid argument") once failed a run on an
    # archive that was intact (sha256 verified afterwards). Retry from a clean destination.
    last = ""
    for attempt in range(1, 4):
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        proc = subprocess.run(["tar", "-xf", str(archive), "-C", str(dest)],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            break
        last = proc.stderr.strip()
        print(f"  tar attempt {attempt} failed on {archive.name}: {last[:200]}", flush=True)
        time.sleep(10 * attempt)
    else:
        raise RuntimeError(f"tar failed on {archive.name} after 3 attempts: {last}")
    found = sorted(dest.rglob("activations.safetensors"))
    if len(found) != 1 or not (found[0].parent / "index.parquet").is_file():
        raise RuntimeError(f"{archive.name}: expected one activations.safetensors + index")
    return found[0].parent


def _read_layers(files, wanted_prompts=None):
    """{(point, layer): {prompt_id: float32 vector}} for every shard in `files`.
    One shard extracted at a time. Only the LAYERS in LAYERS[point] are kept."""
    import numpy as np
    import pandas as pd
    from safetensors import safe_open

    store = {}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for n, archive in enumerate(files, 1):
        shard = _extract(archive, SCRATCH / "one")
        idx = pd.read_parquet(shard / "index.parquet")
        keep = pd.Series(False, index=idx.index)
        for point in POINTS:
            keep |= ((idx["extraction_point"] == point)
                     & idx["layer"].astype(int).isin(LAYERS[point]))
        idx = idx[keep]
        if wanted_prompts is not None:
            idx = idx[idx["prompt_id"].astype(str).isin(wanted_prompts)]
        n_kept = len(idx)
        with safe_open(str(shard / "activations.safetensors"), framework="pt") as fh:
            for row in idx.itertuples():
                v = fh.get_tensor(row.tensor_key).reshape(-1).float().numpy()
                if v.shape[0] != HIDDEN:
                    raise RuntimeError(f"vector size {v.shape[0]}, expected {HIDDEN}")
                key = (str(row.extraction_point), int(row.layer))
                store.setdefault(key, {})[str(row.prompt_id)] = v.astype(np.float32)
        shutil.rmtree(SCRATCH / "one", ignore_errors=True)
        print(f"  [{n:2d}/{len(files)}] kept {n_kept:>6} vectors", flush=True)
    return store


def _subset_prompts(config_files):
    """The 500 prompt ids of a subset config, read from its own index (no text)."""
    import pandas as pd
    ids = set()
    for archive in config_files:
        shard = _extract(archive, SCRATCH / "ids")
        ids |= set(pd.read_parquet(shard / "index.parquet", columns=["prompt_id"])
                   ["prompt_id"].astype(str))
        shutil.rmtree(SCRATCH / "ids", ignore_errors=True)
    return ids


def _a1_denominator() -> float | None:
    """Median A1 displacement (relative L2) at the headline layer, from the saved A1 run."""
    import pandas as pd
    path = RESULTS / "a1_per_pair_llama31-8b.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    return float(df["relative_l2"].median())


def _run(preflight_only: bool) -> dict:
    import numpy as np
    from datetime import datetime, timezone

    t0 = time.time()
    out_volume.reload()
    report = {"reference": REF_CONFIG, "configs": {}}
    ref_base, ref_files = archives_for(REF_CONFIG, REF_SHARDS)
    report["reference_archives"] = len(ref_files)
    if not ref_files:
        raise RuntimeError(f"reference {REF_CONFIG}: no archives found; nothing to compare to")

    present = {}
    for name, (config, expected) in SUBSET_CONFIGS.items():
        try:
            base, files = archives_for(config, expected)
        except RuntimeError as error:
            report["configs"][name] = {"status": "ERROR", "reason": str(error)}
            continue
        if not files:
            report["configs"][name] = {"status": "MISSING",
                                       "reason": f"no archive for {config} yet"}
            continue
        present[name] = (config, base, files)
        report["configs"][name] = {"status": "found", "config": config, "archives": len(files)}

    if preflight_only:
        report["a1_denominator"] = _a1_denominator()
        return {"preflight": "OK", **report, "seconds": round(time.time() - t0, 1)}

    if not present:
        raise RuntimeError("no X4 config archives are present; run the NF4/fp16 jobs first")

    # the 500-prompt set is defined by the fp16 archive (or the first present config)
    first = next(iter(present.values()))[2]
    subset = _subset_prompts(first)
    report["n_subset_prompts"] = len(subset)

    print("reading the bf16 reference for the subset...", flush=True)
    ref = _read_layers(ref_files, wanted_prompts=subset)
    a1 = _a1_denominator()
    rows, summary = [], {}
    for name, (config, base, files) in present.items():
        print(f"reading {config}...", flush=True)
        cfg = _read_layers(files, wanted_prompts=subset)
        summary[name] = {"config": config, "archive_base": base, "per_layer": {}}
        for point in POINTS:
            for layer in LAYERS[point]:
                key = (point, layer)
                if key not in cfg or key not in ref:
                    continue
                common = sorted(set(cfg[key]) & set(ref[key]))
                if len(common) < 2:
                    continue
                X = np.stack([cfg[key][p] for p in common])
                Y = np.stack([ref[key][p] for p in common])
                rl = rel_l2(X, Y)
                cd = cosine_distance(X, Y)
                l2 = boot_median(rl)
                c = boot_median(cd)
                cka = linear_cka(X, Y)
                frac = drift_fraction(l2["median"], a1) if (
                    point == "last_prefill" and layer == HEADLINE_LAYER) else None
                row = {"config": name, "extraction_point": point, "layer": layer,
                       "n_prompts": len(common), "rel_l2_median": l2["median"],
                       "rel_l2_lo": l2["lo"], "rel_l2_hi": l2["hi"],
                       "cos_dist_median": c["median"], "cos_dist_lo": c["lo"],
                       "cos_dist_hi": c["hi"], "linear_cka": cka,
                       "drift_fraction_of_a1": frac}
                rows.append(row)
                summary[name]["per_layer"][f"{point}:{layer}"] = {
                    k: v for k, v in row.items() if k not in ("config", "extraction_point",
                                                              "layer")}
        head = next((r for r in rows if r["config"] == name and r["layer"] == HEADLINE_LAYER
                     and r["extraction_point"] == "last_prefill"), None)
        summary[name]["headline_L31_last_prefill"] = head

    import pandas as pd
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "x4_drift_llama31-8b.csv", index=False)
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
              "analysis": "X4 representation drift per layer vs bf16 reference",
              "owner": "WS2 (Kayson)", "model_alias": MODEL_ALIAS, "dataset": DATASET,
              "reference": REF_CONFIG, "reference_archive_base": ref_base,
              "a1_denominator_rel_l2": a1,
              "configs": report["configs"], "results": summary,
              "statistics": {"interval": "percentile bootstrap over prompts",
                             "n_boot": N_BOOT, "alpha": ALPHA, "seed": SEED},
              "runtime_seconds": round(time.time() - t0, 1)}
    tmp = RESULTS / "x4_summary_llama31-8b.json.partial"
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, RESULTS / "x4_summary_llama31-8b.json")
    out_volume.commit()
    return {"status": "done", "configs": report["configs"], "rows": len(rows)}


@app.function(image=image, volumes=VOLUMES, cpu=4.0, memory=32768, timeout=3600)
def x4_run() -> dict:
    return _run(preflight_only=False)


@app.function(image=image, volumes=VOLUMES, cpu=1.0, memory=4096, timeout=600)
def x4_preflight() -> dict:
    return _run(preflight_only=True)


@app.local_entrypoint()
def main():
    print(json.dumps(x4_preflight.remote(), indent=1, default=str))
