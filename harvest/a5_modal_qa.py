# SPDX-License-Identifier: Apache-2.0
"""QA on the 37 Llama prompt-position archives: NaN, inf, shape, row counts.

CPU only, reads the a5-outputs volume in place, ~10 min.
Writes results/qa_llama31-8b.json and results/qa_llama31-8b.csv.

    modal run --detach a5_modal_qa.py::qa
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "a5-ws2-llama-qa"
CONFIG_NAME = "hf-bf16-promptpos-full"
EXPECTED_SHARDS = 37
EXPECTED_PROMPTS = 18500
DATA = Path("/data")
OUT = Path("/outputs")
SCRATCH = Path("/tmp/qa")

# The same image as the analysis, so Modal serves it from cache.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("pixz", "tar", "git")
    .pip_install(
        "torch==2.8.0",
        "safetensors>=0.4",
        "pandas>=2.0",
        "pyarrow>=15",
        "numpy>=1.23",
    )
)
app = modal.App(APP_NAME)
out_volume = modal.Volume.from_name("a5-outputs", create_if_missing=True)


@app.function(image=image, volumes={str(OUT): out_volume}, cpu=8.0, memory=32768, timeout=3600)
def _qa():
    import numpy as np
    import pandas as pd
    from safetensors.torch import load_file

    t0 = time.time()
    archives = sorted((OUT / "archives").glob(f"*__{CONFIG_NAME}__*shard_*.tpxz"))
    if len(archives) != EXPECTED_SHARDS:
        raise RuntimeError(f"expected {EXPECTED_SHARDS} archives, found {len(archives)}")

    rows, prompts_seen = [], set()
    for n, arc in enumerate(archives, 1):
        dest = SCRATCH / "one"
        subprocess.run(["rm", "-rf", str(dest)], check=True)
        dest.mkdir(parents=True)
        subprocess.run(["tar", "-xf", str(arc), "-C", str(dest)], check=True)
        act = next(dest.rglob("activations.safetensors"))
        idx = pd.read_parquet(next(dest.rglob("index.parquet")))

        tensors = load_file(str(act))
        n_nan = n_inf = n_tensors = 0
        shapes = set()
        for key, t in tensors.items():
            arr = t.float().numpy()
            n_nan += int(np.isnan(arr).sum())
            n_inf += int(np.isinf(arr).sum())
            shapes.add(tuple(arr.shape))
            n_tensors += 1
        pids = set(idx["prompt_id"].astype(str))
        prompts_seen |= pids
        rows.append({
            "shard": arc.name.split("__shard_")[-1].replace(".tpxz", ""),
            "index_rows": len(idx),
            "tensors": n_tensors,
            "prompts": len(pids),
            "nan": n_nan,
            "inf": n_inf,
            "shapes": ";".join(sorted(str(s) for s in shapes)),
        })
        print(f"[{n:2d}/{len(archives)}] {rows[-1]['shard']}  prompts {len(pids):>4}  "
              f"tensors {n_tensors:>5}  nan {n_nan}  inf {n_inf}", flush=True)
        shutil_rm = subprocess.run(["rm", "-rf", str(dest)])

    df = pd.DataFrame(rows)
    summary = {
        "archives": len(archives),
        "distinct_prompts": len(prompts_seen),
        "expected_prompts": EXPECTED_PROMPTS,
        "total_nan": int(df["nan"].sum()),
        "total_inf": int(df["inf"].sum()),
        "shard_prompt_sum": int(df["prompts"].sum()),
        "all_shapes": sorted(set(";".join(df["shapes"]).split(";"))),
        "passed": bool(df["nan"].sum() == 0 and df["inf"].sum() == 0
                       and len(prompts_seen) == EXPECTED_PROMPTS),
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    res = OUT / "results"
    res.mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "qa_llama31-8b.csv", index=False)
    (res / "qa_llama31-8b.json").write_text(json.dumps(summary, indent=2))
    out_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def qa():
    print(json.dumps(_qa.remote(), indent=2))
