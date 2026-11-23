# SPDX-License-Identifier: Apache-2.0
"""A9 noise floor on Llama-3.1-8B, through Undercurrent, on Modal.

Three passes over the same 32 prompts, rendered by your own render_and_check:
  bf16 run A, bf16 run B (identical input, should be exactly 0), fp32.
Writes a9_summary_llama31-8b.csv with comparison == "bf16_vs_fp32", which
a5_modal_analysis.py reads for the snr_l2_vs_floor column.

    modal run --detach a5_modal_a9_llama.py::a9

EXECUTION TIME: 15-25 min on A100-40GB (three model loads, 96 forward passes).
fp32 Llama is ~32 GB of GPU memory, which fits the 40 GB card with room for
single-token forwards. If this fails with CUDA out of memory, say so and we
switch the fp32 pass to the 80 GB card. Nothing else changes.
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from pathlib import Path

import modal

APP_NAME = "a5-ws2-llama-a9"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_ALIAS = "llama31-8b"
SPEC_FILE = "llama31-8b-promptpos.yaml"
N_PROMPTS = 32
SALT = "a9-floor"
GPU = "A100-40GB"

DATA = Path("/data")
OUT = Path("/outputs")

# The same image as the harvest and the analysis, so Modal serves it from cache.
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
VOLUMES = {str(DATA): data_volume, str(OUT): out_volume, "/cache": cache_volume}
hf_secret = modal.Secret.from_name("huggingface")


def _prepare_run_module():
    """The project's own run.py, repointed at the data volume."""
    import importlib.util

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, Path(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    code = DATA / "code"
    load(code / "collector.py", "a5_collector")
    run = load(code / "run.py", "a5_run")
    run.ROOT = DATA
    run.HARVEST = DATA
    run.CSV_PATH = DATA / "pairs_table_v1_final-2.csv"
    run.PAIR_MAP_PATH = DATA / "pair_map__d240ec65.parquet"
    run.SPEC_PATH = DATA / "code" / SPEC_FILE
    return run


def _capture(run, prompts, dtype, label):
    """One pass over the prompt set; returns {prompt|point|tensor: float32 array}."""
    import torch
    from transformers import AutoTokenizer
    from undercurrent import ProbedModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    rendered = []
    for p in prompts:
        text, _ids = run.render_and_check(tokenizer, p["prompt_text"])
        rendered.append((str(p["prompt_id"]), text))

    print(f"\n--- capturing: {label} ---", flush=True)
    kwargs = dict(spec=str(run.SPEC_PATH), backend="hf", device="cuda",
                  tokenizer=tokenizer, dtype=dtype)
    try:
        model = ProbedModel.from_pretrained(MODEL_ID, attn_implementation="sdpa", **kwargs)
    except (TypeError, ValueError):
        model = ProbedModel.from_pretrained(MODEL_ID, **kwargs)

    captures = {}
    with model:
        for i, (pid, text) in enumerate(rendered, 1):
            out = model.generate(text, max_new_tokens=1, temperature=0)
            for point_name, result in out.probe_results.items():
                for key, tensor in result.verdict["tensors"].items():
                    captures[f"{pid}|{point_name}|{key}"] = (
                        tensor.to(dtype=torch.float32).cpu().numpy())
            if i % 8 == 0 or i == len(rendered):
                print(f"  {i}/{len(rendered)} prompts, {len(captures)} activations",
                      flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return captures


def _compare(a, b):
    """Per-activation cosine distance and relative L2, a versus b."""
    import numpy as np

    if set(a) != set(b):
        raise ValueError(f"capture keys differ: {len(set(a) - set(b))} only in A, "
                         f"{len(set(b) - set(a))} only in B")
    cos_d, rel = [], []
    for key in sorted(a):
        x = np.asarray(a[key], dtype="float64").ravel()
        y = np.asarray(b[key], dtype="float64").ravel()
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch at {key}: {x.shape} vs {y.shape}")
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        denom = nx * ny
        cos = float(x @ y / denom) if denom > 0 else float("nan")
        cos_d.append(1.0 - cos)
        rel.append(float(np.linalg.norm(x - y) / nx) if nx > 0 else float("nan"))
    return np.array(cos_d), np.array(rel), len(a)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=[hf_secret],
              cpu=4.0, memory=65536, timeout=5400)
def _a9():
    import numpy as np
    import pandas as pd
    import torch

    run = _prepare_run_module()
    all_prompts = run.load_prompt_rows()

    # A fixed, reproducible sample of 32 prompts, chosen by salted hash.
    prompts = sorted(
        all_prompts,
        key=lambda p: hashlib.sha256((SALT + str(p["prompt_id"])).encode()).hexdigest(),
    )[:N_PROMPTS]
    print(f"A9 prompt set: {len(prompts)} prompts; first three "
          f"{[p['prompt_id'] for p in prompts[:3]]}", flush=True)

    a = _capture(run, prompts, torch.bfloat16, "bf16, run A")
    b = _capture(run, prompts, torch.bfloat16, "bf16, run B (identical input)")
    f = _capture(run, prompts, torch.float32, "fp32")

    rows = []
    for name, (x, y) in {"bf16_rerun": (a, b), "bf16_vs_fp32": (a, f)}.items():
        cd, rl, n = _compare(x, y)
        rows.append({
            "comparison": name,
            "n_activations": n,
            "cos_dist_median": float(np.nanmedian(cd)),
            "cos_dist_p95": float(np.nanpercentile(cd, 95)),
            "cos_dist_max": float(np.nanmax(cd)),
            "rel_l2_median": float(np.nanmedian(rl)),
            "rel_l2_p95": float(np.nanpercentile(rl, 95)),
            "rel_l2_max": float(np.nanmax(rl)),
        })
        print(f"\n{name}: n={n:,}  rel L2 p95 {rows[-1]['rel_l2_p95']:.4e}  "
              f"cos p95 {rows[-1]['cos_dist_p95']:.4e}", flush=True)

    results = OUT / "results"
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results / "a9_summary_llama31-8b.csv", index=False)
    out_volume.commit()
    print(f"\nwritten: {results / 'a9_summary_llama31-8b.csv'}", flush=True)
    return rows


@app.local_entrypoint()
def a9():
    rows = _a9.remote()
    print("\n" + json.dumps(rows, indent=2))
