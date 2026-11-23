# SPDX-License-Identifier: Apache-2.0
"""Judges replies: greedy 256-token replies for the 2,742-prompt judges subset.

The judges archives hold activations only, so the replies were never saved.
This regenerates them with the same prompts, the same rendering (run.render_and_check),
the same spec and greedy decoding, and keeps the reply text.

    modal run --detach a5_modal_replies.py::replies

EXECUTION TIME: about 100-120 min wall clock. Six shards run in parallel, 500 prompts
each, 256 tokens per reply. Writes replies/shard_XXXXX.jsonl on a5-outputs.
Reply text stays on the volume and must never be committed to GitHub.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import modal

APP_NAME = "a5-ws2-llama-replies"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SPEC_FILE = "llama31-8b-promptpos.yaml"
SUBSET_FILE = "llama31-8b_eval_subset.json"
SHARD = 500
MAX_NEW_TOKENS = 256
GPU = "A100-40GB"

DATA = Path("/data")
OUT = Path("/outputs")

# Same image as the harvest and analysis, so Modal serves it from cache.
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
          "A5_PROFILE": "judges", "A5_GPU": "A100-40GB"})
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("a5-data", create_if_missing=True)
out_volume = modal.Volume.from_name("a5-outputs", create_if_missing=True)
cache_volume = modal.Volume.from_name("a5-hf-cache", create_if_missing=True)
VOLUMES = {str(DATA): data_volume, str(OUT): out_volume, "/cache": cache_volume}
hf_secret = modal.Secret.from_name("huggingface")


def _prepare_run_module():
    """The project's run.py, repointed at the data volume (same as the analysis)."""
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


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=[hf_secret],
              cpu=4.0, memory=32768, timeout=9000, retries=1)
def _replies_shard(shard_index: int) -> dict:
    import torch
    from transformers import AutoTokenizer
    from undercurrent import ProbedModel

    run = _prepare_run_module()
    ids = json.loads((DATA / SUBSET_FILE).read_text())["prompt_ids"]
    by_id = {str(p["prompt_id"]): p for p in run.load_prompt_rows()}
    missing = [pid for pid in ids if pid not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} subset ids not in the pairs table")

    chunk = ids[shard_index * SHARD:(shard_index + 1) * SHARD]
    if not chunk:
        return {"shard": shard_index, "status": "out_of_range"}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    kwargs = dict(spec=str(run.SPEC_PATH), backend="hf", device="cuda",
                  tokenizer=tokenizer, dtype=torch.bfloat16)
    try:
        model = ProbedModel.from_pretrained(MODEL_ID, attn_implementation="sdpa", **kwargs)
    except (TypeError, ValueError):
        model = ProbedModel.from_pretrained(MODEL_ID, **kwargs)

    out_dir = OUT / "replies"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shard_{shard_index:05d}.jsonl"
    tmp = path.with_name(path.name + ".partial")

    n = 0
    with model, open(tmp, "w", encoding="utf-8") as fh:
        for pid in chunk:
            text, _ids = run.render_and_check(tokenizer, by_id[pid]["prompt_text"])
            out = model.generate(text, max_new_tokens=MAX_NEW_TOKENS, temperature=0)
            fh.write(json.dumps({"prompt_id": pid, "reply": out.text,
                                 "aborted": bool(out.aborted),
                                 "shard": shard_index}, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    out_volume.commit()
    return {"shard": shard_index, "status": "done", "replies": n, "file": path.name}


@app.local_entrypoint()
def replies():
    n_shards = math.ceil(2742 / SHARD)
    print(f"generating replies for {n_shards} shards in parallel", flush=True)
    results = list(_replies_shard.map(range(n_shards), return_exceptions=True))
    for i, r in enumerate(results):
        print(f"shard {i}: {r}", flush=True)
