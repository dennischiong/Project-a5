# SPDX-License-Identifier: Apache-2.0
"""Project A-5 / WS2 - Llama-3.1-8B bf16 harvest, on Modal.

WHAT THIS IS
------------
Your harvest, moved to a rented GPU. It does NOT rewrite your pipeline: it
imports your own harvest/run.py and harvest/collector.py inside the container
and calls your validated functions - render_and_check, extract_prompt,
write_completed_shard, verify_local_shard - so the chat-template check, the
NaN/inf checks, the safetensors+parquet shard format, the SHA-256 manifests and
the pixz archiving all behave exactly as they do in Colab.

Three things differ, and only three:

  1. The GPU is an A100 instead of a T4, so Llama-3.1-8B runs in bf16 - the
     configuration the project spec calls the reference. No 4-bit, so none of
     the 18% quantization displacement measured in Week 3.
  2. Google Drive is not mountable from a container, so DRIVE_ROOT points at a
     Modal Volume. The archives land there in exactly the format Drive expects,
     and the Colab driver copies them across afterwards.
  3. Each shard is a separate function call, so they run in parallel and each
     writes its own manifest. No shared-state races. Re-running skips any shard
     whose archive already verifies, so an interrupted run resumes.

HOW TO RUN IT
-------------
From Colab, after `modal token set`:

    !modal run a5_modal_harvest.py::prepare        # one-off: upload inputs
    !modal run a5_modal_harvest.py::harvest_all    # the harvest
    !modal run a5_modal_harvest.py::status         # what finished

Then the Colab driver cell pulls the archives into Drive.

COST
----
A100-40GB is about $2-3/hour on Modal. Eight shards of 500 prompts, in
parallel, is roughly 10-15 minutes of wall clock and a few dollars of the
team's $100. Set GPU = "L4" below to spend about a third as much for maybe
twice the wall clock.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import modal

# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "a5-ws2-llama-harvest"

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_ALIAS = "llama31-8b"
DATASET_NAME = "a5-pairs-eval-subset"

# Which prompt-set file on the data volume, and how many tokens to generate.
# Three profiles, matching the checklist:
#   prompts  18,500 prompts, prompt positions only, 1 new token  -> A1, A6, A9
#   judges    2,000 prompts, 256 greedy tokens, replies kept     -> judges, A4
#   decode      500 prompts, 20 greedy tokens, gen positions     -> X6
PROFILE = os.environ.get("A5_PROFILE", "prompts")

PROFILES = {
    # file None = every validated prompt from run.load_prompt_rows() (18,500).
    "prompts": {"file": None, "max_new_tokens": 1,
                "spec": "llama31-8b-promptpos.yaml",
                "config": "hf-bf16-promptpos-full"},
    "judges":  {"file": "llama31-8b_eval_subset.json", "max_new_tokens": 256,
                "spec": "llama31-8b-promptpos.yaml",
                "config": "hf-bf16-replies-2k"},
    "decode":  {"file": "llama31-8b_decode_subset.json", "max_new_tokens": 20,
                "spec": "llama31-8b-decode.yaml",
                "config": "hf-bf16-decode-500"},
}
if PROFILE not in PROFILES:
    raise SystemExit(f"A5_PROFILE must be one of {sorted(PROFILES)}")

SUBSET_FILE = PROFILES[PROFILE]["file"]
SPEC_FILE = PROFILES[PROFILE]["spec"]
CONFIG_NAME = PROFILES[PROFILE]["config"]

SHARD_SIZE = 500
MAX_NEW_TOKENS = PROFILES[PROFILE]["max_new_tokens"]
INFERENCE_DTYPE = "torch.bfloat16"
STORAGE_DTYPE = "torch.bfloat16"

GPU = os.environ.get("A5_GPU", "A100-40GB")   # A5_GPU=L4 is about a third of the cost
TIMEOUT = 5400      # 90 minutes per shard; generous, and it is resumable
MAX_CONTAINERS = 20  # parallel shards; raise if the workspace allows

# Paths inside the container
DATA = Path("/data")        # inputs, read-only in practice
OUT = Path("/outputs")      # shards, archives, manifests, logs
CACHE = Path("/cache")      # Hugging Face weights, so 16 GB downloads once

# ============================================================
# IMAGE AND VOLUMES
# ============================================================

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
          # Read at import time inside the container; without these the
          # container would silently fall back to the prompts profile.
          "A5_PROFILE": PROFILE, "A5_GPU": GPU})
)

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("a5-data", create_if_missing=True)
out_volume = modal.Volume.from_name("a5-outputs", create_if_missing=True)
cache_volume = modal.Volume.from_name("a5-hf-cache", create_if_missing=True)

VOLUMES = {str(DATA): data_volume, str(OUT): out_volume, str(CACHE): cache_volume}

# The HF token. Create it once from Colab with:
#   !modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
hf_secret = modal.Secret.from_name("huggingface")


# ============================================================
# SHARED SETUP, RUN INSIDE THE CONTAINER
# ============================================================


def _prepare_modules():
    """Import the project's own run.py and collector.py, and repoint their paths.

    Returns (run_module, archive_module). Everything the harvest relies on is
    the project's code; only the directory constants change.
    """
    import importlib.util
    import sys

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, Path(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    code = DATA / "code"

    # collector.py registers the "collector" probe type by import.
    load(code / "collector.py", "a5_collector")

    # drive_archive.py, with DRIVE_ROOT moved to the output volume. The archive
    # format, the pixz index check, the SHA-256 manifest and the .partial
    # rename all stay exactly as written.
    archive = load(code / "drive_archive.py", "a5_drive_archive")
    archive.DRIVE_ROOT = OUT / "archives"
    archive.SCRATCH = Path("/tmp/archives")
    archive.DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    archive.SCRATCH.mkdir(parents=True, exist_ok=True)

    # run.py, with its Colab/Drive constants repointed at the volumes.
    run = load(code / "run.py", "a5_run")
    run.ROOT = DATA
    run.HARVEST = DATA
    run.CSV_PATH = DATA / "pairs_table_v1_final-2.csv"
    run.PAIR_MAP_PATH = DATA / "pair_map__d240ec65.parquet"
    run.SPEC_PATH = DATA / "code" / SPEC_FILE
    run.COLLECTOR_PATH = code / "collector.py"
    run.ARCHIVE_MODULE_PATH = code / "drive_archive.py"

    run.MODEL_ID = MODEL_ID
    run.MODEL_ALIAS = MODEL_ALIAS
    run.DATASET_NAME = DATASET_NAME
    run.FULL_CONFIG = CONFIG_NAME
    run.PILOT_CONFIG = CONFIG_NAME + "-pilot"
    run.MAX_NEW_TOKENS = MAX_NEW_TOKENS
    run.PRODUCTION_SHARD_SIZE = SHARD_SIZE
    run.INFERENCE_DTYPE = INFERENCE_DTYPE
    run.STORAGE_DTYPE = STORAGE_DTYPE

    run.LOCAL_OUTPUT = Path("/tmp/harvest")
    run.LOG_DIR = OUT / "logs"
    run.MANIFEST_DIR = OUT / "manifests"
    for folder in (run.LOCAL_OUTPUT, run.LOG_DIR, run.MANIFEST_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    return run, archive


def _subset_prompts(run):
    """The recorded evaluation subset, in a fixed order, validated by run.py."""
    all_prompts = run.load_prompt_rows()
    if SUBSET_FILE is None:
        ids = sorted({p["prompt_id"] for p in all_prompts})
        subset = {"prompt_ids": ids, "n_pairs": None,
                  "pair_ids_sha256": run.sha256_text("\n".join(ids))}
        if len(ids) != 18500:
            raise RuntimeError(f"expected 18,500 prompts, run.py returned {len(ids):,}")
    else:
        subset = json.loads((DATA / SUBSET_FILE).read_text())
    keep = set(subset["prompt_ids"])

    by_id = {p["prompt_id"]: p for p in all_prompts}

    missing = keep - set(by_id)
    if missing:
        raise RuntimeError(
            f"{len(missing)} subset prompt ids are absent from the pairs table; "
            "the table changed after the subset was chosen."
        )

    # Sorted by prompt_id so shard boundaries are deterministic across runs.
    prompts = [by_id[pid] for pid in sorted(keep)]
    return subset, prompts


def _shard_bounds(n_prompts, shard_index):
    start = shard_index * SHARD_SIZE
    end = min(start + SHARD_SIZE, n_prompts)
    if start >= n_prompts:
        return None
    return start, end


# ============================================================
# ENTRYPOINT: UPLOAD INPUTS
# ============================================================


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def _list_data():
    """Report what is on the data volume, for the driver to verify."""
    found = {}
    for path in sorted(DATA.rglob("*")):
        if path.is_file():
            found[str(path.relative_to(DATA))] = path.stat().st_size
    return found


@app.local_entrypoint()
def prepare():
    """Print what the data volume holds. Upload with `modal volume put`."""
    print("Files on the a5-data volume:")
    found = _list_data.remote()
    if not found:
        print("  (empty)")
    for name, size in found.items():
        print(f"  {name:<52} {size:>14,} bytes")

    required = [
        "pairs_table_v1_final-2.csv",
        "pair_map__d240ec65.parquet",
        *([SUBSET_FILE] if SUBSET_FILE else []),
        "code/run.py",
        "code/collector.py",
        "code/drive_archive.py",
        f"code/{SPEC_FILE}",
    ]
    missing = [r for r in required if r not in found]
    if missing:
        print("\nMISSING - upload these before harvesting:")
        for name in missing:
            print("  ", name)
    else:
        print("\nAll required inputs present.")


# ============================================================
# THE HARVEST, ONE SHARD PER CALL
# ============================================================


@app.function(
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    secrets=[hf_secret],
    # Undercurrent's HF adapter loads the weights on CPU and then calls
    # .to("cuda"), so the container needs room for 16 GB of bf16 weights
    # before any of it reaches the GPU.
    cpu=4.0,
    memory=32768,
    timeout=TIMEOUT,
    retries=1,
    max_containers=MAX_CONTAINERS,
)
def harvest_shard(shard_index: int) -> dict:
    """Harvest one shard of the evaluation subset and archive it.

    Uses the project's own extract_prompt and write_completed_shard, so the
    output format and every integrity check are unchanged.
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from undercurrent import ProbedModel

    os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
    (CACHE / "hf").mkdir(parents=True, exist_ok=True)

    started = time.time()
    run, archive = _prepare_modules()
    subset, prompts = _subset_prompts(run)

    bounds = _shard_bounds(len(prompts), shard_index)
    if bounds is None:
        return {"shard_index": shard_index, "status": "out_of_range",
                "n_prompts": len(prompts)}
    start, end = bounds
    shard_name = f"shard_{shard_index:05d}"

    spec_text, parsed_spec, spec_hash = run.load_spec()

    base_archive_name = archive.archive_name(
        MODEL_ALIAS, DATASET_NAME, CONFIG_NAME, spec_hash
    )
    drive_archive_name = base_archive_name[:-5] + f"__{shard_name}.tpxz"

    # Already done? The verified archive is the completion marker, exactly as
    # in Colab, so a re-run is cheap and safe.
    if run.verify_drive_archive(archive, drive_archive_name):
        out_volume.commit()
        return {
            "shard_index": shard_index,
            "shard": shard_name,
            "status": "already_archived",
            "archive": drive_archive_name,
        }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)
    context_limit = run.get_context_limit(config, tokenizer)
    hidden_size = int(config.hidden_size)
    template_hash = run.sha256_text(str(getattr(tokenizer, "chat_template", "")))

    try:
        from huggingface_hub import HfApi

        revision = HfApi().model_info(MODEL_ID).sha
    except Exception:
        revision = "unknown"

    shard_prompts = prompts[start:end]
    error_path = run.LOG_DIR / f"{MODEL_ALIAS}__{CONFIG_NAME}__{shard_name}_errors.jsonl"

    model_kwargs = dict(
        spec=str(run.SPEC_PATH),
        backend="hf",
        device="cuda",
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
    )
    try:
        model = ProbedModel.from_pretrained(
            MODEL_ID, attn_implementation="sdpa", **model_kwargs
        )
        attn_used = "sdpa"
    except (TypeError, ValueError):
        model = ProbedModel.from_pretrained(MODEL_ID, **model_kwargs)
        attn_used = "default"

    completed = []
    load_done = time.time()
    with model:
        for prompt in shard_prompts:
            try:
                result = run.extract_prompt(
                    model, tokenizer, prompt, context_limit,
                    hidden_size, template_hash,
                )
            except Exception as error:
                run.log_failure(error_path, prompt, error)
                out_volume.commit()
                raise
            if result is None:
                raise RuntimeError(
                    f"extract_prompt returned None for {prompt['prompt_id']}; "
                    "refusing to archive a shard with a silently dropped prompt.")
            completed.append({"prompt": prompt, "result": result})
    if len(completed) != len(shard_prompts):
        raise RuntimeError(
            f"{shard_name}: harvested {len(completed)} of {len(shard_prompts)} prompts; "
            "refusing to archive an incomplete shard.")

    del model
    torch.cuda.empty_cache()

    partial_dir = run.LOCAL_OUTPUT / f"{shard_name}.partial"
    final_dir = run.LOCAL_OUTPUT / shard_name
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    manifest = run.write_completed_shard(
        partial_dir, shard_name, completed, revision,
        spec_hash, template_hash, CONFIG_NAME,
    )
    partial_dir.rename(final_dir)

    if not run.verify_local_shard(final_dir, spec_hash, template_hash, revision):
        raise RuntimeError(f"{shard_name} failed local verification after writing.")

    archive.archive_to_drive(str(final_dir), drive_archive_name)
    if not run.verify_drive_archive(archive, drive_archive_name):
        raise RuntimeError(f"{shard_name} archive failed verification.")

    # A per-shard manifest, so parallel shards never contend for one file.
    shard_record = {
        "shard": shard_name,
        "shard_index": shard_index,
        "archive": drive_archive_name,
        "prompt_range": [start, end],
        "prompt_count": manifest["prompt_count"],
        "activation_count": manifest["activation_count"],
        "spec_hash": spec_hash,
        "template_hash": template_hash,
        "model_revision": revision,
        "model_id": MODEL_ID,
        "dataset": DATASET_NAME,
        "config": CONFIG_NAME,
        "backend": "hf",
        "concurrency": 1,
        "inference_dtype": INFERENCE_DTYPE,
        "storage_dtype": STORAGE_DTYPE,
        "quantization": "none",
        "attn_implementation": attn_used,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "subset_pair_ids_sha256": subset.get("pair_ids_sha256"),
        "seconds_model_load": round(load_done - started, 1),
        "seconds_total": round(time.time() - started, 1),
        "completed_at": run.utc_now(),
        "status": "complete",
    }
    run.atomic_json(run.MANIFEST_DIR / f"{shard_name}.json", shard_record)

    shutil.rmtree(final_dir, ignore_errors=True)
    out_volume.commit()
    return shard_record


# ============================================================
# ENTRYPOINTS: RUN AND INSPECT
# ============================================================


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def _plan():
    """How many shards the recorded subset needs."""
    run, _archive = _prepare_modules()
    subset, prompts = _subset_prompts(run)
    n = len(prompts)
    shards = (n + SHARD_SIZE - 1) // SHARD_SIZE
    return {
        "n_prompts": n,
        "n_pairs": subset.get("n_pairs"),
        "shard_size": SHARD_SIZE,
        "n_shards": shards,
        "subset_sha256": subset.get("pair_ids_sha256"),
    }


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=3600)
def _download_weights():
    """Fetch Llama once into the shared cache, so 20 shards do not race on it."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        MODEL_ID,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    )
    cache_volume.commit()
    return path


@app.local_entrypoint()
def plan():
    """Print the shard plan for the current A5_PROFILE without harvesting."""
    print(f"profile={PROFILE}  prompts={SUBSET_FILE}  spec={SPEC_FILE}  "
          f"max_new_tokens={MAX_NEW_TOKENS}  config={CONFIG_NAME}  gpu={GPU}")
    print(json.dumps(_plan.remote(), indent=2))


@app.local_entrypoint()
def harvest_all():
    """Harvest every shard, in parallel, then report."""
    print(f"profile={PROFILE}  prompts={SUBSET_FILE}  "
          f"spec={SPEC_FILE}  max_new_tokens={MAX_NEW_TOKENS}  "
          f"config={CONFIG_NAME}")
    plan = _plan.remote()
    print("Plan:", json.dumps(plan, indent=2))

    print("Caching model weights once (first run ~5 min, later runs seconds) ...")
    print("  cached at", _download_weights.remote())

    indices = list(range(plan["n_shards"]))
    print(f"\nLaunching {len(indices)} shards of {SHARD_SIZE} prompts on {GPU} ...")
    print("Each shard is independent and resumable; re-run this to fill gaps.\n")

    started = time.time()
    done, failed = [], []
    for record in harvest_shard.map(indices, return_exceptions=True):
        if isinstance(record, Exception):
            failed.append(repr(record))
            print("  FAILED:", repr(record)[:300])
        else:
            done.append(record)
            print(f"  {record.get('shard', '?'):<14} {record['status']:<18}"
                  f" {record.get('prompt_count', '')} prompts"
                  f" {record.get('activation_count', '')} activations"
                  f" {record.get('seconds_total', '')} s")

    print(f"\nWall clock: {(time.time() - started) / 60:.1f} min")
    print(f"Succeeded: {len(done)}   Failed: {len(failed)}")
    total_prompts = sum(r.get("prompt_count", 0) or 0 for r in done)
    total_acts = sum(r.get("activation_count", 0) or 0 for r in done)
    print(f"Prompts harvested: {total_prompts:,}")
    print(f"Activations: {total_acts:,}")
    if failed:
        print("\nRe-run `modal run a5_modal_harvest_v2.py::harvest_all` to retry the "
              "failures; completed shards are skipped.")
    else:
        print("\nAll shards archived. Next: pull them into Drive from Colab.")


@app.function(image=image, volumes=VOLUMES, timeout=900)
def _status():
    archives = sorted((OUT / "archives").glob("*.tpxz")) if (OUT / "archives").is_dir() else []
    manifests = sorted((OUT / "manifests").glob("*.json")) if (OUT / "manifests").is_dir() else []
    logs = sorted((OUT / "logs").glob("*.jsonl")) if (OUT / "logs").is_dir() else []
    return {
        "archives": [(p.name, p.stat().st_size) for p in archives],
        "manifests": [p.name for p in manifests],
        "error_logs": [(p.name, p.stat().st_size) for p in logs],
        "total_archive_bytes": sum(p.stat().st_size for p in archives),
    }


@app.local_entrypoint()
def status():
    """What is finished on the output volume."""
    info = _status.remote()
    print(f"Archives: {len(info['archives'])}")
    for name, size in info["archives"]:
        print(f"  {name}  ({size:,} bytes)")
    print(f"\nTotal: {info['total_archive_bytes'] / 1e9:.2f} GB")
    print(f"Shard manifests: {len(info['manifests'])}")
    if info["error_logs"]:
        print("\nError logs present (non-empty means prompts failed):")
        for name, size in info["error_logs"]:
            print(f"  {name}  ({size:,} bytes)")
    else:
        print("No error logs.")
