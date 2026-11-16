# SPDX-License-Identifier: Apache-2.0
"""Project A-5 / WS2 - pack 2: Track 3 configs C3 and C6, and the Qwen2.5-7B harvest.

WHAT THIS RUNS (work plan: Track 3 and section 6.4)
---------------------------------------------------
Llama-3.1-8B-Instruct on the recorded 500-prompt subset, two extraction
profiles each, under two serving configurations:

    C3  fp16 compute               (compute precision changes, weights unchanged)
    C6  4-bit NF4 weights, bf16 compute   (weight quantization changes only)

      job name              config name                 spec                      tokens
      llama-fp16-prompt     hf-fp16-promptpos-sub500    llama31-8b-promptpos      1
      llama-fp16-decode     hf-fp16-decodegen-sub500    llama31-8b-decode         20
      llama-nf4-prompt      hf-nf4-promptpos-sub500     llama31-8b-promptpos      1
      llama-nf4-decode      hf-nf4-decodegen-sub500     llama31-8b-decode         20

and the replication model, full set, bf16 (same configuration as the Llama
reference, so the cross-model comparison is not confounded by precision):

      qwen-bf16-prompt      hf-bf16-promptpos-all       qwen25-7b-promptpos       1
                            (18,500 prompts, 37 shards of 500)

C1 (HF bf16, batch 1) for the 500-prompt subset already exists: it is the
Llama harvest itself, and the hf-bf16-decode-500 archive.

It reuses the project's own run.py / collector.py / drive_archive.py exactly
as a5_modal_harvest_v2.py does: same chat-template check, NaN/inf checks,
shard format, SHA-256 sidecars and pixz archives. Storage stays bf16 for every
configuration, so a C3 shard is directly comparable with a C1 shard.

Differences from a5_modal_harvest_v2.py, all deliberate:
  * Config names avoid the substrings the Drive pull cell counts
    ("promptpos-full", "decode-500", "replies-2k"), so its counts stay valid.
  * Per-shard manifests are named alias__config__shard.json. The v2 launcher
    names them shard_XXXXX.json, which a later job overwrites.
  * run.py's hard-coded "quantization": "none" is patched IN MEMORY so an NF4
    shard is recorded as nf4. The user's run.py file is never modified.
  * A dry-run "smoke" step proves each job's code path on 2 prompts and
    writes a marker; "launch" refuses jobs whose smoke did not pass.

HOW TO RUN (from Colab, after modal token is set; ~times are estimates)
----------------------------------------------------------------------
    !modal run a5_modal_pack2.py::prepare            # ~3-8 min, caches Qwen weights
    !modal run a5_modal_pack2.py::smoke              # ~8-15 min, 5 jobs in parallel
    !modal deploy a5_modal_pack2.py                  # ~1-2 min, makes the app persistent
    python: modal.Function.from_name("a5-ws2-pack2", "launch_remote").remote()
                                                     # seconds; jobs then run on Modal alone
    !modal run a5_modal_pack2.py::status             # any time

Why deploy: calls spawned from a `modal run` entrypoint are cancelled when that app
stops (even with --detach, Modal keeps only the last one alive). Calls spawned from
inside a deployed app belong to that app and survive the notebook closing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

import modal

# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "a5-ws2-pack2"
DATASET_NAME = "a5-pairs-eval-subset"

LLAMA_ID = "meta-llama/Llama-3.1-8B-Instruct"
QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"

GPU = os.environ.get("A5_GPU", "A100-40GB")
SHARD_SIZE = 500
TIMEOUT = 5400          # 90 min per shard call
MAX_CONTAINERS = 20

DATA = Path("/data")
OUT = Path("/outputs")
CACHE = Path("/cache")

POINTS_PROMPT = ("last_prefill", "last_instruction")
POINTS_DECODE = ("decode_generated",)

# precision -> how the model is loaded and what is written into the records
PRECISION = {
    "bf16": {"torch_dtype": "bfloat16", "label": "torch.bfloat16", "quant": "none"},
    "fp16": {"torch_dtype": "float16", "label": "torch.float16", "quant": "none"},
    "nf4": {"torch_dtype": "bfloat16", "label": "torch.bfloat16 (nf4 compute)",
            "quant": "nf4"},
}

LLAMA_SUBSET = "llama31-8b_decode_subset.json"   # the 500-prompt subset already on a5-data


def _job(model_id, alias, subset, spec, config, tokens, points, precision):
    return {
        "model_id": model_id, "alias": alias, "subset": subset, "spec": spec,
        "config": config, "max_new_tokens": tokens, "points": list(points),
        "precision": precision,
    }


JOBS = {
    "llama-fp16-prompt": _job(LLAMA_ID, "llama31-8b", LLAMA_SUBSET,
                              "llama31-8b-promptpos.yaml", "hf-fp16-promptpos-sub500",
                              1, POINTS_PROMPT, "fp16"),
    "llama-fp16-decode": _job(LLAMA_ID, "llama31-8b", LLAMA_SUBSET,
                              "llama31-8b-decode.yaml", "hf-fp16-decodegen-sub500",
                              20, POINTS_DECODE, "fp16"),
    "llama-nf4-prompt": _job(LLAMA_ID, "llama31-8b", LLAMA_SUBSET,
                             "llama31-8b-promptpos.yaml", "hf-nf4-promptpos-sub500",
                             1, POINTS_PROMPT, "nf4"),
    "llama-nf4-decode": _job(LLAMA_ID, "llama31-8b", LLAMA_SUBSET,
                             "llama31-8b-decode.yaml", "hf-nf4-decodegen-sub500",
                             20, POINTS_DECODE, "nf4"),
    "qwen-bf16-prompt": _job(QWEN_ID, "qwen25-7b", None,
                             "qwen25-7b-promptpos.yaml", "hf-bf16-promptpos-all",
                             1, POINTS_PROMPT, "bf16"),
}

# Launch order: the four small Llama jobs first, so they get containers first.
LAUNCH_ORDER = ["llama-fp16-prompt", "llama-fp16-decode",
                "llama-nf4-prompt", "llama-nf4-decode", "qwen-bf16-prompt"]

QWEN_LAYERS = 28            # Qwen2.5-7B-Instruct: 28 layers, hidden size 3584
QWEN_HIDDEN = 3584
QWEN_PROMPTS = 18500

QWEN_SPEC_TEXT = (
    'version: "1"\n'
    "\n"
    "extraction_points:\n"
    "  - name: last_prefill\n"
    f"    layers: [{', '.join(str(i) for i in range(QWEN_LAYERS))}]\n"
    "    tensor_type: residual_stream\n"
    '    position: "prompt[-1]"\n'
    "    probe_type: collector\n"
    "    probe_kind: single_shot\n"
    "\n"
    "  - name: last_instruction\n"
    f"    layers: [{', '.join(str(i) for i in range(1, QWEN_LAYERS, 2))}]\n"
    "    tensor_type: residual_stream\n"
    '    position: "prompt[-6]"\n'
    "    probe_type: collector\n"
    "    probe_kind: single_shot\n"
)

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
        "bitsandbytes>=0.46.1",
        "safetensors>=0.4",
        "pandas>=2.0",
        "pyarrow>=15",
        "PyYAML>=6,<7",
        "numpy>=1.23",
        "huggingface_hub>=0.26",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false",
          "HF_HOME": "/cache/hf"})
)

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("a5-data", create_if_missing=True)
out_volume = modal.Volume.from_name("a5-outputs", create_if_missing=True)
cache_volume = modal.Volume.from_name("a5-hf-cache", create_if_missing=True)
VOLUMES = {str(DATA): data_volume, str(OUT): out_volume, str(CACHE): cache_volume}
hf_secret = modal.Secret.from_name("huggingface")

# ============================================================
# PURE HELPERS (no Modal, no GPU: unit-testable)
# ============================================================

REQUIRED_OLD = 'required_prompt_points = {"last_prefill", "last_instruction"}'
QUANT_LITERAL = re.compile(r'"quantization":\s*"none",')
STORAGE_ANCHOR = 'STORAGE_DTYPE = "torch.bfloat16"'
QUANT_CONSTANT = 'QUANTIZATION = "none"'
QUANT_USE = '"quantization": QUANTIZATION,'


def patch_run_source(src: str, points, quant: str) -> str:
    """Return run.py source with (1) the required extraction points following the
    job and (2) the quantization recorded from a constant instead of "none".

    Both edits are exact-match: if run.py is not what this expects, an NF4 job
    raises rather than silently writing "quantization": "none" for 4-bit data.
    (_assert_quantization_recorded later re-checks the shard that was written.)
    """
    import ast

    if src.count(REQUIRED_OLD) != 1:
        raise RuntimeError("run.py changed: the required-points line was not found "
                           "exactly once, so the patch does not apply.")
    src = src.replace(REQUIRED_OLD,
                      f"required_prompt_points = set({sorted(points)!r})")

    already = QUANT_CONSTANT in src and QUANT_USE in src
    if not already:
        hits = QUANT_LITERAL.findall(src)
        if len(hits) >= 1 and src.count(STORAGE_ANCHOR) == 1:
            src = src.replace(STORAGE_ANCHOR, STORAGE_ANCHOR + "\n" + QUANT_CONSTANT, 1)
            src = QUANT_LITERAL.sub(QUANT_USE, src)
        elif quant != "none":
            raise RuntimeError(
                "run.py does not have the expected '\"quantization\": \"none\",' "
                f"literal (found {len(hits)}); refusing to run an NF4 job that would "
                "be recorded as unquantized.")
    ast.parse(src)
    return src


def expected_activations_per_prompt(spec_text: str) -> int:
    import yaml
    spec = yaml.safe_load(spec_text)
    return sum(len(p["layers"]) for p in spec["extraction_points"])


def _find_key(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and not isinstance(v, (dict, list)):
                yield v
            yield from _find_key(v, key)
    elif isinstance(obj, list):
        for item in obj:
            yield from _find_key(item, key)


def assert_quantization_recorded(shard_dir, quant: str) -> list:
    """Read what run.py actually wrote (JSON manifests and parquet index rows) and
    require every 'quantization' field to equal this job's. For a quantized job at
    least one such field must exist, so 'nothing to check' cannot pass silently."""
    import pandas as pd

    seen = set()
    for path in sorted(Path(shard_dir).rglob("*")):
        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            seen.update(str(v) for v in _find_key(data, "quantization"))
        elif path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            if "quantization" in frame.columns:
                seen.update(str(v) for v in frame["quantization"].unique())
    if seen and seen != {quant}:
        raise RuntimeError(f"shard records say quantization={sorted(seen)}, "
                           f"but this job is {quant!r}.")
    if quant != "none" and not seen:
        raise RuntimeError("no 'quantization' field found in the written shard, so an "
                           f"{quant} job cannot be verified as recorded.")
    return sorted(seen)


def archive_matches(name: str, job: dict) -> bool:
    return name.endswith(".tpxz") and job["alias"] in name and job["config"] in name


def _shard_bounds(n_prompts: int, shard_index: int):
    start = shard_index * SHARD_SIZE
    end = min(start + SHARD_SIZE, n_prompts)
    return None if start >= n_prompts else (start, end)


# ============================================================
# SHARED SETUP, RUN INSIDE THE CONTAINER
# ============================================================


def _prepare_modules(job):
    """Load the project's run.py / collector.py / drive_archive.py, repointed at
    the volumes, with the in-memory patches. Returns (run, archive)."""
    import importlib.util
    import sys

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, Path(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    code = DATA / "code"
    load(code / "collector.py", "a5_collector")

    archive = load(code / "drive_archive.py", "a5_drive_archive")
    archive.DRIVE_ROOT = OUT / "archives"
    archive.SCRATCH = Path("/tmp/archives")
    archive.DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    archive.SCRATCH.mkdir(parents=True, exist_ok=True)

    quant = PRECISION[job["precision"]]["quant"]
    src = (code / "run.py").read_text(encoding="utf-8")
    src = patch_run_source(src, job["points"], quant)
    patched = Path("/tmp/a5_run_patched.py")
    patched.write_text(src, encoding="utf-8")
    run = load(patched, "a5_run")

    run.ROOT = DATA
    run.HARVEST = DATA
    run.CSV_PATH = DATA / "pairs_table_v1_final-2.csv"
    run.PAIR_MAP_PATH = DATA / "pair_map__d240ec65.parquet"
    run.SPEC_PATH = code / job["spec"]
    run.COLLECTOR_PATH = code / "collector.py"
    run.ARCHIVE_MODULE_PATH = code / "drive_archive.py"

    run.MODEL_ID = job["model_id"]
    run.MODEL_ALIAS = job["alias"]
    run.DATASET_NAME = DATASET_NAME
    run.FULL_CONFIG = job["config"]
    run.PILOT_CONFIG = job["config"] + "-pilot"
    run.MAX_NEW_TOKENS = job["max_new_tokens"]
    run.PRODUCTION_SHARD_SIZE = SHARD_SIZE
    run.INFERENCE_DTYPE = PRECISION[job["precision"]]["label"]
    run.STORAGE_DTYPE = "torch.bfloat16"
    if hasattr(run, "QUANTIZATION") or quant != "none":
        run.QUANTIZATION = quant

    run.LOCAL_OUTPUT = Path("/tmp/harvest")
    run.LOG_DIR = OUT / "logs"
    run.MANIFEST_DIR = OUT / "manifests"
    for folder in (run.LOCAL_OUTPUT, run.LOG_DIR, run.MANIFEST_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    return run, archive


def _subset_prompts(run, job):
    all_prompts = run.load_prompt_rows()
    if job["subset"] is None:
        ids = sorted({p["prompt_id"] for p in all_prompts})
        subset = {"prompt_ids": ids, "n_pairs": None,
                  "pair_ids_sha256": run.sha256_text("\n".join(ids))}
        if len(ids) != QWEN_PROMPTS:
            raise RuntimeError(f"expected {QWEN_PROMPTS:,} prompts, got {len(ids):,}")
    else:
        subset = json.loads((DATA / job["subset"]).read_text())
    keep = set(subset["prompt_ids"])
    by_id = {p["prompt_id"]: p for p in all_prompts}
    missing = keep - set(by_id)
    if missing:
        raise RuntimeError(f"{len(missing)} subset prompt ids are absent from the pairs table")
    return subset, [by_id[pid] for pid in sorted(keep)]


def _load_model(job, tokenizer, spec_path):
    """Load through Undercurrent. NF4 follows the call the Week 3 smoke test and the
    runbook use (quantization_config + device_map, no device/dtype); the others use
    device='cuda' with an explicit dtype, as the bf16 harvest does."""
    import torch
    from undercurrent import ProbedModel

    precision = job["precision"]
    kwargs = dict(spec=str(spec_path), backend="hf", tokenizer=tokenizer)
    if precision == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
        # Undercurrent's HF adapter calls model.to(device) after loading, and device
        # defaults to "cpu"; without this line the 4-bit model was moved to the CPU.
        kwargs["device"] = "cuda"
    else:
        kwargs["device"] = "cuda"
        kwargs["dtype"] = getattr(torch, PRECISION[precision]["torch_dtype"])
    try:
        model = ProbedModel.from_pretrained(job["model_id"], attn_implementation="sdpa",
                                            **kwargs)
        attn = "sdpa"
    except (TypeError, ValueError):
        model = ProbedModel.from_pretrained(job["model_id"], **kwargs)
        attn = "default"
    return model, attn


def _check_weights_footprint(job, gb):
    """Independent proof of the precision actually loaded: 7-8B weights are ~15-16 GB
    in bf16/fp16 and ~6 GB in 4-bit NF4."""
    if job["precision"] == "nf4":
        if gb > 10.0:
            raise RuntimeError(f"NF4 job holds {gb:.1f} GB on the GPU after load; "
                               "the weights were not quantized.")
    elif gb < 12.0:
        raise RuntimeError(f"{job['precision']} job holds only {gb:.1f} GB after load; "
                           "expected a full 16-bit model.")
    if job["precision"] == "nf4" and gb < 3.0:
        raise RuntimeError(f"NF4 job holds only {gb:.2f} GB on the GPU after load; "
                           "the model is not on the GPU.")


def placement_report(hf_model, adapter_device, n_layers: int) -> dict:
    """Where every parameter and buffer lives, how many 4-bit linear layers exist, and
    which device Undercurrent sends the inputs to."""
    devices = {}
    for t in list(hf_model.parameters()) + list(hf_model.buffers()):
        devices[str(t.device)] = devices.get(str(t.device), 0) + int(t.numel())
    n4 = sum(1 for m in hf_model.modules() if type(m).__name__ == "Linear4bit")
    return {"tensor_devices": devices, "linear4bit_modules": n4,
            "linear4bit_expected": 7 * n_layers, "adapter_device": str(adapter_device)}


def assert_placement(report: dict, precision: str) -> None:
    off = sorted(d for d in report["tensor_devices"] if not d.startswith("cuda"))
    if off:
        raise RuntimeError(f"model tensors are on {off}, not the GPU: {report}")
    if not report["adapter_device"].startswith("cuda"):
        raise RuntimeError(f"Undercurrent sends inputs to {report['adapter_device']!r}, not cuda")
    if precision == "nf4" and report["linear4bit_modules"] != report["linear4bit_expected"]:
        raise RuntimeError(f"{report['linear4bit_modules']} Linear4bit layers, expected "
                           f"{report['linear4bit_expected']}: the model is not fully NF4")
    if precision != "nf4" and report["linear4bit_modules"]:
        raise RuntimeError("a 16-bit job contains Linear4bit layers")


def _hf_module(probed):
    adapter = getattr(probed, "_adapter", None)
    hf = getattr(adapter, "_model", None) if adapter is not None else None
    if hf is None:
        raise RuntimeError("cannot reach the HF model at ProbedModel._adapter._model")
    return adapter, hf


# ============================================================
# ONE SHARD (REAL RUN) OR TWO PROMPTS (DRY RUN), SAME CODE PATH
# ============================================================


def _execute(job_name: str, shard_index: int, dry_run: bool) -> dict:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    job = JOBS[job_name]
    os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
    (CACHE / "hf").mkdir(parents=True, exist_ok=True)

    started = time.time()
    run, archive = _prepare_modules(job)
    subset, prompts = _subset_prompts(run, job)
    spec_text, _parsed, spec_hash = run.load_spec()

    if dry_run:
        shard_name = "smoke"
        start = end = None
        shard_prompts = sorted(prompts, key=lambda p: len(p["prompt_text"]))[:2]
    else:
        bounds = _shard_bounds(len(prompts), shard_index)
        if bounds is None:
            return {"job": job_name, "shard_index": shard_index, "status": "out_of_range"}
        start, end = bounds
        shard_name = f"shard_{shard_index:05d}"
        shard_prompts = prompts[start:end]

    base_name = archive.archive_name(job["alias"], DATASET_NAME, job["config"], spec_hash)
    archive_name = base_name[:-5] + f"__{shard_name}.tpxz"

    if not dry_run and run.verify_drive_archive(archive, archive_name):
        out_volume.commit()
        return {"job": job_name, "shard": shard_name, "status": "already_archived",
                "archive": archive_name}

    tokenizer = AutoTokenizer.from_pretrained(job["model_id"])
    config = AutoConfig.from_pretrained(job["model_id"])
    context_limit = run.get_context_limit(config, tokenizer)
    hidden_size = int(config.hidden_size)
    template_hash = run.sha256_text(str(getattr(tokenizer, "chat_template", "")))

    tails = []
    if dry_run and "last_instruction" in job["points"]:
        # prompt[-6] must be the last token of the user's own text, not template.
        for prompt in shard_prompts:
            _text, ids = run.render_and_check(tokenizer, prompt["prompt_text"])
            tail = [tokenizer.decode([i]) for i in ids[-8:]]
            tails.append({"tokens": len(ids), "tail": tail, "at_minus6": tail[2]})
            if "<|" in tail[2]:
                raise RuntimeError(f"prompt[-6] is a template token {tail[2]!r}; "
                                   f"tail={tail}. The offset is wrong for this model.")

    try:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(job["model_id"]).sha
    except Exception:
        revision = "unknown"

    model, attn_used = _load_model(job, tokenizer, run.SPEC_PATH)
    gpu_gb = round(torch.cuda.memory_allocated(0) / 1e9, 2)
    _check_weights_footprint(job, gpu_gb)
    _adapter, _hf = _hf_module(model)
    placement = placement_report(_hf, getattr(_adapter, "_device", None),
                                 int(config.num_hidden_layers))
    assert_placement(placement, job["precision"])
    placement["gpu_gb_reserved"] = round(torch.cuda.memory_reserved(0) / 1e9, 2)
    print(f"[{job_name}] placement OK: {placement}", flush=True)
    load_done = time.time()

    error_path = run.LOG_DIR / f"{job['alias']}__{job['config']}__{shard_name}_errors.jsonl"
    completed = []
    with model:
        for prompt in shard_prompts:
            try:
                result = run.extract_prompt(model, tokenizer, prompt, context_limit,
                                            hidden_size, template_hash)
            except Exception as error:
                run.log_failure(error_path, prompt, error)
                out_volume.commit()
                raise
            if result is None:
                raise RuntimeError(f"extract_prompt returned None for {prompt['prompt_id']}; "
                                   "refusing to archive a shard with a dropped prompt.")
            completed.append({"prompt": prompt, "result": result})
            if len(completed) % 50 == 0 or len(completed) == len(shard_prompts):
                print(f"[{job_name}] {len(completed)}/{len(shard_prompts)} prompts, "
                      f"{time.time() - load_done:.0f} s since load", flush=True)
    if len(completed) != len(shard_prompts):
        raise RuntimeError(f"{shard_name}: harvested {len(completed)} of "
                           f"{len(shard_prompts)} prompts; refusing to archive.")

    del model
    torch.cuda.empty_cache()

    work = Path("/tmp/harvest_smoke") if dry_run else run.LOCAL_OUTPUT
    work.mkdir(parents=True, exist_ok=True)
    partial_dir = work / f"{shard_name}.partial"
    final_dir = work / shard_name
    for stale in (partial_dir, final_dir):
        if stale.exists():
            shutil.rmtree(stale)
    partial_dir.mkdir(parents=True)

    manifest = run.write_completed_shard(partial_dir, shard_name, completed, revision,
                                         spec_hash, template_hash, job["config"])
    partial_dir.rename(final_dir)
    if not run.verify_local_shard(final_dir, spec_hash, template_hash, revision):
        raise RuntimeError(f"{shard_name} failed local verification after writing.")

    quant = PRECISION[job["precision"]]["quant"]
    quantization_recorded = assert_quantization_recorded(final_dir, quant)

    per_prompt_expected = expected_activations_per_prompt(spec_text)
    if "last_instruction" in job["points"]:
        want = per_prompt_expected * len(completed)
        if manifest["activation_count"] != want:
            raise RuntimeError(f"{shard_name}: {manifest['activation_count']} activations, "
                               f"expected {want} ({per_prompt_expected} per prompt).")

    evidence = {
        "job": job_name, "config": job["config"], "shard": shard_name,
        "prompt_count": manifest["prompt_count"],
        "activation_count": manifest["activation_count"],
        "activations_per_prompt_expected": per_prompt_expected,
        "gpu_gb_after_load": gpu_gb, "attn_implementation": attn_used,
        "placement": placement,
        "seconds_extraction": round(time.time() - load_done, 1),
        "quantization_recorded": quantization_recorded,
        "seconds_total": round(time.time() - started, 1),
    }

    if dry_run:
        shutil.rmtree(final_dir, ignore_errors=True)
        evidence.update(status="smoke_ok", offset_check=tails)
        return evidence

    archive.archive_to_drive(str(final_dir), archive_name)
    if not run.verify_drive_archive(archive, archive_name):
        raise RuntimeError(f"{shard_name} archive failed verification.")

    try:
        import bitsandbytes
        bnb_version = bitsandbytes.__version__
    except Exception:
        bnb_version = None

    record = {
        "shard": shard_name, "shard_index": shard_index, "job": job_name,
        "archive": archive_name, "prompt_range": [start, end],
        "prompt_count": manifest["prompt_count"],
        "activation_count": manifest["activation_count"],
        "spec_hash": spec_hash, "template_hash": template_hash,
        "model_revision": revision, "model_id": job["model_id"],
        "dataset": DATASET_NAME, "config": job["config"], "backend": "hf",
        "concurrency": 1,
        "inference_dtype": PRECISION[job["precision"]]["label"],
        "storage_dtype": "torch.bfloat16",
        "quantization": quant,
        "bnb_4bit_compute_dtype": "torch.bfloat16" if quant == "nf4" else None,
        "bnb_4bit_double_quant": True if quant == "nf4" else None,
        "bitsandbytes": bnb_version,
        "gpu_gb_after_load": gpu_gb,
        "placement": placement,
        "attn_implementation": attn_used,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "subset_pair_ids_sha256": subset.get("pair_ids_sha256"),
        "seconds_model_load": round(load_done - started, 1),
        "seconds_total": round(time.time() - started, 1),
        "completed_at": run.utc_now(), "status": "complete",
    }
    # Unique per job: the v2 launcher's shard_XXXXX.json would overwrite another job.
    run.atomic_json(run.MANIFEST_DIR / f"{job['alias']}__{job['config']}__{shard_name}.json",
                    record)
    shutil.rmtree(final_dir, ignore_errors=True)
    out_volume.commit()
    return record


# ============================================================
# MODAL FUNCTIONS
# ============================================================


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=[hf_secret],
              cpu=4.0, memory=32768, timeout=TIMEOUT, retries=1,
              max_containers=MAX_CONTAINERS)
def harvest_shard(job_name: str, shard_index: int) -> dict:
    return _execute(job_name, shard_index, dry_run=False)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=[hf_secret],
              cpu=4.0, memory=32768, timeout=1800, max_containers=MAX_CONTAINERS)
def smoke_job(job_name: str) -> dict:
    import traceback
    try:
        return _execute(job_name, 0, dry_run=True)
    except Exception:
        return {"job": job_name, "status": "FAILED",
                "error": traceback.format_exc()[-1800:]}


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=3600)
def _download_weights(model_id: str) -> str:
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
    path = snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"],
    )
    cache_volume.commit()
    return path


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def _prepare_inputs() -> dict:
    """Write the Qwen spec to the data volume (never overwrites a different file) and
    report which required inputs are present."""
    code = DATA / "code"
    code.mkdir(parents=True, exist_ok=True)
    qspec = code / "qwen25-7b-promptpos.yaml"
    if qspec.exists() and qspec.read_text(encoding="utf-8") != QWEN_SPEC_TEXT:
        raise RuntimeError(f"{qspec} exists and differs from the expected Qwen spec; "
                           "not overwriting. Inspect it.")
    qspec.write_text(QWEN_SPEC_TEXT, encoding="utf-8")
    data_volume.commit()

    needed = ["pairs_table_v1_final-2.csv", "pair_map__d240ec65.parquet", LLAMA_SUBSET,
              "code/run.py", "code/collector.py", "code/drive_archive.py",
              "code/llama31-8b-promptpos.yaml", "code/llama31-8b-decode.yaml",
              "code/qwen25-7b-promptpos.yaml"]
    return {"missing": [n for n in needed if not (DATA / n).is_file()]}


@app.function(image=image, volumes=VOLUMES, timeout=900)
def _write_marker(payload: dict) -> str:
    folder = OUT / "pack2"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "smoke_status.json").write_text(json.dumps(payload, indent=2))
    out_volume.commit()
    return str(folder / "smoke_status.json")


@app.function(image=image, volumes=VOLUMES, timeout=900)
def _read_marker() -> dict:
    path = OUT / "pack2" / "smoke_status.json"
    out_volume.reload()
    return json.loads(path.read_text()) if path.is_file() else {}


@app.function(image=image, volumes=VOLUMES, timeout=900)
def _status() -> dict:
    out_volume.reload()
    arch_dir = OUT / "archives"
    names = sorted(p.name for p in arch_dir.glob("*.tpxz")) if arch_dir.is_dir() else []
    logs = OUT / "logs"
    errors = ([(p.name, p.stat().st_size) for p in sorted(logs.glob("*_errors.jsonl"))
               if p.stat().st_size > 0] if logs.is_dir() else [])
    return {"archives": names, "errors": errors}


@app.function(image=image, volumes=VOLUMES, timeout=900)
def launch_remote() -> dict:
    """Spawn every job whose smoke passed. Runs INSIDE the deployed app, so the
    spawned calls belong to the deployed app and keep running after the caller
    (the notebook) disconnects. A spawn from a `modal run` entrypoint does not."""
    out_volume.reload()
    path = OUT / "pack2" / "smoke_status.json"
    if not path.is_file():
        raise RuntimeError("No smoke result found. Run ::smoke first.")
    marker = json.loads(path.read_text())
    passed = [n for n in LAUNCH_ORDER if n in marker.get("passed", [])]
    skipped = [n for n in LAUNCH_ORDER if n not in passed]
    if not passed:
        raise RuntimeError("No job passed smoke; nothing to launch.")
    spawned = {}
    for name in passed:
        calls = [harvest_shard.spawn(name, i) for i in range(_n_shards(name))]
        spawned[name] = [c.object_id for c in calls]
    return {"spawned": {n: len(v) for n, v in spawned.items()}, "not_launched": skipped,
            "call_ids": spawned}


NF4_PROMPT_JOB = "llama-nf4-prompt"
NF4_CHECK = OUT / "pack2" / "nf4_prompt_gpu_check.json"


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=[hf_secret],
              cpu=4.0, memory=32768, timeout=1800)
def nf4_prompt_gpu_check() -> dict:
    """2-prompt dry run of llama-nf4-prompt with the placement checks. Archives nothing."""
    result = _execute(NF4_PROMPT_JOB, 0, dry_run=True)
    NF4_CHECK.parent.mkdir(parents=True, exist_ok=True)
    NF4_CHECK.write_text(json.dumps(result, indent=2, default=str))
    out_volume.commit()
    return result


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=[hf_secret],
              cpu=4.0, memory=32768, timeout=3600)
def nf4_prompt_harvest() -> dict:
    """The single 500-prompt llama-nf4-prompt shard. Refuses unless the GPU check passed."""
    out_volume.reload()
    check = json.loads(NF4_CHECK.read_text()) if NF4_CHECK.is_file() else {}
    if check.get("status") != "smoke_ok" or not check.get("placement"):
        raise RuntimeError("nf4_prompt_gpu_check has not passed; run it first.")
    return _execute(NF4_PROMPT_JOB, 0, dry_run=False)


NF4_DECODE_JOB = "llama-nf4-decode"
NF4_DECODE_CHECK = OUT / "pack2" / "nf4_decode_gpu_check.json"


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=[hf_secret],
              cpu=4.0, memory=32768, timeout=1800)
def nf4_decode_gpu_check() -> dict:
    """2-prompt dry run of llama-nf4-decode (20 greedy tokens) with the placement checks.
    Archives nothing."""
    result = _execute(NF4_DECODE_JOB, 0, dry_run=True)
    NF4_DECODE_CHECK.parent.mkdir(parents=True, exist_ok=True)
    NF4_DECODE_CHECK.write_text(json.dumps(result, indent=2, default=str))
    out_volume.commit()
    return result


@app.function(image=image, gpu=GPU, volumes=VOLUMES, secrets=[hf_secret],
              cpu=4.0, memory=32768, timeout=5400)
def nf4_decode_harvest() -> dict:
    """The single 500-prompt llama-nf4-decode shard. Refuses unless the GPU check passed.
    No retries: a failure is reported, not silently re-run."""
    out_volume.reload()
    check = json.loads(NF4_DECODE_CHECK.read_text()) if NF4_DECODE_CHECK.is_file() else {}
    if check.get("status") != "smoke_ok" or not check.get("placement"):
        raise RuntimeError("nf4_decode_gpu_check has not passed; run it first.")
    return _execute(NF4_DECODE_JOB, 0, dry_run=False)


# ============================================================
# ENTRYPOINTS
# ============================================================


def _n_shards(job_name):
    return 1 if JOBS[job_name]["subset"] else (QWEN_PROMPTS + SHARD_SIZE - 1) // SHARD_SIZE


@app.local_entrypoint()
def prepare():
    """Write the Qwen spec, check inputs, cache both models' weights."""
    info = _prepare_inputs.remote()
    if info["missing"]:
        print("MISSING on the a5-data volume:")
        for name in info["missing"]:
            print("  ", name)
        raise SystemExit("Fix the missing inputs, then run prepare again.")
    print("Inputs present. Qwen spec written to a5-data/code/qwen25-7b-promptpos.yaml")
    for model_id in (LLAMA_ID, QWEN_ID):
        print("caching", model_id, "->", _download_weights.remote(model_id))
    print("\nprepare OK")


def _print_smoke(name, result):
    if result.get("status") == "smoke_ok":
        print(f"\n  ok      {name}: {result['activation_count']} activations on "
              f"{result['prompt_count']} prompts "
              f"({result['activations_per_prompt_expected']}/prompt expected), "
              f"{result['gpu_gb_after_load']} GB on GPU after load, "
              f"quantization recorded {result['quantization_recorded']}, "
              f"{result['seconds_total']} s", flush=True)
        for t in result.get("offset_check", []):
            print(f"          prompt[-6] token {t['at_minus6']!r}   tail {t['tail']}",
                  flush=True)
    else:
        print(f"\n  FAILED  {name}\n{str(result.get('error'))[-900:]}", flush=True)


@app.local_entrypoint()
def smoke():
    """Dry-run jobs on 2 prompts; results print as each finishes; nothing is archived.

    Rerun only some jobs:  A5_SMOKE_JOBS=qwen-bf16-prompt modal run ...::smoke
    The marker is merged, so jobs not rerun keep their earlier result.
    """
    wanted = [n.strip() for n in os.environ.get("A5_SMOKE_JOBS", "").split(",") if n.strip()]
    unknown = [n for n in wanted if n not in JOBS]
    if unknown:
        raise SystemExit(f"Unknown job(s) {unknown}; choose from {LAUNCH_ORDER}")
    names = wanted or list(LAUNCH_ORDER)
    deadline_min = float(os.environ.get("A5_SMOKE_DEADLINE_MIN", "20"))
    print(f"Dry-running {len(names)} job(s) on {GPU}: {names}", flush=True)
    print(f"Waiting at most {deadline_min} min; anything still running then is FAILED.",
          flush=True)
    started = time.time()
    results = {}
    pending = {name: smoke_job.spawn(name) for name in names}
    deadline = started + deadline_min * 60
    while pending and time.time() < deadline:
        for name, call in list(pending.items()):
            try:
                result = call.get(timeout=2)
            except Exception as error:
                if "timeout" in type(error).__name__.lower() or isinstance(error, TimeoutError):
                    continue          # still running
                result = {"job": name, "status": "FAILED", "error": repr(error)[:900]}
            del pending[name]
            results[name] = result
            _print_smoke(name, result)
        time.sleep(1)
    for name in pending:
        results[name] = {"status": "FAILED", "job": name,
                         "error": f"still running after {deadline_min} min; "
                                  "check modal.com/apps for its logs"}
        print(f"\n  FAILED  {name}: still running after {deadline_min} min", flush=True)

    passed_now = [n for n in names if results[n].get("status") == "smoke_ok"]
    failed_now = {n: results[n].get("error") for n in names if n not in passed_now}
    old = _read_marker.remote() if wanted else {}
    passed = [n for n in LAUNCH_ORDER
              if n in passed_now or (n in old.get("passed", []) and n not in names)]
    failed = {n: e for n, e in {**old.get("failed", {}), **failed_now}.items()
              if n not in passed}
    payload = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "passed": passed, "failed": failed, "results": results}
    print("\nmarker:", _write_marker.remote(payload))
    print(f"\nSmoke wall clock: {(time.time() - started) / 60:.1f} min")
    print(f"PASSED: {passed}")
    print(f"FAILED: {list(failed)}")
    if failed:
        print("Failed jobs will NOT be launched.")


@app.local_entrypoint()
def launch():
    """Do not launch from here: jobs spawned from a `modal run` entrypoint are
    cancelled when the app stops, even with --detach. Deploy, then call launch_remote."""
    raise SystemExit(
        "Launch is two steps now:\n"
        "  !modal deploy /content/a5_modal_pack2.py\n"
        "  then, in Python:\n"
        "    import modal\n"
        "    print(modal.Function.from_name('a5-ws2-pack2', 'launch_remote').remote())")


@app.local_entrypoint()
def status():
    """Archives per job, against what each job should produce."""
    info = _status.remote()
    names = info["archives"]
    all_done = True
    for job_name in LAUNCH_ORDER:
        job = JOBS[job_name]
        have = sum(1 for n in names if archive_matches(n, job))
        want = _n_shards(job_name)
        all_done &= have == want
        print(f"  {job_name:<20} {have:>2} / {want:<2} archives   {job['config']}")
    if info["errors"]:
        print("\nNon-empty error logs (prompts failed):")
        for name, size in info["errors"]:
            print(f"  {name}  ({size:,} bytes)")
    else:
        print("\nNo error logs.")
    print("\nALL JOBS COMPLETE" if all_done else "\nNot complete yet (or some jobs not launched).")
