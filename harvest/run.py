
from pathlib import Path
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import traceback
import tempfile
from datetime import datetime, timezone

import pandas as pd
import torch
import undercurrent
import yaml

from safetensors.torch import save_file, load_file
from safetensors import safe_open


# ============================================================
# PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path("/content/gdrive/MyDrive/MYSSP/a5-artifacts")
HARVEST = ROOT / "harvest"

CSV_PATH = (
    ROOT / "Dennis" / "data table" /
    "pairs_table_v1_final-2.csv"
)

PAIR_MAP_PATH = (
    ROOT / "Dennis" / "Activations" /
    "Full harvest Qwen 2.5-7B activations" /
    "pair_maps" / "pair_map__d240ec65.parquet"
)

SPEC_PATH = HARVEST / "specs" / "qwen25-1.5b.yaml"
COLLECTOR_PATH = HARVEST / "collector.py"
ARCHIVE_MODULE_PATH = HARVEST / "drive_archive.py"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_ALIAS = "qwen25-1p5b"
DATASET_NAME = "sorry-bench-paired"

# Pilot and full runs must have different configuration names.
PILOT_CONFIG = "hf-fp32-bf16-pilot8"
FULL_CONFIG = "hf-fp32-bf16-full"

MAX_NEW_TOKENS = 256
PILOT_PROMPT_LIMIT = 8
PRODUCTION_SHARD_SIZE = 500

INFERENCE_DTYPE = "torch.float32"
STORAGE_DTYPE = "torch.bfloat16"

LOCAL_OUTPUT = Path("/content/harvest")
LOG_DIR = HARVEST / "logs"
MANIFEST_DIR = HARVEST / "manifests"

for folder in [LOCAL_OUTPUT, LOG_DIR, MANIFEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# GENERAL UTILITIES
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_key(value):
    value = str(value)
    readable = re.sub(r"[^A-Za-z0-9_.-]", "_", value)[:70]
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{readable}_{suffix}"


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


# ============================================================
# LOAD SPECIFICATION
# ============================================================

def load_spec():
    if not SPEC_PATH.is_file():
        raise FileNotFoundError(SPEC_PATH)

    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    spec = yaml.safe_load(spec_text)

    if not isinstance(spec, dict):
        raise ValueError("YAML specification is not a mapping.")

    points = spec.get("extraction_points")

    if not isinstance(points, list) or not points:
        raise ValueError("No extraction_points found in YAML.")

    names = [p.get("name") for p in points]

    if any(not name for name in names):
        raise ValueError("Every extraction point needs a name.")

    if len(names) != len(set(names)):
        raise ValueError("Duplicate extraction point names.")

    return spec_text, spec, sha256_file(SPEC_PATH)


# ============================================================
# LOAD AND VALIDATE PROMPTS
# ============================================================

EXPECTED_INCOMPLETE_PAIR_IDS = {
    "sb_question_157",
    "sb_question_160",
    "sb_question_166",
    "sb_question_404",
}



def load_prompt_rows():
    pairs = pd.read_csv(CSV_PATH, dtype={"pair_id": str})
    pair_map = pd.read_parquet(PAIR_MAP_PATH)

    required_csv = {
        "pair_id", "source", "source_row_id", "split",
        "holdout_group", "intent", "category", "tactic",
        "vanilla_prompt", "adversarial_prompt",
        "vanilla_token_count", "adversarial_token_count",
    }
    required_map = {"pair_id", "role", "prompt_id"}

    if required_csv - set(pairs.columns):
        raise ValueError(f"Missing CSV columns: {required_csv - set(pairs.columns)}")
    if required_map - set(pair_map.columns):
        raise ValueError(f"Missing pair-map columns: {required_map - set(pair_map.columns)}")

    pairs["pair_id"] = pairs["pair_id"].astype(str)
    pair_map["pair_id"] = pair_map["pair_id"].astype(str)
    pair_map["prompt_id"] = pair_map["prompt_id"].astype(str)
    pair_map["role"] = (
        pair_map["role"].astype(str).str.lower()
        .replace({"adv": "adversarial"})
    )

    if pairs["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair IDs in source CSV.")
    if pair_map.duplicated(["pair_id", "role"]).any():
        raise ValueError("Duplicate pair_id/role mappings.")

    unexpected_roles = set(pair_map["role"]) - {"vanilla", "adversarial"}
    if unexpected_roles:
        raise ValueError(f"Unexpected roles: {sorted(unexpected_roles)}")

    expected_incomplete = {
        "sb_question_157", "sb_question_160",
        "sb_question_166", "sb_question_404",
    }

    incomplete = set(
        pairs.loc[
            pairs["vanilla_prompt"].notna()
            & pairs["adversarial_prompt"].isna(),
            "pair_id",
        ]
    )

    if incomplete != expected_incomplete:
        raise ValueError(f"Unexpected incomplete pairs: {sorted(incomplete)}")

    complete = pairs[
        pairs["vanilla_prompt"].notna()
        & pairs["adversarial_prompt"].notna()
    ].copy()

    if len(complete) != 16515:
        raise ValueError(f"Expected 16,515 pairs; found {len(complete)}")

    complete_ids = set(complete["pair_id"])

    unknown_ids = set(pair_map["pair_id"]) - set(pairs["pair_id"])
    if unknown_ids:
        raise ValueError(f"Unknown pair IDs: {sorted(unknown_ids)[:10]}")

    mapping = pair_map[pair_map["pair_id"].isin(complete_ids)].copy()

    if len(mapping) != 33030:
        raise ValueError(f"Expected 33,030 mappings; found {len(mapping)}")

    role_groups = mapping.groupby("pair_id")["role"].agg(list)
    invalid = [
        (pid, role_groups.get(pid, []))
        for pid in complete_ids
        if sorted(role_groups.get(pid, [])) != ["adversarial", "vanilla"]
    ]
    if invalid:
        raise ValueError(f"Invalid pair-role mappings: {invalid[:10]}")

    joined = mapping.merge(
        complete, on="pair_id", how="left", validate="many_to_one"
    )

    metadata_fields = [
        "pair_id", "role", "source", "source_row_id",
        "split", "holdout_group", "intent", "category", "tactic",
    ]

    def clean_value(value):
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value

    grouped = {}

    for row in joined.to_dict(orient="records"):
        prompt_id = row["prompt_id"]

        if row["role"] == "vanilla":
            prompt_text = row["vanilla_prompt"]
            token_count = row["vanilla_token_count"]
        else:
            prompt_text = row["adversarial_prompt"]
            token_count = row["adversarial_token_count"]

        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(f"Empty prompt: {prompt_id}")

        token_count = clean_value(token_count)

        link = {
            field: clean_value(row[field])
            for field in metadata_fields
        }

        if prompt_id not in grouped:
            grouped[prompt_id] = {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "source_token_count": token_count,
                "pair_links": [],
            }
        else:
            existing = grouped[prompt_id]
            if existing["prompt_text"] != prompt_text:
                raise ValueError(f"Prompt ID maps to multiple texts: {prompt_id}")
            if existing["source_token_count"] != token_count:
                raise ValueError(f"Prompt ID maps to multiple token counts: {prompt_id}")

        grouped[prompt_id]["pair_links"].append(link)

    prompts = []

    for prompt_id in sorted(grouped):
        record = grouped[prompt_id]
        links = sorted(
            record.pop("pair_links"),
            key=lambda x: (x["pair_id"], x["role"]),
        )
        record["pair_link_count"] = len(links)
        record["pair_links_json"] = json.dumps(
            links, ensure_ascii=False, sort_keys=True
        )
        prompts.append(record)

    total_links = sum(p["pair_link_count"] for p in prompts)

    if len(prompts) != 18500:
        raise ValueError(f"Expected 18,500 unique prompts; found {len(prompts)}")
    if total_links != 33030:
        raise ValueError(f"Expected 33,030 links; found {total_links}")

    print("Prompt loader validation passed.")
    print("Complete pairs:", len(complete))
    print("Unique prompts:", len(prompts))
    print("Preserved pair-role mappings:", total_links)
    print("Excluded incomplete pairs:", sorted(incomplete))

    return prompts



# ============================================================
# TEMPLATE AND CONTEXT VALIDATION
# ============================================================


def render_and_check(tokenizer, prompt_text):
    """
    Render the exact model chat template and verify that ordinary
    tokenization reproduces the native chat-template token IDs.

    Remove a literal leading BOS before ordinary tokenization when
    present, because the HF adapter adds special tokens by default.
    """

    messages = [{"role": "user", "content": prompt_text}]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    direct_output = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    # BatchEncoding is mapping-like: extract input_ids explicitly.
    if hasattr(direct_output, "keys") and "input_ids" in direct_output:
        direct_ids = direct_output["input_ids"]
    else:
        direct_ids = direct_output

    if isinstance(direct_ids, torch.Tensor):
        direct_ids = direct_ids.tolist()

    if direct_ids and isinstance(direct_ids[0], list):
        direct_ids = direct_ids[0]

    direct_ids = [int(token_id) for token_id in direct_ids]

    # Prevent accidental double-BOS insertion.
    bos = tokenizer.bos_token
    if bos and rendered.startswith(bos):
        rendered = rendered[len(bos):]

    rendered_ids = tokenizer(
        rendered,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    if isinstance(rendered_ids, torch.Tensor):
        rendered_ids = rendered_ids.tolist()

    if rendered_ids and isinstance(rendered_ids[0], list):
        rendered_ids = rendered_ids[0]

    rendered_ids = [int(token_id) for token_id in rendered_ids]

    if rendered_ids != direct_ids:
        limit = min(len(rendered_ids), len(direct_ids))
        mismatch = next(
            (i for i in range(limit)
             if rendered_ids[i] != direct_ids[i]),
            limit,
        )

        raise ValueError(
            "Chat-template token IDs do not match after BOS handling. "
            f"Rendered count={len(rendered_ids)}, "
            f"native count={len(direct_ids)}, "
            f"first mismatch={mismatch}."
        )

    return rendered, direct_ids



def get_context_limit(model_config, tokenizer):
    candidates = [
        getattr(model_config, "max_position_embeddings", None),
        getattr(model_config, "max_sequence_length", None),
        getattr(tokenizer, "model_max_length", None),
    ]

    for value in candidates:
        if isinstance(value, int) and 0 < value < 10**7:
            return value

    raise ValueError("Could not establish a reliable context limit.")


def check_prompt_length(token_count, context_limit):
    requested_total = token_count + MAX_NEW_TOKENS

    if requested_total > context_limit:
        raise ValueError(
            f"Prompt plus generation exceeds context limit: "
            f"{token_count} + {MAX_NEW_TOKENS} > {context_limit}. "
            "No truncation was applied."
        )


# ============================================================
# DRIVE ARCHIVE API AND VERIFICATION
# ============================================================

def load_archive_module():
    if not ARCHIVE_MODULE_PATH.is_file():
        raise FileNotFoundError(ARCHIVE_MODULE_PATH)

    spec = importlib.util.spec_from_file_location(
        "a5_drive_archive",
        ARCHIVE_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "setup",
        "archive_name",
        "already_archived",
        "archive_to_drive",
        "extract_one",
    ]

    missing = [name for name in required if not hasattr(module, name)]

    if missing:
        raise RuntimeError(
            f"drive_archive.py missing functions: {missing}"
        )

    return module


def shard_archive_name(base_name, shard_name):
    # archive_name() returns "...<hash>.tpxz".
    # Insert the shard ID before the extension.
    if not base_name.endswith(".tpxz"):
        raise ValueError(f"Unexpected archive filename: {base_name}")

    return base_name[:-5] + f"__{shard_name}.tpxz"


def verify_drive_archive(archive_module, name):
    if not archive_module.already_archived(name):
        return False

    drive_root = archive_module.DRIVE_ROOT
    archive_path = drive_root / name
    manifest_path = drive_root / f"{name}.json"

    if not archive_path.is_file() or not manifest_path.is_file():
        return False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = manifest.get("sha256")

    if not expected_sha:
        raise RuntimeError(
            f"Archive manifest has no SHA-256: {name}"
        )

    actual_sha = sha256_file(archive_path)

    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Drive archive SHA-256 mismatch: {name}"
        )

    return True


def load_archived_shard_manifest(archive, archive_name, shard_name):
    """Read only the internal manifest from a verified Drive archive."""
    with tempfile.TemporaryDirectory(prefix="a5_resume_manifest_") as tmp:
        archive.extract_one(
            archive_name,
            f"{shard_name}/manifest.json",
            tmp,
        )

        manifest_path = Path(tmp) / shard_name / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Could not extract internal manifest from {archive_name}."
            )

        return json.loads(
            manifest_path.read_text(encoding="utf-8")
        )


def reconcile_completed_shard(
    run_manifest,
    run_manifest_path,
    shard_name,
    archive_name,
    shard_manifest,
):
    """Idempotently record a verified completed shard in the run manifest."""
    if shard_manifest.get("status") != "complete":
        raise RuntimeError(
            f"{shard_name} does not have a complete shard manifest."
        )

    for field in ("model_revision", "spec_hash", "template_hash"):
        if shard_manifest.get(field) != run_manifest.get(field):
            raise RuntimeError(
                f"{shard_name} manifest differs from the run in {field}."
            )

    entries = run_manifest.setdefault("completed_shards", [])
    existing = next(
        (item for item in entries if item.get("shard") == shard_name),
        None,
    )

    completion = {
        "shard": shard_name,
        "archive": archive_name,
        "prompt_count": int(shard_manifest["prompt_count"]),
        "activation_count": int(shard_manifest["activation_count"]),
    }

    if existing is None:
        completion["completed_at"] = (
            shard_manifest.get("created_at") or utc_now()
        )
        entries.append(completion)
    else:
        completed_at = existing.get("completed_at")
        existing.update(completion)
        existing["completed_at"] = (
            completed_at or shard_manifest.get("created_at") or utc_now()
        )

    entries.sort(key=lambda item: item["shard"])
    atomic_json(run_manifest_path, run_manifest)



# ============================================================
# PROMPT EXTRACTION
# ============================================================

def extract_prompt(
    probed_model,
    tokenizer,
    prompt,
    context_limit,
    expected_hidden_size,
    template_hash,
):
    rendered, input_ids = render_and_check(
        tokenizer,
        prompt["prompt_text"],
    )

    token_count = len(input_ids)
    check_prompt_length(token_count, context_limit)

    # HF backend: one prompt per call.
    output = probed_model.generate(
        rendered,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )

    if getattr(output, "aborted", False):
        raise RuntimeError(
            f"Undercurrent aborted {prompt['prompt_id']}"
        )

    tensors = {}
    index_rows = []
    fired_points = []

    for point_name, result in output.probe_results.items():
        verdict = result.verdict

        if not isinstance(verdict, dict):
            raise TypeError(
                f"Unexpected verdict for {point_name}: {type(verdict)}"
            )

        captured = verdict.get("tensors", {})
        records = verdict.get("records", [])

        if len(captured) != len(records):
            raise ValueError(
                f"Tensor/record count mismatch at {point_name}"
            )

        if captured:
            fired_points.append(point_name)

        for record in records:
            original_key = record["tensor_key"]

            if original_key not in captured:
                raise KeyError(original_key)

            tensor = captured[original_key]

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Not a tensor: {original_key}")

            expected_shape = (expected_hidden_size,)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Unexpected activation shape at {original_key}: "
                    f"expected {expected_shape}, got {tuple(tensor.shape)}"
                )

            if tensor.dtype != torch.bfloat16:
                raise TypeError(
                    f"Expected BF16, got {tensor.dtype} at {original_key}"
                )

            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f"NaN/Inf in {prompt['prompt_id']} / {original_key}"
                )

            storage_key = (
                stable_key(prompt["prompt_id"])
                + "__"
                + stable_key(point_name)
                + "__"
                + stable_key(original_key)
            )

            if storage_key in tensors:
                raise ValueError(f"Duplicate tensor key: {storage_key}")

            tensors[storage_key] = tensor.detach().cpu().contiguous()

            index_rows.append({
                **{
                    k: v for k, v in prompt.items()
                    if k != "prompt_text"
                },
                "tensor_key": storage_key,
                "original_tensor_key": original_key,
                "extraction_point": point_name,
                "layer": int(record["layer"]),
                "token_pos": int(record["token_pos"]),
                "is_generated": bool(record["is_generated"]),
                "tensor_shape": str(tuple(tensor.shape)),
                "source_dtype": record.get("source_dtype"),
                "storage_dtype": STORAGE_DTYPE,
                "token_count": token_count,
                "was_truncated": False,
                "backend": "hf",
                "concurrency": 1,
                "inference_dtype": INFERENCE_DTYPE,
                "quantization": "none",
                "model_id": MODEL_ID,
                "template_hash": template_hash,
                "undercurrent_version": importlib.metadata.version(
                    "undercurrent"
                ),
            })

    if not tensors:
        raise RuntimeError(
            f"No activations captured for {prompt['prompt_id']}"
        )

    # Both prompt-side extraction points must have fired.
    required_prompt_points = {"last_prefill", "last_instruction"}

    if not required_prompt_points.issubset(set(fired_points)):
        raise RuntimeError(
            f"Missing prompt extraction point(s) for "
            f"{prompt['prompt_id']}: "
            f"{required_prompt_points - set(fired_points)}"
        )

    return {
        "tensors": tensors,
        "index_rows": index_rows,
        "token_count": token_count,
        "input_ids": input_ids,
        "fired_points": fired_points,
        "rendered_prompt_hash": sha256_text(rendered),
    }


# ============================================================
# SHARD WRITING AND LOCAL VERIFICATION
# ============================================================

def write_completed_shard(
    shard_dir,
    shard_name,
    completed,
    model_revision,
    spec_hash,
    template_hash,
    config_name,
):
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    index_rows = []
    prompt_records = []

    for item in completed:
        prompt = item["prompt"]
        result = item["result"]

        for key, tensor in result["tensors"].items():
            if key in tensors:
                raise ValueError(f"Duplicate shard key: {key}")
            tensors[key] = tensor

        index_rows.extend(result["index_rows"])

        prompt_records.append({
            "prompt_id": prompt["prompt_id"],
            "pair_links_json": prompt["pair_links_json"],
            "pair_link_count": int(prompt["pair_link_count"]),
            "token_count": result["token_count"],
            "input_ids": result["input_ids"],
            "was_truncated": False,
            "fired_points": result["fired_points"],
            "rendered_prompt_hash": result["rendered_prompt_hash"],
        })

    if not tensors:
        raise ValueError("Cannot write an empty shard.")

    safetensors_path = shard_dir / "activations.safetensors"
    parquet_path = shard_dir / "index.parquet"
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    save_file(
        tensors,
        str(safetensors_path),
        metadata={
            "model_id": MODEL_ID,
            "model_revision": model_revision,
            "backend": "hf",
            "storage_dtype": "bfloat16",
            "spec_hash": spec_hash,
            "template_hash": template_hash,
            "gpu_name": gpu_name,
        },
    )

    pd.DataFrame(index_rows).to_parquet(
        parquet_path,
        index=False,
    )

    # Reopen the outputs and verify all keys.
    reopened = load_file(str(safetensors_path), device="cpu")
    reopened_index = pd.read_parquet(parquet_path)

    if set(reopened) != set(tensors):
        raise RuntimeError("Safetensors keys changed after saving.")

    if set(reopened_index["tensor_key"]) != set(reopened):
        raise RuntimeError("Parquet and Safetensors keys do not match.")

    if len(reopened_index) != len(reopened):
        raise RuntimeError("Parquet row count does not match tensor count.")

    for key, tensor in reopened.items():
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"Saved tensor is not BF16: {key}")

        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Saved tensor is non-finite: {key}")

    manifest = {
        "status": "complete",
        "shard": shard_name,
        "created_at": utc_now(),
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "dataset": DATASET_NAME,
        "config": config_name,
        "spec_hash": spec_hash,
        "template_hash": template_hash,
        "backend": "hf",
        "concurrency": 1,
        "gpu_name": gpu_name,
        "inference_dtype": INFERENCE_DTYPE,
        "storage_dtype": STORAGE_DTYPE,
        "quantization": "none",
        "undercurrent_version": importlib.metadata.version(
            "undercurrent"
        ),
        "python_version": platform.python_version(),
        "prompt_count": len(completed),
        "activation_count": len(tensors),
        "index_rows": len(index_rows),
        "prompts": prompt_records,
        "safetensors_sha256": sha256_file(safetensors_path),
        "index_sha256": sha256_file(parquet_path),
    }

    atomic_json(shard_dir / "manifest.json", manifest)

    return manifest


def verify_local_shard(
    shard_dir,
    expected_spec_hash,
    expected_template_hash,
    expected_revision,
):
    shard_dir = Path(shard_dir)
    manifest_path = shard_dir / "manifest.json"
    tensor_path = shard_dir / "activations.safetensors"
    index_path = shard_dir / "index.parquet"

    if not all(p.is_file() for p in [
        manifest_path,
        tensor_path,
        index_path,
    ]):
        return False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("status") != "complete":
        return False

    if manifest.get("spec_hash") != expected_spec_hash:
        return False

    if manifest.get("template_hash") != expected_template_hash:
        return False

    if manifest.get("model_revision") != expected_revision:
        return False

    if sha256_file(tensor_path) != manifest.get("safetensors_sha256"):
        raise RuntimeError(f"Local Safetensors hash mismatch: {shard_dir}")

    if sha256_file(index_path) != manifest.get("index_sha256"):
        raise RuntimeError(f"Local Parquet hash mismatch: {shard_dir}")

    with safe_open(
        str(tensor_path),
        framework="pt",
        device="cpu",
    ) as f:
        keys = list(f.keys())

    index = pd.read_parquet(index_path)

    if len(keys) != manifest["activation_count"]:
        raise RuntimeError("Local tensor count does not match manifest.")

    if set(keys) != set(index["tensor_key"]):
        raise RuntimeError("Local Parquet/Safetensors keys differ.")

    return True


# ============================================================
# FAILURE LOGGING
# ============================================================

def log_failure(error_path, prompt, error):
    record = {
        "timestamp": utc_now(),
        "prompt_id": prompt["prompt_id"],
        "pair_links_json": prompt["pair_links_json"],
        "pair_link_count": int(prompt["pair_link_count"]),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }

    with open(error_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# MAIN HARVEST FUNCTION
# ============================================================

def run_harvest(
    *,
    probed_model,
    tokenizer,
    model_config,
    model_revision,
    pilot=True,
):
    if importlib.metadata.version("undercurrent") != "0.1.0":
        raise RuntimeError("Expected undercurrent==0.1.0.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    if not COLLECTOR_PATH.is_file():
        raise FileNotFoundError(COLLECTOR_PATH)

    spec_text, parsed_spec, spec_hash = load_spec()
    prompts = load_prompt_rows()

    context_limit = get_context_limit(model_config, tokenizer)

    expected_hidden_size = int(
        getattr(model_config, "hidden_size", 0) or 0
    )
    if expected_hidden_size <= 0:
        raise RuntimeError(
            "Could not determine a valid model hidden_size."
        )

    template_hash = sha256_text(
        str(getattr(tokenizer, "chat_template", ""))
    )

    if pilot:
        config_name = PILOT_CONFIG
        prompts = prompts[:PILOT_PROMPT_LIMIT]
        shard_size = PILOT_PROMPT_LIMIT
    else:
        config_name = FULL_CONFIG
        shard_size = PRODUCTION_SHARD_SIZE

    run_id = (
        f"{MODEL_ALIAS}__{DATASET_NAME}__{config_name}__"
        f"{spec_hash[:8]}"
    )

    out_dir = LOCAL_OUTPUT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    error_path = LOG_DIR / f"{run_id}_errors.jsonl"
    run_manifest_path = MANIFEST_DIR / f"{run_id}.json"

    archive = load_archive_module()
    archive.setup(mount=False)

    base_archive_name = archive.archive_name(
        MODEL_ALIAS,
        DATASET_NAME,
        config_name,
        spec_hash,
    )

    run_manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "dataset": DATASET_NAME,
        "config": config_name,
        "spec_hash": spec_hash,
        "template_hash": template_hash,
        "pilot": pilot,
        "pilot_prompt_limit": PILOT_PROMPT_LIMIT if pilot else None,
        "shard_size": shard_size,
        "completed_shards": [],
    }

    if run_manifest_path.exists():
        previous = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )

        for field in [
            "model_revision",
            "spec_hash",
            "template_hash",
            "config",
        ]:
            if previous.get(field) != run_manifest.get(field):
                raise RuntimeError(
                    f"Existing run manifest differs in {field}."
                )

        run_manifest = previous

    total = len(prompts)
    shard_count = (total + shard_size - 1) // shard_size

    print("Model:", MODEL_ID)
    print("Revision:", model_revision)
    print("Prompts in this run:", total)
    print("Shard size:", shard_size)
    print("Number of shards:", shard_count)
    print("Context limit:", context_limit)
    print("Config:", config_name)
    print("Spec SHA256:", spec_hash)
    print("Template SHA256:", template_hash)

    for shard_number in range(shard_count):
        start = shard_number * shard_size
        end = min(start + shard_size, total)

        shard_name = f"shard_{shard_number:05d}"
        final_dir = out_dir / shard_name
        partial_dir = out_dir / f"{shard_name}.partial"

        drive_archive_name = shard_archive_name(
            base_archive_name,
            shard_name,
        )

        # A valid Drive archive is the authoritative completion marker.
        if verify_drive_archive(archive, drive_archive_name):
            print("Skipping verified Drive archive:", drive_archive_name)

            if verify_local_shard(
                final_dir,
                spec_hash,
                template_hash,
                model_revision,
            ):
                shard_manifest = json.loads(
                    (final_dir / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
            else:
                print(
                    "Local shard is absent or incomplete; "
                    "reading its manifest from the verified archive."
                )
                shard_manifest = load_archived_shard_manifest(
                    archive,
                    drive_archive_name,
                    shard_name,
                )

            reconcile_completed_shard(
                run_manifest,
                run_manifest_path,
                shard_name,
                drive_archive_name,
                shard_manifest,
            )

            continue

        # If local shard exists, validate it before archiving.
        if final_dir.exists():
            if not verify_local_shard(
                final_dir,
                spec_hash,
                template_hash,
                model_revision,
            ):
                raise RuntimeError(
                    f"Existing local shard is not valid: {final_dir}"
                )

            print("Local shard verified; uploading archive:", shard_name)

            archive.archive_to_drive(
                str(final_dir),
                drive_archive_name,
            )

            if not verify_drive_archive(archive, drive_archive_name):
                raise RuntimeError(
                    f"Drive archive verification failed: {drive_archive_name}"
                )

            shard_manifest = json.loads(
                (final_dir / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            reconcile_completed_shard(
                run_manifest,
                run_manifest_path,
                shard_name,
                drive_archive_name,
                shard_manifest,
            )

            continue

        if partial_dir.exists():
            # Partial output has no completion manifest and is not reusable.
            import shutil
            shutil.rmtree(partial_dir)

        partial_dir.mkdir(parents=True)

        shard_prompts = prompts[start:end]
        completed = []

        print(
            f"\nProcessing {shard_name}: "
            f"{len(shard_prompts)} prompts"
        )

        try:
            for prompt in shard_prompts:
                print(
                    "Extracting:",
                    prompt["prompt_id"],
                    prompt["pair_link_count"],
                    "pair links",
                )

                try:
                    result = extract_prompt(
                        probed_model,
                        tokenizer,
                        prompt,
                        context_limit,
                        expected_hidden_size,
                        template_hash,
                    )

                    completed.append({
                        "prompt": prompt,
                        "result": result,
                    })

                except Exception as error:
                    log_failure(error_path, prompt, error)
                    raise

            manifest = write_completed_shard(
                partial_dir,
                shard_name,
                completed,
                model_revision,
                spec_hash,
                template_hash,
                config_name,
            )

            # Atomic rename: final directory means local completion.
            partial_dir.rename(final_dir)

            # Archive the complete directory. The .tpxz suffix stays last.
            archive.archive_to_drive(
                str(final_dir),
                drive_archive_name,
            )

            if not verify_drive_archive(archive, drive_archive_name):
                raise RuntimeError(
                    f"Drive archive failed verification: {drive_archive_name}"
                )

            reconcile_completed_shard(
                run_manifest,
                run_manifest_path,
                shard_name,
                drive_archive_name,
                manifest,
            )

            print(
                f"Completed {shard_name}: "
                f"{manifest['prompt_count']} prompts, "
                f"{manifest['activation_count']} activations."
            )

        except Exception:
            # Keep the partial directory for inspection, but it is never
            # treated as completed by the resume logic.
            print("Shard failed. Partial output retained at:", partial_dir)
            raise

    print("\nHARVEST RUN FINISHED")
    print("Run manifest:", run_manifest_path)
    print("Failure log:", error_path)
    print("Output directory:", out_dir)

    return {
        "run_id": run_id,
        "prompt_count": total,
        "shard_count": shard_count,
        "output_dir": str(out_dir),
        "manifest_path": str(run_manifest_path),
    }
