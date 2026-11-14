# =====================================================================
# PROJECT A-5 / WS2 - STEP 1: A9 NUMERICAL NOISE FLOOR (precision legs)
# Paste this whole thing into ONE new Colab cell and run it.
# Needs nothing installed beyond what preflight just reported.
# Runtime: about 8-12 minutes (downloads Qwen2.5-1.5B once, loads it 3x).
# =====================================================================
#
# WHAT THIS MEASURES, in plain words
# ----------------------------------
# Take one prompt. Do not change a single character of it. Run it through the
# model twice, and then again in a different numeric precision. The activation
# vectors come out slightly different every time, because floating-point
# arithmetic is not exact and the order of operations changes.
#
# That difference is the noise floor. It is the smallest displacement your
# instruments can see. In Week 4, A1 measures how far an adversarial rewrite
# moves the same activations. If a rewrite moves them by less than this floor,
# you have measured your own arithmetic, not a jailbreak.
#
# Three numbers come out:
#   bf16 run A vs bf16 run B   - run-to-run determinism (should be tiny)
#   fp32 vs bf16               - the cost of the precision you store at
#   (vLLM legs come later, under a matched torch version)
#
# WHY A SEPARATE COLLECTOR
# ------------------------
# Your harvest/collector.py casts every captured tensor to bf16 before storing
# it. That is correct for the harvest - it is your storage format - but it
# would destroy this measurement: an fp32 run and a bf16 run would both come
# back as bf16 and look identical. So this writes a second collector that keeps
# the compute precision, registered under a different name. Your production
# collector is not touched.
#
# WHAT SUCCESS LOOKS LIKE
# -----------------------
# A printed table of two comparisons with a median and a 95th-percentile
# cosine distance, and four files written to harvest/qa/a9/. The last line
# reads A9 PRECISION LEGS: DONE.

import gc
import json
import hashlib
import importlib.metadata as md
import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

print("=" * 68)
print("A9 NOISE FLOOR - PRECISION LEGS")
print("=" * 68)

# ---------------------------------------------------------------- paths
CANDIDATE_ROOTS = [
    Path("/content/gdrive/MyDrive/MYSSP/a5-artifacts"),
    Path("/content/drive/MyDrive/MYSSP/a5-artifacts"),
    Path("/content/gdrive/MyDrive/a5-artifacts"),
    Path("/content/drive/MyDrive/a5-artifacts"),
]
ROOT = next((p for p in CANDIDATE_ROOTS if p.is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("Project root not visible. Re-run the preflight cell.")

HARVEST = ROOT / "harvest"
SPECS = HARVEST / "specs"
OUT = HARVEST / "qa" / "a9"
OUT.mkdir(parents=True, exist_ok=True)

PILOT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_LAYERS = 28
N_PROMPTS = 32

print("Project root:", ROOT)
print("Output:", OUT)

# ------------------------------------------- the full-precision collector
FULLPREC = '''# SPDX-License-Identifier: Apache-2.0
"""Collector that preserves the compute dtype, for the A9 noise floor.

harvest/collector.py casts to bf16 on capture, which is right for the harvest
but hides the difference A9 measures. This one widens to float32 for storage
(bf16 -> float32 is lossless) and records the dtype it was handed.

Registered as "collector_fullprec"; the production "collector" is untouched.
"""

from threading import RLock

import torch

from undercurrent.core.probe import Probe, ProbeResult
from undercurrent.core import register_probe


@register_probe("collector_fullprec")
class FullPrecisionCollectorProbe(Probe):
    probe_kind = "single_shot"

    def __init__(self):
        super().__init__()

    def on_start(self, request_ctx):
        self._tensors = {}
        self._records = []
        self._lock = RLock()

    def on_activation(self, record):
        tensor = record.tensor

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        if not tensor.is_floating_point():
            raise TypeError(f"Expected floating-point tensor, got {tensor.dtype}")

        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"NaN or Inf detected at layer {record.layer}")

        key = (
            f"layer_{record.layer:02d}"
            f"_token_{record.token_pos}"
            f"_{'generated' if record.is_generated else 'prompt'}"
        )

        cpu_tensor = (
            tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .clone()
        )

        metadata = dict(record.metadata())
        metadata["source_dtype"] = str(tensor.dtype)
        metadata["storage_dtype"] = "torch.float32"
        metadata["tensor_key"] = key

        with self._lock:
            if key in self._tensors:
                raise ValueError(f"Duplicate activation key: {key}")
            self._tensors[key] = cpu_tensor
            self._records.append(metadata)

        return None

    def on_end(self, request_ctx):
        with self._lock:
            tensors = dict(self._tensors)
            records = list(self._records)

        return ProbeResult(
            request_id=self.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={"tensors": tensors, "records": records},
            metadata={
                "activation_count": len(tensors),
                "storage_dtype": "torch.float32",
            },
        )
'''

collector_path = HARVEST / "collector_fullprec.py"
if not collector_path.is_file() or collector_path.read_text(encoding="utf-8") != FULLPREC:
    collector_path.write_text(FULLPREC, encoding="utf-8")
    print("\nWrote:", collector_path)
else:
    print("\ncollector_fullprec.py already up to date.")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_module(collector_path, "a5_collector_fullprec")
print("Registered probe type: collector_fullprec")

# ------------------------------------------------------------- the spec
layers = ", ".join(str(i) for i in range(N_LAYERS))
SPEC_TEXT = (
    'version: "1"\n'
    "\n"
    "extraction_points:\n"
    "  - name: last_prefill\n"
    f"    layers: [{layers}]\n"
    "    tensor_type: residual_stream\n"
    '    position: "prompt[-1]"\n'
    "    probe_type: collector_fullprec\n"
    "    probe_kind: single_shot\n"
)
spec_path = SPECS / "qwen25-1.5b-a9.yaml"
SPECS.mkdir(parents=True, exist_ok=True)
if not spec_path.is_file() or spec_path.read_text(encoding="utf-8") != SPEC_TEXT:
    spec_path.write_text(SPEC_TEXT, encoding="utf-8")
    print("Wrote:", spec_path)
else:
    print("A9 spec already up to date.")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------- the prompt set
print("\nLoading and validating the pairs table through your own run.py ...")
a5_run = load_module(HARVEST / "run.py", "a5_run_for_a9")
all_prompts = a5_run.load_prompt_rows()


def stable_rank(value, salt="a9-floor"):
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


prompts = sorted(all_prompts, key=lambda p: stable_rank(p["prompt_id"]))[:N_PROMPTS]
print(f"\nA9 prompt set: {len(prompts)} prompts, chosen by hash of prompt_id.")
print("First three:", [p["prompt_id"] for p in prompts[:3]])

# --------------------------------------------------------- the captures
from transformers import AutoTokenizer
from undercurrent import ProbedModel

tokenizer = AutoTokenizer.from_pretrained(PILOT_MODEL)

# Render every prompt once, through your own template check, so all three runs
# see byte-identical input and any template problem surfaces before the GPU work.
rendered = []
for prompt in prompts:
    text, ids = a5_run.render_and_check(tokenizer, prompt["prompt_text"])
    rendered.append((prompt["prompt_id"], text, len(ids)))
print(f"Template check passed for all {len(rendered)} prompts.")
print("Token counts:", min(r[2] for r in rendered), "to", max(r[2] for r in rendered))


def capture(dtype, label):
    """One pass over the prompt set; returns {key: float32 numpy vector}."""
    print(f"\n--- capturing: {label} ---")
    captures = {}
    with ProbedModel.from_pretrained(
        PILOT_MODEL,
        spec=str(spec_path),
        backend="hf",
        device="cuda",
        tokenizer=tokenizer,
        dtype=dtype,
    ) as model:
        for i, (prompt_id, text, _n) in enumerate(rendered, 1):
            out = model.generate(text, max_new_tokens=1, temperature=0)
            for point_name, result in out.probe_results.items():
                for key, tensor in result.verdict["tensors"].items():
                    captures[f"{prompt_id}|{point_name}|{key}"] = (
                        tensor.to(dtype=torch.float32).numpy()
                    )
            if i % 8 == 0 or i == len(rendered):
                print(f"  {i}/{len(rendered)} prompts, {len(captures)} activations")
    gc.collect()
    torch.cuda.empty_cache()
    return captures


cap_bf16_a = capture(torch.bfloat16, "bf16, run A")
cap_bf16_b = capture(torch.bfloat16, "bf16, run B (identical input)")
cap_fp32 = capture(torch.float32, "fp32")

# ------------------------------------------------------- the comparison
def compare(a, b):
    if set(a) != set(b):
        raise ValueError(
            f"Capture keys differ: {len(set(a) - set(b))} only in A, "
            f"{len(set(b) - set(a))} only in B."
        )
    rows = []
    for key in sorted(a):
        x = np.asarray(a[key], dtype="float64")
        y = np.asarray(b[key], dtype="float64")
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch at {key}: {x.shape} vs {y.shape}")
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        denom = nx * ny
        cos = float(x @ y / denom) if denom > 0 else float("nan")
        match = re.search(r"layer_(\d+)_", key)
        rows.append(
            {
                "tensor_key": key,
                "layer": int(match.group(1)) if match else None,
                "cosine_distance": 1.0 - cos,
                "relative_l2": float(np.linalg.norm(x - y) / nx) if nx > 0 else float("nan"),
                "norm_a": nx,
                "norm_b": ny,
            }
        )
    return rows


def stats(values):
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


comparisons = {
    "bf16_rerun": compare(cap_bf16_a, cap_bf16_b),
    "bf16_vs_fp32": compare(cap_fp32, cap_bf16_a),  # fp32 is the reference
}

frames = []
summary_rows = []
print("\n" + "=" * 68)
print("RESULTS")
print("=" * 68)

for name, rows in comparisons.items():
    frame = pd.DataFrame(rows)
    frame["comparison"] = name
    frames.append(frame)

    cos = stats(frame["cosine_distance"])
    l2 = stats(frame["relative_l2"])
    summary_rows.append(
        {
            "comparison": name,
            "n_activations": cos.get("n", 0),
            "cos_dist_median": cos.get("median"),
            "cos_dist_p95": cos.get("p95"),
            "cos_dist_max": cos.get("max"),
            "rel_l2_median": l2.get("median"),
            "rel_l2_p95": l2.get("p95"),
            "rel_l2_max": l2.get("max"),
        }
    )
    print(f"\n{name}  (n = {cos.get('n')})")
    print(f"  cosine distance   median {cos.get('median'):.3e}   p95 {cos.get('p95'):.3e}"
          f"   max {cos.get('max'):.3e}")
    print(f"  relative L2       median {l2.get('median'):.3e}   p95 {l2.get('p95'):.3e}")

per_activation = pd.concat(frames, ignore_index=True)
per_layer = (
    per_activation.groupby(["comparison", "layer"])[["cosine_distance", "relative_l2"]]
    .median()
    .reset_index()
)
summary = pd.DataFrame(summary_rows)

per_activation.to_csv(OUT / "a9_per_activation.csv", index=False)
per_layer.to_csv(OUT / "a9_per_layer.csv", index=False)
summary.to_csv(OUT / "a9_summary.csv", index=False)

meta = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "analysis": "A9 numerical noise floor, precision legs",
    "model_id": PILOT_MODEL,
    "backend": "hf",
    "concurrency": 1,
    "n_prompts": len(prompts),
    "prompt_ids": [p["prompt_id"] for p in prompts],
    "prompt_selection": "sha256('a9-floor:' + prompt_id), ascending, first 32",
    "n_layers": N_LAYERS,
    "extraction_point": "last_prefill at prompt[-1], layers 0..27",
    "spec_sha256": sha256_file(spec_path),
    "collector": "collector_fullprec (preserves compute dtype, stores float32)",
    "legs_completed": sorted(comparisons),
    "legs_outstanding": [
        "hf_vs_vllm (needs vLLM installed, which pins torch - run under matched torch)",
        "concurrency_1_vs_64 (vLLM only)",
    ],
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "undercurrent": md.version("undercurrent"),
    "transformers": md.version("transformers"),
    "note": (
        "The production collector casts to bf16 on capture and cannot see a "
        "precision difference; collector_fullprec was used instead. torch here is "
        f"{torch.__version__}; the project compatibility table records 2.13.0 as "
        "GPU-verified. The vLLM legs must be measured under one torch version."
    ),
}
(OUT / "a9_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("\n" + "=" * 68)
print("FILES WRITTEN")
print("=" * 68)
for name in ["a9_per_activation.csv", "a9_per_layer.csv", "a9_summary.csv", "a9_meta.json"]:
    path = OUT / name
    print(f"  {path}  ({path.stat().st_size:,} bytes)")

floor = summary.loc[summary["comparison"] == "bf16_vs_fp32", "cos_dist_p95"].iloc[0]
print("\nThe number that matters for A1 and figure F1:")
print(f"  noise floor (bf16 vs fp32, 95th percentile cosine distance) = {floor:.3e}")
print("\nIn Week 4, any per-tactic displacement A1 reports must be compared")
print("against this. A displacement near it is not a finding.")

print("\n" + "=" * 68)
print("A9 PRECISION LEGS: DONE")
print("=" * 68)
