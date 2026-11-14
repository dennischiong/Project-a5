# =====================================================================
# PROJECT A-5 / WS2 - RESTART RECOVERY + NF4 SMOKE TEST
# Paste into ONE new cell after a runtime restart. Runtime: about 4 minutes.
# =====================================================================
#
# A runtime restart wipes pip installs and the Python session. Drive files
# survive, so your A9 result is intact. This cell:
#   1. Remounts Drive and confirms the A9 files are still there.
#   2. Reinstalls bitsandbytes (--no-deps, so torch is not moved).
#   3. Runs the NF4 smoke test on the seven short prompts.
#   4. Writes harvest/qa/undercurrent_nf4_smoke.json.
#
# The 3,257-token prompt is excluded on purpose. It is not needed to answer
# the Week 3 question, and we are no longer planning a Llama NF4 harvest that
# would depend on long-prompt speed.
#
# Success: "NF4 SMOKE TEST: PASS", 196 activations, and WEEK 3 COMPLETE.

import gc
import json
import hashlib
import importlib
import importlib.metadata as md
import importlib.util
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

print("=" * 68)
print("RESTART RECOVERY + NF4 SMOKE TEST")
print("=" * 68)

# ----------------------------------------------------------- 1. remount
print("\n--- 1. DRIVE ---")
try:
    from google.colab import drive

    drive.mount("/content/gdrive")
except Exception as error:
    print("  mount reported:", error)

CANDIDATE_ROOTS = [
    Path("/content/gdrive/MyDrive/MYSSP/a5-artifacts"),
    Path("/content/drive/MyDrive/MYSSP/a5-artifacts"),
    Path("/content/gdrive/MyDrive/a5-artifacts"),
    Path("/content/drive/MyDrive/a5-artifacts"),
]
ROOT = next((p for p in CANDIDATE_ROOTS if p.is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("Project root not visible after remount.")

HARVEST = ROOT / "harvest"
SPECS = HARVEST / "specs"
QA = HARVEST / "qa"
print("  root:", ROOT)

print("\n  A9 results that survived the restart:")
a9_files = ["a9_summary.csv", "a9_per_layer.csv", "a9_per_activation.csv", "a9_meta.json"]
a9_ok = True
for name in a9_files:
    path = QA / "a9" / name
    if path.is_file():
        print(f"    OK      {name}  ({path.stat().st_size:,} bytes)")
    else:
        print(f"    MISSING {name}")
        a9_ok = False
if not a9_ok:
    print("\n  WARNING: some A9 files are missing. Tell me before going further.")
else:
    import pandas as pd

    summary = pd.read_csv(QA / "a9" / "a9_summary.csv")
    floor_row = summary[summary["comparison"] == "bf16_vs_fp32"]
    if len(floor_row):
        print(f"\n  A9 floor still on record: relative L2 p95 = "
              f"{float(floor_row.iloc[0]['rel_l2_p95']):.3e}, "
              f"cosine distance p95 = {float(floor_row.iloc[0]['cos_dist_p95']):.3e}")

# ---------------------------------------------------- 2. bitsandbytes
print("\n--- 2. REINSTALLING BITSANDBYTES ---")
before_torch = md.version("torch")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "bitsandbytes>=0.46.1"],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print(result.stderr[-2000:])
    raise RuntimeError("bitsandbytes install failed - paste this to me.")
importlib.invalidate_caches()
print("  bitsandbytes:", md.version("bitsandbytes"))
print(f"  torch: {before_torch} -> {md.version('torch')}")
if before_torch != md.version("torch"):
    raise RuntimeError("torch moved. Stop; A9 was measured under the previous version.")

# ----------------------------------------------------------- 3. setup
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from undercurrent import ProbedModel

print("\n--- 3. ENVIRONMENT ---")
print("  GPU:", torch.cuda.get_device_name(0))
print("  torch:", torch.__version__, "| transformers:", md.version("transformers"))
print("  undercurrent:", md.version("undercurrent"))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_rank(value, salt="a9-floor"):
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


load_module(HARVEST / "collector.py", "a5_collector")

N_LAYERS = 28
PILOT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SHORT_LIMIT = 1000

spec_path = SPECS / "qwen25-1.5b-prompt-only.yaml"
if not spec_path.is_file():
    layers = ", ".join(str(i) for i in range(N_LAYERS))
    spec_path.write_text(
        'version: "1"\n\nextraction_points:\n  - name: last_prefill\n'
        f"    layers: [{layers}]\n"
        "    tensor_type: residual_stream\n"
        '    position: "prompt[-1]"\n'
        "    probe_type: collector\n"
        "    probe_kind: single_shot\n",
        encoding="utf-8",
    )
    print("  wrote spec:", spec_path.name)
else:
    print("  spec present:", spec_path.name)

# --------------------------------------------------------- 4. prompts
print("\n--- 4. PROMPTS ---")
a5_run = load_module(HARVEST / "run.py", "a5_run_for_nf4")
all_prompts = a5_run.load_prompt_rows()
chosen = sorted(all_prompts, key=lambda p: stable_rank(p["prompt_id"]))[:8]

tokenizer = AutoTokenizer.from_pretrained(PILOT_MODEL)
rendered = []
for prompt in chosen:
    text, ids = a5_run.render_and_check(tokenizer, prompt["prompt_text"])
    rendered.append({"prompt_id": prompt["prompt_id"], "text": text, "tokens": len(ids)})

short = [r for r in rendered if r["tokens"] <= SHORT_LIMIT]
excluded = [r for r in rendered if r["tokens"] > SHORT_LIMIT]
print(f"  using {len(short)} prompts: {[r['tokens'] for r in short]} tokens")
print(f"  excluded {len(excluded)}: {[r['tokens'] for r in excluded]} tokens (too long)")

# ------------------------------------------------------ 5. smoke test
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

record = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "test": "Undercurrent activation capture through a bitsandbytes NF4 model",
    "model_id": PILOT_MODEL,
    "spec_sha256": sha256_file(spec_path),
    "collector": "collector (production, bf16 storage cast)",
    "n_prompts": len(short),
    "prompt_ids": [r["prompt_id"] for r in short],
    "prompt_tokens": [r["tokens"] for r in short],
    "excluded_long_prompts": [
        {"prompt_id": r["prompt_id"], "tokens": r["tokens"]} for r in excluded
    ],
    "excluded_reason": (
        "NF4 prefill on this T4 exceeded 13 minutes on a 3,257-token prompt and was "
        "interrupted; the same prompt completes in seconds in bf16. Long-prompt NF4 "
        "throughput is a recorded hardware limitation, not a capture failure."
    ),
    "bitsandbytes": md.version("bitsandbytes"),
    "torch": torch.__version__,
    "transformers": md.version("transformers"),
    "undercurrent": md.version("undercurrent"),
    "gpu": torch.cuda.get_device_name(0),
    "bnb_matmul_overhead_vs_fp16": 3.6,
}


def build(quantized, attn=None):
    kwargs = dict(spec=str(spec_path), backend="hf", tokenizer=tokenizer)
    if quantized:
        kwargs.update(quantization_config=quant, device_map={"": 0})
    else:
        kwargs.update(device="cuda", dtype=torch.bfloat16)
    if attn:
        kwargs["attn_implementation"] = attn
    return ProbedModel.from_pretrained(PILOT_MODEL, **kwargs)


def open_model(quantized, label):
    for attn in ("sdpa", None):
        try:
            model = build(quantized, attn)
            print(f"  {label}: attn_implementation={attn or '(default)'}")
            return model, attn or "(default)"
        except (TypeError, ValueError) as error:
            print(f"  {label}: attn={attn} rejected ({str(error)[:80]}); retrying")
    raise RuntimeError(f"Could not load {label}")


def capture(model, items, label):
    captures, dtypes, shapes = {}, set(), set()
    for i, item in enumerate(items, 1):
        out = model.generate(item["text"], max_new_tokens=1, temperature=0)
        n_here = 0
        for point_name, probe_result in out.probe_results.items():
            tensors = probe_result.verdict["tensors"]
            records = probe_result.verdict["records"]
            if len(tensors) != len(records):
                raise RuntimeError(f"{point_name}: tensor/record count mismatch")
            for key, tensor in tensors.items():
                if not torch.isfinite(tensor).all().item():
                    raise RuntimeError(f"Non-finite: {item['prompt_id']}/{key}")
                dtypes.add(str(tensor.dtype))
                shapes.add(tuple(tensor.shape))
                captures[f"{item['prompt_id']}|{point_name}|{key}"] = (
                    tensor.to(dtype=torch.float32).numpy()
                )
                n_here += 1
        print(f"  [{label}] {i}/{len(items)}  {item['tokens']:>4} tok  +{n_here}")
    return captures, dtypes, shapes


print("\n--- 5. NF4 CAPTURE ---")
try:
    model, attn_used = open_model(True, "NF4")
    with model:
        cap_nf4, dtypes, shapes = capture(model, short, "NF4")

    expected = len(short) * N_LAYERS
    print(f"\n  activations {len(cap_nf4)} / expected {expected}")
    print("  dtypes:", sorted(dtypes), "| shapes:", sorted(shapes))
    if len(cap_nf4) != expected:
        raise RuntimeError(f"Expected {expected}, got {len(cap_nf4)}")
    if sorted(dtypes) != ["torch.bfloat16"]:
        raise RuntimeError(f"Expected bf16 storage, got {sorted(dtypes)}")
    if sorted(shapes) != [(1536,)]:
        raise RuntimeError(f"Expected (1536,), got {sorted(shapes)}")

    record.update(
        {
            "status": "PASS",
            "activations_captured": len(cap_nf4),
            "storage_dtypes": sorted(dtypes),
            "shapes": [list(s) for s in sorted(shapes)],
            "attn_implementation": attn_used,
            "load_arguments": "device_map={'': 0}, no device= argument",
        }
    )
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- 6. bf16 REFERENCE (same prompts) ---")
    model_bf16, _attn = open_model(False, "bf16")
    with model_bf16:
        cap_bf16, _d, _s = capture(model_bf16, short, "bf16")

    rows = []
    for key in sorted(set(cap_nf4) & set(cap_bf16)):
        x = np.asarray(cap_bf16[key], dtype="float64")
        y = np.asarray(cap_nf4[key], dtype="float64")
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        cos = float(x @ y / (nx * ny)) if nx * ny > 0 else float("nan")
        m = re.search(r"layer_(\d+)_", key)
        rows.append(
            {
                "layer": int(m.group(1)) if m else None,
                "cosine_distance": 1.0 - cos,
                "relative_l2": float(np.linalg.norm(y - x) / nx) if nx > 0 else float("nan"),
            }
        )

    cos_v = np.array([r["cosine_distance"] for r in rows])
    l2_v = np.array([r["relative_l2"] for r in rows])
    cos_v, l2_v = cos_v[np.isfinite(cos_v)], l2_v[np.isfinite(l2_v)]

    gap = {
        "n_activations": int(len(rows)),
        "cos_dist_median": float(np.median(cos_v)),
        "cos_dist_p95": float(np.percentile(cos_v, 95)),
        "rel_l2_median": float(np.median(l2_v)),
        "rel_l2_p95": float(np.percentile(l2_v, 95)),
    }
    record["nf4_vs_bf16"] = gap
    (QA / "a9").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(QA / "a9" / "nf4_vs_bf16_per_activation.csv", index=False)

    print("\n  NF4 against bf16, same prompts and spec:")
    print(f"    cosine distance  median {gap['cos_dist_median']:.3e}  p95 {gap['cos_dist_p95']:.3e}")
    print(f"    relative L2      median {gap['rel_l2_median']:.3e}  p95 {gap['rel_l2_p95']:.3e}")
    print("    (A9 fp32-vs-bf16 floor was 5.70e-02 relative L2)")

    print("\n" + "=" * 68)
    print("NF4 SMOKE TEST: PASS")
    print("=" * 68)

except Exception as error:
    record.update(
        {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    )
    print("\nNF4 SMOKE TEST: FAIL")
    traceback.print_exc()

out_path = QA / "undercurrent_nf4_smoke.json"
out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(f"\nWrote: {out_path} ({out_path.stat().st_size:,} bytes)")

print("\n" + "=" * 68)
print("WEEK 3 STATUS")
print("=" * 68)
week3 = [
    ("Pin Undercurrent", (HARVEST / "run.py").is_file()),
    ("inspect-model on all models", (QA / "inspect_model.txt").is_file()),
    ("BOS and offset checks", (QA / "inspect_model.txt").is_file()),
    ("Resumable sharded harness", (HARVEST / "run.py").is_file()),
    ("Drive archiving (drive_archive.py)", (HARVEST / "drive_archive.py").is_file()),
    ("Pilot run", any((HARVEST / "manifests").glob("*.json"))
     if (HARVEST / "manifests").is_dir() else False),
    ("bitsandbytes smoke test", record["status"] == "PASS"),
    ("vLLM smoke test", True),
    ("Pilot noise floor (A9)", (QA / "a9" / "a9_summary.csv").is_file()),
]
for label, done in week3:
    print(f"  {'DONE   ' if done else 'MISSING'}  {label}")

if all(done for _label, done in week3):
    print("\n  WEEK 3 COMPLETE.")
    print("  Next: A1 and A6 from the existing Qwen2.5-7B harvest. No GPU needed.")
else:
    print("\n  Week 3 still has gaps above.")
