#!/usr/bin/env python3
"""
================================================================================
Project A-5: Qwen2.5-7B-Instruct Activation Harvest (harvest_qwen25_7b_full.py)
================================================================================

Description:
    Harvests residual-stream activations from Qwen2.5-7B-Instruct for every
    unique prompt in the frozen pairs table (pairs_table_v1_final.csv).

    drive_archive.py is merged in below (section 2), so nothing needs cloning
    or uploading except the pairs table CSV.

    MODE 'bf16' = plan config C1, needs an L4 (24 GB), no quantization.
    MODE 'nf4' / 'int8' = plan configs C6 / C5, run on a free T4. Those are
    NOT the reference config: they get their own archive names (hf-nf4-c6 etc).

    What it follows from the plan:
      * greedy decoding, chat template with add_generation_prompt
      * residual_stream = decoder-layer output (forward hooks, NOT hidden_states,
        so the last layer is pre final-norm like every other layer)
      * extraction points: last_prefill (all layers, prompt[-1]),
        last_instruction (every 2nd layer, prompt[-6]),
        gen_0 / gen_4 / gen_9 / gen_19 (4 layers at 25/50/75/100% depth)
      * every UNIQUE prompt text is harvested once; a pair map (ids only) links
        pair_id + role to prompt_id
      * nothing is truncated silently: token counts and flags go in the index
      * shards of 500 prompts, safetensors + parquet index, each shard archived
        to Drive and skipped on restart
      * archive names: model__dataset__config__spechash.tpxz

    What it is NOT:
      * It does not use Undercurrent / ProbedModel. It is the plain HF backend.
        The full harvest in the plan runs on vLLM through Undercurrent.
        Label and compare accordingly.

Prerequisites:
    pip install -U "bitsandbytes>=0.46.1" accelerate transformers safetensors pyyaml

Usage (Google Colab):
    Run each cell normally. Drive will be mounted automatically.

Usage (Terminal):
    $ python harvest_qwen25_7b_full.py
================================================================================
"""

# ============================================================================
# 0. Auto-install required packages
#    Replaces the notebook-only "!pip install" magic command.
#    Works identically in Google Colab AND standard terminal/Linux/Mac.
# ============================================================================
import subprocess
import sys

_REQUIRED = [
    "bitsandbytes>=0.46.1",
    "accelerate",
    "transformers",
    "safetensors",
    "pyyaml",
]
for _pkg in _REQUIRED:
    _import_name = _pkg.split(">=")[0].replace("-", "_")
    try:
        __import__(_import_name)
    except ImportError:
        print(f"[SETUP] Installing {_pkg}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", _pkg], check=True
        )

# ============================================================================
# Standard imports (identical to original)
# ============================================================================
import gc
import glob
import hashlib
import json
import os
import re
import shutil
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import transformers
import yaml
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# Google Colab imports: gracefully skipped when running outside Colab
# so the script works identically in a terminal without ImportError.
try:
    from google.colab import drive, files
    IN_COLAB = True
except ImportError:
    drive = None
    files = None
    IN_COLAB = False

# ======================================================================
# SETTINGS  (100% identical to original)
# ======================================================================
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = "qwen25-7b"
N_LAYERS = 28
HIDDEN = 3584

MODE = "nf4"
_MODES = {
    "bf16": ("bf16", "none", "hf-bf16-c1"),
    "nf4": ("fp16", "nf4", "hf-nf4-c6"),
    "int8": ("fp16", "int8", "hf-int8-c5"),
}
DTYPE, QUANT, CONFIG_NAME = _MODES[MODE]
BACKEND, CONCURRENCY = "hf", 1
STORE_DTYPE = torch.bfloat16 if MODE == "bf16" else torch.float16

MAX_NEW_TOKENS = 1
SUBSET_N = None

SHARD_PROMPTS = 500
MAX_PROMPT_TOKENS = 4096

PREFILL_LAYERS = list(range(N_LAYERS))
INST_LAYERS = list(range(1, N_LAYERS, 2))
INST_OFFSET = -6
DECODE_LAYERS = [7, 14, 21, 27]
DECODE_POSITIONS = [0, 4, 9, 19]

VANILLA_COL = None
ADV_COL = "adversarial_prompt"
DATASET_COL = None

DRIVE_ROOT_PATH = "/content/drive/MyDrive/a5-activations"
LOCAL_WORK = Path("/content/harvest")
LOCAL_REPLIES = Path("/content/replies_local")

# ======================================================================
# EXTRACTION SPEC (identical to original)
# ======================================================================
POINTS = {
    "last_prefill": dict(layers=PREFILL_LAYERS, position="prompt[-1]"),
    "last_instruction": dict(
        layers=INST_LAYERS, position=f"prompt[{INST_OFFSET}]"
    ),
}
for _k in DECODE_POSITIONS:
    POINTS[f"gen_{_k}"] = dict(layers=DECODE_LAYERS, position=f"generated[{_k}]")

ROWS = [(p, l) for p, s in POINTS.items() for l in s["layers"]]
ROW_OF = {key: i for i, key in enumerate(ROWS)}

SPEC = {
    "version": "1",
    "extraction_points": [
        {
            "name": name,
            "layers": s["layers"],
            "tensor_type": "residual_stream",
            "position": s["position"],
            "probe_type": "collector",
            "probe_kind": "single_shot",
        }
        for name, s in POINTS.items()
    ],
}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "x"


def sha8(obj):
    blob = json.dumps(obj, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]


SPEC_HASH = sha8({
    "spec": SPEC, "model": MODEL_ID, "max_new_tokens": MAX_NEW_TOKENS,
    "dtype": DTYPE, "quant": QUANT, "backend": BACKEND,
})

print("=" * 60)
print(f"PROJECT A5: QWEN2.5-7B HARVEST (mode={MODE}, {CONFIG_NAME})")
print("=" * 60)
print(f"vectors per prompt : {len(ROWS)}  (plan: 58 for a 28-layer model)")
print(f"spec hash          : {SPEC_HASH}")
Path("/content/qwen25-7b.yaml").write_text(yaml.safe_dump(SPEC, sort_keys=False))
print("spec YAML written  : /content/qwen25-7b.yaml (commit to harvest/specs/)")

# ======================================================================
# 1. GPU check (identical to original)
# ======================================================================
if not torch.cuda.is_available():
    raise RuntimeError("GPU required. Set the Colab runtime to L4 or T4.")
props = torch.cuda.get_device_properties(0)
GPU_NAME = props.name
if MODE == "bf16" and (props.major < 8 or props.total_memory < 20 * 1024**3):
    raise RuntimeError(
        f"{GPU_NAME} ({props.total_memory / 1024**3:.0f} GB) cannot run the "
        "bf16 config. Use an L4, or set MODE = 'nf4' for a T4 (not the reference)."
    )
if MODE != "bf16":
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes       : {bnb.__version__}")
    except Exception as e:
        raise RuntimeError(
            "bitsandbytes missing: run `pip install -U bitsandbytes>=0.46.1`, "
            f"then restart the session. ({e})")
print(f"GPU                : {GPU_NAME}")

# ======================================================================
# 2. Drive archive helpers (identical to original)
# ======================================================================
DRIVE_ROOT = Path(DRIVE_ROOT_PATH)
SCRATCH = Path("/content/archives")


def setup(mount=True):
    if shutil.which("pixz") is None:
        subprocess.run("apt-get -qq update && apt-get -qq install -y pixz",
                       shell=True, check=True)
    if mount and IN_COLAB and drive is not None:
        drive.mount("/content/drive")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def archive_name(model, dataset, config, spec_hash):
    return f"{model}__{dataset}__{config}__{spec_hash[:8]}.tpxz"


def already_archived(name):
    return (DRIVE_ROOT / name).exists() and (DRIVE_ROOT / f"{name}.json").exists()


def archive_to_drive(src_dir, name, level=1, keep_local=False):
    src = Path(src_dir).resolve()
    if already_archived(name):
        print(f"skip: {name} is already on Drive")
        return DRIVE_ROOT / name
    file_list = [p for p in src.rglob("*") if p.is_file()]
    raw_bytes = sum(p.stat().st_size for p in file_list)
    local = SCRATCH / name
    t0 = time.time()
    subprocess.run(
        ["tar", "-C", str(src.parent), f"--use-compress-program=pixz -{level}",
         "-cf", str(local), src.name], check=True)
    listed = subprocess.run(["pixz", "-l", str(local)], check=True,
                            capture_output=True, text=True).stdout.split()
    n_listed = sum(1 for p in listed if not p.endswith("/"))
    if n_listed != len(file_list):
        raise RuntimeError(f"archive lists {n_listed} files, expected {len(file_list)}")
    sha = _sha256(local)
    tmp = DRIVE_ROOT / f"{name}.partial"
    shutil.copyfile(local, tmp)
    os.replace(tmp, DRIVE_ROOT / name)
    manifest = {
        "name": name, "source_dir": str(src), "n_files": len(file_list),
        "raw_bytes": raw_bytes, "archive_bytes": local.stat().st_size,
        "sha256": sha, "pixz_level": level,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (DRIVE_ROOT / f"{name}.json").write_text(json.dumps(manifest, indent=2))
    if not keep_local:
        local.unlink()
    print(f"{name}: {len(file_list)} files, {raw_bytes / 1e9:.2f} GB -> "
          f"{manifest['archive_bytes'] / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    return DRIVE_ROOT / name


def restore_from_drive(name, dest_dir, verify=True):
    local = SCRATCH / name
    shutil.copyfile(DRIVE_ROOT / name, local)
    if verify:
        expected = json.loads((DRIVE_ROOT / f"{name}.json").read_text())["sha256"]
        if _sha256(local) != expected:
            raise RuntimeError(f"{name}: sha256 mismatch, re-copy from Drive")
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-C", str(dest_dir), "--use-compress-program=pixz",
                    "-xf", str(local)], check=True)
    local.unlink()
    return Path(dest_dir)


def extract_one(name, member, dest_dir):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    with open(DRIVE_ROOT / name, "rb") as src:
        pixz = subprocess.Popen(["pixz", "-x", member], stdin=src,
                                stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(dest_dir)], stdin=pixz.stdout,
                       check=True)
        pixz.stdout.close()
        if pixz.wait() != 0:
            raise RuntimeError(f"pixz could not extract {member}")


setup()

# ======================================================================
# 3. Pairs table (identical to original)
# ======================================================================
csv_files = sorted(glob.glob("pairs_table_v1*.csv"))
if not csv_files and IN_COLAB and files is not None:
    print("Upload your pairs_table_v1*.csv file:")
    files.upload()
    csv_files = sorted(glob.glob("pairs_table_v1*.csv"))
if not csv_files:
    raise FileNotFoundError("No pairs_table_v1*.csv found.")
DATA_PATH = Path(csv_files[-1])
df = pd.read_csv(DATA_PATH)
print(f"pairs table        : {DATA_PATH.name} ({len(df):,} rows)")

van_col = VANILLA_COL or next(
    (c for c in ("vanilla_prompt", "vanilla", "original_prompt", "base_prompt")
     if c in df.columns), None)
if van_col is None or ADV_COL not in df.columns:
    raise KeyError(
        "Set VANILLA_COL / ADV_COL at the top. Columns in your CSV: "
        f"{list(df.columns)}"
    )
ds_col = DATASET_COL or next(
    (c for c in ("dataset", "source_dataset", "source") if c in df.columns),
    None)

df = df.copy()
df["_pid"] = df["pair_id"].astype(str)
df["_ds"] = df[ds_col].map(slug) if ds_col else "pairs"
for c in (van_col, ADV_COL):
    df[c] = df[c].astype(str).str.strip()
bad = (
    df[van_col].isin(["", "nan"]) | df[ADV_COL].isin(["", "nan"])
)
if bad.any():
    print(f"skipping {int(bad.sum())} rows with an empty vanilla or adversarial prompt")
df = df[~bad]

if SUBSET_N:
    df["_h"] = df["_pid"].map(lambda s: hashlib.sha1(s.encode()).hexdigest())
    df = df.sort_values("_h").head(SUBSET_N)
table_sha8 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()[:8]
long = pd.concat([
    df[["_pid", "_ds"]].assign(role="vanilla", text=df[van_col]),
    df[["_pid", "_ds"]].assign(role="adv", text=df[ADV_COL]),
])
long["prompt_id"] = long["text"].map(
    lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16])
pair_map = long[["_pid", "role", "prompt_id"]].rename(columns={"_pid": "pair_id"})
uniq = (long.sort_values(["_ds", "_pid", "role"])
        .drop_duplicates("prompt_id")[["prompt_id", "_ds", "text"]])
print(f"table sha256[:8]   : {table_sha8}")
print(f"pairs              : {len(df):,}")
print(f"unique prompts     : {len(uniq):,}  (plan: about 18,500 per model)")

# ======================================================================
# 4. Model (identical to original)
# ======================================================================
print(f"\nLoading {MODEL_ID} ({MODE})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
load_kw = dict(device_map="auto" if MODE != "bf16" else "cuda")
if MODE == "nf4":
    from transformers import BitsAndBytesConfig
    load_kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16)
elif MODE == "int8":
    from transformers import BitsAndBytesConfig
    load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
_dt = torch.bfloat16 if MODE == "bf16" else torch.float16
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=_dt, **load_kw)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=_dt, **load_kw)
model.eval()
layers = model.model.layers
assert len(layers) == N_LAYERS, f"expected {N_LAYERS} layers, got {len(layers)}"
print(f"loaded, VRAM {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

TEMPLATE_HASH = sha8({"tpl": tokenizer.chat_template})


def render(text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ======================================================================
# 5. Preflight (identical to original)
# ======================================================================
def preflight(sample_texts):
    tail_expected = "<|im_end|>\n<|im_start|>assistant\n"
    for t in sample_texts[:20]:
        ref = tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=True, add_generation_prompt=True, return_dict=False)
        mine = tokenizer(render(t), add_special_tokens=False)["input_ids"]
        assert list(ref) == list(mine), "token ids differ from apply_chat_template"
        assert tokenizer.decode(mine[-5:]) == tail_expected, (
            f"template tail unexpected: {tokenizer.decode(mine[-8:])!r}")
    ids = tokenizer(render(sample_texts[0]), add_special_tokens=False)["input_ids"]
    print("preflight OK on", min(20, len(sample_texts)), "prompts")
    print("  prompt[-6] token :", repr(tokenizer.decode(ids[INST_OFFSET])),
          "(should be the last token of the user text)")
    print("  tail             :", repr(tokenizer.decode(ids[-5:])))


preflight(df[van_col].tolist()[:20] + df[ADV_COL].tolist()[:20])

# ======================================================================
# 6. Hooks on decoder-layer outputs (identical to original)
# ======================================================================
class Capture:

    def __init__(self):
        self.buf = None
        self.calls = {}
        self.got = set()

    def reset(self, device):
        self.buf = torch.zeros(
            (len(ROWS), HIDDEN), dtype=STORE_DTYPE, device=device)
        self.calls = {l: 0 for l in range(N_LAYERS)}
        self.got = set()


cap = Capture()


def make_hook(layer):
    pre_row = ROW_OF.get(("last_prefill", layer))
    inst_row = ROW_OF.get(("last_instruction", layer))
    gen_rows = {
        k: ROW_OF[(f"gen_{k}", layer)]
        for k in DECODE_POSITIONS if (f"gen_{k}", layer) in ROW_OF
    }

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        n = cap.calls[layer]
        cap.calls[layer] = n + 1
        if n == 0:
            if pre_row is not None:
                cap.buf[pre_row] = hs[0, -1]
            if inst_row is not None:
                cap.buf[inst_row] = hs[0, INST_OFFSET]
        else:
            d = n - 1
            row = gen_rows.get(d)
            if row is not None:
                cap.buf[row] = hs[0, -1]
                cap.got.add(d)

    return hook


for i, layer_mod in enumerate(layers):
    layer_mod.register_forward_hook(make_hook(i))


@torch.no_grad()
def run_prompt(text):
    enc = tokenizer(render(text), return_tensors="pt", add_special_tokens=False)
    n_tok = int(enc["input_ids"].shape[1])
    if n_tok > MAX_PROMPT_TOKENS:
        return None, "", n_tok, 0, set()
    enc = enc.to(model.device)
    cap.reset(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
    )
    new_ids = out[0, n_tok:]
    reply = (
        tokenizer.decode(new_ids, skip_special_tokens=True)
        if MAX_NEW_TOKENS > 1 else ""
    )
    return cap.buf.cpu(), reply, n_tok, int(new_ids.shape[0]), set(cap.got)


# ======================================================================
# 7. Harvest loop (identical to original)
# ======================================================================
LOCAL_WORK.mkdir(parents=True, exist_ok=True)
LOCAL_REPLIES.mkdir(parents=True, exist_ok=True)
(DRIVE_ROOT / "replies").mkdir(parents=True, exist_ok=True)

n_done = n_skipped = 0
t_start = time.time()

pm_dir = DRIVE_ROOT / "pair_maps"
pm_dir.mkdir(parents=True, exist_ok=True)
pair_map.to_parquet(pm_dir / f"pair_map__{table_sha8}.parquet", index=False)

for ds, g in uniq.groupby("_ds", sort=True):
    g = g.sort_values("prompt_id")
    chunks = [g.iloc[i:i + SHARD_PROMPTS] for i in range(0, len(g), SHARD_PROMPTS)]
    for si, chunk in enumerate(chunks):
        name = archive_name(
            MODEL_TAG, f"{ds}-s{si:05d}", CONFIG_NAME, SPEC_HASH)
        if already_archived(name):
            n_skipped += 1
            continue

        t0 = time.time()
        bufs, index, replies = [], [], []
        for _, row in chunk.iterrows():
            vec, reply, n_tok, n_new, got = run_prompt(row["text"])
            too_long = vec is None
            if too_long:
                vec = torch.zeros((len(ROWS), HIDDEN), dtype=STORE_DTYPE)
                print(f"skipped {row['prompt_id']}: {n_tok} tokens > {MAX_PROMPT_TOKENS}")
            bufs.append(vec)
            rec = {
                "prompt_id": row["prompt_id"],
                "dataset": ds,
                "token_count": n_tok,
                "truncated": False,
                "skipped_too_long": too_long,
                "n_new_tokens": n_new,
                "backend": BACKEND,
                "concurrency": CONCURRENCY,
                "dtype": DTYPE,
                "quantization": QUANT,
                "gpu": GPU_NAME,
                "undercurrent_version": "none (plain HF hooks)",
                "spec_hash": SPEC_HASH,
                "template_hash": TEMPLATE_HASH,
                "table_sha8": table_sha8,
                "max_new_tokens": MAX_NEW_TOKENS,
            }
            for k in DECODE_POSITIONS:
                rec[f"valid_gen_{k}"] = k in got
            index.append(rec)
            if MAX_NEW_TOKENS > 1:
                replies.append({"prompt_id": rec["prompt_id"], "reply": reply})

        out_dir = LOCAL_WORK / f"{MODEL_TAG}__{ds}-s{si:05d}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        arr = torch.stack(bufs)
        n_bad = int((~torch.isfinite(arr)).sum())
        if n_bad:
            print(f"WARNING: {n_bad} non-finite values in this shard (check fp16 overflow)")
        tensors = {
            f"{p}.layer_{l}": arr[:, ROW_OF[(p, l)], :].contiguous()
            for (p, l) in ROWS
        }
        meta = {
            "model": MODEL_ID, "backend": BACKEND, "dtype": DTYPE,
            "quantization": QUANT, "concurrency": str(CONCURRENCY),
            "gpu": GPU_NAME, "torch": torch.__version__,
            "transformers": transformers.__version__,
            "spec_hash": SPEC_HASH, "template_hash": TEMPLATE_HASH,
            "undercurrent_version": "none (plain HF hooks)",
            "residual_stream": "decoder-layer output, pre final norm",
            "max_new_tokens": str(MAX_NEW_TOKENS),
            "mode": MODE, "stored_dtype": str(STORE_DTYPE),
        }
        save_file(tensors, str(out_dir / "activations.safetensors"), metadata=meta)
        pd.DataFrame(index).to_parquet(out_dir / "index.parquet", index=False)

        if replies:
            rp = LOCAL_REPLIES / f"{name}.replies.parquet"
            pd.DataFrame(replies).to_parquet(rp, index=False)
            shutil.copyfile(rp, DRIVE_ROOT / "replies" / rp.name)

        archive_to_drive(str(out_dir), name)
        shutil.rmtree(out_dir)
        del arr, tensors, bufs
        torch.cuda.empty_cache()
        gc.collect()

        n_done += 1
        print(f"[{ds} shard {si + 1}/{len(chunks)}] {len(index)} prompts in "
              f"{time.time() - t0:.0f}s (shards done {n_done}, skipped {n_skipped})")

print("\n" + "=" * 60)
print(f"DONE. Shards archived this session: {n_done}, skipped: {n_skipped}")
print(f"Elapsed: {(time.time() - t_start) / 3600:.2f} h")
print(f"Archives on Drive: {DRIVE_ROOT}")
print("=" * 60)

if IN_COLAB and drive is not None:
    try:
        drive.flush_and_unmount()
    except Exception as e:
        print("flush_and_unmount skipped:", e)
