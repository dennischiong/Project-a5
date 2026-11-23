# SPDX-License-Identifier: Apache-2.0
"""A6 on Mistral-7B-Instruct-v0.3: layer sweep and position split (WS2).

The saved Mistral harvest keeps only layer 31 at the last token, so A6 needs a new forward pass that keeps every layer at two
positions. This script samples pairs, runs Mistral on each prompt once, saves the 32 decoder-block outputs at the last prefill
token and the last instruction token, then measures vanilla-vs-adversarial displacement per layer and position.

Run in Colab with a GPU (a T4 works; layers that do not fit are offloaded to CPU):
    from google.colab import drive; drive.mount('/content/drive')
    !python mistral_m3_a6.py            # or paste the file into one cell
Needs a Hugging Face read token if the model is not cached (Colab secret HF_TOKEN, or huggingface_hub.login()).

Steps
  1. Load the full harvest's prompts.parquet and pairs.parquet (keys and row numbers only; they hold no prompt text), and the
     prompt text from the shared pairs table pairs_table_v1_final-2.csv (vanilla and adversarial text, matched on pair_id).
  2. Sample up to SAMPLE_PER_TACTIC pairs per (source, tactic), fixed seed.
  3. Capture: one batch-1 bf16 forward pass per prompt; forward hooks on every decoder block. One .npz per prompt is written to
     Drive as it goes, so a disconnect loses at most one prompt; re-running skips prompts already done.
  4. Check: layer 31 at the last prefill token against the saved harvest (prenorm_last.npy) for the same prompts.
  5. Pack the capture and build the pair list. A pair is kept only if both of its prompts were captured.
  6. Layer sweep and position split, median with percentile bootstrap (2,000 resamples, seed 0, same as the Llama A6).
  7. Write CSVs, figure and meta.json to Drive.

Positions (Mistral [INST] template, rendered with the tokenizer's own chat template):
  last_prefill      = last token of the rendered prompt ("[/INST]")
  last_instruction  = the token before it, the last token of the user's text (prompt[-2], as in the work plan's model table)
Layer L = output of decoder block L (residual stream before the final norm), L = 0..31, the same definition as the harvest.

EXECUTION TIME (T4): model download and load 3-6 min; capture about 1.5-2 s per prompt, so about 35-45 min for ~1,260 prompts;
check, sweep and write about 1-3 min. A100/L4: capture about 5-10 min. If one prompt takes over 30 s, the GPU has fallen back to
CPU for most layers: restart the runtime with a GPU and run again (finished prompts are kept).
"""
import gc
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_ALIAS = "mistral-7b"
FULL_NAME = "mistral-7b__wj-sorry__hf-bf16-bs1"     # full bf16 harvest folder on Drive
FULL_DIR = None                                      # set a path if the search below cannot find it
PAIRS_CSV = None                                     # path to pairs_table_v1_final-2.csv if the search below cannot find it
PAIRS_CSV_NAME = "pairs_table_v1_final-2.csv"
OUT_DIR = Path("<drive>/MYSSPtemp/a5-artifacts/harvest/results/mistral_a6")
CAPTURE_DIR = OUT_DIR / "capture"                    # per-prompt .npz files (activations; never pushed to GitHub)
SAMPLE_PER_TACTIC = 30
MAX_TOKENS = 1536                                    # longer prompts are skipped to stay inside T4 memory; the count is recorded
N_LAYERS, HIDDEN = 32, 4096
POSITIONS = {"last_prefill": -1, "last_instruction": -2}
N_BOOT, ALPHA, SEED = 2000, 0.05, 0

T_START = time.time()
OUT_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


def say(msg):
    print(f"[{time.time() - T_START:6.0f}s] {msg}", flush=True)


def find_dir(name):
    roots = [Path(p) for p in ("/content/drive/MyDrive", "/content/gdrive/MyDrive") if Path(p).is_dir()]
    if not roots:
        raise SystemExit("Drive is not mounted: from google.colab import drive; drive.mount('/content/drive')")
    for r in roots:
        for pat in ("", "*/", "*/*/", "*/*/*/"):
            for p in sorted(r.glob(pat + name)):
                if p.is_dir():
                    return p
    return None


def find_file(name):
    roots = [Path(p) for p in ("/content", "/content/drive/MyDrive", "/content/gdrive/MyDrive") if Path(p).is_dir()]
    for r in roots:
        for pat in ("", "*/", "*/*/", "*/*/*/", "*/*/*/*/"):
            for p in sorted(r.glob(pat + name)):
                if p.is_file():
                    return p
    return None


def pick(df, candidates, what, regex=None):
    for c in candidates:
        if c in df.columns:
            return c
    if regex:
        hits = [c for c in df.columns if re.search(regex, c, re.I)]
        if len(hits) == 1:
            return hits[0]
    raise SystemExit(f"cannot find the {what} column; columns are {list(df.columns)}")


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def safe_name(key):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(key))


# ------------------------------------------------------------------ 1. harvest tables
full_dir = Path(FULL_DIR) if FULL_DIR else find_dir(FULL_NAME)
if full_dir is None or not full_dir.is_dir():
    raise SystemExit(f"cannot find {FULL_NAME} on Drive; set FULL_DIR at the top of this file")
prompts = pd.read_parquet(full_dir / "prompts.parquet")
pairs = pd.read_parquet(full_dir / "pairs.parquet")
key_col = pick(prompts, ["prompt_key", "key"], "prompt key", regex=r"key")
prompt_keys = prompts[key_col].astype(str).tolist()
row_of_key = {k: i for i, k in enumerate(prompt_keys)}


def to_row(v):
    """pairs.parquet stores either a row number into prompts.parquet or a prompt key."""
    if isinstance(v, (int, np.integer)) or (isinstance(v, str) and v.isdigit() and v not in row_of_key):
        i = int(v)
        return i if 0 <= i < len(prompts) else None
    return row_of_key.get(str(v))


v_col = pick(pairs, ["row_v", "vanilla_key", "vkey"], "vanilla prompt")
a_col = pick(pairs, ["row_a", "adversarial_key", "akey"], "adversarial prompt")
pairs = pairs.assign(pair_id=pairs["pair_id"].astype(str), rv=pairs[v_col].map(to_row), ra=pairs[a_col].map(to_row))
pairs = pairs.dropna(subset=["rv", "ra"]).astype({"rv": int, "ra": int})
say(f"harvest {full_dir.name}: {len(prompts):,} prompts, {len(pairs):,} pairs, "
    f"{pairs.groupby(['source', 'tactic']).ngroups} tactic groups")

# prompt text: only the shared pairs table has it (vanilla and adversarial text per pair_id)
csv_path = Path(PAIRS_CSV) if PAIRS_CSV else find_file(PAIRS_CSV_NAME)
if csv_path is None:
    raise SystemExit(f"cannot find {PAIRS_CSV_NAME}: upload it to /content or set PAIRS_CSV at the top of this file")
header = pd.read_csv(csv_path, nrows=0, engine="python").columns.tolist()


def text_col_of(role, regex):
    for c in (f"{role}_prompt", f"{role}_text", role):
        if c in header:
            return c
    hits = [c for c in header if re.search(regex, c, re.I)]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"cannot find the {role} text column in {csv_path.name}; columns are {header}")


cv, ca = text_col_of("vanilla", r"^van\w*(prompt|text|query)$"), text_col_of("adversarial", r"^adv\w*(prompt|text|query)$")
if "pair_id" not in header:
    raise SystemExit(f"{csv_path.name} has no pair_id column; columns are {header}")
bad_lines = []
tab = pd.read_csv(csv_path, usecols=["pair_id", cv, ca], dtype={"pair_id": str}, engine="python",
                  on_bad_lines=lambda fields: bad_lines.append(len(fields)))   # lines that do not parse are skipped and counted
tab = tab.dropna(subset=[cv, ca]).drop_duplicates("pair_id")
vtext = dict(zip(tab["pair_id"], tab[cv].astype(str)))
atext = dict(zip(tab["pair_id"], tab[ca].astype(str)))
has_text = pairs["pair_id"].isin(vtext) & pairs["pair_id"].isin(atext)
pairs_without_text = int((~has_text).sum())
pairs = pairs[has_text].reset_index(drop=True)
text_of_row = {}
for pid, rv, ra in zip(pairs["pair_id"], pairs["rv"], pairs["ra"]):
    text_of_row[int(rv)] = vtext[pid]
    text_of_row[int(ra)] = atext[pid]
say(f"text from {csv_path.name}: columns {cv} / {ca}; {len(bad_lines)} unparseable lines skipped; "
    f"{pairs_without_text:,} pairs have no text and are left out; {len(pairs):,} pairs usable")

# ------------------------------------------------------------------ 2. sample
rng = np.random.default_rng(SEED)
parts = [g.iloc[rng.permutation(len(g))[:SAMPLE_PER_TACTIC]] for _, g in pairs.groupby(["source", "tactic"], sort=True)]
sample = pd.concat(parts).sort_values("pair_id").reset_index(drop=True)
rows_needed = sorted(set(sample["rv"]) | set(sample["ra"]))
say(f"sample: {len(sample):,} pairs, {len(rows_needed):,} prompts")

# ------------------------------------------------------------------ 3. capture (resumable)
todo = [r for r in rows_needed if not (CAPTURE_DIR / f"{safe_name(prompt_keys[r])}.npz").is_file()]
skipped_long = []
skip_file = CAPTURE_DIR / "_skipped_long.json"
if skip_file.is_file():
    skipped_long = json.loads(skip_file.read_text())
todo = [r for r in todo if prompt_keys[r] not in set(skipped_long)]
gpu_name = None
if todo:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if not torch.cuda.is_available():
        raise SystemExit("no GPU: Runtime > Change runtime type > T4 GPU (or better), then run again")
    gpu_name = torch.cuda.get_device_name(0)
    say(f"loading {MODEL_ID} in bf16 on {gpu_name} ({len(todo):,} prompts left)")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    blocks = model.model.layers
    if len(blocks) != N_LAYERS:
        raise SystemExit(f"model has {len(blocks)} decoder blocks, expected {N_LAYERS}")

    grabbed = {}

    def make_hook(i):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            grabbed[i] = h[0, [POSITIONS["last_prefill"], POSITIONS["last_instruction"]], :].float().cpu().numpy()
        return hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(blocks)]
    try:
        for n, r in enumerate(todo, 1):
            rendered = tok.apply_chat_template([{"role": "user", "content": text_of_row[r]}], tokenize=False, add_generation_prompt=True)
            ids = tok(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"]
            if ids.shape[1] > MAX_TOKENS:
                skipped_long.append(prompt_keys[r])
                write_atomic(skip_file, json.dumps(skipped_long))
                continue
            grabbed.clear()
            with torch.no_grad():
                model.model(input_ids=ids.to(model.device))      # decoder only; the output head is not needed
            act = np.stack([grabbed[i] for i in range(N_LAYERS)], axis=1)   # (2 positions, 32 layers, 4096)
            np.savez(CAPTURE_DIR / f"{safe_name(prompt_keys[r])}.npz", act=act.astype(np.float16), n_tokens=ids.shape[1])
            if n % 50 == 0 or n == len(todo):
                say(f"captured {n:,}/{len(todo):,}")
            if n % 200 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    finally:
        for h in handles:
            h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()
else:
    say("capture already complete on Drive; skipping the model")

# ------------------------------------------------------------------ 4 and 5. pack, check against the harvest, build pairs
have = {r: CAPTURE_DIR / f"{safe_name(prompt_keys[r])}.npz" for r in rows_needed}
have = {r: p for r, p in have.items() if p.is_file()}
cap_rows = sorted(have)
idx_of_row = {r: i for i, r in enumerate(cap_rows)}
cap = np.zeros((len(cap_rows), 2, N_LAYERS, HIDDEN), dtype=np.float16)
n_tok_now = np.zeros(len(cap_rows), dtype=int)
for i, r in enumerate(cap_rows):
    z = np.load(have[r])
    cap[i], n_tok_now[i] = z["act"], int(z["n_tokens"])

pre = np.load(full_dir / "prenorm_last.npy", mmap_mode="r")
saved = np.asarray(pre[cap_rows], dtype=np.float32)
mine = cap[:, 0, N_LAYERS - 1, :].astype(np.float32)
cos = (saved * mine).sum(1) / (np.linalg.norm(saved, axis=1) * np.linalg.norm(mine, axis=1) + 1e-12)
rel = np.linalg.norm(saved - mine, axis=1) / (np.linalg.norm(saved, axis=1) + 1e-12)
check = {"n_prompts": len(cap_rows), "cosine_mean": float(cos.mean()), "cosine_min": float(cos.min()),
         "rel_l2_median": float(np.median(rel)), "rel_l2_p95": float(np.percentile(rel, 95))}
say(f"check vs saved harvest (layer 31, last prefill): cosine mean {check['cosine_mean']:.5f}, min {check['cosine_min']:.5f}, "
    f"relative L2 median {check['rel_l2_median']:.4f}")
if "n_tokens" in prompts.columns:      # same text and template as the harvest should give the same token count
    diff = n_tok_now - prompts["n_tokens"].to_numpy()[cap_rows].astype(int)
    check["n_tokens_equal_share"] = float((diff == 0).mean())
    check["n_tokens_median_difference"] = float(np.median(diff))
    say(f"token counts equal to the harvest's for {check['n_tokens_equal_share']:.1%} of prompts (median difference {np.median(diff):.0f})")
if check["cosine_mean"] < 0.99:
    raise SystemExit("re-captured layer 31 does not match the saved harvest (mean cosine < 0.99): the template or the position is "
                     "different from the harvest. Stop here and compare the rendered prompt with the harvest spec.")

keep = sample["rv"].isin(idx_of_row) & sample["ra"].isin(idx_of_row)
cap_pairs = sample[keep].reset_index(drop=True)
dropped_pairs = int((~keep).sum())
van = cap_pairs["rv"].map(idx_of_row).to_numpy()
adv = cap_pairs["ra"].map(idx_of_row).to_numpy()
say(f"pairs with both prompts captured: {len(cap_pairs):,} of {len(sample):,} "
    f"({dropped_pairs} dropped; {len(skipped_long)} prompts over {MAX_TOKENS} tokens)")
if len(cap_pairs) < 100:
    raise SystemExit("fewer than 100 pairs: something went wrong in the capture; nothing written")


# ------------------------------------------------------------------ 6. sweep and position split
def boot(values):
    x = np.asarray(values, dtype="float64")
    x = x[np.isfinite(x)]
    r = np.random.default_rng(SEED)
    meds = np.median(x[r.integers(0, x.size, size=(N_BOOT, x.size))], axis=1)
    return {"n": int(x.size), "median": float(np.median(x)),
            "lo": float(np.percentile(meds, 100 * ALPHA / 2)), "hi": float(np.percentile(meds, 100 * (1 - ALPHA / 2)))}


def displacement(mat, v, a):
    hv, ha = mat[v], mat[a]
    nv, na = np.linalg.norm(hv, axis=1), np.linalg.norm(ha, axis=1)
    cos_d = 1.0 - (hv * ha).sum(1) / (nv * na + 1e-12)
    rel_l2 = np.linalg.norm(ha - hv, axis=1) / (nv + 1e-12)
    return cos_d, rel_l2


rows = []
for p_idx, p_name in enumerate(POSITIONS):
    for layer in range(N_LAYERS):
        cd, rl = displacement(cap[:, p_idx, layer, :].astype(np.float32), van, adv)
        c, l = boot(cd), boot(rl)
        rows.append({"extraction_point": p_name, "layer": layer, "n_pairs": l["n"],
                     "cos_dist_median": c["median"], "cos_dist_lo": c["lo"], "cos_dist_hi": c["hi"],
                     "rel_l2_median": l["median"], "rel_l2_lo": l["lo"], "rel_l2_hi": l["hi"], "snr_l2_vs_floor": None})
sweep = pd.DataFrame(rows)

a = sweep[sweep.extraction_point == "last_prefill"].set_index("layer")
b = sweep[sweep.extraction_point == "last_instruction"].set_index("layer")
split = pd.DataFrame([{
    "layer": L,
    "last_prefill_rel_l2": a.at[L, "rel_l2_median"], "last_prefill_lo": a.at[L, "rel_l2_lo"], "last_prefill_hi": a.at[L, "rel_l2_hi"],
    "last_instruction_rel_l2": b.at[L, "rel_l2_median"], "last_instruction_lo": b.at[L, "rel_l2_lo"],
    "last_instruction_hi": b.at[L, "rel_l2_hi"],
    "difference": a.at[L, "rel_l2_median"] - b.at[L, "rel_l2_median"],
    "intervals_disjoint": bool(a.at[L, "rel_l2_lo"] > b.at[L, "rel_l2_hi"] or b.at[L, "rel_l2_lo"] > a.at[L, "rel_l2_hi"]),
} for L in range(N_LAYERS)])

# ------------------------------------------------------------------ 7. write
tag = MODEL_ALIAS
sweep.to_csv(OUT_DIR / f"a6_layer_sweep_{tag}.csv", index=False)
split.to_csv(OUT_DIR / f"a6_position_split_{tag}.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
for p_name, color, marker in (("last_prefill", "#2a78d6", "o"), ("last_instruction", "#eb6834", "^")):
    part = sweep[sweep.extraction_point == p_name]
    ax.plot(part["layer"], part["rel_l2_median"], color=color, marker=marker, markersize=3.5, linewidth=1.6,
            label=p_name.replace("_", " "))
    ax.fill_between(part["layer"], part["rel_l2_lo"], part["rel_l2_hi"], color=color, alpha=0.18, linewidth=0)
ax.set_xlabel("Layer (output of decoder block)", fontsize=9)
ax.set_ylabel("Relative L2, adversarial vs vanilla\n(median, 95% CI)", fontsize=9)
ax.grid(True, color="#e6e5e0", linewidth=0.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / f"a6_layer_sweep_{tag}.png", facecolor="#fcfcfb")
plt.close(fig)

meta = {
    "created_at": datetime.now(timezone.utc).isoformat(), "analysis": "A6 layer sweep and position split", "owner": "WS2",
    "script": "mistral_m3_a6.py", "model_id": MODEL_ID, "model_alias": tag, "harvest": full_dir.name,
    "capture": {"backend": "HF transformers, batch 1, bf16", "gpu": gpu_name or "earlier session (capture reused from Drive)",
                "layers": "decoder block outputs 0-31, before the final norm", "positions": POSITIONS,
                "max_tokens": MAX_TOKENS, "prompts_captured": len(cap_rows), "prompts_skipped_over_max_tokens": len(skipped_long),
                "storage_dtype": "float16"},
    "check_vs_saved_harvest_layer31_last_prefill": check,
    "sample": {"per_tactic": SAMPLE_PER_TACTIC, "seed": SEED, "pairs_sampled": int(len(sample)), "pairs_without_text_in_csv": pairs_without_text,
               "csv_lines_unparseable": len(bad_lines),
               "pairs_measured": int(len(cap_pairs)), "pairs_dropped": dropped_pairs,
               "tactic_groups": int(cap_pairs.groupby(["source", "tactic"]).ngroups),
               "pairs_per_source": cap_pairs.groupby("source").size().astype(int).to_dict()},
    "statistics": {"estimator": "median", "interval": "percentile bootstrap", "n_boot": N_BOOT, "alpha": ALPHA, "seed": SEED},
    "caveats": ["sample of up to 30 pairs per tactic group, not all pairs",
                "no A9 precision floor for Mistral, so snr_l2_vs_floor is empty",
                "no paraphrase or length controls in the Mistral harvest"],
    "runtime_seconds": round(time.time() - T_START, 1),
}
write_atomic(OUT_DIR / f"a6_{tag}_meta.json", json.dumps(meta, indent=2, default=str))

for p_name in POSITIONS:
    part = sweep[sweep.extraction_point == p_name]
    peak = part.loc[part["rel_l2_median"].idxmax()]
    l31 = part[part.layer == N_LAYERS - 1].iloc[0]
    print(f"{p_name:17} peak layer {int(peak.layer):2} ({peak.rel_l2_median:.4f})   layer 31 {l31.rel_l2_median:.4f} "
          f"[{l31.rel_l2_lo:.4f}, {l31.rel_l2_hi:.4f}]   n = {int(l31.n_pairs)}")
print(f"position split: intervals disjoint at {int(split.intervals_disjoint.sum())} of {N_LAYERS} layers; "
      f"last_instruction larger at layers {split.loc[split.difference < 0, 'layer'].tolist()}")
say(f"wrote a6_layer_sweep_{tag}.csv/.png, a6_position_split_{tag}.csv, a6_{tag}_meta.json to {OUT_DIR}")
