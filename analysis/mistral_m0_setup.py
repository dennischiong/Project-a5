# Cell M0 - Mistral replication of A1, A4, A6: set-up, shared helpers, load Hasif's files, check them.
# Run this FIRST in Colab. It writes nothing. Cells M1 (A1), M2 (A4), M3 and M4 (A6) reuse what it leaves in memory.
# Reads (Drive): the full bf16 harvest folder  mistral-7b__wj-sorry__hf-bf16-bs1   (prenorm_last.npy, postnorm_last.npy, sweep.npy,
#                prompts.parquet, pairs.parquet, meta.json) and Hasif's hand-off folder  A5_for_Kayson_16Nov_Mistral
#                (judges/.../labels.parquet, replies/.../replies.parquet), plus the shared pairs table pairs_table_v1_final-2.csv.
# EXECUTION TIME: about 1-3 min (reads ~550 MB from Drive). Over 6 min means the Drive mount is stuck: restart the runtime and remount.
import hashlib, json, math, os, re, time
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
FULL_NAME = "mistral-7b__wj-sorry__hf-bf16-bs1"              # full harvest folder (13,229 prompts, C1)
HANDOFF_NAME = "A5_for_Kayson_16Nov_Mistral"                  # Hasif's hand-off folder
SUB_NAME = "mistral-7b__wj-sorry__hf-bf16-bs1__sub4000"       # 4,000-prompt subset harvest (optional, link only)
FULL_DIR = HANDOFF_DIR = SUB_DIR = PAIRS_CSV = RESULTS_DIR = None   # set a Path here if the auto-search below cannot find it
OVERRIDES = {}      # column names, only if the resolver stops and asks, e.g. {"vanilla_key": "van_key"}
FIT_SPLIT = "train"             # the 70% split that r_hat is fitted on (A4)
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_ALIAS = "mistral-7b"
HIDDEN, LAST_LAYER = 4096, 31
N_BOOT, ALPHA, SEED, MIN_PER_CLASS = 2000, 0.05, 0, 5

# ------------------------------------------------------------------ find the folders
DRIVE_ROOTS = [Path(p) for p in ("/content/drive/MyDrive", "/content/gdrive/MyDrive") if Path(p).is_dir()]
if not DRIVE_ROOTS:
    raise SystemExit("Drive is not mounted. Run:  from google.colab import drive; drive.mount('/content/drive')  then run this cell again.")


def find_dir(name, roots=DRIVE_ROOTS, depth=3):
    for r in roots:
        for pat in ["", "*/", "*/*/", "*/*/*/"][:depth + 1]:
            for p in sorted(r.glob(f"{pat}{name}")):
                if p.is_dir():
                    return p
    return None


def find_file(name, roots=DRIVE_ROOTS, depth=4):
    for r in roots:
        for pat in ["", "*/", "*/*/", "*/*/*/", "*/*/*/*/"][:depth + 1]:
            for p in sorted(r.glob(f"{pat}{name}")):
                if p.is_file():
                    return p
    return None


FULL_DIR = Path(FULL_DIR) if FULL_DIR else find_dir(FULL_NAME)
HANDOFF_DIR = Path(HANDOFF_DIR) if HANDOFF_DIR else find_dir(HANDOFF_NAME)
SUB_DIR = Path(SUB_DIR) if SUB_DIR else find_dir(SUB_NAME)
for label, p in (("full harvest folder", FULL_DIR), ("hand-off folder", HANDOFF_DIR)):
    if p is None or not p.is_dir():
        raise SystemExit(f"cannot find the {label} in your Drive. If it is a shortcut that Colab cannot open, open the shared folder in "
                         "Drive, select all its files, 'Make a copy', move the copies into a normal folder, and set FULL_DIR / HANDOFF_DIR above.")
print(f"full harvest : {FULL_DIR}\nhand-off     : {HANDOFF_DIR}\nsub4000      : {SUB_DIR or 'not found (optional)'}")

# Results go to Drive (mentor-visible), next to the Llama results, with 'mistral-7b' in every file name.
if RESULTS_DIR is None:
    _cands = [r / "MYSSP/a5-artifacts/harvest/results" for r in DRIVE_ROOTS]
    RESULTS_DIR = next((c for c in _cands if c.is_dir()), _cands[0])
RESULTS_DIR = Path(RESULTS_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"results go to: {RESULTS_DIR}")


def write_atomic(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def set_hf_token():
    """Take the Hugging Face token from the Colab secret HF_TOKEN (needed for the gated Mistral files)."""
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
        return True
    except Exception:
        return False


def _read_table(path):
    """parquet (Colab has pyarrow); a same-name .csv is read instead if parquet cannot be."""
    path = Path(path)
    try:
        return pd.read_parquet(path)
    except ImportError:
        return pd.read_csv(path.with_suffix(".csv"))


def pick(df, names, what, required=True, regex=None):
    if OVERRIDES.get(what):
        return OVERRIDES[what]
    for c in names:
        if c in df.columns:
            return c
    if regex:
        hits = [c for c in df.columns if re.search(regex, str(c), re.I)]
        if len(hits) == 1:
            return hits[0]
    if required:
        raise SystemExit(f"cannot tell which column is '{what}'. The columns are: {list(df.columns)}. "
                         f"Set OVERRIDES['{what}'] = '<the right column>' at the top of cell M0 and run it again.")
    return None


# ------------------------------------------------------------------ statistics helpers (the Llama A1/A4 code, unchanged in meaning)
def to_bool(v):
    """True / False / None. Understands bools, 1/0, yes/no, true/false and the judge's harmful/unharmful words."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0", "yes", "y", "t", "harmful", "unsafe", "refusal", "refused"):
        return True
    if s in ("false", "0", "0.0", "no", "n", "f", "unharmful", "safe", "benign", "harmless", "compliance", "complied", "no_refusal"):
        return False
    return None


def norm_intent(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in ("harmful", "harm", "unsafe"):
        return "harmful"
    if s in ("benign", "harmless", "safe", "unharmful"):
        return "benign"
    return "UNKNOWN:" + str(v)


def boot_median(values, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "median": None, "lo": None, "hi": None}
    if arr.size == 1:
        return {"n": 1, "median": float(arr[0]), "lo": float(arr[0]), "hi": float(arr[0])}
    rng = np.random.default_rng(seed)
    meds = np.median(arr[rng.integers(0, arr.size, (n_boot, arr.size))], axis=1)
    return {"n": int(arr.size), "median": float(np.median(arr)),
            "lo": float(np.percentile(meds, 100 * alpha / 2)), "hi": float(np.percentile(meds, 100 * (1 - alpha / 2)))}


def boot_median_diff(a, b, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    a = np.asarray(a, dtype="float64"); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype="float64"); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"diff": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    d = (np.median(a[rng.integers(0, a.size, (n_boot, a.size))], axis=1)
         - np.median(b[rng.integers(0, b.size, (n_boot, b.size))], axis=1))
    return {"diff": float(np.median(a) - np.median(b)), "lo": float(np.percentile(d, 100 * alpha / 2)),
            "hi": float(np.percentile(d, 100 * (1 - alpha / 2)))}


def displacement(matrix, van, adv, block=4096):
    """cosine distance and relative L2 between rows van[i] and adv[i] of matrix."""
    cos_d = np.full(len(van), np.nan)
    rel = np.full(len(van), np.nan)
    for s in range(0, len(van), block):
        e = min(s + block, len(van))
        v = matrix[van[s:e]].astype("float32")
        a = matrix[adv[s:e]].astype("float32")
        ok = np.isfinite(v).all(axis=1) & np.isfinite(a).all(axis=1)
        nv, na = np.linalg.norm(v, axis=1), np.linalg.norm(a, axis=1)
        dot = np.einsum("ij,ij->i", v, a)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_d[s:e] = 1.0 - np.where((nv * na > 0) & ok, dot / (nv * na), np.nan)
            rel[s:e] = np.where((nv > 0) & ok, np.linalg.norm(a - v, axis=1) / nv, np.nan)
    return cos_d, rel


# ------------------------------------------------------------------ load the full harvest
meta = json.loads((FULL_DIR / "meta.json").read_text())
print(f"\nharvest meta: {meta.get('model')}  config {meta.get('config_name')}  gpu {meta.get('gpu')}  prompts {meta.get('n_prompts')}  "
      f"position {meta.get('position')}  sweep_layers {meta.get('sweep_layers')}")
if "Mistral" not in str(meta.get("model")):
    raise SystemExit(f"meta.json says model {meta.get('model')}, expected Mistral-7B-Instruct-v0.3")
t0 = time.time()
prompts_df = _read_table(FULL_DIR / "prompts.parquet")
pairs_raw = _read_table(FULL_DIR / "pairs.parquet")
pre = np.load(FULL_DIR / "prenorm_last.npy")
post = np.load(FULL_DIR / "postnorm_last.npy")
sweep = np.load(FULL_DIR / "sweep.npy")
N = int(meta["n_prompts"])
print(f"read in {time.time() - t0:.0f}s   prenorm {pre.shape} {pre.dtype}   postnorm {post.shape}   sweep {sweep.shape} {sweep.dtype}   "
      f"prompts.parquet {prompts_df.shape}   pairs.parquet {pairs_raw.shape}")
if pre.shape != (N, HIDDEN) or post.shape != (N, HIDDEN):
    raise SystemExit(f"prenorm/postnorm shape {pre.shape}/{post.shape}, expected ({N}, {HIDDEN})")
if len(prompts_df) != N:
    raise SystemExit(f"prompts.parquet has {len(prompts_df)} rows, meta says {N}")
if sweep.size != N * HIDDEN * len(meta.get("sweep_layers", [0])):
    print(f"   NOTE: sweep.npy holds {sweep.size / (N * HIDDEN):.2f} layers worth of values; meta says sweep_layers = {meta.get('sweep_layers')}")
bad_rows = int((~np.isfinite(pre).all(axis=1)).sum())
print(f"non-finite prenorm rows: {bad_rows}")

key_col = pick(prompts_df, ["prompt_key", "key"], "prompt_key", regex=r"key")
keys = prompts_df[key_col].astype(str).tolist()
if len(set(keys)) != len(keys):
    raise SystemExit(f"prompts.parquet column {key_col} is not unique")
row_of = {k: i for i, k in enumerate(keys)}
for c in ("row", "row_idx", "npy_row", "idx"):
    if c in prompts_df.columns and sorted(prompts_df[c].astype(int).tolist()) == list(range(N)):
        row_of = {k: int(r) for k, r in zip(keys, prompts_df[c])}
        print(f"   row order taken from prompts.parquet column '{c}'")
        break

# ------------------------------------------------------------------ pairs: vanilla key, adversarial key
pcol = pick(pairs_raw, ["pair_id"], "pair_id")
vk = pick(pairs_raw, ["vanilla_key", "vanilla_prompt_key", "van_key", "vanilla_id", "vanilla_prompt_id", "vanilla"],
          "vanilla_key", required=False, regex=r"^van\w*(key|id)$")
ak = pick(pairs_raw, ["adversarial_key", "adversarial_prompt_key", "adv_key", "adversarial_id", "adversarial_prompt_id", "adversarial"],
          "adversarial_key", required=False, regex=r"^adv\w*(key|id)$")
if vk and ak:
    pairs = pairs_raw[[pcol, vk, ak]].rename(columns={pcol: "pair_id", vk: "vkey", ak: "akey"}).copy()
else:                                                   # long format: one row per (pair, role)
    rcol = pick(pairs_raw, ["role"], "role")
    kcol = pick(pairs_raw, ["prompt_key", "key"], "pair_prompt_key", regex=r"key")
    long = pairs_raw[[pcol, rcol, kcol]].copy()
    long[rcol] = long[rcol].astype(str).str.lower()
    wide = long.drop_duplicates([pcol, rcol]).pivot(index=pcol, columns=rcol, values=kcol)
    if not {"vanilla", "adversarial"} <= set(wide.columns):
        raise SystemExit(f"roles in pairs.parquet are {list(wide.columns)}, expected vanilla and adversarial")
    pairs = wide.reset_index().rename(columns={pcol: "pair_id", "vanilla": "vkey", "adversarial": "akey"})[["pair_id", "vkey", "akey"]]
pairs["pair_id"] = pairs["pair_id"].astype(str)
pairs["vkey"] = pairs["vkey"].astype(str)
pairs["akey"] = pairs["akey"].astype(str)
if pairs["pair_id"].duplicated().any():
    raise SystemExit("pairs.parquet has the same pair_id twice")

# ------------------------------------------------------------------ pair metadata from the shared pairs table (no prompt text is kept)
META_COLS = ["pair_id", "source", "source_row_id", "split", "holdout_group", "intent", "category", "tactic", "may_not_be_understood"]
csv_path = Path(PAIRS_CSV) if PAIRS_CSV else find_file("pairs_table_v1_final-2.csv")
if csv_path is not None and csv_path.is_file():
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    want = str(meta.get("pairs_table_sha256", ""))
    print(f"\npairs table  : {csv_path}\n   sha256 {sha[:16]}  Hasif's harvest used {want}  -> {'SAME FILE' if (want and sha.startswith(want)) else 'DIFFERENT HASH (not fatal; his hash may be computed another way)'}")
    csv_meta = pd.read_csv(csv_path, usecols=lambda c: c in META_COLS, dtype={"pair_id": str})   # text columns are never read
    pairs = pairs.merge(csv_meta.drop_duplicates("pair_id"), on="pair_id", how="left")
else:
    print("\npairs table not found: taking source / tactic / intent from pairs.parquet. Set PAIRS_CSV above if this is wrong.")
    extra = [c for c in META_COLS if c in pairs_raw.columns and c != "pair_id"]
    pairs = pairs.merge(pairs_raw[[pcol] + extra].rename(columns={pcol: "pair_id"}).assign(pair_id=lambda d: d["pair_id"].astype(str))
                        .drop_duplicates("pair_id"), on="pair_id", how="left")
for c in ("source", "tactic", "intent"):
    if c not in pairs.columns or pairs[c].isna().all():
        raise SystemExit(f"no '{c}' for any pair. Set PAIRS_CSV to the shared pairs_table_v1_final-2.csv and run again.")
for c in ("split", "holdout_group", "category", "may_not_be_understood", "source_row_id"):
    if c not in pairs.columns:
        pairs[c] = np.nan
pairs["intent"] = pairs["intent"].map(norm_intent)
pairs["not_understood"] = pairs["may_not_be_understood"].map(lambda v: bool(to_bool(v) or False))
in_v, in_a = pairs["vkey"].isin(row_of), pairs["akey"].isin(row_of)
print(f"\npairs: {len(pairs):,}   with both prompts in the harvest: {int((in_v & in_a).sum()):,}   "
      f"missing vanilla {int((~in_v).sum())}   missing adversarial {int((~in_a).sum())}")
if (in_v & in_a).mean() < 0.99:
    raise SystemExit("fewer than 99% of pairs map onto the harvest rows: the keys in pairs.parquet and prompts.parquet do not line up. "
                     "Check that pairs.parquet and prompts.parquet come from the same harvest.")
print("per source / intent:")
print(pairs[in_v & in_a].groupby(["source", "intent"], dropna=False).size().to_string())
print(f"tactic groups: {pairs.loc[in_v & in_a].groupby(['source', 'tactic']).ngroups}   controls (paraphrase/length baselines) present: "
      f"{bool((pairs['source'].astype(str) == 'Controls').any())}")

# ------------------------------------------------------------------ judge labels and replies (A4 only)
labels = replies = None
lab_dirs = sorted((HANDOFF_DIR / "judges").glob("*/labels.parquet")) if (HANDOFF_DIR / "judges").is_dir() else []
rep_dirs = sorted((HANDOFF_DIR / "replies").glob("*/replies.parquet")) if (HANDOFF_DIR / "replies").is_dir() else []
if lab_dirs:
    labels = _read_table(lab_dirs[0])
    print(f"\nlabels  : {lab_dirs[0]}   rows {len(labels):,}   columns {list(labels.columns)}")
    for c in labels.columns:
        if labels[c].nunique(dropna=False) <= 6:
            print(f"   {c}: {labels[c].value_counts(dropna=False).to_dict()}")
    lk = pick(labels, ["prompt_key", "key"], "label_key", regex=r"key")
    cover = labels[lk].astype(str).isin(row_of).mean()
    print(f"   label keys found in the full harvest: {cover:.1%}")
else:
    print("\nlabels not found under judges/*/labels.parquet: M2 (A4) cannot run.")
if rep_dirs:
    replies = _read_table(rep_dirs[0])
    print(f"replies : {rep_dirs[0]}   rows {len(replies):,}   columns {list(replies.columns)}   (reply text stays in memory, never written)")

print("\nM0 done. Next: M1 (A1), then M2 (A4). A6 needs M3 + M4 (see the instructions).")
