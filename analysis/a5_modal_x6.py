"""
A5 WS2 - X6: decode-position flips on the Llama 500-prompt decode subset.

WHAT X6 REPORTS (work plan: "X6 Decode positions (Llama): flips split by whether the
generated tokens matched, as in Khatri (2026)"; measure: flip rate on same-token vs
diverged rows):

  DEFINITION (confirmed by the project owner): the headline X6 flip is a PROBE-DECISION
  flip. A G1 harmful-vs-benign probe is fitted on the REFERENCE's decode activations
  (train split only), applied unchanged to each config's activations of the same prompt
  and position, and a flip is a prompt whose decision differs. Positions follow the
  work plan's G4: generated tokens 0, 4, 9, 19. Rows are then grouped by whether the
  generated tokens had matched the reference so far. The token-level flip below is kept
  as a secondary diagnostic. Layer is chosen by validation AUROC, never by flips.

  token-level flip      = the next-token argmax under the config differs from the
                          next-token argmax under the reference at the same position.
  same-token row        = the generated prefix (g0 .. g_i, the tokens that are the
                          INPUT to this position) is identical under both configs, so
                          the two models are reading the same text and the comparison
                          is controlled.
  diverged row          = the generations had already parted before this position, so
                          the two models are reading DIFFERENT text. The flip rate
                          there measures how far the trajectories moved apart; it is
                          not a controlled per-position comparison. Reported, labelled.

HOW THE GENERATED TOKENS ARE OBTAINED (the point that blocked this item):

  The decode archives store residual streams, not generated token ids, and the reply
  shards cover only ~73 of the 500 decode prompts. The ids do not have to be stored:
  the stored layer-31 residual IS the final hidden state, so the model's own final
  RMSNorm + lm_head applied to it reproduces the logits that chose the next token.
  Greedy decoding takes the argmax, so argmax(head(norm(h))) IS the generated token.
  This is the same operation A4 already uses for its first-token refusal mass.

  Recovering the ids from the archives (rather than re-generating them on a GPU) is
  also the only option that is guaranteed to correspond to the stored activations:
  a fresh generation run can take a different branch at a near-tie and would then be
  describing tokens that these activations did not produce.

  Positions: the decode spec captures generated[0:20]. Which of the 20 captured
  states is "the state whose input is g_i" versus "the state that produced g_i" is a
  convention, so it is CALIBRATED from data, not assumed: for the decode prompts that
  also have a stored greedy reply, the recovered argmax sequence is scored against the
  reply's own token ids under both conventions and the winner must clear CALIB_MIN and
  beat the loser by CALIB_GAP. If neither does, the job raises and reports both rates.

  g0 (the first generated token) is the input to the first captured decode position,
  so it comes from last_prefill layer 31 - from the decode archive when that archive
  carries last_prefill, otherwise from the config's own prompt-position archive.

NUMERICAL CAVEAT, measured rather than assumed: the residuals are stored in bf16 and
the head is applied in float32, so a position whose top-2 logits are nearly tied can
be recovered as the runner-up. Every row carries its top1-top2 logit margin; the
summary reports the near-tie fraction and re-reports both flip rates with near-tie
rows dropped, as a sensitivity check.

RELEASE POLICY: recovered token ids are model replies. They stay in memory. The CSV
and JSON carry only derived indicators (flip, same-token, margins) - no token ids,
no text - so these outputs are safe for the release package.

INPUTS   (a5-outputs/archives)
  reference decode : hf-bf16-decode-500            1 shard
  C3               : hf-fp16-decodegen-sub500      1 shard
  C6               : hf-nf4-decodegen-sub500       1 shard   (skipped if absent)
  g0 fallback      : hf-bf16-promptpos-full (37), hf-fp16-promptpos-sub500 (1),
                     hf-nf4-promptpos-sub500 (1)
  replies          : a5-outputs/replies/shard_*.jsonl        (calibration only)

OUTPUTS  (a5-outputs)
  results/x6_probe_flips_llama31-8b.csv    PRIMARY: one row per (config, layer, G4
                                           position, prompt): probe flip, same_token
  results/x6_decode_flips_llama31-8b.csv   secondary: token-level, per (config, prompt, row)
  results/x6_summary_llama31-8b.json       headline table + calibration + sensitivity
  qa/x6_calibration.json                   both conventions' match rates

RUN
  modal run a5_modal_x6.py::x6_preflight     # ~4 min, checks inputs only
  modal run a5_modal_x6.py::x6_run           # ~15-25 min
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

APP_NAME = "a5-ws2-x6"
MODEL_ALIAS = "llama31-8b"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

REF_CONFIG = ("hf-bf16-decode-500", 1)
CFG_CONFIGS = {                       # label -> (config name, expected shards)
    "C3": ("hf-fp16-decodegen-sub500", 1),
    "C6": ("hf-nf4-decodegen-sub500", 1),
}
PREFILL_CONFIGS = {                   # label -> (config name, expected shards) for g0
    "REF": ("hf-bf16-promptpos-full", 37),
    "C3": ("hf-fp16-promptpos-sub500", 1),
    "C6": ("hf-nf4-promptpos-sub500", 1),
}

DECODE_POINT = "decode_generated"
PREFILL_POINT = "last_prefill"
HEAD_LAYER = 31
HIDDEN = 4096
N_GEN = 20
MIN_POS = 2          # a prompt needs at least this many positions to contribute a flip

MARGIN_EPS = 0.05        # top1-top2 logit gap below this counts as a near tie
CALIB_MIN = 0.80         # winning convention must match at least this fraction
CALIB_GAP = 0.30         # and beat the other convention by this much
PROBE_LAYERS = (8, 16, 24, 31)      # the decode spec's four layers
G4_POSITIONS = (0, 4, 9, 19)        # work plan G4: generated tokens 0, 4, 9, 19
PROBE_LAMBDAS = (1.0, 10.0, 100.0, 1000.0)   # ridge strengths, chosen on validation
MIN_TRAIN_PER_CLASS = 10
VAL_SPLITS = {"val", "valid", "validation", "dev"}
VAL_FRACTION = 5                    # 1 in 5 train pairs becomes validation if none exists
SPLIT_SALT = "a5-ws2-x6-valsplit-v1"
HARMFUL_VALUES = {"harmful", "harm", "unsafe"}
BENIGN_VALUES = {"benign", "harmless", "safe", "unharmful"}
META_COLS = ["pair_id", "source", "tactic", "split", "holdout_group", "category",
             "intent", "may_not_be_understood"]
N_BOOT = 2000
ALPHA = 0.05
SEED = 0
HEAD_BATCH = 256

DATA = Path("/data")
OUT = Path("/outputs")
RESULTS = OUT / "results"
QA = OUT / "qa"
SCRATCH = Path("/tmp/x6_restore")

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
          "HF_HOME": "/cache/hf", "A5_PROFILE": "prompts", "A5_GPU": "A100-40GB"})
)
app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("a5-data", create_if_missing=True)
out_volume = modal.Volume.from_name("a5-outputs", create_if_missing=True)
cache_volume = modal.Volume.from_name("a5-hf-cache", create_if_missing=True)
VOLUMES = {str(DATA): data_volume, str(OUT): out_volume, "/cache": cache_volume}
hf_secret = modal.Secret.from_name("huggingface")


# ============================================================
# PURE HELPERS (no Modal, no volumes: unit-tested locally)
# ============================================================


def to_bool(v):
    """True / False / None from bools, 'True'/'False' strings, 0/1 and NaN."""
    import math
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


def head_argmax(H, norm_w, head_w, eps: float, batch: int = HEAD_BATCH):
    """Final RMSNorm + lm_head on stored residuals H (n, d) in float32 - the same
    operation A4 applies to get its first-token distribution. Returns the argmax token
    id per row and the top1-top2 LOGIT margin per row (the near-tie measure).

    Logits are not softmaxed: the argmax and the top-2 gap are all X6 needs, and the
    gap in logit space is the quantity that bf16 storage noise has to overcome to
    change the recovered token.
    """
    import numpy as np

    H = np.asarray(H, dtype=np.float32)
    w = np.asarray(norm_w, dtype=np.float32)
    W = np.asarray(head_w, dtype=np.float32)
    if H.ndim != 2 or H.shape[1] != w.shape[0]:
        raise RuntimeError(f"residuals have shape {H.shape}, expected (n, {w.shape[0]})")
    if W.shape[1] != w.shape[0]:
        raise RuntimeError(f"head is {W.shape}, expected (vocab, {w.shape[0]})")
    if W.shape[0] < 2:
        raise RuntimeError("head has fewer than 2 rows; no top-2 margin exists")
    ids, margins = [], []
    for s in range(0, H.shape[0], batch):
        h = H[s:s + batch]
        normed = h / np.sqrt((h * h).mean(axis=-1, keepdims=True) + eps) * w
        logits = normed @ W.T
        pair = np.argpartition(-logits, 1, axis=-1)[:, :2]
        vals = np.take_along_axis(logits, pair, axis=-1)
        order = np.argsort(-vals, axis=-1)
        pair = np.take_along_axis(pair, order, axis=-1)
        vals = np.take_along_axis(vals, order, axis=-1)
        ids += pair[:, 0].astype(np.int64).tolist()
        margins += (vals[:, 0] - vals[:, 1]).astype(np.float64).tolist()
    return [int(i) for i in ids], [float(m) for m in margins]


def flips_and_prefix(ref_ids, cfg_ids, g0_ref, g0_cfg):
    """Per decode row i: did the next-token argmax flip, and was the INPUT prefix
    (g0..g_i) identical under both configs?

    With greedy decoding the generated token at captured position i is the input
    there: g_0 comes from last_prefill, and g_{j+1} is the argmax at position j.
    So the prefix stays matched only while every earlier argmax agreed:
        prefix_0 = (g0_ref == g0_cfg)
        prefix_i = prefix_{i-1} and not flip_{i-1}
    Divergence is therefore absorbing, which is the point of the split.
    """
    if len(ref_ids) != len(cfg_ids):
        raise RuntimeError(f"ref has {len(ref_ids)} positions, config has {len(cfg_ids)}")
    flips = [bool(a != b) for a, b in zip(ref_ids, cfg_ids)]
    prefix, matched = [], bool(g0_ref == g0_cfg)
    for i in range(len(ref_ids)):
        prefix.append(matched)
        matched = matched and not flips[i]
    return flips, prefix


def onset(prefix):
    """Index of the first diverged row, or None if the prefix never parted."""
    for i, ok in enumerate(prefix):
        if not ok:
            return i
    return None


def exact_binom_ci(k: int, n: int, alpha: float = ALPHA):
    """Clopper-Pearson interval for k events in n trials, pure python. A percentile
    bootstrap of a 0-event rate is [0, 0], which says nothing about how large the true
    rate could still be; this interval does."""
    if n <= 0:
        return None, None

    def log_pmf(i, p):
        return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + i * math.log(p) + (n - i) * math.log1p(-p))

    def cdf(i, p):                       # P(X <= i)
        if i < 0:
            return 0.0
        if i >= n:
            return 1.0
        return min(1.0, math.fsum(math.exp(log_pmf(j, p)) for j in range(i + 1)))

    def solve(target_fn):
        lo, hi = 1e-12, 1.0 - 1e-12
        for _ in range(80):
            mid = (lo + hi) / 2
            if target_fn(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1.0 - cdf(k - 1, p) >= alpha / 2)
    upper = 1.0 if k >= n else solve(lambda p: cdf(k, p) <= alpha / 2)
    return float(lower), float(upper)


def cluster_boot_rate(groups, n_boot=N_BOOT, alpha=ALPHA, seed=SEED) -> dict:
    """Rate over pooled rows with a percentile bootstrap that resamples PROMPTS, not
    rows: rows inside one prompt share a generation and are not independent.

    `groups` is [(n_flip, n_row), ...], one entry per prompt. Also returns exact
    (Clopper-Pearson) intervals, at the row level and at the prompt level (a prompt
    counts once if any of its rows flipped), because the bootstrap interval of a zero
    or tiny rate collapses and understates what the data cannot rule out.
    """
    import numpy as np

    g = [(int(a), int(b)) for a, b in groups if int(b) > 0]
    n_row = sum(b for _, b in g)
    if n_row == 0:
        return {"n_prompts": 0, "n_rows": 0, "rate": None, "lo": None, "hi": None,
                "exact_row_ci": [None, None], "prompts_with_flip": 0,
                "exact_prompt_ci": [None, None]}
    k_row = sum(a for a, _ in g)
    rate = k_row / n_row
    k_pr = sum(1 for a, _ in g if a > 0)
    extra = {"exact_row_ci": list(exact_binom_ci(k_row, n_row, alpha)),
             "prompts_with_flip": k_pr,
             "exact_prompt_ci": list(exact_binom_ci(k_pr, len(g), alpha))}
    if len(g) == 1:
        return {"n_prompts": 1, "n_rows": n_row, "rate": float(rate),
                "lo": float(rate), "hi": float(rate), **extra}
    num = np.array([a for a, _ in g], dtype="float64")
    den = np.array([b for _, b in g], dtype="float64")
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), (n_boot, len(g)))
    draws = num[pick].sum(axis=1) / den[pick].sum(axis=1)
    return {"n_prompts": len(g), "n_rows": int(n_row), "rate": float(rate),
            "lo": float(np.percentile(draws, 100 * alpha / 2)),
            "hi": float(np.percentile(draws, 100 * (1 - alpha / 2))), **extra}


def calibration_scores(recovered: dict, reply_ids: dict, n_gen=N_GEN) -> dict:
    """Score the recovered argmax sequence against the stored greedy reply under both
    position conventions.

      'input'  - captured position i holds the state whose INPUT is g_i,
                 so argmax_i == reply[i+1]
      'output' - captured position i holds the state that PRODUCED g_i,
                 so argmax_i == reply[i]

    Returns the match fraction of each and the chosen convention.
    """
    hits = {"input": 0, "output": 0}
    total = {"input": 0, "output": 0}
    for pid, ids in recovered.items():
        reply = reply_ids.get(pid)
        if not reply:
            continue
        for i, got in enumerate(ids[:n_gen]):
            if i + 1 < len(reply):
                total["input"] += 1
                hits["input"] += int(got == reply[i + 1])
            if i < len(reply):
                total["output"] += 1
                hits["output"] += int(got == reply[i])
    rates = {k: (hits[k] / total[k] if total[k] else None) for k in hits}
    ranked = sorted((k for k in rates if rates[k] is not None),
                    key=lambda k: rates[k], reverse=True)
    chosen, reason = None, "no overlapping prompt had a stored reply"
    if ranked:
        best = ranked[0]
        other = rates[ranked[1]] if len(ranked) > 1 else 0.0
        if rates[best] >= CALIB_MIN and rates[best] - other >= CALIB_GAP:
            chosen, reason = best, "clear winner"
        else:
            reason = (f"inconclusive: best '{best}' matched {rates[best]:.3f}, "
                      f"other matched {other:.3f}; need >= {CALIB_MIN} and a "
                      f"gap >= {CALIB_GAP}")
    return {"match_rate": rates, "compared": total, "chosen": chosen,
            "reason": reason, "n_prompts_with_reply": sum(
                1 for p in recovered if reply_ids.get(p))}


def structural_convention(offsets) -> dict:
    """Second, independent read of the position convention, from the index's own
    token_count column rather than from the replies.

    The first captured generated position sits at sequence index `token_pos_min`.
    If that equals the prompt's token_count, the prompt occupies indices
    0..token_count-1 and the first captured state is the one AFTER the prompt, whose
    INPUT is the first generated token -> 'input'. If it is token_count - 1, the first
    captured state is the last prompt position, which PRODUCED the first generated
    token -> 'output'.
    """
    vals = sorted({int(o) for o in offsets})
    if vals == [0]:
        return {"offsets": vals, "convention": "input"}
    if vals == [-1]:
        return {"offsets": vals, "convention": "output"}
    return {"offsets": vals, "convention": None}


def prefix_seed(convention: str, g0_ref, g0_cfg):
    """The g0 pair that seeds the prefix test, which is all the two conventions change.

    'input'  - the first captured position's INPUT is g0. g0 is not in the decode
               archive (it is produced at last_prefill) and may itself differ between
               configs, so it has to be supplied and compared.
    'output' - the first captured position is itself the state that PRODUCED g0, so
               the input prefix there is the prompt alone and matches by construction.
    """
    if convention == "input":
        if g0_ref is None or g0_cfg is None:
            raise RuntimeError("the 'input' convention needs g0 from last_prefill, "
                               "which was not available for this prompt")
        return int(g0_ref), int(g0_cfg)
    if convention == "output":
        return 0, 0
    raise RuntimeError(f"unknown convention {convention!r}")


def needs_prefill(convention: str) -> bool:
    return convention == "input"


# ============================================================
# ARCHIVE DISCOVERY AND READING
# ============================================================


def archives_for(config: str, expected: int):
    """Archive paths for one config, refusing a mixed or partial set (as X4 does)."""
    files = sorted(p for p in (OUT / "archives").glob(f"*__{config}__*shard_*.tpxz"))
    bases = {p.name.split("__shard_")[0] for p in files}
    if not files:
        return None, []
    if len(bases) != 1:
        raise RuntimeError(f"{config}: archives span {len(bases)} bases {sorted(bases)}")
    if len(files) != expected:
        raise RuntimeError(f"{config}: expected {expected} shard(s), found {len(files)}")
    return bases.pop(), files


def _extract(archive: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    # A transient volume read error ("Cannot read: Invalid argument") once failed a run on an
    # archive that was intact (sha256 verified afterwards). Retry from a clean destination.
    last = ""
    for attempt in range(1, 4):
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        proc = subprocess.run(["tar", "-xf", str(archive), "-C", str(dest)],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            break
        last = proc.stderr.strip()
        print(f"  tar attempt {attempt} failed on {archive.name}: {last[:200]}", flush=True)
        time.sleep(10 * attempt)
    else:
        raise RuntimeError(f"tar failed on {archive.name} after 3 attempts: {last}")
    found = sorted(dest.rglob("activations.safetensors"))
    if len(found) != 1 or not (found[0].parent / "index.parquet").is_file():
        raise RuntimeError(f"{archive.name}: expected one activations.safetensors + index")
    return found[0].parent


def _check_provenance(idx):
    models = {str(m).lower() for m in idx["model_id"].unique()}
    if not all("llama-3.1-8b" in m for m in models):
        raise RuntimeError(f"index holds non-Llama model ids {sorted(models)}")


def _index_profile(idx) -> dict:
    """What a decode index actually contains, so the structure is reported, not assumed."""
    points = sorted(str(p) for p in idx["extraction_point"].unique())
    decode = idx[idx["extraction_point"] == DECODE_POINT]
    prof = {"extraction_points": points,
            "layers_at_decode": sorted(int(x) for x in decode["layer"].unique()),
            "rows": int(len(idx)), "decode_rows": int(len(decode)),
            "prompts": int(idx["prompt_id"].nunique())}
    if len(decode):
        pos = decode["token_pos"].astype(int)
        prof["token_pos_min"] = int(pos.min())
        prof["token_pos_max"] = int(pos.max())
        per = decode.groupby("prompt_id")["token_pos"].nunique()
        prof["distinct_positions_per_prompt"] = sorted(int(v) for v in per.unique())
        if "token_count" in decode.columns:
            first = decode.groupby("prompt_id")["token_pos"].min().astype(int)
            count = decode.groupby("prompt_id")["token_count"].first().astype(int)
            prof["structural"] = structural_convention((first - count).tolist())
    if "is_generated" in idx.columns:
        prof["is_generated_values"] = sorted(
            {str(v) for v in decode["is_generated"].unique()})
    return prof


def _read_decode(files, wanted_prompts=None, layers=None):
    """{layer: {prompt_id: (token_pos list, residual matrix)}} at DECODE_POINT, each
    prompt sorted by token_pos. `layers` defaults to the head layer only.
    Returns (stores, index profile, {prompt_id: length} for prompts whose length at the
    head layer is not N_GEN)."""
    import numpy as np
    import pandas as pd
    from safetensors import safe_open

    layers = sorted({int(x) for x in (layers if layers is not None else (HEAD_LAYER,))})
    stores, profile = {L: {} for L in layers}, None
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for n, archive in enumerate(files, 1):
        tick = time.time()
        shard = _extract(archive, SCRATCH / "one")
        idx = pd.read_parquet(shard / "index.parquet")
        _check_provenance(idx)
        if profile is None:
            profile = _index_profile(idx)
        keep = ((idx["extraction_point"] == DECODE_POINT)
                & idx["layer"].astype(int).isin(layers))
        if "is_generated" in idx.columns:
            gen = idx["is_generated"].map(to_bool)
            if bool((gen == True).any()):            # noqa: E712 - None-safe
                keep &= (gen == True)                # noqa: E712
        idx = idx[keep]
        if wanted_prompts is not None:
            idx = idx[idx["prompt_id"].astype(str).isin(wanted_prompts)]
        idx = idx.sort_values(["prompt_id", "token_pos"])
        rows = {}
        with safe_open(str(shard / "activations.safetensors"), framework="pt") as fh:
            for row in idx.itertuples():
                v = fh.get_tensor(row.tensor_key).reshape(-1).float().numpy()
                if v.shape[0] != HIDDEN:
                    raise RuntimeError(f"vector size {v.shape[0]}, expected {HIDDEN}")
                rows.setdefault((int(row.layer), str(row.prompt_id)), []).append(
                    (int(row.token_pos), v.astype(np.float32)))
        for (layer, pid), items in rows.items():
            items.sort(key=lambda t: t[0])
            stores[layer][pid] = ([t[0] for t in items],
                                  np.stack([t[1] for t in items], axis=0))
        shutil.rmtree(SCRATCH / "one", ignore_errors=True)
        print(f"  [{n:2d}/{len(files)}] {len(rows):>5} (layer, prompt) series  "
              f"{time.time() - tick:5.1f}s", flush=True)
    head = stores.get(HEAD_LAYER, {})
    short = {pid: len(v[0]) for pid, v in head.items() if len(v[0]) != N_GEN}
    return stores, profile, short


def _read_prefill(files, wanted_prompts):
    """{prompt_id: residual vector} at last_prefill layer HEAD_LAYER, for g0."""
    import numpy as np
    import pandas as pd
    from safetensors import safe_open

    store = {}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for n, archive in enumerate(files, 1):
        tick = time.time()
        shard = _extract(archive, SCRATCH / "pre")
        idx = pd.read_parquet(shard / "index.parquet")
        _check_provenance(idx)
        idx = idx[(idx["extraction_point"] == PREFILL_POINT)
                  & (idx["layer"].astype(int) == HEAD_LAYER)
                  & (idx["prompt_id"].astype(str).isin(wanted_prompts))]
        with safe_open(str(shard / "activations.safetensors"), framework="pt") as fh:
            for row in idx.itertuples():
                v = fh.get_tensor(row.tensor_key).reshape(-1).float().numpy()
                store[str(row.prompt_id)] = v.astype(np.float32)
        shutil.rmtree(SCRATCH / "pre", ignore_errors=True)
        print(f"  [{n:2d}/{len(files)}] g0 states {len(store):>4}  "
              f"{time.time() - tick:5.1f}s", flush=True)
    return store


# ============================================================
# PROBE (the work plan's G4-style decode-time probe), numpy only
# ============================================================


def norm_intent(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip().lower()
    if s in HARMFUL_VALUES:
        return "harmful"
    if s in BENIGN_VALUES:
        return "benign"
    return "UNKNOWN:" + str(v)


def build_pairs(prompt_rows) -> dict:
    """pair_id -> {vanilla, adversarial, split, ...}, from pair_links_json exactly as
    the A1 and A4 jobs pair them."""
    pairs = {}
    for prompt in prompt_rows:
        for link in json.loads(prompt["pair_links_json"]):
            entry = pairs.setdefault(str(link["pair_id"]), {
                "pair_id": str(link["pair_id"]), "split": link.get("split"),
                "link_intent": link.get("intent")})
            entry[str(link["role"]).lower()] = str(prompt["prompt_id"])
    return pairs


def probe_labels_from(prompt_rows, meta_df) -> tuple:
    """prompt_id -> {y, split, pair_id}: y = 1 for harmful intent, 0 for benign (the G1
    label), split from the pairs table. A prompt that sits in pairs with different
    intents or different splits is dropped from the probe, not guessed."""
    pairs = build_pairs(prompt_rows)
    meta = {}
    if meta_df is not None and "pair_id" in meta_df.columns:
        m = meta_df.copy()
        m["pair_id"] = m["pair_id"].astype(str)
        for r in m.drop_duplicates("pair_id").to_dict("records"):
            meta[r["pair_id"]] = r
    per = {}
    unknown = set()
    for pair_id, rec in pairs.items():
        src = meta.get(pair_id, {})
        raw = src.get("intent") if "intent" in src else rec.get("link_intent")
        intent = norm_intent(raw)
        if intent and intent.startswith("UNKNOWN:"):
            unknown.add(intent)
        split = rec.get("split")
        if split is None and "split" in src:
            split = src.get("split")
        for role in ("vanilla", "adversarial"):
            pid = rec.get(role)
            if pid:
                d = per.setdefault(pid, {"intents": set(), "splits": set(), "pairs": []})
                d["intents"].add(intent)
                d["splits"].add(str(split))
                d["pairs"].append(pair_id)
    if unknown:
        raise RuntimeError(f"intent values the probe cannot map to harmful/benign: "
                           f"{sorted(unknown)}")
    out, drop = {}, {"no_intent": 0, "intent_conflict": 0, "split_conflict": 0}
    for pid, d in per.items():
        ints = {i for i in d["intents"] if i}
        if not ints:
            drop["no_intent"] += 1
        elif len(ints) != 1:
            drop["intent_conflict"] += 1
        elif len(d["splits"]) != 1:
            drop["split_conflict"] += 1
        else:
            out[pid] = {"y": int(next(iter(ints)) == "harmful"),
                        "split": next(iter(d["splits"])), "pair_id": sorted(d["pairs"])[0]}
    return out, drop


def assign_roles(labels: dict) -> tuple:
    """prompt_id -> 'train' | 'val' | 'test'.

    The pairs table's own `split` decides: 'train' trains the probe, a validation split
    (val/valid/validation/dev) selects its ridge strength and the layer, everything else
    is the held-out test. If the table has no validation split, a fixed 1-in-5 of the
    TRAIN pairs (salted hash of pair_id, both twins together) is held out as validation.
    The probe is never fitted on a validation or test prompt either way."""
    roles = {}
    for pid, d in labels.items():
        sp = str(d["split"]).strip().lower()
        roles[pid] = "train" if sp == "train" else ("val" if sp in VAL_SPLITS else "test")
    carved = False
    if not any(r == "val" for r in roles.values()):
        carved = True
        for pid, d in labels.items():
            if roles[pid] == "train":
                h = int(hashlib.sha256((SPLIT_SALT + str(d["pair_id"])).encode())
                        .hexdigest(), 16)
                if h % VAL_FRACTION == 0:
                    roles[pid] = "val"
    return roles, carved


def auroc(score, y):
    """P(score of a positive > score of a negative), ties counted half; None if one class."""
    import numpy as np
    score = np.asarray(score, dtype="float64")
    y = np.asarray(y).astype(int)
    pos, neg = score[y == 1], score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(allv.size)
    i = 0
    while i < allv.size:                       # average ranks over ties
        j = i
        while j + 1 < allv.size and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    inv = np.empty(allv.size, dtype=int)
    inv[order] = np.arange(allv.size)
    r_pos = ranks[inv[:pos.size]].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def fit_ridge_logistic(X, y, lam, max_iter=60) -> dict:
    """L2-penalised logistic regression on standardised features, in numpy.

    d (4096) is far larger than n (a few hundred), so the problem is solved in the
    n-dimensional row space: X = U S V^T, features F = U S, penalty lam/2 * ||v||^2
    (the same as lam/2 * ||w||^2 because V is orthonormal), unpenalised intercept,
    Newton steps with step halving. w = V v maps back to the original features.
    """
    import numpy as np

    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Z = (X - mean) / std
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    F = np.concatenate([U * S, np.ones((Z.shape[0], 1))], axis=1)
    r = F.shape[1] - 1
    pen = np.diag(np.concatenate([np.full(r, float(lam)), [0.0]]))
    theta = np.zeros(r + 1)

    def nll(th):
        z = F @ th
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * th @ pen @ th)

    cur = nll(theta)
    for _ in range(max_iter):
        z = F @ theta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = F.T @ (p - y) + pen @ theta
        W = np.clip(p * (1.0 - p), 1e-9, None)
        H = F.T @ (F * W[:, None]) + pen + 1e-9 * np.eye(r + 1)
        step = np.linalg.solve(H, grad)
        t = 1.0
        while t > 1e-6:
            cand = theta - t * step
            val = nll(cand)
            if val <= cur + 1e-12:
                break
            t *= 0.5
        else:
            break
        done = abs(cur - val) < 1e-10 * max(1.0, abs(cur))
        theta, cur = cand, val
        if done:
            break
    w = Vt.T @ theta[:r]
    return {"mean": mean, "std": std, "w": w, "b": float(theta[r]), "lam": float(lam)}


def probe_prob(model, X):
    import numpy as np
    z = ((np.asarray(X, dtype="float64") - model["mean"]) / model["std"]) @ model["w"] \
        + model["b"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def probe_cell(ref_X, cfg_X, y, role, lambdas=PROBE_LAMBDAS,
               min_train=None) -> dict:
    """One (layer, position) probe. Fitted ONLY on the reference's train prompts; the
    ridge strength is chosen by validation AUROC; the SAME fitted probe then scores
    the reference and the config activations of every prompt, and a flip is a prompt
    whose 0.5-threshold decision differs between the two."""
    import numpy as np

    min_train = MIN_TRAIN_PER_CLASS if min_train is None else min_train
    y = np.asarray(y).astype(int)
    role = np.asarray(role)
    tr, va = role == "train", role == "val"
    n1, n0 = int((y[tr] == 1).sum()), int((y[tr] == 0).sum())
    if min(n1, n0) < min_train:
        return {"status": "skipped",
                "reason": f"train has {n1} harmful / {n0} benign (< {min_train} per class)"}
    best_lam, best_auc = None, None
    for lam in lambdas:
        m = fit_ridge_logistic(ref_X[tr], y[tr], lam)
        a = auroc(probe_prob(m, ref_X[va]), y[va]) if va.any() else None
        if a is not None and (best_auc is None or a > best_auc):
            best_lam, best_auc = lam, a
    chosen_by = "validation AUROC"
    if best_lam is None:
        best_lam, chosen_by = lambdas[len(lambdas) // 2], "default (validation had one class)"
    m = fit_ridge_logistic(ref_X[tr], y[tr], best_lam)
    p_ref, p_cfg = probe_prob(m, ref_X), probe_prob(m, cfg_X)
    te = role == "test"
    return {"status": "ok", "lambda": float(best_lam), "lambda_chosen_by": chosen_by,
            "p_ref": p_ref, "p_cfg": p_cfg,
            "flip": (p_ref >= 0.5) != (p_cfg >= 0.5),
            "val_auroc": auroc(p_ref[va], y[va]) if va.any() else None,
            "test_auroc": auroc(p_ref[te], y[te]) if te.any() else None,
            "train_acc": float(((p_ref[tr] >= 0.5) == (y[tr] == 1)).mean()),
            "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())}


def g4_row(k: int, convention: str) -> int:
    """Row index of generated token k's own residual. Under 'input' (the first captured
    state is the one whose INPUT is g0) that is row k. Under 'output' the first
    captured state PRODUCED g0, so g_k's own residual is row k + 1."""
    return k + (1 if convention == "output" else 0)


def _bos_provenance() -> dict:
    """The Week 3 inspect-model / BOS checks (qa/bos_offset_checks.json), read for Llama.

    X6's position convention rests on one fact those checks establish: the prompt the
    adapter tokenises carries exactly ONE BOS, so the index's token_count includes it
    and the first generated token sits at sequence index token_count. A doubled or
    missing BOS would shift every decode position by one and silently misalign the
    split. So a Llama entry that is not PASS, or whose BOS count is not exactly [1],
    stops the run. A missing file is recorded, not hidden, and does not block.
    """
    path = QA / "bos_offset_checks.json"
    if not path.is_file():
        return {"found": False,
                "note": "qa/bos_offset_checks.json not found; prompt tokenisation was "
                        "not independently verified for this run"}
    data = json.loads(path.read_text())
    hits = [m for m in data.get("models", [])
            if "llama-3.1-8b" in str(m.get("model_id", "")).lower()]
    if not hits:
        return {"found": True, "llama_entry": False, "created_at": data.get("created_at"),
                "note": "the BOS checks file has no Llama-3.1-8B entry"}
    m = hits[0]
    info = {"found": True, "llama_entry": True, "status": m.get("status"),
            "bos_counts_adapter": m.get("bos_counts_adapter"),
            "observed_offset": m.get("observed_offset"),
            "plan_offset": m.get("plan_offset"),
            "created_at": data.get("created_at")}
    if m.get("status") != "PASS":
        raise RuntimeError(f"the BOS / offset check for Llama is {m.get('status')}: "
                           f"{m.get('reasons')}. Decode positions cannot be trusted.")
    if m.get("bos_counts_adapter") != [1]:
        raise RuntimeError(f"the adapter saw BOS counts {m.get('bos_counts_adapter')} on "
                           "the BOS check, not exactly [1]; decode positions would be "
                           "shifted.")
    return info


def _load_head():
    """Final RMSNorm weight and lm_head of the cached checkpoint, float32 (as in A4)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index_path = Path(hf_hub_download(MODEL_ID, "model.safetensors.index.json"))
    cfg = json.loads(Path(hf_hub_download(MODEL_ID, "config.json")).read_text())
    wmap = json.loads(index_path.read_text())["weight_map"]
    head_name = ("model.embed_tokens.weight" if cfg.get("tie_word_embeddings")
                 or "lm_head.weight" not in wmap else "lm_head.weight")
    tensors = {}
    for name in ("model.norm.weight", head_name):
        path = hf_hub_download(MODEL_ID, wmap[name])
        with safe_open(path, framework="pt") as fh:
            tensors[name] = fh.get_tensor(name).float().numpy()
    if cfg.get("final_logit_softcapping"):
        raise RuntimeError("this checkpoint uses final-logit soft-capping; not handled")
    return (tensors["model.norm.weight"], tensors[head_name], float(cfg["rms_norm_eps"]),
            {"head_tensor": head_name, "rms_norm_eps": float(cfg["rms_norm_eps"]),
             "vocab": int(tensors[head_name].shape[0])})


def _reply_token_ids(tokenizer, prompt_ids: set, n: int) -> dict:
    """prompt_id -> first n token ids of the stored greedy reply. Calibration only;
    the text never leaves this function and no ids are written to any output."""
    out = {}
    for path in sorted((OUT / "replies").glob("shard_*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                pid = str(r.get("prompt_id"))
                if pid in prompt_ids and r.get("reply"):
                    ids = tokenizer.encode(r["reply"], add_special_tokens=False)
                    if ids:
                        out[pid] = [int(i) for i in ids[:n + 1]]
    return out


def _recover(store, norm_w, head_w, eps):
    """{prompt_id: (argmax ids, margins)} for every prompt in a decode store."""
    import numpy as np

    pids = sorted(store)
    if not pids:
        return {}
    big = np.concatenate([store[p][1] for p in pids], axis=0)
    ids, margins = head_argmax(big, norm_w, head_w, eps)
    out, cursor = {}, 0
    for pid in pids:
        k = store[pid][1].shape[0]
        out[pid] = (ids[cursor:cursor + k], margins[cursor:cursor + k])
        cursor += k
    return out


def _load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _probe_labels(prompt_ids: set) -> dict:
    """{prompt_id: {y, split, pair_id, role}} for the decode prompts, the same way A4
    pairs prompts and reads intent, plus a report of what was dropped. Reads the pairs
    table WITHOUT its text columns; no prompt text is loaded."""
    import pandas as pd

    code = DATA / "code"
    _load_module(code / "collector.py", "a5_collector")
    run = _load_module(code / "run.py", "a5_run")
    run.ROOT = DATA
    run.HARVEST = DATA
    run.CSV_PATH = DATA / "pairs_table_v1_final-2.csv"
    run.PAIR_MAP_PATH = DATA / "pair_map__d240ec65.parquet"
    run.SPEC_PATH = DATA / "code" / "llama31-8b-promptpos.yaml"
    rows = [{"prompt_id": str(p["prompt_id"]), "pair_links_json": p["pair_links_json"]}
            for p in run.load_prompt_rows()]
    meta_df = pd.read_csv(run.CSV_PATH, usecols=lambda c: c in META_COLS,
                          dtype={"pair_id": str})
    labels, dropped = probe_labels_from(rows, meta_df)
    roles, carved = assign_roles(labels)
    mine = {pid: {**labels[pid], "role": roles[pid]}
            for pid in prompt_ids if pid in labels}
    report = {"decode_prompts": len(prompt_ids), "decode_prompts_labelled": len(mine),
              "dropped_in_whole_table": dropped, "validation_carved_from_train": carved}
    for role in ("train", "val", "test"):
        sel = [v for v in mine.values() if v["role"] == role]
        report[role] = {"harmful": sum(v["y"] for v in sel),
                        "benign": sum(1 - v["y"] for v in sel)}
    return mine, report


def probe_stage(label, config, ref_stores, cfg_stores, labels, prefix_map, shared,
                convention):
    """Probe-decision flips for one config, grouped by whether the generated tokens had
    matched. Returns (csv rows, summary dict). The probe is fitted on the reference
    only; the same probe scores the config."""
    import numpy as np

    rows, cells = [], {}
    for layer in PROBE_LAYERS:
        if layer not in ref_stores or layer not in cfg_stores:
            continue
        for k in G4_POSITIONS:
            r = g4_row(k, convention)
            pids = [p for p in shared
                    if p in labels and p in prefix_map
                    and p in ref_stores[layer] and p in cfg_stores[layer]
                    and len(ref_stores[layer][p][0]) > r
                    and len(cfg_stores[layer][p][0]) > r
                    and len(prefix_map[p]) > r]
            key = f"L{layer}_g{k}"
            if len(pids) < 2 * MIN_TRAIN_PER_CLASS:
                cells[key] = {"layer": layer, "g4_position": k, "status": "skipped",
                              "reason": f"only {len(pids)} prompts reach row {r}"}
                continue
            Xr = np.stack([ref_stores[layer][p][1][r] for p in pids])
            Xc = np.stack([cfg_stores[layer][p][1][r] for p in pids])
            y = np.array([labels[p]["y"] for p in pids])
            role = np.array([labels[p]["role"] for p in pids])
            cell = probe_cell(Xr, Xc, y, role)
            if cell["status"] != "ok":
                cells[key] = {"layer": layer, "g4_position": k, **cell}
                continue
            same = np.array([bool(prefix_map[p][r]) for p in pids])
            held = role != "train"
            for i, pid in enumerate(pids):
                rows.append({"config": config, "config_label": label, "layer": layer,
                             "g4_position": k, "row_index": r, "prompt_id": pid,
                             "role": role[i], "y_harmful": int(y[i]),
                             "same_token": bool(same[i]),
                             "probe_flip": bool(cell["flip"][i]),
                             "ref_prob": round(float(cell["p_ref"][i]), 6),
                             "cfg_prob": round(float(cell["p_cfg"][i]), 6),
                             "held_out": bool(held[i])})

            def grp(mask):
                n = int(mask.sum())
                return {"n": n, "flips": int(cell["flip"][mask].sum()),
                        "rate": (float(cell["flip"][mask].mean()) if n else None)}
            cells[key] = {"layer": layer, "g4_position": k, "status": "ok",
                          "lambda": cell["lambda"],
                          "lambda_chosen_by": cell["lambda_chosen_by"],
                          "val_auroc": cell["val_auroc"], "test_auroc": cell["test_auroc"],
                          "train_acc": cell["train_acc"], "n_train": cell["n_train"],
                          "n_val": cell["n_val"], "n_test": cell["n_test"],
                          "held_out": {"same_token": grp(held & same),
                                       "diverged": grp(held & ~same)},
                          "all_prompts": {"same_token": grp(same), "diverged": grp(~same)}}
    return rows, cells


def pooled_by_layer(rows, held_only=True) -> dict:
    """Per layer: probe-flip rate on same-token vs diverged rows, pooled over the G4
    positions, with a bootstrap that resamples PROMPTS (one prompt contributes up to
    four rows, which are not independent)."""
    by = {}
    for r in rows:
        if held_only and not r["held_out"]:
            continue
        grp = "same_token" if r["same_token"] else "diverged"
        d = by.setdefault(r["layer"], {}).setdefault(grp, {}).setdefault(
            r["prompt_id"], [0, 0])
        d[0] += int(r["probe_flip"]); d[1] += 1
    return {layer: {grp: cluster_boot_rate(list(per.values()))
                    for grp, per in groups.items()}
            for layer, groups in by.items()}


def headline_layer(cells: dict):
    """The layer with the best MEAN reference validation AUROC over the G4 positions.
    Flips play no part in the choice (layer and position are chosen on validation only)."""
    by = {}
    for c in cells.values():
        if c.get("status") == "ok" and c.get("val_auroc") is not None:
            by.setdefault(c["layer"], []).append(c["val_auroc"])
    if not by:
        return None, {}
    means = {layer: sum(v) / len(v) for layer, v in by.items()}
    return max(means, key=lambda k: (means[k], k)), means


def _prompt_ids_of(files) -> set:
    """Prompt ids of a config, read from its own index (no text, no tensors)."""
    import pandas as pd
    ids = set()
    for archive in files:
        shard = _extract(archive, SCRATCH / "ids")
        ids |= set(pd.read_parquet(shard / "index.parquet", columns=["prompt_id"])
                   ["prompt_id"].astype(str))
        shutil.rmtree(SCRATCH / "ids", ignore_errors=True)
    return ids


def _run(preflight_only: bool) -> dict:
    import numpy as np
    import pandas as pd
    from datetime import datetime, timezone
    from transformers import AutoTokenizer

    t0 = time.time()
    out_volume.reload()
    cache_volume.reload()
    print("=" * 72 + f"\nX6 ON {MODEL_ALIAS}"
          f"{'  (PREFLIGHT)' if preflight_only else ''}\n" + "=" * 72, flush=True)

    bos = _bos_provenance()
    print("BOS / inspect-model provenance:", json.dumps(bos), flush=True)

    ref_name, ref_files = archives_for(*REF_CONFIG)
    if not ref_files:
        raise RuntimeError(f"reference decode config {REF_CONFIG[0]} has no archive; "
                           "X6 cannot run without it")
    print(f"reference {REF_CONFIG[0]}: {ref_name}  ({len(ref_files)} shard)", flush=True)

    present, missing = {}, {}
    for label, (config, expected) in CFG_CONFIGS.items():
        base, files = archives_for(config, expected)
        if files:
            present[label] = (config, base, files)
            print(f"  {label}  {config}: present", flush=True)
        else:
            missing[label] = config
            print(f"  {label}  {config}: MISSING, skipped", flush=True)
    if not present:
        raise RuntimeError("no comparison config has an archive; nothing to compare")

    print("\nreading reference decode activations ...", flush=True)
    all_layers = sorted(set(PROBE_LAYERS) | {HEAD_LAYER})
    ref_stores, ref_profile, ref_short = _read_decode(ref_files, layers=all_layers)
    ref_store = ref_stores.get(HEAD_LAYER, {})
    print(json.dumps(ref_profile, indent=1), flush=True)
    if ref_short:
        hist = {}
        for n in ref_short.values():
            hist[n] = hist.get(n, 0) + 1
        print(f"  {len(ref_short)} reference prompt(s) have a length other than "
              f"{N_GEN}; lengths -> prompt counts: {dict(sorted(hist.items()))}. "
              f"They are kept and compared over the positions both configs share.",
              flush=True)
    ref_store = {p: v for p, v in ref_store.items() if len(v[0]) >= MIN_POS}
    if not ref_store and preflight_only:
        report = {"bos_check": bos, "VERDICT": "NO USABLE DECODE ROWS",
                  "reference": {"config": REF_CONFIG[0], "base": ref_name,
                                "profile": ref_profile, "usable_prompts": 0},
                  "note": f"no prompt has >= {MIN_POS} decode rows at layer {HEAD_LAYER}; "
                          "the profile above shows what the archive does hold"}
        print("\n" + json.dumps(report, indent=1), flush=True)
        return report
    if not ref_store:
        raise RuntimeError(
            f"no reference prompt has >= {MIN_POS} decode rows at layer {HEAD_LAYER}. "
            f"The index holds extraction points {ref_profile.get('extraction_points')}, "
            f"decode layers {ref_profile.get('layers_at_decode')}, "
            f"{ref_profile.get('decode_rows')} decode rows over "
            f"{ref_profile.get('prompts')} prompts. Paste the 'reference' profile "
            "printed above; the decode archive's layout is not what this job expects.")

    norm_w, head_w, eps, head_meta = _load_head()
    print(f"head: {head_meta}", flush=True)

    if preflight_only:
        report = {"bos_check": bos,
                  "reference": {"config": REF_CONFIG[0], "base": ref_name,
                                "profile": ref_profile, "usable_prompts": len(ref_store),
                                "dropped_short": len(ref_short)},
                  "head": head_meta, "present": {k: v[0] for k, v in present.items()},
                  "missing": missing, "overlap": {}}
        for label, (config, _base, files) in present.items():
            ids = _prompt_ids_of(files)
            report["overlap"][label] = {"config": config, "prompts": len(ids),
                                        "shared_with_reference":
                                        len(ids & set(ref_store))}
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        replies = _reply_token_ids(tokenizer, set(ref_store), N_GEN)
        report["replies_available_for_calibration"] = len(replies)
        try:
            _lab, report["probe_labels"] = _probe_labels(set(ref_store))
        except Exception as error:      # the preflight must still print the rest
            report["probe_labels"] = {"error": f"{type(error).__name__}: {str(error)[:600]}"}
        report["prefill_archives"] = {}
        for label, (config, expected) in PREFILL_CONFIGS.items():
            _b, files = archives_for(config, expected)
            report["prefill_archives"][label] = {"config": config,
                                                 "shards": len(files),
                                                 "expected": expected}
        report["minutes"] = round((time.time() - t0) / 60, 2)
        print("\n" + json.dumps(report, indent=1), flush=True)
        return report

    print("\nrecovering reference generated tokens (final RMSNorm + lm_head) ...",
          flush=True)
    ref_rec = _recover(ref_store, norm_w, head_w, eps)
    print(f"  {len(ref_rec)} prompts x {N_GEN} positions  "
          f"{(time.time() - t0) / 60:.1f} min elapsed", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    replies = _reply_token_ids(tokenizer, set(ref_rec), N_GEN)
    calib = calibration_scores({p: ref_rec[p][0] for p in ref_rec}, replies)
    print("\ncalibration against stored greedy replies:", flush=True)
    print(json.dumps(calib, indent=1), flush=True)
    structural = ref_profile.get("structural") or {"offsets": None, "convention": None}
    forced = os.environ.get("A5_X6_CONVENTION", "").strip() or None
    if calib["chosen"]:
        convention = calib["chosen"]
        source = "replies"
    elif forced in ("input", "output"):
        convention, source = forced, "A5_X6_CONVENTION override"
    elif structural["convention"]:
        convention, source = structural["convention"], "index token_count (structural)"
    else:
        raise RuntimeError(
            "the decode position convention could not be established. Replies: "
            f"{calib['reason']}. Index token_count: {structural}. X6 is not run on an "
            "assumed alignment; set A5_X6_CONVENTION=input|output to force one.")
    agree = (structural["convention"] is None
             or structural["convention"] == convention)
    print(f"  convention: {convention}   (from {source}; structural says "
          f"{structural['convention']}, offsets {structural['offsets']})", flush=True)
    if not agree:
        print("  WARNING: the replies and the index token_count disagree on the "
              "convention; the reply calibration is used and both are recorded.",
              flush=True)

    labels, label_report = None, {"status": "skipped (A5_X6_SKIP_PROBE=1)"}
    if os.environ.get("A5_X6_SKIP_PROBE", "").strip() != "1":
        print("\nloading G1 intent labels and splits for the decode prompts ...", flush=True)
        labels, label_report = _probe_labels(set(ref_rec))
        print(json.dumps(label_report, indent=1), flush=True)
        if not labels:
            raise RuntimeError("none of the decode prompts has an intent label and split; "
                               "the probe cannot be trained. See the label report above.")

    g0 = {}
    if needs_prefill(convention):
        print("\nreading g0 from last_prefill (needed by this convention) ...",
              flush=True)
        for label in ["REF"] + list(present):
            config, expected = PREFILL_CONFIGS[label]
            _base, files = archives_for(config, expected)
            if not files and label != "REF":
                missing[label] = f"{present[label][0]} (its prompt-position archive " \
                                 f"{config} is missing, so g0 cannot be recovered)"
                del present[label]
                print(f"  {label}: SKIPPED, {config} has no archive yet", flush=True)
                continue
            if not files:
                raise RuntimeError(f"{label}: g0 needs {config}, which has no archive")
            print(f"  {label}  {config}  ({len(files)} shard(s))", flush=True)
            store = _read_prefill(files, set(ref_rec))
            rec = _recover({p: ([0], v.reshape(1, -1)) for p, v in store.items()},
                           norm_w, head_w, eps)
            g0[label] = {p: rec[p][0][0] for p in rec}
            print(f"    g0 recovered for {len(g0[label])} prompts", flush=True)

    if not present:
        raise RuntimeError("no comparison config has both its decode and its "
                           "prompt-position archive; nothing to compare")
    rows, summary_cfg, probe_rows = [], {}, []
    for label, (config, base, files) in present.items():
        print(f"\n--- {label}  {config} ---", flush=True)
        cfg_stores, profile, short = _read_decode(files, wanted_prompts=set(ref_rec),
                                                  layers=all_layers)
        store = cfg_stores.get(HEAD_LAYER, {})
        if short:
            print(f"  dropped {len(short)} prompt(s) without {N_GEN} positions",
                  flush=True)
        rec = _recover(store, norm_w, head_w, eps)
        shared = sorted(set(rec) & set(ref_rec))
        print(f"  {len(shared)} prompts shared with the reference", flush=True)
        if not shared:
            raise RuntimeError(f"{config}: no prompt is shared with the reference")
        no_g0 = []
        if needs_prefill(convention):
            no_g0 = [p for p in shared
                     if p not in g0.get("REF", {}) or p not in g0.get(label, {})]
            if no_g0:
                shared = [p for p in shared if p not in set(no_g0)]
                print(f"  dropped {len(no_g0)} prompt(s) with no last_prefill g0 "
                      f"under this convention", flush=True)
            if not shared:
                raise RuntimeError(f"{config}: every shared prompt is missing g0 at "
                                   f"{PREFILL_POINT} layer {HEAD_LAYER}")

        same_g, div_g, same_g_t, div_g_t = [], [], [], []
        prefix_map = {}
        onsets, never, n_tie = [], 0, 0
        for pid in shared:
            r_ids, r_marg = ref_rec[pid]
            c_ids, c_marg = rec[pid]
            n_common = min(len(r_ids), len(c_ids))
            r_ids, r_marg = r_ids[:n_common], r_marg[:n_common]
            c_ids, c_marg = c_ids[:n_common], c_marg[:n_common]
            seed_pair = prefix_seed(convention,
                                    g0.get("REF", {}).get(pid) if g0 else None,
                                    g0.get(label, {}).get(pid) if g0 else None)
            flips, prefix = flips_and_prefix(r_ids, c_ids, *seed_pair)
            prefix_map[pid] = prefix
            pos = store[pid][0]
            k = onset(prefix)
            if k is None:
                never += 1
            else:
                onsets.append(k)
            s_f = s_n = d_f = d_n = 0
            s_f_t = s_n_t = d_f_t = d_n_t = 0
            for i, (fl, pre) in enumerate(zip(flips, prefix)):
                tie = (r_marg[i] < MARGIN_EPS) or (c_marg[i] < MARGIN_EPS)
                n_tie += int(tie)
                rows.append({"config": config, "config_label": label,
                             "prompt_id": pid, "pos_index": i, "token_pos": pos[i],
                             "same_token": bool(pre), "flip": bool(fl),
                             "ref_margin": round(r_marg[i], 6),
                             "cfg_margin": round(c_marg[i], 6),
                             "near_tie": bool(tie)})
                if pre:
                    s_n += 1; s_f += int(fl)
                    if not tie:
                        s_n_t += 1; s_f_t += int(fl)
                else:
                    d_n += 1; d_f += int(fl)
                    if not tie:
                        d_n_t += 1; d_f_t += int(fl)
            same_g.append((s_f, s_n)); div_g.append((d_f, d_n))
            same_g_t.append((s_f_t, s_n_t)); div_g_t.append((d_f_t, d_n_t))

        n_rows = sum(1 for r in rows if r["config_label"] == label)
        summary_cfg[label] = {
            "config": config, "archive_base": base,
            "n_prompts": len(shared), "n_rows": n_rows,
            "same_token": cluster_boot_rate(same_g),
            "diverged": cluster_boot_rate(div_g),
            "near_tie_rows": n_tie,
            "near_tie_fraction": (n_tie / n_rows) if n_rows else None,
            "sensitivity_near_ties_dropped": {
                "same_token": cluster_boot_rate(same_g_t),
                "diverged": cluster_boot_rate(div_g_t)},
            "prompts_never_diverged": never,
            "prompts_diverged": len(onsets),
            "divergence_onset_median": (float(np.median(onsets)) if onsets else None),
            "prompts_reply_shorter_than_n_gen": len(short),   # kept, compared over the shared positions
            "prompts_dropped_no_g0": len(no_g0),
        }
        s, d = summary_cfg[label]["same_token"], summary_cfg[label]["diverged"]
        print(f"  same-token rows {s['n_rows']:>5}  flip rate "
              f"{s['rate'] if s['rate'] is None else round(s['rate'], 4)}", flush=True)
        print(f"  diverged  rows {d['n_rows']:>5}  flip rate "
              f"{d['rate'] if d['rate'] is None else round(d['rate'], 4)}", flush=True)
        if labels is not None:
            prows, pcells = probe_stage(label, config, ref_stores, cfg_stores, labels,
                                        prefix_map, shared, convention)
            probe_rows += prows
            best, means = headline_layer(pcells)
            pooled = pooled_by_layer(prows, held_only=True)
            summary_cfg[label]["probe"] = {
                "headline_layer": best, "mean_val_auroc_by_layer": means,
                "headline": (pooled.get(best) if best is not None else None),
                "pooled_held_out_by_layer": pooled,
                "pooled_all_prompts_by_layer": pooled_by_layer(prows, held_only=False),
                "cells": pcells}
            print(f"  PROBE flips (held-out prompts), headline layer {best} "
                  f"(chosen on validation AUROC {means.get(best)}):", flush=True)
            for grp in ("same_token", "diverged"):
                v = (pooled.get(best) or {}).get(grp) or {}
                print(f"    {grp:<10} rows {v.get('n_rows', 0):>4}  "
                      f"flip rate {v.get('rate')}   95% [{v.get('lo')}, {v.get('hi')}]",
                      flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / f"x6_decode_flips_{MODEL_ALIAS}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    probe_csv = RESULTS / f"x6_probe_flips_{MODEL_ALIAS}.csv"
    if probe_rows:
        pd.DataFrame(probe_rows).to_csv(probe_csv, index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_alias": MODEL_ALIAS, "model_id": MODEL_ID,
        "reference_config": REF_CONFIG[0], "reference_base": ref_name,
        "decode_point": DECODE_POINT, "head_layer": HEAD_LAYER,
        "generated_tokens_per_prompt": N_GEN,
        "token_ids_source": "recovered from the stored layer-31 residual via the "
                            "model's own final RMSNorm + lm_head (greedy argmax); "
                            "no ids are stored in the archives and none are written "
                            "to these outputs",
        "bos_check": bos,
        "flip_definition": "X6 headline = PROBE-DECISION flip: a G1 harmful-vs-benign "
                           "probe fitted on the reference's decode activations, applied "
                           "unchanged to each config; a flip is a prompt whose 0.5-"
                           "threshold decision differs between reference and config, "
                           "at generated tokens 0, 4, 9, 19. Rows are grouped by "
                           "whether the generated tokens had matched so far. The "
                           "token-level flip (`configs.*.same_token / .diverged`) is a "
                           "secondary diagnostic.",
        "probe_spec": {"type": "L2-regularised logistic regression on standardised "
                               "residuals, numpy",
                       "labels": "G1 intent (harmful=1, benign=0) from the pairs table",
                       "fit_on": "reference decode activations of the decode subset's "
                                 "train split only; ridge strength chosen by validation "
                                 "AUROC; layer chosen by mean validation AUROC",
                       "reported_on": "prompts NOT in the train split; the pooled_all_prompts "
                                      "figures include train prompts the probe was fitted "
                                      "on, so they are in-sample and not held out",
                       "layers": list(PROBE_LAYERS), "g4_positions": list(G4_POSITIONS),
                       "limitation": "trained on the 500-prompt decode subset only, not "
                                     "the full harvest, so absolute probe accuracy is "
                                     "lower than a full-harvest G4 probe; X6 compares the "
                                     "SAME probe across configs, which this does support"},
        "probe_labels": label_report,
        "position_convention": convention,
        "position_convention_source": source,
        "position_convention_structural": structural,
        "position_convention_checks_agree": bool(agree),
        "calibration": calib, "head": head_meta,
        "margin_eps": MARGIN_EPS, "n_boot": N_BOOT, "alpha": ALPHA, "seed": SEED,
        "bootstrap": "percentile bootstrap resampling PROMPTS (rows within a "
                     "generation are dependent)",
        "configs": summary_cfg, "missing_configs": missing,
        "reference_profile": ref_profile,
        "minutes": round((time.time() - t0) / 60, 2),
    }
    (RESULTS / f"x6_summary_{MODEL_ALIAS}.json").write_text(json.dumps(summary, indent=1))
    (QA / "x6_calibration.json").write_text(json.dumps(
        {"calibration": calib, "convention": convention,
         "convention_source": source, "structural": structural, "bos_check": bos,
         "checks_agree": bool(agree), "head": head_meta,
         "reference_profile": ref_profile}, indent=1))
    out_volume.commit()
    print(f"\nwrote {csv_path.name} ({len(rows)} rows), x6_summary_{MODEL_ALIAS}.json, "
          f"qa/x6_calibration.json", flush=True)
    print(f"X6 done in {summary['minutes']} min", flush=True)
    return summary


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret],
              cpu=8.0, memory=65536, timeout=7200)
def x6_run() -> dict:
    return _run(preflight_only=False)


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret],
              cpu=4.0, memory=32768, timeout=1800)
def x6_preflight() -> dict:
    return _run(preflight_only=True)


@app.local_entrypoint()
def main():
    print(json.dumps(x6_run.remote(), indent=1, default=str))
