"""Task 5: own pilot data. NEEDS A GPU (L4). Run only after the user says "go".

Prints ONLY ids, counts and numbers: never prompt text or model replies.
Outputs (repo-relative):
  data/pairs_text.parquet            raw text (git-ignored)
  harvest/out/pairs.parquet          pairs table WITHOUT text
  harvest/out/pilot_acts.safetensors prompt[-1] vector for every hidden_states entry, bf16, key "<pair_id>|<role>"
  harvest/out/pilot_acts_instr.safetensors  same, at the last token of the user instruction (Zhao et al. split)
  harvest/out/pilot_prenorm_last.safetensors  last layer output BEFORE the final norm, shape (1, hidden)
  harvest/out/pilot_nexttoken.parquet  top-20 next tokens and refusal/comply mass at prompt[-1] (ids and numbers)
  harvest/out/pilot_index.parquet    index: key, token count, last-instruction offset, truncation flag, config
  harvest/out/pilot_meta.json        protocol metadata
  harvest/out/noise_floor.json       pilot-only rough floor (batch 1 vs 16), for each layer view

Where tactic labels come from (checked in this order, see tactics_source_for):
  1. data/tactics_map.parquet, if it exists (TACTIC_SOURCE hook, for when Dennis says where tactics come from);
  2. the dataset's own "tactics" column, if it has one;
  3. PILOT ONLY: --allow-missing-tactics sets every tactic to "unknown" and samples at random;
  4. otherwise STOP, exactly as before.

Flags:
  --check-columns           CPU-only pre-check: no model, no GPU
  --allow-missing-tactics   PILOT ONLY: run without tactic labels (recorded in pilot_meta.json)
"""
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import lib  # noqa: E402
from seeds import set_seed  # noqa: E402

SEED = 0
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
N_PER_TYPE = 50
N_NOISE = 50
OUT = REPO / "harvest" / "out"
DATA = REPO / "data"
TACTICS_COL = "tactics"
REQUIRED_COLS = {"vanilla", "adversarial", TACTICS_COL, "data_type"}

# ---- TACTIC_SOURCE hook ---------------------------------------------------------------------------------------------
# When Dennis says where tactic labels come from, save them as data/tactics_map.parquet with two columns:
#   <TACTICS_MAP_KEY>  the row key, matching the same column of the loaded dataset
#   tactic             the tactic label: one string, or a list / "['a', 'b']" string (the first entry is used)
# The default key, source_row_id, is the 0-based data-row number in WildJailbreak train/train.tsv (header not
# counted). It is the same number pairs.parquet stores in source_row_id. To join on another column, change
# TACTICS_MAP_KEY here (the only place it is set). Rows the map does not cover get tactic "unknown".
# TACTICS_MAP_PATH (set by Task 0 from team_inputs.json) points at the file when it lives somewhere else.
TACTICS_MAP = Path(os.environ.get("TACTICS_MAP_PATH") or DATA / "tactics_map.parquet")
TACTICS_MAP_KEY = "source_row_id"

# tactics_source values, written to pilot_meta.json
SRC_MAP = "data/tactics_map.parquet (TACTIC_SOURCE hook)"
SRC_COLUMN = "dataset 'tactics' column (first listed tactic)"
SRC_OVERRIDE = "missing (pilot override)"


def first_tactic(raw) -> str:
    s = str(raw).strip()
    if not s or s in ("None", "nan"):
        return "unknown"
    if s.startswith("["):
        try:
            items = ast.literal_eval(s)
            return str(items[0]).strip() if items else "unknown"
        except Exception:  # noqa: BLE001
            pass
    return s.split(",")[0].strip().strip("'\"") or "unknown"


def _tactic_value(v) -> str:
    """One tactic label from a map cell: a string, a "['a', 'b']" string, or a list/array (first entry used)."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return first_tactic(v[0]) if len(v) else "unknown"
    return first_tactic(v)


def tactics_source_for(cols, allow_missing_tactics=False, map_path=None) -> str:
    """Decide where tactic labels come from, given the dataset's column names. Returns one of the SRC_* labels.

    Order: tactics map file (if it exists) > the dataset's 'tactics' column > pilot override > STOP.
    Without the map file and without the override flag, a missing 'tactics' column STOPs with exactly the
    same message as before. A missing vanilla/adversarial/data_type column always STOPs.
    """
    map_path = TACTICS_MAP if map_path is None else Path(map_path)
    cols = set(cols)
    missing = REQUIRED_COLS - cols
    if map_path.exists():
        source = SRC_MAP
    elif TACTICS_COL in cols:
        source = SRC_COLUMN
    elif allow_missing_tactics:
        source = SRC_OVERRIDE
    else:
        source = None
    if (missing - {TACTICS_COL}) or source is None:
        sys.exit(f"STOP: required column(s) missing: {sorted(missing)}. "
                 "WildJailbreak gives no per-row tactic labels; ask the mentor/Dennis where tactics come from.")
    if source == SRC_OVERRIDE:
        print("WARNING: no tactic labels; PILOT ONLY override is on: every tactic = 'unknown', plain seeded random "
              "sample, per-tactic results skipped (recorded in pilot_meta.json as tactics_source).")
    elif source == SRC_MAP:
        print(f"tactics: from {map_path.name} (TACTIC_SOURCE hook), joined on '{TACTICS_MAP_KEY}'")
    return source


def load_tactics_map(map_path=None, key=None) -> dict:
    """Read the tactics map file into {str(row key): tactic}. STOPs (counts only, never text) if the file lacks
    the key or 'tactic' column, or if a key appears more than once."""
    map_path = TACTICS_MAP if map_path is None else Path(map_path)
    key = TACTICS_MAP_KEY if key is None else key
    m = pd.read_parquet(map_path)
    missing = {key, "tactic"} - set(m.columns)
    if missing:
        sys.exit(f"STOP: {map_path.name} is missing column(s) {sorted(missing)}; expected '{key}' and 'tactic'.")
    dup = m[key].astype(str).duplicated()
    if dup.any():
        sys.exit(f"STOP: {map_path.name} has {int(dup.sum())} duplicated '{key}' value(s); each row needs one tactic.")
    return dict(zip(m[key].astype(str), m["tactic"].map(_tactic_value)))


def assign_tactics(df: pd.DataFrame, source: str, map_path=None, key=None) -> pd.DataFrame:
    """Add the 'tactic' column according to ``source`` (a SRC_* label from tactics_source_for)."""
    df = df.copy()
    key = TACTICS_MAP_KEY if key is None else key
    if source == SRC_MAP:
        if key not in df.columns:
            sys.exit(f"STOP: the dataset has no '{key}' column to join the tactics map on (change TACTICS_MAP_KEY).")
        mapping = load_tactics_map(map_path, key)
        df["tactic"] = df[key].astype(str).map(mapping).fillna("unknown")
        print(f"tactics map: {int((df['tactic'] != 'unknown').sum())} of {len(df)} rows labelled, the rest 'unknown'")
    elif source == SRC_COLUMN:
        df["tactic"] = df[TACTICS_COL].map(first_tactic)
    else:
        df["tactic"] = "unknown"
    return df


def sample_spread(sub: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Round-robin over tactics (random order within and between tactics) to cover as many as possible.

    With one tactic or none (e.g. every row 'unknown' under the pilot override) there is nothing to spread over,
    so it falls back to a plain seeded random sample of n rows.
    """
    if sub["tactic"].nunique() <= 1:
        return sub.sample(n=min(n, len(sub)), random_state=int(rng.integers(1 << 31)))
    groups = {t: g.sample(frac=1, random_state=int(rng.integers(1 << 31))) for t, g in sub.groupby("tactic")}
    order = list(groups)
    rng.shuffle(order)
    picked, i = [], 0
    while len(picked) < n and any(len(g) for g in groups.values()):
        t = order[i % len(order)]
        if len(groups[t]):
            picked.append(groups[t].iloc[0])
            groups[t] = groups[t].iloc[1:]
        i += 1
    return pd.DataFrame(picked)


def template_hash(tok) -> str:
    """Short sha256 of the tokenizer's chat template (work plan section 4 schema / section 6.6 index)."""
    import hashlib

    return hashlib.sha256(str(tok.chat_template).encode("utf-8")).hexdigest()[:16]


def last_instruction_index(tok, prompt: str, user_text: str):
    """Index (from the start) of the last token of the user's own text inside the rendered prompt.

    Uses character offsets: the last token that starts inside the user text and is not zero-width.
    Returns None if the text cannot be located (then we STOP rather than guess).
    """
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    needle = user_text.strip()
    start = prompt.rfind(needle)
    if start < 0 or not needle:
        return None, enc["input_ids"]
    end = start + len(needle)
    idx = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s and s < end and e > start]
    return (max(idx) if idx else None), enc["input_ids"]


# Kickoff extraction step 5: refusal-like vs compliance-like first tokens, using the kickoff's examples ("I", "Sorry"
# vs "Sure", "Here") extended with the same lists as the Week 0 notebook. This is a CRUDE plumbing measure: "I" also
# starts "I'd be happy to". [PROPOSED - confirm with mentor]; the real behaviour label comes from WildGuard.
REFUSAL_WORDS = ["I", "Sorry", "I'm sorry", "I cannot", "I can't", "I apologize", "Unfortunately", "As an AI"]
COMPLY_WORDS = ["Sure", "Here", "Certainly", "Of course", "Absolutely", "Okay"]


def first_token_ids(tok, words):
    ids = set()
    for w in words:
        for variant in (w, " " + w):
            t = tok.encode(variant, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


def load_and_check(allow_missing_tactics=False):
    """Download train.tsv, print ONLY column names and data_type counts, stop if required columns are missing.

    Extra columns (e.g. 'completion', which holds model replies) are allowed but never used or printed.
    Returns (dataframe, tactics_source).
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("allenai/wildjailbreak", "train/train.tsv", repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    cols = list(pd.read_csv(path, sep="\t", nrows=0).columns)  # header only
    print("columns:", cols)
    source = tactics_source_for(cols, allow_missing_tactics)  # STOPs here if tactics are missing and not overridden
    extra = set(cols) - REQUIRED_COLS
    if extra:
        print("note: extra columns present, never loaded:", sorted(extra))
    # usecols: the 'completion' column (model replies) is never read into memory
    df = pd.read_csv(path, sep="\t", keep_default_na=False, usecols=sorted(REQUIRED_COLS & set(cols)))
    print("data_type counts:", df["data_type"].value_counts().to_dict())
    return df, source


def build_pairs(tok, allow_missing_tactics=False):
    df, tactics_source = load_and_check(allow_missing_tactics)
    df = df.reset_index().rename(columns={"index": "source_row_id"})
    df = assign_tactics(df, tactics_source)

    rng = np.random.default_rng(SEED)
    parts = []
    for dt, intent in [("adversarial_harmful", "harmful"), ("adversarial_benign", "benign")]:
        sub = df[(df["data_type"] == dt) & (df["adversarial"].str.strip() != "") & (df["vanilla"].str.strip() != "")]
        # One vanilla request can have many adversarial rewrites. Keep ONE row per vanilla text (chosen at random,
        # seeded) so the same vanilla prompt can never land in two pairs and leak across train/eval splits.
        n_before = len(sub)
        sub = sub.sample(frac=1, random_state=int(rng.integers(1 << 31))).drop_duplicates("vanilla")
        print(f"{dt}: {n_before} rows, {len(sub)} with a unique vanilla text")
        pick = sample_spread(sub, N_PER_TYPE, rng).assign(intent=intent)
        print(f"{dt}: sampled {len(pick)} rows over {pick['tactic'].nunique()} distinct tactics")
        parts.append(pick)
    pick = pd.concat(parts, ignore_index=True)
    pick["pair_id"] = [f"wj{int(r):07d}" for r in pick["source_row_id"]]

    def n_tok(text):  # rendered with the chat template; never truncated
        p = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        return len(tok(p, add_special_tokens=False)["input_ids"])

    t_hash = template_hash(tok)
    rows = []
    for _, r in pick.iterrows():
        for role, col in (("vanilla", "vanilla"), ("adversarial", "adversarial")):
            rows.append(dict(
                pair_id=r["pair_id"], source="wildjailbreak", role=role, intent=r["intent"], tactic=r["tactic"],
                token_count=n_tok(r[col]), split=lib.assign_split(r["pair_id"]), source_row_id=int(r["source_row_id"]),
                template_hash=t_hash,  # work plan section 4 schema; holdout_group waits for Dennis's definition
                text=r[col],
            ))
    full = pd.DataFrame(rows)
    # raises ValueError with counts only if bad; all-'unknown' tactics are accepted only under the pilot override
    lib.validate_pairs(full, allow_unknown_tactics=(tactics_source == SRC_OVERRIDE))
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    full.to_parquet(DATA / "pairs_text.parquet")
    table = full.drop(columns=["text"])
    table.to_parquet(OUT / "pairs.parquet")
    print("pairs validated:", table["pair_id"].nunique(), "pairs | split counts (pairs):",
          table[table.role == "vanilla"]["split"].value_counts().to_dict())
    print("token_count (adversarial) min/median/max:", int(table[table.role == "adversarial"].token_count.min()),
          int(table[table.role == "adversarial"].token_count.median()), int(table[table.role == "adversarial"].token_count.max()))
    return full, tactics_source


# Layer views of the noise floor; the same tags pilot_analysis.py uses for its figures
NOISE_VIEWS = {
    "postnorm_last": "last hidden_states entry (post-final-norm)",
    "layer_minus2": "hidden_states[-2] (second-to-last decoder layer, no final norm)",
    "prenorm_last": "input to the final norm (last decoder layer, pre-final-norm)",
}


def noise_floor_summary(singles, batches, n):
    """Batch-size-1 vs batch-size-16 differences for every layer view.

    ``singles``: list of {view: array (1, hidden)}, one per measured prompt, in prompt order.
    ``batches``: list of {view: array (16, hidden)} in the same prompt order; only the first n rows are compared.
    Returns {view: {layer, cosine_distance: {mean, median, max, ci95}, relative_l2: {...}}}. The CI is a 95%
    bootstrap over prompts (2000 resamples, seed 0).
    """
    out = {}
    for view, label in NOISE_VIEWS.items():
        a = np.concatenate([s[view] for s in singles])[:n]
        b = np.concatenate([bt[view] for bt in batches])[:n]
        assert a.shape == b.shape, f"noise floor shape mismatch for {view}"
        summ = {}
        for name, vals in (("cosine_distance", lib.cosine_distance(a, b)), ("relative_l2", lib.relative_l2(a, b))):
            _, lo, hi = lib.bootstrap_ci(vals, np.mean)
            summ[name] = dict(mean=float(vals.mean()), median=float(np.median(vals)), max=float(vals.max()), ci95=[lo, hi])
        out[view] = dict(layer=label, **summ)
    return out


def main(allow_missing_tactics=False):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(SEED)
    assert torch.cuda.is_available(), "STOP: no GPU. Switch the Colab runtime to GPU (L4)."
    gpu = torch.cuda.get_device_name(0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    full, tactics_source = build_pairs(tok, allow_missing_tactics)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", token=os.environ.get("HF_TOKEN"))
    model.eval()
    max_pos = int(model.config.max_position_embeddings)  # the model's own limit, not a chosen threshold
    max_len = int(full["token_count"].max())
    print(f"longest prompt {max_len} tokens | model limit {max_pos}")
    if max_len > max_pos:
        sys.exit("STOP: a prompt is longer than the model's context; not truncating silently.")

    # In Hugging Face decoders hidden_states = [embeddings, out layer 0, ..., out layer N-2, final_norm(out layer N-1)].
    # The last layer's output BEFORE the final norm is not in that tuple, so capture it with a hook on the final norm.
    final_norm = model.get_decoder().norm
    captured = {}

    def grab(module, inputs, output):
        # last position of every row in the batch, shape (batch, hidden); batch is 1 except in the noise floor
        captured["pre"] = inputs[0][:, -1].detach()
        captured["post"] = output[:, -1].detach()

    hook = final_norm.register_forward_hook(grab)
    refusal_ids, comply_ids = first_token_ids(tok, REFUSAL_WORDS), first_token_ids(tok, COMPLY_WORDS)
    t_hash = template_hash(tok)
    t0 = time.time()
    tensors, instr, prenorm, index, nexttok = {}, {}, {}, [], []
    n_last_is_norm_out = n_minus2_is_prenorm = 0
    for _, r in full.iterrows():
        p = tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True)
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")  # no truncation
        n_tok = int(enc["input_ids"].shape[1])
        assert n_tok == r["token_count"], "token count mismatch"
        li, ids_off = last_instruction_index(tok, p, r["text"])
        if li is None or ids_off != enc["input_ids"][0].tolist():
            sys.exit(f"STOP: could not locate the last instruction token for pair {r['pair_id']} ({r['role']}).")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        n_last_is_norm_out += int(torch.equal(hs[-1][0, -1], captured["post"][0]))
        n_minus2_is_prenorm += int(torch.equal(hs[-2][0, -1], captured["pre"][0]))
        key = lib.make_key(r["pair_id"], r["role"])
        tensors[key] = torch.stack([h[0, -1] for h in hs]).to(torch.bfloat16).cpu().contiguous()  # last prefill
        instr[key] = torch.stack([h[0, li] for h in hs]).to(torch.bfloat16).cpu().contiguous()     # last instruction
        prenorm[key] = captured["pre"][0].to(torch.bfloat16).cpu().reshape(1, -1).contiguous()
        probs = torch.softmax(out.logits[0, -1].float(), dim=-1)
        top = torch.topk(probs, 20)
        nexttok.append(dict(key=key, top20_ids=top.indices.cpu().tolist(), top20_probs=top.values.cpu().tolist(),
                            refusal_mass=float(probs[refusal_ids].sum()), comply_mass=float(probs[comply_ids].sum())))
        index.append(dict(key=key, pair_id=r["pair_id"], role=r["role"], n_tokens=n_tok, n_hidden_states=len(hs),
                          last_instruction_index=li, last_instruction_offset=li - n_tok, truncated=False,
                          template_hash=t_hash, backend="hf", concurrency=1, dtype="bfloat16", quantization="none"))
    vec = tensors[key]
    n = len(tensors)
    offsets = pd.Series([x["last_instruction_offset"] for x in index]).value_counts().sort_index()
    print("last-instruction offset from the end (offset: count):", offsets.to_dict())
    print(f"check: hidden_states[-1] == final-norm output for {n_last_is_norm_out}/{n} prompts; "
          f"hidden_states[-2] == input to final norm for {n_minus2_is_prenorm}/{n} prompts")
    if n_last_is_norm_out != n:
        hook.remove()
        sys.exit("STOP: hidden_states[-1] is not the final-norm output here; the post-norm label would be wrong.")
    import transformers

    backend = (f"transformers {transformers.__version__} + torch {torch.__version__} (CUDA {torch.version.cuda}), "
               f"attn_implementation={getattr(model.config, '_attn_implementation', 'unknown')}, batch size 1")
    meta = dict(
        model=MODEL_ID, dtype="bfloat16", backend=backend, gpu=gpu, seed=str(SEED),
        concurrency="1 (HF backend, one prompt at a time, no padding)", quantization="none",
        undercurrent_version="not used (plain transformers; pilot only)", template_hash=t_hash,
        position="pilot_acts: prompt[-1] (last prefill); pilot_acts_instr: last token of the user instruction",
        next_token="pilot_nexttoken.parquet: top-20 ids/probs and crude refusal/comply mass at prompt[-1] [PROPOSED token sets]",
        system_prompt="Qwen2.5 template adds its DEFAULT system block; kept and recorded here",
        hidden_states_0="embedding output",
        hidden_states_last=f"AFTER the final norm (index -1); verified equal to the final-norm output for {n_last_is_norm_out}/{n} prompts",
        hidden_states_minus2=("output of the second-to-last decoder layer, no final norm (index -2); NOT the last layer before "
                              f"the norm. Equal to the final-norm input for {n_minus2_is_prenorm}/{n} prompts"),
        prenorm_file="pilot_prenorm_last.safetensors: input to the final norm (= last layer output BEFORE the norm), shape (1, hidden)",
        n_hidden_states=str(len(hs)), hidden=str(vec.shape[1]), note="pilot data, ~100 pairs",
        tactics_source=tactics_source,
    )
    save_file(tensors, str(OUT / "pilot_acts.safetensors"), metadata=meta)
    save_file(instr, str(OUT / "pilot_acts_instr.safetensors"), metadata=meta)
    save_file(prenorm, str(OUT / "pilot_prenorm_last.safetensors"), metadata=meta)
    pd.DataFrame(nexttok).to_parquet(OUT / "pilot_nexttoken.parquet")
    pd.DataFrame(index).to_parquet(OUT / "pilot_index.parquet")
    (OUT / "pilot_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"extracted {len(tensors)} vectors of shape {tuple(vec.shape)} in {time.time() - t0:.0f}s")

    # d) rough noise floor: bs 1 vs bs 16 (left padding), prompt[-1], 50 vanilla prompts, for the three layer views.
    # position_ids come from the attention mask so left padding does not shift positions
    # (otherwise we would measure a padding artefact, not numerical noise).
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vanilla = full[full.role == "vanilla"].sample(frac=1, random_state=SEED)
    n_batches = -(-N_NOISE // 16)
    measured = vanilla.iloc[:N_NOISE]
    filler = vanilla.iloc[N_NOISE:N_NOISE + n_batches * 16 - N_NOISE]  # fills the last batch to exactly 16; not compared
    assert len(measured) + len(filler) == n_batches * 16, "not enough vanilla prompts to fill every batch to 16"
    render = lambda t: tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)  # noqa: E731
    prompts = [render(t) for t in measured["text"]] + [render(t) for t in filler["text"]]

    def last_states(batch):
        """prompt[-1] vectors of every row, for each layer view (the pre-norm one comes from the final-norm hook)."""
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        pos = (enc["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            out = model(**enc, position_ids=pos, output_hidden_states=True)
        return {"postnorm_last": out.hidden_states[-1][:, -1].float().cpu().numpy(),
                "layer_minus2": out.hidden_states[-2][:, -1].float().cpu().numpy(),
                "prenorm_last": captured["pre"].float().cpu().numpy()}

    singles = [last_states([p]) for p in prompts[:N_NOISE]]
    batches = [last_states(prompts[i:i + 16]) for i in range(0, len(prompts), 16)]
    hook.remove()
    per_view = noise_floor_summary(singles, batches, N_NOISE)
    post = per_view["postnorm_last"]
    # top-level cosine_distance / relative_l2 stay the post-norm last layer (what earlier code reads); per_view adds
    # the same numbers for every layer view
    nf = dict(label="pilot-only rough floor", n=N_NOISE, layer=post["layer"], model=MODEL_ID,
              dtype="bfloat16", gpu=gpu, backend=backend,
              batch_sizes=f"1 vs 16 (left padding, position_ids from attention mask; {len(filler)} filler prompts complete the last batch and are not compared)",
              cosine_distance=post["cosine_distance"], relative_l2=post["relative_l2"], per_view=per_view)
    (OUT / "noise_floor.json").write_text(json.dumps(nf, indent=2))
    print("noise floor (pilot-only rough floor), mean per layer view:",
          json.dumps({v: {k: per_view[v][k]["mean"] for k in ("cosine_distance", "relative_l2")} for v in per_view}))
    print("DONE. Token counts logged; nothing truncated; no text printed.")


if __name__ == "__main__":
    allow = "--allow-missing-tactics" in sys.argv  # PILOT ONLY
    if "--check-columns" in sys.argv:  # CPU-only pre-check: no model, no GPU
        load_and_check(allow)
        print("column check PASSED")
    else:
        main(allow)
