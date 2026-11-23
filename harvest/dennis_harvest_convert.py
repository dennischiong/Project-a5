"""Convert Dennis's Qwen2.5-7B harvest (.tpxz archives + pair map, 16 Oct, config hf-nf4-c6) into the WS3 extraction
folder format that analysis/ws3_pilot.run_all and analysis/regenerate_all read. CPU only.

What the harvest holds (checked 28 Oct with Colab Notebooks/scratch/inspect_qwen7b_harvest.ipynb):
  <name>.tpxz      one shard: activations.safetensors (500 prompts per tensor, fp16) + index.parquet (prompt_id per row)
  tensors          last_prefill.layer_<0..27> = decoder-layer output, pre final norm (metadata residual_stream);
                   last_instruction.layer_<..> (14 layers); gen_<k>.layer_<..> are invalid (max_new_tokens = 1)
  pair_maps/pair_map__<table sha8>.parquet   pair_id, role (vanilla / adv), prompt_id
So layer 27 at last_prefill is the pre-registered vector (prompt[-1], last layer before the final norm). There is no
post-norm vector and no next-token stats, so the folder has the prenorm view only and no refusal proxy (A8 angle).

Output folder (same files as harvest/ws3_pilot_extract.py, numbers only):
  prompts.parquet    prompt_key (= Dennis's prompt_id), n_tokens, refusal_mass / compliance_mass (NaN), finite
  pairs.parquet      pair_id, source, intent, tactic, split, holdout_group, row_v, row_a
  prenorm_last.npy   (prompts, hidden) float32
  meta.json          model, model_tag, config c6, serving config from the shard index, counts, table hash

Standalone on purpose (numpy, pandas, safetensors, tar + pixz): it also runs pasted into a Colab cell of the Google
account that can see the harvest, without the rest of the repo. Prints only ids, counts and numbers.

    python harvest/dennis_harvest_convert.py --harvest "<folder>" --pairs <pairs_table_v1_final-2.csv> --out <folder> [--sweep-every 4]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = "qwen25-7b"
DATASETS = {"wildjailbreak": "WildJailbreak", "sorry-bench": "SORRY-Bench"}   # archive name -> pairs-table source
POINT, LAYER, N_LAYERS = "last_prefill", 27, 28
TABLE_COLS = ["pair_id", "source", "intent", "tactic", "split", "holdout_group"]   # never a text column


def archive_dataset(name):
    """'qwen25-7b__sorry-bench-s00003__hf-nf4-c6__e7f5ea94.tpxz' -> 'sorry-bench'."""
    return name.split("__")[1].rsplit("-s", 1)[0]


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def restore_archive(archive, work):
    """Copy one archive to local disk, check its sha256 against the manifest (<name>.json, written last by
    drive_archive), unpack it with pixz, return the folder that holds activations.safetensors."""
    archive, work = Path(archive), Path(work)
    manifest = json.loads(Path(f"{archive}.json").read_text())
    local = work / archive.name
    shutil.copy(archive, local)
    if sha256_file(local) != manifest["sha256"]:
        raise ValueError(f"sha256 mismatch for {archive.name}: archive differs from its manifest")
    if shutil.which("pixz") is None:
        subprocess.run("apt-get -qq update && apt-get -qq install -y pixz", shell=True, check=True)
    subprocess.run(["tar", "-I", "pixz", "-xf", str(local), "-C", str(work)], check=True)
    local.unlink()
    found = sorted(p.parent for p in work.rglob("activations.safetensors"))
    if len(found) != 1:
        raise ValueError(f"{archive.name}: expected one activations.safetensors, found {len(found)}")
    return found[0]


def read_shard(shard_dir, point=POINT, layer=LAYER):
    """One restored shard -> DataFrame (prompt_id, n_tokens, finite) and its (rows, hidden) float32 vectors.
    index.parquet row i is tensor row i; rows the harvester skipped as too long count as not finite."""
    from safetensors import safe_open
    shard_dir = Path(shard_dir)
    idx = pd.read_parquet(shard_dir / "index.parquet")
    with safe_open(str(shard_dir / "activations.safetensors"), framework="np") as f:
        X = np.asarray(f.get_tensor(f"{point}.layer_{layer}"), np.float32)
        meta = dict(f.metadata() or {})
    if len(X) != len(idx):
        raise ValueError(f"{shard_dir.name}: {len(X)} tensor rows vs {len(idx)} index rows")
    ok = np.isfinite(X).all(1)
    if "skipped_too_long" in idx:
        ok &= ~idx["skipped_too_long"].to_numpy(bool)
    rows = pd.DataFrame({"prompt_id": idx["prompt_id"].astype(str),
                         "n_tokens": idx["token_count"].astype(int) if "token_count" in idx else -1,
                         "finite": ok})
    serving = {c: str(idx[c].iloc[0]) for c in ("backend", "concurrency", "dtype", "quantization", "gpu", "spec_hash")
               if c in idx and len(idx)}
    return rows, X, {**serving, **{k: meta[k] for k in ("residual_stream", "model", "transformers", "torch") if k in meta}}


def sweep_layers(every, n_layers=N_LAYERS):
    """Exploratory layer sweep in the hidden_states convention of harvest/ws3_harvest.py: index l = output of decoder
    layer l (1-based), i.e. Dennis's layer_<l-1>. No embeddings (index 0) in the harvest; the last layer is the
    prenorm view already. every <= 0: no sweep."""
    return list(range(every, n_layers, every)) if every > 0 else []


def read_sweep(shard_dir, layers, point=POINT):
    """One restored shard -> (rows, len(layers), hidden) float16 at the sweep layers (hidden_states indices)."""
    from safetensors import safe_open
    with safe_open(str(Path(shard_dir) / "activations.safetensors"), framework="np") as f:
        return np.stack([np.asarray(f.get_tensor(f"{point}.layer_{l - 1}"), np.float16) for l in layers], axis=1)


def plan_pairs(pair_map, table, sources):
    """Pairs table (ids and labels only) + pair map -> one row per usable pair with prompt_v / prompt_a, and counts.
    Dropped: pairs not in the map or missing a role, and pairs whose two prompts are the same prompt (e.g. the
    SORRY-Bench base rows: no displacement by construction)."""
    t = table[table["source"].isin(sources)][TABLE_COLS].copy()
    for c in TABLE_COLS:
        t[c] = t[c].fillna("").astype(str)
    roles = pair_map.assign(pair_id=pair_map["pair_id"].astype(str), prompt_id=pair_map["prompt_id"].astype(str))
    wide = roles.pivot_table(index="pair_id", columns="role", values="prompt_id", aggfunc="first")
    wide = wide.reindex(columns=["vanilla", "adv"]).rename(columns={"vanilla": "prompt_v", "adv": "prompt_a"})
    p = t.merge(wide, left_on="pair_id", right_index=True, how="left")
    counts = dict(table_pairs=len(t), not_in_map=int(p[["prompt_v", "prompt_a"]].isna().any(axis=1).sum()))
    p = p.dropna(subset=["prompt_v", "prompt_a"])
    same = p["prompt_v"] == p["prompt_a"]
    counts["identical_prompts"] = int(same.sum())
    return p[~same].reset_index(drop=True), counts


def build_folder(plan, shards, out_dir, meta):
    """plan (from plan_pairs) + shards [(rows DataFrame, vectors) or (rows, vectors, sweep)] -> the WS3 extraction
    folder; sweep.npy is written when every shard has a sweep (meta[sweep_layers] names its layers). Pairs with a
    prompt that no shard holds are dropped and counted. Returns the meta dict written to meta.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.concat([s[0] for s in shards], ignore_index=True) if shards else pd.DataFrame(columns=["prompt_id"])
    X = np.concatenate([s[1] for s in shards]) if shards else np.zeros((0, 0), np.float32)
    S = np.concatenate([s[2] for s in shards]) if shards and all(len(s) == 3 for s in shards) else None
    first = (~rows["prompt_id"].duplicated()).to_numpy()   # a prompt harvested twice keeps its first copy
    rows, X = rows[first].reset_index(drop=True), X[first]
    if S is not None:
        S = S[first]
        # like ws3_harvest: a prompt with a non-finite sweep vector is marked non-finite rather than clipped
        rows["finite"] = rows["finite"].to_numpy(bool) & np.isfinite(S).all(axis=(1, 2))
    have = set(rows["prompt_id"])
    keep = plan["prompt_v"].isin(have) & plan["prompt_a"].isin(have)
    plan = plan[keep].reset_index(drop=True)
    used = sorted(set(plan["prompt_v"]) | set(plan["prompt_a"]))
    pos = pd.Series(np.arange(len(rows)), index=rows["prompt_id"])
    take = pos[used].to_numpy()
    prompts = pd.DataFrame({"prompt_key": used, "n_tokens": rows["n_tokens"].to_numpy()[take].astype(int),
                            "refusal_mass": np.nan, "compliance_mass": np.nan,
                            "finite": rows["finite"].to_numpy()[take].astype(bool)})
    row_of = pd.Series(np.arange(len(used)), index=used)
    pairs = plan[TABLE_COLS].assign(row_v=row_of[plan["prompt_v"]].to_numpy(), row_a=row_of[plan["prompt_a"]].to_numpy())
    H = X[take].astype(np.float32)
    prompts.to_parquet(out / "prompts.parquet", index=False)
    pairs.to_parquet(out / "pairs.parquet", index=False)
    np.save(out / "prenorm_last.npy", H)
    if S is not None:
        np.save(out / "sweep.npy", S[take])
    meta = dict(meta, n_prompts=len(used), n_pairs=pairs["source"].value_counts().to_dict(),
                dropped_prompt_not_harvested=int((~keep).sum()), n_nonfinite=int((~prompts["finite"]).sum()),
                hidden_size=int(H.shape[1]) if H.size else 0)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def main(harvest, pairs_csv, out_dir, datasets=tuple(DATASETS), work_root="/content/dennis_restore", sweep_every=0):
    harvest = Path(harvest)
    layers = sweep_layers(sweep_every)
    archives = sorted(a for a in harvest.glob("*.tpxz") if archive_dataset(a.name) in datasets)
    pm_files = sorted((harvest / "pair_maps").glob("*.parquet"))
    if not archives or not pm_files:
        raise SystemExit(f"no archives for {list(datasets)} or no pair map in {harvest}")
    table = pd.read_csv(pairs_csv, usecols=TABLE_COLS)                    # ids and labels only, no text column
    table_sha8 = sha256_file(pairs_csv)[:8]
    pm_sha8 = pm_files[0].stem.rsplit("__", 1)[-1]
    if pm_sha8 != table_sha8:
        raise SystemExit(f"pair map is for table {pm_sha8}, but {Path(pairs_csv).name} is {table_sha8}")
    plan, counts = plan_pairs(pd.read_parquet(pm_files[0]), table, [DATASETS[d] for d in datasets])
    print(f"pairs: {counts} -> {len(plan):,} usable before the harvest check")
    t0, shards, serving = time.time(), [], {}
    Path(work_root).mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(archives, 1):
        with tempfile.TemporaryDirectory(dir=work_root) as work:
            shard_dir = restore_archive(a, work)
            rows, X, serving = read_shard(shard_dir)
            shards.append((rows, X, read_sweep(shard_dir, layers)) if layers else (rows, X))
        print(f"  [{i}/{len(archives)}] {archive_dataset(a.name)}: {len(rows)} prompts, {int((~rows['finite']).sum())} "
              f"not finite ({time.time() - t0:.0f} s)")
    meta = dict(model=MODEL_ID, model_tag=MODEL_TAG, config="c6", config_name=archives[0].name.split("__")[2],
                num_hidden_layers=N_LAYERS, point=POINT, layer=LAYER, position="prompt[-1]", views=["prenorm"],
                sweep_layers=layers, prenorm=serving.get("residual_stream", "decoder-layer output, pre final norm"),
                postnorm="not stored", refusal_proxy="not stored (no next-token stats in the harvest)",
                serving=serving, source=f"Dennis's harvest, {len(archives)} archives ({', '.join(datasets)})",
                pairs_table=str(pairs_csv), pairs_table_sha8=table_sha8, pair_counts=counts,
                created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    meta = build_folder(plan, shards, out_dir, meta)
    print(f"wrote {out_dir}: {meta['n_prompts']:,} prompts, pairs {meta['n_pairs']}, "
          f"{meta['dropped_prompt_not_harvested']} pairs without a harvested prompt, {meta['n_nonfinite']} not finite")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", required=True, help='the "Full harvest Qwen 2.5-7B activations" folder')
    ap.add_argument("--pairs", required=True, help="pairs_table_v1_final-2.csv (only id/label columns are read)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="wildjailbreak,sorry-bench")
    ap.add_argument("--sweep-every", type=int, default=0,
                    help="also write sweep.npy (float16) at every Nth hidden_states layer (4 gives 4, 8, ..., 24; about 0.5 GB for Qwen-7B); 0 = off")
    a = ap.parse_args()
    main(a.harvest, a.pairs, a.out, tuple(a.datasets.split(",")), sweep_every=a.sweep_every)
