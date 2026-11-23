"""Rerun only the prompts a finished harvest lost to GPU memory (Mistral-7B C1: 244 of 4,000 on a T4). Needs a GPU.

Same model, config and hooks as harvest/ws3_harvest.py. NEVER changes the input folder: it writes a full copy, with the
lost rows filled, to --out (default ../ws3_oom_fill/<name>, outside ws3_harvest/ so regenerate_all does not count the run
twice). Replacing the folder in ws3_harvest/ is a separate manual step, because it changes numbers in the paper.

The filled rows come from a different GPU than the rest, so --check reruns N prompts that already have vectors and reports
how far the new vectors are from the stored ones (relative L2). The result is stored in meta.json under fill.

  python harvest/ws3_fill_lost.py --folder ../ws3_harvest/mistral-7b__wj-sorry__hf-bf16-bs1__sub4000
  python harvest/ws3_fill_lost.py --folder <same> --max-fill 5     # timing test (writes a __smoke folder)
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ROW_FIELDS = ("prenorm", "postnorm", "sweep", "n_tokens", "refusal_mass", "compliance_mass", "finite")


# ------------------------------------------------------------------------------------------------ pure helpers
def apply_fill(arrays, fill, rows):
    """Copy of arrays (dict of per-prompt arrays) with rows replaced by fill's values. A row only takes new vectors if it
    came back finite; one that ran out of memory again keeps its old values and stays non-finite."""
    out = {k: v.copy() for k, v in arrays.items()}
    rows = np.asarray(rows)
    ok = np.asarray(fill["finite"], bool)
    for field in ROW_FIELDS:
        out[field][rows[ok]] = np.asarray(fill[field])[ok]
    return out


def relative_l2(new, old):
    """Per row: ||new - old|| / ||old||."""
    new, old = np.asarray(new, float), np.asarray(old, float)
    return np.linalg.norm(new - old, axis=1) / np.linalg.norm(old, axis=1)


def check_rows(finite, n):
    """The first n rows that already have vectors (deterministic)."""
    return np.flatnonzero(np.asarray(finite, bool))[:n]


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="a finished harvest/ws3_harvest.py folder")
    ap.add_argument("--pairs", default=str(REPO / "data" / "pairs_table_v1_final-2.csv"))
    ap.add_argument("--out", default="")
    ap.add_argument("--max-fill", type=int, default=0, help="timing test: only the first N lost prompts")
    ap.add_argument("--check", type=int, default=20, help="prompts with vectors to rerun as a comparison")
    a = ap.parse_args()

    import pandas as pd
    import torch
    import transformers
    from harvest import ws3_generate as wg
    from harvest import ws3_harvest as wh

    src = Path(a.folder)
    meta = json.loads((src / "meta.json").read_text())
    if wg.file_sha(a.pairs) != meta["pairs_table_sha256"]:
        sys.exit("STOP: the pairs table differs from the one this harvest used")
    prompts_df = pd.read_parquet(src / "prompts.parquet")
    prompts, _ = wg.select_prompts(wg.read_pairs(a.pairs), meta.get("limit") or 0, meta.get("subset") or 0)
    keys = prompts_df["prompt_key"].tolist()
    if [k for k, _ in prompts] != keys:
        sys.exit("STOP: the rebuilt prompt list does not match this harvest; nothing was run")
    texts = dict(prompts)

    lost = np.flatnonzero(~prompts_df["finite"].to_numpy(bool))
    if a.max_fill:
        lost = lost[:a.max_fill]
    if not len(lost):
        print("nothing to fill: every prompt is finite")
        return
    name = src.name + (f"__smoke{a.max_fill}" if a.max_fill else "")
    out = Path(a.out) if a.out else REPO.parent / "ws3_oom_fill" / name
    if out.resolve() == src.resolve():
        sys.exit("STOP: --out is the input folder; the input is never changed")
    if (out / "meta.json").exists():
        print(f"already filled: {out}")
        return

    if not torch.cuda.is_available():
        sys.exit("STOP: no GPU")
    gpu = torch.cuda.get_device_name(0)
    spec = wh.MODELS[meta["model_tag"]]
    tok, model = wh.load_model(spec, meta["config"], meta["dtype"])
    ref_ids, comp_ids = wh.wp_first_ids(tok)
    if ref_ids != meta["refusal_token_ids"] or comp_ids != meta["compliance_token_ids"]:
        sys.exit("STOP: proxy token ids differ from the harvest's; the tokenizer changed")
    layers = list(meta["sweep_layers"])

    arrays = dict(prenorm=np.load(src / "prenorm_last.npy"), postnorm=np.load(src / "postnorm_last.npy"),
                  sweep=np.load(src / "sweep.npy"), n_tokens=prompts_df["n_tokens"].to_numpy(),
                  refusal_mass=prompts_df["refusal_mass"].to_numpy(), compliance_mass=prompts_df["compliance_mass"].to_numpy(),
                  finite=prompts_df["finite"].to_numpy(bool))
    print(f"{meta['model_tag']} {meta['config']} on {gpu} | lost prompts to fill {len(lost)} of {len(keys)} | "
          f"original GPU {meta['gpu']} -> {out}")

    check = {}
    if a.check:
        rows = check_rows(arrays["finite"], a.check)
        chk = wh.run_chunk(tok, model, [texts[keys[i]] for i in rows], layers, ref_ids, comp_ids)
        good = np.asarray(chk["finite"], bool)
        d = relative_l2(chk["prenorm"][good], arrays["prenorm"][rows[good]])
        check = dict(n=int(good.sum()), median_relative_l2=float(np.median(d)), max_relative_l2=float(np.max(d)))
        print(f"comparison with stored vectors (pre-norm): {check}")

    t0 = time.time()
    fill = wh.run_chunk(tok, model, [texts[keys[i]] for i in lost], layers, ref_ids, comp_ids)
    done = apply_fill(arrays, fill, lost)
    n_filled = int(np.asarray(fill["finite"], bool).sum())
    prompts_df = prompts_df.copy()
    for col in ("n_tokens", "refusal_mass", "compliance_mass", "finite"):
        prompts_df[col] = done[col]
    n_nonfinite = int((~done["finite"]).sum())
    meta = dict(meta, gpu=f"{meta['gpu']}; lost prompts on {gpu}", n_nonfinite=n_nonfinite, n_oom=n_nonfinite,
                max_tokens=int(done["n_tokens"].max()),
                fill=dict(source_folder=src.name, n_attempted=int(len(lost)), n_filled=n_filled, n_still_lost=n_nonfinite,
                          filled_row_gpu=gpu, torch=torch.__version__, transformers=transformers.__version__,
                          check=check, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    out.mkdir(parents=True, exist_ok=True)
    for arr, fname in ((done["prenorm"], "prenorm_last.npy"), (done["postnorm"], "postnorm_last.npy"), (done["sweep"], "sweep.npy")):
        np.save(out / fname, arr)
    shutil.copy(src / "pairs.parquet", out / "pairs.parquet")
    prompts_df.to_parquet(out / "prompts.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))   # last: its presence marks a finished folder
    print(f"DONE in {(time.time() - t0) / 60:.1f} min: filled {n_filled} of {len(lost)} | still non-finite {n_nonfinite} | "
          f"input folder unchanged -> {out}")


if __name__ == "__main__":
    main()
