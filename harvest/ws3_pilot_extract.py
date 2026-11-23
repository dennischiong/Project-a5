"""WS3 pilot extraction: Qwen2.5-1.5B-Instruct on a free-tier T4, every unique prompt of the pairs table v1.1
(WildJailbreak + SORRY-Bench), so A2 per tactic, A5 and A8 can run on REAL activations with tactic labels before the
full harvests. NEEDS A GPU. Pilot only: the official harvests use the shared harness (Undercurrent).

Prints ONLY ids, counts and numbers: never prompt text or model replies. Text is read from the pairs table into memory
and is never written anywhere (prompts are identified by a sha256 hash).

Per prompt (HF backend, batch 1, no padding, one forward pass, chat template with add_generation_prompt=True):
  prenorm   last decoder layer output BEFORE the final norm at prompt[-1] (= Undercurrent residual_stream), float32
  postnorm  hidden_states[-1] at prompt[-1] (after the final norm), float32
  sweep     hidden_states[l] at prompt[-1] for l in SWEEP_LAYERS (exploratory layer sweep), float16
  next-token refusal / compliance mass at prompt[-1] (first-word proxy, see analysis/ws3_pilot.py)

Outputs in --out: prompts.parquet, pairs.parquet, prenorm_last.npy, postnorm_last.npy, sweep.npy, meta.json.
Resumable: every CHUNK prompts are saved to --out/chunks/; a rerun skips finished chunks (same prompt list only).

  python harvest/ws3_pilot_extract.py --pairs <pairs table CSV> \
      --out $A5_WORK_ROOT/ws3_pilot/qwen25-1p5b [--dtype float16] [--limit 100]
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import ws3_pilot as wp  # noqa: E402
from seeds import set_seed  # noqa: E402

CHUNK = 1000
SWEEP_LAYERS = [0, 4, 8, 12, 16, 20, 24]   # hidden_states indices (0 = embeddings); the last layer is in prenorm


def first_token_ids(tok, words):
    ids = []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if t and t[0] not in ids:
            ids.append(t[0])
    return ids


def load_model(dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(wp.MODEL_ID)
    td = getattr(torch, dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(wp.MODEL_ID, dtype=td, attn_implementation="sdpa")
    except TypeError:   # older transformers
        model = AutoModelForCausalLM.from_pretrained(wp.MODEL_ID, torch_dtype=td, attn_implementation="sdpa")
    if model.dtype != td:   # a transformers version that ignored the keyword: cast explicitly
        model = model.to(td)
    return tok, model.to("cuda").eval()


def run_chunk(tok, model, texts, ref_ids, comp_ids):
    """One forward pass per prompt. Returns arrays for the chunk (numbers only)."""
    import torch
    cap = {}
    hook = model.model.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("pre", i[0][0, -1].float()))
    n, d = len(texts), model.config.hidden_size
    out = dict(prenorm=np.zeros((n, d), np.float32), postnorm=np.zeros((n, d), np.float32),
               sweep=np.zeros((n, len(SWEEP_LAYERS), d), np.float16), n_tokens=np.zeros(n, np.int32),
               refusal_mass=np.zeros(n, np.float32), compliance_mass=np.zeros(n, np.float32), finite=np.zeros(n, bool))
    try:
        with torch.no_grad():
            for i, text in enumerate(texts):
                rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                   add_generation_prompt=True)
                ids = tok(rendered, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
                try:
                    o = model(ids, output_hidden_states=True, logits_to_keep=1)
                except TypeError:
                    o = model(ids, output_hidden_states=True)
                hs = o.hidden_states
                pre = cap["pre"].cpu().numpy()
                post = hs[-1][0, -1].float().cpu().numpy()
                sw = np.stack([hs[l][0, -1].float().cpu().numpy() for l in SWEEP_LAYERS])
                p = torch.softmax(o.logits[0, -1].float(), -1)
                out["prenorm"][i], out["postnorm"][i], out["sweep"][i] = pre, post, sw.astype(np.float16)
                out["n_tokens"][i] = ids.shape[1]
                out["refusal_mass"][i] = float(p[ref_ids].sum())
                out["compliance_mass"][i] = float(p[comp_ids].sum())
                out["finite"][i] = bool(np.isfinite(pre).all() and np.isfinite(post).all() and np.isfinite(sw).all()
                                        and np.isfinite(out["sweep"][i]).all())
    finally:
        hook.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs table v1.1 (.csv or .parquet, with text columns)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                    help="float16 is fast on a T4; bfloat16 matches the harvest dtype but is emulated (slow) on a T4")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N pairs of each source")
    a = ap.parse_args()

    import torch
    import transformers
    set_seed(0)
    if not torch.cuda.is_available():
        sys.exit("STOP: no GPU (Runtime > Change runtime type > T4 GPU)")
    out = Path(a.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)

    pairs = pd.read_parquet(a.pairs) if a.pairs.endswith(".parquet") else pd.read_csv(a.pairs)
    prompts, plan = wp.prompt_plan(pairs)
    if a.limit:   # smoke test: the first N pairs of each source (by pair_id), and only their prompts
        plan = plan.sort_values("pair_id").groupby("source", sort=False).head(a.limit).reset_index(drop=True)
        need = set(plan["key_v"]) | set(plan["key_a"])
        prompts = [kt for kt in prompts if kt[0] in need]
    keys = [k for k, _ in prompts]
    kh = wp.keys_hash(keys)
    kfile = out / "chunks" / "keys.json"
    if kfile.exists() and json.loads(kfile.read_text())["keys_hash"] != kh:
        sys.exit(f"STOP: {out}/chunks holds chunks from a different prompt list; use a new --out folder")
    kfile.write_text(json.dumps(dict(keys_hash=kh, n_prompts=len(keys), chunk=CHUNK)))
    print(f"pairs: {plan['source'].value_counts().to_dict()} | unique prompts: {len(keys)} | keys {kh} | dtype {a.dtype}")

    tok, model = load_model(a.dtype)
    ref_ids, comp_ids = first_token_ids(tok, wp.REFUSAL_WORDS), first_token_ids(tok, wp.COMPLIANCE_WORDS)
    gpu = torch.cuda.get_device_name(0)
    print(f"model {wp.MODEL_ID} on {gpu} | layers {model.config.num_hidden_layers} | hidden {model.config.hidden_size}")

    n_chunks = (len(keys) + CHUNK - 1) // CHUNK
    t0, done = time.time(), 0
    for c in range(n_chunks):
        f = out / "chunks" / f"chunk_{c:03d}.npz"
        if f.exists():
            print(f"chunk {c + 1}/{n_chunks}: already on Drive, skipped")
            continue
        texts = [t for _, t in prompts[c * CHUNK:(c + 1) * CHUNK]]
        tc = time.time()
        arr = run_chunk(tok, model, texts, ref_ids, comp_ids)
        with tempfile.TemporaryDirectory() as td:   # write locally, then copy: no half-written file on Drive
            tmp = Path(td) / f.name
            np.savez(tmp, **arr)
            shutil.copy(tmp, f)
        done += len(texts)
        rate = (time.time() - t0) / done
        left = sum(min(CHUNK, len(keys) - k * CHUNK) for k in range(c + 1, n_chunks)
                   if not (out / "chunks" / f"chunk_{k:03d}.npz").exists())
        print(f"chunk {c + 1}/{n_chunks}: {len(texts)} prompts in {time.time() - tc:.0f} s | non-finite "
              f"{int((~arr['finite']).sum())} | max tokens {int(arr['n_tokens'].max())} | ETA {left * rate / 60:.1f} min")

    # assemble
    parts = [dict(np.load(out / "chunks" / f"chunk_{c:03d}.npz")) for c in range(n_chunks)]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert len(cat["prenorm"]) == len(keys)
    row = {k: i for i, k in enumerate(keys)}
    plan = plan[plan["key_v"].isin(row) & plan["key_a"].isin(row)].reset_index(drop=True)
    plan["row_v"], plan["row_a"] = plan["key_v"].map(row), plan["key_a"].map(row)
    prompts_df = pd.DataFrame(dict(prompt_key=keys, n_tokens=cat["n_tokens"], refusal_mass=cat["refusal_mass"],
                                   compliance_mass=cat["compliance_mass"], finite=cat["finite"]))
    meta = dict(model=wp.MODEL_ID, dtype=a.dtype, gpu=gpu, torch=torch.__version__, transformers=transformers.__version__,
                backend="HF transformers, attn sdpa, batch 1, no padding, one forward pass", position="prompt[-1]",
                num_hidden_layers=model.config.num_hidden_layers, hidden_size=model.config.hidden_size,
                sweep_layers=SWEEP_LAYERS, prenorm="input of the final norm (last decoder layer, = residual_stream)",
                postnorm="hidden_states[-1] (after the final norm)",
                template_sha256=hashlib.sha256((tok.chat_template or "").encode()).hexdigest()[:16],
                system_prompt="Qwen2.5 template adds its default system block; kept", keys_hash=kh,
                n_prompts=len(keys), n_pairs=plan["source"].value_counts().to_dict(),
                n_nonfinite=int((~cat["finite"]).sum()), max_tokens=int(cat["n_tokens"].max()), truncation="none",
                refusal_words=list(wp.REFUSAL_WORDS), refusal_token_ids=ref_ids,
                compliance_words=list(wp.COMPLIANCE_WORDS), compliance_token_ids=comp_ids,
                pairs_table=a.pairs, pairs_table_sha256=hashlib.sha256(Path(a.pairs).read_bytes()).hexdigest()[:16],
                limit=a.limit or None, seed=0, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        np.save(td / "prenorm_last.npy", cat["prenorm"])
        np.save(td / "postnorm_last.npy", cat["postnorm"])
        np.save(td / "sweep.npy", cat["sweep"])
        prompts_df.to_parquet(td / "prompts.parquet", index=False)
        plan.drop(columns=["key_v", "key_a"]).to_parquet(td / "pairs.parquet", index=False)
        (td / "meta.json").write_text(json.dumps(meta, indent=2))
        for p in td.iterdir():
            shutil.copy(p, out / p.name)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min: {len(keys)} prompts, {len(plan)} pairs, "
          f"{meta['n_nonfinite']} non-finite, max {meta['max_tokens']} tokens, nothing truncated, no text written -> {out}")


if __name__ == "__main__":
    main()
