"""WS3 prompt-position harvest for Hasif's models (Gemma-2-9B-it, Mistral-7B-Instruct-v0.3) with my own HF pipeline,
so A2, A5, A7 and A8 can run without the shared harness. NEEDS A GPU. Same output format as ws3_pilot_extract.py, so
analysis/ws3_pilot.run_all reads it unchanged.

Prints ONLY ids, counts and numbers: never prompt text or model replies. Text is read from the pairs table into memory
and is never written anywhere (prompts are identified by a sha256 hash).

Serving configs (work plan s.7.3). This pipeline is HF transformers, batch 1, no padding, one forward pass per prompt:
  c1  bf16 weights               = Track 3 C1 (HF bf16, batch 1). A full run with --config c1 is also the WS3 dataset.
  c6  4-bit NF4 weights (bnb)    = Track 3 C6. Run it on the same --subset as the C1 subset run for X5.
There is no vLLM reference (R) here: X5 compares C6 against C1 and says so.

Per prompt, at prompt[-1] (chat template with add_generation_prompt=True, BOS from the template only):
  prenorm   last decoder layer output BEFORE the final norm (= Undercurrent residual_stream), float32
  postnorm  hidden_states[-1] (after the final norm), float32
  sweep     hidden_states[l] for l in sweep_layers(n_layers, --sweep-every) (exploratory layer sweep), float16
  next-token refusal / compliance mass (first-word proxy, see analysis/ws3_pilot.py)

Outputs in --out: prompts.parquet, pairs.parquet, prenorm_last.npy, postnorm_last.npy, sweep.npy, meta.json.
Resumable: every CHUNK prompts are saved to --out/chunks/; a rerun skips finished chunks (same prompt list only).

Pairs table: Dennis's pairs_table_v1_final-2.csv (data/, the team's final table; WildJailbreak + SORRY-Bench rows).

  python harvest/ws3_harvest.py --model mistral-7b --config c1
  python harvest/ws3_harvest.py --model mistral-7b --config c6 --subset 4000
  python harvest/ws3_harvest.py --model qwen25-1p5b --config c1 --dtype float16      # the pilot model, free T4
"""
import argparse
import hashlib
import json
import os
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

CHUNK = 500
# A5_WORK_ROOT = the Drive folder that holds ws3_pilot/ and ws3_harvest/ (default: the Colab path used so far)
DRIVE_OUT = f'{os.environ.get("A5_WORK_ROOT", "/content/drive/MyDrive/a5-work")}/ws3_harvest'
# last_instruction offsets come from harvest/token_check.py (figures/offsets.csv, 7 Oct); recorded, not harvested.
MODELS = {
    "gemma2-9b": dict(id="google/gemma-2-9b-it", attn="eager", min_gib_bf16=20.0, fp16_ok=False, last_instruction=-6,
                      note="eager attention: Gemma-2 attention soft-capping is not applied by sdpa"),
    "mistral-7b": dict(id="mistralai/Mistral-7B-Instruct-v0.3", attn="sdpa", min_gib_bf16=14.0, fp16_ok=True,
                       last_instruction=-2, note=""),
    "qwen25-1p5b": dict(id="Qwen/Qwen2.5-1.5B-Instruct", attn="sdpa", min_gib_bf16=4.0, fp16_ok=True,
                        last_instruction=-6, note="plumbing check only"),
}
CONFIGS = {"c1": "hf-bf16-bs1", "c6": "hf-nf4-bs1"}
PAIRS_FINAL = str(REPO / "data" / "pairs_table_v1_final-2.csv")   # git-ignored: the table holds prompt text


def config_name(config, dtype="bfloat16"):
    """Folder / label name of a serving config; a float16 C1 run (pilot model on a T4) says so."""
    return "hf-fp16-bs1" if config == "c1" and dtype == "float16" else CONFIGS[config]


# ------------------------------------------------------------------------------------------------ pure helpers
def sweep_layers(n_layers, every):
    """hidden_states indices 0, every, 2*every, ... below n_layers (index n_layers is the post-norm output, stored
    separately; the pre-norm last layer is stored separately too)."""
    return list(range(0, n_layers, every))


def subset_plan(plan, n_prompts):
    """Track 3 evaluation subset: whole pairs (both twins), taken in sha256(pair_id) order within each
    (source, intent) stratum, round-robin across strata in proportion to their size, until the subset holds at
    least n_prompts unique prompts. Deterministic; both splits are included (work plan s.7.3)."""
    p = plan.copy()
    p["_h"] = p["pair_id"].map(lambda s: hashlib.sha256(str(s).encode()).hexdigest())
    strata = {k: g.sort_values("_h") for k, g in p.groupby(["source", "intent"], sort=True)}
    total = sum(len(g) for g in strata.values())
    taken, keys, idx = [], set(), {k: 0 for k in strata}
    while len(keys) < n_prompts and any(idx[k] < len(g) for k, g in strata.items()):
        for k, g in strata.items():
            share = max(1, round(len(g) / total * 20))   # pairs per round, proportional to the stratum's size
            for _ in range(share):
                if idx[k] >= len(g) or len(keys) >= n_prompts:
                    break
                r = g.iloc[idx[k]]
                idx[k] += 1
                taken.append(r.name)
                keys.update((r["key_v"], r["key_a"]))
    return plan.loc[sorted(taken)].reset_index(drop=True)


def size_estimate_gb(n_prompts, hidden, n_sweep):
    """Drive bytes: prenorm + postnorm in float32, the sweep in float16 (parquet/meta are negligible)."""
    return n_prompts * hidden * (2 * 4 + n_sweep * 2) / 1e9


def out_name(tag, config, subset, limit, dtype="bfloat16"):
    name = f"{tag}__wj-sorry__{config_name(config, dtype)}"
    if subset:
        name += f"__sub{subset}"
    if limit:
        name += f"__smoke{limit}"
    return name


# ------------------------------------------------------------------------------------------------ model
def load_model(spec, config, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec["id"])
    td = getattr(torch, dtype)
    kw = dict(attn_implementation=spec["attn"])
    if config == "c6":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            sys.exit("STOP: C6 needs bitsandbytes (run: pip install -q bitsandbytes), then rerun")
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=td, bnb_4bit_use_double_quant=False)
        kw["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(spec["id"], dtype=td, **kw)
    except TypeError:   # older transformers
        model = AutoModelForCausalLM.from_pretrained(spec["id"], torch_dtype=td, **kw)
    if config == "c1":
        if model.dtype != td:
            model = model.to(td)
        model = model.to("cuda")
    return tok, model.eval()


def run_chunk(tok, model, texts, layers, ref_ids, comp_ids):
    """One forward pass per prompt. Returns arrays for the chunk (numbers only)."""
    import torch
    cap = {}
    hook = model.model.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("pre", i[0][0, -1].float()))
    n, d = len(texts), model.config.hidden_size
    out = dict(prenorm=np.zeros((n, d), np.float32), postnorm=np.zeros((n, d), np.float32),
               sweep=np.zeros((n, len(layers), d), np.float16), n_tokens=np.zeros(n, np.int32),
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
                sw = np.stack([hs[l][0, -1].float().cpu().numpy() for l in layers])
                p = torch.softmax(o.logits[0, -1].float(), -1)
                out["prenorm"][i], out["postnorm"][i], out["sweep"][i] = pre, post, sw.astype(np.float16)
                out["n_tokens"][i] = ids.shape[1]
                out["refusal_mass"][i] = float(p[ref_ids].sum())
                out["compliance_mass"][i] = float(p[comp_ids].sum())
                # float16 sweep can overflow on large residuals: such a prompt is marked non-finite, never clipped
                out["finite"][i] = bool(np.isfinite(pre).all() and np.isfinite(post).all()
                                        and np.isfinite(out["sweep"][i]).all())
    finally:
        hook.remove()
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--config", default="c1", choices=sorted(CONFIGS))
    ap.add_argument("--pairs", default=PAIRS_FINAL,
                    help="pairs table with text columns (default: Dennis's pairs_table_v1_final-2.csv in data/)")
    ap.add_argument("--out", default="", help=f"default: {DRIVE_OUT}/<model>__wj-sorry__<config>[__subN]")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"],
                    help="weights (c1) or compute dtype (c6); bfloat16 is native on an L4/A100, emulated on a T4")
    ap.add_argument("--subset", type=int, default=0, help="Track 3 evaluation subset: N unique prompts (e.g. 4000)")
    ap.add_argument("--sweep-every", type=int, default=8, help="layer sweep spacing (hidden_states indices)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N pairs of each source")
    ap.add_argument("--max-gb", type=float, default=2.5, help="stop if the estimated Drive size is above this")
    a = ap.parse_args()

    import torch
    import transformers
    set_seed(0)
    spec = MODELS[a.model]
    if a.dtype == "float16" and not spec["fp16_ok"]:
        sys.exit(f"STOP: {a.model} in float16 risks overflow (NaN states); use bfloat16")
    if not torch.cuda.is_available():
        sys.exit("STOP: no GPU (Runtime > Change runtime type)")
    gpu = torch.cuda.get_device_name(0)
    gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if a.config == "c1" and gib < spec["min_gib_bf16"]:
        sys.exit(f"STOP: {a.model} in 16-bit needs about {spec['min_gib_bf16']:.0f} GiB; {gpu} has {gib:.1f} GiB "
                 f"(use an L4/A100, or --config c6)")

    pairs = pd.read_parquet(a.pairs) if a.pairs.endswith(".parquet") else pd.read_csv(a.pairs)
    prompts, plan = wp.prompt_plan(pairs)
    if a.limit:   # smoke test: the first N pairs of each source (by pair_id)
        plan = plan.sort_values("pair_id").groupby("source", sort=False).head(a.limit).reset_index(drop=True)
    if a.subset:
        plan = subset_plan(plan, a.subset)
    if a.limit or a.subset:
        need = set(plan["key_v"]) | set(plan["key_a"])
        prompts = [kt for kt in prompts if kt[0] in need]
    keys = [k for k, _ in prompts]
    kh = wp.keys_hash(keys)

    out = Path(a.out or f"{DRIVE_OUT}/{out_name(a.model, a.config, a.subset, a.limit, a.dtype)}")
    if (out / "meta.json").exists() and json.loads((out / "meta.json").read_text()).get("keys_hash") == kh:
        print(f"already harvested (same prompt list): {out}")
        return
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    kfile = out / "chunks" / "keys.json"
    if kfile.exists() and json.loads(kfile.read_text())["keys_hash"] != kh:
        sys.exit(f"STOP: {out}/chunks holds chunks from a different prompt list; use a new --out folder")

    tok, model = load_model(spec, a.config, a.dtype)
    n_layers, hidden = model.config.num_hidden_layers, model.config.hidden_size
    layers = sweep_layers(n_layers, a.sweep_every)
    est = size_estimate_gb(len(keys), hidden, len(layers))
    print(f"{a.model} {a.config} ({config_name(a.config, a.dtype)}) on {gpu} ({gib:.1f} GiB) | layers {n_layers} | hidden {hidden} | "
          f"sweep {layers}\npairs: {plan['source'].value_counts().to_dict()} | unique prompts: {len(keys)} | keys {kh} | "
          f"estimated size on Drive {est:.2f} GB -> {out}")
    if est > a.max_gb:
        sys.exit(f"STOP: {est:.2f} GB is above --max-gb {a.max_gb}: raise --sweep-every or --max-gb")
    kfile.write_text(json.dumps(dict(keys_hash=kh, n_prompts=len(keys), chunk=CHUNK)))
    ref_ids, comp_ids = wp_first_ids(tok)

    n_chunks = (len(keys) + CHUNK - 1) // CHUNK
    t0, done = time.time(), 0
    for c in range(n_chunks):
        f = out / "chunks" / f"chunk_{c:03d}.npz"
        if f.exists():
            print(f"chunk {c + 1}/{n_chunks}: already on Drive, skipped")
            continue
        texts = [t for _, t in prompts[c * CHUNK:(c + 1) * CHUNK]]
        tc = time.time()
        arr = run_chunk(tok, model, texts, layers, ref_ids, comp_ids)
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

    parts = [dict(np.load(out / "chunks" / f"chunk_{c:03d}.npz")) for c in range(n_chunks)]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert len(cat["prenorm"]) == len(keys)
    row = {k: i for i, k in enumerate(keys)}
    plan = plan[plan["key_v"].isin(row) & plan["key_a"].isin(row)].reset_index(drop=True)
    plan["row_v"], plan["row_a"] = plan["key_v"].map(row), plan["key_a"].map(row)
    prompts_df = pd.DataFrame(dict(prompt_key=keys, n_tokens=cat["n_tokens"], refusal_mass=cat["refusal_mass"],
                                   compliance_mass=cat["compliance_mass"], finite=cat["finite"]))
    meta = dict(model=spec["id"], model_tag=a.model, config=a.config, config_name=config_name(a.config, a.dtype),
                quantization="bitsandbytes nf4" if a.config == "c6" else "none", dtype=a.dtype, gpu=gpu,
                torch=torch.__version__, transformers=transformers.__version__,
                backend=f"HF transformers, attn {spec['attn']}, batch 1, no padding, one forward pass",
                reference="none (no vLLM R); X5 compares c6 with c1", position="prompt[-1]",
                last_instruction_offset=spec["last_instruction"], model_note=spec["note"],
                num_hidden_layers=n_layers, hidden_size=hidden, sweep_layers=layers,
                prenorm="input of the final norm (last decoder layer, = residual_stream)",
                postnorm="hidden_states[-1] (after the final norm)",
                template_sha256=hashlib.sha256((tok.chat_template or "").encode()).hexdigest()[:16],
                system_prompt="none added; the model's template defaults are kept", keys_hash=kh,
                n_prompts=len(keys), n_pairs=plan["source"].value_counts().to_dict(),
                n_nonfinite=int((~cat["finite"]).sum()), max_tokens=int(cat["n_tokens"].max()), truncation="none",
                refusal_words=list(wp.REFUSAL_WORDS), refusal_token_ids=ref_ids,
                compliance_words=list(wp.COMPLIANCE_WORDS), compliance_token_ids=comp_ids,
                pairs_table=a.pairs, pairs_table_sha256=hashlib.sha256(Path(a.pairs).read_bytes()).hexdigest()[:16],
                subset=a.subset or None, limit=a.limit or None, seed=0,
                created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
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
    shutil.rmtree(out / "chunks")   # the assembled arrays replace the chunks (halves the Drive footprint)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min: {len(keys)} prompts, {len(plan)} pairs, "
          f"{meta['n_nonfinite']} non-finite, max {meta['max_tokens']} tokens, nothing truncated, no text written -> {out}")


def wp_first_ids(tok):
    """First token ids of the refusal / compliance proxy words (as in ws3_pilot_extract.py)."""
    def first(words):
        ids = []
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if t and t[0] not in ids:
                ids.append(t[0])
        return ids
    return first(wp.REFUSAL_WORDS), first(wp.COMPLIANCE_WORDS)


if __name__ == "__main__":
    main()
