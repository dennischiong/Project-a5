"""WS3 reply generation with vLLM: one greedy reply per unique prompt, for the judges (harvest/ws3_judge.py). Needs a GPU.

Same prompts and same chat template as harvest/ws3_harvest.py (prompt keys match, so labels join to the activations by
prompt_key). Token ids are passed to vLLM directly, so BOS comes from the template only, as in the harvest.
Greedy, max 256 new tokens, no truncation of prompts: a prompt that does not fit the model's position limit is not
generated and is marked finish_reason = not_generated.

SAFETY: replies are raw text, so they are written ONLY under the git-ignored data/ folder (data/ws3_replies/<name>/).
Nothing here prints or logs a prompt or a reply: ids, counts and numbers only.

Output in --out: replies.parquet (prompt_key, reply, finish_reason, n_prompt_tokens, n_reply_tokens), pairs.parquet
(the plan with prompt keys, no text), meta.json. Resumable: every CHUNK prompts are saved to --out/chunks/.

  python harvest/ws3_generate.py --model mistral-7b --limit 10      # timing test
  python harvest/ws3_generate.py --model mistral-7b --subset 4000   # Track 3 subset, same prompts as the activations
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

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis import ws3_pilot as wp  # noqa: E402
from harvest import ws3_harvest as wh  # noqa: E402

CHUNK = 500
PAIRS_DEFAULT = REPO / "data" / "pairs_table_v1_final-2.csv"
REPLIES_ROOT = REPO / "data" / "ws3_replies"
# End-of-turn tokens that a model's generation_config may not list. Gemma-2-9B-it's lists only <eos>, so without
# <end_of_turn> every Gemma reply would run on to the token limit.
EXTRA_STOP_TOKENS = ("<end_of_turn>", "<|im_end|>")


# ------------------------------------------------------------------------------------------------ pure helpers
def select_prompts(pairs, limit=0, subset=0):
    """(prompts, plan) exactly as harvest/ws3_harvest.py selects them, so the prompt keys and their order match."""
    prompts, plan = wp.prompt_plan(pairs)
    if limit:
        plan = plan.sort_values("pair_id").groupby("source", sort=False).head(limit).reset_index(drop=True)
    if subset:
        plan = wh.subset_plan(plan, subset)
    if limit or subset:
        need = set(plan["key_v"]) | set(plan["key_a"])
        prompts = [kt for kt in prompts if kt[0] in need]
    return prompts, plan


def out_name(tag, subset=0, limit=0, dtype="bfloat16"):
    """Folder name. vLLM replies are a different serving config from the HF activations, and the name says so."""
    name = f"{tag}__wj-sorry__vllm-{'bf16' if dtype == 'bfloat16' else 'fp16'}"
    if subset:
        name += f"__sub{subset}"
    if limit:
        name += f"__smoke{limit}"
    return name


def stop_token_ids(tok, generation_eos=None):
    """Every token that ends a turn: the tokenizer's EOS, generation_config's EOS (int or list), and EXTRA_STOP_TOKENS
    when the tokenizer really has them (an unknown token maps to unk or None and is left out)."""
    ids = set()
    for value in (generation_eos, tok.eos_token_id):
        if value is None:
            continue
        ids.update(value if isinstance(value, (list, tuple)) else [value])
    for token in EXTRA_STOP_TOKENS:
        i = tok.convert_tokens_to_ids(token)
        if i is not None and i != tok.unk_token_id:
            ids.add(i)
    return sorted(ids)


def fits(n_prompt_tokens, max_new_tokens, limit):
    """Prompt plus a full-length reply fits the model's position limit."""
    return n_prompt_tokens + max_new_tokens <= limit


def max_model_len(token_counts, max_new_tokens, limit):
    """vLLM's memory limit: the longest prompt that fits plus a full reply, capped at the model's position limit."""
    longest = max((n for n in token_counts if fits(n, max_new_tokens, limit)), default=0)
    return min(limit, longest + max_new_tokens)


def n_chunks(n, chunk=CHUNK):
    return (n + chunk - 1) // chunk


def settings_hash(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def speed_line(elapsed, n_prompts, n_prompt_tokens, n_reply_tokens, target=4000):
    """Numbers for the timing test. Rough: a few prompts may be shorter or longer than the whole set."""
    if not n_prompts or elapsed <= 0:
        return "no prompts generated"
    return (f"{n_prompts} prompts in {elapsed:.0f} s | {n_prompt_tokens / elapsed:.0f} prompt tok/s, "
            f"{n_reply_tokens / elapsed:.0f} reply tok/s | at this speed {target} prompts take about "
            f"{elapsed / n_prompts * target / 3600:.2f} h")


def replace_file(local, target):
    """Copy, then rename: no half-written file on Drive if the machine stops mid-copy."""
    partial = Path(str(target) + ".partial")
    shutil.copy(local, partial)
    os.replace(partial, target)


def save_parquet(df, target):
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "part.parquet"
        df.to_parquet(local, index=False)
        replace_file(local, target)


def read_pairs(path):
    path = str(path)
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ vLLM
def load_llm(model_id, dtype, max_len, gpu_util):
    from vllm import LLM
    # generation_config="vllm": a model repo's own sampling defaults (e.g. a repetition penalty) must not change replies
    return LLM(model=model_id, dtype=dtype, seed=0, max_model_len=max_len, gpu_memory_utilization=gpu_util,
               generation_config="vllm")


def greedy_params(max_new_tokens, stop_ids):
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, stop_token_ids=stop_ids, seed=0)


def run_batch(llm, token_lists, params):
    """Generate for a list of token-id lists. Returns (text, finish_reason, n_reply_tokens), in input order."""
    if not token_lists:
        return []
    outs = llm.generate([{"prompt_token_ids": t} for t in token_lists], sampling_params=params, use_tqdm=False)
    return [(o.outputs[0].text, str(o.outputs[0].finish_reason), len(o.outputs[0].token_ids)) for o in outs]


def gpu_name():
    import torch
    if not torch.cuda.is_available():
        sys.exit("STOP: no GPU")
    return torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(wh.MODELS))
    ap.add_argument("--pairs", default=str(PAIRS_DEFAULT))
    ap.add_argument("--out", default="", help="default: data/ws3_replies/<name>; must stay under a git-ignored folder")
    ap.add_argument("--subset", type=int, default=0, help="Track 3 subset: N unique prompts (e.g. 4000)")
    ap.add_argument("--limit", type=int, default=0, help="timing test: only the first N pairs of each source")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    a = ap.parse_args()

    import torch
    import transformers
    import vllm
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    spec = wh.MODELS[a.model]
    if a.dtype == "float16" and not spec["fp16_ok"]:
        sys.exit(f"STOP: {a.model} in float16 risks overflow; use bfloat16")
    gpu, gib = gpu_name()
    if gib < spec["min_gib_bf16"]:
        sys.exit(f"STOP: {a.model} needs about {spec['min_gib_bf16']:.0f} GiB; {gpu} has {gib:.1f} GiB")

    prompts, plan = select_prompts(read_pairs(a.pairs), a.limit, a.subset)
    keys = [k for k, _ in prompts]
    kh = wp.keys_hash(keys)
    out = Path(a.out) if a.out else REPLIES_ROOT / out_name(a.model, a.subset, a.limit, a.dtype)
    settings = dict(model=spec["id"], dtype=a.dtype, max_new_tokens=a.max_new_tokens, keys_hash=kh, seed=0,
                    pairs_table_sha256=file_sha(a.pairs))
    if (out / "meta.json").exists() and json.loads((out / "meta.json").read_text()).get("settings_hash") == settings_hash(settings):
        print(f"already generated (same settings): {out}")
        return
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    sfile = out / "chunks" / "settings.json"
    if sfile.exists() and json.loads(sfile.read_text()) != settings:
        sys.exit(f"STOP: {out}/chunks holds chunks from different settings; use a new --out folder")
    sfile.write_text(json.dumps(settings))

    tok = AutoTokenizer.from_pretrained(spec["id"])
    rendered = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
                for _, t in prompts]
    token_lists = [tok(r, add_special_tokens=False).input_ids for r in rendered]
    del rendered
    counts = [len(t) for t in token_lists]
    limit = int(getattr(AutoConfig.from_pretrained(spec["id"]), "max_position_embeddings"))
    max_len = max_model_len(counts, a.max_new_tokens, limit)
    try:
        generation_eos = GenerationConfig.from_pretrained(spec["id"]).eos_token_id
    except OSError:
        generation_eos = None
    stops = stop_token_ids(tok, generation_eos)
    print(f"{a.model} vLLM {vllm.__version__} {a.dtype} on {gpu} ({gib:.1f} GiB) | prompts {len(keys)} | keys {kh} | "
          f"longest prompt {max(counts)} tok | position limit {limit} | max_model_len {max_len} | stop ids {stops} -> {out}")

    t_load = time.time()
    llm = load_llm(spec["id"], a.dtype, max_len, a.gpu_memory_utilization)
    params = greedy_params(a.max_new_tokens, stops)
    print(f"model loaded in {time.time() - t_load:.0f} s")

    total = n_chunks(len(keys))
    elapsed = done = done_prompt_tok = done_reply_tok = 0
    for c in range(total):
        f = out / "chunks" / f"chunk_{c:03d}.parquet"
        if f.exists():
            print(f"chunk {c + 1}/{total}: already saved, skipped")
            continue
        idx = range(c * CHUNK, min((c + 1) * CHUNK, len(keys)))
        go = [i for i in idx if fits(counts[i], a.max_new_tokens, limit)]
        tc = time.time()
        results = dict(zip(go, run_batch(llm, [token_lists[i] for i in go], params)))
        took = time.time() - tc
        rows = []
        for i in idx:
            text, reason, n_reply = results.get(i, ("", "not_generated", 0))
            rows.append(dict(prompt_key=keys[i], reply=text, finish_reason=reason, n_prompt_tokens=counts[i],
                             n_reply_tokens=n_reply))
        df = pd.DataFrame(rows)
        save_parquet(df, f)
        elapsed += took
        done += len(go)
        done_prompt_tok += sum(counts[i] for i in go)
        done_reply_tok += int(df["n_reply_tokens"].sum())
        left = sum(min(CHUNK, len(keys) - k * CHUNK) for k in range(c + 1, total)
                   if not (out / "chunks" / f"chunk_{k:03d}.parquet").exists())
        print(f"chunk {c + 1}/{total}: {took:.0f} s | {df['finish_reason'].value_counts().to_dict()} | "
              f"ETA {left * elapsed / max(done, 1) / 60:.1f} min")
    print("speed: " + speed_line(elapsed, done, done_prompt_tok, done_reply_tok))

    cat = pd.concat([pd.read_parquet(out / "chunks" / f"chunk_{c:03d}.parquet") for c in range(total)], ignore_index=True)
    assert cat["prompt_key"].tolist() == keys, "saved chunks do not match the prompt list"
    reasons = cat["finish_reason"].value_counts().to_dict()
    n_empty = int(((cat["reply"].str.strip() == "") & (cat["finish_reason"] != "not_generated")).sum())
    meta = dict(model=spec["id"], model_tag=a.model, backend=f"vLLM {vllm.__version__}, token ids, greedy", dtype=a.dtype,
                quantization="none", gpu=gpu, torch=torch.__version__, transformers=transformers.__version__,
                max_new_tokens=a.max_new_tokens, temperature=0.0, stop_token_ids=stops, position_limit=limit,
                max_model_len=max_len, keys_hash=kh, settings_hash=settings_hash(settings), n_prompts=len(keys),
                finish_reasons=reasons, n_empty=n_empty, truncation="none",
                template_sha256=hashlib.sha256((tok.chat_template or "").encode()).hexdigest()[:16],
                system_prompt="none added; the model's template defaults are kept",
                pairs_table_sha256=settings["pairs_table_sha256"], subset=a.subset or None, limit=a.limit or None, seed=0,
                activations_note="replies are vLLM bf16; the WS3 activations are HF batch 1, so state both next to any H2 or A8 result",
                created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save_parquet(cat, out / "replies.parquet")
    save_parquet(plan, out / "pairs.parquet")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(out / "chunks")
    print(f"DONE: {len(keys)} prompts | {reasons} | {n_empty} empty | no text printed -> {out}")


if __name__ == "__main__":
    main()
