"""SYNTHETIC Undercurrent-style harvest shards for tests and chart previews: random numbers, never text.

Follows the format assumed in lib.EXPECTED_SCHEMA (work plan s.6.1 / s.6.6): shard_00000.safetensors with one
(prompts, hidden) tensor per <point>/layer_<n>, plus shard_00000.parquet with the index columns. bf16 when torch
is installed (as in Colab), float32 otherwise.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Llama-3.1-8B layout scaled down: all layers at last_prefill, every 2nd at last_instruction, 4 at decode positions
def point_layers(n_layers):
    q = [max(0, round(n_layers * f) - 1) for f in (0.25, 0.5, 0.75, 1.0)]
    return {"last_prefill": list(range(n_layers)), "last_instruction": list(range(1, n_layers, 2)),
            "gen_4": q, "gen_9": q, "gen_19": q}


def make_fake_harvest(folder, n_shards=2, prompts_per_shard=6, n_layers=8, hidden=16, seed=0, pair_ids=None,
                      nan_at=None, truncate_rows=(), dtype="auto"):
    """Write a synthetic harvest. pair_ids: list of pair ids; prompt ids are <pair_id>|<role> (vanilla, adversarial
    alternating). nan_at: (point, layer) to plant one NaN in shard 0. Returns the list of prompt ids in order."""
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n = n_shards * prompts_per_shard
    if pair_ids is None:
        pair_ids = [f"syn{i:04d}" for i in range(n // 2)]
    ids = [f"{p}|{r}" for p in pair_ids for r in ("vanilla", "adversarial")][:n]
    try:
        import torch
        from safetensors.torch import save_file
        use_torch = dtype in ("auto", "bf16")
    except ImportError:
        use_torch = False
    if not use_torch:
        from safetensors.numpy import save_file
    layers = point_layers(n_layers)
    for s in range(n_shards):
        rows = ids[s * prompts_per_shard:(s + 1) * prompts_per_shard]
        tensors = {}
        for point, ls in layers.items():
            for L in ls:
                x = rng.normal(size=(len(rows), hidden)).astype(np.float32) * (1 + L / n_layers)
                if nan_at == (point, L) and s == 0:
                    x[0, 0] = np.nan
                key = f"{point}/layer_{L}"
                tensors[key] = torch.from_numpy(x).to(torch.bfloat16) if use_torch else x
        save_file(tensors, str(folder / f"shard_{s:05d}.safetensors"))
        idx = pd.DataFrame({
            "prompt_id": rows, "token_count": rng.integers(20, 600, len(rows)),
            "truncated": [(s * prompts_per_shard + k) in truncate_rows for k in range(len(rows))],
            "backend": "vllm", "concurrency": 64, "dtype": "bfloat16" if use_torch else "float32",
            "quantization": "none", "undercurrent_version": "0.1.0-SYNTHETIC", "template_hash": "synthetic",
        })
        idx.to_parquet(folder / f"shard_{s:05d}.parquet")
    return ids
