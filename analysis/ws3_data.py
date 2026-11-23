"""WS3 data plumbing: restored harvest shards + the pairs table -> (Hv, Ha, meta) at one layer and position, and a
numbers-only JSON writer. Used by WS3_analyses.ipynb; the Drive restore itself stays in the notebook (it needs Colab).
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import lib


def pairs_from_harvest(shard_dir, pairs, point="last_prefill", layer=None):
    """Read one extraction point (and one layer; None = the last stored layer) from the shards under shard_dir,
    join to the pairs table, keep complete vanilla/adversarial pairs.

    Returns (Hv, Ha, meta, info): Hv/Ha (pairs, hidden) float32, meta one row per pair (pair_id, intent, tactic,
    split), info = dict(layer, point, n_rows, n_pairs) for titles and the results file."""
    layers = lib.harvest_layers(shard_dir, point)  # key names only: no tensor is read
    if not layers:
        raise ValueError(f"no tensors for point '{point}' under {shard_dir}")
    use = layers[-1] if layer is None else int(layer)
    if use not in layers:
        raise KeyError(f"layer {use} is not stored at point '{point}' (stored: {layers})")
    acts, index = lib.load_harvest(shard_dir, point=point, layers=[use])
    joined = lib.join_pairs(index, pairs)
    Hv, Ha, meta = lib.build_2x2(acts, joined)
    if "split" not in meta.columns:
        meta["split"] = meta["pair_id"].map(lib.assign_split)
    if "tactic" not in meta.columns:
        meta["tactic"] = "unknown"
    return Hv[:, 0], Ha[:, 0], meta, dict(layer=use, point=point, n_rows=len(index), n_pairs=len(meta))


def jsonable(o):
    """numpy / tuples / NaN -> plain JSON (NaN and inf become null). Never includes text from the data."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict())
    return o


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, allow_nan=False))
    return path
