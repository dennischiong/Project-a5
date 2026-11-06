"""Analysis library for the A5 adversarial-internals project.

Conventions
-----------
* A *pairs table* has one row per (pair_id, role). role is vanilla or adversarial.
  Columns: pair_id, source, role, intent (harmful/benign), tactic, token_count, split,
  source_row_id, and (only in git-ignored data/) text.
* Activations are stored as safetensors, one tensor per key <pair_id>|<role>, shape
  (n_layers + 1, hidden): the prompt[-1] vector for every entry of ``hidden_states``.
* Nothing here prints or returns raw prompt text.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

KEY_SEP = "|"
SPLITS = ("train", "validation", "test")


# --------------------------------------------------------------------------- loading
def load_pairs(path: str) -> pd.DataFrame:
    """Read a pairs table from parquet and check its basic structure.

    Asserts the required columns exist and that role/intent/split values are legal.
    Does not look at text content (see ``validate_pairs`` for that).
    """
    df = pd.read_parquet(path)
    required = {"pair_id", "role", "intent", "tactic", "token_count", "split"}
    missing = required - set(df.columns)
    assert not missing, f"pairs table missing columns: {sorted(missing)}"
    assert set(df["role"].unique()) <= {"vanilla", "adversarial"}, "unexpected role value"
    assert set(df["intent"].unique()) <= {"harmful", "benign"}, "unexpected intent value"
    assert set(df["split"].unique()) <= set(SPLITS), "unexpected split value"
    return df


def make_key(pair_id: str, role: str) -> str:
    return f"{pair_id}{KEY_SEP}{role}"


def load_activations(
    path: str,
    n_layers_plus_1: Optional[int] = None,
    hidden: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load a safetensors file into {key: float32 array (layers+1, hidden)}.

    Asserts every tensor is 2-D, all tensors share one shape (and match the expected shape if
    given), and no value is NaN or inf. bf16 is upcast to float32 (numpy has no bf16).
    """
    from safetensors import safe_open

    out: Dict[str, np.ndarray] = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).float().numpy()
    assert out, "activation file is empty"
    shapes = {v.shape for v in out.values()}
    assert len(shapes) == 1, f"inconsistent tensor shapes: {shapes}"
    shape = next(iter(shapes))
    assert len(shape) == 2, f"expected (layers+1, hidden), got {shape}"
    if n_layers_plus_1 is not None:
        assert shape[0] == n_layers_plus_1, f"layers+1 {shape[0]} != {n_layers_plus_1}"
    if hidden is not None:
        assert shape[1] == hidden, f"hidden {shape[1]} != {hidden}"
    for k, v in out.items():
        assert np.isfinite(v).all(), f"NaN/inf in activations for key {k}"
    return out


# --------------------------------------------------------------------------- splits
def assign_split(pair_id: str, fractions=(0.70, 0.15, 0.15)) -> str:
    """Stable split from a sha256 hash of pair_id (never Python's hash()).

    Both rows of a pair share pair_id and therefore a split. Default 70/15/15.
    """
    assert abs(sum(fractions) - 1.0) < 1e-9
    digest = hashlib.sha256(str(pair_id).encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64  # uniform in [0, 1)
    if u < fractions[0]:
        return "train"
    if u < fractions[0] + fractions[1]:
        return "validation"
    return "test"


# --------------------------------------------------------------------------- direction
def fit_direction(h_harm: np.ndarray, h_benign: np.ndarray) -> np.ndarray:
    """r_hat = (mean(h_harm) - mean(h_benign)) / norm. Inputs shape (n, d). Returns unit (d,)."""
    h_harm, h_benign = np.asarray(h_harm, float), np.asarray(h_benign, float)
    assert h_harm.ndim == 2 and h_benign.ndim == 2 and h_harm.shape[1] == h_benign.shape[1]
    assert len(h_harm) > 0 and len(h_benign) > 0, "need rows in both classes"
    diff = h_harm.mean(0) - h_benign.mean(0)
    norm = np.linalg.norm(diff)
    assert norm > 0, "class means are identical; direction undefined"
    return diff / norm


def require_train_only(df: pd.DataFrame) -> None:
    """Raise ValueError if any row's split is not 'train' (guards against fitting on eval data)."""
    bad = df["split"] != "train"
    if bad.any():
        raise ValueError(f"refusing to fit: {int(bad.sum())} row(s) have split != 'train'")


def fit_direction_train(h: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Fit r_hat from rows ``h`` (n, d) described by ``meta`` (same order, columns intent/split).

    ``meta`` must contain only vanilla-train rows; any non-train row raises ValueError.
    """
    assert len(h) == len(meta)
    require_train_only(meta)
    if "role" in meta.columns and (meta["role"] != "vanilla").any():
        raise ValueError("refusing to fit: r_hat is fitted on vanilla prompts only")
    harm = (meta["intent"] == "harmful").to_numpy()
    return fit_direction(h[harm], h[~harm])


def project(delta: np.ndarray, r_hat: np.ndarray) -> np.ndarray:
    """dp = delta . r_hat. delta (..., d) -> (...). Negative = adversarial looks less harmful."""
    return np.asarray(delta, float) @ np.asarray(r_hat, float)


# --------------------------------------------------------------------------- distances
def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - cos(a, b) row-wise. a, b shape (n, d) or (d,)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return 1.0 - num / den


def relative_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """||a - b|| / ||a|| row-wise. a is the reference (vanilla)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(a, axis=-1)


# --------------------------------------------------------------------------- statistics
def bootstrap_ci(
    values: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n: int = 2000,
    seed: int = 0,
    level: float = 0.95,
):
    """Percentile bootstrap CI. ``values`` has ONE entry per pair (resampling is over pairs).

    Returns (point_estimate, lo, hi).
    """
    values = np.asarray(values)
    m = len(values)
    assert m > 0, "no values"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    boots = np.array([stat(values[i]) for i in idx])
    alpha = (1 - level) / 2
    return float(stat(values)), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


def cohens_d(a: np.ndarray, b: Optional[np.ndarray] = None) -> float:
    """Cohen's d. Two samples: (mean a - mean b) / pooled SD. One sample (b=None): mean / SD
    (the paired form when ``a`` holds per-pair differences, e.g. dp)."""
    a = np.asarray(a, float)
    if b is None:
        sd = a.std(ddof=1)
        return float(a.mean() / sd) if sd > 0 else float("nan")
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def auroc_per_group(y, score, groups, min_per_class: int = 3) -> Dict[str, float]:
    """AUROC within each group; NaN (n/a) unless both classes have >= min_per_class members."""
    from sklearn.metrics import roc_auc_score

    y, score, groups = np.asarray(y), np.asarray(score), np.asarray(groups)
    out: Dict[str, float] = {}
    for g in pd.unique(groups):
        m = groups == g
        pos, neg = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        out[g] = float(roc_auc_score(y[m], score[m])) if min(pos, neg) >= min_per_class else float("nan")
    return out


# --------------------------------------------------------------------------- validation
def validate_pairs(df: pd.DataFrame, check_text: bool = True, allow_unknown_tactics: bool = False) -> None:
    """Raise ValueError describing every problem found. Never echoes text content.

    Fails if: an adversarial row has empty/NaN/None text; an adversarial token_count <= 1;
    a pair_id lacks exactly one vanilla and one adversarial row; duplicate (pair_id, role);
    or ALL harmful adversarial rows have tactic unknown (skipped only when allow_unknown_tactics=True, which
    the pilot sets under its --allow-missing-tactics override).
    """
    problems = []
    adv = df[df["role"] == "adversarial"]

    if check_text:
        if "text" not in df.columns:
            problems.append("no 'text' column to check")
        else:
            t = adv["text"]
            bad = t.isna() | t.astype(str).str.strip().isin(["", "None", "nan", "NaN"])
            if bad.any():
                problems.append(f"{int(bad.sum())} adversarial row(s) with empty/NaN/'None' text")

    if (adv["token_count"].isna() | (adv["token_count"] <= 1)).any():
        problems.append("adversarial row(s) with token_count <= 1")

    dup = df.duplicated(["pair_id", "role"], keep=False)
    if dup.any():
        problems.append(f"{int(dup.sum())} row(s) share a duplicate (pair_id, role)")

    counts = df.groupby(["pair_id", "role"]).size().unstack(fill_value=0)
    for role in ("vanilla", "adversarial"):
        if role not in counts.columns:
            counts[role] = 0
    broken = counts[(counts["vanilla"] != 1) | (counts["adversarial"] != 1)]
    if len(broken):
        problems.append(f"{len(broken)} pair_id(s) without exactly one vanilla and one adversarial row")

    harm_adv = adv[adv["intent"] == "harmful"]
    if not allow_unknown_tactics and len(harm_adv) and (harm_adv["tactic"].astype(str) == "unknown").all():
        problems.append("all harmful adversarial rows have tactic 'unknown'")

    if problems:
        raise ValueError("validate_pairs failed: " + "; ".join(problems))


# --------------------------------------------------------------------------- Undercurrent harvest shards (sample shard from Kayson)
# ASSUMED format of the team harvest (work plan s.6.1 and s.6.6): one safetensors file per shard of up to 500
# prompts in native bf16, plus a parquet index per shard. The exact tensor key names are UNKNOWN until a real sample
# arrives. >>> EDIT THIS ONCE KAYSON'S SAMPLE ARRIVES <<<  Everything below reads the format from this dict only.
EXPECTED_SCHEMA = {
    "shard_glob": "*.safetensors",            # shard files (searched in sub-folders too)
    "index_suffix": ".parquet",               # the index of shard_00000.safetensors is shard_00000.parquet
    "tensor_key_regex": r"^(?P<point>[A-Za-z0-9_]+)/layer_(?P<layer>\d+)$",  # assumed key <point>/layer_<n>
    "tensor_key_example": "last_prefill/layer_31",  # each tensor: (prompts in the shard, hidden), rows = index rows
    "dtype": "bfloat16",
    "prompts_per_shard": 500,
    # extraction points (work plan s.6.1): all layers at last_prefill, every 2nd at last_instruction, 4 layers at decode
    "points": ["last_prefill", "last_instruction", "gen_4", "gen_9", "gen_19"],
    "index_columns": ["prompt_id", "token_count", "truncated", "backend", "concurrency", "dtype", "quantization",
                      "undercurrent_version", "template_hash"],
    "id_column": "prompt_id",
    # join to the pairs table: harvest prompt_id == pairs prompt_id; if the pairs table has no prompt_id column it is
    # built as <pair_id>|<role> (the pilot's key). Change pairs_key to e.g. source_row_id if that is the link.
    "join": {"index_key": "prompt_id", "pairs_key": "prompt_id"},
}


def _tensor_reader():
    """safetensors framework: 'pt' (handles bf16; Colab has torch) or 'numpy' (float16/32 only)."""
    try:
        import torch  # noqa: F401
        return "pt"
    except ImportError:
        return "numpy"


def _parse_key(key, schema=None):
    import re

    m = re.match((schema or EXPECTED_SCHEMA)["tensor_key_regex"], key)
    return (m.group("point"), int(m.group("layer"))) if m else None


def harvest_shards(shard_dir, schema=None):
    """Sorted shard files under shard_dir."""
    from pathlib import Path

    return sorted(Path(shard_dir).rglob((schema or EXPECTED_SCHEMA)["shard_glob"]))


def _index_path(shard, schema=None):
    return shard.with_suffix((schema or EXPECTED_SCHEMA)["index_suffix"])


def harvest_layers(shard_dir, point="last_prefill", schema=None):
    """Layer numbers stored for one extraction point, from the first shard's key names (no tensor is read)."""
    from safetensors import safe_open

    shards = harvest_shards(shard_dir, schema) if str(shard_dir).strip() else []
    if not shards:
        raise FileNotFoundError(f"no shard files under {shard_dir!r}")
    with safe_open(str(shards[0]), framework=_tensor_reader()) as f:
        parsed = [_parse_key(k, schema) for k in f.keys()]
    return sorted(pk[1] for pk in parsed if pk and pk[0] == point)


def load_harvest(shard_dir, point="last_prefill", layers=None, schema=None):
    """Read every shard under ``shard_dir`` for one extraction point.

    Returns (activations, index): activations is float32 (prompts, layers, hidden) and row i belongs to the prompt
    in index row i (index[id_column] is unique). index.attrs[layers] lists the layer numbers of axis 1 and
    index.attrs[point] the point. ``layers`` (a list) keeps only those layers; a missing one raises KeyError.
    Raises FileNotFoundError / ValueError with a plain message if the format does not match EXPECTED_SCHEMA.
    """
    from safetensors import safe_open

    schema = schema or EXPECTED_SCHEMA
    shards = harvest_shards(shard_dir, schema) if str(shard_dir).strip() else []
    if not shards:
        raise FileNotFoundError(f"no {schema['shard_glob']} files under {shard_dir!r}")
    fw = _tensor_reader()
    acts, idxs, layer_ref = [], [], None
    for sh in shards:
        ip = _index_path(sh, schema)
        if not ip.exists():
            raise FileNotFoundError(f"{sh.name}: no index file {ip.name}")
        idx = pd.read_parquet(ip)
        with safe_open(str(sh), framework=fw) as f:
            found = {}
            for k in f.keys():
                pk = _parse_key(k, schema)
                if pk and pk[0] == point:
                    found[pk[1]] = k
            if not found:
                raise ValueError(f"{sh.name}: no tensors for point '{point}' (check EXPECTED_SCHEMA['tensor_key_regex'])")
            sel = sorted(found) if layers is None else list(layers)
            missing = [L for L in sel if L not in found]
            if missing:
                raise KeyError(f"{sh.name}: point '{point}' has no layer(s) {missing}")
            if layer_ref is None:
                layer_ref = sel
            elif sel != layer_ref:
                raise ValueError(f"{sh.name}: layers {sel} differ from the first shard's {layer_ref}")
            mats = []
            for L in sel:
                t = f.get_tensor(found[L])
                t = t.float().numpy() if fw == "pt" else np.asarray(t, dtype=np.float32)
                if t.ndim != 2 or t.shape[0] != len(idx):
                    raise ValueError(f"{sh.name}: tensor {found[L]} has shape {tuple(t.shape)}, expected ({len(idx)}, hidden)")
                mats.append(t)
        acts.append(np.stack(mats, axis=1))
        idxs.append(idx.assign(shard=sh.name, row_in_shard=np.arange(len(idx))))
    index = pd.concat(idxs, ignore_index=True)
    idc = schema["id_column"]
    if idc not in index.columns:
        raise ValueError(f"index has no '{idc}' column (check EXPECTED_SCHEMA['id_column'])")
    if index[idc].duplicated().any():
        raise ValueError(f"{int(index[idc].duplicated().sum())} duplicated {idc} value(s) across shards")
    index.attrs["layers"], index.attrs["point"] = layer_ref, point
    return np.concatenate(acts, axis=0), index


def check_harvest(shard_dir, schema=None, verbose=True):
    """Plain-English report on a harvest folder. Never raises: every problem is reported instead.

    Returns a dict used by the Task 8 charts: shards, prompts_per_shard, points {point: layers}, dtypes, hidden,
    nan_inf {(point, layer): count}, mean_norm {(point, layer): mean L2 norm over prompts}, token_count list,
    truncated list (bools), schema [(field, ok, hint)], problems [str].
    """
    from pathlib import Path

    schema = schema or EXPECTED_SCHEMA
    rep = dict(shard_dir=str(shard_dir), shards=[], prompts_per_shard={}, points={}, dtypes=set(), hidden=set(),
               nan_inf={}, mean_norm={}, token_count=[], truncated=[], schema=[], problems=[], unparsed_keys=[],
               index_columns=set(), n_prompts=0, ids_unique=None, rows_match=True)
    try:
        if not str(shard_dir).strip() or not Path(shard_dir).is_dir():
            rep["problems"].append(f"folder does not exist: {shard_dir!r}")
            shards = []
        else:
            shards = harvest_shards(shard_dir, schema)
        fw = _tensor_reader()
        ids = []
        norm_acc = {}  # (point, layer) -> (sum of norms, number of prompts)
        for sh in shards:
            rep["shards"].append(sh.name)
            try:
                ip = _index_path(sh, schema)
                idx = pd.read_parquet(ip) if ip.exists() else None
                if idx is None:
                    rep["problems"].append(f"{sh.name}: index file {ip.name} missing")
                else:
                    rep["index_columns"] |= set(idx.columns)
                    rep["prompts_per_shard"][sh.name] = len(idx)
                    if schema["id_column"] in idx.columns:
                        ids += idx[schema["id_column"]].astype(str).tolist()
                    if "token_count" in idx.columns:
                        rep["token_count"] += [int(x) for x in idx["token_count"]]
                        tr = idx["truncated"].astype(bool).tolist() if "truncated" in idx.columns else [False] * len(idx)
                        rep["truncated"] += tr
                from safetensors import safe_open

                with safe_open(str(sh), framework=fw) as f:
                    for k in f.keys():
                        pk = _parse_key(k, schema)
                        try:
                            rep["dtypes"].add(str(f.get_slice(k).get_dtype()))
                        except Exception:  # noqa: BLE001
                            pass
                        if pk is None:
                            rep["unparsed_keys"].append(k)
                            continue
                        try:
                            t = f.get_tensor(k)
                            t = t.float().numpy() if fw == "pt" else np.asarray(t, dtype=np.float32)
                        except Exception as e:  # noqa: BLE001
                            rep["problems"].append(f"{sh.name}: cannot read {k} ({type(e).__name__}; bf16 needs torch)")
                            continue
                        rep["points"].setdefault(pk[0], set()).add(pk[1])
                        if t.ndim == 2:
                            rep["hidden"].add(int(t.shape[1]))
                            if idx is not None and t.shape[0] != len(idx):
                                rep["rows_match"] = False
                                rep["problems"].append(f"{sh.name}: {k} has {t.shape[0]} rows, index has {len(idx)}")
                            bad = int((~np.isfinite(t)).sum())
                            rep["nan_inf"][pk] = rep["nan_inf"].get(pk, 0) + bad
                            norms = np.linalg.norm(np.where(np.isfinite(t), t, 0), axis=1)
                            s, n = norm_acc.get(pk, (0.0, 0))
                            norm_acc[pk] = (s + float(norms.sum()), n + len(norms))
                        else:
                            rep["problems"].append(f"{sh.name}: {k} is {t.ndim}-D, expected (prompts, hidden)")
            except Exception as e:  # noqa: BLE001
                rep["problems"].append(f"{sh.name}: could not read ({type(e).__name__})")
        rep["mean_norm"] = {pk: s / n for pk, (s, n) in norm_acc.items() if n}
        rep["n_prompts"] = sum(rep["prompts_per_shard"].values())
        rep["ids_unique"] = (len(ids) == len(set(ids))) if ids else None
    except Exception as e:  # noqa: BLE001
        rep["problems"].append(f"unexpected error ({type(e).__name__}): {e}")

    # ---- the schema check, one row per expected field
    S = rep["schema"]
    ex = ", ".join(rep["unparsed_keys"][:3])
    S.append(("shard files " + schema["shard_glob"], bool(rep["shards"]),
              "no shard files: check the folder, or EXPECTED_SCHEMA['shard_glob']"))
    S.append(("one parquet index per shard", bool(rep["shards"]) and len(rep["prompts_per_shard"]) == len(rep["shards"]),
              "edit EXPECTED_SCHEMA['index_suffix'] to match the index file names"))
    for c in schema["index_columns"]:
        S.append((f"index column '{c}'", c in rep["index_columns"], f"edit EXPECTED_SCHEMA['index_columns'] (found: "
                  f"{sorted(rep['index_columns'])[:12]})"))
    S.append(("tensor keys like " + schema["tensor_key_example"], bool(rep["points"]) and not rep["unparsed_keys"],
              f"edit EXPECTED_SCHEMA['tensor_key_regex'] to match keys such as: {ex or 'none found'}"))
    for p in schema["points"]:
        S.append((f"extraction point '{p}'", p in rep["points"], f"point missing; found {sorted(rep['points'])}"))
    want = {"bfloat16": {"BF16", "torch.bfloat16", "bfloat16"}}.get(schema["dtype"], {schema["dtype"]})
    S.append((f"dtype {schema['dtype']}", bool(rep["dtypes"]) and rep["dtypes"] <= want,
              f"found dtypes {sorted(rep['dtypes'])}; edit EXPECTED_SCHEMA['dtype'] if this is intended"))
    S.append(("tensor rows == index rows", rep["rows_match"] and bool(rep["prompts_per_shard"]), "see problems below"))
    S.append((f"unique {schema['id_column']}", bool(rep["ids_unique"]), "duplicated or missing prompt ids"))
    S.append(("no NaN / inf values", bool(rep["nan_inf"]) and sum(rep["nan_inf"].values()) == 0,
              "NaN/inf found: fp16 overflow? (work plan s.6.6: keep bf16)"))
    S.append(("no truncated prompts", bool(rep["truncated"]) and not any(rep["truncated"]),
              f"{sum(rep['truncated'])} truncated prompt(s), or no 'truncated' column"))
    if verbose:
        for line in harvest_report_lines(rep):
            print(line)
    return rep


def harvest_report_lines(rep):
    lines = [f"harvest folder: {rep['shard_dir']}",
             f"shards found: {len(rep['shards'])} | prompts: {rep['n_prompts']} | hidden size: {sorted(rep['hidden'])} | "
             f"dtypes: {sorted(rep['dtypes'])}"]
    for p, ls in sorted(rep["points"].items()):
        ls = sorted(ls)
        lines.append(f"  point {p}: {len(ls)} layer(s) {ls[:6]}{' ...' if len(ls) > 6 else ''}")
    lines.append(f"NaN/inf values: {sum(rep['nan_inf'].values())} | truncated prompts: {sum(rep['truncated'])}")
    bad = [(f, h) for f, ok, h in rep["schema"] if not ok]
    lines.append(f"schema check: {len(rep['schema']) - len(bad)}/{len(rep['schema'])} fields match EXPECTED_SCHEMA")
    for f, h in bad:
        lines.append(f"  MISMATCH {f}: {h}")
    for p in rep["problems"]:
        lines.append(f"  PROBLEM {p}")
    return lines


def join_pairs(index, pairs, schema=None):
    """Attach pair_id / role / intent / tactic / split from the pairs table to every harvest row (same order as
    ``index``). Rows without a match get NaN; prints matched counts only."""
    j = (schema or EXPECTED_SCHEMA)["join"]
    ik, pk = j["index_key"], j["pairs_key"]
    p = pairs.copy()
    if pk not in p.columns:
        if {"pair_id", "role"} <= set(p.columns):
            p[pk] = [make_key(a, b) for a, b in zip(p["pair_id"], p["role"])]
        else:
            raise KeyError(f"pairs table has no '{pk}' column and no pair_id/role to build it from")
    keep = [c for c in ("pair_id", "role", "intent", "tactic", "split", "source", "source_row_id") if c in p.columns and c != pk]
    p = p[[pk] + keep].rename(columns={pk: ik})
    p[ik] = p[ik].astype(str)
    left = pd.DataFrame({ik: index[ik].astype(str).to_numpy()})
    out = left.merge(p, on=ik, how="left", validate="many_to_one")
    print(f"join: {int(out['pair_id'].notna().sum()) if 'pair_id' in out else 0} of {len(out)} harvest rows matched a pairs-table row")
    return out


def build_2x2(acts, joined):
    """Pairs with exactly one vanilla and one adversarial row in the harvest.

    Returns (Hv, Ha, meta): Hv/Ha shape (pairs, layers, hidden), meta one row per pair (pair_id, intent, tactic, split).
    """
    j = joined.reset_index(drop=True)  # row k of j is row k of acts
    assert len(j) == len(acts), "joined table and activations must have the same rows"
    ok = j["pair_id"].notna() & j["role"].isin(["vanilla", "adversarial"])
    j = j[ok]
    counts = j.groupby(["pair_id", "role"]).size().unstack(fill_value=0)
    for r in ("vanilla", "adversarial"):
        if r not in counts.columns:
            counts[r] = 0
    good = counts[(counts["vanilla"] == 1) & (counts["adversarial"] == 1)].index
    j = j[j["pair_id"].isin(good)]
    van = j[j.role == "vanilla"].set_index("pair_id")
    ids = sorted(good)
    rows_v = j.index[j.role == "vanilla"]
    rows_a = j.index[j.role == "adversarial"]
    pos_v = dict(zip(j.loc[rows_v, "pair_id"], rows_v))
    pos_a = dict(zip(j.loc[rows_a, "pair_id"], rows_a))
    Hv = acts[[pos_v[i] for i in ids]]
    Ha = acts[[pos_a[i] for i in ids]]
    cols = [c for c in ("intent", "tactic", "split") if c in van.columns]
    meta = van.loc[ids, cols].reset_index()
    return Hv, Ha, meta
