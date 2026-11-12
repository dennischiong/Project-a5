"""WS3 pilot on REAL activations with tactic labels (Qwen2.5-1.5B-Instruct, free-tier T4), before the full harvests.

Data: harvest/ws3_pilot_extract.py runs the model once per unique prompt of the pairs table v1.1 (WildJailbreak and
SORRY-Bench) and writes numbers only. SORRY-Bench gives 19 labelled tactics (436 base instructions x 19 mutations) but
no benign twins, so the harmfulness direction r_hat is fitted on WildJailbreak vanilla TRAIN pairs (harmful vs benign)
and SORRY-Bench displacements are projected onto it.

What run_all() produces (every figure regenerates from this one function; labelled PILOT for this pilot and RESULT
for a harvest/ws3_harvest.py folder, see run_labels; A5/A8 always EXPLORATORY):
  A2   WildJailbreak F2 histograms; per-tactic masking on SORRY-Bench: mean dp and the share of adversarial prompts
       below the WildJailbreak midpoint, with 95% bootstrap CIs over held-out pairs
  A7   style vs content on WildJailbreak held-out pairs (H3 part a): style = benign pairs' mean dp, content = harmful
       minus benign, pooled over tactics (WildJailbreak has no tactic labels yet; with labels, rows per tactic appear)
  A5   cosine between the 19 tactics' mean displacements, PCA, principal angles between tactic subspaces (k = 3),
       same-tactic share of k-NN neighbours vs chance, and how much of each tactic's move lies along r_hat
  A8   singular-value spectrum of the displacements (all and per tactic), and the angle between r_hat and a refusal
       direction from a NEXT-TOKEN PROXY (refusal-word mass at prompt[-1]), not judge labels. When WildGuard labels
       exist (analysis/judge_labels.py), the angle from those labels is reported next to the proxy's
Numbers only: no prompt text or reply is ever read here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import geometry as g
from analysis import lib

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_TAG = "qwen25-1p5b"
SOURCES = ("WildJailbreak", "SORRY-Bench")
# SORRY-Bench mutation families (SORRY-Bench paper); the work plan puts encoded mutations in a may not be understood
# bucket, and a 1.5B model may not understand low-resource translations either: both are flagged in every output.
FAMILIES = {
    "encoded (may not be understood)": ("ascii", "atbash", "caesar", "morse"),
    "translation": ("translate-fr", "translate-ml", "translate-mr", "translate-ta", "translate-zh-cn"),
    "writing style": ("misspellings", "question", "role_play", "slang", "technical_terms", "uncommon_dialects"),
    "persuasion": ("authority_endorsement", "evidence-based_persuasion", "expert_endorsement", "logical_appeal",
                   "misrepresentation"),
}
FAMILY_OF = {t: f for f, ts in FAMILIES.items() for t in ts}
# Next-token refusal / compliance proxy (same first words as the a5 pilot extraction; a working choice,
# so everything built on it is EXPLORATORY and labelled proxy).
REFUSAL_WORDS = ("I", "Sorry", "As", "Unfortunately")
COMPLIANCE_WORDS = ("Sure", "Certainly", "Here", "Absolutely")


# ------------------------------------------------------------------------------------------------ prompt plan
def prompt_key(text):
    """Stable id of one prompt: sha256 of its text (first 20 hex). Identical prompts (e.g. a SORRY-Bench base
    instruction shared by 19 mutations) get one key and are run once. A hash, never the text, is stored."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def prompt_plan(pairs, sources=SOURCES):
    """From the pairs table (with text columns) to (prompts, plan) WITHOUT text in either output.

    prompts: list of (key, text) for every unique prompt, sorted by key (deterministic order; text stays in memory).
    plan: DataFrame one row per pair: pair_id, source, intent, tactic, split, holdout_group, key_v, key_a."""
    df = pairs[pairs["source"].isin(sources)].copy()
    need = {"pair_id", "source", "intent", "tactic", "split", "vanilla_prompt", "adversarial_prompt"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"pairs table lacks columns {sorted(missing)}")
    df = df.dropna(subset=["vanilla_prompt", "adversarial_prompt"])
    # Pairs whose two prompts are identical (SORRY-Bench base and StrongREJECT none rows in pairs_table_v1_final-2)
    # have no displacement by construction: they would add zero vectors to every per-tactic analysis, so they go.
    df = df[df["vanilla_prompt"] != df["adversarial_prompt"]]
    df["key_v"] = df["vanilla_prompt"].map(prompt_key)
    df["key_a"] = df["adversarial_prompt"].map(prompt_key)
    texts = dict(zip(df["key_v"], df["vanilla_prompt"]))
    texts.update(zip(df["key_a"], df["adversarial_prompt"]))
    prompts = sorted(texts.items())
    if "holdout_group" not in df.columns:
        df["holdout_group"] = ""
    for c in ("pair_id", "source", "intent", "tactic", "split", "holdout_group"):   # one type per column for parquet
        df[c] = df[c].fillna("").astype(str)
    plan = df[["pair_id", "source", "intent", "tactic", "split", "holdout_group", "key_v", "key_a"]].reset_index(drop=True)
    return prompts, plan


def keys_hash(keys):
    """Fingerprint of the ordered prompt list (a resumed extraction refuses to mix chunks from a different list)."""
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ loading
def load(folder, view="prenorm"):
    """Read the extraction folder. view: prenorm (pre-registered vector), postnorm, or an int = a sweep layer
    (hidden_states index, exploratory). Returns dict(H, prompts, pairs, meta, view)."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    prompts = pd.read_parquet(folder / "prompts.parquet")
    pairs = pd.read_parquet(folder / "pairs.parquet")
    if view == "prenorm":
        H = np.load(folder / "prenorm_last.npy")
    elif view == "postnorm":
        H = np.load(folder / "postnorm_last.npy")
    else:
        layers = list(meta["sweep_layers"])
        if int(view) not in layers:
            raise KeyError(f"layer {view} not in the sweep {layers}")
        H = np.load(folder / "sweep.npy", mmap_mode="r")[:, layers.index(int(view))].astype(np.float32)
    assert len(H) == len(prompts), "array rows and prompts.parquet disagree"
    return dict(H=np.asarray(H, np.float32), prompts=prompts, pairs=pairs, meta=meta, view=view)


def layer_label(d):
    n = d["meta"].get("num_hidden_layers", "last")
    if d["view"] == "prenorm":
        return f"{n} (last, pre-norm)"
    if d["view"] == "postnorm":
        return f"{n} (last, post-norm)"
    return f"{d['view']} (no norm)"


def pair_arrays(d, source):
    """(Hv, Ha, meta) for one source, rows aligned; pairs with a non-finite prompt are dropped (count in meta.attrs)."""
    p = d["pairs"][d["pairs"]["source"] == source].reset_index(drop=True)
    pr = d["prompts"]
    finite = pr["finite"].to_numpy(bool)
    ok = finite[p["row_v"].to_numpy()] & finite[p["row_a"].to_numpy()]
    p = p[ok].reset_index(drop=True)
    Hv, Ha = d["H"][p["row_v"].to_numpy()], d["H"][p["row_a"].to_numpy()]
    meta = p[["pair_id", "source", "intent", "tactic", "split", "row_v", "row_a"]].copy()
    meta.attrs["dropped_nonfinite"] = int((~ok).sum())
    return Hv, Ha, meta


# ------------------------------------------------------------------------------------------------ analyses
def _share_ci(x, n_boot, seed):
    x = np.asarray(x, float)
    return lib.bootstrap_ci(x, np.mean, n=n_boot, seed=seed) if len(x) else (np.nan,) * 3


def per_tactic_masking(Hv, Ha, meta, r_hat, midpoint, n_boot=2000, seed=0):
    """A2 per tactic (exploratory): for held-out pairs of each tactic, mean dp along r_hat and the share of adversarial
    prompts whose projection falls below the midpoint (the H1 line), each with a 95% bootstrap CI over pairs; plus
    the share of the tactic's VANILLA prompts below the midpoint (the baseline the adversarial share is compared to)."""
    pv, pa = lib.project(Hv, r_hat), lib.project(Ha, r_hat)
    dp = pa - pv
    ev = (meta["split"] != "train").to_numpy()
    tac = meta["tactic"].astype(str).to_numpy()
    out = {}
    for t in sorted(set(tac[ev])):
        m = ev & (tac == t)
        if m.sum() < 2:
            continue
        out[t] = dict(mean_dp=lib.bootstrap_ci(dp[m], np.mean, n=n_boot, seed=seed),
                      cohens_d=lib.cohens_d(dp[m]),
                      share_adv_below=_share_ci(pa[m] < midpoint, n_boot, seed),
                      share_van_below=float(np.mean(pv[m] < midpoint)),
                      n=int(m.sum()), family=FAMILY_OF.get(t, "other"))
    return out


def tactic_alignment(delta, tactics, r_hat):
    """How much of each tactic's average move lies along the harmfulness direction: cos(mean delta_t, r_hat) (-1 =
    straight towards benign) and the length of the mean move. {tactic: dict(cos_to_r, mean_norm, n)}."""
    means = g.mean_displacements(delta, tactics)
    tac = np.asarray(tactics).astype(str)
    return {t: dict(cos_to_r=float(v @ r_hat / np.linalg.norm(v)), mean_norm=float(np.linalg.norm(v)),
                    n=int((tac == t).sum())) for t, v in means.items()}


def angle_matrix(delta, tactics, k=3):
    """Smallest principal angle (degrees) between every two tactics' k-dimensional displacement subspaces."""
    tac = np.asarray(tactics).astype(str)
    names = sorted(set(tac))
    U = {t: g.tactic_subspace(np.asarray(delta)[tac == t], k=k) for t in names}
    A = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            A[i, j] = 0.0 if i == j else g.principal_angles(U[a], U[b])[0]
    return names, A


def knn_self_share(delta, tactics, k=10):
    """Share of each tactic's k nearest neighbours (cosine) that have the same tactic, and the chance level."""
    comp = g.knn_composition(delta, tactics, k=k)
    tac = pd.Series(np.asarray(tactics).astype(str))
    chance = tac.value_counts(normalize=True)
    return {t: dict(self_share=float(comp.loc[t, t]), chance=float(chance[t])) for t in comp.index}


def rank_by_tactic(delta, tactics):
    """A8 per tactic: effective rank and directions holding 90% of the movement (needs >= 2 pairs)."""
    tac = np.asarray(tactics).astype(str)
    out = {}
    for t in sorted(set(tac)):
        D = np.asarray(delta)[tac == t]
        if len(D) >= 2:
            s = g.singular_values(D)
            out[t] = dict(effective_rank=g.effective_rank(s), directions_90=g.energy_rank(s, 0.9), n=len(D))
    return out


def refusal_proxy(prompts, threshold=0.5):
    """Next-token proxy label per prompt: refusal-word mass at prompt[-1] >= threshold. NOT judge labels.
    None when the extraction has no next-token stats (e.g. a converted harvest: refusal_mass all NaN)."""
    mass = prompts["refusal_mass"].to_numpy(float) if "refusal_mass" in prompts else np.array([np.nan])
    if not np.isfinite(mass).any():
        return None
    return mass >= threshold


# ------------------------------------------------------------------------------------------------ figures
def _fig_tools():
    import matplotlib.pyplot as plt
    from analysis import figures as F
    from analysis import plotstyle as ps
    return plt, F, ps


FAMILY_COLORS = {"encoded (may not be understood)": "#9a9a92", "translation": "#7a5195",
                 "writing style": "#2f6db5", "persuasion": "#d9822b", "other": "#52514e"}


def plot_tactic_masking(rows, wj, path, kind, model, layer, position="prompt[-1]"):
    """Two panels per tactic, sorted by mean dp: (a) mean dp with 95% CI; (b) share of adversarial prompts below the
    WildJailbreak midpoint (with CI) next to the share of the same tactic's vanilla prompts (black tick)."""
    plt, F, ps = _fig_tools()
    ps.apply_style()
    tacs = sorted(rows, key=lambda t: rows[t]["mean_dp"][0])
    y = np.arange(len(tacs))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.36 * len(tacs) + 2.2), sharey=True)
    col = [FAMILY_COLORS[rows[t]["family"]] for t in tacs]
    est = [rows[t]["mean_dp"][0] for t in tacs]
    a1.barh(y, est, color=col)
    a1.errorbar(est, y, xerr=F._ci_err(est, [rows[t]["mean_dp"][1] for t in tacs], [rows[t]["mean_dp"][2] for t in tacs]),
                fmt="none", ecolor=ps.INK2, capsize=2, lw=1)
    if wj is not None:
        a1.axvline(wj[0], color=ps.INK, ls="--", lw=1)
        a1.text(wj[0], len(tacs) - 0.4, " WildJailbreak harmful", fontsize=6.5, color=ps.INK, va="bottom")
    a1.axvline(0, color=ps.INK, lw=1)
    a1.set_yticks(y, tacs, fontsize=7)
    a1.set(xlabel="mean dp per pair along r_hat (negative = looks less harmful)")
    sh = [rows[t]["share_adv_below"][0] for t in tacs]
    a2.barh(y, sh, color=col)
    a2.errorbar(sh, y, xerr=F._ci_err(sh, [rows[t]["share_adv_below"][1] for t in tacs],
                                      [rows[t]["share_adv_below"][2] for t in tacs]), fmt="none", ecolor=ps.INK2, capsize=2, lw=1)
    a2.scatter([rows[t]["share_van_below"] for t in tacs], y, marker="|", s=120, color=ps.INK, zorder=4,
               label="same tactic's vanilla prompts")
    a2.axvline(0.5, color=ps.MUTED, ls=":", lw=1)
    a2.set(xlabel="share of adversarial prompts below the midpoint (H1 line)", xlim=(0, 1))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values() if c in col]
    labels = [f for f, c in FAMILY_COLORS.items() if c in col]
    # above the panel: inside it, the legend covers the longest bars (100% rows sit at the bottom)
    a2.legend(handles + a2.get_legend_handles_labels()[0], labels + ["same tactic's vanilla prompts"], fontsize=6.5,
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    return F._finish(fig, kind, model, layer, position, "A2 per tactic (SORRY-Bench): masking along the WildJailbreak r_hat",
                     "held-out pairs; 95% bootstrap CIs over pairs (2000, seed 0); r_hat and midpoint from WildJailbreak vanilla TRAIN", path)


def plot_angle_matrix(names, A, path, kind, model, layer, position="prompt[-1]", k=3):
    plt, F, ps = _fig_tools()
    ps.apply_style()
    order = g.cluster_order(np.cos(np.radians(A)))
    A = A[np.ix_(order, order)]
    names = [names[i] for i in order]
    fig, ax = plt.subplots(figsize=(0.55 * len(names) + 3, 0.5 * len(names) + 2))
    im = ax.imshow(A, cmap="viridis_r", vmin=0, vmax=90)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=5.5, color="white" if A[i, j] < 45 else ps.INK)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(names)), names, fontsize=6.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="smallest principal angle (degrees; 0 = shared direction)")
    return F._finish(fig, kind, model, layer, position, f"A5: principal angles between tactic subspaces (k = {k})",
                     "harmful SORRY-Bench pairs, all splits; ordered by clustering", path)


def plot_knn_self(ks, path, kind, model, layer, position="prompt[-1]", k=10):
    plt, F, ps = _fig_tools()
    ps.apply_style()
    tacs = sorted(ks, key=lambda t: ks[t]["self_share"])
    y = np.arange(len(tacs))
    fig, ax = plt.subplots(figsize=(7.5, 0.33 * len(tacs) + 2))
    ax.barh(y, [ks[t]["self_share"] for t in tacs], color=[FAMILY_COLORS[FAMILY_OF.get(t, "other")] for t in tacs])
    ax.scatter([ks[t]["chance"] for t in tacs], y, marker="|", s=120, color=ps.INK, zorder=4, label="chance")
    ax.set_yticks(y, tacs, fontsize=7)
    ax.set(xlabel=f"share of the {k} nearest neighbours with the same tactic", xlim=(0, 1))
    ax.legend(fontsize=7, loc="lower right")
    return F._finish(fig, kind, model, layer, position, "A5: does each tactic form its own cluster? (k-NN on displacements)",
                     "held-out harmful SORRY-Bench pairs; cosine neighbours; tick = chance level", path)


def plot_alignment(al, path, kind, model, layer, position="prompt[-1]"):
    plt, F, ps = _fig_tools()
    ps.apply_style()
    tacs = sorted(al, key=lambda t: al[t]["cos_to_r"])
    y = np.arange(len(tacs))
    fig, ax = plt.subplots(figsize=(7.5, 0.33 * len(tacs) + 2))
    ax.barh(y, [al[t]["cos_to_r"] for t in tacs], color=[FAMILY_COLORS[FAMILY_OF.get(t, "other")] for t in tacs])
    ax.axvline(0, color=ps.INK, lw=1)
    ax.set_yticks(y, tacs, fontsize=7)
    ax.set(xlabel="cos(tactic's mean displacement, r_hat)   (-1 = straight towards benign, 0 = sideways)", xlim=(-1, 1))
    return F._finish(fig, kind, model, layer, position, "A5: how much of each tactic's move is along the harmfulness direction",
                     "harmful SORRY-Bench pairs, all splits; r_hat from WildJailbreak vanilla TRAIN", path)


# ------------------------------------------------------------------------------------------------ one script
def run_labels(meta):
    """(model label, KIND, note) for titles and results. The Qwen-1.5B pilot (meta without model_tag) is PILOT; a
    folder from harvest/ws3_harvest.py is RESULT, and names its serving config unless it is the full C1 run."""
    pilot_note = "PILOT (plumbing-scale model). A2 per tactic, A5 and A8 are EXPLORATORY; not results."
    if "model_tag" not in meta:
        return MODEL_TAG, "PILOT", pilot_note
    model = meta["model_tag"]
    if model == MODEL_TAG:   # the pilot model re-run through harvest/ws3_harvest.py (e.g. on another pairs table)
        return f"{model} [{Path(meta.get('pairs_table', '')).stem or 'pairs table'}]", "PILOT", pilot_note
    if meta.get("config", "c1") != "c1" or meta.get("subset"):
        model += f" [{meta.get('config_name', meta.get('config'))}" + (f", sub{meta['subset']}" if meta.get("subset") else "") + "]"
    return model, "RESULT", ("WS3 harvest (own HF pipeline, no vLLM reference). A2 H1 is pre-registered; A2 per tactic, "
                             "A5 and A8 are EXPLORATORY.")


def run_all(folder, view="prenorm", fig_dir="figures/WS3/pilot", results_path=None, n_boot=2000, seed=0, k_angle=3, labels=None):
    """Every WS3 pilot analysis and figure from one call. Returns (results dict, {name: png path}).
    labels: optional WildGuard labels (analysis/judge_labels.py); they only add to A8, nothing else changes."""
    plt, F, ps = _fig_tools()
    d = load(folder, view)
    model, kind, note = run_labels(d["meta"])
    layer, pos = layer_label(d), "prompt[-1]"
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if view == "prenorm" else f"_{view}"
    figs, res = {}, dict(label=ps.tag(kind, model, layer, pos), view=view, extraction=d["meta"], note=note)

    # A2 on WildJailbreak: r_hat, midpoint, F2
    Hv, Ha, mw = pair_arrays(d, "WildJailbreak")
    a2 = g.a2_projection(Hv, Ha, mw, n_boot=n_boot, seed=seed)
    ev = (mw["split"] != "train").to_numpy()
    harm = (mw["intent"] == "harmful").to_numpy()
    te = (mw["split"] == "test").to_numpy() & harm
    F.TITLE_NOTE = "PILOT (Qwen2.5-1.5B)" if kind == "PILOT" else None   # A2 / H1 is pre-registered
    figs["A2_F2_wildjailbreak"] = F.a2_f2_histograms(a2, mw, fig_dir / f"A2_F2_wildjailbreak{tag}.png", kind, model, layer, pos)
    res["A2_wildjailbreak"] = dict(
        n_pairs=len(mw), dropped_nonfinite=mw.attrs["dropped_nonfinite"], midpoint=a2["midpoint"],
        mean_dp_heldout=a2["overall"],
        cohens_d_heldout={i: lib.cohens_d(a2["dp"][ev & (harm if i == "harmful" else ~harm)]) for i in ("harmful", "benign")},
        h1_test=dict(n=int(te.sum()), share_adv_below=_share_ci(a2["pa"][te] < a2["midpoint"], n_boot, seed),
                     share_van_below=_share_ci(a2["pv"][te] < a2["midpoint"], n_boot, seed),
                     difference=_share_ci((a2["pa"][te] < a2["midpoint"]).astype(float) -
                                          (a2["pv"][te] < a2["midpoint"]).astype(float), n_boot, seed)))

    # A7 on WildJailbreak held-out pairs: style = benign dp, content = harmful - benign (H3 part a; parts b and c need
    # the G2 and harmfulness probes). One tactic label so far, so the pooled row is the same as that tactic's row.
    sc = g.style_content(a2["dp"][ev], mw["tactic"].astype(str).to_numpy()[ev], mw["intent"].to_numpy()[ev],
                         n_boot=n_boot, seed=seed)
    if sc:
        shown = {"_pooled": sc["_pooled"]} if len(sc) == 2 else sc
        figs["A7_wildjailbreak"] = F.a7_style_content(shown, fig_dir / f"A7_wildjailbreak{tag}.png", kind, model, layer, pos)
    res["A7_wildjailbreak"] = dict(
        pooled=sc.get("_pooled"), per_tactic={t: v for t, v in sc.items() if t != "_pooled"},
        note="held-out pairs; style = benign mean dp, content = harmful minus benign; bootstrap within tactic. "
             "H3 part (a) only: G2 probe (b) and harmfulness-probe margin (c) are not computed here")
    r_hat, mid = a2["r_hat"], a2["midpoint"]

    # SORRY-Bench: per-tactic masking along the WildJailbreak r_hat
    Sv, Sa, ms = pair_arrays(d, "SORRY-Bench")
    if len(ms) == 0:
        F.TITLE_NOTE = None
        res["SORRY-Bench"] = "no SORRY-Bench pairs in the extraction"
        return res, figs
    sdelta = Sa - Sv
    stac = ms["tactic"].astype(str).to_numpy()
    s_ev = (ms["split"] != "train").to_numpy()
    rows = per_tactic_masking(Sv, Sa, ms, r_hat, mid, n_boot=n_boot, seed=seed)
    base_side = float(np.mean(lib.project(Sv, r_hat) > mid))
    F.TITLE_NOTE = "EXPLORATORY"   # the title already starts with the kind (tag)
    figs["A2_tactics_sorry"] = plot_tactic_masking(rows, a2["overall"]["harmful"], fig_dir / f"A2_tactics_sorry{tag}.png",
                                                   kind, model, layer, pos)
    res["A2_sorry_per_tactic"] = dict(n_pairs=len(ms), dropped_nonfinite=ms.attrs["dropped_nonfinite"],
                                      sorry_vanilla_share_on_harmful_side=base_side, per_tactic=rows)

    # A5: tactic geometry on SORRY-Bench displacements
    means = g.mean_displacements(sdelta, stac)
    figs["A5_cosine"] = F.a5_cosine_heatmap(means, fig_dir / f"A5_cosine{tag}.png", kind, model, layer, pos)
    figs["A5_pca"] = F.a5_pca_scatter(means, fig_dir / f"A5_pca{tag}.png", kind, model, layer, pos)
    names, A = angle_matrix(sdelta, stac, k=k_angle)
    figs["A5_angles"] = plot_angle_matrix(names, A, fig_dir / f"A5_angles{tag}.png", kind, model, layer, pos, k=k_angle)
    ks = knn_self_share(sdelta[s_ev], stac[s_ev], k=10)
    figs["A5_knn"] = plot_knn_self(ks, fig_dir / f"A5_knn{tag}.png", kind, model, layer, pos)
    al = tactic_alignment(sdelta, stac, r_hat)
    figs["A5_alignment"] = plot_alignment(al, fig_dir / f"A5_alignment{tag}.png", kind, model, layer, pos)
    C = g.cosine_matrix(np.array([means[t] for t in names]))
    res["A5"] = dict(tactics=names, families={t: FAMILY_OF.get(t, "other") for t in names},
                     cosine={a: {b: float(C[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)},
                     principal_angles_deg={a: {b: float(A[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)},
                     k_angle=k_angle, knn_self_share=ks, alignment_with_r_hat=al)

    # A8: low-rank structure + harmfulness vs refusal (next-token proxy)
    s_all = g.singular_values(sdelta)
    figs["A8_spectrum_sorry"] = F.a8_spectrum(s_all, fig_dir / f"A8_spectrum_sorry{tag}.png", kind, model, layer, pos)
    s_wj = g.singular_values(Ha - Hv)
    figs["A8_spectrum_wildjailbreak"] = F.a8_spectrum(s_wj, fig_dir / f"A8_spectrum_wildjailbreak{tag}.png", kind, model, layer, pos)
    refused = refusal_proxy(d["prompts"])
    rows_tr = np.r_[mw.loc[mw["split"] == "train", "row_v"], mw.loc[mw["split"] == "train", "row_a"],
                    ms.loc[ms["split"] == "train", "row_a"]]
    rows_tr = np.unique(rows_tr)
    angle = None
    if refused is not None and refused[rows_tr].any() and (~refused[rows_tr]).any():
        rdir = lib.fit_direction(d["H"][rows_tr][refused[rows_tr]], d["H"][rows_tr][~refused[rows_tr]])
        angle = g.angle_between(r_hat, rdir)
        F.TITLE_NOTE = "EXPLORATORY; refusal = next-token PROXY, not judge labels"
        figs["A8_angle"] = F.a8_angle_arc(angle, fig_dir / f"A8_angle_proxy{tag}.png", kind, model, layer, pos)
    res["A8"] = dict(sorry=dict(effective_rank=g.effective_rank(s_all), directions_90=g.energy_rank(s_all, 0.9),
                                singular_values_top50=s_all[:50]),
                     wildjailbreak=dict(effective_rank=g.effective_rank(s_wj), directions_90=g.energy_rank(s_wj, 0.9),
                                        singular_values_top50=s_wj[:50]),
                     per_tactic=rank_by_tactic(sdelta, stac),
                     angle_harm_vs_refusal_proxy_deg=angle,
                     refusal_proxy=("not in this extraction (no next-token stats)" if refused is None else
                                    dict(words=REFUSAL_WORDS, threshold=0.5, train_rows=int(len(rows_tr)),
                                         share_refused_train=float(refused[rows_tr].mean()))))
    if labels is not None:
        from analysis import judge_labels as jl
        wg_a8 = jl.refusal_angle(d["H"], rows_tr, r_hat, jl.align(d["prompts"], labels))
        res["A8"]["refusal_wildguard"] = wg_a8
        if wg_a8["angle_deg"] is not None:
            F.TITLE_NOTE = "EXPLORATORY; refusal = WildGuard on vLLM replies"
            figs["A8_angle_wildguard"] = F.a8_angle_arc(wg_a8["angle_deg"], fig_dir / f"A8_angle_wildguard{tag}.png", kind, model, layer, pos)
    F.TITLE_NOTE = None
    if results_path:
        from analysis import ws3_data
        ws3_data.save_json(res, results_path)
    return res, figs


def summary_lines(res):
    """Key numbers as plain lines (for the notebook and the progress log)."""
    w = res["A2_wildjailbreak"]
    f = lambda c: f"{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]"  # noqa: E731
    out = [res["label"],
           f"WildJailbreak: mean dp harmful {f(w['mean_dp_heldout']['harmful'])}, benign {f(w['mean_dp_heldout']['benign'])}; "
           f"H1 share {w['h1_test']['share_adv_below'][0]:.2f} vs vanilla {w['h1_test']['share_van_below'][0]:.2f}"]
    p = res.get("A7_wildjailbreak", {}).get("pooled")
    if p:
        out.append(f"A7 WildJailbreak (pooled): style {f(p['style'])}, content {f(p['content'])}")
    if "A2_sorry_per_tactic" in res:
        s = res["A2_sorry_per_tactic"]
        out.append(f"SORRY-Bench base instructions on the harmful side of the midpoint: {s['sorry_vanilla_share_on_harmful_side']:.2f}")
        for t, r in sorted(s["per_tactic"].items(), key=lambda kv: kv[1]["mean_dp"][0]):
            out.append(f"  {t:28s} dp {f(r['mean_dp'])}  below midpoint {r['share_adv_below'][0]:.2f} "
                       f"(vanilla {r['share_van_below']:.2f})  n={r['n']}  [{r['family']}]")
        a8 = res["A8"]
        out.append(f"A8: effective rank SORRY {a8['sorry']['effective_rank']:.1f} ({a8['sorry']['directions_90']} dirs for 90%), "
                   f"WildJailbreak {a8['wildjailbreak']['effective_rank']:.1f} ({a8['wildjailbreak']['directions_90']}); "
                   f"angle r_hat vs refusal proxy: {a8['angle_harm_vs_refusal_proxy_deg']}")
        if "refusal_wildguard" in a8:
            out.append(f"A8: angle r_hat vs WildGuard refusal: {a8['refusal_wildguard']['angle_deg']}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--view", default="prenorm")
    a = ap.parse_args()
    v = a.view if a.view in ("prenorm", "postnorm") else int(a.view)
    r, _ = run_all(a.folder, v, results_path=f"results/WS3_pilot_{MODEL_TAG}_{v}.json")
    print("\n".join(summary_lines(r)))
