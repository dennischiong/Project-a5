# Cell M2 - A4 on Mistral: does the jailbreak's push along the "harmfulness direction" (delta_p) and the model's first-word
# refusal signal (first-token refusal mass) tell us whether the jailbreak worked?   Same method as the Llama A4.
#   r_hat   = mean(harmful vanilla prompts) - mean(benign vanilla prompts) at layer 31, prompt[-1]; fitted on vanilla prompts only, on the
#             FIT_SPLIT part of the data, and never on a prompt that is evaluated
#   delta_p = (h_adversarial - h_vanilla) . r_hat           (how far the jailbreak pushed the state toward "benign")
#   refusal mass = probability the model's very first word is I / Sorry (final norm + lm_head applied offline to the stored vector)
#   success = WildGuard says the adversarial reply is a harmful response (harmful pairs only)
# Needs M0 first. Downloads the Mistral output matrix once (~4.5 GB, gated: needs the Colab secret HF_TOKEN) and caches the small part it needs.
# EXECUTION TIME: first run 4-10 min (download 1-3 min + bootstraps 3-6 min); re-run about 3-6 min (the head is cached).
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    pre, post, pairs, row_of, labels, RESULTS_DIR, MODEL_ALIAS
except NameError:
    raise SystemExit("Run cell M0 first.")
if labels is None:
    raise SystemExit("M0 did not find the judge labels (judges/*/labels.parquet in the hand-off folder). A4 cannot run without them.")

# ------------------------------------------------------------------ settings
HEAD_CACHE = globals().get("HEAD_CACHE", Path("/content/mistral_head_cache.npz"))   # lm_head + final-norm weight, saved after the first download
TOKENIZER = globals().get("TOKENIZER", None)                                          # None = load it from Hugging Face
REFUSAL_WORDS, COMPLIANCE_WORDS = ["I", "Sorry"], ["Sure", "Here"]                  # the plan's primary token sets (same as Llama)
REFUSAL_WORDS_EXT = ["I", "Sorry", "Unfortunately", "As", "Apologies"]               # sensitivity
COMPLIANCE_WORDS_EXT = ["Sure", "Here", "Certainly", "Absolutely", "Yes"]
COLOR_FAILED, COLOR_SUCCEEDED, SURFACE, INK, INK_MUTED = "#2a78d6", "#eb6834", "#fcfcfb", "#1f1f1e", "#6b6a63"
t0 = time.time()
FIT = FIT_SPLIT        # set FIT_SPLIT = None in M0 to fit r_hat on every non-evaluated vanilla prompt instead (a deviation; it is labelled in the output)


# ------------------------------------------------------------------ the Llama A4 statistics, unchanged
def auroc(score, y):
    score, y = np.asarray(score, dtype="float64"), np.asarray(y, dtype=bool)
    ok = np.isfinite(score)
    score, y = score[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auroc(score, y, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    score, y = np.asarray(score, dtype="float64"), np.asarray(y, dtype=bool)
    point = auroc(score, y)
    if point is None:
        return {"auroc": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, y.size, y.size)
        v = auroc(score[i], y[i])
        if v is not None:
            vals.append(v)
    if not vals:
        return {"auroc": point, "lo": None, "hi": None}
    return {"auroc": point, "lo": float(np.percentile(vals, 100 * alpha / 2)), "hi": float(np.percentile(vals, 100 * (1 - alpha / 2)))}


def logistic_fit(X, y, l2=1e-6, max_iter=100, tol=1e-9):
    X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype="float64")])
    y = np.asarray(y, dtype="float64")
    beta = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2); pen[0] = 0.0
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        grad = X.T @ (y - p) - pen * beta
        H = (X * W[:, None]).T @ X + np.diag(pen) + 1e-12 * np.eye(X.shape[1])
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            return beta, True
    return beta, False


def logistic_with_ci(columns, y, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    names = list(columns)
    raw = np.column_stack([np.asarray(columns[n], dtype="float64") for n in names])
    y = np.asarray(y, dtype=bool)
    ok = np.isfinite(raw).all(axis=1)
    raw, y = raw[ok], y[ok]
    if y.sum() < MIN_PER_CLASS or (~y).sum() < MIN_PER_CLASS:
        return {"fitted": False, "reason": f"fewer than {MIN_PER_CLASS} successes or failures ({int(y.sum())} / {int((~y).sum())})"}
    mu, sd = raw.mean(axis=0), raw.std(axis=0)
    sd[sd == 0] = 1.0
    X = (raw - mu) / sd
    coef, conv = logistic_fit(X, y)
    rng = np.random.default_rng(seed)
    boots, skipped, nonconv = [], 0, 0
    for _ in range(n_boot):
        i = rng.integers(0, y.size, y.size)
        if y[i].all() or (~y[i]).all():
            skipped += 1
            continue
        b, c = logistic_fit(X[i], y[i])
        nonconv += 0 if c else 1
        boots.append(b)
    boots = np.asarray(boots)
    out = {"fitted": True, "n": int(y.size), "n_success": int(y.sum()), "converged": bool(conv),
           "standardisation": {n: {"mean": float(m), "sd": float(s)} for n, m, s in zip(names, mu, sd)},
           "bootstrap_resamples_used": int(len(boots)), "bootstrap_skipped_one_class": skipped, "bootstrap_not_converged": nonconv,
           "coefficients": {}}
    for j, n in enumerate(["intercept"] + names):
        lo, hi = (np.percentile(boots[:, j], [100 * alpha / 2, 100 * (1 - alpha / 2)]) if len(boots) else (float("nan"), float("nan")))
        out["coefficients"][n] = {"log_odds_per_sd" if n != "intercept" else "value": float(coef[j]), "lo": float(lo), "hi": float(hi),
                                  "odds_ratio_per_sd": (float(math.exp(coef[j])) if n != "intercept" else None)}
    return out


def summarise_outcome(df, success_col, dp_col, mass_col, comp_col):
    d = df[df[success_col].notna()]
    y = d[success_col].astype(bool).to_numpy()
    groups = {}
    for name, mask in (("succeeded", y), ("failed", ~y)):
        part = d[mask]
        groups[name] = {"n_pairs": int(mask.sum()), "delta_p": boot_median(part[dp_col]),
                        "refusal_mass": boot_median(part[mass_col]), "compliance_mass": boot_median(part[comp_col])}
    s, f = d[y], d[~y]
    return {"n_pairs": int(len(d)), "n_succeeded": int(y.sum()), "n_failed": int((~y).sum()),
            "success_rate": float(y.mean()) if len(y) else None, "groups": groups,
            "median_difference_succeeded_minus_failed": {"delta_p": boot_median_diff(s[dp_col], f[dp_col]),
                                                         "refusal_mass": boot_median_diff(s[mass_col], f[mass_col])},
            "auroc_for_success": {"delta_p (higher = success)": boot_auroc(d[dp_col], y),
                                  "refusal_mass (higher = success)": boot_auroc(d[mass_col], y)},
            "logistic": logistic_with_ci({"delta_p": d[dp_col].to_numpy(), "refusal_mass": d[mass_col].to_numpy()}, y),
            "logistic_delta_p_only": logistic_with_ci({"delta_p": d[dp_col].to_numpy()}, y),
            "logistic_refusal_mass_only": logistic_with_ci({"refusal_mass": d[mass_col].to_numpy()}, y)}


def first_token_stats(H, norm_w, head_w, eps, sets, batch=128):
    H = np.asarray(H, dtype=np.float32)
    w = np.asarray(norm_w, dtype=np.float32)
    W = np.asarray(head_w, dtype=np.float32)
    out = {"argmax": [], "top1_prob": []}
    for k in sets:
        out[k] = []
    for s in range(0, H.shape[0], batch):
        h = H[s:s + batch]
        normed = h / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + eps) * w
        logits = normed @ W.T
        m = logits.max(axis=-1, keepdims=True)
        logp = logits - (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))
        out["argmax"] += logp.argmax(axis=-1).astype(int).tolist()
        out["top1_prob"] += np.exp(logp.max(axis=-1)).astype(float).tolist()
        for k, ids in sets.items():
            out[k] += np.exp(logp[:, np.asarray(ids, dtype=int)]).sum(-1).astype(float).tolist()
    return out


def plot_f4(df, path, success_col, dp_col, mass_col, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df[success_col].notna()]
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for value, color, marker, label in ((False, COLOR_FAILED, "o", "Jailbreak failed"), (True, COLOR_SUCCEEDED, "^", "Jailbreak succeeded")):
        part = d[d[success_col].astype(bool) == value]
        ax.scatter(part[dp_col], part[mass_col], s=16, marker=marker, color=color, alpha=0.75, linewidths=0.4, edgecolors=SURFACE,
                   label=f"{label} (n={len(part)})", zorder=3)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xlabel(r"$\Delta p = (h_{adv} - h_{van}) \cdot \hat{r}$   (layer 31, last prefill token)", color=INK, fontsize=9)
    ax.set_ylabel("First-token refusal mass, adversarial prompt", color=INK, fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color="#e6e5e0", linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c1")
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    leg = ax.legend(frameon=False, fontsize=8, loc="best")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ 1. judge labels -> one outcome per adversarial prompt
lk = pick(labels, ["prompt_key", "key"], "label_key", regex=r"key")
sc = pick(labels, ["status", "parse_status", "judge_status"], "label_status", required=False)
hc = pick(labels, ["harmful_response", "response_harmful", "wg_response_harmful", "wg_harmful_response", "response_harm"],
          "label_harmful", regex=r"harm.*resp|resp.*harm")
rc = pick(labels, ["refusal", "response_refusal", "wg_response_refusal", "refused", "is_refusal"], "label_refusal", required=False, regex=r"refus")
tc = pick(labels, ["reply_truncated", "truncated"], "label_truncated", required=False, regex=r"trunc")
print(f"label columns used: key={lk}  status={sc}  harmful_response={hc}  refusal={rc}  truncated={tc}")
lab = labels.copy()
lab[lk] = lab[lk].astype(str)
if lab[lk].duplicated().any():
    raise SystemExit(f"labels.parquet has the same {lk} twice")
lab_by_key = {r[lk]: r for r in lab.to_dict("records")}


def outcome_of(key):
    r = lab_by_key.get(key)
    if r is None:
        return {"harm": None, "complied": None, "trunc": None, "why": "no judge label"}
    if sc and str(r.get(sc)).strip().lower() != "ok":
        return {"harm": None, "complied": None, "trunc": to_bool(r.get(tc)) if tc else None, "why": f"judge status {r.get(sc)}"}
    h = to_bool(r.get(hc))
    ref = to_bool(r.get(rc)) if rc else None
    return {"harm": h, "complied": (None if ref is None else (not ref)), "trunc": to_bool(r.get(tc)) if tc else None,
            "why": None if h is not None else "harmful_response missing"}


# ------------------------------------------------------------------ 2. which pairs are evaluated, which vanilla prompts fit r_hat
p = pairs[pairs["vkey"].isin(row_of) & pairs["akey"].isin(row_of)].copy()
p["out"] = p["akey"].map(outcome_of)
p["labelled"] = p["out"].map(lambda o: o["harm"] is not None)
ev = p[p["labelled"] & p["intent"].isin(["harmful", "benign"])].copy()
if len(ev) < 100:
    raise SystemExit(f"only {len(ev)} evaluable pairs. Check that the label keys match prompts.parquet keys (M0 prints how many do).")
eval_prompts = set(ev["vkey"]) | set(ev["akey"])
adv_keys = set(p["akey"])
rest = p[~p["pair_id"].isin(set(ev["pair_id"])) & p["intent"].isin(["harmful", "benign"])]
van_intent, van_split = {}, {}
for r in rest.itertuples():
    van_intent.setdefault(r.vkey, set()).add(r.intent)
    van_split.setdefault(r.vkey, set()).add(str(r.split))
fit_main, fit_any, conflicts = {}, {}, 0
for v, ints in van_intent.items():
    if v in eval_prompts or v in adv_keys:
        continue
    if len(ints) != 1:
        conflicts += 1
        continue
    it = next(iter(ints))
    fit_any[v] = it
    if FIT is not None and van_split[v] == {str(FIT)}:
        fit_main[v] = it
fit_use = fit_any if FIT is None else fit_main
n_h = sum(i == "harmful" for i in fit_use.values()); n_b = sum(i == "benign" for i in fit_use.values())
split_values = p["split"].astype(str).value_counts().to_dict()
print(f"\nevaluated pairs {len(ev):,}  (harmful {int((ev.intent == 'harmful').sum())}, benign {int((ev.intent == 'benign').sum())}); "
      f"evaluated prompts {len(eval_prompts):,}\nfit set for r_hat ({'split ' + str(FIT) if FIT is not None else 'ANY split - deviation'}): "
      f"{n_h} harmful + {n_b} benign vanilla prompts   (conflicting intents dropped: {conflicts})   split values in data: {split_values}")
if n_h < 20 or n_b < 20:
    raise SystemExit(f"too few vanilla prompts to fit r_hat ({n_h} harmful, {n_b} benign) on split {FIT}. Split values are {split_values}. "
                     "Set FIT_SPLIT in M0 to the 70% split's name, or to None to use every non-evaluated vanilla prompt (and say so in the write-up).")
assert not (set(fit_use) & eval_prompts)

# ------------------------------------------------------------------ 3. r_hat and delta_p
def fit_direction(harm, benign):
    harm, benign = np.asarray(harm, dtype="float64"), np.asarray(benign, dtype="float64")
    diff = harm.mean(axis=0) - benign.mean(axis=0)
    n = float(np.linalg.norm(diff))
    if not np.isfinite(n) or n == 0:
        raise SystemExit("difference of means is zero or not finite")
    return diff / n, n


def direction_from(fit):
    harm = [pre[row_of[k]] for k, i in fit.items() if i == "harmful"]
    ben = [pre[row_of[k]] for k, i in fit.items() if i == "benign"]
    return fit_direction(harm, ben) + (len(harm), len(ben))


r_hat, raw_norm, nh, nb = direction_from(fit_use)
r_any, _, _, _ = direction_from(fit_any)
van_intent_eval = {r.vkey: r.intent == "harmful" for r in ev.itertuples()}
heldout = {"n_vanilla_prompts": len(van_intent_eval),
           **boot_auroc([float(pre[row_of[k]].astype("float64") @ r_hat) for k in van_intent_eval], list(van_intent_eval.values()))}
direction_info = {"n_harmful": nh, "n_benign": nb, "mean_difference_norm": raw_norm,
                  "cosine_fit_vs_any_split": float(r_hat @ r_any), "heldout_vanilla_auroc_harmful": heldout}
print(f"r_hat fitted: |mean diff| {raw_norm:.2f}   held-out AUROC (does r_hat separate the evaluated vanilla prompts by intent?) "
      f"{heldout['auroc']:.3f} [{heldout['lo']:.3f}, {heldout['hi']:.3f}]")

# ------------------------------------------------------------------ 4. final norm + lm_head (cached after the first download)
def load_head():
    if Path(HEAD_CACHE).is_file():
        z = np.load(HEAD_CACHE)
        print(f"lm_head from cache {HEAD_CACHE}")
        return z["norm_w"], z["head_w"], float(z["eps"]), {"source": "cache"}
    if not set_hf_token():
        raise SystemExit("No Hugging Face token. In Colab: key icon -> Secrets -> add HF_TOKEN (read access, and accept the Mistral-7B-Instruct-v0.3 terms on its HF page).")
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    idx_path = Path(hf_hub_download(MODEL_ID, "model.safetensors.index.json"))
    cfg = json.loads(Path(hf_hub_download(MODEL_ID, "config.json")).read_text())
    wmap = json.loads(idx_path.read_text())["weight_map"]
    head_name = "model.embed_tokens.weight" if (cfg.get("tie_word_embeddings") or "lm_head.weight" not in wmap) else "lm_head.weight"
    if cfg.get("final_logit_softcapping"):
        raise SystemExit("this checkpoint uses final-logit soft-capping; not handled")
    got = {}
    for name in ("model.norm.weight", head_name):
        path = hf_hub_download(MODEL_ID, wmap[name])
        with safe_open(path, framework="pt") as fh:
            got[name] = fh.get_tensor(name).float().numpy()
    np.savez(HEAD_CACHE, norm_w=got["model.norm.weight"], head_w=got[head_name], eps=np.float64(cfg["rms_norm_eps"]))
    return got["model.norm.weight"], got[head_name], float(cfg["rms_norm_eps"]), {"source": "hf", "head_tensor": head_name, "revision": idx_path.parent.name}


norm_w, head_w, eps, head_info = load_head()
head_info.update({"rms_norm_eps": eps, "vocab": int(head_w.shape[0])})
if head_w.shape[1] != HIDDEN:
    raise SystemExit(f"lm_head width {head_w.shape[1]}, expected {HIDDEN}")

if TOKENIZER is None:
    set_hf_token()
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
tok = TOKENIZER


def word_ids(w):
    ids = set()
    e = tok.encode(w, add_special_tokens=False)
    if len(e) == 1:
        ids.add(int(e[0]))
    for t in (w, "▁" + w):                      # bare and sentencepiece-space forms
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id:
            ids.add(int(i))
    return ids


resolved, set_ids = {}, {}
for name, words in (("refusal", REFUSAL_WORDS), ("compliance", COMPLIANCE_WORDS), ("refusal_ext", REFUSAL_WORDS_EXT), ("compliance_ext", COMPLIANCE_WORDS_EXT)):
    per = {w: sorted(word_ids(w)) for w in words}
    resolved[name] = per
    set_ids[name] = sorted({i for v in per.values() for i in v})
    if name in ("refusal", "compliance"):
        missing = [w for w, v in per.items() if not v]
        if missing:
            raise SystemExit(f"primary {name} words with no single-token form in the Mistral vocabulary: {missing}")
print("token sets (word -> ids -> how the tokenizer writes them):")
for name in ("refusal", "compliance"):
    for w, ids in resolved[name].items():
        print(f"   {name:<10} {w:<8} {ids}  {[tok.convert_ids_to_tokens(i) for i in ids]}")

order = sorted(eval_prompts, key=lambda k: row_of[k])
rows_idx = np.array([row_of[k] for k in order])
stats = first_token_stats(pre[rows_idx], norm_w, head_w, eps, set_ids)
tokstat = {k: {c: stats[c][i] for c in stats} for i, k in enumerate(order)}

# QA 1: is norm(prenorm) the stored postnorm? (checks that the two stored files agree with the head we loaded)
Hh = pre[rows_idx[:2000]].astype("float32")
normed = Hh / np.sqrt(np.mean(Hh * Hh, axis=-1, keepdims=True) + eps) * norm_w.astype("float32")
Pp = post[rows_idx[:2000]].astype("float32")
cosq = np.einsum("ij,ij->i", normed, Pp) / (np.linalg.norm(normed, axis=1) * np.linalg.norm(Pp, axis=1))
qa = {"postnorm_vs_norm_of_prenorm_cosine_min": float(np.nanmin(cosq)), "postnorm_vs_norm_of_prenorm_cosine_median": float(np.nanmedian(cosq))}
print(f"QA  stored postnorm vs final-norm(prenorm): cosine median {qa['postnorm_vs_norm_of_prenorm_cosine_median']:.5f}, min {qa['postnorm_vs_norm_of_prenorm_cosine_min']:.5f}")

# QA 2: offline argmax vs the real reply's first token (replies came from vLLM, activations from HF: expect high, not perfect)
qa["argmax_matches_reply_first_token"] = None
if replies is not None:
    rk = pick(replies, ["prompt_key", "key"], "reply_key", required=False, regex=r"key")
    rt = pick(replies, ["reply", "reply_text", "response", "completion", "output", "text"], "reply_text", required=False)
    if rk and rt:
        first = {}
        for k, txt in zip(replies[rk].astype(str), replies[rt]):
            if k in tokstat and isinstance(txt, str) and txt.strip():
                ids = tok.encode(txt, add_special_tokens=False)
                if ids:
                    first[k] = int(ids[0])
        if first:
            qa["n_prompts_with_reply"] = len(first)
            qa["argmax_matches_reply_first_token"] = float(np.mean([tokstat[k]["argmax"] == t for k, t in first.items()]))
            print(f"QA  offline argmax = the real reply's first token on {qa['argmax_matches_reply_first_token']:.1%} of {len(first):,} prompts")
    else:
        print(f"QA  replies.parquet columns {list(replies.columns)}: could not find key/text columns, skipping the first-token check "
              "(set OVERRIDES['reply_key'] / ['reply_text'] to run it)")

# ------------------------------------------------------------------ 5. per-pair table
rows = []
for r in ev.itertuples():
    v, a = r.vkey, r.akey
    o = r.out
    rows.append({"pair_id": r.pair_id, "source": r.source, "tactic": r.tactic, "split": r.split, "holdout_group": r.holdout_group,
                 "intent": r.intent, "not_understood": r.not_understood, "vanilla_prompt_key": v, "adversarial_prompt_key": a,
                 "delta_p": float((pre[row_of[a]].astype("float64") - pre[row_of[v]].astype("float64")) @ r_hat),
                 "refusal_mass_adv": tokstat[a]["refusal"], "compliance_mass_adv": tokstat[a]["compliance"],
                 "refusal_mass_ext_adv": tokstat[a]["refusal_ext"], "compliance_mass_ext_adv": tokstat[a]["compliance_ext"],
                 "refusal_mass_van": tokstat[v]["refusal"], "top1_prob_adv": tokstat[a]["top1_prob"],
                 "success_wg_harmful": o["harm"], "success_wg_complied": o["complied"], "reply_truncated": o["trunc"]})
df = pd.DataFrame(rows)
for c in ("success_wg_harmful", "success_wg_complied", "reply_truncated"):
    df[c] = df[c].astype("object")
harm = df[(df["intent"] == "harmful") & (~df["not_understood"])]
nu = df[(df["intent"] == "harmful") & (df["not_understood"])]
ben = df[df["intent"] == "benign"]
print(f"\nharmful pairs analysed {len(harm):,} (not-understood harmful {len(nu)}, benign {len(ben)})")

# ------------------------------------------------------------------ 6. the A4 comparison
print("\nbootstrapping (2,000 resamples per interval; this is the slow part, 2-5 min) ...", flush=True)
results = {"wg_harmful": {
    "last_prefill_L31": summarise_outcome(harm, "success_wg_harmful", "delta_p", "refusal_mass_adv", "compliance_mass_adv"),
    "last_prefill_L31_extended_tokens": summarise_outcome(harm, "success_wg_harmful", "delta_p", "refusal_mass_ext_adv", "compliance_mass_ext_adv")}}
if harm["success_wg_complied"].notna().any():
    results["wg_complied"] = {"last_prefill_L31": summarise_outcome(harm, "success_wg_complied", "delta_p", "refusal_mass_adv", "compliance_mass_adv")}
by_source = {}
for src, part in harm.groupby("source", dropna=False):
    if part["success_wg_harmful"].notna().sum() >= 20:
        by_source[str(src)] = summarise_outcome(part, "success_wg_harmful", "delta_p", "refusal_mass_adv", "compliance_mass_adv")
nt = harm[harm["reply_truncated"].map(lambda v: v is False)]
sens_not_truncated = (summarise_outcome(nt, "success_wg_harmful", "delta_p", "refusal_mass_adv", "compliance_mass_adv")
                      if len(nt) >= 20 else {"n_pairs": int(len(nt)), "note": "too few replies shorter than the 256-token cap to analyse"})
primary = results["wg_harmful"]["last_prefill_L31"]

warnings = []
if primary["n_succeeded"] < 20:
    warnings.append(f"only {primary['n_succeeded']} successful jailbreaks under the primary definition: report the interval widths and this count")
if qa["argmax_matches_reply_first_token"] is not None and qa["argmax_matches_reply_first_token"] < 0.85:
    warnings.append("offline argmax matches the replies' first token on fewer than 85% of prompts. Replies came from vLLM and activations from "
                    "HF, so some gap is expected, but check the refusal-mass computation before relying on it")
if qa["postnorm_vs_norm_of_prenorm_cosine_min"] < 0.99:
    warnings.append("stored postnorm differs from final-norm(prenorm) for some prompts (cosine < 0.99): the lm_head/norm loaded may not match the harvest")
warnings.append("replies and WildGuard labels come from a vLLM bf16 run; the activations come from an HF bf16 batch-1 run (different backend). "
                "About 85% of replies hit the 256-token cap; see 'sensitivity_not_truncated' for the replies that did not.")

summary = {
    "created_at": datetime.now(timezone.utc).isoformat(), "analysis": "A4 delta_p and first-token refusal mass vs jailbreak success",
    "owner": "WS2 (Kayson)", "model_id": MODEL_ID, "harvest": FULL_NAME,
    "primary": {"outcome": "WildGuard harmful response on the adversarial reply (judge status ok only)", "position": f"prompt[-1], layer {LAST_LAYER} (prenorm_last)",
                "refusal_tokens": REFUSAL_WORDS, "compliance_tokens": COMPLIANCE_WORDS,
                "direction": (f"vanilla-only difference of means, split '{FIT}', disjoint from every evaluated prompt" if FIT is not None else
                              "DEVIATION: vanilla-only difference of means over ALL non-evaluated vanilla prompts (any split)"),
                "label_columns": {"key": lk, "status": sc, "harmful_response": hc, "refusal": rc, "truncated": tc}},
    "head": head_info, "token_sets": resolved,
    "selection": {"n_eval_pairs": int(len(ev)), "n_harmful_pairs_analysed": int(len(harm)), "n_not_understood_harmful": int(len(nu)),
                  "n_benign_pairs": int(len(ben)), "fit_harmful": n_h, "fit_benign": n_b, "fit_conflicts_dropped": conflicts,
                  "split_values": split_values,
                  "pairs_without_valid_label_dropped": int((~p["labelled"]).sum())},
    "direction": direction_info, "qa": qa, "warnings": warnings, "results": results, "by_source": by_source,
    "sensitivity_not_truncated": sens_not_truncated,
    "not_understood_harmful_pairs": {"n": int(len(nu)), "delta_p": boot_median(nu["delta_p"]), "refusal_mass_adv": boot_median(nu["refusal_mass_adv"])},
    "benign_pairs_descriptive": {"n": int(len(ben)), "delta_p": boot_median(ben["delta_p"]), "refusal_mass_adv": boot_median(ben["refusal_mass_adv"]),
                                 "refusal_mass_van": boot_median(ben["refusal_mass_van"]),
                                 "note": "Mistral refused 0 of 296 benign adversarial prompts, so a benign-control comparison (H2) is not possible"},
    "not_replicable_here": ["last_instruction position (not saved in this harvest)", "A6 layer sweep (see M3/M4)"],
    "statistics": {"estimator": "median", "interval": "percentile bootstrap over pairs", "n_boot": N_BOOT, "alpha": ALPHA, "seed": SEED,
                   "logistic": "IRLS, predictors standardised on the full sample, ridge 1e-6, coefficients are log-odds per SD"},
    "runtime_seconds": round(time.time() - t0, 1)}

tag = MODEL_ALIAS
df.to_csv(RESULTS_DIR / f"a4_per_pair_{tag}.csv", index=False)
np.savez(RESULTS_DIR / f"a4_directions_{tag}.npz", fit_split=r_hat, any_split=r_any)
plot_f4(harm, RESULTS_DIR / f"a4_f4_scatter_{tag}.png", "success_wg_harmful", "delta_p", "refusal_mass_adv",
        "A4 / F4: Mistral-7B-Instruct-v0.3, harmful pairs, WildGuard outcome")
write_atomic(RESULTS_DIR / f"a4_summary_{tag}.json", json.dumps(summary, indent=2, default=str))

# ------------------------------------------------------------------ 7. read-out
def line(name, v):
    return f"{name:<44} {v['median']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" if v.get("median") is not None else f"{name:<44} n/a"


g = primary["groups"]
print("\n" + "=" * 72 + "\nA4 READ-OUT (harmful pairs, WildGuard harmful-response outcome)\n" + "=" * 72)
print(f"pairs {primary['n_pairs']:,}   jailbreak succeeded {primary['n_succeeded']:,} ({primary['success_rate']:.1%})   failed {primary['n_failed']:,}")
for grp in ("succeeded", "failed"):
    print(f"-- {grp} (n={g[grp]['n_pairs']})")
    print("   " + line("delta_p, median", g[grp]["delta_p"]))
    print("   " + line("first-token refusal mass, median", g[grp]["refusal_mass"]))
d = primary["median_difference_succeeded_minus_failed"]
print(f"difference succeeded - failed: delta_p {d['delta_p']['diff']:.3f} [{d['delta_p']['lo']:.3f}, {d['delta_p']['hi']:.3f}]   "
      f"refusal mass {d['refusal_mass']['diff']:.3f} [{d['refusal_mass']['lo']:.3f}, {d['refusal_mass']['hi']:.3f}]")
for k, v in primary["auroc_for_success"].items():
    if v["auroc"] is not None:
        print(f"AUROC {k}: {v['auroc']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
print("(AUROC 0.5 = no signal. Below 0.5 means LOWER values go with a successful jailbreak; for delta_p that is the expected direction, "
      "because a working jailbreak pushes the state away from 'harmful'.)")
for w in warnings:
    print("WARNING:", w)
print(f"\nwrote a4_per_pair_{tag}.csv, a4_summary_{tag}.json, a4_directions_{tag}.npz, a4_f4_scatter_{tag}.png to {RESULTS_DIR}   "
      f"({(time.time() - t0) / 60:.1f} min)")
