"""WS3 geometry: the maths behind A2, A5, A7, A8 and X5 (work plan s.7.1 and s.7.3). Numbers only, never text.

Shapes used everywhere:
  Hv, Ha   (pairs, hidden)  hidden state of the vanilla and the adversarial prompt of each pair, at ONE layer/position
  delta    (pairs, hidden)  Ha - Hv, how far the rewrite moved the pair's hidden state
  meta     DataFrame, one row per pair, columns intent ("harmful"/"benign"), tactic, split ("train"/...)
Fitting always uses TRAIN pairs only; numbers are reported on held-out (validation + test) pairs.
"""
import numpy as np
import pandas as pd

from analysis import lib


# ------------------------------------------------------------------------------------------------ A2
def a2_projection(Hv, Ha, meta, n_boot=2000, seed=0):
    """A2: the harmfulness direction r_hat and how far each rewrite moves along it.

    In words: average the vanilla harmful prompts and the vanilla benign prompts (TRAIN only); the arrow from the
    benign average to the harmful average, scaled to length 1, is r_hat ("which way is more harmful"). Each pair's
    dp is the shadow of its displacement on that arrow: dp = (h_adv - h_van) . r_hat. dp < 0 means the rewrite made
    the prompt look LESS harmful to the model. The midpoint between the two TRAIN averages (projected on r_hat) is
    the H1 line.

    Returns dict: r_hat, pv, pa (projections of every pair), dp, midpoint, overall {intent: (mean, lo, hi)} and
    per_tactic {tactic: {intent: (mean, lo, hi, n)}} on held-out pairs (95% bootstrap CI over pairs).
    """
    tr = (meta["split"] == "train").to_numpy()
    ev = ~tr
    tmeta = pd.DataFrame({"intent": meta["intent"][tr].to_numpy(), "split": "train", "role": "vanilla"})
    r_hat = lib.fit_direction_train(np.asarray(Hv)[tr], tmeta)
    pv, pa = lib.project(Hv, r_hat), lib.project(Ha, r_hat)
    dp = pa - pv
    harm = (meta["intent"] == "harmful").to_numpy()
    mid = (pv[tr & harm].mean() + pv[tr & ~harm].mean()) / 2
    out = dict(r_hat=r_hat, pv=pv, pa=pa, dp=dp, midpoint=float(mid), overall={}, per_tactic={})
    for intent, m in (("harmful", harm), ("benign", ~harm)):
        v = dp[ev & m]
        out["overall"][intent] = lib.bootstrap_ci(v, np.mean, n=n_boot, seed=seed) if len(v) else (np.nan,) * 3
    tac = meta["tactic"].astype(str).to_numpy()
    for t in sorted(set(tac[ev])):
        row = {}
        for intent, m in (("harmful", harm), ("benign", ~harm)):
            v = dp[ev & m & (tac == t)]
            row[intent] = (*lib.bootstrap_ci(v, np.mean, n=n_boot, seed=seed), len(v)) if len(v) >= 2 else (np.nan, np.nan, np.nan, len(v))
        out["per_tactic"][t] = row
    return out


# ------------------------------------------------------------------------------------------------ A5
def mean_displacements(delta, tactics, mask=None):
    """A5: the average displacement of each tactic, {tactic: vector}.

    In words: for every tactic, average the arrows (h_adv - h_van) of its pairs. If two tactics move the hidden state
    the same way, their average arrows point the same way. ``mask`` keeps only some pairs (e.g. harmful ones)."""
    delta, tactics = np.asarray(delta, float), np.asarray(tactics).astype(str)
    mask = np.ones(len(delta), bool) if mask is None else np.asarray(mask, bool)
    return {t: delta[mask & (tactics == t)].mean(0) for t in sorted(set(tactics[mask]))}


def cosine_matrix(vectors):
    """Cosine similarity between every pair of vectors (rows). 1 = same direction, 0 = unrelated (at right angles),
    -1 = opposite."""
    V = np.asarray(vectors, float)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return np.clip(V @ V.T, -1, 1)


def cluster_order(C):
    """Order of rows so that similar tactics sit next to each other (average-linkage clustering on 1 - cosine)."""
    n = len(C)
    if n < 3:
        return list(range(n))
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        D = np.clip(1 - np.asarray(C), 0, 2)
        np.fill_diagonal(D, 0)
        return list(leaves_list(linkage(squareform(D, checks=False), method="average")))
    except ImportError:  # no scipy: keep the given order
        return list(range(n))


def pca_2d(vectors):
    """Flatten many-dimensional vectors to 2-D for a picture: the two directions along which the vectors differ most
    (principal components). Returns (coords (n, 2), share of variance explained by each of the two)."""
    V = np.asarray(vectors, float)
    Vc = V - V.mean(0)
    _, s, Wt = np.linalg.svd(Vc, full_matrices=False)
    k = min(2, Wt.shape[0])
    coords = np.zeros((len(V), 2))
    coords[:, :k] = Vc @ Wt[:k].T
    var = s ** 2
    share = var[:2] / var.sum() if var.sum() > 0 else np.zeros(2)
    return coords, np.pad(share, (0, 2 - len(share)))


def tactic_subspace(delta_t, k=1):
    """The k directions that best describe one tactic's displacements (top-k right singular vectors, NOT
    mean-centred so the tactic's average direction is included). Returns an orthonormal (hidden, k) basis."""
    D = np.asarray(delta_t, float)
    _, _, Wt = np.linalg.svd(D, full_matrices=False)
    return Wt[:k].T


def principal_angles(U, V):
    """Principal angles (degrees, smallest first) between two subspaces with orthonormal bases U, V (hidden, k).

    In words: the smallest angle you can find between any arrow in subspace U and any arrow in subspace V, then the
    next smallest at right angles to those, and so on. 0 degrees = they share a direction; 90 = completely unrelated.
    Computed as arccos of the singular values of U^T V."""
    s = np.linalg.svd(np.asarray(U).T @ np.asarray(V), compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def knn_composition(X, labels, k=10):
    """For each row, the tactics of its k nearest other rows (cosine similarity). Returns a DataFrame
    (query tactic x neighbour tactic) of fractions; each row sums to 1.

    In words: for every adversarial harmful prompt, look at its 10 closest neighbours; if most share its tactic,
    tactics form separate clusters; if they are mixed, tactics overlap."""
    X = np.asarray(X, float)
    labels = np.asarray(labels).astype(str)
    S = cosine_matrix(X)
    np.fill_diagonal(S, -np.inf)  # never count yourself
    k = min(k, len(X) - 1)
    nn = np.argsort(-S, axis=1)[:, :k]
    tacs = sorted(set(labels))
    out = pd.DataFrame(0.0, index=tacs, columns=tacs)
    for i, row in enumerate(nn):
        for j in row:
            out.loc[labels[i], labels[j]] += 1
    return out.div(out.sum(axis=1), axis=0)


# ------------------------------------------------------------------------------------------------ A7
def style_content(dp, tactics, intents, n_boot=2000, seed=0):
    """A7 / H3: split each tactic's move along r_hat into a STYLE part and a CONTENT part.

    In words: benign prompts rewritten in tactic t have no harmful content, so whatever they move along r_hat is due
    to the jailbreak STYLE alone: style(t) = mean dp of benign pairs of t = d_benign(t) . r_hat. Harmful pairs move
    by style + content, so content(t) = mean dp(harmful, t) - mean dp(benign, t) = (d_harmful(t) - d_benign(t)) . r_hat.
    (Projection is linear, so averaging dp is the same as projecting the average displacement.)
    95% CIs: bootstrap that resamples harmful and benign pairs separately within the tactic.

    Returns {tactic: dict(style=(est, lo, hi), content=(est, lo, hi), n_harmful, n_benign)} for tactics with at least
    2 pairs of each intent, plus "_pooled" = the same averaged over those tactics (resampling within each tactic).
    """
    dp = np.asarray(dp, float)
    tac = np.asarray(tactics).astype(str)
    harm = np.asarray(intents) == "harmful"
    rng = np.random.default_rng(seed)
    out, boots_s, boots_c, ests = {}, [], [], []
    for t in sorted(set(tac)):
        h, b = dp[(tac == t) & harm], dp[(tac == t) & ~harm]
        if len(h) < 2 or len(b) < 2:
            continue
        bh = h[rng.integers(0, len(h), (n_boot, len(h)))].mean(1)
        bb = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
        s_b, c_b = bb, bh - bb
        s, c = b.mean(), h.mean() - b.mean()
        out[t] = dict(style=(float(s), *np.quantile(s_b, [0.025, 0.975]).tolist()),
                      content=(float(c), *np.quantile(c_b, [0.025, 0.975]).tolist()), n_harmful=len(h), n_benign=len(b))
        boots_s.append(s_b), boots_c.append(c_b), ests.append((s, c))
    if ests:
        S, Cc = np.mean(boots_s, axis=0), np.mean(boots_c, axis=0)
        e = np.mean(ests, axis=0)
        out["_pooled"] = dict(style=(float(e[0]), *np.quantile(S, [0.025, 0.975]).tolist()),
                              content=(float(e[1]), *np.quantile(Cc, [0.025, 0.975]).tolist()),
                              n_harmful=int(sum(v["n_harmful"] for v in out.values())),
                              n_benign=int(sum(v["n_benign"] for v in out.values())))
    return out


# ------------------------------------------------------------------------------------------------ A8
def singular_values(delta, center=False):
    """A8: singular values of the stacked displacements (one row per pair), largest first. Each one says how much
    of the movement happens along one direction; a fast drop means the rewrites move the state in only a few ways."""
    D = np.asarray(delta, float)
    if center:
        D = D - D.mean(0)
    return np.linalg.svd(D, compute_uv=False)


def effective_rank(s):
    """Effective rank (Roy and Vetterli, 2007): exp(entropy of the singular values scaled to sum to 1).

    In words: how many directions the movement really uses. k equal singular values give exactly k; one dominant
    direction gives close to 1."""
    s = np.asarray(s, float)
    s = s[s > 0]
    if not len(s):
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def energy_rank(s, share=0.9):
    """How many directions (largest first) hold ``share`` of the movement (sum of squared singular values).
    Easier to read than the entropy version because tiny noise directions barely change it."""
    e = np.asarray(s, float) ** 2
    if e.sum() == 0:
        return 0
    return int(np.searchsorted(np.cumsum(e) / e.sum(), share - 1e-12) + 1)


def refusal_direction(H, refused, train_mask=None):
    """Refusal direction: (mean hidden state of prompts the model refused - mean of prompts it answered), length 1,
    on TRAIN rows. ``refused`` comes from the judge labels (WildGuard refusal)."""
    H = np.asarray(H, float)
    refused = np.asarray(refused, bool)
    m = np.ones(len(H), bool) if train_mask is None else np.asarray(train_mask, bool)
    return lib.fit_direction(H[m & refused], H[m & ~refused])


def angle_between(u, v):
    """Angle in degrees between two directions (0 = same, 90 = unrelated, 180 = opposite)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# ------------------------------------------------------------------------------------------------ X5
def direction_stability(by_config, meta, reference, n_boot=2000, seed=0):
    """X5: refit r_hat under each serving config and compare it with the reference config R.

    by_config: {config: (Hv, Ha)} for the SAME pairs (rows aligned with meta). For each config returns
    cos_to_reference (1 = the harmfulness direction did not move) and the held-out mean dp of harmful and benign pairs
    with 95% CIs, so we can see whether the masking effect (H1) keeps its size on, e.g., a 4-bit model."""
    out, ref_r = {}, None
    for cfg in [reference] + [c for c in by_config if c != reference]:
        Hv, Ha = by_config[cfg]
        a = a2_projection(Hv, Ha, meta, n_boot=n_boot, seed=seed)
        if cfg == reference:
            ref_r = a["r_hat"]
        out[cfg] = dict(cos_to_reference=float(a["r_hat"] @ ref_r), dp=a["overall"], r_hat=a["r_hat"])
    return out


# ------------------------------------------------------------------------------------------------ synthetic data
def synthetic_ws3(n_pairs=240, d=48, tactics=("roleplay", "persona", "obfuscation", "hypothetical"), seed=0,
                  style=-0.8, content=-1.2, configs=None):
    """SYNTHETIC pairs with a KNOWN planted answer (for tests and chart previews; never results).

    Planted: r direction u; every tactic t has its own unit direction w_t; benign pairs move style*u + w_t, harmful
    pairs move (style+content)*u + w_t (+ small noise). So A7 should find style ~ style, content ~ content; A5 should
    find each tactic's displacement mostly along w_t. configs: {name: rotation angle in degrees} gives X5 copies whose
    r direction is rotated by that angle."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    W = {}
    for t in tactics:
        w = rng.normal(size=d)
        w -= (w @ u) * u
        W[t] = w / np.linalg.norm(w)
    rows, Hv, Ha = [], [], []
    for i in range(n_pairs):
        intent = "harmful" if i % 2 == 0 else "benign"
        t = tactics[(i // 2) % len(tactics)]
        pid = f"syn{i:05d}"
        base = rng.normal(size=d) + (2.0 if intent == "harmful" else -2.0) * u
        shift = (style + (content if intent == "harmful" else 0.0)) * u + 1.5 * W[t] + 0.15 * rng.normal(size=d)
        rows.append(dict(pair_id=pid, intent=intent, tactic=t, split=lib.assign_split(pid)))
        Hv.append(base)
        Ha.append(base + shift)
    meta = pd.DataFrame(rows)
    out = dict(Hv=np.array(Hv), Ha=np.array(Ha), meta=meta, u=u, W=W, style=style, content=content)
    if configs:
        by = {}
        for name, deg in configs.items():
            # rotate the u-component of every vector towards a fixed orthogonal direction by `deg` degrees
            z = rng.normal(size=d)
            z -= (z @ u) * u
            z /= np.linalg.norm(z)
            th = np.radians(deg)
            R = np.eye(d) + (np.cos(th) - 1) * (np.outer(u, u) + np.outer(z, z)) + np.sin(th) * (np.outer(z, u) - np.outer(u, z))
            by[name] = (out["Hv"] @ R.T, out["Ha"] @ R.T)
        out["by_config"] = by
    return out
