"""Kill-test for Idea 2: is the meta-gradient sign-agreement excess ARCHITECTURAL?

Idea 2 claims we can cluster parameters by measured meta-gradient sign-agreement and
that the clusters will CROSS architectural boundaries (which is what separates it from
SGG, arXiv:2506.01049, which clusters by momentum WITHIN a layer).  If the whole excess
over 50% is explained by "coordinates in the same tensor / same block agree", then the
existing layerwise and 6-block partitions already capture all of it and Idea 2 is dead.

WHAT THE PROBE ACTUALLY STORES (HF.py::_probe, lines 344-403):
  frac_neg / frac_zero : GLOBAL scalars over all n_tot coordinates of the arm.  Per-record.
  z_mean / z_std / snr : length-62 arrays -- one entry per PARAM TENSOR, always, on every
                         arm.  z_mean is the RUNNING (cumulative-over-steps) mean of the
                         per-tensor value; z_std is its TEMPORAL std, not a spatial std.
                         On scalar/layerwise/blockwise arms the per-tensor value is the
                         exact per-group meta-gradient; on nodewise/weightwise arms it is
                         the SPATIAL MEAN of that tensor's per-coordinate meta-gradients.
  beta                 : length-62 array, same convention, but NOT a running mean -- it is
                         the instantaneous per-group (or per-tensor-mean) log-stepsize.

CONSEQUENCE: per-COORDINATE signs are never written, and neither is a per-tensor sign
count.  Within-tensor sign-agreement among the weights of one tensor is NOT recoverable.
Across-tensor agreement IS recoverable, exactly, because each tensor contributes one
number per record.  See the report for the minimal probe change that would close the gap.

STATISTIC.  For groups i != j we use the PAIRWISE same-sign rate

    A_ij = P( sign z_i(t) == sign z_j(t) )

whose null under coordinate independence is EXACTLY 0.5 with no finite-sample bias.  The
published "53.26%" statistic is instead the cross-sectional majority fraction
max(p_r, 1-p_r), which is biased UP by 0.5*sqrt(2/pi)/sqrt(n) under the null (6.4pp at
n=62, 20pp at n=6) and is therefore useless for a within-vs-across comparison where the
two sides have different group counts.  We report both; we CONCLUDE on the pairwise one.
"""
import json, os, sys, math
import numpy as np

RESNET18_BLOCKS = [3, 12, 15, 15, 15, 2]      # HF.py::polish_the_stepsize_groups
SQ2PI = math.sqrt(2.0 / math.pi)
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------- loading
def load(d):
    p = os.path.join(d, "probe.jsonl")
    recs = []
    for l in open(p):
        l = l.strip()
        if l:
            try:
                recs.append(json.loads(l))
            except Exception:
                pass
    with open(os.path.join(d, "block_sizes.json")) as fh:
        bs = json.load(fh)
    return recs, bs


def block_labels():
    lab = []
    for b, k in enumerate(RESNET18_BLOCKS):
        lab += [b] * k
    return np.array(lab)


# --------------------------------------------- per-interval reconstruction
def interval_z(recs, key="z_mean"):
    """Undo the running mean.  mean_k = S_k / n_k with n_k = step_k + 1 (the accumulator
    is bumped on EVERY optimizer step; records are written every PROBE steps).
    Returns Z[T-1, G]: the average per-group meta-gradient over each inter-record window.
    """
    steps = np.array([r["step"] for r in recs], dtype=np.float64)
    M = np.array([r[key] for r in recs], dtype=np.float64)
    n = (steps + 1.0)[:, None]
    S = M * n
    dS = np.diff(S, axis=0)
    dn = np.diff(n, axis=0)
    return dS / dn


def interval_beta(recs):
    """Delta beta between consecutive records: the REALISED meta-update direction.
    Exactly stored (no running-mean differencing), so it validates interval_z."""
    B = np.array([r["beta"] for r in recs], dtype=np.float64)
    return np.diff(B, axis=0)


# --------------------------------------------------------------- statistics
def pair_agreement(S):
    """S[T,G] of signs in {-1,0,+1}.  Returns A[G,G] = P(same sign | both nonzero)."""
    T, G = S.shape
    nz = (S != 0).astype(np.float64)
    same = (S.T @ S)                      # +1 per agreeing pair, -1 per disagreeing pair
    both = (nz.T @ nz)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = (both + same) / (2.0 * both)
    return A, both


def split_within_across(A, both, lab, min_n=1):
    """Size-unweighted mean of A_ij over same-label pairs and different-label pairs."""
    G = A.shape[0]
    iu = np.triu_indices(G, 1)
    a = A[iu]
    w = both[iu]
    same = (lab[iu[0]] == lab[iu[1]])
    ok = np.isfinite(a) & (w >= min_n)
    def m(mask):
        s = mask & ok
        return (float(np.mean(a[s])), int(s.sum())) if s.sum() else (float("nan"), 0)
    return m(same), m(~same), m(np.ones(G * (G - 1) // 2, dtype=bool))


def moving_block_boot(S, lab, n_boot=400, L=40):
    """Moving-block bootstrap over records -> CI on (within, across, all)."""
    T = S.shape[0]
    nb = max(1, T // L)
    out = []
    for _ in range(n_boot):
        starts = RNG.integers(0, max(1, T - L), size=nb)
        idx = np.concatenate([np.arange(s, s + L) for s in starts])
        idx = idx[idx < T]
        A, bo = pair_agreement(S[idx])
        (w, _), (x, _), (al, _) = split_within_across(A, bo, lab)
        out.append((w, x, al))
    o = np.array(out)
    return np.nanpercentile(o, [2.5, 97.5], axis=0)


def majority_stat(S):
    """The PUBLISHED statistic: per-record max(p_neg, p_pos) among nonzero, averaged;
    plus its independence floor 0.5 + sqrt(2/pi)/(2 sqrt(n))."""
    vals, ns = [], []
    for row in S:
        nz = row[row != 0]
        if nz.size < 2:
            continue
        p = float((nz < 0).mean())
        vals.append(max(p, 1 - p))
        ns.append(nz.size)
    if not vals:
        return float("nan"), float("nan")
    nbar = float(np.mean(ns))
    return float(np.mean(vals)), 0.5 + SQ2PI * 0.5 / math.sqrt(nbar)


# ---------------------------------------------------------------- reporting
def headline(dirs, label):
    """(d) reproduce the published per-arm agreement from the GLOBAL frac_neg field."""
    rows = []
    for d in dirs:
        recs, bs = load(d)
        w = recs[len(recs) // 2:]                       # steady window: last 50%
        fz = np.array([r["frac_zero"] for r in w])
        fn = np.array([r["frac_neg"] for r in w])
        keep = fz < 1.0
        p_step = fn[keep] / (1 - fz[keep])
        agree_step = float(np.mean(np.maximum(p_step, 1 - p_step)))
        p_sys = float(np.mean(fn[keep])) / (1 - float(np.mean(fz[keep])))
        agree_sys = max(p_sys, 1 - p_sys)
        # n_tot inferred from the rational denominator of frac_neg (block_sizes lies on nodewise)
        from fractions import Fraction
        nt = 1
        for v in list(fn[:200]) + list(fz[:200]):
            if 0 < v < 1:
                nt = max(nt, Fraction(v).limit_denominator(100_000_000).denominator)
        floor = 0.5 + SQ2PI * 0.5 / math.sqrt(max(nt * (1 - float(fz.mean())), 1))
        rows.append((os.path.basename(d), bs["stepsize_type"], nt,
                     100 * agree_sys, 100 * agree_step, 100 * floor))
    print(f"\n### {label}  (steady window = last 50% of records)")
    h = f"{'run':34s} {'type':11s} {'n_tot':>12} {'sys%':>9} {'step%':>9} {'null%':>9} {'step-null':>10}"
    print(h); print("-" * len(h))
    for r in rows:
        print(f"{r[0]:34s} {r[1]:11s} {r[2]:>12,} {r[3]:>9.4f} {r[4]:>9.4f} {r[5]:>9.4f} {r[4]-r[5]:>10.4f}")
    return rows


def analyse_arm(dirs, label, note=""):
    print(f"\n\n### {label}")
    if note:
        print(f"    {note}")
    lab = block_labels()
    agg = []
    for d in dirs:
        recs, bs = load(d)
        recs = recs[len(recs) // 2:]                    # steady window
        Z = interval_z(recs)
        Bd = interval_beta(recs)
        Sz = np.sign(Z)
        Sb = np.sign(Bd)
        # validation: do the two independent direction estimators agree?
        m = (Sz != 0) & (Sb != 0)
        conc = float((Sz[m] == Sb[m]).mean()) if m.sum() else float("nan")
        conc = max(conc, 1 - conc)                      # sign convention is global
        A, both = pair_agreement(Sz)
        (w, nw), (x, nx), (al, na) = split_within_across(A, both, lab)
        ci = moving_block_boot(Sz, lab)
        maj, floor = majority_stat(Sz)
        agg.append(dict(run=os.path.basename(d), T=Sz.shape[0], conc=conc,
                        w=w, x=x, al=al, ci=ci, maj=maj, floor=floor, A=A, S=Sz,
                        nb=np.array(bs["n_b"], dtype=float)))
    h = (f"{'run':30s} {'T':>5} {'z/beta':>7} {'ALL pairs%':>11} {'within-blk%':>12} "
         f"{'across-blk%':>12} {'across 95%CI':>16} {'majority%':>10} {'(null)':>8}")
    print(h); print("-" * len(h))
    for a in agg:
        lo, hi = a["ci"][0][1], a["ci"][1][1]
        print(f"{a['run']:30s} {a['T']:>5} {100*a['conc']:>6.1f}% {100*a['al']:>11.3f} "
              f"{100*a['w']:>12.3f} {100*a['x']:>12.3f} "
              f"{'['+format(100*lo,'.2f')+', '+format(100*hi,'.2f')+']':>16} "
              f"{100*a['maj']:>10.3f} {100*a['floor']:>8.3f}")
    W = np.array([a["w"] for a in agg]); X = np.array([a["x"] for a in agg]); AL = np.array([a["al"] for a in agg])
    print(f"{'MEAN +- sd over seeds':30s} {'':>5} {'':>7} "
          f"{100*AL.mean():>11.3f} {100*W.mean():>12.3f} {100*X.mean():>12.3f}")
    print(f"{'':30s} {'':>5} {'':>7} {'+-'+format(100*AL.std(ddof=1),'.3f'):>11} "
          f"{'+-'+format(100*W.std(ddof=1),'.3f'):>12} {'+-'+format(100*X.std(ddof=1),'.3f'):>12}")
    return agg


def block_size_analysis(agg, label):
    """(4) how does agreement vary with block SIZE?"""
    print(f"\n\n### {label} -- agreement vs block size")
    nb = agg[0]["nb"]
    G = len(nb)
    lab = block_labels()
    # per-tensor: mean pairwise agreement against every OTHER tensor, and temporal
    # sign persistence (fraction of intervals matching the tensor's own modal sign)
    rows = []
    for g in range(G):
        aij, pers = [], []
        for a in agg:
            A = a["A"]
            o = np.delete(A[g], g)
            aij.append(np.nanmean(o))
            s = a["S"][:, g]; s = s[s != 0]
            pers.append(max(float((s < 0).mean()), float((s > 0).mean())) if s.size else np.nan)
        rows.append((g, nb[g], lab[g], float(np.mean(aij)), float(np.mean(pers))))
    order = np.argsort(nb)
    print(f"{'n_b (params)':>14} {'ntensors':>9} {'mean pair-agree vs others%':>28} {'temporal persistence%':>23}")
    print("-" * 78)
    # decade bins
    lo = np.floor(np.log10(nb.min())); hi = np.ceil(np.log10(nb.max()))
    e = 10 ** np.arange(lo, hi + 1)
    for i in range(len(e) - 1):
        sel = [r for r in rows if e[i] <= r[1] < e[i + 1]]
        if not sel:
            continue
        print(f"{format(int(e[i]),',')+'-'+format(int(e[i+1]),','):>14} {len(sel):>9} "
              f"{100*np.mean([r[3] for r in sel]):>28.3f} {100*np.mean([r[4] for r in sel]):>23.3f}")
    # log-log style correlation
    x = np.log10(np.array([r[1] for r in rows]))
    y = np.array([r[3] for r in rows]) - 0.5
    z = np.array([r[4] for r in rows]) - 0.5
    print(f"\n  Pearson r(log10 n_b, pairwise-agreement excess) = {np.corrcoef(x, y)[0,1]:+.3f}  (n={len(x)} tensors)")
    print(f"  Pearson r(log10 n_b, temporal-persistence excess) = {np.corrcoef(x, z)[0,1]:+.3f}")
    sl = np.polyfit(x, y, 1)
    print(f"  OLS: pairwise excess = {100*sl[0]:+.3f}pp per decade of n_b  (intercept {100*sl[1]:+.3f}pp)")
    return rows


def cluster_test(agg, label):
    """Does the empirical correlation structure line up with the 6-block partition, or
    does it cross it?  Spectral-cluster the agreement matrix into k=6 and score against
    the architectural labels with the adjusted Rand index."""
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import adjusted_rand_score
    lab = block_labels()
    print(f"\n\n### {label} -- do agreement clusters respect architecture?")
    print(f"{'run':30s} {'ARI vs 6-block':>16} {'ARI vs contiguity':>19} {'best within/across gap':>24}")
    print("-" * 92)
    for a in agg:
        A = np.nan_to_num(a["A"], nan=0.5)
        S = np.abs(2 * A - 1)                      # affinity: |agreement excess|
        np.fill_diagonal(S, 1.0)
        try:
            cl = SpectralClustering(n_clusters=6, affinity="precomputed",
                                    random_state=0, assign_labels="kmeans").fit_predict(S)
        except Exception as e:
            print(f"{a['run']:30s}  spectral failed: {e}")
            continue
        ari = adjusted_rand_score(lab, cl)
        # contiguity null: a random CONTIGUOUS partition into 6 of the same sizes
        aris_c = []
        for _ in range(200):
            cuts = np.sort(RNG.choice(np.arange(1, 62), size=5, replace=False))
            lab2 = np.zeros(62, dtype=int)
            for j, c in enumerate(cuts):
                lab2[c:] = j + 1
            aris_c.append(adjusted_rand_score(lab2, cl))
        (w, _), (x, _), _ = split_within_across(a["A"], np.ones_like(a["A"]), cl)
        print(f"{a['run']:30s} {ari:>16.3f} {np.mean(aris_c):>19.3f} "
              f"{100*(w-x):>23.3f}pp")


def precision_check(d):
    """Differencing a float32 running mean loses precision.  Quantify it: recompute the
    interval signs at stride k (k x larger increments, k x less differencing noise) and
    report how stable the within/across split is."""
    recs, _ = load(d)
    recs = recs[len(recs) // 2:]
    lab = block_labels()
    print(f"\n  {os.path.basename(d)}")
    print(f"  {'stride':>7} {'T':>6} {'ALL%':>8} {'within%':>9} {'across%':>9}")
    for k in (1, 2, 5, 10, 25):
        sub = recs[::k]
        S = np.sign(interval_z(sub))
        A, bo = pair_agreement(S)
        (w, _), (x, _), (al, _) = split_within_across(A, bo, lab)
        print(f"  {k:>7} {S.shape[0]:>6} {100*al:>8.3f} {100*w:>9.3f} {100*x:>9.3f}")


def beta_replicate(dirs, label):
    """Independent estimator: sign of the REALISED meta-update (delta beta), which is
    stored exactly and needs no running-mean differencing."""
    lab = block_labels()
    print(f"\n\n### {label} -- same split from delta-beta (precision-exact, independent of z_mean)")
    h = f"{'run':30s} {'ALL pairs%':>11} {'within-blk%':>12} {'across-blk%':>12}"
    print(h); print("-" * len(h))
    out = []
    for d in dirs:
        recs, _ = load(d)
        recs = recs[len(recs) // 2:]
        S = np.sign(interval_beta(recs))
        A, bo = pair_agreement(S)
        (w, _), (x, _), (al, _) = split_within_across(A, bo, lab)
        out.append((w, x, al))
        print(f"{os.path.basename(d):30s} {100*al:>11.3f} {100*w:>12.3f} {100*x:>12.3f}")
    o = np.array(out)
    print(f"{'MEAN':30s} {100*o[:,2].mean():>11.3f} {100*o[:,0].mean():>12.3f} {100*o[:,1].mean():>12.3f}")
    return o


def permutation_test(agg, label, n_perm=2000):
    """Is the within-vs-across-block gap bigger than a RANDOM partition of the same
    block sizes would give?  This is the correct null for claim (c): it holds the group
    sizes fixed and destroys only the architectural assignment."""
    lab = block_labels()
    print(f"\n\n### {label} -- permutation null for the within/across-block gap")
    print(f"{'run':30s} {'observed gap':>13} {'perm mean':>11} {'perm 95th':>11} {'p':>8}")
    print("-" * 78)
    for a in agg:
        A, bo = a["A"], np.ones_like(a["A"])
        (w, _), (x, _), _ = split_within_across(A, bo, lab)
        obs = w - x
        gaps = []
        for _ in range(n_perm):
            p = RNG.permutation(lab)
            (w2, _), (x2, _), _ = split_within_across(A, bo, p)
            gaps.append(w2 - x2)
        g = np.array(gaps)
        pval = float((g >= obs).mean())
        print(f"{a['run']:30s} {100*obs:>12.3f}pp {100*g.mean():>10.3f}pp "
              f"{100*np.percentile(g,95):>10.3f}pp {pval:>8.4f}")


def aggregation_ladder(agg, label):
    """(4) THE block-size question, done inside ONE run so the optimizer is held fixed.
    The blockwise optimizer's group meta-gradient is the SUM of its tensors' (HF.py:149),
    so we can synthesise any coarser partition from the 62 tensor series.  Compare
    ARCHITECTURAL groupings against RANDOM groupings of identical sizes: if random does
    as well, the agreement gain from coarsening is pure averaging, not architecture."""
    lab = block_labels()
    print(f"\n\n### {label} -- agreement vs GROUP SIZE (architectural vs random, same sizes)")
    print(f"  Coarsening a partition raises agreement even under pure independence, because a")
    print(f"  group meta-gradient is the SUM of its members'.  The random control isolates that.")
    print(f"\n  {'tensors/group':>14} {'groups':>7} {'architectural%':>16} {'random (same sizes)%':>22} {'arch - random':>15}")
    print("  " + "-" * 80)
    for k in (1, 2, 4, 8, 16, 31):
        arch_v, rand_v = [], []
        for a in agg:
            Z = None
            # rebuild the raw interval-z (signs alone cannot be summed)
            recs, _ = load(f"analysis/killtest_data/mx/{a['run']}")
            Z = interval_z(recs[len(recs) // 2:])
            G = Z.shape[1]
            # architectural: contiguous runs of k tensors (respects depth ordering)
            grp = np.arange(G) // k
            arch_v.append(_agg_agree(Z, grp))
            # random: same group sizes, shuffled membership
            rv = []
            for _ in range(20):
                rv.append(_agg_agree(Z, RNG.permutation(grp)))
            rand_v.append(np.mean(rv))
        na = len(np.unique(np.arange(62) // k))
        print(f"  {k:>14} {na:>7} {100*np.mean(arch_v):>16.3f} {100*np.mean(rand_v):>22.3f} "
              f"{100*(np.mean(arch_v)-np.mean(rand_v)):>14.3f}pp")
    # and the true 6-block partition vs random partitions of exactly [3,12,15,15,15,2]
    arch_v, rand_v = [], []
    for a in agg:
        recs, _ = load(f"analysis/killtest_data/mx/{a['run']}")
        Z = interval_z(recs[len(recs) // 2:])
        arch_v.append(_agg_agree(Z, lab))
        rand_v.append(np.mean([_agg_agree(Z, RNG.permutation(lab)) for _ in range(40)]))
    print(f"\n  TRUE resnet18_blocks [3,12,15,15,15,2]: architectural {100*np.mean(arch_v):.3f}%  "
          f"vs random same-sizes {100*np.mean(rand_v):.3f}%  "
          f"-> {100*(np.mean(arch_v)-np.mean(rand_v)):+.3f}pp")


def _agg_agree(Z, grp):
    """Sum the tensor meta-gradients within each group, then mean pairwise same-sign rate."""
    ug = np.unique(grp)
    if len(ug) < 2:
        return float("nan")
    Y = np.stack([Z[:, grp == g].sum(axis=1) for g in ug], axis=1)
    S = np.sign(Y)
    A, bo = pair_agreement(S)
    iu = np.triu_indices(len(ug), 1)
    v = A[iu]
    return float(np.nanmean(v))


def circshift_null(S, lab, n_null=400):
    """THE null that matters.  Two coordinates that each have a persistent direction agree
    above 50% even when they are statistically INDEPENDENT: if coordinate i is negative a
    fraction p_i of the time, then A_ij = p_i p_j + q_i q_j > 0.5 with no dependence at all.
    Marginal bias like that is already captured by giving each tensor its own step size --
    it is not correlation structure and it cannot justify clustering.

    Circularly shifting each coordinate's sign series by an independent random offset
    destroys cross-coordinate dependence while preserving EXACTLY each coordinate's
    marginal bias and its own autocorrelation.  Anything above this null is real
    dependence.  Returns (null_within, null_across, null_all) distributions and the
    per-pair null threshold.
    """
    T, G = S.shape
    outs = []
    pair_null = []
    for _ in range(n_null):
        Sh = np.empty_like(S)
        for g in range(G):
            Sh[:, g] = np.roll(S[:, g], int(RNG.integers(0, T)))
        A, bo = pair_agreement(Sh)
        (w, _), (x, _), (al, _) = split_within_across(A, bo, lab)
        outs.append((w, x, al))
        iu = np.triu_indices(G, 1)
        pair_null.append(A[iu])
    return np.array(outs), np.array(pair_null)


def marginal_adjusted(agg, label, root="analysis/killtest_data/mx"):
    """Re-run the within/across split against the marginal-preserving circular-shift null."""
    lab = block_labels()
    print(f"\n\n### {label} -- against the MARGINAL-PRESERVING null (independent circular shifts)")
    print("  Excess over 50% conflates (i) each tensor having its own persistent direction --")
    print("  which layerwise step sizes already capture -- with (ii) genuine cross-tensor")
    print("  dependence.  Only (ii) can justify clustering.  This null removes (i).")
    print(f"\n{'run':28s} {'obs within%':>12} {'null within%':>13} {'d':>8} | "
          f"{'obs across%':>12} {'null across%':>13} {'d':>8} {'p(across)':>10}")
    print("-" * 116)
    res = []
    for a in agg:
        recs, _ = load(f"{root}/{a['run']}")
        S = np.sign(interval_z(recs[len(recs) // 2:]))
        A, bo = pair_agreement(S)
        (w, _), (x, _), (al, _) = split_within_across(A, bo, lab)
        nulls, pair_null = circshift_null(S, lab)
        nw, nx = nulls[:, 0].mean(), nulls[:, 1].mean()
        p = float((nulls[:, 1] >= x).mean())
        res.append((w - nw, x - nx, p))
        print(f"{a['run']:28s} {100*w:>12.3f} {100*nw:>13.3f} {100*(w-nw):>+8.3f} | "
              f"{100*x:>12.3f} {100*nx:>13.3f} {100*(x-nx):>+8.3f} {p:>10.4f}")
        # Per-pair counts.  A Bonferroni tail of 0.05/(2*1506) = 1.7e-5 cannot be read off
        # 400 draws as a percentile, so standardise each pair by the null's own mean and sd
        # (both well estimated at n=400) and threshold on z.  Then CALIBRATE: apply the
        # identical procedure to held-out null draws to get the actual false-positive count.
        from scipy.stats import norm
        G = S.shape[1]
        iu = np.triu_indices(G, 1)
        cross = lab[iu[0]] != lab[iu[1]]
        npairs = int(cross.sum())
        zc = norm.ppf(1 - 0.05 / (2 * npairs))
        half = len(pair_null) // 2
        mu = pair_null[:half].mean(axis=0)
        sd = pair_null[:half].std(axis=0, ddof=1) + 1e-12
        z_obs = (A[iu] - mu) / sd
        hi = int(((z_obs > zc) & cross).sum())
        low = int(((z_obs < -zc) & cross).sum())
        fp = []
        for k in range(half, len(pair_null)):
            zk = (pair_null[k] - mu) / sd
            fp.append((((zk > zc) | (zk < -zc)) & cross).sum())
        print(f"{'':28s}   cross-block pairs beyond Bonferroni |z|>{zc:.2f}: "
              f"{hi} above / {low} below of {npairs}   "
              f"[held-out null gives {np.mean(fp):.2f} +- {np.std(fp):.2f} false positives]")
    r = np.array(res)
    print(f"\n  MEAN over seeds: within {100*r[:,0].mean():+.3f}pp, "
          f"across {100*r[:,1].mean():+.3f}pp above the marginal-preserving null")
    return r


def eff_T(S):
    """Effective sample size of the sign series, from the mean lag-1 autocorrelation.
    T_eff = T * (1-rho)/(1+rho).  The per-pair null SE is 0.5/sqrt(T_eff), not 0.5/sqrt(T)."""
    T = S.shape[0]
    rs = []
    for g in range(S.shape[1]):
        s = S[:, g].astype(float)
        if s.std() < 1e-12:
            continue
        rs.append(float(np.corrcoef(s[:-1], s[1:])[0, 1]))
    rho = float(np.nanmedian(rs)) if rs else 0.0
    rho = min(max(rho, 0.0), 0.95)
    return T * (1 - rho) / (1 + rho), rho


def ulp_noise(d):
    """How much of the reconstructed interval-z is float32 differencing noise?
    S_k = mean_k * n_k is formed from a float32 mean, so its absolute error is
    ~eps32*|S_k|; the interval mean inherits eps32*|S_k|/dn.  Report the median ratio
    of that noise to the reconstructed signal."""
    recs, _ = load(d)
    recs = recs[len(recs) // 2:]
    steps = np.array([r["step"] for r in recs], float)
    M = np.array([r["z_mean"] for r in recs], float)
    n = (steps + 1.0)[:, None]
    S = M * n
    dn = np.diff(n, axis=0)
    sig = np.abs(np.diff(S, axis=0) / dn)
    noise = np.finfo(np.float32).eps * np.abs(S[1:]) * math.sqrt(2) / dn
    r = noise / (sig + 1e-300)
    return float(np.median(r)), float((r > 1).mean())


def common_mode_test(agg, label, root="analysis/killtest_data/mx"):
    """Is the residual across-block agreement ONE global common direction -- which the
    SCALAR partition already captures, so clustering would buy nothing -- or heterogeneous
    structure?  Two diagnostics:

      (1) rank-1 share: eigenvalue share of the leading eigenvector of the centred
          agreement-excess matrix (2A-1).  A single common mode is exactly rank 1.
      (2) significantly ANTI-correlated cross-block pairs.  No single common mode can
          produce a pair that agrees LESS than chance, so these are the only evidence
          that could keep Idea 2 alive.  Significance uses a moving-block bootstrap so
          the autocorrelation of the sign series is respected.
    """
    lab = block_labels()
    print(f"\n\n### {label} -- is the cross-block residual just ONE global common mode?")
    print(f"{'run':28s} {'T':>5} {'rho1':>6} {'T_eff':>6} {'rank1 share':>12} "
          f"{'x-blk pairs >50':>16} {'x-blk pairs <50':>16} {'expect/side':>12}")
    print("-" * 112)
    from scipy.stats import norm
    for a in agg:
        recs, _ = load(f"{root}/{a['run']}")
        Z = interval_z(recs[len(recs) // 2:])
        S = np.sign(Z)
        A, _ = pair_agreement(S)
        Te, rho = eff_T(S)
        E = np.nan_to_num(2 * A - 1.0, nan=0.0)
        np.fill_diagonal(E, 0.0)
        ev = np.linalg.eigvalsh((E + E.T) / 2)
        rank1 = float(np.max(np.abs(ev)) / np.sum(np.abs(ev)))
        iu = np.triu_indices(S.shape[1], 1)
        cross = lab[iu[0]] != lab[iu[1]]
        v = A[iu][cross]
        npairs = int(cross.sum())
        se = 0.5 / math.sqrt(Te)
        zc = norm.ppf(1 - 0.05 / (2 * npairs))          # Bonferroni, two-sided 5%
        hi = int((v > 0.5 + zc * se).sum())
        lo = int((v < 0.5 - zc * se).sum())
        print(f"{a['run']:28s} {S.shape[0]:>5} {rho:>6.3f} {Te:>6.0f} {rank1:>12.3f} "
              f"{str(hi)+' / '+str(npairs):>16} {str(lo)+' / '+str(npairs):>16} {0.025:>12.3f}")


if __name__ == "__main__":
    root = "analysis/killtest_data"
    mx = lambda g: sorted(f"{root}/mx/probe_sig_{g}_s{s}" for s in (0, 1, 2))

    print("=" * 100)
    print("KILL-TEST: is the meta-gradient sign-agreement excess explained by ARCHITECTURE?")
    print("=" * 100)

    print("\n\n" + "=" * 100)
    print("(d) REPRODUCTION CHECK -- published per-arm agreement, from the GLOBAL frac_neg field")
    print("=" * 100)
    for g in ("scalar", "resnet18_blocks", "layerwise", "nodewise", "weightwise"):
        headline(mx(g), f"mx/probe_sig_{g}")
    print("\n" + "=" * 100)
    print("gate3 (20-epoch, 500-record, independent batch)")
    print("=" * 100)
    for g in ("scal", "blk6", "layer"):
        headline(sorted(f"{root}/gate3/probe_{g}_s{s}" for s in (0, 1, 2)), f"gate3/probe_{g}")

    print("\n\n" + "=" * 100)
    print("(b)/(c) WITHIN vs ACROSS, at TENSOR granularity -- pairwise same-sign rate, null = 50.000%")
    print("=" * 100)
    lay = analyse_arm(mx("layerwise"), "LAYERWISE arm (m=62): each coordinate IS one param tensor",
                      "'ALL pairs' = across-tensor agreement (b).  within/across-blk = the 6-block split (c).")
    wgt = analyse_arm(mx("weightwise"), "WEIGHTWISE arm (m=11.17M): per-tensor SPATIAL MEAN of the per-weight meta-gradient",
                      "This is the arm the 53.1%/11.17M headline was measured on.")
    nod = analyse_arm(mx("nodewise"), "NODEWISE arm (m=14,420): per-tensor spatial mean of the per-node meta-gradient")

    print("\n\n" + "=" * 100)
    print("ROBUSTNESS: does the split survive an independent, precision-exact estimator?")
    print("=" * 100)
    print("\n### float32 running-mean differencing -- noise budget and stability vs stride")
    for d in mx("layerwise")[:1] + mx("weightwise")[:1]:
        med, frac = ulp_noise(d)
        print(f"\n  {os.path.basename(d)}: median float32 noise/signal = {med:.2e}, "
              f"fraction of entries where noise > signal = {100*frac:.2f}%")
        precision_check(d)
    beta_replicate(mx("layerwise"), "LAYERWISE arm")
    beta_replicate(mx("weightwise"), "WEIGHTWISE arm")

    print("\n\n" + "=" * 100)
    print("IS THE SPLIT ARCHITECTURAL?  permutation + random-partition + common-mode controls")
    print("=" * 100)
    permutation_test(lay, "LAYERWISE arm")
    permutation_test(wgt, "WEIGHTWISE arm")
    common_mode_test(lay, "LAYERWISE arm")
    common_mode_test(wgt, "WEIGHTWISE arm")
    marginal_adjusted(lay, "LAYERWISE arm")
    marginal_adjusted(wgt, "WEIGHTWISE arm")

    print("\n\n" + "=" * 100)
    print("(4) BLOCK SIZE")
    print("=" * 100)
    block_size_analysis(lay, "LAYERWISE arm")
    block_size_analysis(wgt, "WEIGHTWISE arm")
    aggregation_ladder(lay, "LAYERWISE arm")

    cluster_test(lay, "LAYERWISE arm")
    cluster_test(wgt, "WEIGHTWISE arm")
