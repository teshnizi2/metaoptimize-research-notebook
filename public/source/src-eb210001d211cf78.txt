#!/usr/bin/env python3
"""
c99_qcalibration.py -- what the paper's Cochran Q is actually distributed as, and the
weight-free permutation that does not depend on the standard errors at all.

    python3 analysis/c99_qcalibration.py                 # the headline calibration
    python3 analysis/c99_qcalibration.py --endpoints      # ...on all four endpoints
    python3 analysis/c99_qcalibration.py --draws 200000   # more Monte-Carlo precision

WHY THIS FILE EXISTS.
=====================
Section 4.4 pools fourteen within-batch contrasts with inverse-variance weights
w_i = se_i^-2 and refers Cochran's Q to chi-square on k-1 df.  That reference is only
valid when the se_i are known, or estimated at enough degrees of freedom to be treated
as known.  They are not: every se_i here is a Welch standard error formed from two arms
of three to six runs, and the Welch-Satterthwaite degrees of freedom of the fourteen
live cells run from 2.04 to 9.68 with a MEDIAN OF 2.91.  At three degrees of freedom a
sample variance is so unstable that the resulting Q is enormously over-dispersed
relative to chi2_{k-1}: this file measures the over-dispersion by simulation, and the
null mean of Q on 13 df comes out near 27, not 13.

The paper already concedes exactly this problem one layer down.  Section 3.3's
Multiplicity paragraph says that at n = 3 v 3 the normal approximation and the
Welch-Satterthwaite value "differ materially at 2--4 degrees of freedom", and every t in
the paper is Welch-corrected on that ground.  The concession was never carried into the
Q layer.  This file carries it.

WHAT IT DOES NOT DO.
====================
It does not delete the decomposition.  The heterogeneity is real -- it survives at
Monte-Carlo p 0.013 -- and the base-optimiser structure is confirmed by an exact
permutation test that uses no standard errors at all (eta^2 = 0.840, exact p = 0.00020
over all 45,045 partitions of the realised shape).  What moves is the SIZE of the
p-values and the WIDTH of the intervals, not the sign or the direction of any claim.

HOW THE NULL IS BUILT (and it goes through the paper's own estimator).
======================================================================
Under the null every cell shares one true D.  For cell i we draw n_c uniform-arm values
and n_n aligned-arm values from normals with the cell's own arm standard deviations, at
the cell's own observed n, form (D*, se*) with c98_figures.welch -- the same function
the paper uses -- and pool the fourteen with c98_figures.meta, the same DerSimonian-Laird
routine the paper uses.  Q* is read off that pool.  Nothing about the estimator changes;
only the data are resampled.  Two nulls are reported because they bracket the answer:

  percell : cell i's two arm sds are taken as truth.  Faithful to the observed
            heteroscedasticity, but each "truth" is itself a 2-5 df estimate, so this
            null inherits the very instability it is measuring and is the CONSERVATIVE
            (larger-p) of the two.
  common  : one pooled within-arm sd, estimated on all 70 within-arm df at once, is used
            for every arm.  Much better determined, but it assumes the cells are
            homoscedastic.  This is the ANTI-CONSERVATIVE (smaller-p) of the two.

The paper prints the percell number as its calibrated p and the common number beside it,
so the reader sees the bracket rather than the flattering end of it.

SPEED.  _welch is a float two-pass rewrite of c98_figures.welch (which routes through
statistics.variance and its exact-Fraction arithmetic, ~18x slower).  It is not trusted:
check_estimator_fidelity() asserts the two agree to 1e-12 on all fourteen observed cells
AND on the first 200 simulated draws before any figure is reported, and it is called
from main() and from the c98_reproduce.py section.

HOUSE RULES.  plateau5 is the primary and is what every headline number here reads; the
other three end-of-training columns are read side by side under --endpoints as the same
endpoint DISCLOSURE section 4.4 already makes, and none of them is made primary.  The
cell set is c98_figures.POOL12 minus the withdrawn GroupNorm cell -- the same fourteen
section 4.4 pools -- and sm4 can never enter it because c98_figures gives it its own
base string.
"""
import argparse, itertools, math, os, statistics as st, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c98_figures as F
from c98_figures import load, welch, meta, chi2_sf

# ---- REGISTERED CONSTANTS.  Changing either changes every number this file prints.
SEED   = 20260902
DRAWS  = 20000
BASES  = ["SGDm", "SGD", "RMSProp", "AdamW"]
SHAPE  = (8, 2, 2, 2)          # the realised base-optimiser shape on the fourteen cells


# ------------------------------------------------------------------ the cell set
def live_arms(adm, arm_fn=None):
    """The fourteen live cells of S4.4, each with its two arms' raw values.

    Same selection as c98_reproduce.heterogeneity(): POOL12 minus the withdrawn
    GroupNorm cell.  `arm_fn` defaults to c98_figures.arm (plateau5); pass
    c98_reproduce._arm_on(col) to read another end-of-training column.
    """
    arm_fn = arm_fn or F.arm
    out = []
    for (lab, net, ds, base, eta, ep, ch, nd, c23, n1d) in F.CELLS:
        if lab not in F.POOL12 or lab == F.GN_CELL:
            continue
        cv, _ = arm_fn(adm, *ch)
        nv, _ = arm_fn(adm, *nd)
        out.append(dict(label=lab, base=base, cv=cv, nv=nv))
    assert len(out) == 14, "the live cell set moved: %d cells" % len(out)
    assert all(c["base"] != "AdamW+RMS" for c in out), "sm4 leaked into the pool"
    assert sorted(sum(1 for c in out if c["base"] == b) for b in BASES) == \
        sorted(SHAPE), "the base-optimiser shape moved"
    return out


def welch_df(cell):
    """Welch-Satterthwaite df of one cell's D.  This is the number that makes the
    chi-square reference invalid, so it is reported rather than assumed."""
    a, b = cell["cv"], cell["nv"]
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    return (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))


# ---------------------------------------------------------------- the estimator
def _welch(a, b):
    """Float two-pass rewrite of c98_figures.welch.  Fidelity is ASSERTED, not assumed."""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    return ma - mb, math.sqrt(va / na + vb / nb)


def check_estimator_fidelity(cs, seed=SEED, draws=200, tol=1e-12):
    """_welch must agree with the paper's own welch() on the observed cells and on
    simulated draws, or the speed-up is not free and nothing below may be reported."""
    worst = 0.0
    for c in cs:
        d1, s1, _ = welch(c["cv"], c["nv"])
        d2, s2 = _welch(c["cv"], c["nv"])
        worst = max(worst, abs(d1 - d2), abs(s1 - s2))
    rng = random.Random(seed)
    S = arm_sds(cs, "percell")
    for _ in range(draws):
        for c, (sc, sn) in zip(cs, S):
            cv = [rng.gauss(0.0, sc) for _ in range(len(c["cv"]))]
            nv = [rng.gauss(0.0, sn) for _ in range(len(c["nv"]))]
            d1, s1, _ = welch(cv, nv)
            d2, s2 = _welch(cv, nv)
            worst = max(worst, abs(d1 - d2), abs(s1 - s2))
    assert worst <= tol, "the fast estimator does not reproduce welch(): %.3e" % worst
    return worst


# -------------------------------------------------------------------- the nulls
def arm_sds(cs, mode):
    """The per-arm standard deviations the null draws from.  See the module docstring."""
    if mode == "percell":
        return [(st.stdev(c["cv"]), st.stdev(c["nv"])) for c in cs]
    if mode == "common":
        num = sum((len(c["cv"]) - 1) * st.variance(c["cv"]) +
                  (len(c["nv"]) - 1) * st.variance(c["nv"]) for c in cs)
        den = sum((len(c["cv"]) - 1) + (len(c["nv"]) - 1) for c in cs)
        return [(math.sqrt(num / den), math.sqrt(num / den))] * len(cs)
    raise ValueError("mode must be 'percell' or 'common'")


def pooled_arm_sd(cs):
    """(pooled within-arm sd, its df) -- what the 'common' null draws from."""
    num = sum((len(c["cv"]) - 1) * st.variance(c["cv"]) +
              (len(c["nv"]) - 1) * st.variance(c["nv"]) for c in cs)
    den = sum((len(c["cv"]) - 1) + (len(c["nv"]) - 1) for c in cs)
    return math.sqrt(num / den), den


def observed(cs):
    """Everything S4.4 prints, re-derived: pool, se, Q, df, tau, the within/between
    split on the base optimiser, and the per-level Q's."""
    items = [_welch(c["cv"], c["nv"]) for c in cs]
    bs = [c["base"] for c in cs]
    m, sem, Q, df, tau = meta(items)
    per, within, wdf = {}, 0.0, 0
    for b in BASES:
        it = [x for x, bb in zip(items, bs) if bb == b]
        if not it:
            continue
        per[b] = meta(it)[2]
        within += per[b]
        wdf += len(it) - 1
    return dict(items=items, m=m, se=sem, Q=Q, df=df, tau=tau, per=per,
                within=within, wdf=wdf, between=Q - within, bdf=df - wdf,
                share=(Q - within) / Q if Q else float("nan"))


def qnull(cs, draws=DRAWS, seed=SEED, mode="percell"):
    """Simulate the paper's own estimator under 'every cell shares one true D'.

    Returns parallel lists: Q, within-level Q, between-level Q, the between share,
    the DerSimonian-Laird tau, the pooled estimate and its reported se, plus the
    per-level Q's.  The true D is set to zero; Q, tau and the split are invariant to
    a common shift, and coverage is asked as |m*| <= c * se*, so nothing depends on it.
    """
    rng = random.Random(seed)
    S = arm_sds(cs, mode)
    bs = [c["base"] for c in cs]
    out = dict(Q=[], W=[], B=[], SH=[], TAU=[], M=[], SE=[],
               LV={b: [] for b in BASES})
    for _ in range(draws):
        items = []
        for c, (sc, sn) in zip(cs, S):
            cv = [rng.gauss(0.0, sc) for _ in range(len(c["cv"]))]
            nv = [rng.gauss(0.0, sn) for _ in range(len(c["nv"]))]
            items.append(_welch(cv, nv))
        m, sem, Q, df, tau = meta(items)
        w = 0.0
        for b in BASES:
            it = [x for x, bb in zip(items, bs) if bb == b]
            if not it:
                continue
            q = meta(it)[2]
            w += q
            out["LV"][b].append(q)
        out["Q"].append(Q); out["W"].append(w); out["B"].append(Q - w)
        out["SH"].append((Q - w) / Q if Q else 0.0)
        out["TAU"].append(tau); out["M"].append(m); out["SE"].append(sem)
    return out


def quant(v, p):
    v = sorted(v)
    i = p * (len(v) - 1)
    lo = int(math.floor(i)); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (i - lo) * (v[hi] - v[lo])


def mc_p(v, obs):
    """Monte-Carlo upper-tail p, (1 + #{>= obs}) / (1 + N): never reports zero, and is
    the standard conservative estimator for a simulated reference distribution."""
    return (1.0 + sum(1 for x in v if x >= obs)) / (len(v) + 1.0)


def re_pool(items, tau):
    """DerSimonian-Laird RANDOM-effects pool: (estimate, se), weights 1/(se^2 + tau^2).
    meta() returns the FIXED-effect pool, which is what every 'pooled D' in the paper has
    meant; when tau > 0 that is not the right primary and S4.4 now says so."""
    w = [1.0 / (se ** 2 + tau ** 2) for _, se in items]
    W = sum(w)
    return sum(wi * yi for wi, (yi, _) in zip(w, items)) / W, 1.0 / math.sqrt(W)


def coverage(sim, c=1.96):
    return sum(1 for m, s in zip(sim["M"], sim["SE"]) if abs(m) <= c * s) / len(sim["M"])


def half_width(sim, p=0.95):
    """The calibrated 95% half-width in pp: the p-quantile of |m* - mu| under the null.
    An interval of this width has exact coverage p by construction."""
    return quant([abs(x) for x in sim["M"]], p)


# ------------------------------------------------- the weight-free permutation
def eta2(groups):
    """Unweighted between-group share of the total sum of squares of the D's.  No se
    enters this statistic anywhere, which is the entire point of it."""
    allv = [x for g in groups for x in g]
    gm = st.mean(allv)
    sst = sum((x - gm) ** 2 for x in allv)
    ssb = sum(len(g) * (st.mean(g) - gm) ** 2 for g in groups)
    return ssb / sst


def _shape_partitions(n=14, big=8):
    """All distinct partitions of n labelled cells into one block of `big` and three
    UNORDERED blocks of 2.  Yields index tuples; there are C(14,8)*15*6/6 = 45,045."""
    idx = list(range(n))
    for eight in itertools.combinations(idx, big):
        rest = [i for i in idx if i not in eight]
        a0 = rest[0]                                   # fix the smallest element ...
        for b0 in rest[1:]:
            r2 = [i for i in rest if i not in (a0, b0)]
            a1 = r2[0]                                 # ... and again, so the three
            for b1 in r2[1:]:                          # doubletons are never permuted
                r3 = [i for i in r2 if i not in (a1, b1)]
                yield list(eight), [a0, b0], [a1, b1], r3


def permutation_eta2(cs):
    """Exact permutation test of the base-optimiser partition on the raw D's.

    Returns (observed eta^2, rank, total, exact p, null median, null p95, null max).
    The p is EXACT -- every partition of the realised shape is enumerated, so there is
    no Monte-Carlo error in it at all.
    """
    D = [_welch(c["cv"], c["nv"])[0] for c in cs]
    bs = [c["base"] for c in cs]
    e0 = eta2([[d for d, b in zip(D, bs) if b == bb] for bb in BASES])
    ge, tot, vals = 0, 0, []
    for g1, g2, g3, g4 in _shape_partitions(len(cs), SHAPE[0]):
        e = eta2([[D[i] for i in g] for g in (g1, g2, g3, g4)])
        vals.append(e); tot += 1
        if e >= e0 - 1e-12:
            ge += 1
    return e0, ge, tot, ge / tot, quant(vals, 0.5), quant(vals, 0.95), max(vals)


def permutation_share(cs):
    """The same enumeration on the WEIGHTED share of Q that S4.4 already prints.
    Carried only as a cross-check that this enumeration reproduces the paper's own
    published permutation figure (rank 14, p 0.00031)."""
    items = [_welch(c["cv"], c["nv"]) for c in cs]
    Qt = meta(items)[2]
    bs = [c["base"] for c in cs]
    obs_idx = [[i for i in range(len(cs)) if bs[i] == b] for b in BASES]

    def sh(gidx):
        w = sum(meta([items[i] for i in g])[2] for g in gidx)
        return (Qt - w) / Qt
    s0 = sh(obs_idx)
    ge, tot = 0, 0
    for g1, g2, g3, g4 in _shape_partitions(len(cs), SHAPE[0]):
        if sh((g1, g2, g3, g4)) >= s0 - 1e-12:
            ge += 1
        tot += 1
    return s0, ge, tot, ge / tot


# ------------------------------------------------- the Q-profile limit on tau
#   S4.4 prints "one-sided 95% Q-profile upper limit 0.109 pp" for the SGDm level.
#   That limit inverts Q_gen(tau^2) against the 5th percentile of chi2_{k-1}, so it
#   carries the same defect as every other chi-square reference in the subsection.
#   Here the reference is simulated AT EACH TRIAL tau -- a between-cell effect
#   N(0, tau^2) is added to the cells' true D's -- so the limit is properly
#   calibrated rather than approximated from the tau = 0 null.
def q_gen(items, t2):
    w = [1.0 / (se ** 2 + t2) for _, se in items]
    W = sum(w)
    m = sum(wi * yi for wi, (yi, _) in zip(w, items)) / W
    return sum(wi * (yi - m) ** 2 for wi, (yi, _) in zip(w, items))


def tau_upper(cs, draws=DRAWS, seed=SEED, mode="percell", alpha=0.05, hi=0.09):
    """One-sided (1-alpha) Q-profile upper limit on tau, with the reference simulated
    at each trial tau.  hi is an upper bracket on tau^2 (0.09 -> tau <= 0.3 pp)."""
    items = [_welch(c["cv"], c["nv"]) for c in cs]
    S = arm_sds(cs, mode)

    def ref(t2):
        rng = random.Random(seed)
        tau, out = math.sqrt(t2), []
        for _ in range(draws):
            its = []
            for c, (sc, sn) in zip(cs, S):
                u = rng.gauss(0.0, tau) if tau > 0 else 0.0
                cv = [rng.gauss(u, sc) for _ in range(len(c["cv"]))]
                nv = [rng.gauss(0.0, sn) for _ in range(len(c["nv"]))]
                its.append(_welch(cv, nv))
            out.append(q_gen(its, t2))
        return quant(out, alpha)

    def g(t2):
        return q_gen(items, t2) - ref(t2)
    if g(0.0) <= 0:
        return 0.0
    a, b = 0.0, hi
    for _ in range(18):
        mid = 0.5 * (a + b)
        if g(mid) > 0: a = mid
        else: b = mid
    return math.sqrt(0.5 * (a + b))


# -------------------------------------------- the registered 1-df decision rule
#   c97_bm2_score.py's registered line is "Q > 3.841 => the level is not a stable
#   quantity" -- the chi2_1 5% critical value.  A two-cell level's Q is not chi2_1
#   here either, so the rule's REAL size is measured rather than assumed.
RULE_1DF = 3.841

def rule_size(cs, draws=DRAWS, seed=SEED, mode="percell"):
    """(realised size of the Q > 3.841 rule, its size-0.05 critical value), pooled over
    the three two-cell levels."""
    v = []
    for b in ("SGD", "RMSProp", "AdamW"):
        sub = [c for c in cs if c["base"] == b]
        v += qnull(sub, draws, seed, mode)["Q"]
    return sum(1 for x in v if x > RULE_1DF) / len(v), quant(v, 0.95)


def subpool_null(cs, draws=DRAWS, seed=SEED, mode="percell"):
    """(Q, df, chi2 p, MC p, null mean) for the Cochran Q of an arbitrary sub-pool of the
    fourteen Dstat cells -- e.g. the ten momentum-present cells whose residual Q S4.5's
    2x2 collapse reads.  Same estimator, same null construction as qnull()."""
    o = observed(cs)
    v = qnull(cs, draws, seed, mode)["Q"]
    return o["Q"], o["df"], chi2_sf(o["Q"], o["df"]), mc_p(v, o["Q"]), sum(v) / len(v)


# ------------------------------------------------- the G family (S5.4 multiplicity)
def gfamily_arms(adm):
    """(twelve-cell, fourteen-cell) arm pairs for the Gstat family of S5.4.

    Gstat = mu(chunk2325) - mu(nodewise1d), so this is a DIFFERENT contrast from the
    Dstat pool above and needs its own null.  The cell set is exactly
    c98_reproduce.tail()'s: every cell that carries a Gstat, and that set minus the two
    AdamW-base cells sm3 and sm4.  Fidelity is ASSERTED below, not assumed.
    """
    spec = {t[0]: (t[8], t[9]) for t in F.CELLS}
    fam14 = [c for c in F.cells(adm) if c["G"] is not None]
    fam12 = [c for c in fam14 if c["label"] not in ("sm3", "sm4")]
    assert len(fam14) == 14 and len(fam12) == 12, \
        "the G family moved: %d / %d cells" % (len(fam14), len(fam12))
    def pull(fam):
        out = []
        for c in fam:
            c23, n1d = spec[c["label"]]
            a, _ = F.arm(adm, *c23)
            b, _ = F.arm(adm, *n1d)
            out.append((a, b))
        return out
    return pull(fam12), pull(fam14)


def gfamily_null(adm, draws=DRAWS, seed=SEED):
    """Calibrate S5.4's two Gstat-family Cochran Q's against their own simulated null.

    Returns {"12": (Q, df, chi2 p, MC p, null mean), "14": ..., "15": ...} -- the
    pre-specified twelve, the enlarged fourteen and the fifteen that adds bn1.  Before it reports
    anything it re-derives both Q's from the raw arms through the paper's own welch()
    and meta() and asserts they equal the printed 18.21 and 28.25 -- the same discipline
    check_estimator_fidelity() makes for the Dstat pool.
    """
    out = {}
    A12, A14 = gfamily_arms(adm)
    A15 = A14 + [(F.arm(adm, "bn1-c23", "chunk2325")[0],
                  F.arm(adm, "bn1-n1d", "nodewise1d")[0])]
    for tag, AA, printed in (("12", A12, 18.21), ("14", A14, 28.25), ("15", A15, 42.98)):
        items = [welch(a, b)[:2] for a, b in AA]
        m, sem, Q, df, tau = meta(items)
        assert abs(Q - printed) < 0.005, \
            "the G family's Q moved: %.4f against the printed %.2f" % (Q, printed)
        rng = random.Random(seed)
        S = [(st.stdev(a), st.stdev(b)) for a, b in AA]
        v = []
        for _ in range(draws):
            it = []
            for (a, b), (sa, sb) in zip(AA, S):
                x = [rng.gauss(0.0, sa) for _ in range(len(a))]
                y = [rng.gauss(0.0, sb) for _ in range(len(b))]
                it.append(_welch(x, y))
            v.append(meta(it)[2])
        out[tag] = (Q, df, chi2_sf(Q, df), mc_p(v, Q), sum(v) / len(v))
    return out


# ------------------------------------------------------------------------ report
def _hdr(t):
    print("\n" + t)
    print("-" * len(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--endpoints", action="store_true",
                    help="repeat the Q calibration on all four end-of-training columns")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    rows, adm = load(a.csv) if a.csv else load()
    cs = live_arms(adm)
    worst = check_estimator_fidelity(cs, a.seed)
    print("c99_qcalibration -- %d cells, %d draws, seed %d "
          "(fast estimator agrees with welch() to %.1e)" % (len(cs), a.draws, a.seed, worst))

    _hdr("[1] WHY THE CHI-SQUARE REFERENCE IS WRONG: the Welch df of the fourteen cells")
    dfs = []
    for c in cs:
        d, se = _welch(c["cv"], c["nv"])
        dfs.append(welch_df(c))
        print("   %-16s %-9s n %d v %d   D %+0.4f  se %0.4f   Welch df %6.2f"
              % (c["label"], c["base"], len(c["cv"]), len(c["nv"]), d, se, dfs[-1]))
    print("   Welch df: min %.2f  median %.2f  mean %.2f  max %.2f  -- the se's are NOT known"
          % (min(dfs), st.median(dfs), st.mean(dfs), max(dfs)))
    psd, pdf = pooled_arm_sd(cs)
    print("   pooled within-arm sd %.4f pp on %d df (the 'common' null draws from this)"
          % (psd, pdf))

    obs = observed(cs)
    _hdr("[2] THE OBSERVED VALUES, RE-DERIVED")
    print("   fixed-effect pool %+0.4f +- %0.4f   Q %.4f on %d df   tau %.4f   I2 %.1f%%"
          % (obs["m"], obs["se"], obs["Q"], obs["df"], obs["tau"],
             100 * (obs["Q"] - obs["df"]) / obs["Q"]))
    print("   within-level Q %.4f on %d df | between-base Q %.4f on %d df | share %.2f%%"
          % (obs["within"], obs["wdf"], obs["between"], obs["bdf"], 100 * obs["share"]))
    for b in BASES:
        k = sum(1 for c in cs if c["base"] == b)
        print("      level %-8s k=%d   Q %6.4f on %d df   chi2 p %.4f"
              % (b, k, obs["per"][b], k - 1, chi2_sf(obs["per"][b], k - 1)))

    _hdr("[3] THE CALIBRATED REFERENCE DISTRIBUTION OF Q")
    sims = {}
    for mode in ("percell", "common"):
        sims[mode] = s = qnull(cs, a.draws, a.seed, mode)
        print("   ---- %s null" % mode)
        for nm, key, o, df_ in (("Q total", "Q", obs["Q"], obs["df"]),
                                ("within-level Q", "W", obs["within"], obs["wdf"]),
                                ("between-base Q", "B", obs["between"], obs["bdf"])):
            v = s[key]
            print("     %-16s null mean %7.3f med %7.3f p95 %7.3f  (chi2_%d: %d, %.2f, %.2f)"
                  % (nm, st.mean(v), quant(v, .5), quant(v, .95), df_, df_,
                     _chi2_med(df_), _chi2_p95(df_)))
            print("     %-16s obs %8.3f   chi2 p %-10.4g   MONTE-CARLO p %.4f"
                  % ("", o, chi2_sf(o, df_), mc_p(v, o)))
        print("     between share    null mean %.3f med %.3f p95 %.3f | obs %.4f  MC p %.5f"
              % (st.mean(s["SH"]), quant(s["SH"], .5), quant(s["SH"], .95),
                 obs["share"], mc_p(s["SH"], obs["share"])))
        print("     tau              null mean %.3f med %.3f p95 %.3f | obs %.4f  MC p %.5f"
              % (st.mean(s["TAU"]), quant(s["TAU"], .5), quant(s["TAU"], .95),
                 obs["tau"], mc_p(s["TAU"], obs["tau"])))
        for b in BASES:
            k = sum(1 for c in cs if c["base"] == b)
            v = s["LV"][b]
            print("     level %-8s k=%d  Q %6.3f  chi2 p %.4f | null med %6.3f p95 %7.3f"
                  "  MC p %.4f  (obs at pct %.0f of its own null)"
                  % (b, k, obs["per"][b], chi2_sf(obs["per"][b], k - 1),
                     quant(v, .5), quant(v, .95), mc_p(v, obs["per"][b]),
                     100.0 * sum(1 for x in v if x < obs["per"][b]) / len(v)))

    _hdr("[4] tau AND I^2 ARE UPPER BOUNDS: they subtract k-1 where the null mean is ~27")
    w = [1.0 / se ** 2 for _, se in obs["items"]]
    W = sum(w); den = W - sum(x ** 2 for x in w) / W
    for nm, e in (("chi2 k-1 = %d" % obs["df"], float(obs["df"]),),
                  ("percell null mean", st.mean(sims["percell"]["Q"])),
                  ("common null mean", st.mean(sims["common"]["Q"]))):
        t2 = max(0.0, (obs["Q"] - e) / den)
        print("   centred on %-22s (%6.3f)  ->  tau %.4f   I^2 %.1f%%"
              % (nm, e, math.sqrt(t2), 100 * max(0.0, (obs["Q"] - e) / obs["Q"])))

    _hdr("[5] THE INTERVALS UNDERCOVER: the eight SGDm cells, and the fourteen")
    for nm, sub in (("SGDm k=8", [c for c in cs if c["base"] == "SGDm"]),
                    ("all k=14", cs)):
        o = observed(sub)
        for mode in ("percell", "common"):
            s = qnull(sub, a.draws, a.seed, mode)
            print("   %-9s [%-7s] pool %+0.4f +- %0.4f | +-1.96se covers %5.2f%% | "
                  "true sd of m %.4f | calibrated 95%% half-width %.4f pp (nominal %.4f)"
                  % (nm, mode, o["m"], o["se"], 100 * coverage(s, 1.96),
                     st.stdev(s["M"]), half_width(s), 1.96 * o["se"]))
        o8 = observed(sub)
        print("   %-9s Q %.4f on %d df, chi2 p %.4g" % (nm, o8["Q"], o8["df"],
                                                        chi2_sf(o8["Q"], o8["df"])))

    sg = [c for c in cs if c["base"] == "SGDm"]
    print("   SGDm one-sided 95%% Q-profile upper limit on tau: CALIBRATED %.4f pp "
          "(the paper prints 0.109, referred to chi2_7)" % tau_upper(sg, a.draws, a.seed))
    print("   -- the ONE quantity the chi-square reference errs CONSERVATIVELY on: "
          "calibration tightens this limit rather than loosening it.")

    _hdr("[6] THE REGISTERED 1-df DECISION RULE, Q > 3.841, HAS THE WRONG SIZE")
    for mode in ("percell", "common"):
        sz, cv = rule_size(cs, a.draws, a.seed, mode)
        print("   [%-7s] P(Q > %.3f | homogeneous level) = %.4f, not 0.05; the size-0.05"
              " critical value is %.3f" % (mode, RULE_1DF, sz, cv))
    print("   -- the rule fires about twice as often as advertised, so a level that does"
          " NOT fire it\n      is stronger evidence of homogeneity than the rule claims,"
          " not weaker.")

    _hdr("[7] THE WEIGHT-FREE PERMUTATION -- no standard error enters this at all")
    e0, ge, tot, p, med, p95, mx = permutation_eta2(cs)
    print("   eta^2 = SSB/SST on the raw D's, base-optimiser grouping: %.4f" % e0)
    print("   exact enumeration of all %d partitions of shape %s: rank %d, EXACT p = %.6f"
          % (tot, "{%d,%d,%d,%d}" % SHAPE, ge, p))
    print("   null eta^2: median %.4f  p95 %.4f  max %.4f" % (med, p95, mx))
    s0, ge2, tot2, p2 = permutation_share(cs)
    print("   cross-check on the WEIGHTED share S4.4 already publishes: obs %.4f, rank %d "
          "of %d, p %.5f  (the paper prints rank 14, p 0.00031)" % (s0, ge2, tot2, p2))

    if a.endpoints:
        _hdr("[8] THE ENDPOINT DISCLOSURE, CALIBRATED (plateau5 stays primary)")
        import c98_reproduce as R
        for col in R.MS_METRICS:
            e = live_arms(adm, R._arm_on(col))
            o = observed(e)
            row = "   %-11s Q %7.3f/%d  chi2 p %-9.4g" % (col, o["Q"], o["df"],
                                                          chi2_sf(o["Q"], o["df"]))
            for mode in ("percell", "common"):
                s = qnull(e, a.draws, a.seed, mode)
                row += " | MC p (%s) %.4f" % (mode, mc_p(s["Q"], o["Q"]))
                if mode == "percell":
                    bq = mc_p(s["B"], o["between"])
            print(row)
            print("               between-base Q %7.3f/3  chi2 p %-9.4g | MC p (percell) %.4f"
                  % (o["between"], chi2_sf(o["between"], o["bdf"]), bq))
            pe = permutation_eta2(e)
            print("               weight-free eta^2 %.4f  rank %d of %d  EXACT p %.6f"
                  % (pe[0], pe[1], pe[2], pe[3]))

    _hdr("[9] THE Gstat FAMILY OF S5.4, CALIBRATED")
    g = gfamily_null(adm, a.draws, a.seed)
    for tag, name in (("12", "pre-specified twelve"), ("14", "enlarged fourteen"),
                      ("15", "fifteen, with bn1")):
        Q, df, cp, mp, mean = g[tag]
        print("   %-20s Q %6.2f/%-2d  chi2 p %-8.4g | null mean %5.1f | MC p %.3f"
              % (name, Q, df, cp, mean, mp))
    print("   None of the three resolves against its own null; the chi2 contrast between the")
    print("   twelve and the fourteen does not survive, so S5.4 no longer reads the fourteen as")
    print("   heterogeneous where the twelve was not, and the fifteen is no longer 'heterogeneous'.")
    mp = [c for c in cs if c["base"] in ("SGDm", "AdamW")]
    Q, df, cp, mp_, mean = subpool_null(mp, a.draws, a.seed)
    print("   momentum-present residual (S4.5's 2x2 collapse): Q %6.2f/%-2d  chi2 p %-8.4g"
          "| null mean %5.1f | MC p %.3f" % (Q, df, cp, mean, mp_))


def _chi2_med(k):
    """Median of chi2_k, by bisection on the paper's own chi2_sf.  Reference only."""
    lo, hi = 0.0, 10.0 * k + 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_sf(mid, k) > 0.5: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def _chi2_p95(k):
    lo, hi = 0.0, 10.0 * k + 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_sf(mid, k) > 0.05: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    main()
