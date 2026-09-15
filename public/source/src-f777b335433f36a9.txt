"""N_eff(m) across MODEL SIZES and DATASETS, from probe batches already on disk.

Applies the exact-inversion effective-sample-size estimator of frozen_agreement.py
(validated in analysis/neff_validate.py) to every probe batch it is pointed at, and fits
the exponent s in  N_eff ~ m^s.  s = 1 is the independence assumption that the
Adam-mini / Adalayer / SGG line makes when it argues coarse granularity averages noise away.

CAVEAT THAT MUST TRAVEL WITH THESE NUMBERS.  `p7free` / `p6free` / `mx` are FREE-ADAPTING
(beta moves), so the granularity arms are on different trajectories by the time agreement is
read -- agreement and trajectory are confounded.  Only `fz-*` is frozen-beta and therefore
controlled; it exists only at R18/CIFAR-10.  These batches test whether the *exponent*
survives across models and datasets, they do not replace the controlled measurement.

m is INFERRED per run from the rational denominator of frac_neg (block_sizes.json reports
11.17M for nodewise arms -- CORRECTIONS 16), never read from the config.
"""
import os, glob, collections, math, sys
import numpy as np
from frozen_agreement import arm_stats

BATCHES = {
    # a0 is READ FROM THE RUNS (beta at step 0 = ln a0), not from prose.  This dict said
    # p7free was a0=1e-6; all 92 dirs are a0=1e-3 (cycle 44).  The distinction matters:
    # a0=1e-3 is what makes p7free byte-matched to fz-*-a3 except --alg-meta, which is the
    # controlled frozen/free pair CORRECTIONS 27 and FINDINGS 44.3 both rest on.
    "p7free": "20-ep free-adaptation ladder, a0=1e-3, SGDm+Lion",
    "p6free": "20-ep free-adaptation ladder (earlier batch), a0=1e-3",
    "mx":     "100-ep free-adaptation ladder, R18/CIFAR-10, a0=1e-6",
    "gate3":  "20-ep, R18/CIFAR-10, a0=1e-6",
}
ROOT = os.path.join(os.path.dirname(__file__), "killtest_data")


def parse(name):
    """-> (family, granularity_token).  Returns None when the name is not a ladder arm."""
    t = name.replace("probe_sig_", "").replace("probe_", "")
    t = t.rsplit("-s", 1)[0].rsplit("_s", 1)[0]
    p = t.replace("_", "-").split("-")
    if p and p[0] in ("p7", "p6f", "p9"):
        p = p[1:]
    gtok = p[-1]
    fam = "-".join(p[:-1]) or "R18/CIFAR-10"
    canon = {"blk6": "blk6", "resnet18": "blk6", "lay": "lay", "layerwise": "lay",
             "node": "node", "nodewise": "node", "w": "w", "weightwise": "w",
             "scal": "scal", "scalar": "scal", "blocks": "blk6"}
    if gtok not in canon:
        return None
    return fam, canon[gtok]


def run_batch(batch):
    root = os.path.join(ROOT, batch)
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    by = collections.defaultdict(list)
    for d in dirs:
        pr = parse(os.path.basename(d))
        if pr is None:
            continue
        try:
            s = arm_stats(d)
        except Exception:
            continue
        if s and s["nbar"] >= 2:
            by[pr].append(s)
    if not by:
        return
    print("\n" + "=" * 104)
    print(f"{batch}   -- {BATCHES.get(batch,'')}")
    print("=" * 104)
    fams = sorted({k[0] for k in by})
    for fam in fams:
        gs = [g for g in ("blk6", "lay", "node", "w") if by.get((fam, g))]
        if len(gs) < 3:
            continue
        print(f"\n  family = {fam}")
        h = (f"  {'granularity':12s} {'m (inferred)':>14s} {'seeds':>6s} {'agree%':>10s} "
             f"{'null%':>10s} {'N_eff':>12s} {'N/Neff':>10s}")
        print(h); print("  " + "-" * (len(h) - 2))
        for g in gs:
            v = by[(fam, g)]
            m = float(np.mean([s["nbar"] for s in v]))
            print(f"  {g:12s} {m:>14,.0f} {len(v):>6d} "
                  f"{100*np.mean([s['agree'] for s in v]):>10.4f} "
                  f"{100*np.mean([s['null'] for s in v]):>10.4f} "
                  f"{np.mean([s['neff'] for s in v]):>12,.1f} "
                  f"{np.mean([s['n_over_neff'] for s in v]):>10,.1f}")
        nseed = min(len(by[(fam, g)]) for g in gs)
        x = np.array([math.log10(np.mean([s["nbar"] for s in by[(fam, g)]])) for g in gs])
        sl = []
        for k in range(nseed):
            y = np.array([math.log10(max(by[(fam, g)][k]["neff"], 1e-9)) for g in gs])
            sl.append(np.polyfit(x, y, 1)[0])
        print(f"    N_eff ~ m^({np.mean(sl):.3f} +- "
              f"{(np.std(sl, ddof=1) if len(sl) > 1 else 0):.3f})   over {nseed} seeds, "
              f"m spanning {10**x.min():,.0f} to {10**x.max():,.0f}")


if __name__ == "__main__":
    for b in (sys.argv[1:] or list(BATCHES)):
        if os.path.isdir(os.path.join(ROOT, b)):
            run_batch(b)
