"""cX1_reduction_report.py -- read cX1_reduction_probe.py's per-tensor terms and
answer, for each group, the two questions that decide whether ANY reduction
intervention has power on this optimizer.

INPUT   <prefix>.terms.npy  (steps x 62 float64) and <prefix>.meta.json, both
        written by analysis/cX1_reduction_probe.py.

Q1 DOMINATION.  For each group G and each step: is sign(z_G) = sign(t_j) where
   j = argmax_{i in G} |t_i|?  A rate near 1 means the group's whole meta-signal
   is set by ONE tensor -- the briefing's claimed mechanism.

Q2 POWER.  For each candidate reweighting w, how often does
   sign(sum_i w_i t_i) DIFFER from sign(sum_i t_i)?  The Lion meta-update
   consumes ONLY that sign (cX1_reduction_noop_proof.py), so this rate IS the
   fraction of meta-steps a reweighted reduction would change.  A rate of 0
   means the intervention is a null BY CONSTRUCTION and the batch must not run.

   w = 1/n_i        per-coordinate MEAN (the natural fix for an unnormalised sum)
   w = 1/sqrt(n_i)  a milder variant
   w = 1/|t_i|      pure sign-vote (each tensor one vote, magnitude discarded)

Q3 the same, but restricted to the steps that matter: this also reports the rate
   over the LAST HALF of the recorded prefix, because early steps run at the
   alpha0 initialisation where every alpha is 1e-6 and h_condenced is still near
   zero.

RUN:  python3 analysis/cX1_reduction_report.py <prefix> [<prefix> ...]
"""
import json, sys

import numpy as np


def groups_of(meta):
    n = len(meta["names"])
    pgi = meta.get("param_groups_indices") or []
    if pgi:
        return [list(g) for g in pgi]
    if meta["stepsize_type"] == "scalar":
        return [list(range(n))]
    return [[i] for i in range(n)]


def rate(a, b):
    """fraction of entries where the two sign vectors differ"""
    sa, sb = np.sign(a), np.sign(b)
    return float(np.mean(sa != sb))


def report(prefix):
    t = np.load(prefix + ".terms.npy")
    meta = json.load(open(prefix + ".meta.json"))
    numel = np.asarray(meta["numel"], dtype=np.float64)
    G = groups_of(meta)
    S = t.shape[0]
    half = S // 2
    print("=" * 78)
    print("%s   spec=%s  steps=%d  type=%s  groups=%d"
          % (prefix.split("/")[-1], meta["spec"], S, meta["stepsize_type"], len(G)))
    for gi, g in enumerate(G):
        idx = np.asarray(g, dtype=int)
        tg = t[:, idx]
        n = numel[idx]
        z = tg.sum(axis=1)
        # Q1 domination
        j = np.argmax(np.abs(tg), axis=1)
        tmax = tg[np.arange(S), j]
        dom = float(np.mean(np.sign(z) == np.sign(tmax)))
        # how concentrated is the sum?
        share = np.abs(tmax) / np.maximum(np.abs(tg).sum(axis=1), 1e-300)
        print("  group %d: %d tensors, %d params" % (gi, len(idx), int(n.sum())))
        print("    Q1 sign(z) == sign(largest |t|) in %.4f of steps "
              "(last half %.4f)"
              % (dom, float(np.mean(np.sign(z[half:]) == np.sign(tmax[half:])))))
        print("       largest |t| is %.4f of sum|t| on average (median %.4f)"
              % (float(share.mean()), float(np.median(share))))
        cnt = np.bincount(j, minlength=len(idx))
        top = np.argsort(-cnt)[:3]
        # UNANIMITY.  A high Q1 rate is NOT evidence of domination if the terms
        # simply agree: with all 62 signs equal, ANY positive weighting gives the
        # same sign.  This separates the two.
        sg = np.sign(tg)
        pos = (sg > 0).sum(axis=1)
        neg = (sg < 0).sum(axis=1)
        unan = float(np.mean((pos == 0) | (neg == 0)))
        print("       terms UNANIMOUS in sign in %.4f of steps (last half %.4f); "
              "minority share of |t| median %.4f"
              % (unan, float(np.mean((pos[half:] == 0) | (neg[half:] == 0))),
                 float(np.median(
                     np.where(np.sign(z)[:, None] == sg, 0.0, np.abs(tg)).sum(axis=1)
                     / np.maximum(np.abs(tg).sum(axis=1), 1e-300)))))
        print("       argmax|t| is tensor %s" % ", ".join(
            "%s (%d%%)" % (meta["names"][idx[k]], round(100.0 * cnt[k] / S))
            for k in top))
        # Q2 power of reweightings
        for nm, w in (("w=1/n      ", 1.0 / n),
                      ("w=1/sqrt(n)", 1.0 / np.sqrt(n)),
                      ("sign-vote  ", None)):
            zz = np.sign(tg).sum(axis=1) if w is None else (tg * w).sum(axis=1)
            print("    Q2 %s sign differs from the standard sum in %.4f of "
                  "steps (last half %.4f)"
                  % (nm, rate(z, zz), rate(z[half:], zz[half:])))
        # Q4 ARITHMETIC SAFETY.  The patched branch accumulates in the tensors'
        # own float32, left to right, exactly as the standard branch does.  Does
        # that reproduce the float64 sign?  If not, the intervention would be
        # measuring float32 cancellation rather than the reweighting.
        w = (1.0 / n).astype(np.float64)
        z64 = (tg * w).sum(axis=1)
        acc = np.zeros(S, dtype=np.float32)
        for k in range(len(idx)):
            acc = (acc + (tg[:, k] * w[k]).astype(np.float32)).astype(np.float32)
        print("    Q4 w=1/n float32 left-to-right sum disagrees in sign with the "
              "float64 sum in %.4f of steps" % rate(z64, acc.astype(np.float64)))
        acc0 = np.zeros(S, dtype=np.float32)
        for k in range(len(idx)):
            acc0 = (acc0 + tg[:, k].astype(np.float32)).astype(np.float32)
        print("    Q4 STANDARD float32 left-to-right sum disagrees in sign with "
              "its own float64 sum in %.4f of steps"
              % rate(z, acc0.astype(np.float64)))
    print()


# =============================================================================
# THE TWO EXTRA MEASUREMENTS THAT DECIDED THE crn1 DESIGN.
#   `--steer NAME`  for one named tensor: how often does REMOVING it from its
#                   group flip that group's sign, under the standard weighting
#                   and under w = 1/n?  This is the direct test of "one tensor
#                   dictates the sign for its whole group".
#   `--bnshare`     what fraction of a group's sum|t| is held by the 1-D
#                   (BatchNorm scale/shift) tensors, under each weighting?
# Both are pure re-reads of the same .terms.npy; they run no model.
# =============================================================================
def extras(prefix, steer_names=(), bnshare=False):
    t = np.load(prefix + ".terms.npy")
    meta = json.load(open(prefix + ".meta.json"))
    numel = np.asarray(meta["numel"], dtype=np.float64)
    names = meta["names"]
    G = groups_of(meta)
    S = t.shape[0]
    half = S // 2
    print("-" * 78)
    print("%s   spec=%s  steps=%d" % (prefix.split("/")[-1], meta["spec"], S))
    for gi, g in enumerate(G):
        idx = np.asarray(g, dtype=int)
        tg = t[:, idx]
        w = 1.0 / numel[idx]
        for nm in steer_names:
            if nm not in names or names.index(nm) not in list(idx):
                continue
            k = list(idx).index(names.index(nm))
            keep = [j for j in range(len(idx)) if j != k]
            z, zw = tg.sum(1), tg[:, keep].sum(1)
            zn = (tg * w).sum(1)
            znw = (tg[:, keep] * w[keep]).sum(1)
            print("  g%d DROP %-30s (n=%d)" % (gi, nm, numel[idx][k]))
            print("     STANDARD sign flips in %.4f of steps (last half %.4f)"
                  % (rate(z, zw), rate(z[half:], zw[half:])))
            print("     w=1/n    sign flips in %.4f of steps (last half %.4f)"
                  % (rate(zn, znw), rate(zn[half:], znw[half:])))
        if bnshare:
            def is1d(n2):
                return not (n2.endswith("conv1.weight") or n2.endswith("conv2.weight")
                            or ".shortcut.0." in n2 or n2 == "conv1.weight"
                            or n2.startswith("linear"))
            bn = np.array([is1d(names[i]) for i in idx])
            a = np.abs(tg)
            den = np.maximum(a.sum(1), 1e-300)
            aw = a * w
            denw = np.maximum(aw.sum(1), 1e-300)
            print("  g%d %d tensors, %d of them 1-D BN (%.1f%% of tensors, "
                  "%.4f%% of params)"
                  % (gi, len(idx), int(bn.sum()), 100.0 * bn.mean(),
                     100.0 * numel[idx][bn].sum() / numel[idx].sum()))
            print("     BN share of sum|t|:  STANDARD %.4f   w=1/n %.4f"
                  % (float(np.mean(a[:, bn].sum(1) / den)),
                     float(np.mean(aw[:, bn].sum(1) / denw))))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(2)
    steer, pfx, want_bn = [], [], False
    i = 0
    while i < len(args):
        if args[i] == "--steer":
            steer.append(args[i + 1])
            i += 2
        elif args[i] == "--bnshare":
            want_bn = True
            i += 1
        else:
            pfx.append(args[i])
            i += 1
    for p in pfx:
        report(p)
    if steer or want_bn:
        print("=" * 78)
        print("EXTRAS -- single-tensor steering power, and the BN share")
        for p in pfx:
            extras(p, steer, want_bn)
