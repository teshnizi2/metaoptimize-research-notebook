"""Which cross-block tensor pairs survive the marginal-preserving null?

The mean cross-block dependence is zero, but ~2-5% of cross-block pairs clear a Bonferroni
threshold against a null that already reproduces every tensor's own marginal bias and
autocorrelation.  If those surviving pairs concentrate on a few nameable tensors, that is
structure crossing architectural boundaries and Idea 2 has a (much narrower) target.  If
they are scattered uniformly, it is residual mis-specification and there is nothing to cluster.
"""
import json, math, sys
import numpy as np
from scipy.stats import norm
from killtest_idea2 import (load, interval_z, pair_agreement, block_labels,
                            split_within_across, RESNET18_BLOCKS)

RNG = np.random.default_rng(1)
ROOT = "analysis/killtest_data/mx"


def tensor_names(nb):
    """Reconstruct the 62 ResNet18 parameter names from the numel signature, and ASSERT
    every reconstructed numel against block_sizes.json.  The shortcut/downsample triple
    sits immediately after the FIRST BasicBlock of each stage, not at the end of the stage
    -- reading it the other way silently mislabels 9 of the 62 tensors."""
    names, num = ["conv1.weight", "bn1.weight", "bn1.bias"], [64 * 3 * 3 * 3, 64, 64]
    planes = [64, 128, 256, 512]
    inp = 64
    for L, p in enumerate(planes, start=1):
        for b in range(2):
            cin = inp if b == 0 else p
            names += [f"layer{L}.{b}.conv1.weight", f"layer{L}.{b}.bn1.weight", f"layer{L}.{b}.bn1.bias",
                      f"layer{L}.{b}.conv2.weight", f"layer{L}.{b}.bn2.weight", f"layer{L}.{b}.bn2.bias"]
            num += [p * cin * 9, p, p, p * p * 9, p, p]
            if b == 0 and L > 1:
                names += [f"layer{L}.0.downsample.0.weight",
                          f"layer{L}.0.downsample.1.weight", f"layer{L}.0.downsample.1.bias"]
                num += [p * inp, p, p]
        inp = p
    names += ["linear.weight", "linear.bias"]
    num += [512 * 10, 10]
    assert len(names) == len(nb), (len(names), len(nb))
    bad = [(j, names[j], num[j], int(nb[j])) for j in range(len(nb)) if num[j] != int(nb[j])]
    assert not bad, f"numel mismatch, name reconstruction is wrong: {bad}"
    return names


def kind(n):
    if "downsample.0" in n or ("conv" in n and "weight" in n):
        return "conv"
    if "downsample.1" in n or "bn" in n:
        return "bn"
    return "fc"


def surviving_pairs(run, n_null=600):
    recs, bs = load(f"{ROOT}/{run}")
    S = np.sign(interval_z(recs[len(recs) // 2:]))
    T, G = S.shape
    A, _ = pair_agreement(S)
    null = []
    for _ in range(n_null):
        Sh = np.empty_like(S)
        for g in range(G):
            Sh[:, g] = np.roll(S[:, g], int(RNG.integers(0, T)))
        An, _ = pair_agreement(Sh)
        null.append(An[np.triu_indices(G, 1)])
    null = np.array(null)
    iu = np.triu_indices(G, 1)
    mu, sd = null.mean(axis=0), null.std(axis=0, ddof=1) + 1e-12
    z = (A[iu] - mu) / sd
    lab = block_labels()
    cross = lab[iu[0]] != lab[iu[1]]
    zc = norm.ppf(1 - 0.05 / (2 * int(cross.sum())))
    sig = (np.abs(z) > zc) & cross
    return iu, z, sig, cross, zc, np.array(bs["n_b"], float)


if __name__ == "__main__":
    runs = [f"probe_sig_{g}_s{s}" for g in ("layerwise", "weightwise") for s in (0, 1, 2)]
    lab = block_labels()
    nb0 = np.array(json.load(open(f"{ROOT}/probe_sig_layerwise_s0/block_sizes.json"))["n_b"], float)
    names = tensor_names(nb0)
    kinds = np.array([kind(n) for n in names])

    print("=" * 100)
    print("WHICH cross-block pairs survive the marginal-preserving null?")
    print("=" * 100)
    print(f"\n  ResNet18 tensor-name reconstruction check (numel signature):")
    for j in (0, 3, 21, 22, 23, 60, 61):
        print(f"    [{j:2d}] block {lab[j]}  n_b={int(nb0[j]):>9,}  {names[j]}")

    deg = np.zeros(62)
    reps = {}
    for run in runs:
        iu, z, sig, cross, zc, nb = surviving_pairs(run)
        for a, b in zip(iu[0][sig], iu[1][sig]):
            deg[a] += 1; deg[b] += 1
            reps[(a, b)] = reps.get((a, b), 0) + 1
        print(f"\n  {run}: {int(sig.sum())} of {int(cross.sum())} cross-block pairs "
              f"survive |z|>{zc:.2f}")

    print(f"\n\n### Are the surviving edges concentrated?  (degree = surviving cross-block")
    print(f"### edges per tensor, summed over 6 runs; uniform scatter would give ~equal degree)")
    o = np.argsort(-deg)
    print(f"\n  {'tensor':34s} {'block':>6} {'n_b':>10} {'kind':>5} {'x-block edges':>14}")
    print("  " + "-" * 76)
    for j in o[:14]:
        print(f"  {names[j]:34s} {lab[j]:>6} {int(nb0[j]):>10,} {kinds[j]:>5} {int(deg[j]):>14}")
    print(f"  {'...':34s}")
    print(f"  {'(median tensor)':34s} {'':>6} {'':>10} {'':>5} {int(np.median(deg)):>14}")
    print(f"\n  top-5 tensors hold {100*deg[o[:5]].sum()/deg.sum():.1f}% of all surviving cross-block edges")
    print(f"  top-10 tensors hold {100*deg[o[:10]].sum()/deg.sum():.1f}%   (uniform would be 8.1% / 16.1%)")

    print(f"\n\n### By tensor kind -- surviving cross-block edge RATE per available pair")
    for k in ("conv", "bn", "fc"):
        m = kinds == k
        # available cross-block pairs with at least one endpoint of this kind
        iu = np.triu_indices(62, 1)
        cross = lab[iu[0]] != lab[iu[1]]
        touch = (m[iu[0]] | m[iu[1]]) & cross
        got = sum(1 for (a, b), c in reps.items() if (m[a] or m[b]) for _ in range(c))
        print(f"  {k:>5}: {m.sum():>2} tensors, {int(touch.sum()):>5} cross-block pairs available, "
              f"{got:>4} surviving edge-instances over 6 runs "
              f"({100*got/(6*max(touch.sum(),1)):.2f}% vs 0.0017% expected)")

    print(f"\n\n### Do the SAME pairs recur across the 6 independent runs?")
    from collections import Counter
    c = Counter(reps.values())
    tot = sum(reps.values())
    print(f"  {len(reps)} distinct cross-block pairs ever survive; {tot} edge-instances total")
    for k in sorted(c, reverse=True):
        print(f"    survived in {k}/6 runs: {c[k]:>4} pairs")
    exp2 = None
    # expected recurrence if each run independently flags ~m pairs at random
    m = tot / 6.0
    p1 = m / 1506.0
    exp2 = 1506 * (6 * p1 ** 2 * (1 - p1) ** 4 * 15 / 6)  # ~binomial P(>=2) leading term
    print(f"\n  If each run flagged {m:.0f} pairs at random out of 1506, the expected number")
    print(f"  surviving in >=2 runs would be ~{1506*(1-(1-p1)**6-6*p1*(1-p1)**5):.1f}")
    rec = sum(v for k, v in c.items() if k >= 2)
    print(f"  Observed surviving in >=2 runs: {rec}")
    if rec:
        print(f"\n  Pairs surviving in >=3 runs:")
        for (a, b), v in sorted(reps.items(), key=lambda kv: -kv[1]):
            if v >= 3:
                print(f"    {v}/6  blk{lab[a]}:{names[a]:32s} <-> blk{lab[b]}:{names[b]:32s}")
