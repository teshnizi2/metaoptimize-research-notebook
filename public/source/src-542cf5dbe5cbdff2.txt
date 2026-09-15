#!/usr/bin/env python3
"""
Analytic re-derivation of named_parameters() shape lists for the ResNets in
build_network.py, and the exact group census that HF.py's PATCH_GRANULARITY /
PATCH_NODEBN / PATCH_CHUNKWISE would allocate.

Instrument (verified against HF.py lines 72, 286-311, 341-348):
  net_param_names_and_size = [(name, p.data.size()) for name,p in net.named_parameters()]
  nodewise   : beta_i = ones(p_size[0])                      -> m = sum(shape[0])
  nodewise1d : beta_i = ones(1 if ndim==1 else p_size[0])    -> m = sum(1 if 1d else shape[0])
  chunkwise K: beta_i = ones(ceil(numel/K))                  -> m = sum(ceil(numel/K))
No torch required; pure shape arithmetic.
"""
import math

def conv(cout, cin, k):   return (cout, cin, k, k)
def bn(c):                return [(c,), (c,)]

def basicblock(in_planes, planes, stride):
    exp = 1
    P = [("conv1", conv(planes, in_planes, 3))]
    P += [("bn1.w", (planes,)), ("bn1.b", (planes,))]
    P += [("conv2", conv(planes, planes, 3))]
    P += [("bn2.w", (planes,)), ("bn2.b", (planes,))]
    if stride != 1 or in_planes != exp * planes:
        P += [("sc.conv", conv(exp * planes, in_planes, 1))]
        P += [("sc.bn.w", (exp * planes,)), ("sc.bn.b", (exp * planes,))]
    return P, exp * planes

def bottleneck(in_planes, planes, stride):
    exp = 4
    P = [("conv1", conv(planes, in_planes, 1))]
    P += [("bn1.w", (planes,)), ("bn1.b", (planes,))]
    P += [("conv2", conv(planes, planes, 3))]
    P += [("bn2.w", (planes,)), ("bn2.b", (planes,))]
    P += [("conv3", conv(exp * planes, planes, 1))]
    P += [("bn3.w", (exp * planes,)), ("bn3.b", (exp * planes,))]
    if stride != 1 or in_planes != exp * planes:
        P += [("sc.conv", conv(exp * planes, in_planes, 1))]
        P += [("sc.bn.w", (exp * planes,)), ("sc.bn.b", (exp * planes,))]
    return P, exp * planes

def resnet_params(block, num_blocks, num_classes=10):
    exp = 1 if block is basicblock else 4
    P = [("conv1", conv(64, 3, 3)), ("bn1.w", (64,)), ("bn1.b", (64,))]
    in_planes = 64
    for planes, nb, stride in zip([64,128,256,512], num_blocks, [1,2,2,2]):
        strides = [stride] + [1]*(nb-1)
        for s in strides:
            bp, in_planes = block(in_planes, planes, s)
            P += bp
    P += [("linear.w", (num_classes, 512*exp)), ("linear.b", (num_classes,))]
    return P

def numel(s):
    n = 1
    for d in s: n *= d
    return n

def census(P):
    total = sum(numel(s) for _, s in P)
    node_m = sum(s[0] for _, s in P)
    # singleton nodewise groups: numel//shape[0] == 1  <=> ndim == 1
    sing_groups = sum(s[0] for _, s in P if numel(s)//s[0] == 1)
    sing_weights = sum(numel(s) for _, s in P if numel(s)//s[0] == 1)
    n1d_m = sum(1 if len(s) == 1 else s[0] for _, s in P)
    n_1d_tensors = sum(1 for _, s in P if len(s) == 1)
    # nodewise1d's own singletons: groups of size 1 under the n1d rule
    n1d_sing = 0
    for _, s in P:
        g = 1 if len(s) == 1 else s[0]
        if numel(s)//g == 1: n1d_sing += g
    return dict(tensors=len(P), params=total, node_m=node_m,
                singletons=sing_groups, sing_frac=sing_groups/node_m,
                sing_weights=sing_weights, sing_wfrac=sing_weights/total,
                n1d_m=n1d_m, n_1d_tensors=n_1d_tensors, n1d_sing=n1d_sing)

def chunk_count(P, K):
    return sum(math.ceil(numel(s)/K) for _, s in P)

def match_K(P, target):
    """smallest-|error| K whose chunk count is closest to target; also exact hits."""
    best = None
    exact = []
    lo, hi = 1, max(1, int(sum(numel(s) for _,s in P)/max(target,1))*4 + 10)
    for K in range(1, hi):
        c = chunk_count(P, K)
        d = abs(c - target)
        if c == target: exact.append(K)
        if best is None or d < best[1]: best = (K, d, c)
    return best, exact

ARCHS = {
 "ResNet10":      (basicblock, [1,1,1,1], 10),
 "ResNet18":      (basicblock, [2,2,2,2], 10),
 "ResNet34":      (basicblock, [3,4,6,3], 10),
 "ResNet50":      (bottleneck, [3,4,6,3], 10),
 "ResNet10_c100": (basicblock, [1,1,1,1], 100),
 "ResNet18_c100": (basicblock, [2,2,2,2], 100),
 "ResNet34_c100": (basicblock, [3,4,6,3], 100),
 "ResNet50_c100": (bottleneck, [3,4,6,3], 100),   # NOT in build_network.py -- reported only
}

rows = []
for name,(blk, nb, nc) in ARCHS.items():
    P = resnet_params(blk, nb, nc)
    c = census(P); c["name"] = name; c["P"] = P
    rows.append(c)

hdr = f'{"arch":16} {"tensors":>7} {"params":>11} {"node_m":>8} {"sing":>7} {"sing%":>7} {"w_cov%":>8} {"n1d_m":>7} {"1d_tens":>7}'
print(hdr); print("-"*len(hdr))
for c in rows:
    print(f'{c["name"]:16} {c["tensors"]:7d} {c["params"]:11d} {c["node_m"]:8d} {c["singletons"]:7d} '
          f'{100*c["sing_frac"]:7.2f} {100*c["sing_wfrac"]:8.4f} {c["n1d_m"]:7d} {c["n_1d_tensors"]:7d}')

print()
print("chunk-matched K (count closest to nodewise m, and to nodewise1d m):")
for c in rows:
    b1, e1 = match_K(c["P"], c["node_m"])
    b2, e2 = match_K(c["P"], c["n1d_m"])
    print(f'  {c["name"]:16} node_m={c["node_m"]:6d} -> bestK={b1[0]:5d} count={b1[2]:6d} (exact Ks: {e1[:6]})'
          f'   n1d_m={c["n1d_m"]:6d} -> bestK={b2[0]:5d} count={b2[2]:6d} (exact Ks: {e2[:6]})')

print()
print("CROSS-CHECK ResNet18 vs FINDINGS 77.6 / cycle 81:")
r18 = [c for c in rows if c["name"]=="ResNet18"][0]
checks = [("params", r18["params"], 11173962),
          ("nodewise m", r18["node_m"], 14420),
          ("singletons", r18["singletons"], 9610),
          ("singleton frac %", round(100*r18["sing_frac"],2), 66.64),
          ("weight coverage %", round(100*r18["sing_wfrac"],2), 0.09),
          ("nodewise1d m", r18["n1d_m"], 4851),
          ("n 1-D tensors", r18["n_1d_tensors"], 41)]
ok = True
for lbl, got, want in checks:
    good = (got == want)
    ok &= good
    print(f'  {lbl:20} got={got!r:<14} expected={want!r:<12} {"OK" if good else "MISMATCH"}')
print("INSTRUMENT VALIDATED" if ok else "INSTRUMENT MISMATCH -- STOP")
