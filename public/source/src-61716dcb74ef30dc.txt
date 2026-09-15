#!/usr/bin/env python3
"""ctd1_attack_rederive.py -- INDEPENDENT re-derivation of cTD2 section [B] and the
five attacks on H-DOMINATE (tensor level), CORRECTIONS 184.  Written from the
definitions in cTD2's header, NOT from its code; numpy throughout; nothing here
is a registered scorer and nothing here gates anything.  cTD2 is not edited.

    python3 analysis/ctd1_attack_rederive.py <runsdir>      (runsdir/ctd1/probe_* and runsdir/cru1/probe_*)

Sections:
  B   [B] re-derived: R_T pooled/per seed, PINNED/UNPINNED split, DOWN, NAMED-CARRIED,
      carrying sets and carrier frequencies, on L_i (PRIMARY) and z_i (SECONDARY).
  A1  gradient-scale artefact: median |L_i|, |z_i| per tensor (total and per element),
      carriers vs median tensor, per arm; sign census by tensor class on pinned records.
  A2  layerwise companion: terminal beta, floor occupancy, first-pin step, own-vote sign,
      for the carriers, the NAMED set, and the class summary.
  A3  the narrowing: sign vs magnitude of the NAMED convs relative to the carriers, by phase.
  A5  seed independence: pin steps, peak steps, max |beta_s - beta_s'|, DISAGREE-set overlap.
"""
import glob, json, os, sys, collections
import numpy as np

NAMES = ("conv1.weight", "bn1.weight", "bn1.bias", "layer1.0.conv1.weight",
    "layer1.0.bn1.weight", "layer1.0.bn1.bias", "layer1.0.conv2.weight",
    "layer1.0.bn2.weight", "layer1.0.bn2.bias", "layer1.1.conv1.weight",
    "layer1.1.bn1.weight", "layer1.1.bn1.bias", "layer1.1.conv2.weight",
    "layer1.1.bn2.weight", "layer1.1.bn2.bias", "layer2.0.conv1.weight",
    "layer2.0.bn1.weight", "layer2.0.bn1.bias", "layer2.0.conv2.weight",
    "layer2.0.bn2.weight", "layer2.0.bn2.bias",
    "layer2.0.shortcut.0.weight", "layer2.0.shortcut.1.weight",
    "layer2.0.shortcut.1.bias", "layer2.1.conv1.weight",
    "layer2.1.bn1.weight", "layer2.1.bn1.bias", "layer2.1.conv2.weight",
    "layer2.1.bn2.weight", "layer2.1.bn2.bias", "layer3.0.conv1.weight",
    "layer3.0.bn1.weight", "layer3.0.bn1.bias", "layer3.0.conv2.weight",
    "layer3.0.bn2.weight", "layer3.0.bn2.bias",
    "layer3.0.shortcut.0.weight", "layer3.0.shortcut.1.weight",
    "layer3.0.shortcut.1.bias", "layer3.1.conv1.weight",
    "layer3.1.bn1.weight", "layer3.1.bn1.bias", "layer3.1.conv2.weight",
    "layer3.1.bn2.weight", "layer3.1.bn2.bias", "layer4.0.conv1.weight",
    "layer4.0.bn1.weight", "layer4.0.bn1.bias", "layer4.0.conv2.weight",
    "layer4.0.bn2.weight", "layer4.0.bn2.bias",
    "layer4.0.shortcut.0.weight", "layer4.0.shortcut.1.weight",
    "layer4.0.shortcut.1.bias", "layer4.1.conv1.weight",
    "layer4.1.bn1.weight", "layer4.1.bn1.bias", "layer4.1.conv2.weight",
    "layer4.1.bn2.weight", "layer4.1.bn2.bias", "linear.weight", "linear.bias")
CARRIERS = ("layer4.1.bn2.weight", "layer4.0.bn2.weight", "layer4.0.shortcut.1.weight")
NAMED = ("layer4.0.conv2.weight", "layer4.1.conv2.weight", "linear.weight")
IDX = {n: i for i, n in enumerate(NAMES)}
CI = [IDX[n] for n in CARRIERS]
NI = [IDX[n] for n in NAMED]
LO = -15.0
EPS = 1e-6
MS = 1e-3


def tclass(n):
    if n.endswith("bias"):
        return "bias"
    if n.endswith("conv1.weight") or n.endswith("conv2.weight") or n == "conv1.weight" or n.endswith("shortcut.0.weight"):
        return "conv"
    if n.endswith("bn1.weight") or n.endswith("bn2.weight") or n.endswith("shortcut.1.weight") or n == "bn1.weight":
        return "bnscale"
    if n == "linear.weight":
        return "linear"
    return "other"


CLASS = [tclass(n) for n in NAMES]


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return recs


def load_run(root, arm, seed):
    d = os.path.join(root, "ctd1", "probe_ctd1-%s-s%d" % (arm, seed))
    recs = load(os.path.join(d, "probe.jsonl"))
    numel = np.array(json.load(open(os.path.join(d, "probe_tensor.json")))["param_numels"], float)
    R = {}
    R["step"] = np.array([r["step"] for r in recs])
    R["z"] = np.array([r["z_tensor"] for r in recs])
    R["m"] = np.array([r["m_tensor"] for r in recs])
    b2 = np.array([r["pt_b2"] for r in recs])[:, None]
    R["L"] = b2 * R["m"] + (1 - b2) * R["z"]
    R["zh"] = np.array([r["z_agg"][0] for r in recs])          # (n, ngroups)
    R["mh"] = np.array([r["mom_pre"][0] for r in recs])
    R["Lh"] = b2 * R["mh"] + (1 - b2) * R["zh"]
    R["beta"] = np.array([r["beta"] for r in recs])            # post-update
    R["beta_pre"] = np.array([r["beta_pre"][0] for r in recs])
    R["n_at_lo"] = np.array([r["n_at_lo"] for r in recs])
    R["h_absmax"] = np.array([r["h_absmax"] for r in recs])
    R["numel"] = numel
    return R


def carrying(X, s):
    """smallest prefix of |X|-ranked tensors whose partial sum has sign s and |partial| > remaining sum|X|."""
    o = np.argsort(-np.abs(X))
    part = np.cumsum(X[o])
    rem = np.abs(X).sum() - np.cumsum(np.abs(X[o]))
    ok = (np.sign(part) == s) & (np.abs(part) > rem)
    k = int(np.argmax(ok)) + 1 if ok.any() else len(X)
    return o[:k]


def section_B(runs, key, hkey, label):
    print("\n[B-%s] X = %s, s_agg from harness %s" % (label, key, hkey))
    pooled = collections.Counter()
    carr = collections.Counter()
    sizes = collections.Counter()
    disrec = {}
    per_seed = {}
    for seed, R in sorted(runs.items()):
        X = R[key]
        s_agg = np.sign(R[hkey][:, 0])
        s_sum = np.sign(X.sum(1))
        pos = (X > 0).sum(1)
        neg = (X < 0).sum(1)
        s_maj = np.sign(pos - neg)
        det = (s_agg != 0) & (s_sum == s_agg) & (s_maj != 0)
        dis = det & (s_agg != s_maj)
        pinned = R["n_at_lo"] == 1
        c = dict(det=det.sum(), dis=dis.sum(), pin_det=(det & pinned).sum(), pin_dis=(dis & pinned).sum(),
                 pin_aggneg=(det & pinned & (s_agg < 0)).sum(), unp_det=(det & ~pinned).sum(),
                 unp_dis=(dis & ~pinned).sum(), dis_down=(dis & (s_agg > 0)).sum(),
                 ind_agg=((s_agg == 0) | (s_sum != s_agg)).sum(), ind_maj=(det | dis).size - det.sum() - ((s_agg == 0) | (s_sum != s_agg)).sum())
        named = 0
        for i in np.where(dis)[0]:
            cs = carrying(X[i], s_agg[i])
            sizes[len(cs)] += 1
            for j in cs:
                carr[NAMES[j]] += 1
            named += set(cs.tolist()) <= set(NI)
        c["named"] = named
        per_seed[seed] = c
        disrec[seed] = set(R["step"][dis].tolist())
        for k, v in c.items():
            pooled[k] += v
    p = pooled
    print("  determinate %d (indeterminate: aggregate %d, tie %d); DISAGREE %d; R_T = %.4f"
          % (p["det"], p["ind_agg"], p["ind_maj"], p["dis"], p["dis"] / p["det"]))
    for s, c in per_seed.items():
        print("    seed %d: R_T %.4f (%d/%d)  pinned det %d dis %d  unpinned det %d dis %d"
              % (s, c["dis"] / c["det"], c["dis"], c["det"], c["pin_det"], c["pin_dis"], c["unp_det"], c["unp_dis"]))
    print("  PINNED det %d R_T %.4f agg<0 %.4f | UNPINNED det %d R_T %.4f | PINNED-FRAC %.4f"
          % (p["pin_det"], p["pin_dis"] / p["pin_det"], p["pin_aggneg"] / p["pin_det"], p["unp_det"],
             p["unp_dis"] / p["unp_det"], p["pin_det"] / p["det"]))
    print("  DOWN among DISAGREE %.4f   NAMED-CARRIED %.4f" % (p["dis_down"] / p["dis"], p["named"] / p["dis"]))
    print("  carrying-set sizes: %s" % dict(sorted(sizes.items())))
    print("  carriers (share of DISAGREE):")
    for n, k in carr.most_common(8):
        print("    %-30s %5d  %.3f" % (n, k, k / p["dis"]))
    return disrec


def section_A1(sc, lay):
    print("\n[A1] IS BN-SCALE DOMINANCE A GRADIENT-SCALE ARTEFACT?  median |X_i| over records, per arm")
    numel = sc[18]["numel"]
    for arm, runs, key in (("SCALAR", sc, "L"), ("SCALAR", sc, "z"), ("LAYERWISE", lay, "Lh"), ("LAYERWISE", lay, "zh")):
        X = np.concatenate([R[key] for R in runs.values()])
        pin = np.concatenate([R["n_at_lo"] == 1 if arm == "SCALAR" else np.zeros(len(R["step"]), bool) for R in runs.values()])
        for phase, mask in (("ALL", np.ones(len(X), bool)),) + ((("PINNED", pin), ("UNPINNED", ~pin)) if arm == "SCALAR" else ()):
            med = np.median(np.abs(X[mask]), 0)
            medpe = med / numel
            rank = np.argsort(-med)
            medtensor = np.median(med)
            print("  %s %s %-8s: median-tensor median|X| %.3e; carriers: %s"
                  % (arm, key, phase, medtensor,
                     ", ".join("%s %.3e (x%.0f, rank %d)" % (n.replace("layer4.", "l4."), med[IDX[n]], med[IDX[n]] / medtensor, list(rank).index(IDX[n]) + 1) for n in CARRIERS)))
            print("      NAMED: %s" % ", ".join("%s %.3e (x%.1f, rank %d)" % (n.replace("layer4.", "l4."), med[IDX[n]], med[IDX[n]] / medtensor, list(rank).index(IDX[n]) + 1) for n in NAMED))
            print("      top-6 by median|X|: %s" % ", ".join("%s %.2e" % (NAMES[i], med[i]) for i in rank[:6]))
            rk = np.argsort(-medpe)
            print("      per-ELEMENT median|X|/numel top-6: %s" % ", ".join("%s %.2e" % (NAMES[i], medpe[i]) for i in rk[:6]))
            # class shares of sum |X|
            share = collections.defaultdict(float)
            tot = np.abs(X[mask]).sum(1)
            for c in set(CLASS):
                cols = [i for i in range(62) if CLASS[i] == c]
                share[c] = float(np.median(np.abs(X[mask][:, cols]).sum(1) / tot))
            print("      median share of sum|X| by class: %s" % {k: round(v, 3) for k, v in sorted(share.items())})
    print("\n  SIGN census on scalar PINNED records (fraction of records with L_i > 0 = votes DOWN), by class:")
    L = np.concatenate([R["L"][R["n_at_lo"] == 1] for R in sc.values()])
    Lu = np.concatenate([R["L"][R["n_at_lo"] != 1] for R in sc.values()])
    fpos = (L > 0).mean(0)
    fposu = (Lu > 0).mean(0)
    for c in ("bnscale", "conv", "linear", "bias"):
        cols = [i for i in range(62) if CLASS[i] == c]
        print("    %-8s n=%2d  mean P(L_i>0|pinned) %.3f  [min %.3f max %.3f]   unpinned %.3f"
              % (c, len(cols), fpos[cols].mean(), fpos[cols].min(), fpos[cols].max(), fposu[cols].mean()))
    print("    carriers P(L>0|pinned): %s" % ", ".join("%s %.3f" % (n, fpos[IDX[n]]) for n in CARRIERS))
    print("    NAMED    P(L>0|pinned): %s   unpinned: %s" % (", ".join("%s %.3f" % (n, fpos[IDX[n]]) for n in NAMED),
                                                             ", ".join("%.3f" % fposu[IDX[n]] for n in NAMED)))
    print("    all 62 BN scales that vote DOWN >= 0.5 on pinned: %d of %d" % (sum(fpos[i] >= 0.5 for i in range(62) if CLASS[i] == "bnscale"), CLASS.count("bnscale")))
    print("    #pos among 62 on pinned records: mean %.1f (min %d max %d); #pos of the 22 BN scales: mean %.1f"
          % ((L > 0).sum(1).mean(), (L > 0).sum(1).min(), (L > 0).sum(1).max(),
             (L[:, [i for i in range(62) if CLASS[i] == "bnscale"]] > 0).sum(1).mean()))
    # layerwise: is |L| of the carriers also large under layerwise (per-element and total)?
    Ly = np.concatenate([R["Lh"] for R in lay.values()])
    med = np.median(np.abs(Ly), 0)
    rank = list(np.argsort(-med))
    print("  LAYERWISE arm, rank of each carrier by median|L_j| among 62: %s" % ", ".join("%s %d" % (n, rank.index(IDX[n]) + 1) for n in CARRIERS))


def section_A2(sc, lay):
    print("\n[A2] LAYERWISE COMPANION -- each tensor under its OWN beta (from the harness's own per-group z_agg/mom_pre/beta)")
    print("  columns: tensor | seed | beta_T (terminal, step 49900) | beta_min | pin_share (records with beta_j <= -15+1e-6) | first_pin_step"
          " | P(L_j>0) all records (own vote DOWN) | P(L_j>0) last 50% | net sign votes sum(sign L_j) over 500 records")
    rows = []
    for name in CARRIERS + NAMED:
        j = IDX[name]
        for seed, R in sorted(lay.items()):
            b = R["beta"][:, j]
            Lj = R["Lh"][:, j]
            pin = b <= LO + EPS
            fp = int(R["step"][np.argmax(pin)]) if pin.any() else -1
            half = len(Lj) // 2
            print("    %-28s s%d  beta_T %8.3f  beta_min %8.3f  pin_share %.3f  first_pin %6d  P(L>0) %.3f  last50%% %.3f  net %+d"
                  % (name, seed, b[-1], b.min(), pin.mean(), fp, (Lj > 0).mean(), (Lj[half:] > 0).mean(), int(np.sign(Lj).sum())))
    print("  class summary under LAYERWISE (pooled 3 seeds): class | n | mean beta_T | share of (tensor,seed) with beta_T <= -14 | mean P(L_j>0)")
    for c in ("bnscale", "conv", "linear", "bias"):
        cols = [i for i in range(62) if CLASS[i] == c]
        bT = np.concatenate([R["beta"][-1, cols] for R in lay.values()])
        fp = np.concatenate([(R["Lh"][:, cols] > 0).mean(0) for R in lay.values()])
        print("    %-8s n=%2d  mean beta_T %7.3f  frac beta_T<=-14 %.3f  mean P(L>0) %.3f" % (c, len(cols), bT.mean(), (bT <= -14).mean(), fp.mean()))
    print("  per-tensor terminal beta under layerwise, seed-mean, the 14 layer4 tensors + linear:")
    for j in range(45, 62):
        bT = np.mean([R["beta"][-1, j] for R in lay.values()])
        fp = np.mean([(R["Lh"][:, j] > 0).mean() for R in lay.values()])
        print("    %2d %-28s beta_T %8.3f  P(L>0) %.3f" % (j, NAMES[j], bT, fp))
    # PATH re-derivation for carriers and NAMED
    print("  PATH re-derived (same seed, same step): tensor | agree | opp(sc>0 & lay<0) | P(L<0|sc) | P(L<0|lay)")
    for name in CARRIERS + NAMED:
        j = IDX[name]
        a = o = nsc = nly = n = 0
        for seed in (18, 19, 20):
            Ls = sc[seed]["L"][:, j]
            Ly = lay[seed]["Lh"][:, j]
            a += (np.sign(Ls) == np.sign(Ly)).sum(); o += ((Ls > 0) & (Ly < 0)).sum()
            nsc += (Ls < 0).sum(); nly += (Ly < 0).sum(); n += len(Ls)
        print("    %-28s %.3f  %.3f  %.3f  %.3f" % (name, a / n, o / n, nsc / n, nly / n))


def section_A3(sc):
    print("\n[A3] THE NARROWING -- NAMED convs vs the carriers under SCALAR dynamics, by phase")
    print("  phase bins by step; per bin: P(L_i>0) for NAMED and carriers; median |L_i|; ratio median|L_NAMED|/median|L_carrier|; h_absmax; sum|L| all 62")
    bins = [(0, 8600), (8600, 12000), (12000, 18600), (18600, 30000), (30000, 50000)]
    for lo, hi in bins:
        Ls, hs, pins = [], [], []
        for R in sc.values():
            m = (R["step"] >= lo) & (R["step"] < hi)
            Ls.append(R["L"][m]); hs.append(R["h_absmax"][m]); pins.append(R["n_at_lo"][m])
        L = np.concatenate(Ls); h = np.concatenate(hs); pin = np.concatenate(pins)
        medc = np.median(np.abs(L[:, CI]), 0)
        medn = np.median(np.abs(L[:, NI]), 0)
        print("  steps [%5d,%5d) n=%3d pinned %.2f  h_absmax med %.2e  sum|L| med %.2e  #pos med %.0f"
              % (lo, hi, len(L), pin.mean(), np.median(h), np.median(np.abs(L).sum(1)), np.median((L > 0).sum(1))))
        print("      carriers P(L>0): %s   median|L|: %s" % (" ".join("%.2f" % (L[:, i] > 0).mean() for i in CI), " ".join("%.2e" % v for v in medc)))
        print("      NAMED    P(L>0): %s   median|L|: %s   ratio to carrier-median: %s"
              % (" ".join("%.2f" % (L[:, i] > 0).mean() for i in NI), " ".join("%.2e" % v for v in medn),
                 " ".join("%.3f" % (v / np.median(medc)) for v in medn)))
        # do NAMED tensors' L_i have the sign of s_agg (i.e. also vote down) in this phase?
        sagg = np.sign(np.concatenate([R["Lh"][(R["step"] >= lo) & (R["step"] < hi), 0] for R in sc.values()]))
        print("      NAMED sign == s_agg: %s ; carriers sign == s_agg: %s"
              % (" ".join("%.2f" % (np.sign(L[:, i]) == sagg).mean() for i in NI), " ".join("%.2f" % (np.sign(L[:, i]) == sagg).mean() for i in CI)))
    # among DISAGREE records: which mechanism removes convs from the carrying set
    print("  On PINNED records: NAMED tensors with L_i > 0 (same sign as aggregate) but outside the size-3 carrying set -- magnitude, not sign, is why they are absent:")
    for R in sc.values():
        pass
    L = np.concatenate([R["L"][R["n_at_lo"] == 1] for R in sc.values()])
    c3 = np.abs(L[:, CI]).sum(1)
    rest = np.abs(L).sum(1) - c3
    print("    pinned records: median sum|L| carriers %.3e vs rest-59 %.3e; carriers > rest on %.4f of records; NAMED sum|L| / carriers sum|L| median %.4f"
          % (np.median(c3), np.median(rest), (c3 > rest).mean(), np.median(np.abs(L[:, NI]).sum(1) / c3)))
    print("    pinned records: P(all three NAMED have L_i>0) %.3f ; P(all three carriers L_i>0) %.3f" % (((L[:, NI] > 0).all(1)).mean(), ((L[:, CI] > 0).all(1)).mean()))
    # m-trajectory of convs: sign flips
    print("  per-tensor m_i trajectory sign for NAMED (seed 18): number of sign changes across 500 records, and P(m>0) pre-pin / pinned")
    R = sc[18]
    pin = R["n_at_lo"] == 1
    for name in NAMED + CARRIERS:
        j = IDX[name]
        m = R["m"][:, j]
        flips = int((np.sign(m[1:]) != np.sign(m[:-1])).sum())
        print("    %-28s flips %3d  P(m>0|pre-pin) %.3f  P(m>0|pinned) %.3f  median|m| pre %.2e pinned %.2e"
              % (name, flips, (m[~pin] > 0).mean(), (m[pin] > 0).mean(), np.median(np.abs(m[~pin])), np.median(np.abs(m[pin]))))


def section_A5(sc, cru):
    print("\n[A5] SEED INDEPENDENCE -- scalar beta trajectories, ctd1 s18-20 and cru1 s15-17 (same cell)")
    allr = dict(sc); allr.update(cru)
    for seed, R in sorted(allr.items()):
        b = R["beta"][:, 0]
        pin = R["n_at_lo"] == 1
        first = int(R["step"][np.argmax(pin)]) if pin.any() else -1
        pk = int(np.argmax(b))
        print("    seed %d: beta0 %.3f peak %.3f at step %d, first n_at_lo==1 at step %d, floor share %.4f, pinned records %d, beta_T %.3f"
              % (seed, b[0], b[pk], R["step"][pk], first, pin.mean(), pin.sum(), b[-1]))
    seeds = sorted(allr)
    print("  max |beta_s(t) - beta_s'(t)| over 500 records, and mean |diff|:")
    for a in seeds:
        for b in seeds:
            if a < b:
                d = np.abs(allr[a]["beta"][:, 0] - allr[b]["beta"][:, 0])
                print("    s%d vs s%d: max %.3f  mean %.3f  at step %d" % (a, b, d.max(), d.mean(), allr[a]["step"][np.argmax(d)]))
    print("  sign(L) agreement across ctd1 seeds at the same record (harness aggregate):")
    for a in (18, 19, 20):
        for b in (18, 19, 20):
            if a < b:
                sa, sb = np.sign(sc[a]["Lh"][:, 0]), np.sign(sc[b]["Lh"][:, 0])
                print("    s%d vs s%d: %.4f" % (a, b, (sa == sb).mean()))
    print("  A trajectory moves by exactly +-ms per step; the ascent length in steps = (peak - beta0)/ms:")
    for seed in (18, 19, 20):
        b = sc[seed]["beta"][:, 0]
        print("    seed %d: (peak-beta0)/ms = %.0f steps; descent to floor = (peak - (-15))/ms = %.0f steps; peak step + descent = %.0f"
              % (seed, (b.max() - b[0]) / MS, (b.max() + 15) / MS, sc[seed]["step"][np.argmax(b)] + (b.max() + 15) / MS))


def main():
    root = sys.argv[1]
    sc = {s: load_run(root, "sc", s) for s in (18, 19, 20)}
    lay = {s: load_run(root, "lay", s) for s in (18, 19, 20)}
    cru = {}
    for s in (15, 16, 17):
        d = os.path.join(root, "cru1", "probe_cru1-sc-m1e-3-a1e-6-s%d" % s)
        if os.path.isfile(os.path.join(d, "probe.jsonl")):
            recs = load(os.path.join(d, "probe.jsonl"))
            cru[s] = {"step": np.array([r["step"] for r in recs]), "beta": np.array([r["beta"] for r in recs]),
                      "n_at_lo": np.array([r["n_at_lo"] for r in recs])}
    print("loaded: scalar %s layerwise %s cru1 %s; records %s" % (sorted(sc), sorted(lay), sorted(cru), [len(R["step"]) for R in sc.values()]))
    # sanity: n_at_lo==1 vs beta<=-15+eps agreement in scalar
    for s, R in sc.items():
        print("  seed %d: n_at_lo==1 on %d records; beta_post<=-15+1e-6 on %d; disagree %d" % (s, (R["n_at_lo"] == 1).sum(), (R["beta"][:, 0] <= LO + EPS).sum(), ((R["n_at_lo"] == 1) != (R["beta"][:, 0] <= LO + EPS)).sum()))
    disL = section_B(sc, "L", "Lh", "PRIMARY-L")
    disz = section_B(sc, "z", "zh", "SECONDARY-z")
    section_A1(sc, lay)
    section_A2(sc, lay)
    section_A3(sc)
    section_A5(sc, cru)
    print("\n  DISAGREE-record step sets (PRIMARY) across seeds: Jaccard s18/s19 %.3f s18/s20 %.3f s19/s20 %.3f"
          % tuple(len(disL[a] & disL[b]) / len(disL[a] | disL[b]) for a, b in ((18, 19), (18, 20), (19, 20))))


if __name__ == "__main__":
    main()
