#!/usr/bin/env python3
"""cvi1_attack_indep.py -- an INDEPENDENT re-derivation of `cvi1`, sharing NO
code with analysis/cVI1_vggiso_identity_score.py (nothing imported from it,
no function copied from it; every regex, reader and statistic written afresh).

Usage:  python3 analysis/cvi1_attack_indep.py <runsdir> [--prefix cvi1]

<runsdir> holds the batch's `.out` files and `<prefix>-PARTITION-MANIFEST.txt`;
probe records are read from <runsdir>/<prefix>/probe_<run>/.  `--prefix cvg1`
re-runs the same code over cvg1 (arms k01/kL only) as a calibration: it must
re-derive 194's published cvg1 numbers before its cvi1 numbers mean anything.

It reads ONLY raw `.out` files, the batch manifest and the runs' own probe
JSON.  It never reads results/all_runs.csv, the CSV `plateau` column, or
best_test.  plateau5 = mean TEST accuracy over epochs 95..99 (the last five of
100), TRAIN beside it.

Sections: (A) per-run soundness + ARGS/ENV/PROBE_TENSOR; (B) per-run and
per-arm plateau5 / train5, seed ranges; (C) contrasts; (D) anchors vs cvg1;
(E) block arity from block_sizes.json; (F) beta trajectories; (G) the carrier
nomination re-derived on THIS batch's OWN scalar-arm pinned records.
"""
import json
import math
import os
import re
import statistics as st
import sys

STEPS_PER_EPOCH = 500.0          # 50,000 meta steps / 100 epochs (manifest META_STEPS)
LO = -15.0
PIN_EPS = 1e-3

EPOCH_LINE = re.compile(r"^Epoch (\d+), Train Accuracy: ([\d.]+) %, Test Accuracy: ([\d.]+) %")
RUN_FILE = r"^%s-(k01|kL|ISO|CTL)-s(\d\d)-(\d+)\.out$"


def die(msg):
    print("FATAL: " + msg)
    sys.exit(2)


def load_manifest(path):
    names, numel = {}, {}
    for ln in open(path):
        t = ln.split()
        if len(t) == 4 and t[0] == "TENSOR":
            names[int(t[1])] = t[2]
            numel[int(t[1])] = int(t[3])
    return names, numel


def parse_out(path):
    ep = {}
    dup = 0
    args = env = pt = None
    minutes = None
    done = tb = False
    for ln in open(path, errors="replace"):
        ln = ln.rstrip("\n")
        m = EPOCH_LINE.match(ln)
        if m:
            e = int(m.group(1))
            if e in ep:
                dup += 1
            ep[e] = (float(m.group(2)), float(m.group(3)))
        elif ln.startswith("ARGS: "):
            args = ln[6:].split()
        elif ln.startswith("ENV: "):
            env = ln[5:].split()
        elif ln.startswith("PROBE_TENSOR: "):
            pt = ln
        elif ln.strip() == "RUN_DONE":
            done = True
        elif "Traceback" in ln:
            tb = True
        else:
            mm = re.match(r"^(\d+)\s+minutes$", ln.strip())
            if mm:
                minutes = int(mm.group(1))
    return dict(ep=ep, dup=dup, args=args, env=env, pt=pt, done=done, tb=tb, minutes=minutes)


def plateau(ep, which):
    k = 1 if which == "test" else 0
    tail = [ep[e][k] for e in range(95, 100) if e in ep]
    return sum(tail) / 5.0 if len(tail) == 5 else None


def argdict(tokens):
    d, i = {}, 0
    while tokens and i < len(tokens):
        if tokens[i].startswith("--"):
            key = tokens[i]
            val = tokens[i + 1] if i + 1 < len(tokens) and not tokens[i + 1].startswith("--") else ""
            d.setdefault(key, []).append(val)
            i += 2 if val != "" else 1
        else:
            i += 1
    return d


def main():
    if len(sys.argv) < 2:
        die("usage: cvi1_attack_indep.py <runsdir> [--prefix cvi1]")
    R = sys.argv[1]
    P = "cvi1"
    if "--prefix" in sys.argv:
        P = sys.argv[sys.argv.index("--prefix") + 1]
    arms = ["k01", "kL", "ISO", "CTL"] if P == "cvi1" else ["k01", "kL"]
    names, numel = load_manifest(os.path.join(R, P + "-PARTITION-MANIFEST.txt"))
    print("manifest: %d tensors, %d params" % (len(names), sum(numel.values())))

    rx = re.compile(RUN_FILE % P)
    runs = {}
    for fn in sorted(os.listdir(R)):
        m = rx.match(fn)
        if m:
            key = (m.group(1), int(m.group(2)))
            if key in runs:
                print("  WARNING duplicate (arm,seed) %s: %s and %s" % (key, runs[key]["file"], fn))
            o = parse_out(os.path.join(R, fn))
            o["file"], o["job"] = fn, m.group(3)
            runs[key] = o
    seeds = sorted(set(s for _, s in runs))
    print("runs found: %d   seeds %s" % (len(runs), seeds))

    # ---------------- (A) soundness + lines ----------------
    print("\n(A) SOUNDNESS AND THE THREE CONFIG LINES")
    envset, ptset = set(), {}
    bad = 0
    for (a, s), o in sorted(runs.items()):
        eps = sorted(o["ep"])
        ok_ep = eps == list(range(100)) and o["dup"] == 0
        ad = argdict(o["args"] or [])
        rep = [k for k, v in ad.items() if len(v) > 1]
        sg = (ad.get("--stepsize-groups") or ["?"])[0]
        sd_ = (ad.get("--seed") or ["?"])[0]
        rn = (ad.get("--run-name") or ["?"])[0]
        envs = " ".join(t for t in (o["env"] or []) if not t.startswith("PROBE_DIR="))
        envset.add(envs)
        ptype = re.search(r"type=(\w+)", o["pt"] or "")
        ptype = ptype.group(1) if ptype else None
        ptset[ptype] = ptset.get(ptype, 0) + 1
        good = ok_ep and o["done"] and not o["tb"] and not rep and sd_ == str(s) \
            and rn == "%s-%s-s%d" % (P, a, s)
        bad += 0 if good else 1
        print("  %-14s job %s  epochs %3d  RUN_DONE %s  tb %s  repflags %s  seed %s  sg %-28s pt %-9s %s min  %s"
              % ("%s-s%d" % (a, s), o["job"], len(eps), o["done"], o["tb"], rep or "-", sd_, sg, ptype,
                 o["minutes"], "ok" if good else "BAD"))
    print("  distinct ENV lines (PROBE_DIR masked): %d" % len(envset))
    for e in envset:
        print("     " + e)
    print("  PROBE_TENSOR types: %s" % ptset)
    print("  runs failing soundness: %d" % bad)

    # ---------------- (B) levels ----------------
    print("\n(B) LEVELS -- plateau5 over epochs 95..99, TEST and TRAIN")
    lv, tr = {}, {}
    for a in arms:
        lv[a], tr[a] = [], []
        for s in seeds:
            o = runs.get((a, s))
            if o is None:
                continue
            pt_, pr_ = plateau(o["ep"], "test"), plateau(o["ep"], "train")
            if pt_ is None:
                continue
            lv[a].append(pt_)
            tr[a].append(pr_)
            print("  %-4s s%d  TEST %.4f  TRAIN %.4f   e99 test %.2f" % (a, s, pt_, pr_, o["ep"][99][1]))
    M, T = {}, {}
    for a in arms:
        if not lv[a]:
            print("  %-4s NONE" % a)
            continue
        M[a], T[a] = st.mean(lv[a]), st.mean(tr[a])
        print("  %-4s MEAN TEST %.4f  sd %.4f  range %.4f (min %.4f max %.4f)  n=%d   MEAN TRAIN %.4f  range %.4f"
              % (a, M[a], st.stdev(lv[a]) if len(lv[a]) > 1 else float("nan"),
                 max(lv[a]) - min(lv[a]), min(lv[a]), max(lv[a]), len(lv[a]), T[a],
                 max(tr[a]) - min(tr[a])))

    # test trajectory every 10 epochs, arm means
    print("\n  TEST trajectory (arm mean, every 10 epochs):")
    for a in arms:
        row = []
        for e in list(range(0, 100, 10)) + [99]:
            v = [runs[(a, s)]["ep"][e][1] for s in seeds if (a, s) in runs and e in runs[(a, s)]["ep"]]
            row.append("e%d=%.1f" % (e, st.mean(v)) if v else "e%d=?" % e)
        print("   %-4s %s" % (a, " ".join(row)))

    if not all(len(lv.get(a, [])) == 3 for a in arms):
        print("\nINCOMPLETE COVERAGE -- no contrast is computed.")
        return 1

    # ---------------- (C) contrasts ----------------
    ss = sum(sum((x - st.mean(v)) ** 2 for x in v) for v in lv.values())
    df = sum(len(v) - 1 for v in lv.values())
    sig_in = math.sqrt(ss / df)
    sig_prior = 0.586232      # the scorer's frozen floor, typed in (not imported) for the SE comparison
    sig_use = max(sig_in, sig_prior)
    se = sig_use * math.sqrt(2.0 / 3.0)
    se_in = sig_in * math.sqrt(2.0 / 3.0)
    print("\n(C) CONTRASTS, all within batch")
    print("  SIGMA_INBATCH %.6f (df %d)   SE_inbatch %.6f   SE at max(prior 0.586232, inbatch) %.6f"
          % (sig_in, df, se_in, se))
    if P == "cvi1":
        C = [("DELTA_ID = ISO-CTL", M["ISO"] - M["CTL"], T["ISO"] - T["CTL"]),
             ("D_ISO    = ISO-k01", M["ISO"] - M["k01"], T["ISO"] - T["k01"]),
             ("D_CTL    = CTL-k01", M["CTL"] - M["k01"], T["CTL"] - T["k01"]),
             ("D_GAP    = kL-k01 ", M["kL"] - M["k01"], T["kL"] - T["k01"]),
             ("ISO-kL           ", M["ISO"] - M["kL"], T["ISO"] - T["kL"])]
    else:
        C = [("D_GAP    = kL-k01 ", M["kL"] - M["k01"], T["kL"] - T["k01"])]
    for nm, d, dt in C:
        print("  %s  TEST %+.4f pp = %+.2f SE (%+.2f SE_inbatch)   TRAIN %+.4f pp"
              % (nm, d, d / se, d / se_in, dt))
    if P == "cvi1":
        dgap = M["kL"] - M["k01"]
        print("  RECOVERY     D_ISO/D_GAP = %.4f   RECOVERY_CTL D_CTL/D_GAP = %.4f (descriptive)"
              % ((M["ISO"] - M["k01"]) / dgap, (M["CTL"] - M["k01"]) / dgap))
        # bars applied HERE, by this parser, in the registered order
        d_iso, d_id, d_ctl = M["ISO"] - M["k01"], M["ISO"] - M["CTL"], M["CTL"] - M["k01"]
        rng_ok = all(max(v) - min(v) <= 5.0 for v in lv.values())
        floor_ok = max(M.values()) >= 15.0
        if not floor_ok:
            br = "HARNESS-UNSOUND"
        elif not rng_ok:
            br = "UNRESOLVED-DIVERGED"
        elif d_iso < 10.0:
            br = "NO-RESCUE-VGG"
        elif d_id >= 15.0:
            br = "IDENTITY-OPERATIVE-VGG"
        elif d_id <= -15.0:
            br = "CONTROL-DOMINATES"
        elif d_ctl >= 10.0 and abs(d_id) <= 2.0:
            br = "CLASS-OPERATIVE-VGG"
        else:
            br = "IDENTITY-ATTENUATED"
        print("  BRANCH (bars applied by THIS parser: rescue 10, identity 15, null 2, diverge 5, floor 15): %s" % br)

    # ---------------- (D) anchors ----------------
    print("\n(D) ANCHORS vs cvg1 (194: k01 35.0173, kL 66.2860, D +31.2687) -- BETWEEN-BATCH, descriptive")
    for a, ref in (("k01", 35.0173), ("kL", 66.2860)):
        print("  %-4s cvi1 %.4f  cvg1 %.4f  diff %+.4f pp = %+.2f SE" % (a, M[a], ref, M[a] - ref, (M[a] - ref) / se))
    print("  D_GAP cvi1 %+.4f  cvg1 +31.2687  diff %+.4f pp" % (M["kL"] - M["k01"], M["kL"] - M["k01"] - 31.2687))

    # ---------------- (E) arity ----------------
    print("\n(E) BLOCK ARITY from each run's own block_sizes.json")
    pdirs = {}
    for (a, s) in sorted(runs):
        pd = os.path.join(R, P, "probe_%s-%s-s%d" % (P, a, s))
        pdirs[(a, s)] = pd
        try:
            bs = json.load(open(os.path.join(pd, "block_sizes.json")))
            print("  %-9s type %-10s m=%d  n_b[:4]=%s  sum %d"
                  % ("%s-s%d" % (a, s), bs.get("stepsize_type"), len(bs["n_b"]), bs["n_b"][:4], sum(bs["n_b"])))
        except Exception as ex:
            print("  %-9s block_sizes.json unreadable: %s" % ("%s-s%d" % (a, s), ex))

    # ---------------- (F) beta trajectories ----------------
    print("\n(F) BETA TRAJECTORIES (epoch = step/500; pinned := beta <= -15 + 1e-3)")
    recs_by = {}
    for (a, s), pd in sorted(pdirs.items()):
        fp = os.path.join(pd, "probe.jsonl")
        if not os.path.exists(fp):
            print("  %s-s%d: no probe.jsonl" % (a, s))
            continue
        recs = []
        for ln in open(fp):
            ln = ln.strip()
            if ln:
                try:
                    recs.append(json.loads(ln))
                except ValueError:
                    pass
        recs_by[(a, s)] = recs
        nb = len(recs[0]["beta"])
        print("  %s-s%d: %d records, len(beta)=%d, last step %d" % (a, s, len(recs), nb, recs[-1]["step"]))
        for g in range(nb if nb <= 2 else 0):
            b = [r["beta"][g] for r in recs]
            first = next((i for i, x in enumerate(b) if x <= LO + PIN_EPS), None)
            pk = max(range(len(b)), key=lambda i: b[i])
            npin = sum(1 for x in b if x <= LO + PIN_EPS)
            # pinned from first pin to end?
            stays = first is not None and all(x <= LO + PIN_EPS for x in b[first:])
            lab = ("the single beta" if nb == 1 else ("beta[0] complement (25 tensors)" if g == 0
                                                       else "beta[1] isolated (1 tensor)"))
            print("     %-32s start %.4f  peak %.4f @ep %.1f  terminal %.4f  first pin %s  pinned %d/%d  stays-pinned %s"
                  % (lab, b[0], b[pk], recs[pk]["step"] / STEPS_PER_EPOCH, b[-1],
                     ("ep %.1f" % (recs[first]["step"] / STEPS_PER_EPOCH)) if first is not None else "never",
                     npin, len(b), stays))
        if nb > 2:
            fin = recs[-1]["beta"]
            pinned = [i + 1 for i in range(nb) if fin[i] <= LO + PIN_EPS]
            free = [i + 1 for i in range(nb) if fin[i] > LO + PIN_EPS]
            print("     layerwise terminal: %d pinned %s ; %d free, max beta %.3f"
                  % (len(pinned), pinned, len(free), max(fin)))
            print("     bn8(23) terminal %.3f  bn7(20) terminal %.3f" % (fin[22], fin[19]))

    # ---------------- (G) the nomination on THIS batch's k01 ----------------
    print("\n(G) CARRIER NOMINATION on %s's OWN scalar-arm PINNED records" % P)
    print("    L_i = b2*m_i + (1-b2)*z_i, b2 from the record's pt_b2; pinned := beta[0] <= -15+1e-3")
    SETS = [("{23}", [23]), ("{20}", [20]), ("{14}", [14]), ("{17}", [17]),
            ("{14,17,20}", [14, 17, 20]), ("{22}", [22]), ("{25}", [25]), ("{20,23}", [20, 23])]
    shares, fracpos, flips, doms, sgnpos = [], [], {k: [] for k, _ in SETS}, [], []
    for s in seeds:
        recs = recs_by.get(("k01", s))
        if not recs:
            continue
        pin = [r for r in recs if r["beta"][0] <= LO + PIN_EPS]
        # cross-check the pin definition against the record's own n_at_lo
        alt = [r for r in recs if r.get("n_at_lo") == r.get("n_beta") and r.get("n_beta", 0) > 0]
        n = len(names)
        mass = [0.0] * n
        pos = [0] * n
        spos = 0
        fl = {k: 0 for k, _ in SETS}
        dom = 0
        for r in pin:
            b2 = r["pt_b2"]
            L = [b2 * r["m_tensor"][i] + (1.0 - b2) * r["z_tensor"][i] for i in range(n)]
            tot = sum(L)
            spos += tot > 0
            for i in range(n):
                mass[i] += abs(L[i])
                pos[i] += L[i] > 0
            for k, idx in SETS:
                rem = tot - sum(L[j - 1] for j in idx)
                if (rem > 0) != (tot > 0):
                    fl[k] += 1
            if abs(L[22]) > abs(tot - L[22]):
                dom += 1
        tm = sum(mass)
        share = [x / tm for x in mass]
        shares.append(share)
        fracpos.append([p / len(pin) for p in pos])
        sgnpos.append(spos / len(pin))
        doms.append(dom / len(pin))
        for k, _ in SETS:
            flips[k].append(fl[k] / len(pin))
        order = sorted(range(n), key=lambda i: -share[i])
        print("  s%d: pinned %d/%d (n_at_lo==n_beta gives %d); sum L > 0 on %.4f; top-5 %s"
              % (s, len(pin), len(recs), len(alt), spos / len(pin),
                 ", ".join("%d %s %.4f" % (i + 1, names[i + 1], share[i]) for i in order[:5])))
    if shares:
        k = len(shares)
        avg = [sum(sh[i] for sh in shares) / k for i in range(len(names))]
        fp = [sum(f[i] for f in fracpos) / k for i in range(len(names))]
        order = sorted(range(len(names)), key=lambda i: -avg[i])
        rank = {i: r + 1 for r, i in enumerate(order)}
        print("  MEAN OVER %d SEEDS -- share of sum mean|L_i|:" % k)
        for i in order[:6]:
            print("     rank %2d idx %2d %-14s numel %9d share %.4f  frac L>0 %.4f"
                  % (rank[i], i + 1, names[i + 1], numel[i + 1], avg[i], fp[i]))
        for idx in (20, 14, 17):
            i = idx - 1
            print("     rank %2d idx %2d %-14s numel %9d share %.4f  frac L>0 %.4f"
                  % (rank[i], idx, names[idx], numel[idx], avg[i], fp[i]))
        print("  sum_i L_i > 0 (beta DOWN) on %s of pinned records per seed" % ["%.4f" % x for x in sgnpos])
        print("  |L_23| > |remainder| (domination) per seed: %s" % ["%.4f" % x for x in doms])
        print("  REMAINDER-SIGN FLIP when the set is removed (per seed; mean):")
        for kk, _ in SETS:
            print("     %-12s %s  mean %.4f" % (kk, ["%.4f" % x for x in flips[kk]], sum(flips[kk]) / k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
