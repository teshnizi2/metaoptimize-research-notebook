#!/usr/bin/env python3
# =============================================================================
# cVK2_invariance_check.py -- THE O2 ACCEPTANCE TEST (CORRECTIONS 208).
#
# On the SAME 27 synthetic cvk1 runs -- 207's near-bar set: the frozen
# analysis/cVK1_smoke_gen.py with ONE substitution, "k22": 64.0 -> 61.2 --
# score them with the FROZEN analysis/cVK1_vggcut_score.py AND the successor
# analysis/cVK2_vggcut_score.py against every corpus given, plus two corpora
# built from the first one by adding 3 INVENTED VGG11_bn_c100 std-cell rows
# (207's mk_stress.py recipe, prefix zstress-).  Prints both FINAL lines side
# by side, and each scorer's SIGMA_USED at full precision.
#
# PASSES (exit 0) iff:
#   1. cVK2 emits ONE FINAL line across every corpus;
#   2. cVK1 emits a DIFFERENT FINAL on at least one stress corpus (the flip
#      O2 exists to remove) -- so the test is not vacuous;
#   3. on every NON-stress corpus |SIGMA_USED(cVK2) - SIGMA_USED(cVK1)| < 1.1e-7.
#
# Every accuracy is INVENTED; nothing here is evidence about the real batch.
# Neither scorer is edited; both are imported by path and run unedited.
#
# usage: python3 analysis/cVK2_invariance_check.py <workdir> <base.csv> [more.csv ...]
#        (<base.csv> seeds the stress corpora; 208 used the 2,809-row corpus)
# =============================================================================
import contextlib
import csv
import importlib.util
import io
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stress(src, dst, vals):
    """207's mk_stress.py: cvi1's scalar row as template, 3 invented plateau5 values"""
    rows = list(csv.DictReader(open(src)))
    hdr = list(rows[0].keys())
    tmpl = next(r for r in rows if r["run"].startswith("cvi1-") and r["granularity"] == "scalar")
    new = []
    for i, v in enumerate(vals):
        r = dict(tmpl)
        r.update(run="zstress-k01-s%d" % (90 + i), job_id=str(9990000 + i), seed=str(90 + i),
                 plateau5="%.4f" % v, plateau="", best_test="", final_test="", dup_group="")
        new.append(r)
    w = csv.DictWriter(open(dst, "w", newline=""), fieldnames=hdr)
    w.writeheader()
    for r in rows + new:
        w.writerow(r)


def run_score(m, runs, csvpath):
    del m.GATE_FAIL[:]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.score(runs, csvpath, "")
    out = buf.getvalue()
    fin = [ln for ln in out.splitlines() if ln.startswith("FINAL:")][-1]
    ps = re.search(r"PEAK_SET (\[.*?\])", out)
    return rc, fin, (ps.group(1) if ps else "-")


def sigma_exact(m, runs, csvpath, v2):
    """SIGMA_USED at full precision, by each scorer's OWN composition"""
    got = m.read_runs(runs)
    arm_v = dict((a, [got[(a, s)]["plateau5"] for s in m.SEEDS]) for a in m.ARMS)
    ss = sum(sum((x - m.mean(v)) ** 2 for x in v) for v in arm_v.values())
    df = sum(len(v) - 1 for v in arm_v.values())
    s_in = math.sqrt(ss / df)
    if v2:
        return m.compose_sigma(s_in)[1], s_in
    cands = [m.SIGMA_PRIOR, s_in]                # cVK1 score() lines 851-862, verbatim logic
    rows = m.read_corpus(csvpath)
    if rows:
        sn = m.pooled_sigma(rows, lambda r: r["network"] == "ResNet18_c100"
                            and r["dataset"] == "CIFAR100")[0]
        sv = m.pooled_sigma(rows, lambda r: r["network"] == m.NET)[0]
        cands += [x for x in (sn, sv) if x]
    return max(cands), s_in


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__ or "usage: cVK2_invariance_check.py <workdir> <base.csv> [more.csv ...]")
    work, base, more = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(work, exist_ok=True)
    v1 = load("cvk1_frozen", os.path.join(HERE, "cVK1_vggcut_score.py"))
    v2 = load("cvk2_successor", os.path.join(HERE, "cVK2_vggcut_score.py"))

    # ---- 207's near-bar runs, regenerated from the frozen generator -------------
    src = open(os.path.join(HERE, "cVK1_smoke_gen.py")).read()
    assert src.count('"k22": 64.0') == 1, "the generator's k22 literal moved"
    gen = os.path.join(work, "gen_near.py")
    open(gen, "w").write(src.replace('"k22": 64.0', '"k22": 61.2'))
    man = os.path.join(work, "manifest.txt")
    open(man, "w").write(v2._e_manifest_text())
    runs = os.path.join(work, "runs_near")
    subprocess.check_call([sys.executable, gen, runs, man])

    # ---- the corpora --------------------------------------------------------------
    corpora = [(os.path.basename(p), p, False) for p in [base] + more]
    for vals in ((35.0, 35.5, 38.5), (35.0, 35.0, 41.0)):
        p = os.path.join(work, "stress_%s.csv" % "_".join("%.1f" % v for v in vals))
        stress(base, p, vals)
        corpora.append(("%s +3 VGG %s" % (os.path.basename(base), "/".join("%.1f" % v for v in vals)),
                        p, True))

    print("SAME 27 RUNS (%s), INVENTED NUMBERS.  cVK1 = frozen, cVK2 = O2 successor." % runs)
    fins2, flips, dmax = set(), 0, 0.0
    for nm, p, is_stress in corpora:
        r1 = run_score(v1, runs, p)
        r2 = run_score(v2, runs, p)
        s1, s_in = sigma_exact(v1, runs, p, False)
        s2, _ = sigma_exact(v2, runs, p, True)
        fins2.add(r2[1])
        if is_stress and r1[1] != r2[1].replace("SIGMA-NARROW-PINNED", "SIGMA-NARROW-LIVE"):
            flips += 1
        if not is_stress:
            dmax = max(dmax, abs(s2 - s1))
        print("\n== %s%s" % (nm, "   [STRESS: 3 INVENTED ROWS]" if is_stress else ""))
        print("   in-batch sigma %.10f" % s_in)
        print("   cVK1 SIGMA_USED %.13f  PEAK_BAR %.4f  PEAK_SET %-12s rc %s"
              % (s1, 2 * s1 * math.sqrt(2 / 3.0), r1[2], r1[0]))
        print("   cVK2 SIGMA_USED %.13f  PEAK_BAR %.4f  PEAK_SET %-12s rc %s"
              % (s2, 2 * s2 * math.sqrt(2 / 3.0), r2[2], r2[0]))
        print("   |cVK2 - cVK1| sigma = %.3e" % abs(s2 - s1))
        print("   cVK1 %s" % r1[1])
        print("   cVK2 %s" % r2[1])
    floor_only = max(v2.SIGMA_PRIOR, s_in)
    print("\nfloor-only composition (O2 read as 'drop them'): %.13f; the NARROW pin exceeds it by %.3e"
          % (floor_only, v2.SIGMA_NARROW_PIN - floor_only))
    ok1, ok2, ok3 = len(fins2) == 1, flips >= 1, dmax < 1.1e-7
    print("\n1. cVK2 emits ONE FINAL across %d corpora: %s" % (len(corpora), "PASS" if ok1 else "FAIL"))
    print("2. cVK1's FINAL moves on %d of 2 stress corpora (the test is not vacuous): %s"
          % (flips, "PASS" if ok2 else "FAIL"))
    print("3. max |sigma cVK2 - cVK1| on the non-stress corpora = %.3e < 1.1e-7: %s"
          % (dmax, "PASS" if ok3 else "FAIL"))
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(main())
