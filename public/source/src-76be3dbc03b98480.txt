"""INERTNESS AND BITE OF PATCH_COMPHOLD OVER A SHORT *REAL* RUN, FOR cvt6's EXACT STRINGS.  CORRECTIONS 242.

The REAL code path of a cvt6 run -- train.py's own parse_args, build_network (PlainNet18_c100), build_optimizer,
load_data (CIFAR-100, AUGMENT=1), the loss, HF.step with BETA_CLIP, PROBE and PROBE_TENSOR -- for --steps optimizer
steps on a GPU, from TWO trees:

  --tree-pre   cvt4's tree ($WS/harness_cvt4/cifar10, HF.py 84b345ad... = the cvt6 tree's HF.py.pre_comphold),
               used READ-ONLY (PYTHONDONTWRITEBYTECODE=1)
  --tree-post  the cvt6 tree (PATCH_COMPHOLD on top of it)

It IMPORTS cvt1's registered tests/test_voteweight_realrun.py UNEDITED and reuses its child process (`--child`), its
`run_variant` / `same` / `chk`; BETA_HOLD and COMP_HOLD are passed through the environment run_variant copies.  The
strings, witnesses and schedules come from analysis/cVT6_complementpath_score.py, so the test and the scorer cannot
disagree.

  RR0  DETERMINISM CONTROL: the unpatched (cvt4) tree run twice -> beta at every step, loss, probe.jsonl bytes, every
       final parameter and buffer identical (else the bitwise claims cannot be tested; FAIL).
  RR1  OFF: COMP_HOLD unset and COMP_HOLD empty, on k01, HEAD, HOLDLOW and HOLDHIGH (each with its own BETA_HOLD) ==
       cvt4's tree BITWISE (beta every step, loss every step, probe.jsonl bytes, every final parameter and buffer); each
       printed exactly one `COMP_HOLD: off`, its arm's BETA_HOLD witness (cvt4's line) and `VOTE_W: off`.
  RR2  THE EXACT STRINGS ON THE REAL PATH (HIGHHEADPATH, LOWMUTEPATH, LOWHEADPATH) and two test-only strings whose
       complement schedule departs from the natural rise inside the run (LOW + tri:100; HIGH + rec:cvt6_testdown):
       exactly one BETA_HOLD and one COMP_HOLD line == the scorer's witnesses; beta[0] AND beta[1] after EVERY step ==
       the registered schedules; at every probe record the scorer's own G-BITE halves hold (both groups on schedule
       in beta and beta_pre; bh_* and ch_* audit: n == step + 2, nat == Lion recomputed, held == beta, active
       non-decreasing) -- bite_check() itself for the registered arms.
  RR3  NON-VACUITY: the two test-only strings' complement departs from cvt4's same-BETA_HOLD run by > 0.05 from step 150
       with ch_active > 0; the three registered strings' complement beta and ch_active are REPORTED against cvt4's
       HOLDLOW / HOLDHIGH (in 300 steps the free complement rises at the maximum rate, as both registered schedules
       do, so a registered string can coincide with Lion up to float32 accumulation; non-gating).
  RR4  OPEN LOOP: on the three registered strings the applied beta vector after every step is BITWISE the schedules --
       independent of the run's z (a function of n only), which RR2 proves on the real path.

RUN (on a GPU node; ~15 min):
  python3 tests/test_comphold_realrun.py --tree-pre $WS/harness_cvt4/cifar10 --tree-post $WS/harness_cvt6/cifar10 \\
      --steps 300 --every 10 --seed 96 --work /tmp/cvt6_realrun
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RR = _load("test_voteweight_realrun", os.path.join(HERE, "test_voteweight_realrun.py"))
SC = _load("cvt6_scorer", os.path.join(REPO, "analysis", "cVT6_complementpath_score.py"))
chk, same, run_variant = RR.chk, RR.same, RR.run_variant


def run_h(a, tag, tree, spec, bh, ch):
    old = dict((k, os.environ.pop(k, None)) for k in ("BETA_HOLD", "COMP_HOLD"))
    if bh is not None:
        os.environ["BETA_HOLD"] = bh
    if ch is not None:
        os.environ["COMP_HOLD"] = ch
    try:
        return run_variant(a, tag, tree, spec, None)
    finally:
        for k in ("BETA_HOLD", "COMP_HOLD"):
            os.environ.pop(k, None)
            if old[k] is not None:
                os.environ[k] = old[k]


def lines(R, prefix):
    return [ln for ln in (R["stdout"] if R else []) if ln.startswith(prefix)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-pre", required=True)
    ap.add_argument("--tree-post", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=96)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    os.makedirs(a.work, exist_ok=True)
    for k in ("BETA_HOLD", "COMP_HOLD", "VOTE_W"):
        os.environ.pop(k, None)
    print("DRIVER_SHA256 %s" % hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
    print("REGISTERED_TEST_SHA256 %s" % hashlib.sha256(open(RR.__file__, "rb").read()).hexdigest())
    print("SCORER_SHA256 %s" % hashlib.sha256(open(SC.__file__, "rb").read()).hexdigest())
    for t, lab in ((pre, "pre"), (post, "post")):
        hf = open(os.path.join(t, "Optimizers", "HF.py"), "rb").read()
        bn = open(os.path.join(t, "build_network.py"), "rb").read()
        print("TREE_%s %s HF %s build_network %s" % (lab.upper(), t, hashlib.sha256(hf).hexdigest(), hashlib.sha256(bn).hexdigest()))
    hpost = open(os.path.join(post, "Optimizers", "HF.py"), "rb").read()
    hpre = open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read()
    chk(hashlib.sha256(hpost).hexdigest() == SC.HF_POST_SHA, "--tree-post's HF.py is the scorer's HF_POST_SHA (cvt6's tree)")
    chk(hashlib.sha256(hpre).hexdigest() == SC.HF_PRE_SHA and b"PATCH_COMPHOLD" not in hpre,
        "--tree-pre's HF.py is cvt4's (HF_PRE_SHA) and carries no PATCH_COMPHOLD")
    chpre = os.path.join(post, "Optimizers", "HF.py.pre_comphold")
    chk(os.path.exists(chpre) and open(chpre, "rb").read() == hpre,
        "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_comphold")
    chk(open(os.path.join(pre, "build_network.py"), "rb").read() == open(os.path.join(post, "build_network.py"), "rb").read(),
        "the two trees' build_network.py are BYTE-IDENTICAL")
    rj = os.path.join(post, "Optimizers", SC.HEADPATH_ID + ".json")
    chk(os.path.exists(rj) and hashlib.sha256(open(rj, "rb").read()).hexdigest() == SC.HEADPATH_SHA,
        "--tree-post carries the replay file %s.json with sha HEADPATH_SHA" % SC.HEADPATH_ID)
    tj = os.path.join(post, "Optimizers", SC.TESTDOWN_ID + ".json")
    chk(os.path.exists(tj) and hashlib.sha256(open(tj, "rb").read()).hexdigest() == SC.TESTDOWN_SHA,
        "--tree-post carries the TEST-ONLY file %s.json with sha TESTDOWN_SHA" % SC.TESTDOWN_ID)
    TD = json.load(open(tj)) if os.path.exists(tj) else {"n": [2, 3], "v": [-14.0, -14.0]}
    print("\nsteps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026" % (a.steps, a.every, a.seed))

    print("\nRR0 DETERMINISM CONTROL (unpatched tree = cvt4's, twice)")
    A1 = run_h(a, "pre_k01_a", pre, "scalar", None, None)
    A2 = run_h(a, "pre_k01_b", pre, "scalar", None, None)
    same(A1, A2, "RR0 unpatched k01 vs itself")
    REF = {"k01": A1}
    for arm in ("HEAD", "HOLDLOW", "HOLDHIGH"):
        REF[arm] = run_h(a, "pre_%s" % arm.lower(), pre, SC.SPEC[arm], SC.BETAHOLD[arm] or None, None)

    print("\nRR1 OFF (COMP_HOLD unset / empty == cvt4's tree, bitwise)")
    for arm in ("k01", "HEAD", "HOLDLOW", "HOLDHIGH"):
        for val in (None, ""):
            lab = "unset" if val is None else "empty"
            R = run_h(a, "post_%s_%s" % (arm.lower(), lab), post, SC.SPEC[arm], SC.BETAHOLD[arm] or None, val)
            same(REF[arm], R, "RR1 %-8s COMP_HOLD %s" % (arm, lab))
            chk(lines(R, "COMP_HOLD") == ["COMP_HOLD: off"] and lines(R, "BETA_HOLD") == [SC.WITNESS_BH[arm]]
                and lines(R, "VOTE_W") == ["VOTE_W: off"],
                "RR1 %-8s COMP_HOLD %s printed exactly `COMP_HOLD: off`, its BETA_HOLD witness and `VOTE_W: off`" % (arm, lab))

    print("\nRR2 THE EXACT STRINGS BITE ON THE REAL PATH")
    tests = [(arm, SC.BETAHOLD[arm], SC.COMPHOLD[arm]) for arm in SC.FORCED]
    tests += [("TRI100", SC.BETAHOLD["HOLDLOW"], "tri:100"), ("TESTDOWN", SC.BETAHOLD["HOLDHIGH"], "rec:" + SC.TESTDOWN_ID)]
    got = {}
    for lab, bh, ch in tests:
        R = run_h(a, "post_%s" % lab.lower(), post, SC.HEADSPEC, bh, ch)
        if R is None:
            continue
        got[lab] = R
        arm50 = "HOLDLOW" if bh.endswith(":floor") else "HOLDHIGH"
        if lab in SC.FORCED:
            wbh, wch = SC.WITNESS_BH[lab], SC.WITNESS_CH[lab]
        elif lab == "TRI100":
            wbh, wch = SC.WITNESS_BH["HOLDLOW"], SC.witness_ch_of("tri", 100)
        else:
            wbh = SC.WITNESS_BH["HOLDHIGH"]
            wch = SC.witness_ch_of("rec", SC.TESTDOWN_ID, sha=SC.TESTDOWN_SHA, knots=len(TD["n"]), n0=TD["n"][0],
                                   n1=TD["n"][-1], vmax=max(TD["v"]), vlast=TD["v"][-1])
        chk(lines(R, "BETA_HOLD") == [wbh] and lines(R, "COMP_HOLD") == [wch] and lines(R, "VOTE_W") == ["VOTE_W: off"],
            "RR2 %-12s printed exactly ONE BETA_HOLD and ONE COMP_HOLD line == the scorer's witnesses, and VOTE_W: off" % lab,
            (lines(R, "COMP_HOLD") or ["(none)"])[0][:80])

        def c_of(n):
            if lab in SC.FORCED:
                return SC.hold_c_value(lab, n)
            if lab == "TRI100":
                return SC.v_of(n, 100)
            return SC.rec_value(n, TD["n"], TD["v"])
        bad_step = sum(1 for n, b in enumerate(R["betas"], 1)
                       if b[1] != SC.hold_value(arm50, n) or b[0] != c_of(n))
        chk(len(R["betas"]) == a.steps and bad_step == 0,
            "RR2 %-12s beta[0] AND beta[1] after EVERY one of %d steps == the registered schedules" % (lab, a.steps),
            "%d bad" % bad_step)
        recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
        bad = {"sched": 0, "bh": 0, "ch": 0}
        pb = pc = -1
        for r in recs:
            s = r["step"]
            bad["sched"] += (abs(r["beta"][1] - SC.hold_value(arm50, s + 2)) > SC.BH_TOL
                             or abs(r["beta_pre"][0][1] - SC.hold_value(arm50, s + 1)) > SC.BH_TOL
                             or abs(r["beta"][0] - c_of(s + 2)) > SC.BH_TOL
                             or abs(r["beta_pre"][0][0] - c_of(s + 1)) > SC.BH_TOL)
            for pfx, g in (("bh", 1), ("ch", 0)):
                nat, tie = SC.lion_natural(r["beta_pre"][0][g], r["mom_pre"][0][g], r["z_agg"][0][g])
                prev = pb if pfx == "bh" else pc
                ok = (r.get(pfx + "_n") == s + 2 and abs(r.get(pfx + "_held", 1e9) - r["beta"][g]) <= SC.BH_TOL
                      and (tie or abs(r.get(pfx + "_nat", 1e9) - nat) <= SC.BH_TOL) and r.get(pfx + "_active", -2) >= prev)
                bad[pfx] += not ok
                if pfx == "bh":
                    pb = r.get("bh_active", pb)
                else:
                    pc = r.get("ch_active", pc)
        chk(len(recs) == a.steps // a.every and not any(bad.values()),
            "RR2 %-12s at every probe record: both groups on schedule (beta, beta_pre); bh_* and ch_* audit" % lab,
            "%d records; %s; last bh_active %s ch_active %s" % (len(recs), bad, pb, pc))

    print("\nRR3 NON-VACUITY")
    for lab, ref_arm in (("TRI100", "HOLDLOW"), ("TESTDOWN", "HOLDHIGH")):
        T, U = got.get(lab), REF.get(ref_arm)
        if T and U:
            pre_pk = max(abs(x[0] - y[0]) for x, y in zip(T["betas"][:100], U["betas"][:100]))
            post_d = min(abs(x[0] - y[0]) for x, y in zip(T["betas"][150:], U["betas"][150:]))
            last = json.loads(T["probe_bytes"].decode().splitlines()[-1])
            chk(post_d > 0.05 and last.get("ch_active", 0) > 0,
                "RR3 %-8s the complement departs from cvt4's %s complement (min |diff| from step 150 %.3f; before step 100 "
                "max %.1e), ch_active %s > 0" % (lab, ref_arm, post_d, pre_pk, last.get("ch_active")))
            print("  NOTE RR3 %-8s final weights %s cvt4's %s (non-gating)"
                  % (lab, "differ from" if T["weights_sha256"] != U["weights_sha256"] else "are identical to", ref_arm))
    for lab in SC.FORCED:
        R, U = got.get(lab), REF.get(SC.NULL_TWIN[lab])
        if R and U:
            nd = sum(1 for x, y in zip(R["betas"], U["betas"]) if x[0] != y[0])
            mx = max(abs(x[0] - y[0]) for x, y in zip(R["betas"], U["betas"]))
            last = json.loads(R["probe_bytes"].decode().splitlines()[-1])
            print("  NOTE RR3 %-12s complement differs from cvt4's %s on %d of %d steps (max |diff| %.2e); ch_active %s; "
                  "final weights %s (non-gating)" % (lab, SC.NULL_TWIN[lab], nd, a.steps, mx, last.get("ch_active"),
                                                     "differ" if R["weights_sha256"] != U["weights_sha256"] else "identical"))

    print("\nRR4 OPEN LOOP: on the real records of the registered strings the applied beta vector is a function of n only")
    for lab in SC.FORCED:
        R = got.get(lab)
        if R:
            recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
            beta_ok = len(recs) > 0 and all(r["beta"] == [SC.hold_c_value(lab, r["step"] + 2), SC.hold_value(lab, r["step"] + 2)]
                                            for r in recs)
            chk(beta_ok, "RR4 %-12s at every real record the applied beta vector IS the pair of schedules, bitwise (a function of n only)" % lab)

    print("\n%s" % ("ALL PASS" if not RR.FAILED else "FAILURES: %r" % RR.FAILED))
    raise SystemExit(1 if RR.FAILED else 0)


if __name__ == "__main__":
    main()
