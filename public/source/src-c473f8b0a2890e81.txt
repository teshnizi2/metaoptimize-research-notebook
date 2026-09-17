"""INERTNESS AND BITE OF PATCH_RESTHOLD OVER A SHORT *REAL* ResNet18_c100 RUN, FOR cvt8's EXACT STRINGS.  CORRECTIONS 248.

The REAL code path of a cvt8 run -- train.py's own parse_args, build_network (ResNet18_c100), build_optimizer, load_data
(CIFAR-100, AUGMENT=1), the loss, HF.step with BETA_CLIP, PROBE and PROBE_TENSOR -- for --steps optimizer steps on a GPU,
from TWO trees:

  --tree-pre   cvt7's tree ($WS/harness_cvt7/cifar10, HF.py 2396f2be... = the cvt8 tree's HF.py.pre_resthold), READ-ONLY
  --tree-post  the cvt8 tree (PATCH_RESTHOLD + the replay files on top of it)

It IMPORTS cvt7's registered tests/test_grouphold_realrun.py UNEDITED and reuses its child process (`--child`, itself
cvt1's registered child with the network a parameter), `run_variant`, `same` and `chk`; GROUP_HOLD is passed as
run_variant's argument and REST_HOLD through the environment run_variant copies.  cvt7's registered test proved cvt7's
tree OFF == the LIVE tree bitwise (243.4, job 5022464); this test proves cvt8's tree OFF == cvt7's tree bitwise, so the
chain reaches the tree every ISO source run of the replay ran from.  The strings, witnesses and schedules come from
analysis/cVT8_doseroute_score.py.

  RR0  DETERMINISM CONTROL: cvt7's tree run twice (k01) -> identical beta, loss, probe bytes, final weights.
  RR1  OFF: REST_HOLD unset and REST_HOLD empty, on k01, ISO, and GROUP_HOLD floor / tri:8609 / tri:9428 (ISO's spec) ==
       cvt7's tree with the same GROUP_HOLD, BITWISE (beta every step, loss every step, probe.jsonl bytes, every final
       parameter and buffer); each printed exactly `REST_HOLD: off`, its GROUP_HOLD line, `BETA_HOLD: off`, `VOTE_W: off`.
  RR2  THE EXACT STRINGS ON THE REAL PATH (HIGHISOPATH, BIGISOPATH, LOWISOPATH) and two test-only strings whose complement
       schedule departs from the natural rise inside the run (floor + tri:100; tri:9428 + rec:cvt8_testdown): exactly one
       GROUP_HOLD and one REST_HOLD line == the scorer's witnesses; beta[0] AND beta[1] after EVERY step == the schedules;
       at every probe record both groups on schedule in beta and beta_pre and the gh_* / rh_* records audit, no bh_* / ch_*
       key -- and the scorer's own bite_check on the real records reports no Lion / schedule / audit / key failure.
  RR3  NON-VACUITY: the two test-only strings' complement departs from cvt7's same-GROUP_HOLD free complement by > 0.05
       from step 150 with rh_active > 0; the registered strings' departures are REPORTED (in 300 steps ISOPATH is the
       max-rate rise, as a free complement is; non-gating).
  RR4  OPEN LOOP: at every real record of the three registered strings the applied beta vector IS the pair of schedules.

RUN (on a GPU node, as a Slurm job):
  python3 tests/test_resthold_realrun.py --tree-pre $WS/harness_cvt7/cifar10 --tree-post $WS/harness_cvt8/cifar10 \\
      --steps 300 --every 10 --seed 102 --work /tmp/cvt8_realrun
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


RR = _load("test_grouphold_realrun", os.path.join(HERE, "test_grouphold_realrun.py"))
SC = _load("cvt8_scorer", os.path.join(REPO, "analysis", "cVT8_doseroute_score.py"))
chk, same, run_variant = RR.chk, RR.same, RR.run_variant


def run_h(a, tag, tree, spec, gh, rh):
    old = os.environ.pop("REST_HOLD", None)
    if rh is not None:
        os.environ["REST_HOLD"] = rh
    try:
        return run_variant(a, tag, tree, spec, gh)
    finally:
        os.environ.pop("REST_HOLD", None)
        if old is not None:
            os.environ["REST_HOLD"] = old


def lines(R, prefix):
    return [ln for ln in (R["stdout"] if R else []) if ln.startswith(prefix)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-pre", required=True)
    ap.add_argument("--tree-post", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=102)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    a.net = SC.NET
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    os.makedirs(a.work, exist_ok=True)
    for k in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "COMP_HOLD", "REST_HOLD"):
        os.environ.pop(k, None)
    print("DRIVER_SHA256 %s" % hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
    print("REGISTERED_TEST_SHA256 %s" % hashlib.sha256(open(RR.__file__, "rb").read()).hexdigest())
    print("SCORER_SHA256 %s" % hashlib.sha256(open(SC.__file__, "rb").read()).hexdigest())
    for t, lab in ((pre, "pre"), (post, "post")):
        hf = open(os.path.join(t, "Optimizers", "HF.py"), "rb").read()
        bn = open(os.path.join(t, "build_network.py"), "rb").read()
        tr = open(os.path.join(t, "train.py"), "rb").read()
        print("TREE_%s %s HF %s build_network %s train %s" % (lab.upper(), t, hashlib.sha256(hf).hexdigest(),
                                                             hashlib.sha256(bn).hexdigest(), hashlib.sha256(tr).hexdigest()))
    hpost = open(os.path.join(post, "Optimizers", "HF.py"), "rb").read()
    hpre = open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read()
    chk(hashlib.sha256(hpost).hexdigest() == SC.HF_POST_SHA, "--tree-post's HF.py is the scorer's HF_POST_SHA (cvt8's tree)")
    chk(hashlib.sha256(hpre).hexdigest() == SC.HF_PRE_SHA and b"PATCH_RESTHOLD" not in hpre,
        "--tree-pre's HF.py is cvt7's (HF_PRE_SHA 2396f2be...) and carries no PATCH_RESTHOLD")
    rpre = os.path.join(post, "Optimizers", "HF.py.pre_resthold")
    chk(os.path.exists(rpre) and open(rpre, "rb").read() == hpre, "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_resthold")
    for f in ("build_network.py", "train.py", "load_data.py", "Optimizers/build_optimizer.py"):
        chk(open(os.path.join(pre, f), "rb").read() == open(os.path.join(post, f), "rb").read(), "the two trees' %s are BYTE-IDENTICAL" % f)
    rj = os.path.join(post, "Optimizers", SC.ISOPATH_ID + ".json")
    chk(os.path.exists(rj) and hashlib.sha256(open(rj, "rb").read()).hexdigest() == SC.ISOPATH_SHA,
        "--tree-post carries the replay file %s.json with sha ISOPATH_SHA" % SC.ISOPATH_ID)
    tj = os.path.join(post, "Optimizers", SC.TESTDOWN_ID + ".json")
    chk(os.path.exists(tj) and hashlib.sha256(open(tj, "rb").read()).hexdigest() == SC.TESTDOWN_SHA,
        "--tree-post carries the TEST-ONLY file %s.json with sha TESTDOWN_SHA" % SC.TESTDOWN_ID)
    TD = json.load(open(tj)) if os.path.exists(tj) else {"n": [2, 3], "v": [-14.0, -14.0]}
    print("\nnet %s, steps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026" % (a.net, a.steps, a.every, a.seed))

    GH = {"k01": None, "ISO": None, "floor": SC.GROUPHOLD["LOWISOPATH"], "tri8609": SC.GROUPHOLD["HOLDHIGH"],
          "tri9428": SC.GROUPHOLD["HOLDBIG"]}
    SP = {"k01": "scalar", "ISO": SC.ISOSPEC, "floor": SC.ISOSPEC, "tri8609": SC.ISOSPEC, "tri9428": SC.ISOSPEC}
    WG = {"k01": "GROUP_HOLD: off", "ISO": "GROUP_HOLD: off", "floor": SC.WITNESS_GH["LOWISOPATH"],
          "tri8609": SC.WITNESS_GH["HOLDHIGH"], "tri9428": SC.WITNESS_GH["HOLDBIG"]}

    print("\nRR0 DETERMINISM CONTROL (cvt7's tree, twice)")
    A1 = run_h(a, "pre_k01_a", pre, "scalar", None, None)
    A2 = run_h(a, "pre_k01_b", pre, "scalar", None, None)
    same(A1, A2, "RR0 cvt7-tree k01 vs itself")
    REF = {"k01": A1}
    for key in ("ISO", "floor", "tri8609", "tri9428"):
        REF[key] = run_h(a, "pre_%s" % key.lower(), pre, SP[key], GH[key], None)
    chk(all(lines(REF[k], "REST_HOLD") == [] for k in REF), "RR0 cvt7's tree prints no REST_HOLD line (it carries no PATCH_RESTHOLD)")

    print("\nRR1 OFF (REST_HOLD unset / empty == cvt7's tree, bitwise)")
    for key in ("k01", "ISO", "floor", "tri8609", "tri9428"):
        for val in (None, ""):
            lab = "unset" if val is None else "empty"
            R = run_h(a, "post_%s_%s" % (key.lower(), lab), post, SP[key], GH[key], val)
            same(REF[key], R, "RR1 %-7s REST_HOLD %s" % (key, lab))
            chk(lines(R, "REST_HOLD") == ["REST_HOLD: off"] and lines(R, "GROUP_HOLD") == [WG[key]]
                and lines(R, "BETA_HOLD") == ["BETA_HOLD: off"] and lines(R, "VOTE_W") == ["VOTE_W: off"],
                "RR1 %-7s REST_HOLD %s printed exactly `REST_HOLD: off`, its GROUP_HOLD line, `BETA_HOLD: off`, `VOTE_W: off`" % (key, lab))

    print("\nRR2 THE EXACT STRINGS BITE ON THE REAL PATH")
    tests = [(arm, SC.GROUPHOLD[arm], SC.RESTHOLD[arm]) for arm in SC.FORCED]
    tests += [("TRI100", SC.GROUPHOLD["LOWISOPATH"], "tri:100"), ("TESTDOWN", SC.GROUPHOLD["HOLDBIG"], "rec:" + SC.TESTDOWN_ID)]
    got = {}
    for lab, gh, rh in tests:
        R = run_h(a, "post_%s" % lab.lower(), post, SC.ISOSPEC, gh, rh)
        if R is None:
            continue
        got[lab] = R
        carm = {"TRI100": "LOWISOPATH", "TESTDOWN": "BIGISOPATH"}.get(lab, lab)
        if lab in SC.FORCED:
            wrh = SC.WITNESS_RH[lab]
        elif lab == "TRI100":
            wrh = SC.witness_rh_of("tri", 100)
        else:
            wrh = SC.witness_rh_of("rec", SC.TESTDOWN_ID, sha=SC.TESTDOWN_SHA, knots=len(TD["n"]), n0=TD["n"][0], n1=TD["n"][-1],
                                   vmax=max(TD["v"]), vlast=TD["v"][-1])
        chk(lines(R, "GROUP_HOLD") == [SC.WITNESS_GH[carm]] and lines(R, "REST_HOLD") == [wrh]
            and lines(R, "BETA_HOLD") == ["BETA_HOLD: off"] and lines(R, "VOTE_W") == ["VOTE_W: off"] and not lines(R, "COMP_HOLD"),
            "RR2 %-12s printed exactly ONE GROUP_HOLD and ONE REST_HOLD line == the scorer's witnesses, BETA_HOLD: off, VOTE_W: off" % lab,
            (lines(R, "REST_HOLD") or ["(none)"])[0][:80])

        def r_of(n):
            if lab in SC.FORCED:
                return SC.hold_r_value(lab, n)
            if lab == "TRI100":
                return SC.v_of(n, 100)
            return SC.rec_value(n, TD["n"], TD["v"])
        bad_step = sum(1 for n, b in enumerate(R["betas"], 1) if b[1] != SC.hold_value(carm, n) or b[0] != r_of(n))
        chk(len(R["betas"]) == a.steps and bad_step == 0,
            "RR2 %-12s beta[0] AND beta[1] after EVERY one of %d steps == the registered schedules" % (lab, a.steps), "%d bad" % bad_step)
        recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
        bad = {"sched": 0, "gh": 0, "rh": 0, "foreign": 0}
        pg = pr = -1
        for r in recs:
            s = r["step"]
            bad["sched"] += (abs(r["beta"][1] - SC.hold_value(carm, s + 2)) > SC.BH_TOL
                             or abs(r["beta_pre"][0][1] - SC.hold_value(carm, s + 1)) > SC.BH_TOL
                             or abs(r["beta"][0] - r_of(s + 2)) > SC.BH_TOL or abs(r["beta_pre"][0][0] - r_of(s + 1)) > SC.BH_TOL)
            bad["foreign"] += any(k in r for k in SC.FOREIGN_KEYS)
            for pfx, gi in (("gh", 1), ("rh", 0)):
                prev = pg if pfx == "gh" else pr
                ok, _act, ca = SC._audit(r, pfx, s + 2, gi, [float(x) for x in r["beta"]], r["beta_pre"][0], r["mom_pre"][0],
                                         r["z_agg"][0], prev)
                bad[pfx] += not ok
                if ca is not None:
                    if pfx == "gh":
                        pg = ca
                    else:
                        pr = ca
        chk(len(recs) == a.steps // a.every and not any(bad.values()),
            "RR2 %-12s at every probe record: both groups on schedule (beta, beta_pre); gh_* and rh_* audit; no bh_* / ch_* key" % lab,
            "%d records; %s; last gh_active %s rh_active %s" % (len(recs), bad, pg, pr))
        if lab in SC.FORCED:
            _ok, d = SC.bite_check(lab, recs)
            chk(d["bad_rec"] == 0 and d["bad_lion"] == 0 and d["bad_sched"] == 0 and d["bad_sched_pre"] == 0 and d["bad_rsched"] == 0
                and d["bad_rsched_pre"] == 0 and d["bad_gh"] == 0 and d["bad_rh"] == 0 and d["keys_wrong"] == 0 and d["n"] == len(recs),
                "RR2 %-12s the scorer's own bite_check on the %d real records: no record, Lion, schedule, audit or key failure" % (lab, len(recs)),
                "sched %d/%d rsched %d/%d gh %d rh %d keys %d gh_active %s rh_active %s" % (
                    d["bad_sched"], d["bad_sched_pre"], d["bad_rsched"], d["bad_rsched_pre"], d["bad_gh"], d["bad_rh"], d["keys_wrong"],
                    d["gh_active_last"], d["rh_active_last"]))
            print("  NOTE RR2 %-12s bite_check's record-count (500) and step (step == 100 k) halves do not apply to a %d-record PROBE=%d "
                  "run: bad_step %d of %d" % (lab, len(recs), a.every, d["bad_step"], len(recs)))

    print("\nRR3 NON-VACUITY")
    for lab, ref in (("TRI100", "floor"), ("TESTDOWN", "tri9428")):
        T, U = got.get(lab), REF.get(ref)
        if T and U:
            post_d = min(abs(x[0] - y[0]) for x, y in zip(T["betas"][150:], U["betas"][150:]))
            last = json.loads(T["probe_bytes"].decode().splitlines()[-1])
            chk(post_d > 0.05 and last.get("rh_active", 0) > 0,
                "RR3 %-8s the complement departs from cvt7's GROUP_HOLD %s free complement (min |diff| from step 150 %.3f), rh_active %s > 0"
                % (lab, ref, post_d, last.get("rh_active")))
    for lab, ref in (("HIGHISOPATH", "tri8609"), ("BIGISOPATH", "tri9428"), ("LOWISOPATH", "floor")):
        R, U = got.get(lab), REF.get(ref)
        if R and U:
            nd = sum(1 for x, y in zip(R["betas"], U["betas"]) if x[0] != y[0])
            mx = max(abs(x[0] - y[0]) for x, y in zip(R["betas"], U["betas"]))
            last = json.loads(R["probe_bytes"].decode().splitlines()[-1])
            print("  NOTE RR3 %-12s complement differs from cvt7's GROUP_HOLD %s free complement on %d of %d steps (max |diff| %.2e); "
                  "rh_active %s; final weights %s (non-gating)" % (lab, ref, nd, a.steps, mx, last.get("rh_active"),
                                                                 "differ" if R["weights_sha256"] != U["weights_sha256"] else "identical"))

    print("\nRR4 OPEN LOOP: on the real records of the registered strings the applied beta vector is a function of n only")
    for lab in SC.FORCED:
        R = got.get(lab)
        if R:
            recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
            chk(len(recs) > 0 and all(r["beta"] == [SC.hold_r_value(lab, r["step"] + 2), SC.hold_value(lab, r["step"] + 2)] for r in recs),
                "RR4 %-12s at every real record the applied beta vector IS the pair of schedules, bitwise" % lab)

    print("\n%s" % ("ALL PASS" if not RR.FAILED else "FAILURES: %r" % RR.FAILED))
    raise SystemExit(1 if RR.FAILED else 0)


if __name__ == "__main__":
    main()
