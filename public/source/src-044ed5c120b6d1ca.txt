"""INERTNESS AND BITE OF PATCH_WINDOWHOLD OVER A SHORT *REAL* RUN, FOR cvt9's EXACT STRINGS.  CORRECTIONS 249.

The REAL code path of a cvt9 run (train.py's own parse_args, build_network PlainNet18_c100, build_optimizer, load_data
CIFAR-100 AUGMENT=1, the loss, HF.step with BETA_CLIP, PROBE and PROBE_TENSOR) for --steps optimizer steps on a GPU, from
TWO trees:
  --tree-pre   cvt6's tree ($WS/harness_cvt6/cifar10, HF.py f1b9c8aa... = the cvt9 tree's HF.py.pre_windowhold), READ-ONLY
  --tree-post  the cvt9 tree (PATCH_WINDOWHOLD on top of it)
It IMPORTS cvt1's registered tests/test_voteweight_realrun.py UNEDITED (its child process, run_variant, same, chk);
BETA_HOLD, COMP_HOLD and WINDOW_HOLD are passed through the environment run_variant copies.  Strings, witnesses and
schedules come from analysis/cVT9_dosewindow_score.py.

  RR0  DETERMINISM CONTROL: cvt6's tree run twice (k01) -> identical beta, loss, probe bytes, final weights.
  RR1  OFF: WINDOW_HOLD unset and empty on k01, LOWHEADPATH, HIGHHEADPATH, MIDDOSE and RESDOSE (each with its own
       BETA_HOLD / COMP_HOLD) == cvt6's tree BITWISE; each printed exactly its BETA_HOLD / COMP_HOLD witnesses and
       `WINDOW_HOLD: off`, `VOTE_W: off`.  The dose rungs' beta[1] after every step == tri:7235 / tri:8609 (the two new
       integers need no new code: this is their string-level real-run proof).
  RR2  THE EXACT STRINGS (EARLY 0:9429, LATE 9429:end) and two test-only windows that switch inside the run (0:150,
       150:end): one line of each kind == the scorer's witnesses; beta[1] AND beta[0] after EVERY step == the windowed
       schedule and HEADPATH; at every probe record the bh_* / ch_* / wh_* records audit.
  RR3  NON-VACUITY: 0:150 departs from HIGHHEADPATH's 50 from step 150 and 150:end from LOWHEADPATH's, wh_active > 0.
       Inside 300 steps EARLY's schedule IS HIGHHEADPATH's and LATE's IS LOWHEADPATH's: their beta must be BITWISE those
       cvt6-tree runs' (gating), weights reported.
  RR4  OPEN LOOP: on EARLY / LATE the applied beta vector at every record IS the pair of schedules.

  python3 tests/test_windowhold_realrun.py --tree-pre $WS/harness_cvt6/cifar10 --tree-post $WS/harness_cvt9/cifar10 \\
      --steps 300 --every 10 --seed 105 --work /tmp/cvt9_realrun
"""
import argparse
import hashlib
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RR = _load("test_voteweight_realrun", os.path.join(HERE, "test_voteweight_realrun.py"))
SC = _load("cvt9_scorer", os.path.join(REPO, "analysis", "cVT9_dosewindow_score.py"))
chk, same, run_variant = RR.chk, RR.same, RR.run_variant
KEYS = ("BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD")


def run_h(a, tag, tree, spec, bh, ch, wh):
    old = dict((k, os.environ.pop(k, None)) for k in KEYS)
    for k, v in zip(KEYS, (bh, ch, wh)):
        if v is not None:
            os.environ[k] = v
    try:
        return run_variant(a, tag, tree, spec, None)
    finally:
        for k in KEYS:
            os.environ.pop(k, None)
            if old[k] is not None:
                os.environ[k] = old[k]


def lines(R, prefix):
    return [ln for ln in (R["stdout"] if R else []) if ln.startswith(prefix)]


def win_sched(wv, n):
    n0, n1 = wv.split(":")
    n0, n1 = int(n0), (None if n1 == "end" else int(n1))
    if n >= n0 and (n1 is None or n < n1):
        return SC.v_of(n, SC.P_HIGH)
    return SC.f32(SC.LO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-pre", required=True)
    ap.add_argument("--tree-post", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=105)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    os.makedirs(a.work, exist_ok=True)
    for k in KEYS + ("VOTE_W",):
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
    chk(hashlib.sha256(hpost).hexdigest() == SC.HF_POST_SHA, "--tree-post's HF.py is the scorer's HF_POST_SHA (cvt9's tree)")
    chk(hashlib.sha256(hpre).hexdigest() == SC.HF_PRE_SHA and b"PATCH_WINDOWHOLD" not in hpre,
        "--tree-pre's HF.py is cvt6's (HF_PRE_SHA) and carries no PATCH_WINDOWHOLD")
    wpre = os.path.join(post, "Optimizers", "HF.py.pre_windowhold")
    chk(os.path.exists(wpre) and open(wpre, "rb").read() == hpre, "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_windowhold")
    chk(open(os.path.join(pre, "build_network.py"), "rb").read() == open(os.path.join(post, "build_network.py"), "rb").read(),
        "the two trees' build_network.py are BYTE-IDENTICAL")
    for t in (pre, post):
        rj = os.path.join(t, "Optimizers", SC.HEADPATH_ID + ".json")
        chk(os.path.exists(rj) and hashlib.sha256(open(rj, "rb").read()).hexdigest() == SC.HEADPATH_SHA,
            "%s carries the replay file with sha HEADPATH_SHA" % t)
    print("\nsteps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026" % (a.steps, a.every, a.seed))

    print("\nRR0 DETERMINISM CONTROL (cvt6's tree, twice)")
    A1 = run_h(a, "pre_k01_a", pre, "scalar", None, None, None)
    A2 = run_h(a, "pre_k01_b", pre, "scalar", None, None, None)
    same(A1, A2, "RR0 unpatched k01 vs itself")
    REF = {"k01": A1}
    OFF_ARMS = ("k01", "LOWHEADPATH", "HIGHHEADPATH", "MIDDOSE", "RESDOSE")
    for arm in OFF_ARMS[1:]:
        REF[arm] = run_h(a, "pre_%s" % arm.lower(), pre, SC.SPEC[arm], SC.BETAHOLD[arm], SC.COMPHOLD[arm], None)

    print("\nRR1 OFF (WINDOW_HOLD unset / empty == cvt6's tree, bitwise)")
    for arm in OFF_ARMS:
        for val in (None, ""):
            lab = "unset" if val is None else "empty"
            R = run_h(a, "post_%s_%s" % (arm.lower(), lab), post, SC.SPEC[arm], SC.BETAHOLD[arm] or None, SC.COMPHOLD[arm] or None, val)
            same(REF[arm], R, "RR1 %-12s WINDOW_HOLD %s" % (arm, lab))
            chk(lines(R, "WINDOW_HOLD") == ["WINDOW_HOLD: off"] and lines(R, "BETA_HOLD") == [SC.WITNESS_BH[arm]]
                and lines(R, "COMP_HOLD") == [SC.WITNESS_CH[arm]] and lines(R, "VOTE_W") == ["VOTE_W: off"],
                "RR1 %-12s WINDOW_HOLD %s printed `WINDOW_HOLD: off` and its BETA_HOLD / COMP_HOLD witnesses, `VOTE_W: off`" % (arm, lab))
            if arm in SC.HELD and R:
                bad = sum(1 for n, b in enumerate(R["betas"], 1) if b[1] != SC.hold_value(arm, n) or b[0] != SC.hold_c_value(n))
                chk(len(R["betas"]) == a.steps and bad == 0,
                    "RR1 %-12s WINDOW_HOLD %s: beta[1] == %s and beta[0] == HEADPATH after every step" % (arm, lab, SC.BETAHOLD[arm].split(":", 1)[1]),
                    "%d bad" % bad)

    print("\nRR2 THE EXACT STRINGS AND TWO TEST-ONLY WINDOWS BITE ON THE REAL PATH")
    tests = [("EARLY", SC.WINDOWHOLD["EARLY"]), ("LATE", SC.WINDOWHOLD["LATE"]), ("WIN0_150", "0:150"), ("WIN150_END", "150:end")]
    got = {}
    for lab, wv in tests:
        R = run_h(a, "post_%s" % lab.lower(), post, SC.HEADSPEC, SC.BETAHOLD["EARLY"], SC.COMPHOLD["EARLY"], wv)
        if R is None:
            continue
        got[lab] = R
        n0, n1 = wv.split(":")
        wwit = SC.witness_wh_of((int(n0), None if n1 == "end" else int(n1)))
        chk(lines(R, "BETA_HOLD") == [SC.WITNESS_BH["EARLY"]] and lines(R, "COMP_HOLD") == [SC.WITNESS_CH["EARLY"]]
            and lines(R, "WINDOW_HOLD") == [wwit] and lines(R, "VOTE_W") == ["VOTE_W: off"],
            "RR2 %-10s printed ONE line of each kind == the scorer's witnesses" % lab, (lines(R, "WINDOW_HOLD") or ["(none)"])[0][:90])
        bad_step = sum(1 for n, b in enumerate(R["betas"], 1) if b[1] != win_sched(wv, n) or b[0] != SC.hold_c_value(n))
        chk(len(R["betas"]) == a.steps and bad_step == 0,
            "RR2 %-10s beta[1] (windowed) AND beta[0] (HEADPATH) after EVERY one of %d steps == the schedules" % (lab, a.steps), "%d bad" % bad_step)
        recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
        bad = {"sched": 0, "bh": 0, "ch": 0, "wh": 0}
        prev = {"bh": -1, "ch": -1, "wh": -1}
        for r in recs:
            s = r["step"]
            bad["sched"] += (abs(r["beta"][1] - win_sched(wv, s + 2)) > SC.BH_TOL or abs(r["beta_pre"][0][1] - win_sched(wv, s + 1)) > SC.BH_TOL
                             or abs(r["beta"][0] - SC.hold_c_value(s + 2)) > SC.BH_TOL or abs(r["beta_pre"][0][0] - SC.hold_c_value(s + 1)) > SC.BH_TOL)
            for pfx, g in (("bh", 1), ("ch", 0), ("wh", 1)):
                if pfx == "wh":
                    nat, tie = SC.v_of(s + 2, SC.P_HIGH), False
                else:
                    nat, tie = SC.lion_natural(r["beta_pre"][0][g], r["mom_pre"][0][g], r["z_agg"][0][g])
                ok = (r.get(pfx + "_n") == s + 2 and abs(r.get(pfx + "_held", 1e9) - r["beta"][g]) <= SC.BH_TOL
                      and (tie or abs(r.get(pfx + "_nat", 1e9) - nat) <= SC.BH_TOL) and r.get(pfx + "_active", -2) >= prev[pfx])
                bad[pfx] += not ok
                prev[pfx] = r.get(pfx + "_active", prev[pfx])
        chk(len(recs) == a.steps // a.every and not any(bad.values()),
            "RR2 %-10s at every probe record: both groups on schedule (beta, beta_pre); bh_* / ch_* / wh_* audit" % lab,
            "%d records; %s; last active %s" % (len(recs), bad, prev))

    print("\nRR3 NON-VACUITY")
    for lab, ref in (("WIN0_150", "HIGHHEADPATH"), ("WIN150_END", "LOWHEADPATH")):
        T, U = got.get(lab), REF.get(ref)
        if T and U:
            before = max(abs(x[1] - y[1]) for x, y in zip(T["betas"][:149], U["betas"][:149]))
            after = min(abs(x[1] - y[1]) for x, y in zip(T["betas"][150:], U["betas"][150:]))
            last = json.loads(T["probe_bytes"].decode().splitlines()[-1])
            chk(before == 0.0 and after > 0.05 and last.get("wh_active", 0) > 0,
                "RR3 %-10s 50's beta == cvt6's %s before step 150 (max %.1e) and departs after (min |diff| %.3f); wh_active %s > 0"
                % (lab, ref, before, after, last.get("wh_active")))
    for lab, ref in (("EARLY", "HIGHHEADPATH"), ("LATE", "LOWHEADPATH")):
        T, U = got.get(lab), REF.get(ref)
        if T and U:
            chk(T["betas"] == U["betas"], "RR3 %-10s inside %d steps the windowed schedule IS %s's: beta BITWISE cvt6's %s run" % (lab, a.steps, ref, ref))
            print("  NOTE RR3 %-10s loss %s, final weights %s cvt6's %s (non-gating)"
                  % (lab, "identical" if T["losses"] == U["losses"] else "differs", "identical to" if T["weights_sha256"] == U["weights_sha256"] else "differ from", ref))

    print("\nRR4 OPEN LOOP: on EARLY / LATE the applied beta vector at every real record is the pair of schedules")
    for lab in ("EARLY", "LATE"):
        R = got.get(lab)
        if R:
            recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
            chk(len(recs) > 0 and all(r["beta"] == [SC.hold_c_value(r["step"] + 2), SC.hold_value(lab, r["step"] + 2)] for r in recs),
                "RR4 %-10s the applied beta vector IS the pair of schedules at every record, bitwise" % lab)

    print("\n%s" % ("ALL PASS" if not RR.FAILED else "FAILURES: %r" % RR.FAILED))
    raise SystemExit(1 if RR.FAILED else 0)


if __name__ == "__main__":
    main()
