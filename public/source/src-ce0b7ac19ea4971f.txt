"""INERTNESS AND BITE OF PATCH_VOTEWEIGHT FOR cvt3's EXACT VOTE_W STRINGS, OVER A SHORT *REAL* RUN.
CORRECTIONS 232.

cvt3 adds NO harness code: it runs cvt1's isolated tree ($WS/harness_cvt1, HF.py 3f2b98e1..., unchanged) with
two NEW VOTE_W strings of 21 items each (MUTEDOWN, MUTECTL) and cvt1's MUTE string (MUTE50).  cvt1's
tests/test_voteweight_realrun.py proved the switch on ONE-item strings.  This driver re-proves it on the
exact strings cvt3 submits, on the same real code path.  It IMPORTS that registered test UNEDITED and reuses
its child process (`--child`: train.py's own parse_args, build_network, build_optimizer, load_data, CIFAR-100,
AUGMENT=1, BETA_CLIP, PROBE, PROBE_TENSOR) and its `run_variant` / `same` / `chk`; the strings and witness
lines are read from analysis/cVT3_downcoalition_score.py, so the test and the scorer cannot disagree.

  RR0  DETERMINISM CONTROL: the unpatched tree run twice -> beta at every step, loss, probe.jsonl bytes,
       every final parameter and buffer identical (else the bitwise claims cannot be tested; FAIL).
  RR1  OFF: VOTE_W unset and VOTE_W empty on k01's spec == the unpatched tree, bitwise; `VOTE_W: off` printed.
  RR2  IDENTITY ON THE EXACT NAME LISTS: MUTEDOWN's and MUTECTL's 21 names, every weight 1 -> bitwise == the
       unpatched tree (the 21-item parse and the reweighted sum path are the original sum), and the witness
       prints all 21 items at w=1.0.
  RR3  THE EXACT STRINGS BITE: MUTE50, MUTEDOWN, MUTECTL print the scorer's WITNESS byte for byte; at every
       probe record z_agg == sum_i w_i z_tensor_i and mom_pre == sum_i w_i m_tensor_i (rel 1e-5); z_agg differs
       from the unpatched run's on >= 1 record; and for MUTEDOWN / MUTECTL (the SET-level bite the 50-only
       check cannot see) the registered set's sum differs from the 50-ONLY sum by > 1e-3 relative on >= 1
       record and z_agg matches the 50-only sum on NONE of those records, nor the other arm's set.

RUN (on a GPU node; ~10 min):
  python3 tests/test_voteweight_realrun_cvt3.py --tree-pre $WS/harness_cpl1/cifar10 \\
      --tree-post $WS/harness_cvt1/cifar10 --steps 300 --work /tmp/cvt3_realrun
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
SC = _load("cvt3_scorer", os.path.join(REPO, "analysis", "cVT3_downcoalition_score.py"))
chk, same, run_variant = RR.chk, RR.same, RR.run_variant
K01 = "scalar"


def rel(a, b, scale):
    return abs(a - b) / (scale + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-pre", required=True)
    ap.add_argument("--tree-post", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=84)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    os.makedirs(a.work, exist_ok=True)
    print("DRIVER_SHA256 %s" % hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
    print("REGISTERED_TEST_SHA256 %s" % hashlib.sha256(open(RR.__file__, "rb").read()).hexdigest())
    print("SCORER_SHA256 %s" % hashlib.sha256(open(SC.__file__, "rb").read()).hexdigest())
    for t, lab in ((pre, "pre"), (post, "post")):
        hf = open(os.path.join(t, "Optimizers", "HF.py"), "rb").read()
        bn = open(os.path.join(t, "build_network.py"), "rb").read()
        print("TREE_%s %s HF %s build_network %s" % (lab.upper(), t, hashlib.sha256(hf).hexdigest(), hashlib.sha256(bn).hexdigest()))
    chk(hashlib.sha256(open(os.path.join(post, "Optimizers", "HF.py"), "rb").read()).hexdigest() == SC.HF_POST_SHA,
        "--tree-post's HF.py is the scorer's HF_POST_SHA (cvt1's tree, unchanged)")
    chk(b"PATCH_VOTEWEIGHT" not in open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read(),
        "--tree-pre is unpatched")
    vwpre = os.path.join(post, "Optimizers", "HF.py.pre_voteweight")
    chk(os.path.exists(vwpre) and open(vwpre, "rb").read() == open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read(),
        "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_voteweight")
    chk(open(os.path.join(pre, "build_network.py"), "rb").read() == open(os.path.join(post, "build_network.py"), "rb").read(),
        "the two trees' build_network.py are BYTE-IDENTICAL")
    print("\nsteps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026" % (a.steps, a.every, a.seed))

    print("\nRR0 DETERMINISM CONTROL (unpatched tree, twice)")
    A1 = run_variant(a, "pre_k01_a", pre, K01, None)
    A2 = run_variant(a, "pre_k01_b", pre, K01, None)
    same(A1, A2, "RR0 unpatched k01 vs itself")

    print("\nRR1 OFF")
    same(A1, run_variant(a, "post_k01_unset", post, K01, None), "RR1 k01 VOTE_W unset")
    E = run_variant(a, "post_k01_empty", post, K01, "")
    same(A1, E, "RR1 k01 VOTE_W empty")
    chk(E is not None and "VOTE_W: off" in E["stdout"], "RR1 the patched run printed `VOTE_W: off`")

    print("\nRR2 IDENTITY on the exact 21-name lists")
    for arm in ("MUTEDOWN", "MUTECTL"):
        s1 = SC.VOTEW[arm].replace(":0", ":1")
        assert s1.count(":1") == 21 and ":0" not in s1
        R = run_variant(a, "post_%s_id" % arm.lower(), post, K01, s1)
        same(A1, R, "RR2 %s names at weight 1" % arm)
        w1 = SC.WITNESS[arm].replace(":w=0.0:", ":w=1.0:")
        chk(R is not None and w1 in R["stdout"], "RR2 %s printed its 21 items at w=1.0" % arm)

    print("\nRR3 THE EXACT STRINGS BITE ON THE REAL PATH")
    groups = [list(range(1, 54))]
    got = {}
    for arm in ("MUTE50", "MUTEDOWN", "MUTECTL"):
        R = run_variant(a, "post_%s" % arm.lower(), post, K01, SC.VOTEW[arm])
        if R is None:
            continue
        got[arm] = R
        chk(SC.WITNESS[arm] in R["stdout"] and sum(1 for ln in R["stdout"] if ln.startswith("VOTE_W")) == 1,
            "RR3 %-8s printed exactly ONE VOTE_W line == the scorer's WITNESS (%d items)" % (arm, len(SC.SILENCED[arm])))
        recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
        refr = [json.loads(x) for x in A1["probe_bytes"].decode().splitlines()] if A1 else []
        W = SC.WEIGHTS[arm]
        bad = 0
        worst = 0.0
        set_inf = set_match = other_match = 0
        other = {"MUTEDOWN": SC.WEIGHTS["MUTECTL"], "MUTECTL": SC.WEIGHTS["MUTEDOWN"]}.get(arm)
        for r in recs:
            zt, mt = r["z_tensor"], r["m_tensor"]
            sz = sum(abs(v) for v in zt)
            sm = sum(abs(v) for v in mt)
            dz = rel(sum(W.get(i, 1.0) * zt[i - 1] for i in groups[0]), r["z_agg"][0][0], sz)
            dm = rel(sum(W.get(i, 1.0) * mt[i - 1] for i in groups[0]), r["mom_pre"][0][0], sm)
            worst = max(worst, dz, dm)
            bad += dz > 1e-5 or dm > 1e-5
            if other is not None:
                o50 = sum(SC.WEIGHTS["MUTE50"].get(i, 1.0) * zt[i - 1] for i in groups[0])
                ws = sum(W.get(i, 1.0) * zt[i - 1] for i in groups[0])
                if rel(ws, o50, sz) > 1e-3:
                    set_inf += 1
                    set_match += rel(o50, r["z_agg"][0][0], sz) <= 1e-5
                xo = sum(other.get(i, 1.0) * zt[i - 1] for i in groups[0])
                if rel(ws, xo, sz) > 1e-3:
                    other_match += rel(xo, r["z_agg"][0][0], sz) <= 1e-5
        chk(recs and bad == 0, "RR3 %-8s z_agg == sum_i w_i z_tensor_i AND mom_pre == sum_i w_i m_tensor_i at every record" % arm,
            "%d records, worst rel %.2e" % (len(recs), worst))
        ndiff = sum(x["z_agg"] != y["z_agg"] for x, y in zip(recs, refr))
        chk(ndiff >= 1, "RR3 %-8s z_agg differs from the unpatched run's on >= 1 record" % arm, "%d of %d" % (ndiff, len(recs)))
        if other is not None:
            chk(set_inf >= 1 and set_match == 0 and other_match == 0,
                "RR3 %-8s SET-level bite: the 50-only sum differs on %d records and matches none; the other set matches none"
                % (arm, set_inf), "50-only match %d, other-set match %d" % (set_match, other_match))

    print("\n%s" % ("ALL PASS" if not RR.FAILED else "FAILURES: %r" % RR.FAILED))
    raise SystemExit(1 if RR.FAILED else 0)


if __name__ == "__main__":
    main()
