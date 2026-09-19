"""INERTNESS AND BITE OF PATCH_SHADOWVOTE OVER A SHORT *REAL* RUN, FOR csv1's EXACT STRINGS.  CORRECTIONS 262.

The REAL code path of a csv1 run (train.py's own parse_args, build_network PlainNet18_c100, build_optimizer, load_data
CIFAR-100 AUGMENT=1, the loss, HF.step with BETA_CLIP, PROBE and PROBE_TENSOR) for --steps optimizer steps on a GPU, from
TWO trees:
  --tree-pre   cvt9's tree ($WS/harness_cvt9/cifar10, HF.py 816e3357... = the csv1 tree's HF.py.pre_shadowvote), READ-ONLY
  --tree-post  the csv1 tree (PATCH_SHADOWVOTE on top of it)
It IMPORTS cvt1's registered tests/test_voteweight_realrun.py UNEDITED (its child process, run_variant, same, chk);
SHADOW_VOTE is passed through the environment run_variant copies.  Strings, witnesses and the audit come from
analysis/cSV1_shadowvote_score.py.

  RR0  DETERMINISM CONTROL: cvt9's tree run twice (k01) -> identical beta, loss, probe bytes, final weights.
  RR1  OFF: SHADOW_VOTE unset and empty on k01, MUTE (VOTE_W on) and HEAD == cvt9's tree BITWISE (beta, loss, probe.jsonl
       bytes, every final parameter and buffer); each printed `SHADOW_VOTE: off` and its own VOTE_W witness.
  RR2  THE INERT ON-PATHS (INERT's registered string shadow:shared, and natural:shared): beta at every step, every loss and
       every final parameter and buffer BITWISE cvt9's k01 run; the probe records EQUAL to cvt9's after deleting the sv_*
       keys; the witness line == the scorer's; the scorer's G-BITE (Lion, z_agg, sv_* audit incl. the INERT identity) passes.
  RR3  THE EXACT LOW STRINGS BITE (SHADOWLOW shadow:floor, NAIVELOW natural:floor): witness == the scorer's; the scorer's
       G-BITE passes on the real records (sv_a_applied == the floor, sv_z_vote == z_tensor[50], the trace ordering, ...);
       final weights DIFFER from cvt9's k01; 50's vote term differs between SHADOWLOW and NAIVELOW on every record after
       the first; SHADOWLOW's 50 term vs cvt9's k01 term reported (non-gating).
  RR4  the registered controls' real records pass the scorer's G-BITE as their arms (k01, MUTE, HEAD) on BOTH trees.

  python3 tests/test_shadowvote_realrun.py --tree-pre $WS/harness_cvt9/cifar10 --tree-post $WS/harness_csv1/cifar10 \\
      --steps 300 --every 10 --seed 136 --work /tmp/csv1_realrun
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
SC = _load("csv1_scorer", os.path.join(REPO, "analysis", "cSV1_shadowvote_score.py"))
chk, same, run_variant = RR.chk, RR.same, RR.run_variant
KEYS = ("SHADOW_VOTE", "BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD", "GROUP_HOLD")


def run_s(a, tag, tree, spec, sv, vw):
    old = dict((k, os.environ.pop(k, None)) for k in KEYS)
    if sv is not None:
        os.environ["SHADOW_VOTE"] = sv
    try:
        return run_variant(a, tag, tree, spec, vw)
    finally:
        for k in KEYS:
            os.environ.pop(k, None)
            if old[k] is not None:
                os.environ[k] = old[k]


def lines(R, prefix):
    return [ln for ln in (R["stdout"] if R else []) if ln.startswith(prefix)]


def recs_of(R):
    return [json.loads(x) for x in R["probe_bytes"].decode().splitlines()] if R else []


def strip_sv(r):
    return dict((k, v) for k, v in r.items() if not k.startswith("sv_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-pre", required=True)
    ap.add_argument("--tree-post", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=136)
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
    chk(hashlib.sha256(hpost).hexdigest() == SC.HF_POST_SHA, "--tree-post's HF.py is the scorer's HF_POST_SHA (csv1's tree)")
    chk(hashlib.sha256(hpre).hexdigest() == SC.HF_PRE_SHA and b"PATCH_SHADOWVOTE" not in hpre,
        "--tree-pre's HF.py is cvt9's (HF_PRE_SHA) and carries no PATCH_SHADOWVOTE")
    spre = os.path.join(post, "Optimizers", "HF.py.pre_shadowvote")
    chk(os.path.exists(spre) and open(spre, "rb").read() == hpre, "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_shadowvote")
    chk(open(os.path.join(pre, "build_network.py"), "rb").read() == open(os.path.join(post, "build_network.py"), "rb").read(),
        "the two trees' build_network.py are BYTE-IDENTICAL")
    nrec = a.steps // a.every
    print("\nsteps %d, PROBE every %d (%d records), PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026" % (a.steps, a.every, nrec, a.seed))

    print("\nRR0 DETERMINISM CONTROL (cvt9's tree, twice)")
    A1 = run_s(a, "pre_k01_a", pre, "scalar", None, None)
    A2 = run_s(a, "pre_k01_b", pre, "scalar", None, None)
    same(A1, A2, "RR0 unpatched k01 vs itself")
    REF = {"k01": A1}
    REF["MUTE"] = run_s(a, "pre_mute", pre, SC.SPEC["MUTE"], None, SC.VOTEW["MUTE"])
    REF["HEAD"] = run_s(a, "pre_head", pre, SC.SPEC["HEAD"], None, None)

    print("\nRR1 OFF (SHADOW_VOTE unset / empty == cvt9's tree, bitwise)")
    for arm in ("k01", "MUTE", "HEAD"):
        for val in ((None, "") if arm == "k01" else (None,)):
            lab = "unset" if val is None else "empty"
            R = run_s(a, "post_%s_%s" % (arm.lower(), lab), post, SC.SPEC[arm], val, SC.VOTEW[arm] or None)
            same(REF[arm], R, "RR1 %-5s SHADOW_VOTE %s" % (arm, lab))
            chk(lines(R, "SHADOW_VOTE") == ["SHADOW_VOTE: off"] and lines(R, "VOTE_W") == [SC.WITNESS_VW[arm]]
                and lines(R, "BETA_HOLD") == ["BETA_HOLD: off"] and lines(R, "WINDOW_HOLD") == ["WINDOW_HOLD: off"],
                "RR1 %-5s SHADOW_VOTE %s printed `SHADOW_VOTE: off` and its VOTE_W witness; holds off" % (arm, lab))

    print("\nRR2 THE INERT ON-PATHS == cvt9's k01, bitwise")
    SC.MODE["INERTNAT"] = ("natural", "shared")
    refr = recs_of(REF["k01"])
    for lab, arm, sv in (("INERT", "INERT", SC.SHADOWVOTE["INERT"]), ("INERTNAT", "INERTNAT", "natural:shared:" + SC.NAME_HEAD)):
        R = run_s(a, "post_%s" % lab.lower(), post, "scalar", sv, None)
        if R is None:
            continue
        want = SC.WITNESS_SV["INERT"] if arm == "INERT" else SC.witness_sv_of(("natural", "shared"))
        chk(lines(R, "SHADOW_VOTE") == [want] and lines(R, "VOTE_W") == ["VOTE_W: off"],
            "RR2 %-8s printed ONE SHADOW_VOTE line == the scorer's witness" % lab, (lines(R, "SHADOW_VOTE") or ["(none)"])[0][:100])
        B = REF["k01"]
        chk(B is not None and R["betas"] == B["betas"], "RR2 %-8s beta IDENTICAL to cvt9's k01 at every step" % lab)
        chk(B is not None and R["losses"] == B["losses"], "RR2 %-8s per-step loss IDENTICAL" % lab)
        chk(B is not None and R["weights_sha256"] == B["weights_sha256"], "RR2 %-8s every parameter and buffer after the last step BITWISE-IDENTICAL" % lab,
            R["weights_sha256"][:16])
        rr = recs_of(R)
        chk(len(rr) == nrec == len(refr) and [strip_sv(x) for x in rr] == refr,
            "RR2 %-8s probe records EQUAL to cvt9's k01 records after deleting the sv_* keys" % lab, "%d records" % len(rr))
        SC.SV_ARMS_BAK = SC.SV_ARMS
        if arm == "INERTNAT":
            SC.SV_ARMS = SC.SV_ARMS + ("INERTNAT",)
            SC.SPEC["INERTNAT"] = "scalar"
        ok, d = SC.bite_check(arm, rr, every=a.every, n_records=nrec)
        chk(ok, "RR2 %-8s the scorer's G-BITE passes on the real records (Lion, z_agg, sv_* audit, INERT identity)" % lab,
            str(dict((k, v) for k, v in d.items() if k.startswith("sv_") or k in ("bad_lion", "bad_zagg"))))
        SC.SV_ARMS = SC.SV_ARMS_BAK

    print("\nRR3 THE EXACT LOW STRINGS BITE ON THE REAL PATH")
    got = {}
    for arm in SC.LOW_ARMS:
        R = run_s(a, "post_%s" % arm.lower(), post, "scalar", SC.SHADOWVOTE[arm], None)
        if R is None:
            continue
        got[arm] = R
        chk(lines(R, "SHADOW_VOTE") == [SC.WITNESS_SV[arm]] and lines(R, "VOTE_W") == ["VOTE_W: off"],
            "RR3 %-9s printed ONE SHADOW_VOTE line == the scorer's witness" % arm, (lines(R, "SHADOW_VOTE") or ["(none)"])[0][:100])
        rr = recs_of(R)
        ok, d = SC.bite_check(arm, rr, every=a.every, n_records=nrec)
        chk(ok, "RR3 %-9s the scorer's G-BITE passes on the real records (floor applied, vote term, trace ordering, ...)" % arm,
            str(dict((k, v) for k, v in d.items() if k.startswith("sv_") or k in ("bad_lion", "bad_zagg"))))
        chk(len(rr) == nrec and all(abs(x["sv_a_applied"][0] - SC.FLOOR_ALPHA) <= 1e-6 * SC.FLOOR_ALPHA and x["sv_a_shared"][0] > 2 * SC.FLOOR_ALPHA
                                    for x in rr), "RR3 %-9s every record: applied == FLOOR_ALPHA while the shared step is > 2x it" % arm)
        chk(REF["k01"] is not None and R["weights_sha256"] != REF["k01"]["weights_sha256"],
            "RR3 %-9s final weights DIFFER from cvt9's k01 (the switch bites)" % arm)
        print("  NOTE RR3 %-9s beta %s cvt9's k01 over %d steps; last record ||h_vote|| %.3e ||h_other|| %.3e, ||dw|| %.3e ||dref|| %.3e"
              % (arm, "identical to" if R["betas"] == REF["k01"]["betas"] else "differs from", a.steps, rr[-1]["sv_hv_norm"][0],
                 rr[-1]["sv_ho_norm"][0], rr[-1]["sv_dw"][0], rr[-1]["sv_dref"][0]))
    if len(got) == 2:
        rs, rn = recs_of(got["SHADOWLOW"]), recs_of(got["NAIVELOW"])
        diff = sum(1 for x, y in zip(rs[1:], rn[1:]) if x["z_tensor"][49] != y["z_tensor"][49])
        chk(diff == nrec - 1, "RR3 50's vote term differs between SHADOWLOW and NAIVELOW on every record after the first", "%d/%d" % (diff, nrec - 1))
        ratio = [abs(x["z_tensor"][49]) / (abs(y["z_tensor"][49]) + 1e-30) for x, y in zip(rs[1:], refr[1:])]
        ratio_n = [abs(x["z_tensor"][49]) / (abs(y["z_tensor"][49]) + 1e-30) for x, y in zip(rn[1:], refr[1:])]
        print("  NOTE RR3 |z50| / cvt9 k01's |z50| (non-gating): SHADOWLOW median %.3g (min %.3g max %.3g); NAIVELOW median %.3g"
              % (sorted(ratio)[len(ratio) // 2], min(ratio), max(ratio), sorted(ratio_n)[len(ratio_n) // 2]))

    print("\nRR4 the controls' real records pass the scorer's G-BITE as their arms")
    for arm in ("k01", "MUTE", "HEAD"):
        R = REF.get(arm)
        if R:
            ok, d = SC.bite_check(arm, recs_of(R), every=a.every, n_records=nrec)
            chk(ok, "RR4 %-5s (cvt9's tree) G-BITE passes" % arm, "lion %d zagg %d worst %.1e" % (d["bad_lion"], d["bad_zagg"], d["worst_zagg"]))

    print("\n%s" % ("ALL PASS" if not RR.FAILED else "FAILURES: %r" % RR.FAILED))
    raise SystemExit(1 if RR.FAILED else 0)


if __name__ == "__main__":
    main()
