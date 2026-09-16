"""INERTNESS AND BITE OF PATCH_VOTEWEIGHT FOR cvt2's EXACT VOTE_W STRINGS, OVER A SHORT *REAL* RUN.
CORRECTIONS 233.

cvt2 runs from cvt1's isolated tree ($WS/harness_cvt1, UNCHANGED) with five weights on the twin
(layer4.1.bn1.weight x 13 / 33 / 152 / 691 / 2000) inside HEAD's complement, plus the two control arms
(k01, HEAD) with VOTE_W unset.  cvt1's tests/test_voteweight_realrun.py proved the patch for x691, x0 and
x0.1 only.  This file re-proves it for EXACTLY the strings cvt2 registers, on the real code path, by
IMPORTING the registered test module UNEDITED and reusing its child process, its run_variant() and its
bitwise comparison (every variant runs train.py's own parse_args, build_network, build_optimizer and
load_data for --steps optimizer steps on a GPU, from its own tree, cudnn deterministic).

The VOTE_W strings and witness lines are READ FROM THE cvt2 SCORER (--scorer), not re-typed here, and are
printed as `LADDER <arm> <VOTE_W> <witness>` lines the launcher's guard compares with the scorer it runs.

  RR0  DETERMINISM CONTROL: the unpatched tree (k01) run twice -> bitwise identical (else nothing is testable).
  RR1  INERTNESS for the batch's control arms: VOTE_W UNSET (as the launcher submits k01 and HEAD) on the
       patched tree == the unpatched tree, bitwise: beta at every step, per-step loss, probe.jsonl bytes, and
       the sha256 of every parameter and buffer after the last step.  Also VOTE_W EMPTY on HEAD.
  RR2  IDENTITY on HEAD's spec: layer4.1.bn1.weight:1 == unpatched HEAD, bitwise (same four equalities).
  RR3  BITE, for EACH of the five ladder strings on HEAD's spec:
         (a) the run prints EXACTLY the scorer's registered witness line, once;
         (b) at every probe record, z_agg[g] == sum_{i in g} w_i z_tensor_i and mom_pre[g] == sum w_i m_tensor_i
             (rel 1e-5 of sum |w_i term_i|), w = the registered K on idx 47 only;
         (c) the ISOLATED group's z_agg (idx 50 alone) is BITWISE equal to the unpatched HEAD run's at every
             record (the weight touched the complement only), while the complement's z_agg differs from the
             unpatched run's on >= 1 record, by (K - 1) * z_tensor_47 (rel 1e-5);
         (d) the non-vacuity count: records where the weighted and unweighted complement sums differ by > 1e-2
             relative (the scorer's INFORM) are counted and printed.
       Disclosed limit (as 227.4): in a few hundred steps no weight flips a meta-gradient sign, so beta and final
       weights of the intervened arms may equal the control's; the bite is proved at the level of z.

RUN (on a GPU node; ~6 min):
  python3 tests/test_voteweight_ladder_realrun.py --tree-pre $WS/harness_cpl1/cifar10 \\
      --tree-post $WS/harness_cvt1/cifar10 --scorer analysis/cVT2_injectladder_score.py --steps 300 --work /tmp/x
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("test_voteweight_realrun", os.path.join(HERE, "test_voteweight_realrun.py"))
RR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RR)
chk, same, run_variant, FAILED = RR.chk, RR.same, RR.run_variant, RR.FAILED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-pre", required=True)
    ap.add_argument("--tree-post", required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=87)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    os.makedirs(a.work, exist_ok=True)
    sp = importlib.util.spec_from_file_location("cvt2sc", os.path.abspath(a.scorer))
    S = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(S)
    print("REGISTERED_TEST_SHA256 %s" % hashlib.sha256(open(os.path.join(HERE, "test_voteweight_realrun.py"), "rb").read()).hexdigest())
    print("THIS_TEST_SHA256 %s" % hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
    print("SCORER %s %s" % (os.path.abspath(a.scorer), hashlib.sha256(open(a.scorer, "rb").read()).hexdigest()))
    for t, lab in ((pre, "pre"), (post, "post")):
        hf = open(os.path.join(t, "Optimizers", "HF.py"), "rb").read()
        bn = open(os.path.join(t, "build_network.py"), "rb").read()
        print("TREE_%s %s HF %s build_network %s" % (lab.upper(), t, hashlib.sha256(hf).hexdigest(), hashlib.sha256(bn).hexdigest()))
    for arm in S.ARMS:
        print("LADDER %s %s %s" % (arm, S.VOTEW[arm] or "unset", S.WITNESS[arm]))
    chk(b"PATCH_VOTEWEIGHT" not in open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read()
        and b"PATCH_VOTEWEIGHT" in open(os.path.join(post, "Optimizers", "HF.py"), "rb").read(),
        "--tree-pre is unpatched, --tree-post carries PATCH_VOTEWEIGHT")
    vwpre = os.path.join(post, "Optimizers", "HF.py.pre_voteweight")
    chk(os.path.exists(vwpre) and open(vwpre, "rb").read() == open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read(),
        "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_voteweight")
    chk(open(os.path.join(pre, "build_network.py"), "rb").read() == open(os.path.join(post, "build_network.py"), "rb").read(),
        "the two trees' build_network.py are BYTE-IDENTICAL")
    chk(hashlib.sha256(open(os.path.join(post, "Optimizers", "HF.py"), "rb").read()).hexdigest() == S.HF_POST_SHA,
        "--tree-post's HF.py is the scorer's HF_POST_SHA (cvt1's tree, unchanged)")
    HEAD, K01 = S.SPEC["HEAD"], S.SPEC["k01"]
    print("\nsteps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026" % (a.steps, a.every, a.seed))

    print("\nRR0 DETERMINISM CONTROL (unpatched tree, twice)")
    A1 = run_variant(a, "pre_k01_a", pre, K01, None)
    A2 = run_variant(a, "pre_k01_b", pre, K01, None)
    same(A1, A2, "RR0 unpatched k01 vs itself")
    H0 = run_variant(a, "pre_head", pre, HEAD, None)

    print("\nRR1 INERTNESS for the control arms (VOTE_W unset, as submitted)")
    E1 = run_variant(a, "post_k01_unset", post, K01, None)
    same(A1, E1, "RR1 k01  VOTE_W unset")
    E2 = run_variant(a, "post_head_unset", post, HEAD, None)
    same(H0, E2, "RR1 HEAD VOTE_W unset")
    for tag, R in (("k01", E1), ("HEAD", E2)):
        chk(R is not None and [x for x in R["stdout"] if x.startswith("VOTE_W")] == [S.WITNESS[tag]],
            "RR1 %-4s printed exactly the registered witness %r" % (tag, S.WITNESS[tag]))
    same(H0, run_variant(a, "post_head_empty", post, HEAD, ""), "RR1 HEAD VOTE_W empty")

    print("\nRR2 IDENTITY on HEAD's spec")
    same(H0, run_variant(a, "post_head_id", post, HEAD, "layer4.1.bn1.weight:1"), "RR2 HEAD layer4.1.bn1.weight:1")

    print("\nRR3 BITE, for every registered ladder string")
    refr = [json.loads(x) for x in H0["probe_bytes"].decode().splitlines()] if H0 else []
    g0 = [i for i in range(53) if i != S.IDX_HEAD - 1]
    t0 = S.IDX_TWIN - 1
    for arm in S.LADDER:
        R = run_variant(a, "post_" + arm.lower(), post, HEAD, S.VOTEW[arm])
        if R is None:
            continue
        K = S.WEIGHTS[arm][S.IDX_TWIN]
        wl = [x for x in R["stdout"] if x.startswith("VOTE_W")]
        chk(wl == [S.WITNESS[arm]], "RR3 %-6s printed exactly ONE VOTE_W line == the scorer's witness" % arm, repr(wl[:2]))
        recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
        badz = badm = 0
        worst = 0.0
        iso_same = comp_diff = delta_ok = inform = 0
        for r, u in zip(recs, refr):
            zt, mt = r["z_tensor"], r["m_tensor"]
            for gi, g in enumerate((g0, [S.IDX_HEAD - 1])):
                w = [K if i == t0 else 1.0 for i in g]
                sz = sum(abs(wi * zt[i]) for wi, i in zip(w, g)) + 1e-30
                sm = sum(abs(wi * mt[i]) for wi, i in zip(w, g)) + 1e-30
                ez = abs(sum(wi * zt[i] for wi, i in zip(w, g)) - r["z_agg"][0][gi]) / sz
                em = abs(sum(wi * mt[i] for wi, i in zip(w, g)) - r["mom_pre"][0][gi]) / sm
                worst = max(worst, ez, em)
                badz += ez > 1e-5
                badm += em > 1e-5
            iso_same += r["z_agg"][0][1] == u["z_agg"][0][1]
            comp_diff += r["z_agg"][0][0] != u["z_agg"][0][0]
            sz0 = sum(abs((K if i == t0 else 1.0) * zt[i]) for i in g0) + 1e-30
            delta_ok += abs((r["z_agg"][0][0] - u["z_agg"][0][0]) - (K - 1.0) * zt[t0]) / sz0 <= 1e-5
            inform += abs((K - 1.0) * zt[t0]) / sz0 > 1e-2
        n = len(recs)
        chk(n > 0 and n == len(refr) and badz == 0 and badm == 0,
            "RR3 %-6s z_agg / mom_pre == sum_i w_i term_i in BOTH groups at every record (w = %g on idx 47)" % (arm, K),
            "%d records, worst rel %.2e" % (n, worst))
        chk(n > 0 and iso_same == n, "RR3 %-6s the ISOLATED group's z_agg is BITWISE the unpatched HEAD's at every record" % arm,
            "%d / %d" % (iso_same, n))
        chk(comp_diff >= 1 and delta_ok == n,
            "RR3 %-6s the complement's z_agg differs from unpatched HEAD's by (K-1)*z_47 at every record" % arm,
            "%d of %d records differ; %d match (K-1)*z_47" % (comp_diff, n, delta_ok))
        print("  NOTE RR3 %-6s informative records (weighted vs unweighted complement sum differ > 1e-2 rel): %d / %d" % (arm, inform, n))

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %r" % FAILED))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
