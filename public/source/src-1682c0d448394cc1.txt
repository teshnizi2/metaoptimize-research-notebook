"""INERTNESS AND BITE OF PATCH_BETAHOLD OVER A SHORT *REAL* RUN, FOR cvt4's EXACT STRINGS.  CORRECTIONS 237.

The REAL code path of a cvt4 run -- train.py's own parse_args, build_network (PlainNet18_c100), build_optimizer,
load_data (CIFAR-100, AUGMENT=1), the loss, HF.step with BETA_CLIP, PROBE and PROBE_TENSOR -- for --steps optimizer
steps on a GPU, from TWO trees:

  --tree-pre   cvt1's tree ($WS/harness_cvt1/cifar10, HF.py 3f2b98e1... = the cvt4 tree's HF.py.pre_betahold),
               used READ-ONLY (PYTHONDONTWRITEBYTECODE=1)
  --tree-post  the cvt4 tree (PATCH_BETAHOLD on top of it)

It IMPORTS cvt1's registered tests/test_voteweight_realrun.py UNEDITED and reuses its child process (`--child`), its
`run_variant` / `same` / `chk`; BETA_HOLD is passed through the environment run_variant copies.  The strings,
witnesses and the schedule come from analysis/cVT4_betahold_score.py, so the test and the scorer cannot disagree.

  RR0  DETERMINISM CONTROL: the unpatched tree run twice -> beta at every step, loss, probe.jsonl bytes, every final
       parameter and buffer identical (else the bitwise claims cannot be tested; FAIL).
  RR1  OFF: BETA_HOLD unset and BETA_HOLD empty, on k01's spec AND on HEAD's spec == the unpatched tree BITWISE (beta
       every step, loss every step, probe.jsonl bytes, every final parameter and buffer); each printed exactly one
       `BETA_HOLD: off` and one `VOTE_W: off`.
  RR2  THE EXACT STRINGS ON THE REAL PATH (HOLDLOW, HOLDSHARED, HOLDHIGH, HOLDHEAD) and a test-only `tri:100` whose
       peak falls inside the run: exactly one BETA_HOLD line == the scorer's witness; beta[1] after EVERY step == the
       registered hold (floor / tri schedule / beta[0]); at every probe record the scorer's own G-BITE halves hold
       (group 0 == Lion recomputed from the record; beta[1] and beta_pre[1] == the hold; bh_n == step + 2; bh_nat ==
       Lion recomputed for group 1; bh_held == beta[1]; bh_active non-decreasing).
  RR3  NON-VACUITY: HOLDLOW's beta[1] differs from the unpatched HEAD's at every step and its final weights differ;
       tri:100's beta[1] tracks the natural rise to its peak (float32 accumulation only) and departs after it, with
       bh_active > 0; HOLDSHARED, HOLDHIGH and HOLDHEAD each differ from the unpatched HEAD's beta on >= 1 step with
       bh_active > 0 (in 300 steps 50's own Lion step already disagrees with the hold on some updates).  Final
       weights of those four are REPORTED, not gated: a step-size difference of ~1e-7 per step on BatchNorm scales
       near 1.0 can sit below float32 resolution within 300 steps (first attempt, 237.4).
  RR4  HOLDSHARED vs MUTE (scalar, VOTE_W=layer4.1.bn2.weight:0, same tree): the exact-arithmetic identity of 237.3.
       beta IDENTICAL at every step (gating: HOLDSHARED's two entries and MUTE's one); loss and final weights
       REPORTED (non-gating: the float paths differ -- 53- vs 52-term python sums with an exact +0.0, np.exp on a
       0-d value vs a 2-vector, beta on GPU vs CPU).

RUN (on a GPU node; ~15 min):
  python3 tests/test_betahold_realrun.py --tree-pre $WS/harness_cvt1/cifar10 --tree-post $WS/harness_cvt4/cifar10 \\
      --steps 300 --every 10 --seed 90 --work /tmp/cvt4_realrun
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
SC = _load("cvt4_scorer", os.path.join(REPO, "analysis", "cVT4_betahold_score.py"))
chk, same, run_variant = RR.chk, RR.same, RR.run_variant
K01 = "scalar"
HEAD = SC.HEADSPEC
TRI100 = SC.NAME_HEAD + ":tri:100"


def run_bh(a, tag, tree, spec, bh, vw=None):
    old = os.environ.pop("BETA_HOLD", None)
    if bh is not None:
        os.environ["BETA_HOLD"] = bh
    try:
        return run_variant(a, tag, tree, spec, vw)
    finally:
        os.environ.pop("BETA_HOLD", None)
        if old is not None:
            os.environ["BETA_HOLD"] = old


def lines(R, prefix):
    return [ln for ln in (R["stdout"] if R else []) if ln.startswith(prefix)]


def hold_after(mode, P, n, beta0):
    if mode == "floor":
        return SC.LO
    if mode == "shared":
        return beta0
    return SC.v_of(n, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-pre", required=True)
    ap.add_argument("--tree-post", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=90)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    os.makedirs(a.work, exist_ok=True)
    os.environ.pop("BETA_HOLD", None)
    os.environ.pop("VOTE_W", None)
    print("DRIVER_SHA256 %s" % hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
    print("REGISTERED_TEST_SHA256 %s" % hashlib.sha256(open(RR.__file__, "rb").read()).hexdigest())
    print("SCORER_SHA256 %s" % hashlib.sha256(open(SC.__file__, "rb").read()).hexdigest())
    for t, lab in ((pre, "pre"), (post, "post")):
        hf = open(os.path.join(t, "Optimizers", "HF.py"), "rb").read()
        bn = open(os.path.join(t, "build_network.py"), "rb").read()
        print("TREE_%s %s HF %s build_network %s" % (lab.upper(), t, hashlib.sha256(hf).hexdigest(), hashlib.sha256(bn).hexdigest()))
    hpost = open(os.path.join(post, "Optimizers", "HF.py"), "rb").read()
    hpre = open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read()
    chk(hashlib.sha256(hpost).hexdigest() == SC.HF_POST_SHA, "--tree-post's HF.py is the scorer's HF_POST_SHA (cvt4's tree)")
    chk(hashlib.sha256(hpre).hexdigest() == SC.HF_PRE_SHA and b"PATCH_BETAHOLD" not in hpre,
        "--tree-pre's HF.py is cvt1's (HF_PRE_SHA) and carries no PATCH_BETAHOLD")
    bhpre = os.path.join(post, "Optimizers", "HF.py.pre_betahold")
    chk(os.path.exists(bhpre) and open(bhpre, "rb").read() == hpre,
        "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_betahold")
    chk(open(os.path.join(pre, "build_network.py"), "rb").read() == open(os.path.join(post, "build_network.py"), "rb").read(),
        "the two trees' build_network.py are BYTE-IDENTICAL")
    print("\nsteps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026" % (a.steps, a.every, a.seed))

    print("\nRR0 DETERMINISM CONTROL (unpatched tree, twice)")
    A1 = run_bh(a, "pre_k01_a", pre, K01, None)
    A2 = run_bh(a, "pre_k01_b", pre, K01, None)
    same(A1, A2, "RR0 unpatched k01 vs itself")
    H0 = run_bh(a, "pre_head", pre, HEAD, None)

    print("\nRR1 OFF")
    for spec, ref, lab in ((K01, A1, "k01"), (HEAD, H0, "HEAD")):
        for val in (None, ""):
            R = run_bh(a, "post_%s_%s" % (lab.lower(), "unset" if val is None else "empty"), post, spec, val)
            same(ref, R, "RR1 %-4s BETA_HOLD %s" % (lab, "unset" if val is None else "empty"))
            chk(lines(R, "BETA_HOLD") == ["BETA_HOLD: off"] and lines(R, "VOTE_W") == ["VOTE_W: off"],
                "RR1 %-4s BETA_HOLD %s printed exactly `BETA_HOLD: off` and `VOTE_W: off`" % (lab, "unset" if val is None else "empty"))

    print("\nRR2 THE EXACT STRINGS BITE ON THE REAL PATH")
    got = {}
    for arm in SC.HELD + ("TRI100",):
        bh = SC.BETAHOLD.get(arm) or TRI100
        mode = bh.split(":", 1)[1].split(":")[0]
        P = int(bh.rsplit(":", 1)[1]) if mode == "tri" else None
        wit = SC.WITNESS_BH.get(arm) or SC.witness_bh_of("tri", 100)
        R = run_bh(a, "post_%s" % arm.lower(), post, HEAD, bh)
        if R is None:
            continue
        got[arm] = R
        chk(lines(R, "BETA_HOLD") == [wit] and lines(R, "VOTE_W") == ["VOTE_W: off"],
            "RR2 %-10s printed exactly ONE BETA_HOLD line == the scorer's witness, and VOTE_W: off" % arm)
        bad_step = sum(1 for n, b in enumerate(R["betas"], 1)
                       if b[1] != hold_after(mode, P, n, b[0]))
        chk(len(R["betas"]) == a.steps and bad_step == 0,
            "RR2 %-10s beta[1] after EVERY one of %d steps == the registered hold" % (arm, a.steps), "%d bad" % bad_step)
        recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
        bad_lion = bad_sched = bad_bh = 0
        prev = -1
        for r in recs:
            s = r["step"]
            nat0, tie0 = SC.lion_natural(r["beta_pre"][0][0], r["mom_pre"][0][0], r["z_agg"][0][0])
            bad_lion += (not tie0) and abs(nat0 - r["beta"][0]) > SC.BH_TOL
            h2 = hold_after(mode, P, s + 2, r["beta"][0])
            h1 = hold_after(mode, P, s + 1, r["beta_pre"][0][0])
            bad_sched += abs(r["beta"][1] - h2) > SC.BH_TOL or abs(r["beta_pre"][0][1] - h1) > SC.BH_TOL
            nat1, tie1 = SC.lion_natural(r["beta_pre"][0][1], r["mom_pre"][0][1], r["z_agg"][0][1])
            ok = (r.get("bh_n") == s + 2 and abs(r.get("bh_held", 1e9) - r["beta"][1]) <= SC.BH_TOL
                  and (tie1 or abs(r.get("bh_nat", 1e9) - nat1) <= SC.BH_TOL) and r.get("bh_active", -2) >= prev)
            bad_bh += not ok
            prev = r.get("bh_active", prev)
        chk(len(recs) == a.steps // a.every and bad_lion == 0 and bad_sched == 0 and bad_bh == 0,
            "RR2 %-10s at every probe record: complement == Lion; beta/beta_pre[1] == hold; bh_n / bh_nat / bh_held / bh_active audit"
            % arm, "%d records; lion %d sched %d bh %d; last bh_active %s" % (len(recs), bad_lion, bad_sched, bad_bh, prev))

    print("\nRR3 NON-VACUITY")
    L = got.get("HOLDLOW")
    if L and H0:
        chk(all(x[1] != y[1] for x, y in zip(L["betas"], H0["betas"])),
            "RR3 HOLDLOW beta[1] differs from the unpatched HEAD's at EVERY step")
        chk(L["weights_sha256"] != H0["weights_sha256"], "RR3 HOLDLOW's final parameters differ from the unpatched HEAD's",
            "%s vs %s" % (L["weights_sha256"][:12], H0["weights_sha256"][:12]))
    T = got.get("TRI100")
    if T and H0:
        pre_pk = max(abs(x[1] - y[1]) for x, y in zip(T["betas"][:101], H0["betas"][:101]))
        post_pk = min(abs(x[1] - y[1]) for x, y in zip(T["betas"][150:], H0["betas"][150:]))
        last = json.loads(T["probe_bytes"].decode().splitlines()[-1])
        chk(pre_pk <= 1e-3 and post_pk > 0.05 and last.get("bh_active", 0) > 0,
            "RR3 tri:100 tracks the natural rise to its peak (max |diff| %.1e: float32 accumulation), then departs "
            "(min |diff| from step 150 %.3f), bh_active %s > 0" % (pre_pk, post_pk, last.get("bh_active")))
        print("  NOTE RR3 tri:100 final weights %s the unpatched HEAD's (non-gating)"
              % ("differ from" if T["weights_sha256"] != H0["weights_sha256"] else "are identical to"))
    for arm in ("HOLDSHARED", "HOLDHIGH", "HOLDHEAD"):
        R = got.get(arm)
        if R and H0:
            nd = sum(1 for x, y in zip(R["betas"], H0["betas"]) if x != y)
            last = json.loads(R["probe_bytes"].decode().splitlines()[-1])
            chk(nd >= 1 and last.get("bh_active", 0) > 0,
                "RR3 %-10s beta differs from the unpatched HEAD's on >= 1 step and bh_active > 0" % arm,
                "%d of %d steps differ; bh_active %s" % (nd, a.steps, last.get("bh_active")))
            print("  NOTE RR3 %-10s final weights %s the unpatched HEAD's (non-gating)"
                  % (arm, "differ from" if R["weights_sha256"] != H0["weights_sha256"] else "are identical to"))

    print("\nRR4 HOLDSHARED vs MUTE (the exact-arithmetic identity, 237.3)")
    S = got.get("HOLDSHARED")
    MU = run_bh(a, "post_mute", post, K01, None, vw="layer4.1.bn2.weight:0")
    if S and MU:
        chk(lines(MU, "VOTE_W") == ["VOTE_W: on type=scalar items=50:layer4.1.bn2.weight:w=0.0:group=0:groupsize=53"]
            and lines(MU, "BETA_HOLD") == ["BETA_HOLD: off"], "RR4 MUTE printed cvt1's MUTE witness and BETA_HOLD: off")
        chk(len(S["betas"]) == len(MU["betas"]) and all(x[0] == x[1] == y[0] for x, y in zip(S["betas"], MU["betas"])),
            "RR4 beta IDENTICAL at every step: HOLDSHARED [b0, b1] == MUTE's single b (gating)")
        first = next((k for k, (x, y) in enumerate(zip(S["losses"], MU["losses"])) if x != y), None)
        mx = max(abs(x - y) for x, y in zip(S["losses"], MU["losses"]))
        print("  NOTE RR4 loss: %s; max |diff| %.3e; final weights %s (non-gating)"
              % ("identical at every step" if first is None else "first differs at step %d" % (first + 1), mx,
                 "BITWISE IDENTICAL" if S["weights_sha256"] == MU["weights_sha256"] else "differ"))

    print("\n%s" % ("ALL PASS" if not RR.FAILED else "FAILURES: %r" % RR.FAILED))
    raise SystemExit(1 if RR.FAILED else 0)


if __name__ == "__main__":
    main()
