"""THE HARNESS-NULL PROOF FOR `caw2` (CORRECTIONS 290), ON THE REAL GPU PATH.

`caw2` (the successor of the withdrawn `caw1`) adds NO harness code.  But two of its four pairings have never run on
ResNet18_c100 / CIFAR-100 in this campaign (SGDm + Adam meta; AdamW + Lion meta), none of those two has ever run with
PROBE_TENSOR=1, and the X cell runs the AdamW base at wd 1.0, a value no landed run has carried.  The batch's readings
rest on the HARNESS-NULL being excluded: that a run is NOT what its ARGS say (a base or meta alg silently falling back,
the decay not entering the step, two arms being the same experiment).  A NO-COLLAPSE produced by such a run would be a
harness artefact; this job is what separates the two.

REUSE, NOT COPY.  The per-step verifying CHILD (every base and meta update checked against its formula, bitwise) and
`run_arm` are caw1's tests/test_caw1_realrun.py, imported UNEDITED (its sha is pinned below and printed): that child is
already generic over SGDm / AdamW bases and Adam / Lion metas and was proven on the L4 by job 5079187 (ALL PASS).  Only
the orchestration (which arms, which checks per pairing) is new here.

On cwd1's tree, UNCHANGED, with the launcher's own environment, each arm's OWN ARGS (through train.py's own
parse_args and its -1 -> None conversion), 300 real CIFAR-100 steps per variant:
  RC1  BASE, BITWISE, WITH THE PASSED wd (0.1, or 1.0 on X), every step and tensor: SGDm (w', h', m') or AdamW (w', h',
       m', v' and the lambda clock).  NON-VACUITY: the same run against the formula at the OTHER decay FAILS; an AdamW
       run FAILS the SGDm formula (the base did not fall back); args_base carries the registered alg and wd.
  RC2  META, BITWISE: Adam (beta, M, T) or Lion (beta, M) at every step; the same run FAILS the other meta's formula.
  RC3  GRAIN: scalar arms carry ONE step size, layerwise arms 62.
  RC4  WITNESSES: the five `off` lines, and ONE `PROBE_TENSOR: on ...` line == the design's witness for the arm (with
       the ONE token every=<period> substituted, since this job probes every 10 steps) followed by ` dir=` -- this is
       where the Lion_beta2 / momentum_param tail is MEASURED on all four pairings.
  RC5  THE VOTE DECOMPOSITION DOM_C READS: on every record of every scalar arm, sum_i L_i equals the harness's applied
       momentum term (Adam: b1/(1-b1)*m_tensor_i + z_i vs b1*mom_pre + z_agg; Lion: b2*m_tensor_i + (1-b2)*z_i vs
       b2*mom_pre + (1-b2)*z_agg) within 1e-3 of sum_i |L_i|, with its sign.
  RC6  DETERMINISM: AS run twice is bitwise identical.
  RC7  THE NINE ARMS ARE DISTINCT EXPERIMENTS: final weights PAIRWISE DISTINCT.
  RC8  THE DECAY ROUTE: tensor-steps at which the decay term changes the float32 update are COUNTED per arm; at a
       TEST-ONLY alpha0 1e-2 AS (wd 0.1) and XS (wd 1.0) still satisfy the AdamW formula bitwise, the decay reaches
       the weights, and their final weights DIFFER (the X cell's dose axis bites).

    python3 tests/test_caw2_realrun.py --tree $WS/harness_cwd1/cifar10 --steps 300 --every 10 --seed 160 --work /tmp/caw2_rr
"""
import argparse
import hashlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
KEYS_OFF = ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "COMP_HOLD", "REST_HOLD", "WINDOW_HOLD", "DECAY_MASK")
CAW1_TEST_SHA = "72bcb3e17ab7a8dfd41e2057671d237c46e1c3f5c4a5b09e0cbdac9959483f5a"   # tests/test_caw1_realrun.py @ 16faa5f


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sh(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def decomposition(recs, meta):
    """RC5 on real records: -> (n, bad_sum, bad_sign)."""
    n = bad = bads = 0
    for r in recs:
        if "z_tensor" not in r:
            continue
        n += 1
        mp, b2 = r["pt_mp"], r["pt_b2"]
        z, m = r["z_tensor"], r["m_tensor"]
        if meta == "Adam":
            L = [mp / (1 - mp) * mi + zi for mi, zi in zip(m, z)]
            app = mp * r["mom_pre"][0][0] + r["z_agg"][0][0]
        else:
            L = [b2 * mi + (1 - b2) * zi for mi, zi in zip(m, z)]
            app = b2 * r["mom_pre"][0][0] + (1 - b2) * r["z_agg"][0][0]
        s = sum(L)
        scale = sum(abs(x) for x in L)
        bad += not (abs(s - app) <= 1e-3 * scale or abs(s - app) <= 1e-30)
        bads += ((s > 0) != (app > 0)) and abs(app) > 1e-3 * scale
    return n, bad, bads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=160)
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    D = _load("caw2_design", os.path.join(REPO, "analysis", "caw2_design.py"))
    T1 = _load("test_caw1_realrun", os.path.join(HERE, "test_caw1_realrun.py"))
    RR = _load("test_grouphold_realrun", os.path.join(HERE, "test_grouphold_realrun.py"))
    chk = RR.chk
    a.tree = os.path.abspath(a.tree)
    os.makedirs(a.work, exist_ok=True)
    for k in KEYS_OFF:
        os.environ.pop(k, None)

    print("BATCH caw2")
    print("DRIVER_SHA256 %s" % sh(os.path.abspath(__file__)))
    print("REUSED_CHILD_SHA256 %s" % sh(T1.__file__))
    print("DESIGN_SHA256 %s" % sh(D.__file__))
    print("CWD_DESIGN_SHA256 %s" % sh(D.W.__file__))
    print("REGISTERED_TEST_SHA256 %s" % sh(RR.__file__))
    print("TREE %s HF %s build_network %s train %s" % (a.tree, sh(os.path.join(a.tree, "Optimizers", "HF.py")),
                                                        sh(os.path.join(a.tree, "build_network.py")),
                                                        sh(os.path.join(a.tree, "train.py"))))
    print("ARMS %s" % ",".join("%s=%s+%s/%s/%s" % (x, D.BASE_OF[x], D.META_OF[x], D.SPEC[x], D.WD[x]) for x in D.ARMS))
    chk(sh(T1.__file__) == CAW1_TEST_SHA, "the reused child tests/test_caw1_realrun.py is caw1's registered file, UNEDITED")
    chk(sh(os.path.join(a.tree, "Optimizers", "HF.py")) == D.HF_SHA, "the tree's HF.py is cwd1's, UNCHANGED (94aedc33...)")
    chk(sh(os.path.join(a.tree, "build_network.py")) == D.BN_SHA, "the tree's build_network.py is cvt8's (c7998883...)")
    chk(sh(D.W.__file__) == D.CWD_DESIGN_SHA, "analysis/cwd_design.py is the registered file, UNEDITED (RULE 16)")
    print("\nsteps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026, HIER=none, net %s"
          % (a.steps, a.every, a.seed, D.NET))

    runs = {}
    print("\nRC1-RC4 EACH ARM'S OWN ARGS, THROUGH train.py's parse_args, ON THE REAL GPU PATH")
    for arm in D.ARMS:
        V = T1.run_arm(a, D, "arm_" + arm, arm)
        runs[arm] = V
        if V is None:
            chk(False, "RC %s ran" % arm)
            continue
        base, meta = D.BASE_OF[arm], D.META_OF[arm]
        s = V["stats"]
        chk(V["steps"] == a.steps and V["alg_base"] == base and V["alg_meta"] == meta and V["wd_seen"] == D.WDF[arm],
            "RC1 %s the optimiser's own args carry base %s, meta %s and wd %r" % (arm, base, meta, D.WDF[arm]))
        nv = 4 if base == "AdamW" else 3
        chk(s["expect"][:nv] == [0] * nv and "unexpected_base" not in s,
            "RC1 %s %s base: w', h', m'%s BITWISE the formula with THE PASSED wd %s at every one of %d steps, every"
            " tensor" % (arm, base, ", v'" if base == "AdamW" else "", D.WD[arm], a.steps), str(s["expect"]))
        chk(sum(s["wrongwd"]) > 0,
            "RC1 %s NON-VACUITY: the same run checked against the formula at the OTHER decay FAILS" % arm,
            "wrongwd %s" % s["wrongwd"])
        if base == "AdamW":
            chk(s.get("lam_ok") == a.steps, "RC1 %s the second-moment bias-correction clock lambda advanced as b2^t" % arm)
            chk(s["sgdm"][0] > 0 and s["sgdm"][2] > 0,
                "RC1 %s THE HARNESS-NULL IS EXCLUDED: the run FAILS the SGDm base formula (w and m)" % arm, str(s["sgdm"]))
        chk(s["meta_expect"] == 0 and "unexpected_meta" not in s,
            "RC2 %s %s meta: beta (and its moments) BITWISE the formula at every step" % (arm, meta))
        chk(s["meta_wrongalg"] > 0, "RC2 %s NON-VACUITY: the same run FAILS the other meta alg's formula" % arm,
            "wrongalg %d" % s["meta_wrongalg"])
        want_n = 1 if D.SPEC[arm] == "scalar" else D.NTENS
        chk(V["stepsize_type"] == D.SPEC[arm] and V["n_beta"] == want_n and V["len_beta_list"] == 1,
            "RC3 %s grain: stepsize_type=%s with %d step size(s)" % (arm, D.SPEC[arm], want_n),
            "%s / %d" % (V["stepsize_type"], V["n_beta"]))
        offs = dict((k, [ln for ln in V["stdout"] if ln.startswith(k)]) for k in KEYS_OFF)
        chk(all(offs[k.split(":")[0]] == [k] for k in D.OFF_LINES) and not offs["COMP_HOLD"] and not offs["WINDOW_HOLD"],
            "RC4 %s printed exactly the five `off` witnesses and no COMP_HOLD / WINDOW_HOLD line" % arm)
        pt = [ln for ln in V["stdout"] if ln.startswith("PROBE_TENSOR:")]
        wit = D.pt_witness_prefix(arm).replace("every=%d " % D.PROBE, "every=%d " % a.every, 1)
        chk(len(pt) == 1 and pt[0].startswith(wit + " dir="),
            "RC4 %s printed ONE PROBE_TENSOR line == the design's witness (every=%d here) `%s dir=...`" % (arm, a.every, wit),
            (pt or ["(none)"])[0][:170])
        n_rec = len(V["records"])
        chk(n_rec == a.steps // a.every and all("z_tensor" in r and len(r["z_tensor"]) == D.NTENS for r in V["records"]),
            "RC4 %s wrote %d probe records, each with a 62-tensor z_tensor" % (arm, a.steps // a.every), "%d" % n_rec)
        print("  RC8 %s decay term changed the float32 weight update at %d of %d tensor-steps (alpha0 %s, wd %s)"
              % (arm, s["decay_bites"], s["tensor_steps"], D.A0, D.WD[arm]))

    print("\nRC5 THE VOTE DECOMPOSITION DOM_C READS, ON REAL RECORDS (every scalar arm, both meta algs)")
    for arm in D.SCALAR_ARMS:
        V = runs.get(arm)
        if V is None:
            continue
        n, bad, bads = decomposition(V["records"], D.META_OF[arm])
        chk(n == a.steps // a.every and bad == 0 and bads == 0,
            "RC5 %s (%s meta) sum_i L_i == the harness's applied momentum term (within 1e-3 of sum|L_i|) and has its sign"
            " on every record" % (arm, D.META_OF[arm]), "%d records, %d off, %d sign" % (n, bad, bads))

    print("\nRC6 DETERMINISM (AS twice)")
    V2 = T1.run_arm(a, D, "arm_AS_again", "AS")
    chk(V2 is not None and runs.get("AS") is not None and V2["weights_sha256"] == runs["AS"]["weights_sha256"],
        "RC6 AS run twice: identical final weights")

    print("\nRC7 THE NINE ARMS ARE DISTINCT EXPERIMENTS")
    shas = dict((k, v["weights_sha256"]) for k, v in runs.items() if v is not None)
    chk(len(shas) == len(D.ARMS) and len(set(shas.values())) == len(D.ARMS),
        "RC7 the nine arms' final weights are PAIRWISE DISTINCT", " ".join("%s %s" % (k, v[:10]) for k, v in sorted(shas.items())))

    print("\nRC8 THE DECAY ROUTE AT A TEST-ONLY alpha0 1e-2 (NOT a batch value)")
    B1 = T1.run_arm(a, D, "big_AS", "AS", alpha0="1e-2")
    B2 = T1.run_arm(a, D, "big_XS", "XS", alpha0="1e-2")
    for tag, T in (("AS", B1), ("XS", B2)):
        chk(T is not None and T["stats"]["expect"] == [0, 0, 0, 0] and T["stats"]["meta_expect"] == 0
            and T["stats"]["decay_bites"] > 0,
            "RC8 %s at alpha0 1e-2 the AdamW formula holds bitwise and the decay term reaches the weights" % tag,
            str(T["stats"] if T else None))
    chk(B1 is not None and B2 is not None and B1["weights_sha256"] != B2["weights_sha256"],
        "RC8 at alpha0 1e-2 the wd-0.1 (AS) and wd-1.0 (XS) runs' final weights DIFFER")

    fails = list(RR.FAILED)
    print("\n%s" % ("ALL PASS" if not fails else "FAILURES: %r" % fails))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
