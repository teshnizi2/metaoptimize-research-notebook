"""BITE OF PATCH_GROUPHOLD ON ONE-TENSOR GROUPS AND ON A THREE-GROUP PARTITION, OVER A SHORT *REAL* ResNet18_c100 RUN,
FOR cvt10's EXACT STRINGS.  CORRECTIONS 258.

cvt10 adds NO harness code: it runs from cvt8's tree ($WS/harness_cvt8/cifar10, HF.py 5197dc2e..., unchanged) with
cvt8's runner.  PATCH_GROUPHOLD (243) names a group by its COMPLETE membership (one or more names) and resolves its
index at construction, but every landed GROUP_HOLD run held a THREE-tensor group 1 of a TWO-group partition.  cvt10 holds
one-tensor groups ([61,1]) and group 1 of a THREE-group partition ([59,1,2]).  This test proves, before launch, on the
REAL code path, that those strings do what the scorer's G-BITE will require -- and that the scorer's bite_check reads
three-group probe records.  It is NOT an inertness proof (no new patch; cvt7 / cvt8 proved OFF bitwise, 243.4 / 248.4).

It IMPORTS cvt7's registered tests/test_grouphold_realrun.py UNEDITED and reuses its child process (`--child`: train.py's
own parse_args, build_network, build_optimizer, load_data, CIFAR-100, AUGMENT=1, the loss, HF.step with BETA_CLIP, PROBE
and PROBE_TENSOR), `run_variant`, `same` and `chk`.  Strings, witnesses and the schedule come from
analysis/cVT10_onevsthree_score.py.

  RB0  DETERMINISM CONTROL: ONE50 (free) run twice -> identical beta, loss, probe bytes, final weights.
  RB1  THE FIVE REGISTERED HELD STRINGS (HOLDBIG3, ONE50BIG, ONE59BIG, ONE53BIG, ISOSPLIT) on their registered specs:
       exactly one GROUP_HOLD line == the scorer's witness, `REST_HOLD: off`, `BETA_HOLD: off`, `VOTE_W: off`, no
       COMP_HOLD / WINDOW_HOLD line; beta[1] after EVERY step == tri:9428; the scorer's OWN bite_check on the real
       records (every=10, 30 records) passes: every unheld group == Lion recomputed, group 1 on schedule in beta and
       beta_pre, gh_* audit, no foreign key.
  RB2  THE FOUR FREE SPECS NEVER RUN (ONE59, ONE53, and ISOSPLIT's [59,1,2] partition free) plus ONE50: `GROUP_HOLD: off`
       and the other off-lines; the scorer's bite_check passes as a control for ONE59 / ONE53 / ONE50; the three-group
       free run is Lion on every group at every record (checked here directly).
  RB3  NON-VACUITY ON THE REAL PATH, test-only strings that depart at step 1: `layer4.0.shortcut.1.weight:floor` on ONE53's
       spec and `layer4.0.shortcut.1.weight+layer4.1.bn2.weight:floor` on ISOSPLIT's spec (GROUP 2 -- the index is
       resolved by membership): the witness names the right group; the held group sits at -15 after every step and
       differs from the free run's at every step; gh_active > 0; every other group Lion; the free runs read as the
       registered held arms by bite_check FAIL (the floor string's records do too).  The registered tri:9428 strings'
       departures from their free twins over 300 steps are REPORTED, not gated (the triangle is the max-rate rise a free
       group also takes early on, 248.4).
  RB4  LOUDNESS on the live model, CPU, inside the job: a GROUP_HOLD naming one carrier on ISO's [59,3] spec, a pair
       that is not a group, and the three carriers on ONE50's spec each raise ValueError at construction.

RUN (on a GPU node, as a Slurm job; bin/cVT10_realrun_bite.sbatch):
  python3 tests/test_cvt10_hold_realrun.py --tree $WS/harness_cvt8/cifar10 --steps 300 --every 10 --seed 120 --work /tmp/x
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
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
SC = _load("cvt10_scorer", os.path.join(REPO, "analysis", "cVT10_onevsthree_score.py"))
chk, same, run_variant = RR.chk, RR.same, RR.run_variant
T53_FLOOR = "layer4.0.shortcut.1.weight:floor"
TSPLIT_G2_FLOOR = "layer4.0.shortcut.1.weight+layer4.1.bn2.weight:floor"


def lines(R, prefix):
    return [ln for ln in (R["stdout"] if R else []) if ln.startswith(prefix)]


def off_lines_ok(R):
    return (lines(R, "REST_HOLD") == ["REST_HOLD: off"] and lines(R, "BETA_HOLD") == ["BETA_HOLD: off"]
            and lines(R, "VOTE_W") == ["VOTE_W: off"] and not lines(R, "COMP_HOLD") and not lines(R, "WINDOW_HOLD"))


def recs_of(R):
    return [json.loads(x) for x in R["probe_bytes"].decode().splitlines()] if R else []


def lion_all_groups(recs, skip=()):
    bad = n = 0
    for r in recs:
        for g in range(len(r["beta"])):
            if g in skip:
                continue
            nat, tie = SC.lion_natural(r["beta_pre"][0][g], r["mom_pre"][0][g], r["z_agg"][0][g])
            n += 1
            bad += (not tie) and abs(nat - r["beta"][g]) > SC.BH_TOL
    return bad, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=120)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    a.net = SC.NET
    tree = os.path.abspath(a.tree)
    os.makedirs(a.work, exist_ok=True)
    for k in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "COMP_HOLD", "REST_HOLD", "WINDOW_HOLD"):
        os.environ.pop(k, None)
    print("DRIVER_SHA256 %s" % hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
    print("SCORER_SHA256 %s" % hashlib.sha256(open(SC.__file__, "rb").read()).hexdigest())
    print("REGISTERED_TEST_SHA256 %s" % hashlib.sha256(open(RR.__file__, "rb").read()).hexdigest())
    hf = open(os.path.join(tree, "Optimizers", "HF.py"), "rb").read()
    bn = open(os.path.join(tree, "build_network.py"), "rb").read()
    print("TREE %s HF %s build_network %s" % (tree, hashlib.sha256(hf).hexdigest(), hashlib.sha256(bn).hexdigest()))
    chk(hashlib.sha256(hf).hexdigest() == SC.HF_POST_SHA, "--tree's HF.py is the scorer's HF_POST_SHA (cvt8's tree, unchanged)")
    chk(hashlib.sha256(bn).hexdigest() == SC.LIVE_BN_SHA, "--tree's build_network.py is the LIVE one")
    print("\nnet %s, steps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026"
          % (a.net, a.steps, a.every, a.seed))
    nrec = a.steps // a.every

    print("\nRB0 DETERMINISM CONTROL (ONE50 free, twice)")
    F50a = run_variant(a, "one50_a", tree, SC.SPEC["ONE50"], None)
    F50b = run_variant(a, "one50_b", tree, SC.SPEC["ONE50"], None)
    same(F50a, F50b, "RB0 ONE50 free vs itself")

    print("\nRB2 THE FREE SPECS")
    free = {"ONE50": F50a}
    for arm in ("ONE59", "ONE53"):
        free[arm] = run_variant(a, arm.lower(), tree, SC.SPEC[arm], None)
    free["SPLITFREE"] = run_variant(a, "splitfree", tree, SC.SPLITSPEC, None)
    for arm, R in free.items():
        if R is None:
            continue
        chk(lines(R, "GROUP_HOLD") == ["GROUP_HOLD: off"] and off_lines_ok(R),
            "RB2 %-9s printed `GROUP_HOLD: off`, `REST_HOLD: off`, `BETA_HOLD: off`, `VOTE_W: off`, no COMP/WINDOW line" % arm)
        recs = recs_of(R)
        if arm != "SPLITFREE":
            ok, d = SC.bite_check(arm, recs, every=a.every, n_records=nrec)
            chk(ok and d["ng"] == 2, "RB2 %-9s the scorer's bite_check PASSES as a control on the %d real records" % (arm, len(recs)),
                "lion %d ties %d keys %d" % (d["bad_lion"], d["ties"], d["keys_wrong"]))
        else:
            bad, n = lion_all_groups(recs)
            chk(len(recs) == nrec and all(len(r["beta"]) == 3 for r in recs) and bad == 0 and not any("gh_n" in r for r in recs),
                "RB2 SPLITFREE [59,1,2]: 3 groups at every record, every group == Lion recomputed, no gh_* key", "%d checks, %d bad" % (n, bad))

    print("\nRB1 THE REGISTERED HELD STRINGS BITE ON THE REAL PATH")
    held = {}
    for arm in SC.HELD:
        R = run_variant(a, arm.lower(), tree, SC.SPEC[arm], SC.GROUPHOLD[arm])
        if R is None:
            continue
        held[arm] = R
        chk(lines(R, "GROUP_HOLD") == [SC.WITNESS_GH[arm]] and off_lines_ok(R),
            "RB1 %-9s printed exactly ONE GROUP_HOLD line == the scorer's witness, and every off-line" % arm,
            (lines(R, "GROUP_HOLD") or ["(none)"])[0][:120])
        bad_step = sum(1 for n, b in enumerate(R["betas"], 1) if b[1] != SC.v_of(n))
        chk(len(R["betas"]) == a.steps and bad_step == 0 and all(len(b) == len(SC.SIZES[arm]) for b in R["betas"]),
            "RB1 %-9s beta[1] after EVERY one of %d steps == tri:9428; %d groups" % (arm, a.steps, len(SC.SIZES[arm])), "%d bad" % bad_step)
        recs = recs_of(R)
        ok, d = SC.bite_check(arm, recs, every=a.every, n_records=nrec)
        chk(ok and d["ng"] == len(SC.SIZES[arm]),
            "RB1 %-9s the scorer's OWN bite_check PASSES on the %d real records (unheld groups Lion, group 1 on schedule, gh audit, no foreign key)"
            % (arm, len(recs)), "lion %d ties %d sched %d/%d gh %d keys %d gh_active %s"
            % (d["bad_lion"], d["ties"], d["bad_sched"], d["bad_sched_pre"], d["bad_gh"], d["keys_wrong"], d["gh_active_last"]))
        twin = {"HOLDBIG3": None, "ONE50BIG": F50a, "ONE59BIG": free.get("ONE59"), "ONE53BIG": free.get("ONE53"),
                "ISOSPLIT": free.get("SPLITFREE")}[arm]
        if twin:
            nd = sum(1 for x, y in zip(R["betas"], twin["betas"]) if x != y)
            print("  NOTE RB1 %-9s beta differs from its free twin's on %d of %d steps; final weights %s (non-gating)"
                  % (arm, nd, a.steps, "differ" if R["weights_sha256"] != twin["weights_sha256"] else "identical"))

    print("\nRB3 NON-VACUITY: test-only floor strings, one on a singleton, one on GROUP 2 of the three-group partition")
    T53 = run_variant(a, "test_one53_floor", tree, SC.SPEC["ONE53"], T53_FLOOR)
    T2 = run_variant(a, "test_split_g2_floor", tree, SC.SPLITSPEC, TSPLIT_G2_FLOOR)
    for R, twin, g, lab, gs in ((T53, free.get("ONE53"), 1, "ONE53 floor", 1), (T2, free.get("SPLITFREE"), 2, "SPLIT group-2 floor", 2)):
        if R is None or twin is None:
            chk(False, "RB3 %s ran" % lab)
            continue
        gl = lines(R, "GROUP_HOLD")
        chk(len(gl) == 1 and gl[0].startswith("GROUP_HOLD: on type=blockwise group=%d groupsize=%d names=" % (g, gs))
            and gl[0].endswith("mode=floor value=-15.0") and off_lines_ok(R),
            "RB3 %-20s witness names group %d, groupsize %d, mode floor" % (lab, g, gs), gl[0][:140] if gl else "(none)")
        at_floor = all(b[g] == -15.0 for b in R["betas"])
        differ = all(x[g] != y[g] for x, y in zip(R["betas"], twin["betas"]))
        recs = recs_of(R)
        bad, n = lion_all_groups(recs, skip=(g,))
        last = recs[-1] if recs else {}
        chk(at_floor and differ and bad == 0 and last.get("gh_active", 0) > 0 and len(recs) == nrec,
            "RB3 %-20s the held group sits at -15 after every step, differs from the free run's at every step, every other group Lion, gh_active > 0"
            % lab, "lion checks %d bad %d gh_active %s" % (n, bad, last.get("gh_active")))
    if T53 is not None and free.get("ONE53") is not None:
        okf, df = SC.bite_check("ONE53BIG", recs_of(T53), every=a.every, n_records=nrec)
        chk(not okf and df["bad_sched"] > 0, "RB3 the ONE53 floor records read as ONE53BIG FAIL the tri:9428 schedule", "sched %d" % df["bad_sched"])
    if T2 is not None:
        okf, df = SC.bite_check("ISOSPLIT", recs_of(T2), every=a.every, n_records=nrec)
        chk(not okf and (df["bad_sched"] > 0 or df["bad_lion"] > 0),
            "RB3 the group-2 floor records read as ISOSPLIT FAIL (group 1 is not held; group 2 is not Lion)",
            "sched %d lion %d" % (df["bad_sched"], df["bad_lion"]))

    print("\nRB4 LOUDNESS on the live model (CPU, inside the job)")
    sys.path.insert(0, tree)
    cwd = os.getcwd()
    os.chdir(tree)
    try:
        import torch
        from build_network import build_network
        from Optimizers.HF import HF
        net = build_network(SC.NET, "cpu")
        nps = [(n, p.data.size()) for n, p in net.named_parameters()]
        chk([n for n, _ in nps] == SC.NAMES, "RB4 the live names == the scorer's NAMES")

        def construct(spec, gh):
            os.environ["GROUP_HOLD"] = gh
            o = HF.__new__(HF)
            o.num_layers = len(nps)
            o._device = torch.device("cpu")
            o.args_meta = {"alg": "Lion", "meta_stepsize": 1e-3, "momentum_param": 0.99, "Lion_beta2": 0.9, "weight_decay": 0}
            o._beta_lo, o._beta_hi = -15.0, -2.3026
            o._hier = "none"
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    HF.init_meta(o, spec, nps, 1e-6)
                return None, buf.getvalue()
            except ValueError as ex:
                return ex, buf.getvalue()
            finally:
                os.environ.pop("GROUP_HOLD", None)
        for spec, gh, lab in ((SC.ISOSPEC, "layer4.0.bn2.weight:tri:9428", "one carrier on ISO's [59,3]"),
                              (SC.SPLITSPEC, "layer4.0.bn2.weight+layer4.1.bn2.weight:tri:9428", "a pair that is not a group"),
                              (SC.ONE50SPEC, SC.GROUPHOLD["HOLDBIG3"], "the three carriers on ONE50's [61,1]")):
            ex, _out = construct(spec, gh)
            chk(isinstance(ex, ValueError), "RB4 %-40s raises ValueError at construction" % lab, str(ex)[:100])
        for arm in SC.HELD:
            ex, out = construct(SC.SPEC[arm], SC.GROUPHOLD[arm])
            gl = [ln for ln in out.splitlines() if ln.startswith("GROUP_HOLD")]
            chk(ex is None and gl == [SC.WITNESS_GH[arm]], "RB4 %-9s constructs on CPU with exactly its registered witness" % arm)
    finally:
        os.chdir(cwd)

    print("\n%s" % ("ALL PASS" if not RR.FAILED else "FAILURES: %r" % RR.FAILED))
    raise SystemExit(1 if RR.FAILED else 0)


if __name__ == "__main__":
    main()
