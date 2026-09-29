"""INERTNESS AND BITE OF PATCH_DECAYROUTE OVER SHORT *REAL* RUNS (the real train.py, on a GPU).  CORRECTIONS 305.

Each variant is ONE child process that executes the tree's OWN, UNCHANGED train.py as __main__ (runpy.run_path) with the
audit core cell's ARGS line (ResNet18 / CIFAR-10 / SGDm 0.99 + Lion 0.99/0.9, ms 1e-4, alpha0 1e-3, gamma 1, bs 100;
--num-epochs E) and the campaign ENV (AUGMENT=1, BETA_CLIP=-15:-2.3026, HIER=none, SCHED=none, PROBE=5, PROBE_DIR=<work>),
cwd = the tree, with deterministic kernels set identically in every child BEFORE train.py runs (cudnn.deterministic,
no benchmark, use_deterministic_algorithms(warn_only), CUBLAS_WORKSPACE_CONFIG=:4096:8) -- the driver form 302.4
established after its first proof failed its determinism control.  Proof runs write only under --work (deleted by the
sbatch); they are NOT campaign runs and never reach runs/ or any CSV.
Two trees:
  --tree-pre   the cgw1 tree ($WS/harness_cgw1/cifar10, HF 4732b74a...), READ-ONLY
  --tree-post  the cdr1 tree (PATCH_DECAYROUTE on the same bytes)

  RR0  DETERMINISM CONTROL: the pre tree, chunk777, wd 0.1, twice -> identical Epoch lines and probe.jsonl bytes.
  RR1  OFF: the post tree with DECAY_ROUTE UNSET and EMPTY, chunk777 -> Epoch lines and probe.jsonl BYTE-IDENTICAL to the
       pre tree's; ONE `DECAY_ROUTE: off` line; stdout differs from the pre tree's by that line only.
  RR2  BITE, scalar grain, against the pre tree at the same ARGS:
       shrink_only (wd 0.1)       ONE witness line == the registered string; probe records carry dr_* with dr_n == step+2;
                                  the measured shrink == 0.1*a (a = exp(beta), rel 1e-2); h_absmax and the probe bytes
                                  DIFFER from the pre tree's (the trace factor bit); Epoch lines may or may not differ
                                  (reported, not gated: the weights move differently only through beta)
       trace_only (wd 0.1)        witness; measured shrink == 0 within 1e-6; Epoch lines DIFFER (the weights are undecayed)
       alpha_indep:5e-05 (wd 0)   witness (lambda=5e-05); measured shrink == 5e-05 within rel 1e-3 at EVERY record,
                                  dr_lam == 5e-05; Epoch lines DIFFER from the pre tree at wd 0
  RR3  LOUD: `alpha_indep` (no LAMBDA), `alpha_indep:5e-05` at wd 0.1, `shrink_only` at wd 0 -> each run exits non-zero
       with `ValueError: PATCH_DECAYROUTE` before any Epoch line.

  python3 tests/test_decayroute_realrun.py --tree-pre $WS/harness_cgw1/cifar10 --tree-post $WS/harness_cdr1/cifar10 \\
      --epochs 2 --work /tmp/cdr1_realrun
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []
NPASS = [0]
EP_RE = re.compile(r"^Epoch (\d+), Train Accuracy: ")
KEYS = ("DECAY_ROUTE", "VAL_SPLIT", "VOTE_W", "BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD", "GROUP_HOLD", "REST_HOLD",
        "DECAY_MASK", "PROBE_TENSOR", "SHADOW_VOTE")
SEED = 1


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    if cond:
        NPASS[0] += 1
    else:
        FAILED.append(label)
    return bool(cond)


def args_for(grain, wd, epochs, save):
    return ["--optimizer", "HF", "--alg-base", "SGDm", "--momentum-param-base", "0.99", "--weight-decay-base", wd,
            "--alg-meta", "Lion", "--momentum-param-meta", "0.99", "--Lion-beta2-meta", "0.9", "--weight-decay-meta", "0",
            "--dataset", "CIFAR10", "--NN-name", "ResNet18", "--batch-size", "100", "--max-time", "999:00:00",
            "--gamma", "1", "--meta-stepsize", "1e-4", "--alpha0", "1e-3", "--num-epochs", str(epochs),
            "--stepsize-groups", grain, "--seed", str(SEED), "--save-directory", save, "--run-name", "rr"]


def run(a, tag, tree, grain, wd, dr):
    work = os.path.join(a.work, tag)
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ)
    for k in KEYS:
        env.pop(k, None)
    env.update(AUGMENT="1", BETA_CLIP="-15:-2.3026", HIER="none", SCHED="none", PROBE="5",
               PROBE_DIR=os.path.join(work, "probe"), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if dr is not None:
        env["DECAY_ROUTE"] = dr
    p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--child", tree] + args_for(grain, wd, a.epochs, work),
                       cwd=tree, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    pp = os.path.join(work, "probe", "probe.jsonl")
    probe = open(pp, "rb").read() if os.path.exists(pp) else b""
    out = p.stdout.splitlines()
    print("  run %-16s tree=%s grain=%s wd=%s DECAY_ROUTE=%s rc=%d epochs=%d probe=%s" % (
        tag, os.path.basename(os.path.dirname(tree)), grain, wd, "UNSET" if dr is None else repr(dr), p.returncode,
        len([l for l in out if EP_RE.match(l)]), hashlib.sha256(probe).hexdigest()[:16]))
    for l in out:
        if EP_RE.match(l) or l.startswith("DECAY_ROUTE") or "Error" in l:
            print("      | " + l[:160])
    return dict(rc=p.returncode, out=out, probe=probe, recs=[json.loads(x) for x in probe.decode().splitlines()] if probe else [])


def eplines(R):
    return [l for l in R["out"] if EP_RE.match(l)]


def child(tree, argv):
    import runpy
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.chdir(tree)
    sys.path.insert(0, tree)
    sys.argv = [os.path.join(tree, "train.py")] + argv
    runpy.run_path(os.path.join(tree, "train.py"), run_name="__main__")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(os.path.abspath(sys.argv[2]), sys.argv[3:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-pre", required=True)
    ap.add_argument("--tree-post", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    E = a.epochs
    print("DRIVER_SHA256 %s" % hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
    rb = lambda t, f: open(os.path.join(t, f), "rb").read()  # noqa
    for t, lab in ((pre, "PRE"), (post, "POST")):
        print("TREE_%s %s HF %s train %s" % (lab, t, hashlib.sha256(rb(t, "Optimizers/HF.py")).hexdigest(),
                                             hashlib.sha256(rb(t, "train.py")).hexdigest()))
    chk(b"PATCH_DECAYROUTE" not in rb(pre, "Optimizers/HF.py"), "--tree-pre is UNPATCHED")
    chk(b"PATCH_DECAYROUTE" in rb(post, "Optimizers/HF.py"), "--tree-post carries PATCH_DECAYROUTE")
    chk(rb(pre, "Optimizers/HF.py") == rb(post, "Optimizers/HF.py.pre_decayroute"),
        "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_decayroute")
    for f in ("train.py", "build_network.py", "load_data.py", "Optimizers/build_optimizer.py", "tin_data.py"):
        chk(rb(pre, f) == rb(post, f), "the two trees' %s are BYTE-IDENTICAL" % f)

    print("\nRR0 DETERMINISM CONTROL (pre tree, chunk777, wd 0.1, twice)")
    A1 = run(a, "pre_a", pre, "chunk777", "0.1", None)
    A2 = run(a, "pre_b", pre, "chunk777", "0.1", None)
    chk(A1["rc"] == 0 and len(eplines(A1)) == E, "RR0 the pre run completes %d epochs" % E)
    chk(eplines(A1) == eplines(A2) and A1["probe"] == A2["probe"] and len(A1["probe"]) > 0,
        "RR0 pre vs itself: Epoch lines and probe.jsonl bytes IDENTICAL")

    print("\nRR1 OFF (post tree, DECAY_ROUTE unset / empty) == pre tree, bitwise")
    for dr, lab in ((None, "unset"), ("", "empty")):
        R = run(a, "post_off_" + lab, post, "chunk777", "0.1", dr)
        chk(R["rc"] == 0 and eplines(R) == eplines(A1), "RR1 %s: Epoch lines (train + test accuracy) IDENTICAL to the pre tree's" % lab)
        chk(R["probe"] == A1["probe"] and len(R["probe"]) > 0, "RR1 %s: probe.jsonl BYTE-IDENTICAL (beta at every 5th step, full precision)" % lab)
        chk([l for l in R["out"] if l.startswith("DECAY_ROUTE")] == ["DECAY_ROUTE: off"], "RR1 %s: exactly ONE `DECAY_ROUTE: off` line" % lab)
        extra = [l for l in R["out"] if l not in A1["out"]]
        miss = [l for l in A1["out"] if l not in R["out"]]
        chk(all(l == "DECAY_ROUTE: off" or l.strip().endswith("minutes") for l in extra) and all(l.strip().endswith("minutes") for l in miss),
            "RR1 %s: stdout differs from the pre tree's ONLY by the witness line (and the wall-clock `N minutes` line)" % lab,
            "extra %r missing %r" % (extra[:3], miss[:3]))
        chk(not any("dr_mode" in r for r in R["recs"]), "RR1 %s: no probe record carries a dr_* key" % lab)

    print("\nRR2 BITE (scalar grain; each against the pre tree at the same ARGS)")
    B01 = run(a, "pre_scalar_wd0.1", pre, "scalar", "0.1", None)
    B00 = run(a, "pre_scalar_wd0", pre, "scalar", "0", None)
    chk(B01["rc"] == 0 and B00["rc"] == 0 and len(eplines(B01)) == E and len(eplines(B00)) == E, "RR2 the two pre-tree references complete")
    for mode, wd, ref in (("shrink_only", "0.1", B01), ("trace_only", "0.1", B01), ("alpha_indep:5e-05", "0", B00)):
        R = run(a, "post_" + mode.split(":")[0], post, "scalar", wd, mode)
        lam = 5e-05 if mode.startswith("alpha") else None
        want = "DECAY_ROUTE: on mode=%s base=SGDm wd=%r lambda=%s lambda_f32=%s gamma=1.0" % (
            mode.split(":")[0], float(wd), "na" if lam is None else "5e-05", "na" if lam is None else "4.999999873689376e-05")
        chk(R["rc"] == 0 and len(eplines(R)) == E, "RR2 %s completes %d epochs" % (mode, E))
        wl = [l for l in R["out"] if l.startswith("DECAY_ROUTE")]
        chk(wl == [want], "RR2 %s: ONE witness line == the registered string" % mode, (wl or ["(none)"])[0])
        recs = R["recs"]
        ok = len(recs) == 100 * E and all(r.get("dr_mode") == mode.split(":")[0] and r.get("dr_n") == r["step"] + 2
                                          and r.get("dr_lam") == lam for r in recs)
        chk(ok, "RR2 %s: %d probe records, each with dr_mode / dr_lam, dr_n == step + 2" % (mode, len(recs)))
        ss = [r.get("dr_shrink_meas") for r in recs]
        if mode == "shrink_only":
            dev = max(abs(r["dr_shrink_meas"] / (0.1 * math.exp(r["beta"][0])) - 1) for r in recs) if all(s is not None for s in ss) else 9
            chk(dev < 1e-2, "RR2 shrink_only: measured per-step shrink == 0.1*exp(beta) at every record (max rel dev %.2e)" % dev)
            hd = [r["h_absmax"] for r in recs] != [r["h_absmax"] for r in ref["recs"]]
            chk(hd and R["probe"] != ref["probe"], "RR2 shrink_only: h_absmax and the probe bytes DIFFER from the pre tree's (the trace factor bites)")
            bd = sum(1 for r, q in zip(recs, ref["recs"]) if r["beta"] != q["beta"])
            print("  INFO RR2 shrink_only: beta differs from the pre tree at %d / %d records; Epoch lines %s"
                  % (bd, len(recs), "DIFFER" if eplines(R) != eplines(ref) else "IDENTICAL"))
        elif mode == "trace_only":
            dev = max(abs(s) for s in ss) if all(s is not None for s in ss) else 9
            chk(dev < 1e-6, "RR2 trace_only: measured per-step shrink == 0 at every record (max |s| %.2e)" % dev)
            chk(eplines(R) != eplines(ref), "RR2 trace_only: Epoch lines DIFFER from the pre tree at wd 0.1 (the weights are undecayed)")
        else:
            dev = max(abs(s / 5e-05 - 1) for s in ss) if all(s is not None for s in ss) else 9
            chk(dev < 1e-3, "RR2 alpha_indep:5e-05: measured per-step shrink == 5e-05 at every record (max rel dev %.2e)" % dev)
            chk(all(abs(r["dr_shrink_applied"] / 5e-05 - 1) < 1e-9 and abs(r["dr_trace_decay"] - (1 - 5e-05)) < 1e-12 for r in recs),
                "RR2 alpha_indep:5e-05: dr_shrink_applied == LAMBDA, dr_trace_decay == 1 - LAMBDA")
            chk(eplines(R) != eplines(ref), "RR2 alpha_indep:5e-05: Epoch lines DIFFER from the pre tree at wd 0 (the decay bites)")
        print("  INFO RR2 %s: shrink at step 0 %s, at the last record %s" % (mode, ss[0] if ss else None, ss[-1] if ss else None))

    print("\nRR3 LOUD")
    for dr, wd in (("alpha_indep", "0"), ("alpha_indep:5e-05", "0.1"), ("shrink_only", "0")):
        R = run(a, "post_bad_%s_%s" % (dr.replace(":", "-"), wd), post, "scalar", wd, dr)
        chk(R["rc"] != 0 and any("ValueError" in l and "PATCH_DECAYROUTE" in l for l in R["out"]) and not eplines(R),
            "RR3 DECAY_ROUTE=%s at wd %s FAILS loudly before any Epoch line" % (dr, wd), (R["out"] or [""])[-1][:110])

    print("\n%d PASS / %d FAIL" % (NPASS[0], len(FAILED)))
    print("ALL PASS" if not FAILED else "FAILURES (%d): %r" % (len(FAILED), FAILED))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
