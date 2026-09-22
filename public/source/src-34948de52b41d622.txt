"""INERTNESS AND BITE OF PATCH_VALSPLIT OVER SHORT *REAL* RUNS (the real train.py, on a GPU).  CORRECTIONS 302.

Each variant is ONE child process that executes the tree's OWN, UNCHANGED train.py as __main__ (runpy.run_path) with the
cvl1 ARGS line (--num-epochs E) and the cvl1 ENV (AUGMENT=1, BETA_CLIP=-15:-2.3026, HIER=none, SCHED=none, PROBE=5,
PROBE_DIR=<work>), cwd = the tree -- what the runner does, minus Slurm -- with ONE addition set identically in every child
BEFORE train.py runs: torch.backends.cudnn.deterministic=True, cudnn.benchmark=False,
torch.use_deterministic_algorithms(True, warn_only=True), CUBLAS_WORKSPACE_CONFIG=:4096:8 (as tests/test_voteweight_realrun.py
sets cudnn).  WHY: the first proof job (5080195) ran plain `python train.py` and its RR0 FAILED -- the unpatched tree
against ITSELF gave different Epoch lines and probe bytes -- so without deterministic kernels no bitwise comparison can be
read.  Production runs do not set these flags; any kernel nondeterminism there acts on patched and unpatched trees alike.
Two trees:
  --tree-pre   the cgw1 tree ($WS/harness_cgw1/cifar10: load_data b52b58a3..., train 3fea309e...), READ-ONLY
  --tree-post  the cvl1 tree (PATCH_VALSPLIT on the same bytes)

  RR0  DETERMINISM CONTROL: the pre tree, chunk777, twice -> identical Epoch lines and probe.jsonl bytes (without it, RR1
       could not be read as bitwise).
  RR1  OFF: the post tree with VAL_SPLIT UNSET and EMPTY -> Epoch lines and probe.jsonl BYTE-IDENTICAL to the pre tree's;
       exactly ONE `VAL_SPLIT: off` line and NO `VAL:` line; the stdout differs from the pre tree's by that ONE line only.
  RR2  BITE, the registered string VAL_SPLIT=5000:302, chunk777, run seeds 184 AND 185:
       ONE witness `VAL_SPLIT: on dataset=CIFAR10 n_val=5000 n_train=45000 classes=10 per_class=500 split_seed=302 ...`;
       the val_sha / train_sha are IDENTICAL across the two run seeds (the split does not move with the seed) and equal
       --expect-val-sha when given; E `VAL:` lines, epochs 0..E-1, each right after its Epoch line, every val_acc a
       multiple of 0.02 in (0, 100], n_val 5000; the model TRAINED ON 45,000: probe records at steps 0,5,..., last step
       == 450*E - 5 (vs 500*E - 5 OFF), i.e. 450 batches of 100 per epoch; the Epoch lines DIFFER from OFF's.
  RR3  NON-VACUITY: VAL_SPLIT=5000:303 (scalar, seed 184) -> a DIFFERENT val_sha; a malformed value (5000) -> the run
       FAILS loudly (non-zero exit, ValueError PATCH_VALSPLIT) before any Epoch line.

  python3 tests/test_valsplit_realrun.py --tree-pre $WS/harness_cgw1/cifar10 --tree-post $WS/harness_cvl1/cifar10 \\
      --epochs 2 --work /tmp/cvl1_realrun
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAILED = []
STEPS_PER_EPOCH_ON = 45000 // 100
STEPS_PER_EPOCH_OFF = 50000 // 100
WIT_RE = re.compile(r"^VAL_SPLIT: on dataset=CIFAR10 n_val=5000 n_train=45000 classes=10 per_class=500 split_seed=(\d+) "
                    r"val_sha=([0-9a-f]{64}) train_sha=([0-9a-f]{64})$")
VAL_RE = re.compile(r"^VAL: epoch (\d+) val_acc ([0-9.]+) % n_val (\d+)$")
EP_RE = re.compile(r"^Epoch (\d+), Train Accuracy: ")
KEYS = ("VAL_SPLIT", "VOTE_W", "BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD", "GROUP_HOLD", "REST_HOLD", "DECAY_MASK",
        "PROBE_TENSOR", "SHADOW_VOTE")


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def args_for(grain, seed, epochs, save):
    return ["--optimizer", "HF", "--alg-base", "SGDm", "--momentum-param-base", "0.99", "--weight-decay-base", "0.1",
            "--alg-meta", "Lion", "--momentum-param-meta", "0.99", "--Lion-beta2-meta", "0.9", "--weight-decay-meta", "0",
            "--dataset", "CIFAR10", "--NN-name", "ResNet18", "--batch-size", "100", "--max-time", "999:00:00",
            "--gamma", "1", "--meta-stepsize", "1e-4", "--alpha0", "1e-3", "--num-epochs", str(epochs),
            "--stepsize-groups", grain, "--seed", str(seed), "--save-directory", save, "--run-name", "rr"]


def run(a, tag, tree, grain, seed, vs):
    work = os.path.join(a.work, tag)
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ)
    for k in KEYS:
        env.pop(k, None)
    env.update(AUGMENT="1", BETA_CLIP="-15:-2.3026", HIER="none", SCHED="none", PROBE="5",
               PROBE_DIR=os.path.join(work, "probe"), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    if vs is not None:
        env["VAL_SPLIT"] = vs
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--child", tree] + args_for(grain, seed, a.epochs, work),
                       cwd=tree, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    pp = os.path.join(work, "probe", "probe.jsonl")
    probe = open(pp, "rb").read() if os.path.exists(pp) else b""
    out = p.stdout.splitlines()
    print("  run %-14s tree=%s grain=%s seed=%d VAL_SPLIT=%s rc=%d epochs=%d probe=%s" % (
        tag, os.path.basename(os.path.dirname(tree)), grain, seed, "UNSET" if vs is None else repr(vs), p.returncode,
        len([l for l in out if EP_RE.match(l)]), hashlib.sha256(probe).hexdigest()[:16]))
    return dict(rc=p.returncode, out=out, probe=probe)


def eplines(R):
    return [l for l in R["out"] if EP_RE.match(l)]


def last_step(R):
    import json
    L = R["probe"].decode().splitlines()
    return (json.loads(L[-1])["step"], len(L), [json.loads(x)["step"] for x in L]) if L else (None, 0, [])


def child(tree, argv):
    """Execute the tree's OWN train.py, unchanged, as __main__, with deterministic kernels."""
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
    ap.add_argument("--expect-val-sha", default=None)
    a = ap.parse_args()
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    E = a.epochs
    print("DRIVER_SHA256 %s" % hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
    for t, lab in ((pre, "PRE"), (post, "POST")):
        print("TREE_%s %s load_data %s train %s HF %s" % (lab, t, *[hashlib.sha256(open(os.path.join(t, f), "rb").read()).hexdigest()
              for f in ("load_data.py", "train.py", "Optimizers/HF.py")]))
    chk(b"PATCH_VALSPLIT" not in open(os.path.join(pre, "load_data.py"), "rb").read() + open(os.path.join(pre, "train.py"), "rb").read(),
        "--tree-pre is UNPATCHED")
    chk(open(os.path.join(pre, "load_data.py"), "rb").read() == open(os.path.join(post, "load_data.py.pre_valsplit"), "rb").read()
        and open(os.path.join(pre, "train.py"), "rb").read() == open(os.path.join(post, "train.py.pre_valsplit"), "rb").read(),
        "--tree-pre's load_data.py / train.py are BYTE-IDENTICAL to --tree-post's *.pre_valsplit backups")
    for f in ("Optimizers/HF.py", "build_network.py", "Optimizers/build_optimizer.py"):
        chk(open(os.path.join(pre, f), "rb").read() == open(os.path.join(post, f), "rb").read(), "the two trees' %s are BYTE-IDENTICAL" % f)

    print("\nRR0 DETERMINISM CONTROL (pre tree, chunk777, seed 184, twice)")
    A1 = run(a, "pre_a", pre, "chunk777", 184, None)
    A2 = run(a, "pre_b", pre, "chunk777", 184, None)
    chk(A1["rc"] == 0 and len(eplines(A1)) == E, "RR0 the pre run completes %d epochs" % E)
    chk(eplines(A1) == eplines(A2) and A1["probe"] == A2["probe"] and len(A1["probe"]) > 0,
        "RR0 pre vs itself: Epoch lines and probe.jsonl bytes IDENTICAL (the GPU path is deterministic here)")

    print("\nRR1 OFF (post tree, VAL_SPLIT unset / empty) == pre tree, bitwise")
    for vs, lab in ((None, "unset"), ("", "empty")):
        R = run(a, "post_off_" + lab, post, "chunk777", 184, vs)
        chk(R["rc"] == 0 and eplines(R) == eplines(A1), "RR1 %s: Epoch lines (train + test accuracy) IDENTICAL to the pre tree's" % lab)
        chk(R["probe"] == A1["probe"], "RR1 %s: probe.jsonl BYTE-IDENTICAL (beta at every 5th step, full precision)" % lab)
        vsl = [l for l in R["out"] if l.startswith("VAL_SPLIT")]
        chk(vsl == ["VAL_SPLIT: off"] and not [l for l in R["out"] if l.startswith("VAL:")],
            "RR1 %s: ONE `VAL_SPLIT: off` line, NO `VAL:` line" % lab)
        extra = [l for l in R["out"] if l not in A1["out"]]
        miss = [l for l in A1["out"] if l not in R["out"]]
        extra_ok = all(l == "VAL_SPLIT: off" or l.strip().endswith("minutes") for l in extra)
        chk(extra_ok and all(l.strip().endswith("minutes") for l in miss),
            "RR1 %s: stdout differs from the pre tree's ONLY by the witness line (and the wall-clock `N minutes` line)" % lab,
            "extra %r missing %r" % (extra[:3], miss[:3]))
        step, n, _st = last_step(R)
        chk(step == STEPS_PER_EPOCH_OFF * E - 5, "RR1 %s: last probe step %s == 500*E - 5 (trained on all 50,000)" % (lab, step))

    print("\nRR2 BITE: VAL_SPLIT=5000:302 (the registered string)")
    ON = {}
    for seed in (184, 185):
        R = run(a, "post_on_s%d" % seed, post, "chunk777", seed, "5000:302")
        ON[seed] = R
        chk(R["rc"] == 0 and len(eplines(R)) == E, "RR2 s%d completes %d epochs" % (seed, E))
        w = [l for l in R["out"] if l.startswith("VAL_SPLIT")]
        m = WIT_RE.match(w[0]) if len(w) == 1 else None
        chk(m is not None and m.group(1) == "302",
            "RR2 s%d: ONE witness, n_val 5000 / n_train 45000 / 10 classes / 500 per class / split_seed 302" % seed, (w or ["(none)"])[0][:100])
        R["wit"] = m.groups() if m else None
        vl = [(i, l) for i, l in enumerate(R["out"]) if l.startswith("VAL:")]
        ok = len(vl) == E
        for k, (i, l) in enumerate(vl):
            mv = VAL_RE.match(l)
            ok = ok and mv is not None and int(mv.group(1)) == k and int(mv.group(3)) == 5000
            ok = ok and mv is not None and 0 < float(mv.group(2)) <= 100 and abs(float(mv.group(2)) * 50 - round(float(mv.group(2)) * 50)) < 1e-6
            ok = ok and i > 0 and EP_RE.match(R["out"][i - 1]) is not None and int(EP_RE.match(R["out"][i - 1]).group(1)) == k
        chk(ok, "RR2 s%d: %d VAL lines, epochs 0..%d, each right after its Epoch line, n_val 5000, val_acc a multiple of 0.02"
            % (seed, E, E - 1), " | ".join(l for _i, l in vl))
        step, n, st = last_step(R)
        chk(step == STEPS_PER_EPOCH_ON * E - 5 and st == list(range(0, STEPS_PER_EPOCH_ON * E, 5)),
            "RR2 s%d: probe records at 0,5,..,%d -- 450 batches of 100 per epoch: TRAINED ON 45,000" % (seed, step or -1))
    chk(eplines(ON[184]) != eplines(A1), "RR2 the ON run's Epoch lines DIFFER from OFF's (the switch bites)")
    w1, w2 = ON[184].get("wit"), ON[185].get("wit")
    chk(w1 is not None and w1 == w2, "RR2 the split (val_sha AND train_sha) is IDENTICAL for run seeds 184 and 185",
        "val %s train %s" % (w1[1][:16], w1[2][:16]) if w1 else "")
    if a.expect_val_sha:
        chk(w1 is not None and w1[1] == a.expect_val_sha, "RR2 val_sha == --expect-val-sha (V3's CPU value)")

    print("\nRR3 NON-VACUITY")
    R3 = run(a, "post_on_303", post, "scalar", 184, "5000:303")
    w = [l for l in R3["out"] if l.startswith("VAL_SPLIT")]
    m = WIT_RE.match(w[0]) if len(w) == 1 else None
    chk(R3["rc"] == 0 and m is not None and m.group(1) == "303" and w1 is not None and m.group(2) != w1[1],
        "RR3 split seed 303 -> a DIFFERENT val_sha")
    R4 = run(a, "post_bad", post, "scalar", 184, "5000")
    chk(R4["rc"] != 0 and any("PATCH_VALSPLIT" in l and "ValueError" in l for l in R4["out"]) and not eplines(R4),
        "RR3 VAL_SPLIT=5000 (malformed) FAILS loudly before any Epoch line", (R4["out"] or [""])[-1][:100])

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES (%d): %r" % (len(FAILED), FAILED)))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
