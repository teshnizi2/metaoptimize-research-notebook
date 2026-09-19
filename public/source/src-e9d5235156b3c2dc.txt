"""INERTNESS AND BITE OF PATCH_DECAYMASK OVER A SHORT *REAL* RUN, FOR cwd1's / cwd2's EXACT STRINGS.  CORRECTIONS 260 / 261.

The REAL code path of a cwd run (train.py's own parse_args, build_network, build_optimizer, load_data CIFAR-100 AUGMENT=1,
the loss, HF.step with BETA_CLIP, PROBE and PROBE_TENSOR) for --steps optimizer steps on a GPU, from TWO trees:
  --tree-pre   the parent tree, READ-ONLY: cvt8's ($WS/harness_cvt8/cifar10) for cwd1, cvt9's for cwd2
  --tree-post  the cwd tree (PATCH_DECAYMASK on top of the parent's HF.py)
For the OFF comparisons it IMPORTS a registered real-run test UNEDITED and uses its child process, run_variant, same and
chk: cwd1 (ResNet18_c100) cvt7's tests/test_grouphold_realrun.py (its child takes the network); cwd2 (PlainNet18_c100)
cvt1's tests/test_voteweight_realrun.py (BETA_HOLD / COMP_HOLD ride the environment run_variant copies, as cvt9 did).
Strings and witnesses come from analysis/cwd_design.py.

  RR0  DETERMINISM CONTROL: the parent tree run twice (k01) -> identical beta, loss, probe bytes, final weights.
  RR1  OFF: DECAY_MASK unset and empty, on every distinct unmasked configuration of the batch (cwd1: scalar, layerwise;
       cwd2: k01, HIGHHEADPATH, LOWHEADPATH with cvt9's hold strings) == the parent tree BITWISE (beta every step, loss
       every step, probe.jsonl bytes, every final parameter and buffer); `DECAY_MASK: off` and the parent's own witness
       lines printed.
  RR2  THE EXACT MASKED STRINGS ON THE REAL PATH (run_variant): exactly the design's DECAY_MASK witness; at every probe
       record dm_n == step + 2, dm_skipped == dm_n * k, dm_masked == k, dm_wdterm > 0, one finite norm per masked tensor;
       no dm_* key on an unmasked run.  cwd2 held arms: beta[1] and beta[0] after EVERY step == cvt9's schedules, and the
       masked arm's beta at every step is BITWISE its unmasked twin's from the parent tree (the mask touches no step size).
  RR3  UPDATE IDENTITY ON THE REAL PATH (this file's own child, the registered child's code plus a verifier wrapped around
       optimizer.base_update): at EVERY step, for EVERY tensor, w', h' and m' are BITWISE the registered formula with
       wd_i = 0 on the design's masked indices and wd 0.1 elsewhere.  Controls: the parent tree (OFF) verified with an
       EMPTY mask -> 0 mismatches; a test-only alpha0 1e-2 masked run -> the WD term is visible at float resolution on
       masked tensors (bites > 0), the SAME records checked against the EMPTY-mask formula FAIL (the verifier can fail),
       and its final weights differ from the OFF run's.

  python3 tests/test_decaymask_realrun.py --batch cwd1 --tree-pre $WS/harness_cvt8/cifar10 --tree-post $WS/harness_cwd1/cifar10 \\
      --steps 300 --every 10 --seed 128 --work /tmp/cwd1_realrun
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAILED_LOCAL = []


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def child_verify(a):
    """The registered child's code (tests/test_grouphold_realrun.py child) with --net, --alpha0 and a verifier."""
    tree = os.path.abspath(a.tree)
    os.chdir(tree)
    sys.path.insert(0, tree)
    import numpy as np
    import torch
    import torch.nn as nn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from torch.utils.tensorboard import SummaryWriter
    from build_network import build_network
    from Optimizers.build_optimizer import build_optimizer
    from load_data import load_data
    import ast as _ast, argparse as _argparse, types as _types
    _tsrc = open(os.path.join(tree, "train.py")).read()
    _fn = [n for n in _ast.parse(_tsrc).body if isinstance(n, _ast.FunctionDef) and n.name == "parse_args"]
    assert len(_fn) == 1, "train.py has no single parse_args"
    TR = _types.ModuleType("train_parse_args")
    TR.argparse = _argparse
    exec(compile(_ast.Module(body=_fn, type_ignores=[]), os.path.join(tree, "train.py"), "exec"), TR.__dict__)
    sys.argv = ["train.py", "--optimizer", "HF", "--alg-base", "SGDm", "--momentum-param-base", "0.99",
                "--weight-decay-base", "0.1", "--alg-meta", "Lion", "--momentum-param-meta", "0.99",
                "--Lion-beta2-meta", "0.9", "--weight-decay-meta", "0", "--dataset", "CIFAR100",
                "--NN-name", a.net, "--batch-size", "100", "--max-time", "999:00:00",
                "--gamma", "1", "--meta-stepsize", "1e-3", "--alpha0", a.alpha0, "--num-epochs", "100",
                "--stepsize-groups", a.spec, "--seed", str(a.seed), "--save-directory", a.save,
                "--run-name", a.tag]
    args = TR.parse_args()
    for k in ("normalizer_param_base", "momentum_param_base", "weight_decay_base", "Lion_beta2_base",
              "normalizer_param_meta", "momentum_param_meta", "weight_decay_meta", "Lion_beta2_meta",
              "meta_stepsize"):
        if getattr(args, k) == -1:
            setattr(args, k, None)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    writer = SummaryWriter(os.path.join(args.save_directory, "Tensorboard_outputs", args.run_name))
    criterion = nn.CrossEntropyLoss().to(args.device)
    net = build_network(args.NN_name, args.device)
    optimizer = build_optimizer(net, args, writer)
    trainloader, _testloader = load_data(args.dataset, args.batch_size, args.seed)
    params = list(net.parameters())
    masks = {}
    for lab, s in (("expect", a.expect_mask), ("wrong", a.wrong_mask)):
        masks[lab] = set(int(x) - 1 for x in s.split(",")) if s else set()
    stats = {"expect": [0, 0, 0], "wrong": [0, 0, 0]}   # mismatches in w, h, m
    bites = [0]
    orig = optimizer.base_update

    def wrapped(netx, g):
        wb = [p.data.clone() for p in params]
        mb = [m.clone() if torch.is_tensor(m) else m for m in optimizer.momentum_base]   # 0.0 floats before the first update
        hb = [h.clone() for h in optimizer.h_condenced]
        orig(netx, g)
        mp = optimizer.args_base["momentum_param"]
        wd = optimizer.args_base["weight_decay"]
        for i, p in enumerate(params):
            al = optimizer.alpha[i]
            m_exp = mp * mb[i] + (1 - mp) * g[i]
            nod = al * mb[i]
            wdd = al * (mb[i] + wd * wb[i])
            if i in masks["expect"] and not torch.equal(wb[i] - nod, wb[i] - wdd):
                bites[0] += 1
            for lab in ("expect", "wrong"):
                if i in masks[lab]:
                    w_e, h_e = wb[i] - nod, optimizer.gamma * hb[i] - nod
                else:
                    w_e, h_e = wb[i] - wdd, optimizer.gamma * (1 - wd * al) * hb[i] - wdd
                stats[lab][0] += not torch.equal(p.data, w_e)
                stats[lab][1] += not torch.equal(optimizer.h_condenced[i], h_e)
                stats[lab][2] += not torch.equal(optimizer.momentum_base[i], m_exp)

    optimizer.base_update = wrapped
    betas, t = [], 0
    while t < a.steps:
        for data in trainloader:
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            loss = criterion(net(inputs), labels)
            optimizer.step(net, loss)
            betas.append([float(x) for b in optimizer.beta for x in b.detach().reshape(-1).tolist()])
            t += 1
            if t >= a.steps:
                break
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    writer.close()
    json.dump({"stats": stats, "bites": bites[0], "steps": t, "betas": betas, "weights_sha256": h.hexdigest(),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}, open(a.out, "w"))


def run_verify(a, tag, tree, spec, env, alpha0, expect_mask, wrong_mask):
    work = os.path.join(a.work, tag)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    e = dict(os.environ)
    for k in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "COMP_HOLD", "REST_HOLD", "WINDOW_HOLD", "DECAY_MASK"):
        e.pop(k, None)
    e.update(env)
    e.update({"AUGMENT": "1", "BETA_CLIP": "-15:-2.3026", "HIER": "", "SCHED": "", "PROBE": str(a.every),
              "PROBE_DIR": os.path.join(work, "probe"), "PROBE_TENSOR": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
    out = os.path.join(work, "result.json")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child-verify", "--tree", tree, "--spec", spec, "--net", a.net,
                        "--alpha0", alpha0, "--expect-mask", expect_mask, "--wrong-mask", wrong_mask,
                        "--steps", str(a.steps), "--seed", str(a.seed), "--save", work, "--tag", tag, "--out", out],
                       env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode != 0 or not os.path.exists(out):
        print(p.stdout[-3000:])
        return None
    r = json.load(open(out))
    r["stdout"] = p.stdout.splitlines()
    print("  ran %-22s (verifier) tree=%s spec=%s env=%s alpha0=%s  %d steps on %s  stats %s bites %d"
          % (tag, tree.rstrip("/").split("/")[-2], spec[:14], env, alpha0, r["steps"], r["gpu"], r["stats"], r["bites"]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child-verify", action="store_true")
    ap.add_argument("--batch", default="")
    ap.add_argument("--tree", default="")
    ap.add_argument("--spec", default="")
    ap.add_argument("--net", default="")
    ap.add_argument("--alpha0", default="1e-6")
    ap.add_argument("--expect-mask", default="")
    ap.add_argument("--wrong-mask", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tree-pre", default="")
    ap.add_argument("--tree-post", default="")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    if a.child_verify:
        child_verify(a)
        return

    D = _load("cwd_design", os.path.join(REPO, "analysis", "cwd_design.py"))
    B = {"cwd1": D.CWD1, "cwd2": D.CWD2}[a.batch]
    a.net = B.NET
    if a.batch == "cwd1":
        RR = _load("test_grouphold_realrun", os.path.join(HERE, "test_grouphold_realrun.py"))
    else:
        RR = _load("test_voteweight_realrun", os.path.join(HERE, "test_voteweight_realrun.py"))
    chk, same = RR.chk, RR.same
    S9 = D.S9
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    os.makedirs(a.work, exist_ok=True)
    for k in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "COMP_HOLD", "REST_HOLD", "WINDOW_HOLD", "DECAY_MASK"):
        os.environ.pop(k, None)

    def sh(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    print("BATCH %s" % a.batch)
    print("DRIVER_SHA256 %s" % sh(os.path.abspath(__file__)))
    print("REGISTERED_TEST_SHA256 %s" % sh(RR.__file__))
    print("DESIGN_SHA256 %s" % sh(D.__file__))
    print("PATCH_SHA256 %s" % sh(os.path.join(REPO, "patches", "patch_decaymask.py")))
    for t, lab in ((pre, "pre"), (post, "post")):
        print("TREE_%s %s HF %s build_network %s" % (lab.upper(), t, sh(os.path.join(t, "Optimizers", "HF.py")), sh(os.path.join(t, "build_network.py"))))
    chk(sh(os.path.join(post, "Optimizers", "HF.py")) == B.HF_POST_SHA, "--tree-post's HF.py is the design's HF_POST_SHA (%s's tree)" % a.batch)
    hpre = open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read()
    chk(hashlib.sha256(hpre).hexdigest() == B.HF_PARENT_SHA and b"PATCH_DECAYMASK" not in hpre,
        "--tree-pre's HF.py is the parent's (%s) and carries no PATCH_DECAYMASK" % B.PARENT_TREE)
    wpre = os.path.join(post, "Optimizers", "HF.py.pre_decaymask")
    chk(os.path.exists(wpre) and open(wpre, "rb").read() == hpre, "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_decaymask")
    chk(sh(os.path.join(pre, "build_network.py")) == sh(os.path.join(post, "build_network.py")) == B.BN_SHA,
        "the two trees' build_network.py are BYTE-IDENTICAL and == the design's BN_SHA")
    print("\nsteps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026, net %s" % (a.steps, a.every, a.seed, B.NET))

    def env_of(arm, dm):
        e = {}
        if a.batch == "cwd2":
            if B.BETAHOLD[arm]:
                e["BETA_HOLD"] = B.BETAHOLD[arm]
            if B.COMPHOLD[arm]:
                e["COMP_HOLD"] = B.COMPHOLD[arm]
        if dm is not None:
            e["DECAY_MASK"] = dm
        return e

    def run_arm(tag, tree, arm, dm):
        e = env_of(arm, dm)
        old = dict((k, os.environ.pop(k, None)) for k in ("BETA_HOLD", "COMP_HOLD", "DECAY_MASK"))
        os.environ.update(e)
        try:
            if a.batch == "cwd1":
                return RR.run_variant(a, tag, tree, B.SPEC[arm], None)
            return RR.run_variant(a, tag, tree, B.SPEC[arm], None)
        finally:
            for k in ("BETA_HOLD", "COMP_HOLD", "DECAY_MASK"):
                os.environ.pop(k, None)
                if old[k] is not None:
                    os.environ[k] = old[k]

    def lines(R, prefix):
        return [ln for ln in (R["stdout"] if R else []) if ln.startswith(prefix)]

    print("\nRR0 DETERMINISM CONTROL (the parent tree, twice)")
    A1 = run_arm("pre_k01_a", pre, "k01", None)
    A2 = run_arm("pre_k01_b", pre, "k01", None)
    same(A1, A2, "RR0 parent k01 vs itself")

    # the distinct unmasked configurations: an arm stands for its twin (same SPEC and holds)
    if a.batch == "cwd1":
        OFF = [("k01", "k01"), ("kLNWD", "layerwise")]          # (arm whose SPEC/holds, label)
    else:
        OFF = [("k01", "k01"), ("HIGHHEADPATH", "HIGHHEADPATH"), ("LOWWD0", "LOWHEADPATH")]
    REF = {"k01": A1}
    print("\nRR1 OFF (DECAY_MASK unset / empty == the parent tree, bitwise)")
    for arm, lab in OFF:
        if lab not in REF:
            REF[lab] = run_arm("pre_%s" % lab.lower(), pre, arm, None)
        for val in (None, ""):
            vl = "unset" if val is None else "empty"
            R = run_arm("post_%s_%s" % (lab.lower(), vl), post, arm, val)
            same(REF[lab], R, "RR1 %-12s DECAY_MASK %s" % (lab, vl))
            want = ["DECAY_MASK: off"]
            ok = lines(R, "DECAY_MASK") == want and lines(R, "VOTE_W") == ["VOTE_W: off"]
            if a.batch == "cwd1":
                ok = ok and all(lines(R, k.split(":")[0]) == [k] for k in B.OFF_LINES)
            else:
                ok = ok and lines(R, "BETA_HOLD") == [B.WITNESS_BH[arm]] and lines(R, "COMP_HOLD") == [B.WITNESS_CH[arm]] \
                    and lines(R, "WINDOW_HOLD") == ["WINDOW_HOLD: off"]
            chk(ok, "RR1 %-12s DECAY_MASK %s printed `DECAY_MASK: off` and the parent's own witness lines" % (lab, vl))
            if R and R["probe_bytes"]:
                chk(b'"dm_' not in R["probe_bytes"], "RR1 %-12s DECAY_MASK %s: no dm_* key in probe.jsonl" % (lab, vl))

    print("\nRR2 THE EXACT MASKED STRINGS ON THE REAL PATH")
    ON = {}
    for arm in B.MASKED:
        R = run_arm("post_%s" % arm.lower(), post, arm, B.DMASK[arm])
        ON[arm] = R
        if R is None:
            continue
        k = len(D.dm_indices(B.DMASK[arm], B.TENSORS))
        chk(lines(R, "DECAY_MASK") == [B.WITNESS_DM[arm]] and lines(R, "VOTE_W") == ["VOTE_W: off"],
            "RR2 %-12s printed exactly the design's DECAY_MASK witness" % arm, (lines(R, "DECAY_MASK") or ["(none)"])[0][:100])
        recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
        bad = 0
        for r in recs:
            nrm = r.get("dm_norm")
            bad += not (r.get("dm_n") == r["step"] + 2 and r.get("dm_skipped") == r.get("dm_n", -1) * k and r.get("dm_masked") == k
                        and isinstance(r.get("dm_wdterm"), float) and r["dm_wdterm"] > 0 and isinstance(nrm, list) and len(nrm) == k
                        and all(isinstance(x, float) and x == x and abs(x) < 1e30 for x in nrm)
                        and isinstance(r.get("dm_small"), int) and isinstance(r.get("dm_absmin"), float))
        chk(len(recs) == a.steps // a.every and bad == 0,
            "RR2 %-12s at every probe record dm_n == step+2, dm_skipped == dm_n*%d, dm_wdterm > 0, %d finite norm(s)" % (arm, k, k),
            "%d records, %d bad; last %s" % (len(recs), bad, {kk: recs[-1].get(kk) for kk in ("dm_n", "dm_skipped", "dm_wdterm", "dm_absmin")} if recs else ""))
        if a.batch == "cwd2" and arm in B.HELD:
            twin = B.AS_CVT9[arm]
            badb = sum(1 for n, b in enumerate(R["betas"], 1) if b[1] != S9.hold_value(twin, n) or b[0] != S9.hold_c_value(n))
            chk(len(R["betas"]) == a.steps and badb == 0,
                "RR2 %-12s beta[1] == cvt9's %s schedule and beta[0] == HEADPATH after EVERY step" % (arm, twin), "%d bad" % badb)
            ref = REF["HIGHHEADPATH" if twin == "HIGHHEADPATH" else "LOWHEADPATH"]
            chk(ref is not None and R["betas"] == ref["betas"],
                "RR2 %-12s beta at every step BITWISE its unmasked twin's from the parent tree (the mask touches no step size)" % arm)
            print("  NOTE RR2 %-12s final weights %s the parent twin's (non-gating: step sizes ~1e-6 may sit below float resolution)"
                  % (arm, "identical to" if R["weights_sha256"] == ref["weights_sha256"] else "differ from"))

    print("\nRR3 UPDATE IDENTITY ON THE REAL PATH (verifier around base_update)")
    V0 = run_verify(a, "verify_pre_k01", pre, B.SPEC["k01"], env_of("k01", None), "1e-6", "", "")
    chk(V0 is not None and V0["steps"] == a.steps and V0["stats"]["expect"] == [0, 0, 0],
        "RR3 CONTROL the parent tree (OFF) matches the registered formula with an EMPTY mask at every step, every tensor",
        str(V0["stats"] if V0 else None))
    for arm in B.MASKED:
        idx = ",".join("%d" % (i + 1) for i in D.dm_indices(B.DMASK[arm], B.TENSORS))
        V = run_verify(a, "verify_%s" % arm.lower(), post, B.SPEC[arm], env_of(arm, B.DMASK[arm]), "1e-6", idx, "")
        chk(V is not None and V["steps"] == a.steps and V["stats"]["expect"] == [0, 0, 0]
            and [ln for ln in V["stdout"] if ln.startswith("DECAY_MASK")] == [B.WITNESS_DM[arm]],
            "RR3 %-12s w', h', m' BITWISE the formula with wd 0 on idx {%s} and 0.1 elsewhere, at every one of %d steps"
            % (arm, idx if len(idx) < 40 else idx[:36] + "...", a.steps), "%s; bites at float resolution %s" % (V["stats"] if V else None, V["bites"] if V else None))
    arm = B.MASKED[0]
    idx = ",".join("%d" % (i + 1) for i in D.dm_indices(B.DMASK[arm], B.TENSORS))
    VB = run_verify(a, "verify_big_on", post, B.SPEC["k01"], {"DECAY_MASK": B.DMASK[arm]}, "1e-2", idx, "")
    VO = run_verify(a, "verify_big_off", post, B.SPEC["k01"], {}, "1e-2", "", "")
    chk(VB is not None and VB["stats"]["expect"] == [0, 0, 0] and VB["bites"] > 0 and sum(VB["stats"]["wrong"][:2]) > 0,
        "RR3 NON-VACUITY (test-only alpha0 1e-2, DECAY_MASK=%s, scalar): the formula holds, the WD term is visible on masked tensors "
        "(bites > 0), and the SAME run checked against the EMPTY-mask formula FAILS" % B.DMASK[arm],
        "stats %s bites %s" % (VB["stats"] if VB else None, VB["bites"] if VB else None))
    chk(VO is not None and VB is not None and VO["stats"]["expect"] == [0, 0, 0] and VO["weights_sha256"] != VB["weights_sha256"],
        "RR3 NON-VACUITY: the test-only alpha0 1e-2 OFF run matches the empty-mask formula and its final weights DIFFER from the masked run's")

    fails = list(RR.FAILED)
    print("\n%s" % ("ALL PASS" if not fails else "FAILURES: %r" % fails))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
