"""BITE OF THE WEIGHT-DECAY FLAG ITSELF, AND OF PATCH_DECAYMASK AT A NEW WEIGHT DECAY.  `cwd5`, CORRECTIONS 281.

`cwd5` adds NO harness code.  Its axis is an EXISTING registered CLI float of train.py, `--weight-decay-base`
(default 0.1; cmo1 already ran it at 0).  But the whole batch rests on ONE premise that no landed proof covers:

    THAT CHANGING THAT FLAG CHANGES THE UPDATE, AT EVERY RUNG, IN THE WAY THE LADDER ASSUMES.

Every registered real-run proof in this campaign (tests/test_decaymask_realrun.py, tests/test_grouphold_realrun.py)
hard-codes `--weight-decay-base 0.1` in the child's argv, so none of them has ever exercised another value.  And
PATCH_DECAYMASK has never run at a weight decay other than 0.1, so `cwd5`'s CARW2 witness -- which prints `wd=0.01`
-- is a string the harness has never produced.  This driver closes both gaps on the real GPU path.

THE NEW CODE IS ONE THING ONLY: a child process that is a FAITHFUL TRANSCRIPTION of the registered child-verify of
tests/test_decaymask_realrun.py with the weight decay PARAMETERISED instead of fixed at 0.1.  Nothing in the harness
is touched, nothing registered is edited (RULE 16), and the transcription is not taken on trust:

  RW-EQ  EQUIVALENCE TO THE REGISTERED DRIVER.  At wd 0.1 -- the ONE value the registered driver can run -- this
         driver's child and tests/test_decaymask_realrun.py's REGISTERED child must agree BITWISE on the same tree,
         spec, seed and mask: identical final-weight sha256, identical beta trajectory, identical stats and bites.
         Run OFF and with the carrier mask.  If the transcription had drifted, this fails.
  RW0    DETERMINISM CONTROL: the parent tree (cvt8's, no PATCH_DECAYMASK) run twice at wd 0.1 -> identical.
  RW1    INERTNESS AT THE ANCHOR: on cwd1's tree with DECAY_MASK unset, wd 0.1 is BITWISE the parent tree.
  RW2    THE LADDER'S PREMISE, THE POINT OF THIS JOB.  For EACH of cwd5's four rung tokens (0.1, 1e-2, 1e-3, 5e-4),
         an UNMASKED run on cwd1's tree: (a) at EVERY step and EVERY tensor w', h' and m' are BITWISE
             w' = w - a*(m + wd*w)   h' = gamma*(1 - wd*a)*h - a*(m + wd*w)   m' = mp*m + (1-mp)*g
         with THE PASSED wd; (b) the SAME run checked against the formula at a DIFFERENT wd FAILS (non-vacuity --
         the verifier can tell the rungs apart); (c) the four runs' final weights are PAIRWISE DISTINCT, so the flag
         genuinely bites and no two rungs are secretly the same experiment.
  RW3    PATCH_DECAYMASK AT A NEW WEIGHT DECAY: the three ctd1 carriers masked at wd 1e-2 -- (a) prints EXACTLY
         cwd5's registered witness, `wd=0.01 ... masked=3 ... idx=50,53,59`; (b) every probe record carries
         dm_n == step+2, dm_skipped == dm_n*3, dm_masked == 3, dm_wdterm > 0 and three finite norms; (c) the update
         is bitwise the formula with wd 0 on {50,53,59} and 0.01 elsewhere; (d) WHICH ROUTE BITES IS MEASURED, NOT
         ASSUMED -- at the run's own alpha0 1e-6 the skipped term is a*wd*|w| ~ 1e-8, BELOW float32 resolution next
         to |w| ~ 1, so the WEIGHT UPDATE is bitwise unchanged while the META TRACE differs at every step; at a
         test-only alpha0 1e-2 both routes bite and the masked and unmasked runs' final weights differ.  The
         coincidence of the two weight trajectories at alpha0 1e-6 over 300 steps is therefore EXPLAINED and
         DECLARED, and is a limit of the PROOF (300 steps at the clamped initial step size), not of the batch,
         whose runs are 50,000 steps at a LEARNED, rising alpha.
  RW4    LOUDNESS AT THE NEW VALUE: at wd 1e-2 a misspelt name, a duplicate, an empty `+` token, a name outside the
         model and a bad character each raise ValueError at construction; and at wd 0 a mask raises the patch's own
         "a mask would be vacuous" ValueError.  A typo cannot run silently unmasked at any rung.

IMPORTED, UNEDITED: tests/test_decaymask_realrun.py (the REGISTERED child, for RW-EQ), tests/test_grouphold_realrun.py
(chk / same / FAILED), analysis/cwd5_design.py (the rung tokens, the mask string, the witnesses, the indices).

  python3 tests/test_decaymask_realrun_cwd5.py --tree-pre $WS/harness_cvt8/cifar10 --tree-post $WS/harness_cwd1/cifar10 \\
      --steps 300 --every 10 --seed 146 --work /tmp/cwd5_realrun
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

REUSED_DRIVER_SHA = "0c64e50ffaac6a2a483f68f81e1b9634199ef7a2a82545964351cad3cfb396bb"   # tests/test_decaymask_realrun.py


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# =====================================================================================================================
# THE CHILD -- a faithful transcription of tests/test_decaymask_realrun.py's registered child_verify with the base
# weight decay PARAMETERISED.  RW-EQ proves the transcription against the registered child at wd 0.1.
# =====================================================================================================================
def child_verify_wd(a):
    tree = a.tree
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
                "--weight-decay-base", a.wd, "--alg-meta", "Lion", "--momentum-param-meta", "0.99",
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
    # stats["expect"] uses the run's OWN weight decay; stats["wrongwd"] uses --wrong-wd (non-vacuity across rungs)
    stats = {"expect": [0, 0, 0], "wrong": [0, 0, 0], "wrongwd": [0, 0, 0]}
    bites = [0]
    wd_wrong = float(a.wrong_wd) if a.wrong_wd else None
    orig = optimizer.base_update
    seen = {"wd": None}

    def wrapped(netx, g):
        wb = [p.data.clone() for p in params]
        mb = [m.clone() if torch.is_tensor(m) else m for m in optimizer.momentum_base]
        hb = [h.clone() for h in optimizer.h_condenced]
        orig(netx, g)
        mp = optimizer.args_base["momentum_param"]
        wd = optimizer.args_base["weight_decay"]
        seen["wd"] = float(wd)
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
            if wd_wrong is not None:
                wdd2 = al * (mb[i] + wd_wrong * wb[i])
                if i in masks["expect"]:
                    w_e2, h_e2 = wb[i] - nod, optimizer.gamma * hb[i] - nod
                else:
                    w_e2, h_e2 = wb[i] - wdd2, optimizer.gamma * (1 - wd_wrong * al) * hb[i] - wdd2
                stats["wrongwd"][0] += not torch.equal(p.data, w_e2)
                stats["wrongwd"][1] += not torch.equal(optimizer.h_condenced[i], h_e2)
                stats["wrongwd"][2] += not torch.equal(optimizer.momentum_base[i], m_exp)

    optimizer.base_update = wrapped
    betas, losses, t = [], [], 0
    while t < a.steps:
        for data in trainloader:
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            loss = criterion(net(inputs), labels)
            losses.append(float(loss.detach().item()))
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
    json.dump({"stats": stats, "bites": bites[0], "steps": t, "betas": betas, "losses": losses,
               "weights_sha256": h.hexdigest(), "wd_seen": seen["wd"],
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}, open(a.out, "w"))


def run_wd(a, tag, tree, spec, env, alpha0, wd, expect_mask="", wrong_mask="", wrong_wd=""):
    work = os.path.join(a.work, tag)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    e = dict(os.environ)
    for k in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "COMP_HOLD", "REST_HOLD", "WINDOW_HOLD", "DECAY_MASK"):
        e.pop(k, None)
    e.update(env)
    e.update({"AUGMENT": "1", "BETA_CLIP": "-15:-2.3026", "HIER": "", "SCHED": "", "PROBE": str(a.every),
              "PROBE_DIR": os.path.join(work, "probe"), "PROBE_TENSOR": "1", "PYTHONDONTWRITEBYTECODE": "1",
              "PYTHONUNBUFFERED": "1"})
    out = os.path.join(work, "result.json")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child-verify-wd", "--tree", tree, "--spec", spec,
                        "--net", a.net, "--alpha0", alpha0, "--wd", wd, "--expect-mask", expect_mask,
                        "--wrong-mask", wrong_mask, "--wrong-wd", wrong_wd, "--steps", str(a.steps),
                        "--seed", str(a.seed), "--save", work, "--tag", tag, "--out", out],
                       env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode != 0 or not os.path.exists(out):
        print(p.stdout[-3000:])
        return None
    r = json.load(open(out))
    r["stdout"] = p.stdout.splitlines()
    pj = os.path.join(work, "probe", "probe.jsonl")
    r["probe_bytes"] = open(pj, "rb").read() if os.path.exists(pj) else b""
    print("  ran %-26s tree=%s wd=%-6s alpha0=%s env=%s  %d steps on %s  stats %s bites %d  weights %s..."
          % (tag, tree.rstrip("/").split("/")[-2], wd, alpha0, env, r["steps"], r["gpu"], r["stats"], r["bites"],
             r["weights_sha256"][:12]))
    return r


def child_loud(tree, wd):
    """RW4 in a clean process: HF.init_meta on the live model with bad DECAY_MASK values AT THIS WEIGHT DECAY."""
    os.chdir(tree)
    sys.path.insert(0, tree)
    import io, contextlib
    import torch
    from build_network import build_network
    from Optimizers.HF import HF
    net = build_network("ResNet18_c100", "cpu")
    nps = [(n, p.data.size()) for n, p in net.named_parameters()]
    good = "layer4.0.bn2.weight+layer4.0.shortcut.1.weight+layer4.1.bn2.weight"
    cases = [("misspelt-one", "layer4.0.bn2.weigth", float(wd)),
             ("misspelt-triple", "layer4.0.bn2.weight+layer4.0.shortcut.1.weigth+layer4.1.bn2.weight", float(wd)),
             ("duplicate", "layer4.0.bn2.weight+layer4.0.bn2.weight", float(wd)),
             ("empty-token", "layer4.0.bn2.weight++layer4.1.bn2.weight", float(wd)),
             ("not-a-name", "layer5.0.bn2.weight", float(wd)),
             ("bad-char", "layer4.0.bn2.weight,layer4.1.bn2.weight", float(wd)),
             ("vacuous-at-wd0", good, 0.0)]
    for lab, v, wdv in cases:
        for k in ("VOTE_W", "BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD", "GROUP_HOLD", "REST_HOLD", "DECAY_MASK"):
            os.environ.pop(k, None)
        os.environ["DECAY_MASK"] = v
        o = HF.__new__(HF)
        o.num_layers = len(nps)
        o._device = torch.device("cpu")
        o.args_meta = {"alg": "Lion", "meta_stepsize": 1e-3, "momentum_param": 0.99, "Lion_beta2": 0.9, "weight_decay": 0}
        o.args_base = {"alg": "SGDm", "weight_decay": wdv, "momentum_param": 0.99}
        o._beta_lo, o._beta_hi = -15.0, -2.3026
        o._hier = "none"
        o.base_update = o.SGDm_base_update
        raised = "none"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                HF.init_meta(o, "scalar", nps, 1e-6)
        except ValueError as ex:
            raised = "ValueError: %s" % str(ex)[:100]
        except Exception as ex:
            raised = "OTHER %s" % type(ex).__name__
        print("LOUD %s wd=%r %s" % (lab, wdv, raised))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child-verify-wd", action="store_true")
    ap.add_argument("--child-loud", action="store_true")
    ap.add_argument("--tree", default="")
    ap.add_argument("--spec", default="scalar")
    ap.add_argument("--net", default="ResNet18_c100")
    ap.add_argument("--alpha0", default="1e-6")
    ap.add_argument("--wd", default="0.1")
    ap.add_argument("--wrong-wd", default="")
    ap.add_argument("--expect-mask", default="")
    ap.add_argument("--wrong-mask", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tree-pre", default="")
    ap.add_argument("--tree-post", default="")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=146)
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    if a.child_verify_wd:
        child_verify_wd(a)
        return
    if a.child_loud:
        child_loud(a.tree, a.wd)
        return

    D = _load("cwd5_design", os.path.join(REPO, "analysis", "cwd5_design.py"))
    B = D.CWD5
    TD = _load("test_decaymask_realrun", os.path.join(HERE, "test_decaymask_realrun.py"))
    RR = _load("test_grouphold_realrun", os.path.join(HERE, "test_grouphold_realrun.py"))
    chk, same = RR.chk, RR.same
    a.net = B.NET
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    os.makedirs(a.work, exist_ok=True)
    for k in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "COMP_HOLD", "REST_HOLD", "WINDOW_HOLD", "DECAY_MASK"):
        os.environ.pop(k, None)

    def sh(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    CARSPEC = D.CARRIER_SPEC
    CARIDX = ",".join("%d" % (i + 1) for i in D.dm_indices(CARSPEC, B.TENSORS))

    print("BATCH cwd5")
    print("DRIVER_SHA256 %s" % sh(os.path.abspath(__file__)))
    print("REUSED_DRIVER_SHA256 %s" % sh(TD.__file__))
    print("REGISTERED_TEST_SHA256 %s" % sh(RR.__file__))
    print("DESIGN_SHA256 %s" % sh(D.__file__))
    print("CWD_DESIGN_SHA256 %s" % sh(D.W.__file__))
    print("PATCH_SHA256 %s" % sh(os.path.join(REPO, "patches", "patch_decaymask.py")))
    for t, lab in ((pre, "pre"), (post, "post")):
        print("TREE_%s %s HF %s build_network %s"
              % (lab.upper(), t, sh(os.path.join(t, "Optimizers", "HF.py")), sh(os.path.join(t, "build_network.py"))))
    print("WD_LADDER %s" % ",".join("%s=%s" % (r, D.WD_TOKEN[r]) for r in D.RUNG_IDS))
    print("CARRIER_SET idx=%s spec=%s" % (CARIDX, CARSPEC))
    chk(sh(os.path.join(post, "Optimizers", "HF.py")) == B.HF_POST_SHA,
        "--tree-post's HF.py is cwd1's PATCH_DECAYMASK tree, UNCHANGED (94aedc33...)")
    hpre = open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read()
    chk(hashlib.sha256(hpre).hexdigest() == B.HF_PARENT_SHA and b"PATCH_DECAYMASK" not in hpre,
        "--tree-pre's HF.py is cvt8's and carries no PATCH_DECAYMASK")
    wpre = os.path.join(post, "Optimizers", "HF.py.pre_decaymask")
    chk(os.path.exists(wpre) and open(wpre, "rb").read() == hpre,
        "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_decaymask")
    chk(sh(os.path.join(pre, "build_network.py")) == sh(os.path.join(post, "build_network.py")) == B.BN_SHA,
        "the two trees' build_network.py are BYTE-IDENTICAL and == the design's BN_SHA")
    chk(sh(TD.__file__) == REUSED_DRIVER_SHA and sh(D.W.__file__) == D.CWD_DESIGN_SHA
        and sh(os.path.join(REPO, "patches", "patch_decaymask.py")) == D.PATCH_DECAYMASK_SHA,
        "the reused registered driver, cwd_design.py and patch_decaymask.py are cwd1's registered files, UNEDITED (RULE 16)")
    print("\nsteps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026, net %s, scalar"
          % (a.steps, a.every, a.seed, B.NET))

    # ---- RW-EQ ------------------------------------------------------------------------------------------------------
    print("\nRW-EQ EQUIVALENCE OF THE NEW wd-PARAMETERISED CHILD TO THE REGISTERED ONE, AT wd 0.1 (the only value the")
    print("      registered child can run).  If the transcription had drifted, this is where it shows.")
    for lab, env, em in (("off", {}, ""), ("carriers", {"DECAY_MASK": CARSPEC}, CARIDX)):
        Vr = TD.run_verify(a, "eq_reg_%s" % lab, post, "scalar", env, "1e-6", em, "")
        Vn = run_wd(a, "eq_new_%s" % lab, post, "scalar", env, "1e-6", "0.1", em, "")
        ok = (Vr is not None and Vn is not None and Vr["weights_sha256"] == Vn["weights_sha256"]
              and Vr["betas"] == Vn["betas"] and Vr["steps"] == Vn["steps"]
              and Vr["stats"]["expect"] == Vn["stats"]["expect"] == [0, 0, 0] and Vr["bites"] == Vn["bites"])
        chk(ok, "RW-EQ %-8s the new child and the REGISTERED child agree BITWISE at wd 0.1 (weights, betas, stats, bites)"
            % lab, "" if ok else "reg %s new %s" % (Vr and Vr["weights_sha256"][:12], Vn and Vn["weights_sha256"][:12]))
        if Vn is not None:
            chk(Vn["wd_seen"] == 0.1, "RW-EQ %-8s the optimiser's own args_base['weight_decay'] is 0.1" % lab)

    # ---- RW0 / RW1 --------------------------------------------------------------------------------------------------
    print("\nRW0 DETERMINISM CONTROL (the parent tree, twice, wd 0.1)")
    A1 = run_wd(a, "pre_k01_a", pre, "scalar", {}, "1e-6", "0.1")
    A2 = run_wd(a, "pre_k01_b", pre, "scalar", {}, "1e-6", "0.1")
    same(A1, A2, "RW0 parent k01 vs itself")

    print("\nRW1 INERTNESS AT THE ANCHOR (cwd1's tree, DECAY_MASK unset, wd 0.1 == the parent tree, bitwise)")
    R1 = run_wd(a, "post_k01_unset", post, "scalar", {}, "1e-6", "0.1")
    same(A1, R1, "RW1 k01 DECAY_MASK unset")
    if R1 is not None:
        chk([ln for ln in R1["stdout"] if ln.startswith("DECAY_MASK")] == ["DECAY_MASK: off"]
            and all([ln for ln in R1["stdout"] if ln.startswith(k.split(":")[0])] == [k] for k in B.OFF_LINES),
            "RW1 printed `DECAY_MASK: off` and the parent's own witness lines")
        chk(b'"dm_' not in R1["probe_bytes"], "RW1 no dm_* key in probe.jsonl on an unmasked run")

    # ---- RW2: THE LADDER'S PREMISE ----------------------------------------------------------------------------------
    print("\nRW2 THE LADDER'S PREMISE -- the four rung tokens on the real GPU path, unmasked")
    shas = {}
    for r in D.RUNG_IDS:
        tok = D.WD_TOKEN[r]
        other = D.WD_TOKEN["W2" if r != "W2" else "W1"]
        V = run_wd(a, "rung_%s" % r, post, "scalar", {}, "1e-6", tok, "", "", other)
        if V is None:
            chk(False, "RW2 %s (wd %s) ran" % (r, tok))
            continue
        shas[r] = V["weights_sha256"]
        chk(V["steps"] == a.steps and V["stats"]["expect"] == [0, 0, 0] and V["wd_seen"] == D.WD_VALUE[r],
            "RW2 %s wd %-5s: w', h', m' BITWISE the coupled formula with THE PASSED wd at every one of %d steps, every"
            " tensor; the optimiser's own args_base['weight_decay'] is %r" % (r, tok, a.steps, D.WD_VALUE[r]),
            str(V["stats"]))
        chk(sum(V["stats"]["wrongwd"]) > 0,
            "RW2 %s wd %-5s: NON-VACUITY -- the SAME run checked against the formula at wd %s FAILS, so the verifier"
            " genuinely distinguishes the rungs" % (r, tok, other), "wrongwd %s" % V["stats"]["wrongwd"])
    chk(len(shas) == 4 and len(set(shas.values())) == 4,
        "RW2 THE FLAG BITES: the four rungs' final weights are PAIRWISE DISTINCT -- no two rungs are secretly the same"
        " experiment", " ".join("%s %s" % (k, v[:10]) for k, v in sorted(shas.items())))

    # ---- RW3: PATCH_DECAYMASK AT A NEW WEIGHT DECAY -----------------------------------------------------------------
    rung = D.CARRIER_RUNG
    tok = D.WD_TOKEN[rung]
    print("\nRW3 PATCH_DECAYMASK AT A WEIGHT DECAY IT HAS NEVER RUN AT (%s carriers at wd %s)" % (CARIDX, tok))
    VM = run_wd(a, "carw2", post, "scalar", {"DECAY_MASK": CARSPEC}, "1e-6", tok, CARIDX, "", "")
    if VM is None:
        chk(False, "RW3 the masked run at wd %s ran" % tok)
    else:
        got = [ln for ln in VM["stdout"] if ln.startswith("DECAY_MASK")]
        chk(got == [B.WITNESS_DM["CARW2"]],
            "RW3 printed EXACTLY cwd5's registered witness -- `wd=0.01 ... masked=3 ... idx=%s`" % CARIDX,
            (got or ["(none)"])[0][:150])
        recs = [json.loads(x) for x in VM["probe_bytes"].decode().splitlines()]
        bad = 0
        for rr in recs:
            nrm = rr.get("dm_norm")
            bad += not (rr.get("dm_n") == rr["step"] + 2 and rr.get("dm_skipped") == rr.get("dm_n", -1) * 3
                        and rr.get("dm_masked") == 3 and isinstance(rr.get("dm_wdterm"), float) and rr["dm_wdterm"] > 0
                        and isinstance(nrm, list) and len(nrm) == 3
                        and all(isinstance(x, float) and x == x and abs(x) < 1e30 for x in nrm)
                        and isinstance(rr.get("dm_small"), int) and isinstance(rr.get("dm_absmin"), float))
        chk(len(recs) == a.steps // a.every and bad == 0,
            "RW3 at every probe record dm_n == step+2, dm_skipped == dm_n*3, dm_masked == 3, dm_wdterm > 0, 3 finite norms",
            "%d records, %d bad; last %s" % (len(recs), bad,
                                             {k: recs[-1].get(k) for k in ("dm_n", "dm_skipped", "dm_wdterm")} if recs else ""))
        chk(VM["stats"]["expect"] == [0, 0, 0] and VM["wd_seen"] == D.WD_VALUE[rung],
            "RW3 w', h', m' BITWISE the formula with wd 0 on idx {%s} and %s elsewhere, at every step" % (CARIDX, tok),
            str(VM["stats"]))
        # WHICH ROUTE BITES AT THE RUN'S OWN alpha0, MEASURED RATHER THAN ASSUMED.  `wrong` here is the FULLY UNMASKED
        # formula (wrong_mask is empty), so wrong[0] / wrong[1] count the steps at which the mask changed the WEIGHT
        # UPDATE / the META TRACE.  At alpha0 1e-6 and wd 1e-2 the skipped term is a*wd*|w| ~ 1e-8, BELOW float32
        # resolution next to |w| ~ 1, so the weight update is bitwise unchanged while the trace is not.  That is a fact
        # about the FIRST 300 STEPS AT THE CLAMPED INITIAL STEP SIZE, not about the 50,000-step batch runs, and it is
        # reported, not asserted away.
        chk(VM["stats"]["wrong"][1] > 0,
            "RW3 THE MASK BITES THE META TRACE at the run's own alpha0: h differs from the unmasked formula at %d of"
            " %d tensor-steps" % (VM["stats"]["wrong"][1], 3 * a.steps),
            "wrong (vs the unmasked formula) w=%d h=%d m=%d; weight-update bites at float resolution %d"
            % (VM["stats"]["wrong"][0], VM["stats"]["wrong"][1], VM["stats"]["wrong"][2], VM["bites"]))
        chk((VM["bites"] == 0) == (VM["stats"]["wrong"][0] == 0),
            "RW3 CONSISTENCY: the weight update differs from the unmasked one on exactly the steps where the skipped"
            " decay term is representable in float32 (bites %d, w-mismatches %d)" % (VM["bites"], VM["stats"]["wrong"][0]))
        VW = run_wd(a, "carw2_vs_unmasked", post, "scalar", {"DECAY_MASK": CARSPEC}, "1e-2", tok, CARIDX, "")
        chk(VW is not None and VW["stats"]["expect"] == [0, 0, 0] and VW["bites"] > 0
            and sum(VW["stats"]["wrong"][:2]) > 0,
            "RW3 at a test-only alpha0 1e-2 the masked run matches ITS OWN set exactly, the skipped decay term is"
            " representable (bites %d), and the run FAILS the fully unmasked formula" % (VW["bites"] if VW else -1),
            "stats %s" % (VW["stats"] if VW else None))
        VU = run_wd(a, "unmasked_w2_big", post, "scalar", {}, "1e-2", tok, "", CARIDX)
        chk(VU is not None and VU["stats"]["expect"] == [0, 0, 0] and sum(VU["stats"]["wrong"][:2]) > 0,
            "RW3 NON-VACUITY: at alpha0 1e-2 an UNMASKED run at the same wd FAILS the masked formula -- the two are"
            " distinguishable on the real path", "stats %s" % (VU["stats"] if VU else None))
        chk(VW is not None and VU is not None and VW["weights_sha256"] != VU["weights_sha256"],
            "RW3 at alpha0 1e-2 the masked run's final weights DIFFER from the unmasked run's at the same weight decay",
            "%s vs %s" % (VW["weights_sha256"][:12] if VW else None, VU["weights_sha256"][:12] if VU else None))
        chk("W2" in shas and (VM["weights_sha256"] == shas["W2"]) == (VM["stats"]["wrong"][0] == 0),
            "RW3 DECLARED LIMIT, MEASURED: at alpha0 1e-6 the masked and unmasked weight trajectories coincide over 300"
            " steps EXACTLY BECAUSE the weight update never differed in float32; the trace difference has not reached"
            " the weights in 300 steps.  The batch's runs are 50,000 steps at a LEARNED, rising alpha",
            "masked %s vs rung W2 %s" % (VM["weights_sha256"][:12], shas.get("W2", "")[:12]))

    # ---- RW4: loudness ----------------------------------------------------------------------------------------------
    print("\nRW4 LOUDNESS AT THE NEW WEIGHT DECAY (CPU, the real HF.init_meta)")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child-loud", "--tree", post, "--wd", tok],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
    loud = [ln for ln in p.stdout.splitlines() if ln.startswith("LOUD ")]
    for ln in loud:
        print("  " + ln)
    chk(p.returncode == 0 and len(loud) == 7 and all(" ValueError: PATCH_DECAYMASK" in ln for ln in loud),
        "RW4 at wd %s a misspelt name, a misspelt triple, a duplicate, an empty + token, a non-name and a bad character"
        " EACH raise ValueError -- and at wd 0 a mask raises the patch's own vacuity error" % tok)
    chk(any("vacuous" in ln for ln in loud),
        "RW4 the wd-0 case raises the patch's `a mask would be vacuous` error specifically")

    fails = list(RR.FAILED)
    print("\n%s" % ("ALL PASS" if not fails else "FAILURES: %r" % fails))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
