"""BITWISE INERTNESS OF PATCH_VOTEWEIGHT OVER A SHORT *REAL* RUN.  CORRECTIONS 227.

tests/test_voteweight.py proves structure and arithmetic on synthetic batches (the
registered test_probe_tensor pattern).  This file closes the remaining gap: the
REAL code path of a cvt1 run -- train.py's own parse_args, build_network
(PlainNet18_c100), build_optimizer, load_data (CIFAR-100, AUGMENT=1, the seed's
shuffle and crops), the loss, HF.step with BETA_CLIP, PROBE and PROBE_TENSOR
exactly as the batch sets them -- run for --steps optimizer steps on a GPU, from
TWO trees:

  --tree-pre   an UNPATCHED tree with the SAME build_network.py (cpl1's POST_SHA)
               and the SAME HF.py bytes the cvt1 tree's HF.py.pre_voteweight holds
               ($WS/harness_cpl1/cifar10; used READ-ONLY, PYTHONDONTWRITEBYTECODE=1)
  --tree-post  the cvt1 tree (PATCH_VOTEWEIGHT)

Each variant runs in its OWN python process (so each imports its own tree), with
cudnn.deterministic=True and cudnn.benchmark=False set identically in every child.

  RR0  DETERMINISM CONTROL: the unpatched tree run twice gives identical beta at
       every step, identical probe.jsonl bytes and identical final weights.  If
       this fails the bitwise claim cannot be tested on this node, and the whole
       test FAILS (it never passes vacuously).
  RR1  OFF (VOTE_W unset, and VOTE_W empty) on k01's spec and on HEAD's spec: beta
       at every step, the per-step loss, probe.jsonl bytes and the sha256 of every
       parameter AND buffer after the last step == the unpatched tree's.
  RR2  IDENTITY (the arm's tensor at weight 1): the same four equalities.
  RR3  THE SWITCH BITES, on the real path: MUTE, DOSE (k01's spec) and INJECT
       (HEAD's spec) print the registered witness line; at every probe record
       z_agg == sum_i w_i z_tensor_i (rel 1e-5); and z_agg DIFFERS from the
       unpatched run's on at least one record.

RUN (on a GPU node; ~10 min):
  python3 tests/test_voteweight_realrun.py --tree-pre $WS/harness_cpl1/cifar10 \\
      --tree-post $WS/harness_cvt1/cifar10 --steps 300 --work $WS/runs/cvt1/realrun
"""
import argparse, hashlib, json, os, shutil, subprocess, sys

FAILED = []
K01 = "scalar"
HEAD = "sets:1-49,51-53/layer4.1.bn2.weight"
VOTEW = {"MUTE": "layer4.1.bn2.weight:0", "DOSE": "layer4.1.bn2.weight:0.1", "INJECT": "layer4.1.bn1.weight:691"}
WEIGHTS = {"MUTE": {49: 0.0}, "DOSE": {49: 0.1}, "INJECT": {46: 691.0}}
WITNESS = {
    "MUTE": "VOTE_W: on type=scalar items=50:layer4.1.bn2.weight:w=0.0:group=0:groupsize=53",
    "DOSE": "VOTE_W: on type=scalar items=50:layer4.1.bn2.weight:w=0.1:group=0:groupsize=53",
    "INJECT": "VOTE_W: on type=blockwise items=47:layer4.1.bn1.weight:w=691.0:group=0:groupsize=52",
}


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def child(a):
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
    # train.py cannot be imported (its last two statements, writer.close() and the
    # minutes print, sit at module level OUTSIDE the __main__ guard), so the tree's
    # OWN parse_args is lifted out of its source by AST and executed unchanged.
    import ast as _ast, argparse as _argparse, types as _types
    _tsrc = open(os.path.join(tree, "train.py")).read()
    _fn = [n for n in _ast.parse(_tsrc).body if isinstance(n, _ast.FunctionDef) and n.name == "parse_args"]
    assert len(_fn) == 1, "train.py has no single parse_args"
    TR = _types.ModuleType("train_parse_args")
    TR.argparse = _argparse
    exec(compile(_ast.Module(body=_fn, type_ignores=[]), os.path.join(tree, "train.py"), "exec"), TR.__dict__)
    sys.argv =["train.py", "--optimizer", "HF", "--alg-base", "SGDm", "--momentum-param-base", "0.99",
                "--weight-decay-base", "0.1", "--alg-meta", "Lion", "--momentum-param-meta", "0.99",
                "--Lion-beta2-meta", "0.9", "--weight-decay-meta", "0", "--dataset", "CIFAR100",
                "--NN-name", "PlainNet18_c100", "--batch-size", "100", "--max-time", "999:00:00",
                "--gamma", "1", "--meta-stepsize", "1e-3", "--alpha0", "1e-6", "--num-epochs", "100",
                "--stepsize-groups", a.spec, "--seed", str(a.seed), "--save-directory", a.save,
                "--run-name", a.tag]
    args = TR.parse_args()
    # train.py's own None-normalisation, verbatim in effect
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
    betas, losses, t = [], [], 0
    while t < a.steps:
        for data in trainloader:
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
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
    json.dump({"betas": betas, "losses": losses, "weights_sha256": h.hexdigest(),
               "device": str(args.device),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
              open(a.out, "w"))


def run_variant(a, tag, tree, spec, vw):
    work = os.path.join(a.work, tag)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    env = dict(os.environ)
    env.pop("VOTE_W", None)
    if vw is not None:
        env["VOTE_W"] = vw
    env.update({"AUGMENT": "1", "BETA_CLIP": "-15:-2.3026", "HIER": "", "SCHED": "",
                "PROBE": str(a.every), "PROBE_DIR": os.path.join(work, "probe"), "PROBE_TENSOR": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
    out = os.path.join(work, "result.json")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", "--tree", tree, "--spec", spec,
                        "--steps", str(a.steps), "--seed", str(a.seed), "--save", work, "--tag", tag, "--out", out],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode != 0 or not os.path.exists(out):
        print(p.stdout[-3000:])
        chk(False, "%s child ran" % tag, "rc %d" % p.returncode)
        return None
    res = json.load(open(out))
    pj = os.path.join(work, "probe", "probe.jsonl")
    res["probe_bytes"] = open(pj, "rb").read() if os.path.exists(pj) else b""
    res["stdout"] = p.stdout.splitlines()
    print("  ran %-14s tree=%s spec=%s VOTE_W=%s  %d steps on %s  weights %s..."
          % (tag, os.path.basename(os.path.dirname(tree.rstrip("/"))), spec[:14], vw, len(res["betas"]),
             res["gpu"], res["weights_sha256"][:12]))
    return res


def same(x, y, lab):
    ok = x is not None and y is not None
    chk(ok and x["betas"] == y["betas"], "%s: beta IDENTICAL at every step" % lab,
        "%d steps" % (len(x["betas"]) if x else 0))
    chk(ok and x["losses"] == y["losses"], "%s: per-step loss IDENTICAL" % lab)
    chk(ok and x["probe_bytes"] == y["probe_bytes"] and len(x["probe_bytes"]) > 0,
        "%s: probe.jsonl BYTE-IDENTICAL" % lab, "%d records" % (x["probe_bytes"].count(b"\n") if x else 0))
    chk(ok and x["weights_sha256"] == y["weights_sha256"], "%s: every parameter and buffer after the last step BITWISE-IDENTICAL" % lab,
        (x["weights_sha256"][:16] if x else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--tree", default="")
    ap.add_argument("--spec", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tree-pre", default="")
    ap.add_argument("--tree-post", default="")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=78)
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    if a.child:
        return child(a)
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    os.makedirs(a.work, exist_ok=True)
    for t, lab in ((pre, "pre"), (post, "post")):
        hf = open(os.path.join(t, "Optimizers", "HF.py"), "rb").read()
        bn = open(os.path.join(t, "build_network.py"), "rb").read()
        print("TREE_%s %s HF %s build_network %s" % (lab.upper(), t, hashlib.sha256(hf).hexdigest(), hashlib.sha256(bn).hexdigest()))
    chk(b"PATCH_VOTEWEIGHT" not in open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read()
        and b"PATCH_VOTEWEIGHT" in open(os.path.join(post, "Optimizers", "HF.py"), "rb").read(),
        "--tree-pre is unpatched, --tree-post carries PATCH_VOTEWEIGHT")
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
    H0 = run_variant(a, "pre_head", pre, HEAD, None)

    print("\nRR1 OFF")
    same(A1, run_variant(a, "post_k01_unset", post, K01, None), "RR1 k01  VOTE_W unset")
    E = run_variant(a, "post_k01_empty", post, K01, "")
    same(A1, E, "RR1 k01  VOTE_W empty")
    chk(E is not None and "VOTE_W: off" in E["stdout"], "RR1 the patched run printed `VOTE_W: off`")
    same(H0, run_variant(a, "post_head_unset", post, HEAD, None), "RR1 HEAD VOTE_W unset")

    print("\nRR2 IDENTITY")
    same(A1, run_variant(a, "post_k01_id", post, K01, "layer4.1.bn2.weight:1"), "RR2 k01  layer4.1.bn2.weight:1")
    same(H0, run_variant(a, "post_head_id", post, HEAD, "layer4.1.bn1.weight:1"), "RR2 HEAD layer4.1.bn1.weight:1")

    print("\nRR3 THE SWITCH BITES ON THE REAL PATH")
    for arm, spec, ref in (("MUTE", K01, A1), ("DOSE", K01, A1), ("INJECT", HEAD, H0)):
        R = run_variant(a, "post_" + arm.lower(), post, spec, VOTEW[arm])
        if R is None:
            continue
        chk(WITNESS[arm] in R["stdout"], "RR3 %-6s printed the registered witness line" % arm)
        recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
        refr = [json.loads(x) for x in ref["probe_bytes"].decode().splitlines()] if ref else []
        W = WEIGHTS[arm]
        groups = [list(range(53))] if spec == K01 else [[i for i in range(53) if i != 49], [49]]
        bad = 0
        worst = 0.0
        for r in recs:
            zt, za = r["z_tensor"], r["z_agg"][0]
            sz = sum(abs(v) for v in zt) + 1e-30
            d = max(abs(sum(W.get(i, 1.0) * zt[i] for i in g) - q) for g, q in zip(groups, za)) / sz
            worst = max(worst, d)
            bad += d > 1e-5
        chk(recs and bad == 0, "RR3 %-6s z_agg == sum_i w_i z_tensor_i at every record" % arm,
            "%d records, worst rel %.2e" % (len(recs), worst))
        chk(any(x["z_agg"] != y["z_agg"] for x, y in zip(recs, refr)),
            "RR3 %-6s z_agg differs from the unpatched run's on >= 1 record" % arm,
            "%d of %d records differ" % (sum(x["z_agg"] != y["z_agg"] for x, y in zip(recs, refr)), len(recs)))

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %r" % FAILED))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
