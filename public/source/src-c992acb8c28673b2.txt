"""INERTNESS AND BITE OF THE cvt7 TREE OVER A SHORT *REAL* ResNet18_c100 RUN, FOR cvt7's EXACT STRINGS.  CORRECTIONS 243.

The REAL code path of a cvt7 run -- train.py's own parse_args, build_network (ResNet18_c100), build_optimizer,
load_data (CIFAR-100, AUGMENT=1), the loss, HF.step with BETA_CLIP, PROBE and PROBE_TENSOR -- for --steps optimizer
steps on a GPU, from TWO trees:

  --tree-pre   the LIVE shared ResNet harness ($WS/MetaOptimize/.../cifar10, HF.py 4732b74a...) -- THE TREE ciso1,
               cdep1 and ciso2 RAN FROM -- used READ-ONLY (PYTHONDONTWRITEBYTECODE=1; nothing is written under it)
  --tree-post  the cvt7 tree (PATCH_VOTEWEIGHT + PATCH_BETAHOLD + PATCH_GROUPHOLD on top of the live HF.py)

So OFF is proved against the ISO batches' own harness, not against an intermediate: the three patches together, all
off, are bitwise the tree whose runs the replays were derived from.

The child is a COPY of tests/test_voteweight_realrun.py's child (227, registered; that file hard-codes
PlainNet18_c100 and is therefore not importable for this network), with the network name a parameter; nothing else in
it differs (a structural check below compares the two child functions' source with the network literal substituted).
The strings, witnesses and the schedule come from analysis/cVT7_grouphold_score.py, so the test and the scorer cannot
disagree.

  RR0  DETERMINISM CONTROL: the live tree run twice (k01) -> beta at every step, loss, probe.jsonl bytes, every final
       parameter and buffer identical (else the bitwise claims cannot be tested; FAIL).
  RR1  OFF: GROUP_HOLD unset and GROUP_HOLD empty, on k01's spec AND on ISO's spec == the live tree BITWISE (beta every
       step, loss every step, probe.jsonl bytes, every final parameter and buffer); each printed exactly one
       `GROUP_HOLD: off`, one `BETA_HOLD: off` and one `VOTE_W: off`.
  RR2  THE EXACT STRINGS ON THE REAL PATH (HOLDLOW, HOLDHIGH, HOLDISO) and a test-only `tri:100` whose peak falls inside
       the run: exactly one GROUP_HOLD line == the scorer's witness (and BETA_HOLD: off, VOTE_W: off); beta[1] after
       EVERY step == the registered hold; at every probe record the scorer's own G-BITE halves hold (group 0 == Lion
       recomputed from the record; beta[1] and beta_pre[1] == the hold; gh_n == step + 2; gh_nat == Lion recomputed for
       group 1; gh_held == beta[1]; gh_active non-decreasing; no bh_* key) -- and the scorer's bite_check itself,
       applied to the 30 records, reports no schedule, Lion or audit failure.
  RR3  NON-VACUITY: HOLDLOW's beta[1] differs from the live ISO's at every step and its final weights differ; tri:100's
       beta[1] tracks the natural rise to its peak and departs after it, with gh_active > 0; HOLDHIGH and HOLDISO each
       differ from the live ISO's beta on >= 1 step with gh_active > 0 (their final weights REPORTED, not gated: a
       step-size difference of ~1e-7 per step can sit below float32 resolution within 300 steps, 237.4).

RUN (on a GPU node; ~10 min):
  python3 tests/test_grouphold_realrun.py --tree-pre $WS/MetaOptimize/codes/Supervised_tasks/MetaOptimize/cifar10 \\
      --tree-post $WS/harness_cvt7/cifar10 --steps 300 --every 10 --seed 99 --work /tmp/cvt7_realrun
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAILED = []


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SC = _load("cvt7_scorer", os.path.join(REPO, "analysis", "cVT7_grouphold_score.py"))
K01 = "scalar"
ISO = SC.ISOSPEC
TRI100 = "+".join(SC.CARRIERS) + ":tri:100"


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
                "--NN-name", a.net, "--batch-size", "100", "--max-time", "999:00:00",
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


def run_variant(a, tag, tree, spec, gh):
    work = os.path.join(a.work, tag)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    env = dict(os.environ)
    for k in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD"):
        env.pop(k, None)
    if gh is not None:
        env["GROUP_HOLD"] = gh
    env.update({"AUGMENT": "1", "BETA_CLIP": "-15:-2.3026", "HIER": "", "SCHED": "",
                "PROBE": str(a.every), "PROBE_DIR": os.path.join(work, "probe"), "PROBE_TENSOR": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
    out = os.path.join(work, "result.json")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", "--tree", tree, "--spec", spec, "--net", a.net,
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
    print("  ran %-16s tree=%s spec=%s GROUP_HOLD=%s  %d steps on %s  weights %s..."
          % (tag, tree.rstrip("/").split("/")[-2], spec[:14], (gh or repr(gh))[-16:], len(res["betas"]),
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


def lines(R, prefix):
    return [ln for ln in (R["stdout"] if R else []) if ln.startswith(prefix)]


def hold_after(mode, P, n):
    return SC.LO if mode == "floor" else SC.v_of(n, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--tree", default="")
    ap.add_argument("--spec", default="")
    ap.add_argument("--net", default=SC.NET)
    ap.add_argument("--save", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tree-pre", default="")
    ap.add_argument("--tree-post", default="")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    if a.child:
        return child(a)
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    os.makedirs(a.work, exist_ok=True)
    for k in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD"):
        os.environ.pop(k, None)
    print("DRIVER_SHA256 %s" % hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest())
    print("SCORER_SHA256 %s" % hashlib.sha256(open(SC.__file__, "rb").read()).hexdigest())
    for t, lab in ((pre, "pre"), (post, "post")):
        hf = open(os.path.join(t, "Optimizers", "HF.py"), "rb").read()
        bn = open(os.path.join(t, "build_network.py"), "rb").read()
        tr = open(os.path.join(t, "train.py"), "rb").read()
        print("TREE_%s %s HF %s build_network %s train %s" % (lab.upper(), t, hashlib.sha256(hf).hexdigest(),
                                                             hashlib.sha256(bn).hexdigest(), hashlib.sha256(tr).hexdigest()))
    hpost = open(os.path.join(post, "Optimizers", "HF.py"), "rb").read()
    hpre = open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read()
    chk(a.net == SC.NET, "the network is the scorer's %s" % SC.NET)
    chk(hashlib.sha256(hpost).hexdigest() == SC.HF_POST_SHA, "--tree-post's HF.py is the scorer's HF_POST_SHA (cvt7's tree)")
    chk(hashlib.sha256(hpre).hexdigest() == SC.HF_LIVE_SHA and b"PATCH_VOTEWEIGHT" not in hpre,
        "--tree-pre's HF.py is the LIVE one (HF_LIVE_SHA 4732b74a..., the ISO batches' recorded HF_SHA256), unpatched")
    vwpre = os.path.join(post, "Optimizers", "HF.py.pre_voteweight")
    chk(os.path.exists(vwpre) and open(vwpre, "rb").read() == hpre,
        "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_voteweight")
    for f in ("build_network.py", "train.py", "load_data.py", "Optimizers/build_optimizer.py"):
        chk(open(os.path.join(pre, f), "rb").read() == open(os.path.join(post, f), "rb").read(),
            "the two trees' %s are BYTE-IDENTICAL" % f)
    vwr = os.path.join(HERE, "test_voteweight_realrun.py")
    if os.path.exists(vwr):
        RRV = _load("test_voteweight_realrun", vwr)
        mine = inspect.getsource(child).replace('"--NN-name", a.net,', '"--NN-name", "PlainNet18_c100",')
        chk(mine == inspect.getsource(RRV.child),
            "this child == tests/test_voteweight_realrun.py's registered child, with ONLY the network literal made a parameter")
    print("\nnet %s, steps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026"
          % (a.net, a.steps, a.every, a.seed))

    print("\nRR0 DETERMINISM CONTROL (the LIVE tree, twice)")
    A1 = run_variant(a, "pre_k01_a", pre, K01, None)
    A2 = run_variant(a, "pre_k01_b", pre, K01, None)
    same(A1, A2, "RR0 live k01 vs itself")
    I0 = run_variant(a, "pre_iso", pre, ISO, None)
    chk(I0 is not None and not lines(I0, "GROUP_HOLD") and not lines(I0, "BETA_HOLD") and not lines(I0, "VOTE_W"),
        "RR0 the LIVE tree prints no GROUP_HOLD / BETA_HOLD / VOTE_W line (it carries none of the three patches)")

    print("\nRR1 OFF == the LIVE tree")
    for spec, ref, lab in ((K01, A1, "k01"), (ISO, I0, "ISO")):
        for val in (None, ""):
            R = run_variant(a, "post_%s_%s" % (lab.lower(), "unset" if val is None else "empty"), post, spec, val)
            same(ref, R, "RR1 %-3s GROUP_HOLD %s" % (lab, "unset" if val is None else "empty"))
            chk(lines(R, "GROUP_HOLD") == ["GROUP_HOLD: off"] and lines(R, "BETA_HOLD") == ["BETA_HOLD: off"]
                and lines(R, "VOTE_W") == ["VOTE_W: off"],
                "RR1 %-3s GROUP_HOLD %s printed exactly `GROUP_HOLD: off`, `BETA_HOLD: off` and `VOTE_W: off`"
                % (lab, "unset" if val is None else "empty"))

    print("\nRR2 THE EXACT STRINGS BITE ON THE REAL PATH")
    got = {}
    for arm in SC.HELD + ("TRI100",):
        gh = SC.GROUPHOLD.get(arm) or TRI100
        mode = gh.split(":", 1)[1].split(":")[0]
        P = int(gh.rsplit(":", 1)[1]) if mode == "tri" else None
        wit = SC.WITNESS_GH.get(arm) or SC.witness_gh_of("tri", 100)
        R = run_variant(a, "post_%s" % arm.lower(), post, ISO, gh)
        if R is None:
            continue
        got[arm] = R
        chk(lines(R, "GROUP_HOLD") == [wit] and lines(R, "BETA_HOLD") == ["BETA_HOLD: off"] and lines(R, "VOTE_W") == ["VOTE_W: off"],
            "RR2 %-9s printed exactly ONE GROUP_HOLD line == the scorer's witness, BETA_HOLD: off and VOTE_W: off" % arm)
        bad_step = sum(1 for n, b in enumerate(R["betas"], 1) if b[1] != hold_after(mode, P, n))
        chk(len(R["betas"]) == a.steps and bad_step == 0,
            "RR2 %-9s beta[1] after EVERY one of %d steps == the registered hold" % (arm, a.steps), "%d bad" % bad_step)
        recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
        bad_lion = bad_sched = bad_gh = bh_keys = 0
        prev = -1
        for r in recs:
            s = r["step"]
            nat0, tie0 = SC.lion_natural(r["beta_pre"][0][0], r["mom_pre"][0][0], r["z_agg"][0][0])
            bad_lion += (not tie0) and abs(nat0 - r["beta"][0]) > SC.BH_TOL
            bad_sched += (abs(r["beta"][1] - hold_after(mode, P, s + 2)) > SC.BH_TOL
                          or abs(r["beta_pre"][0][1] - hold_after(mode, P, s + 1)) > SC.BH_TOL)
            nat1, tie1 = SC.lion_natural(r["beta_pre"][0][1], r["mom_pre"][0][1], r["z_agg"][0][1])
            ok = (r.get("gh_n") == s + 2 and abs(r.get("gh_held", 1e9) - r["beta"][1]) <= SC.BH_TOL
                  and (tie1 or abs(r.get("gh_nat", 1e9) - nat1) <= SC.BH_TOL) and r.get("gh_active", -2) >= prev)
            bad_gh += not ok
            bh_keys += any(k in r for k in SC.BH_KEYS)
            prev = r.get("gh_active", prev)
        chk(len(recs) == a.steps // a.every and bad_lion == 0 and bad_sched == 0 and bad_gh == 0 and bh_keys == 0,
            "RR2 %-9s at every probe record: complement == Lion; beta/beta_pre[1] == hold; gh_n / gh_nat / gh_held / gh_active audit; no bh_* key"
            % arm, "%d records; lion %d sched %d gh %d bh-keys %d; last gh_active %s" % (len(recs), bad_lion, bad_sched, bad_gh, bh_keys, prev))
        if arm in SC.HELD:
            _ok, d = SC.bite_check(arm, recs)
            chk(d["bad_rec"] == 0 and d["bad_lion"] == 0 and d["bad_sched"] == 0
                and d["bad_sched_pre"] == 0 and d["bad_gh"] == 0 and d["bh_any"] == 0 and d["n"] == len(recs),
                "RR2 %-9s the scorer's own bite_check on the %d real records: no record, Lion, schedule, audit or bh_* failure"
                % (arm, len(recs)), "lion %d ties %d sched %d/%d gh %d bh %d active records %d gh_active %s"
                % (d["bad_lion"], d["ties"], d["bad_sched"], d["bad_sched_pre"], d["bad_gh"], d["bh_any"],
                   d["active_records"], d["gh_active_last"]))
            print("  NOTE RR2 %-9s bite_check's record-count half (500) and step half (step == 100 k) do not apply to a "
                  "%d-record PROBE=%d run: bad_step %d of %d (expected %d)" % (arm, len(recs), a.every, d["bad_step"],
                                                                              len(recs), len(recs) - 1))

    print("\nRR3 NON-VACUITY")
    L = got.get("HOLDLOW")
    if L and I0:
        chk(all(x[1] != y[1] for x, y in zip(L["betas"], I0["betas"])),
            "RR3 HOLDLOW beta[1] differs from the live ISO's at EVERY step")
        chk(L["weights_sha256"] != I0["weights_sha256"], "RR3 HOLDLOW's final parameters differ from the live ISO's",
            "%s vs %s" % (L["weights_sha256"][:12], I0["weights_sha256"][:12]))
    T = got.get("TRI100")
    if T and I0:
        pre_pk = max(abs(x[1] - y[1]) for x, y in zip(T["betas"][:101], I0["betas"][:101]))
        post_pk = min(abs(x[1] - y[1]) for x, y in zip(T["betas"][150:], I0["betas"][150:]))
        last = json.loads(T["probe_bytes"].decode().splitlines()[-1])
        chk(pre_pk <= 5e-3 and post_pk > 0.05 and last.get("gh_active", 0) > 0,
            "RR3 tri:100 tracks the natural rise to its peak (max |diff| %.1e), then departs "
            "(min |diff| from step 150 %.3f), gh_active %s > 0" % (pre_pk, post_pk, last.get("gh_active")))
        print("  NOTE RR3 tri:100 final weights %s the live ISO's (non-gating)"
              % ("differ from" if T["weights_sha256"] != I0["weights_sha256"] else "are identical to"))
    for arm in ("HOLDHIGH", "HOLDISO"):
        R = got.get(arm)
        if R and I0:
            nd = sum(1 for x, y in zip(R["betas"], I0["betas"]) if x != y)
            last = json.loads(R["probe_bytes"].decode().splitlines()[-1])
            chk(nd >= 1 and last.get("gh_active", 0) > 0,
                "RR3 %-9s beta differs from the live ISO's on >= 1 step and gh_active > 0" % arm,
                "%d of %d steps differ; gh_active %s" % (nd, a.steps, last.get("gh_active")))
            print("  NOTE RR3 %-9s final weights %s the live ISO's (non-gating)"
                  % (arm, "differ from" if R["weights_sha256"] != I0["weights_sha256"] else "are identical to"))

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %r" % FAILED))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
