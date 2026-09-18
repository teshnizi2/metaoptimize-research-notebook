#!/usr/bin/env python3
"""tests/test_cmo1_flags_bite.py -- cmo1 (CORRECTIONS 255): the two CLI factors reach the base update of the LIVE harness.

cmo1 changes no code: its factors are `--momentum-param-base 0.9` and `--weight-decay-base 0`.  This test proves, on the
LIVE files the batch runs (read-only; nothing is written into the tree), CPU only, seconds:

  F0  the live train.py / Optimizers/HF.py / Optimizers/build_optimizer.py are the pinned files (sha256)
  F1  train.py's parser turns each arm's two flags into args.momentum_param_base / args.weight_decay_base = the registered
      floats (0.99 / 0.9 and 0.1 / 0.0), with no -1 -> None surprise (0 is NOT -1)
  F2  build_optimizer() hands them to HF as args_base['momentum_param'] / args_base['weight_decay'] (HF constructed for real on
      a tiny CPU net, scalar step size)
  F3  HF.SGDm_base_update applies them: w, momentum_base and the meta trace h after two updates equal a hand computation
      m <- mu m + (1-mu) g ; delta = a (m_prev + wd w) ; w <- w - delta ; h <- gamma (1 - wd a) h - delta   (to 1e-12, float64)
  F4  BITE: the three configurations give three different (w, h) after two updates on identical inputs
      (the BROKEN-FLAG null -- a flag that does not reach the update -- would make M9 or W0 equal the anchor)

Usage (login node is fine: CPU, a 3x2 linear layer):
    python3 tests/test_cmo1_flags_bite.py --live $METAOPT_WS/MetaOptimize/codes/Supervised_tasks/MetaOptimize/cifar10
Exit 0 and a final `ALL PASS` line, or 1.
"""
import argparse
import hashlib
import os
import sys

TRAIN_SHA = "3fea309e172609c1ee1be143d5f1847404426df31869a63c4cefbb1ebddfcab7"
HF_SHA = "4732b74aa3a10508896e92eccced5c89051c353fa8a01a3af21aab9aeda0cecd"
BO_SHA = "25a899b3745e9d66fbf630795a1202c6075e2067daffa077e4a54c28e54c7ec2"
CONFIGS = {"anchor": ("0.99", "0.1"), "M9": ("0.9", "0.1"), "W0": ("0.99", "0")}
FAILS = []


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    if not cond:
        FAILS.append(label)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    a = ap.parse_args()
    live = os.path.abspath(a.live)
    tp, hp, bp = (os.path.join(live, x) for x in ("train.py", "Optimizers/HF.py", "Optimizers/build_optimizer.py"))
    print("test_cmo1_flags_bite on %s" % live)
    chk(sha(tp) == TRAIN_SHA, "F0 train.py is the pinned live file %s..." % TRAIN_SHA[:12], sha(tp)[:12])
    chk(sha(hp) == HF_SHA, "F0 Optimizers/HF.py is the pinned live file %s..." % HF_SHA[:12], sha(hp)[:12])
    chk(sha(bp) == BO_SHA, "F0 Optimizers/build_optimizer.py is the pinned live file %s..." % BO_SHA[:12], sha(bp)[:12])
    for k in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "COMP_HOLD", "REST_HOLD", "WINDOW_HOLD", "DECAY_MASK", "SHADOW_VOTE", "PROBE", "PROBE_TENSOR"):
        os.environ.pop(k, None)

    sys.path.insert(0, live)
    os.chdir(live)
    import torch
    import torch.nn as nn
    torch.set_default_dtype(torch.float64)

    # train.py cannot be imported (a module-level `writer.close()` sits outside its __main__ guard), so its OWN parse_args
    # function is lifted out of the file by ast, unedited, and executed with argparse
    import ast
    import types
    tree = ast.parse(open(tp).read())
    fdefs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args"]
    chk(len(fdefs) == 1, "F1 train.py defines exactly one top-level parse_args()")
    train = types.SimpleNamespace()
    ns = {"argparse": __import__("argparse")}
    exec(compile(ast.Module(body=fdefs, type_ignores=[]), tp, "exec"), ns)
    train.parse_args = ns["parse_args"]
    from Optimizers.build_optimizer import build_optimizer
    from Optimizers.HF import HF

    class W:
        def add_text(self, *x, **k):
            pass

        def add_scalar(self, *x, **k):
            pass

    finals = {}
    for name, (mu, wd) in CONFIGS.items():
        argv = ["train.py", "--optimizer", "HF", "--alg-base", "SGDm", "--momentum-param-base", mu, "--weight-decay-base", wd,
                "--alg-meta", "Lion", "--momentum-param-meta", "0.99", "--Lion-beta2-meta", "0.9", "--weight-decay-meta", "0",
                "--gamma", "1", "--meta-stepsize", "1e-3", "--alpha0", "1e-6", "--stepsize-groups", "scalar"]
        old = sys.argv
        sys.argv = argv
        try:
            args = train.parse_args()
        finally:
            sys.argv = old
        for attr in ("normalizer_param_base", "momentum_param_base", "weight_decay_base", "Lion_beta2_base"):
            if getattr(args, attr) == -1:
                setattr(args, attr, None)
        chk(args.momentum_param_base == float(mu) and args.weight_decay_base == float(wd),
            "F1 %-6s parse: momentum_param_base %r weight_decay_base %r" % (name, args.momentum_param_base, args.weight_decay_base))
        torch.manual_seed(0)
        net = nn.Linear(3, 2)
        opt = build_optimizer(net, args, W())
        chk(isinstance(opt, HF) and opt.args_base.get("alg") == "SGDm" and opt.args_base.get("momentum_param") == float(mu)
            and opt.args_base.get("weight_decay") == float(wd),
            "F2 %-6s build_optimizer -> HF.args_base momentum_param %r weight_decay %r"
            % (name, opt.args_base.get("momentum_param"), opt.args_base.get("weight_decay")))
        chk(opt.base_update.__func__ is HF.SGDm_base_update, "F2 %-6s HF.base_update is SGDm_base_update" % name)
        # F3: two base updates on fixed gradients, against a hand computation
        a0 = 0.05
        opt.alpha = [torch.tensor(a0) for _ in net.parameters()]
        opt.momentum_base = [torch.zeros_like(p.data) for p in net.parameters()]
        opt.h_condenced = [torch.zeros_like(p.data) for p in net.parameters()]
        opt.gamma = 1.0
        w0 = [p.data.clone() for p in net.parameters()]
        g = torch.Generator().manual_seed(7)
        grads = [[torch.randn(p.shape, generator=g) for p in net.parameters()] for _ in range(2)]
        for gr in grads:
            opt.SGDm_base_update(net, gr)
        f_mu, f_wd = float(mu), float(wd)
        ok = True
        for j, w_init in enumerate(w0):
            w, m, h = w_init.clone(), torch.zeros_like(w_init), torch.zeros_like(w_init)
            for gr in grads:
                delta = a0 * (m + f_wd * w)
                w = w - delta
                m = f_mu * m + (1 - f_mu) * gr[j]
                h = 1.0 * (1 - f_wd * a0) * h - delta
            p = list(net.parameters())[j]
            ok = ok and torch.allclose(p.data, w, atol=1e-12, rtol=0) and torch.allclose(opt.momentum_base[j], m, atol=1e-12, rtol=0) \
                and torch.allclose(opt.h_condenced[j], h, atol=1e-12, rtol=0)
        chk(ok, "F3 %-6s SGDm_base_update == m <- mu m + (1-mu) g; delta = a(m_prev + wd w); h <- gamma(1-wd a)h - delta (2 updates)" % name)
        finals[name] = torch.cat([p.data.reshape(-1) for p in net.parameters()] + [x.reshape(-1) for x in opt.h_condenced])
    chk(not torch.equal(finals["anchor"], finals["M9"]), "F4 BITE: momentum 0.9 changes (w, h) vs the anchor",
        "max|d| %.3e" % float((finals["anchor"] - finals["M9"]).abs().max()))
    chk(not torch.equal(finals["anchor"], finals["W0"]), "F4 BITE: weight decay 0 changes (w, h) vs the anchor",
        "max|d| %.3e" % float((finals["anchor"] - finals["W0"]).abs().max()))
    print("ALL PASS" if not FAILS else "FAILURES %d" % len(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
