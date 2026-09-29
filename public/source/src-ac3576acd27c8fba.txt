#!/usr/bin/env python3
"""crd1_route_cpu_check.py <harness cifar10 dir> <crd1_design.py> -- CORRECTIONS 307: DOES EACH ARM's OWN ARGS + ITS OWN
DECAY_ROUTE VALUE REACH THE PATCHED UPDATE IT REGISTERS?

The harness-null for crd1's varying knobs (DECAY_ROUTE, --weight-decay-base, the grain), on the login node's CPU (no GPU,
no data, no Slurm job).  PATCH_DECAYROUTE itself was proved at 305 (410 / 0); this check proves the WIRING of THIS
batch's six arms on THIS cell (ResNet18_c100), which 305 did not run:
  R0  the tree's HF.py / train.py / build_optimizer.py / build_network.py are the registered bytes; HF.py carries
      PATCH_DECAYROUTE and none of the hold / mask patches.
  R1  per arm: with DECAY_ROUTE=<the arm's value> and the arm's OWN registered ARGS (crd1_design.args_pairs) through
      train.py's own parse_args and build_optimizer, construction prints EXACTLY ONE `DECAY_ROUTE:` line and it is the
      arm's registered witness (crd1_design.dr_witness); base_update is `_dr_SGDm_base_update`, meta_update
      `Lion_meta_update`; _dr_mode / _dr_lam / wd / grain / gamma are the registered ones; no hold / mask witness line.
  R2  per arm, K real HF.step()s on the live ResNet18_c100 (a batch of 4 random CIFAR-shaped inputs, CPU): on every
      tensor of every step the new weights AND the new trace are BITWISE the arm's registered formula
          shrink_only  delta = a*(m + wd*w)   w1 = w - delta              h1 = gamma*h - delta
          trace_only   delta = a*m            w1 = w - delta              h1 = gamma*(1 - wd*a)*h - delta
          alpha_indep  delta = a*m            w1 = w - LAMBDA*w - delta    h1 = gamma*(1 - LAMBDA)*h - delta
  R3  NON-VACUITY: the UNPATCHED (OFF) formula  w - a*(m + wd*w),  gamma*(1 - wd*a)*h - a*(m + wd*w)  FAILS on at least
      one tensor-step of every arm (shrink_only: on the trace -- its weights are OFF's by design; trace_only and
      alpha_indep: on the weights), so R2 can tell the routes apart.
  R4  LOUD: DECAY_ROUTE=alpha_indep:3.15e-4 at wd 0.1, and shrink_only at wd 0, each raise `ValueError: PATCH_DECAYROUTE`.
Prints `ROUTE CPU CHECK: ALL PASS` iff every check passes; exit 0 / 1.  READ-ONLY on the tree.
"""
import argparse
import ast
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import types

harn, desp = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
K = 3
spm = importlib.util.spec_from_file_location("crd1_design", desp)
D = importlib.util.module_from_spec(spm)
spm.loader.exec_module(D)
RES = {"PASS": 0, "FAIL": 0}


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    RES["PASS" if cond else "FAIL"] += 1
    return bool(cond)


print("crd1 route CPU check: tree %s  design sha %s" % (harn, D.file_sha(desp)))
print("\nR0  the tree")
for rel, want in (("Optimizers/HF.py", D.HF_SHA), ("train.py", D.TRAIN_SHA),
                  ("Optimizers/build_optimizer.py", D.BUILD_OPT_SHA), ("build_network.py", D.BN_SHA)):
    got = D.file_sha(os.path.join(harn, rel))
    chk(got == want, "R0 tree %s == %s..." % (rel, want[:16]), got[:16])
hsrc = open(os.path.join(harn, "Optimizers", "HF.py")).read()
chk("PATCH_DECAYROUTE" in hsrc and not any(m in hsrc for m in ("PATCH_BETAHOLD", "PATCH_GROUPHOLD", "PATCH_RESTHOLD",
                                                                 "PATCH_VOTEWEIGHT", "PATCH_DECAYMASK")),
    "R0 HF.py carries PATCH_DECAYROUTE and no hold / mask patch")

sys.path.insert(0, harn)
os.chdir(harn)
import torch  # noqa: E402

torch.set_num_threads(2)
from build_network import build_network  # noqa: E402
from Optimizers.build_optimizer import build_optimizer  # noqa: E402

tsrc = open(os.path.join(harn, "train.py")).read()
fn = [n for n in ast.parse(tsrc).body if isinstance(n, ast.FunctionDef) and n.name == "parse_args"]
TR = types.ModuleType("train_parse_args")
TR.argparse = argparse
exec(compile(ast.Module(body=fn, type_ignores=[]), os.path.join(harn, "train.py"), "exec"), TR.__dict__)


class _W:
    def add_scalar(self, *a, **k):
        pass

    def add_histogram(self, *a, **k):
        pass


KEYS = ("VOTE_W", "BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD", "GROUP_HOLD", "REST_HOLD", "DECAY_MASK", "DECAY_ROUTE",
        "PROBE", "PROBE_TENSOR", "PROBE_DIR")


def make(arm, route, argpairs, tmpd):
    for k in KEYS:
        os.environ.pop(k, None)
    os.environ.update({"BETA_CLIP": D.CLIP, "HIER": "none", "SCHED": "none", "AUGMENT": "1", "DECAY_ROUTE": route})
    argv = []
    for f, v in argpairs:
        argv += ["--" + f, v]
    sys.argv = ["train.py"] + argv
    args = TR.parse_args()
    for k in ("normalizer_param_base", "momentum_param_base", "weight_decay_base", "Lion_beta2_base",
              "normalizer_param_meta", "momentum_param_meta", "weight_decay_meta", "Lion_beta2_meta", "meta_stepsize"):
        if getattr(args, k) == -1:
            setattr(args, k, None)
    args.device = torch.device("cpu")
    torch.manual_seed(1234)
    net = build_network(D.NET, "cpu")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt = build_optimizer(net, args, _W())
    return net, opt, buf.getvalue().splitlines()


tmpd = tempfile.mkdtemp(prefix="crd1_rcheck_")
try:
    for arm in D.ARMS:
        mode, lam, wd = D.MODE[arm], D.LAM_OF[arm], D.WD[arm]
        print("\n%s  cell %s  %s  wd %s  DECAY_ROUTE=%s" % (arm, D.CELL_OF[arm], D.SPEC[arm], D.WDTOK[arm], D.ROUTE[arm]))
        net, opt, out = make(arm, D.ROUTE[arm], D.args_pairs(arm, D.SEEDS[0], tmpd), tmpd)
        drl = [ln for ln in out if ln.startswith("DECAY_ROUTE")]
        chk(drl == [D.dr_witness(arm)], "R1 exactly one DECAY_ROUTE line == the registered witness", repr(drl)[:220])
        other = [ln for ln in out if ln.startswith(D.NO_LINE_KINDS)]
        chk(not other, "R1 no hold / mask witness line (the cdr1 tree has none of those patches)", repr(other)[:160])
        chk(opt.base_update.__name__ == "_dr_SGDm_base_update" and opt.meta_update.__name__ == "Lion_meta_update"
            and getattr(opt, "_dr_on", False) and opt._dr_mode == mode and opt._dr_lam == lam
            and float(opt.args_base["weight_decay"]) == wd and opt.stepsize_type == D.SPEC[arm]
            and isinstance(opt.gamma, float) and opt.gamma == 1.0
            and int(opt.beta[0].reshape(-1).numel()) == (1 if D.SPEC[arm] == "scalar" else D.NTENS),
            "R1 base %s meta %s mode %s lambda %r wd %r grain %s gamma %r"
            % (opt.base_update.__name__, opt.meta_update.__name__, getattr(opt, "_dr_mode", None),
               getattr(opt, "_dr_lam", None), opt.args_base["weight_decay"], opt.stepsize_type, opt.gamma))
        orig = opt.base_update
        log = {"ok_h": 0, "bad_h": 0, "ok_w": 0, "bad_w": 0, "off_w_differs": 0, "off_h_differs": 0, "steps": 0}

        def wrapped(net_, g_, _orig=orig, _opt=opt, _log=log, _mode=mode, _lam=lam, _wd=wd):
            ws = [p.data.clone() for p in net_.parameters()]
            ms = [m.clone() if torch.is_tensor(m) else m for m in _opt.momentum_base]
            hs = [h.clone() if torch.is_tensor(h) else h for h in _opt.h_condenced]
            al = list(_opt.alpha)
            gm = _opt.gamma
            _orig(net_, g_)
            _log["steps"] += 1
            for i, p in enumerate(net_.parameters()):
                a = al[i]
                if _mode == "shrink_only":
                    delta = a * (ms[i] + _wd * ws[i])
                    w_want = ws[i] - delta
                    h_want = gm * hs[i] - delta
                elif _mode == "trace_only":
                    delta = a * ms[i]
                    w_want = ws[i] - delta
                    h_want = gm * (1 - _wd * a) * hs[i] - delta
                else:
                    delta = a * ms[i]
                    w_want = ws[i] - _lam * ws[i] - delta
                    h_want = gm * (1 - _lam) * hs[i] - delta
                d_off = a * (ms[i] + _wd * ws[i])
                w_off = ws[i] - d_off
                h_off = gm * (1 - _wd * a) * hs[i] - d_off
                _log["ok_h" if torch.equal(_opt.h_condenced[i], h_want) else "bad_h"] += 1
                _log["ok_w" if torch.equal(p.data, w_want) else "bad_w"] += 1
                _log["off_w_differs"] += not torch.equal(p.data, w_off)
                if _log["steps"] >= 2:
                    _log["off_h_differs"] += not torch.equal(_opt.h_condenced[i], h_off)

        opt.base_update = wrapped
        crit = torch.nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(99)
        for step in range(K):
            x = torch.randn(4, 3, 32, 32, generator=gen)
            y = torch.randint(0, 100, (4,), generator=gen)
            loss = crit(net(x), y)
            with contextlib.redirect_stdout(io.StringIO()):
                opt.step(net, loss)
        nt = D.NTENS * K
        chk(log["steps"] == K and log["ok_w"] == nt and log["bad_w"] == 0,
            "R2 weights BITWISE the %s formula on %d/%d tensor-steps" % (mode, log["ok_w"], nt))
        chk(log["ok_h"] == nt and log["bad_h"] == 0,
            "R2 trace BITWISE the %s formula on %d/%d tensor-steps" % (mode, log["ok_h"], nt))
        if mode == "shrink_only":
            chk(log["off_w_differs"] == 0 and log["off_h_differs"] > 0,
                "R3 NON-VACUITY: weights == OFF's on every tensor-step (by design), trace != OFF's on %d tensor-steps"
                % log["off_h_differs"])
        else:
            chk(log["off_w_differs"] > 0, "R3 NON-VACUITY: OFF's weight formula FAILS on %d tensor-steps" % log["off_w_differs"])

    print("\nR4  LOUD on this cell's wrong combinations")
    for lab, route, wdtok in (("alpha_indep at wd 0.1", "alpha_indep:" + D.LAMBDA_TOKEN, "0.1"), ("shrink_only at wd 0", "shrink_only", "0")):
        pairs = [(f, (wdtok if f == "weight-decay-base" else v)) for f, v in D.args_pairs("SRS", D.SEEDS[0], tmpd)]
        try:
            make("SRS", route, pairs, tmpd)
            chk(False, "R4 %s raises" % lab, "constructed silently")
        except ValueError as ex:
            chk("PATCH_DECAYROUTE" in str(ex), "R4 %s raises ValueError: PATCH_DECAYROUTE" % lab, str(ex)[:120])
finally:
    os.environ.pop("DECAY_ROUTE", None)
    shutil.rmtree(tmpd, ignore_errors=True)

print("\ncrd1 route CPU check: %d PASS / %d FAIL" % (RES["PASS"], RES["FAIL"]))
print("ROUTE CPU CHECK: %s" % ("ALL PASS" if RES["FAIL"] == 0 else "FAILED"))
raise SystemExit(0 if RES["FAIL"] == 0 else 1)
