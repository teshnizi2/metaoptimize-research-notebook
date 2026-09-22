#!/usr/bin/env python3
"""csh1_gamma_cpu_check.py <harness cifar10 dir> <csh1_design.py> -- CORRECTIONS 301: DOES --gamma REACH THE TRACE?

The harness-null for csh1's ONE varying optimiser flag, on the login node's CPU (no GPU, no data, no Slurm job):
  G0  the source: in HF.py every read of `self.gamma` sits in __init__ (the assignment) or in a *_base_update trace line
      of the form  h_condenced[i] = self.gamma * ... ; train.py declares `--gamma` as type=float default 1;
      build_optimizer passes `gamma=args.gamma` to HF.
  G1  per arm: the arm's OWN registered ARGS (csh1_design.args_pairs), through train.py's own parse_args and
      build_optimizer, give an HF whose .gamma == float(the arm's token), base_update SGDm_base_update, meta_update
      Lion_meta_update, weight_decay 5e-4, the registered grain.
  G2  per arm, K steps of the REAL HF.step on the live ResNet18_c100 (batch of 4 random CIFAR-shaped inputs, CPU): on
      every tensor of every step the new trace is BITWISE  gamma*(1 - wd*a)*h_old - a*(m_old + wd*w_old)  with the
      arm's gamma, and the new weights are BITWISE  w_old - a*(m_old + wd*w_old).
  G3  NON-VACUITY: from step 2 on (h != 0) the same formula with the OTHER gamma (1 for a gamma < 1 arm, 0.99941 for the
      gamma-1 arm) FAILS on at least one tensor -- so G2 can tell the gammas apart.
Prints `GAMMA CPU CHECK: ALL PASS` iff every check passes; exit 0 / 1.  READ-ONLY on the tree; writes nothing but a
temp dir it removes.
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
spm = importlib.util.spec_from_file_location("csh1_design", desp)
D = importlib.util.module_from_spec(spm)
spm.loader.exec_module(D)
RES = {"PASS": 0, "FAIL": 0}


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    RES["PASS" if cond else "FAIL"] += 1
    return bool(cond)


print("csh1 gamma CPU check: tree %s  design sha %s" % (harn, D.file_sha(desp)))
for rel, want in (("Optimizers/HF.py", D.HF_SHA), ("train.py", D.TRAIN_SHA),
                  ("Optimizers/build_optimizer.py", D.BUILD_OPT_SHA), ("build_network.py", D.BN_SHA)):
    got = D.file_sha(os.path.join(harn, rel))
    chk(got == want, "tree %s == %s..." % (rel, want[:16]), got[:16])

print("\nG0  the source")
hsrc = open(os.path.join(harn, "Optimizers", "HF.py")).read()
tree = ast.parse(hsrc)
reads = []
for fn in ast.walk(tree):
    if isinstance(fn, ast.FunctionDef):
        for n in ast.walk(fn):
            if isinstance(n, ast.Attribute) and n.attr == "gamma" and isinstance(n.value, ast.Name) and n.value.id == "self":
                reads.append((fn.name, n.lineno, isinstance(n.ctx, ast.Store)))
lines = hsrc.splitlines()
bad = [r for r in reads if not ((r[0] == "__init__" and r[2]) or
                                 (r[0].endswith("_base_update") and not r[2]
                                  and lines[r[1] - 1].strip().startswith("self.h_condenced[i] = self.gamma*")))]
chk(not bad and any(r[0] == "SGDm_base_update" for r in reads),
    "G0 HF.py: every self.gamma is the __init__ store or a *_base_update trace line; SGDm_base_update reads it "
    "(%d sites: %s)" % (len(reads), ", ".join("%s:%d" % (r[0], r[1]) for r in reads)), str(bad))
sgdm = [lines[r[1] - 1].strip() for r in reads if r[0] == "SGDm_base_update"]
chk(sgdm == ["self.h_condenced[i] = self.gamma*(1-self.args_base['weight_decay']*a)*self.h_condenced[i] - delta"],
    "G0 SGDm_base_update's trace line is  h <- gamma*(1 - wd*a)*h - delta", str(sgdm))
tsrc = open(os.path.join(harn, "train.py")).read()
chk('parser.add_argument("--gamma", type=float, default=1, help="decay factor")' in tsrc,
    "G0 train.py declares --gamma type=float default=1")
bsrc = open(os.path.join(harn, "Optimizers", "build_optimizer.py")).read()
chk("gamma=args.gamma" in bsrc, "G0 build_optimizer passes gamma=args.gamma to HF")

sys.path.insert(0, harn)
os.chdir(harn)
import torch  # noqa: E402

torch.set_num_threads(2)
from build_network import build_network  # noqa: E402
from Optimizers.build_optimizer import build_optimizer  # noqa: E402

fn = [n for n in ast.parse(tsrc).body if isinstance(n, ast.FunctionDef) and n.name == "parse_args"]
TR = types.ModuleType("train_parse_args")
TR.argparse = argparse
exec(compile(ast.Module(body=fn, type_ignores=[]), os.path.join(harn, "train.py"), "exec"), TR.__dict__)


class _W:
    def add_scalar(self, *a, **k):
        pass

    def add_histogram(self, *a, **k):
        pass


KEYS = ("VOTE_W", "BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD", "GROUP_HOLD", "REST_HOLD", "DECAY_MASK", "PROBE",
        "PROBE_TENSOR", "PROBE_DIR")
tmpd = tempfile.mkdtemp(prefix="csh1_gcheck_")
try:
    for arm in D.ARMS:
        print("\n%s  cell %s  %s  gamma token %s" % (arm, D.CELL_OF[arm], D.SPEC[arm], D.GTOK[arm]))
        for k in KEYS:
            os.environ.pop(k, None)
        os.environ.update({"BETA_CLIP": D.CLIP, "HIER": "none", "SCHED": "none", "AUGMENT": "1"})
        argv = []
        for f, v in D.args_pairs(arm, D.SEEDS[0], tmpd):
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
        with contextlib.redirect_stdout(io.StringIO()):
            opt = build_optimizer(net, args, _W())
        g = D.GF[arm]
        chk(isinstance(args.gamma, float) and args.gamma == g and opt.gamma == g,
            "G1 parse_args -> args.gamma %r (float) -> HF.gamma %r == float(%s)" % (args.gamma, opt.gamma, D.GTOK[arm]))
        chk(opt.base_update.__name__ == "SGDm_base_update" and opt.meta_update.__name__ == "Lion_meta_update"
            and float(opt.args_base["weight_decay"]) == D.WD and opt.stepsize_type == D.SPEC[arm],
            "G1 base SGDm_base_update, meta Lion_meta_update, wd %r, grain %s" % (opt.args_base["weight_decay"], opt.stepsize_type))
        other = 1.0 if g != 1.0 else D.GAMMA["P"]
        orig = opt.base_update
        log = {"ok_h": 0, "bad_h": 0, "ok_w": 0, "bad_w": 0, "other_differs": 0, "steps": 0}

        def wrapped(net_, g_, _orig=orig, _opt=opt, _log=log, _g=g, _other=other):
            wd = _opt.args_base["weight_decay"]
            ws = [p.data.clone() for p in net_.parameters()]
            ms = [m.clone() if torch.is_tensor(m) else m for m in _opt.momentum_base]
            hs = [h.clone() if torch.is_tensor(h) else h for h in _opt.h_condenced]
            al = list(_opt.alpha)
            _orig(net_, g_)
            _log["steps"] += 1
            for i, p in enumerate(net_.parameters()):
                a = al[i]
                delta = a * (ms[i] + wd * ws[i])
                h_want = _g * (1 - wd * a) * hs[i] - delta
                h_oth = _other * (1 - wd * a) * hs[i] - delta
                if torch.equal(_opt.h_condenced[i], h_want):
                    _log["ok_h"] += 1
                else:
                    _log["bad_h"] += 1
                if torch.equal(p.data, ws[i] - delta):
                    _log["ok_w"] += 1
                else:
                    _log["bad_w"] += 1
                if _log["steps"] >= 2 and not torch.equal(_opt.h_condenced[i], h_oth):
                    _log["other_differs"] += 1

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
        chk(log["steps"] == K and log["ok_h"] == nt and log["bad_h"] == 0,
            "G2 trace BITWISE gamma*(1-wd*a)*h - a*(m + wd*w) with gamma=%s on %d/%d tensor-steps" % (D.GTOK[arm], log["ok_h"], nt))
        chk(log["ok_w"] == nt and log["bad_w"] == 0, "G2 weights BITWISE w - a*(m + wd*w) on %d/%d tensor-steps" % (log["ok_w"], nt))
        chk(log["other_differs"] > 0,
            "G3 NON-VACUITY: the gamma=%r formula FAILS on %d tensor-steps (steps 2..%d)" % (other, log["other_differs"], K))
finally:
    shutil.rmtree(tmpd, ignore_errors=True)

print("\ncsh1 gamma CPU check: %d PASS / %d FAIL" % (RES["PASS"], RES["FAIL"]))
print("GAMMA CPU CHECK: %s" % ("ALL PASS" if RES["FAIL"] == 0 else "FAILED"))
raise SystemExit(0 if RES["FAIL"] == 0 else 1)
