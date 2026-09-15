"""Correctness AND BACKWARD-COMPATIBILITY tests for PATCH_REDNORM
(`--stepsize-groups tn:<spec>`).

STRATEGY, copied from tests/test_namesets.py.  The patch's whole claim is that
it is ADDITIVE and OPT-IN.  So the test does not compare the patched code
against my expectation of it; it compares the patched code against the
UNPATCHED code, on the specs the corpus itself has actually used, through the
two functions that actually consume a spec: `init_meta` and `block_product`.

  R0  the two HF.py files differ ONLY by the patch: `PATCH_REDNORM` is absent
      from the pre file and present in the post file, and deleting the THREE
      inserted regions from the post file reproduces the pre file BYTE FOR BYTE.
  R1  BACKWARD COMPATIBILITY THROUGH init_meta.  For every spec drawn from the
      corpus's own `granularity` column, a fully constructed HF has identical
      `stepsize_type`, `len_beta_list`, beta shapes AND VALUES,
      `param_groups_indices`, `map_layers_to_blocks` and `param_numels` before
      and after the patch -- or raises the identical exception type on both.
  R2  BACKWARD COMPATIBILITY THROUGH block_product, which is the function the
      patch actually edits.  For every corpus spec, the SAME (u, v) fed to the
      pre and post objects returns BITWISE-IDENTICAL reductions.
  R3  `tn:X` IS `X`, structurally.  For every spec, `tn:X` and `X` give the same
      `stepsize_type`, the same groups, the same beta.  The ONLY difference
      anywhere in the object is `_rednorm`.
  R4  THE INTERVENTION IS LIVE, AND IS WHAT IT SAYS.  For `scalar` and for
      multi-tensor `blockwise` specs, `tn:X`'s reduction (a) DIFFERS from `X`'s
      and (b) equals, to float tolerance, the independently computed
      sum_i <u_i,v_i>/numel_i over each group.
  R5  THE NO-OP AT SINGLE-TENSOR GRANULARITIES, asserted rather than argued:
      for layerwise / nodewise / weightwise / nodewise1d / chunk<K> / permnode<S>
      the patch touches no branch, so `tn:X` and `X` return BITWISE-IDENTICAL
      reductions and the two runs are identical by construction.
  R6  RULE 20 SAFETY: every `tn:` spec is a single shell token, contains no
      whitespace, no quote, no `=`, does not begin with `-`, and round-trips
      through analysis/argsline_guard.py's own tokenizer as ONE value.
  R7  THE GRAMMAR IS LOUD: `tn:` alone and `tn:<garbage>` raise, they do not
      silently fall through to a default partition.

Any failure means an existing run is not reproducible, or the intervention is
not the intervention it is registered as, and every number from a `tn:` batch is
void.

RUN (needs torch; run it on the cluster against the real tree):
  python3 tests/test_rednorm.py --pre  /path/to/HF.py.pre_rednorm \\
                                --post /path/to/Optimizers/HF.py \\
                                [--cifar-dir /path/to/cifar10] \\
                                [--argsline-guard analysis/argsline_guard.py]
"""
import argparse, importlib.util, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- the specs the CORPUS has actually used, plus the three this batch adds.
CORPUS_SPECS = [
    "scalar", "layerwise", "nodewise", "weightwise", "nodewise1d",
    "resnet18_blocks", "chunk1024", "chunk65536", "permnode0", "permnode101",
    "[2,60]", "[16,16,15,15]", "[17,45]", "[24,38]", "[31,31]", "[38,24]",
    "[42,20]", "[45,17]", "[46,16]", "[47,15]", "[48,14]", "[49,13]", "[50,12]",
    "[51,11]", "[52,10]", "[53,9]", "[54,8]", "[55,7]", "[60,2]",
    "[8,8,8,8,8,8,7,7]", "[4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3]",
    "sets:1-49/50-62",
    "sets:1-48,layer4.0.bn2.weight/layer4.0.conv2.weight,51-62",
    "sets:1-48,layer4.0.shortcut.0.weight/layer4.0.conv2.weight,50-51,53-62",
]

# specs whose groups all hold EXACTLY ONE tensor -> the patch cannot bite (R5)
SINGLE_TENSOR = ["layerwise", "nodewise", "weightwise", "nodewise1d",
                 "chunk1024", "chunk65536", "permnode0", "permnode101"]

# specs where a group holds >= 2 tensors -> the patch MUST bite (R4)
MULTI_TENSOR = ["scalar", "[49,13]", "[50,12]", "resnet18_blocks",
                "sets:1-49/50-62"]

# the batch's own arms, written once here and once in the launcher
BATCH_SPECS = ["scalar", "sets:1-49/50-62", "tn:sets:1-49/50-62",
               "sets:1-50/51-62", "tn:sets:1-50/51-62"]

FAILED = []


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label,
                           ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)


def load_hf(path, tag):
    spec = importlib.util.spec_from_file_location("hf_" + tag, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hf_" + tag] = mod
    spec.loader.exec_module(mod)
    return mod


ABASE = {"alg": "SGDm", "momentum_param": 0.99, "weight_decay": 0.1}
AMETA = {"alg": "Lion", "momentum_param": 0.99, "Lion_beta2": 0.9,
         "weight_decay": 0.0, "meta_stepsize": 1e-3}


def build(mod, spec, net):
    """Return ('ok', HF) or ('exc', ExceptionClassName)."""
    try:
        return ("ok", mod.HF(net, stepsize_groups=spec, alpha0=1e-6,
                             args_base=dict(ABASE), args_meta=dict(AMETA),
                             gamma=1, writer=None))
    except Exception as e:               # noqa: BLE001 -- the comparison IS the test
        return ("exc", type(e).__name__)


def fingerprint(o):
    """Everything about a built HF that a run depends on, EXCEPT _rednorm."""
    import torch
    b = [(tuple(x.shape), float(x.reshape(-1)[0]), float(x.sum())) for x in o.beta]
    return (o.stepsize_type, o.len_beta_list, b,
            getattr(o, "param_groups_indices", None),
            getattr(o, "map_layers_to_blocks", None),
            list(o.param_numels),
            getattr(o, "num_blocks", None),
            getattr(o, "chunk_size", None), getattr(o, "perm_seed", None))


def reduce_with(o, u, v):
    r = o.block_product(u, v)
    return [x.detach().double().cpu().clone() for x in r]


def same_red(a, b):
    import torch
    if len(a) != len(b):
        return False
    return all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="UNPATCHED HF.py")
    ap.add_argument("--post", required=True, help="PATCHED HF.py")
    ap.add_argument("--cifar-dir", default=os.environ.get("CIFAR10_DIR", ""))
    ap.add_argument("--net", default="ResNet18_c100")
    ap.add_argument("--argsline-guard", default="")
    a = ap.parse_args()

    import torch

    pre_src, post_src = open(a.pre).read(), open(a.post).read()

    print("=" * 78)
    print("R0  the two files differ ONLY by PATCH_REDNORM")
    chk("PATCH_REDNORM" not in pre_src, "the --pre file is UNPATCHED")
    chk("PATCH_REDNORM" in post_src, "the --post file IS patched")
    ins1 = ("        # --- PATCH_REDNORM: `tn:<spec>` = <spec>, with the per-group reduction\n"
            "        # weighted 1/numel PER TENSOR.  The prefix is stripped HERE, before any\n"
            "        # routing test runs, so `tn:X` routes and parses EXACTLY as `X` does.\n"
            "        # Every pre-existing spec fails the test and gets _rednorm = False,\n"
            "        # which is the state every prior run was in implicitly. ---\n"
            "        self._rednorm = bool(isinstance(stepsize_groups, str)\n"
            "                             and stepsize_groups.startswith('tn:'))\n"
            "        if self._rednorm:\n"
            "            stepsize_groups = stepsize_groups[3:]\n"
            "        # --- end PATCH_REDNORM ---\n")
    ins2 = ("            # --- PATCH_REDNORM ---\n"
            "            if getattr(self, '_rednorm', False):\n"
            "                return [sum([(u_*v_).sum()/_n for u_,v_,_n\n"
            "                             in zip(u,v,self.param_numels)])]\n"
            "            # --- end PATCH_REDNORM ---\n")
    ins3 = ("            # --- PATCH_REDNORM ---\n"
            "            if getattr(self, '_rednorm', False):\n"
            "                return [torch.tensor([sum([(u[i]*v[i]).sum()/self.param_numels[i]"
            " for i in group_indices]) for group_indices in self.param_groups_indices])]\n"
            "            # --- end PATCH_REDNORM ---\n")
    stripped = post_src
    for k, ins in enumerate((ins1, ins2, ins3), 1):
        chk(stripped.count(ins) == 1, "inserted region %d appears exactly once" % k,
            "count=%d" % stripped.count(ins))
        stripped = stripped.replace(ins, "", 1)
    chk(stripped == pre_src,
        "deleting the three inserted regions reproduces --pre BYTE FOR BYTE",
        "" if stripped == pre_src else "(%d vs %d bytes)" % (len(stripped), len(pre_src)))
    chk(post_src.count("PATCH_REDNORM") == 6,
        "PATCH_REDNORM appears exactly 6 times in the patched file "
        "(1 opening comment + 1 closer in init_meta, and one of each in the two "
        "block_product branches)",
        "%d" % post_src.count("PATCH_REDNORM"))

    pre, post = load_hf(a.pre, "pre"), load_hf(a.post, "post")

    cif = a.cifar_dir or os.getcwd()
    sys.path.insert(0, cif)
    cwd = os.getcwd()
    os.chdir(cif)
    from build_network import build_network
    net = build_network(a.net, "cpu")
    os.chdir(cwd)
    names = [n for n, _ in net.named_parameters()]
    numel = [int(p.numel()) for _, p in net.named_parameters()]
    T = len(names)
    print("\nlive %s: %d parameter tensors, %d parameters" % (a.net, T, sum(numel)))

    # one fixed (u, v) pair, reused for every spec so the comparison is exact.
    torch.manual_seed(20260908)
    U = [torch.randn(p.shape) * 1e-4 for p in net.parameters()]
    V = [torch.randn(p.shape) * 1e-3 for p in net.parameters()]

    print("\nR1  BACKWARD COMPATIBILITY through init_meta over %d corpus specs"
          % len(CORPUS_SPECS))
    same = diff = built = 0
    for s in CORPUS_SPECS:
        rp, rq = build(pre, s, net), build(post, s, net)
        if rp[0] != rq[0]:
            diff += 1
            print("  FAIL %-24s pre=%r post=%r" % (s, rp[0], rq[0]))
            continue
        if rp[0] == "exc":
            same += (rp[1] == rq[1])
            diff += (rp[1] != rq[1])
            continue
        built += 1
        if fingerprint(rp[1]) == fingerprint(rq[1]):
            same += 1
        else:
            diff += 1
            print("  FAIL %-24s fingerprints differ" % s)
    chk(diff == 0, "%d/%d specs build identically before and after"
        % (same, len(CORPUS_SPECS)))
    chk(built >= 30, "at least 30 corpus specs really constructed an optimizer",
        "%d built" % built)

    print("\nR2  BACKWARD COMPATIBILITY through block_product (the edited function)")
    bad = 0
    for s in CORPUS_SPECS:
        rp, rq = build(pre, s, net), build(post, s, net)
        if rp[0] != "ok" or rq[0] != "ok":
            continue
        if not same_red(reduce_with(rp[1], U, V), reduce_with(rq[1], U, V)):
            bad += 1
            print("  FAIL %-24s reduction differs pre vs post" % s)
    chk(bad == 0, "every corpus spec's reduction is BITWISE IDENTICAL pre vs post")

    print("\nR3  `tn:X` IS `X` structurally")
    bad = 0
    for s in CORPUS_SPECS:
        rx, rt = build(post, s, net), build(post, "tn:" + s, net)
        if rx[0] != "ok" or rt[0] != "ok":
            bad += (rx[0] != rt[0])
            continue
        if fingerprint(rx[1]) != fingerprint(rt[1]):
            bad += 1
            print("  FAIL %-24s tn: changes the structure" % s)
        if not (getattr(rt[1], "_rednorm") is True
                and getattr(rx[1], "_rednorm") is False):
            bad += 1
            print("  FAIL %-24s _rednorm flags are wrong" % s)
    chk(bad == 0, "`tn:X` and `X` give the same groups, type and beta; only "
                  "_rednorm differs")

    print("\nR5  THE NO-OP at every single-tensor granularity")
    bad = 0
    for s in SINGLE_TENSOR:
        rx, rt = build(post, s, net), build(post, "tn:" + s, net)
        if rx[0] != "ok" or rt[0] != "ok":
            bad += 1
            continue
        if not same_red(reduce_with(rx[1], U, V), reduce_with(rt[1], U, V)):
            bad += 1
            print("  FAIL %-24s tn: changed a single-tensor granularity" % s)
    chk(bad == 0, "`tn:X` == `X` BITWISE for %s" % ", ".join(SINGLE_TENSOR))

    print("\nR4  THE INTERVENTION IS LIVE, AND IS THE ONE REGISTERED")
    bad = 0
    for s in MULTI_TENSOR:
        rx, rt = build(post, s, net), build(post, "tn:" + s, net)
        if rx[0] != "ok" or rt[0] != "ok":
            bad += 1
            print("  FAIL %-24s did not build" % s)
            continue
        zx, zt = reduce_with(rx[1], U, V), reduce_with(rt[1], U, V)
        if same_red(zx, zt):
            bad += 1
            print("  FAIL %-24s tn: did NOT change a multi-tensor reduction" % s)
        # independent recomputation of the registered estimand
        o = rt[1]
        if o.stepsize_type == "scalar":
            groups = [list(range(T))]
        else:
            groups = [list(g) for g in o.param_groups_indices]
        want = torch.tensor(
            [float(sum(float((U[i] * V[i]).double().sum()) / numel[i] for i in g))
             for g in groups], dtype=torch.float64)
        got = zt[0].reshape(-1).double()
        rel = float((got - want).abs().max() / want.abs().max())
        ok = rel < 1e-5
        if not ok:
            bad += 1
        print("       %-24s groups=%d  max rel dev from sum_i <u,v>/n_i = %.2e  %s"
              % (s, len(groups), rel, "ok" if ok else "MISMATCH"))
    chk(bad == 0, "`tn:` changes every multi-tensor reduction and equals "
                  "sum_i <u_i,v_i>/numel_i")

    print("\nR7  THE GRAMMAR IS LOUD")
    for s in ("tn:", "tn:nonsense", "tn:sets:1-61", "tn:[49,14]"):
        r = build(post, s, net)
        chk(r[0] == "exc", "%-16s raises rather than silently defaulting" % s,
            str(r[1] if r[0] == "exc" else "BUILT"))

    print("\nR6  RULE 20 safety of the literal spec strings")
    for s in BATCH_SPECS:
        ok = (" " not in s and "\t" not in s and "'" not in s and '"' not in s
              and "=" not in s and not s.startswith("-")
              and re.match(r"^[A-Za-z0-9_.,:/-]+$", s) is not None)
        chk(ok, "%-24s is a single safe shell token" % s)
    if a.argsline_guard and os.path.exists(a.argsline_guard):
        cmd = ("--optimizer HF --stepsize-groups %s --seed 15"
               % BATCH_SPECS[2])
        p = subprocess.run([sys.executable, a.argsline_guard, "--cmdline", cmd],
                           capture_output=True, text=True)
        hit = BATCH_SPECS[2] in p.stdout
        chk(p.returncode == 0 and hit,
            "argsline_guard.py round-trips `%s` as ONE value" % BATCH_SPECS[2],
            "rc=%d" % p.returncode)
    else:
        print("  SKIP argsline_guard.py not given")

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %r" % FAILED))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
