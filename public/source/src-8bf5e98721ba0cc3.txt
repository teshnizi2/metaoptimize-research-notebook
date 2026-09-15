"""Inertness AND correctness tests for PATCH_PROBE_TENSOR (patches/patch_probe_tensor.py).

STRATEGY, copied from tests/test_rednorm.py.  The patch's claim is that it is
ADDITIVE, OPT-IN and READ-ONLY on the optimizer.  So the test compares the
patched code against the UNPATCHED code on the real optimizer driving the real
ResNet18_c100 on CPU, three ways, and requires bit-identity where the patch
claims it.

  R0  STRUCTURE.  `PATCH_PROBE_TENSOR` is absent from --pre and present in
      --post; deleting the THREE inserted regions from --post reproduces --pre
      BYTE FOR BYTE; the marker appears exactly 3 times.
  R1  CONTROL.  The unpatched optimizer run twice on identical data gives
      bit-identical beta trajectories and byte-identical probe.jsonl -- the
      CPU path is deterministic, so a later mismatch is the patch's fault.
  R2  INERTNESS.  For `scalar` and `layerwise`:
        (a) patched with PROBE_TENSOR UNSET == unpatched: beta identical at
            EVERY step (bitwise, all coordinates) and probe.jsonl identical
            byte for byte;
        (b) patched with PROBE_TENSOR=1 == unpatched: beta identical at EVERY
            step (the optimizer's arithmetic is untouched), and every probe
            record equals the unpatched record byte for byte once the
            appended keys are removed (the pre-existing keys keep their
            values AND their order).
  R3  DECOMPOSITION, on the PROBE_TENSOR=1 records.  At every record:
        scalar:    sum_i z_tensor[i] == z_agg[0][0]   (the harness's own z)
                   sum_i m_tensor[i] == mom_pre[0][0] (the harness's own EMA)
        layerwise: z_tensor == z_agg[0] elementwise; m_tensor == mom_pre[0]
      to relative tolerance TOL against sum_i |term_i| (float32 with
      cancellation), and the exact-equality rate is printed.
  R4  THE APPLIED SIGN.  With `--weight-decay-meta 0`, Lion moves each beta by
      exactly -ms*sign(b2*mom_pre + (1-b2)*z_agg) before the clamp.  For every
      unclamped coordinate at every record, -(beta - beta_pre)/ms must equal
      that sign.
  R5  PROBE_TENSOR=1 WITHOUT PROBE is a no-op: no probe_tensor.json, no
      probe.jsonl, the optimizer prints the OFF line and runs as unpatched.

RUN (needs torch; run it on the cluster against the real tree):
  python3 tests/test_probe_tensor.py --pre /path/HF.py.pre_probe_tensor \\
        --post /path/Optimizers/HF.py --cifar-dir /path/cifar10 \\
        --work /path/scratch [--steps 130 --every 40 --batch 8]
  python3 tests/test_probe_tensor.py --pre ... --post ... --structural-only
"""
import argparse, copy, importlib.util, json, os, shutil, sys

FAILED = []


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label,
                           ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)


INS1 = "            self._pt_capture(self.h_condenced, g, HtT_gradft)  # PATCH_PROBE_TENSOR\n"
INS2 = "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n"
INS3_HEAD = "    # ----------------------------------------------------- PATCH_PROBE_TENSOR\n"
INS3_TAIL = ("###################################################\n"
             "###################################################\n"
             "    # Updates")
PT_KEYS = ("z_tensor", "m_tensor", "z_agg", "mom_pre", "beta_pre", "pt_mp", "pt_b2",
           "pt_step_mismatch")

ABASE = {"alg": "SGDm", "momentum_param": 0.99, "weight_decay": 0.1}
AMETA = {"alg": "Lion", "momentum_param": 0.99, "Lion_beta2": 0.9,
         "weight_decay": 0.0, "meta_stepsize": 1e-3}
CLIP = "-15:-2.3026"


def structural(pre_src, post_src):
    print("=" * 78)
    print("R0  the two files differ ONLY by PATCH_PROBE_TENSOR")
    chk("PATCH_PROBE_TENSOR" not in pre_src, "the --pre file is UNPATCHED")
    chk("PATCH_PROBE_TENSOR" in post_src, "the --post file IS patched")
    s = post_src
    chk(s.count(INS1) == 1, "inserted region 1 (capture call) appears exactly once")
    s = s.replace(INS1, "", 1)
    chk(s.count(INS2) == 1, "inserted region 2 (attach call) appears exactly once")
    s = s.replace(INS2, "", 1)
    i0 = s.find(INS3_HEAD)
    i1 = s.find(INS3_TAIL)
    chk(i0 >= 0 and i1 > i0, "inserted region 3 (methods) sits before the Updates marker")
    if i0 >= 0 and i1 > i0:
        s = s[:i0] + s[i1:]
    chk(s == pre_src, "deleting the three inserted regions reproduces --pre BYTE FOR BYTE",
        "" if s == pre_src else "(%d vs %d bytes)" % (len(s), len(pre_src)))
    chk(post_src.count("PATCH_PROBE_TENSOR") == 3,
        "PATCH_PROBE_TENSOR appears exactly 3 times (one per region)",
        "%d" % post_src.count("PATCH_PROBE_TENSOR"))


class _W:
    def add_scalar(self, *a, **k): pass
    def add_scalars(self, *a, **k): pass
    def add_text(self, *a, **k): pass
    def close(self): pass


def load_hf(path, tag):
    # an explicit loader: the --pre file is named HF.py.pre_probe_tensor, whose
    # suffix spec_from_file_location does not recognise (loader would be None)
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("hf_" + tag, path)
    spec = importlib.util.spec_from_loader("hf_" + tag, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hf_" + tag] = mod
    spec.loader.exec_module(mod)
    return mod


def run_one(mod, spec, net0, data, every, work, pt, steps, probe=True):
    """Run HF from `mod` on a deep copy of net0 over `data`; return
    (betas: list[list[float]] per step, probe.jsonl bytes or None, stdout lines)."""
    import torch, io, contextlib
    pdir = os.path.join(work, "probe")
    shutil.rmtree(pdir, ignore_errors=True)
    for k in ("PROBE", "PROBE_DIR", "PROBE_TENSOR"):
        os.environ.pop(k, None)
    if probe:
        os.environ["PROBE"] = str(every)
        os.environ["PROBE_DIR"] = pdir
    if pt:
        os.environ["PROBE_TENSOR"] = "1"
    os.environ["BETA_CLIP"] = CLIP
    os.environ["HIER"] = ""
    net = copy.deepcopy(net0)
    net.train()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        opt = mod.HF(net, stepsize_groups=spec, alpha0=1e-6, args_base=dict(ABASE),
                     args_meta=dict(AMETA), gamma=1, writer=_W())
    crit = torch.nn.CrossEntropyLoss()
    betas = []
    with contextlib.redirect_stdout(buf):
        for t in range(steps):
            x, y = data[t]
            out = net(x)
            loss = crit(out, y)
            opt.step(net, loss)
            betas.append([float(v) for b in opt.beta for v in b.detach().reshape(-1).tolist()])
    pj = os.path.join(pdir, "probe.jsonl")
    rec = open(pj, "rb").read() if os.path.exists(pj) else None
    ptj = os.path.exists(os.path.join(pdir, "probe_tensor.json"))
    return betas, rec, buf.getvalue().splitlines(), ptj


def strip_pt(line):
    d = json.loads(line)
    for k in PT_KEYS:
        d.pop(k, None)
    return json.dumps(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--cifar-dir", default=os.environ.get("CIFAR10_DIR", ""))
    ap.add_argument("--net", default="ResNet18_c100")
    ap.add_argument("--work", default="")
    ap.add_argument("--steps", type=int, default=130)
    ap.add_argument("--every", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--structural-only", action="store_true")
    a = ap.parse_args()

    pre_src, post_src = open(a.pre).read(), open(a.post).read()
    structural(pre_src, post_src)
    if a.structural_only:
        print("\n%s" % ("ALL PASS (structural only)" if not FAILED else "FAILURES: %r" % FAILED))
        raise SystemExit(1 if FAILED else 0)

    import torch
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    work = a.work or os.path.join(os.getcwd(), "pt_test_work")
    os.makedirs(work, exist_ok=True)
    pre, post = load_hf(a.pre, "pre"), load_hf(a.post, "post")

    cif = a.cifar_dir or os.getcwd()
    sys.path.insert(0, cif)
    cwd = os.getcwd()
    os.chdir(cif)
    from build_network import build_network
    torch.manual_seed(20260908)
    net0 = build_network(a.net, "cpu")
    os.chdir(cwd)
    T = len(list(net0.parameters()))
    print("\nlive %s: %d parameter tensors" % (a.net, T))
    g = torch.Generator().manual_seed(7)
    data = [(torch.randn(a.batch, 3, 32, 32, generator=g),
             torch.randint(0, 100, (a.batch,), generator=g)) for _ in range(a.steps)]
    ms = AMETA["meta_stepsize"]; mp = AMETA["momentum_param"]; b2 = AMETA["Lion_beta2"]
    lo, hi = [float(x) for x in CLIP.split(":")]

    print("\nR1  CONTROL -- the unpatched optimizer is deterministic on this CPU path")
    b1, r1, _, _ = run_one(pre, "scalar", net0, data, a.every, work, False, a.steps)
    b2_, r2, _, _ = run_one(pre, "scalar", net0, data, a.every, work, False, a.steps)
    chk(b1 == b2_, "unpatched scalar: beta trajectory identical on two runs (%d steps)" % a.steps)
    chk(r1 is not None and r1 == r2, "unpatched scalar: probe.jsonl byte-identical on two runs",
        "%d records" % (r1.count(b"\n") if r1 else 0))

    for spec in ("scalar", "layerwise"):
        print("\nR2  INERTNESS -- %s" % spec)
        bp, rp, _, ptp = run_one(pre, spec, net0, data, a.every, work, False, a.steps)
        bu, ru, ou, ptu = run_one(post, spec, net0, data, a.every, work, False, a.steps)
        bo, ro, oo, pto = run_one(post, spec, net0, data, a.every, work, True, a.steps)
        nrec = rp.count(b"\n") if rp else 0
        chk(bp == bu, "(a) PROBE_TENSOR unset: beta identical to unpatched at every step",
            "%d steps x %d coords" % (len(bp), len(bp[0])))
        chk(rp is not None and rp == ru, "(a) PROBE_TENSOR unset: probe.jsonl byte-identical",
            "%d records" % nrec)
        chk(not ptu and not any("PROBE_TENSOR" in l for l in ou),
            "(a) PROBE_TENSOR unset: no probe_tensor.json, no PROBE_TENSOR line")
        chk(bp == bo, "(b) PROBE_TENSOR=1: beta identical to unpatched at every step")
        lp = rp.decode().splitlines() if rp else []
        lo_ = ro.decode().splitlines() if ro else []
        same = len(lp) == len(lo_) and all(strip_pt(x) == y for x, y in zip(lo_, lp))
        chk(same, "(b) PROBE_TENSOR=1: every record == unpatched record once the appended keys are removed",
            "%d vs %d records" % (len(lp), len(lo_)))
        chk(pto and any(l.startswith("PROBE_TENSOR: on") for l in oo),
            "(b) PROBE_TENSOR=1: probe_tensor.json written and the ON line printed")
        allk = all(all(k in json.loads(x) for k in ("z_tensor", "m_tensor", "z_agg", "mom_pre", "beta_pre"))
                   for x in lo_)
        chk(allk and lo_, "(b) every record carries the five appended fields")

        print("\nR3  DECOMPOSITION -- %s" % spec)
        recs = [json.loads(x) for x in lo_]
        bad_z = bad_m = 0; exact_z = exact_m = 0; n = 0
        worst_z = worst_m = 0.0
        for r in recs:
            zt = r["z_tensor"]; mt = r["m_tensor"]
            za = r["z_agg"][0]; ma = r["mom_pre"][0]
            chk(len(zt) == T and len(mt) == T, "record step %d: %d per-tensor terms" % (r["step"], len(zt)))
            if spec == "scalar":
                got_z, want_z = [sum(zt)], za
                got_m, want_m = [sum(mt)], ma
            else:
                got_z, want_z = zt, za
                got_m, want_m = mt, ma
            n += 1
            sz = sum(abs(v) for v in zt) + 1e-30
            sm = sum(abs(v) for v in mt) + 1e-30
            dz = max(abs(p - q) for p, q in zip(got_z, want_z)) / sz
            dm = max(abs(p - q) for p, q in zip(got_m, want_m)) / sm
            worst_z = max(worst_z, dz); worst_m = max(worst_m, dm)
            bad_z += dz > a.tol; bad_m += dm > a.tol
            exact_z += got_z == want_z; exact_m += got_m == want_m
        chk(n > 0 and bad_z == 0, "sum/elementwise of z_tensor == the harness's z at every record",
            "worst rel %.2e, exact on %d/%d" % (worst_z, exact_z, n))
        chk(n > 0 and bad_m == 0, "sum/elementwise of m_tensor == the harness's momentum_meta at every record",
            "worst rel %.2e, exact on %d/%d" % (worst_m, exact_m, n))
        chk(all(abs(r["pt_mp"] - mp) < 1e-12 and abs(r["pt_b2"] - b2) < 1e-12 for r in recs),
            "the recorded coefficients are the harness's momentum_param and Lion_beta2")

        print("\nR4  THE APPLIED SIGN -- %s" % spec)
        nchk = nbad = nclamp = 0
        for r in recs:
            bpre = r["beta_pre"][0]; bpost = r["beta"]
            za = r["z_agg"][0]; ma = r["mom_pre"][0]
            for i in range(len(bpre)):
                L = b2 * ma[i] + (1 - b2) * za[i]
                s = 1 if L > 0 else (-1 if L < 0 else 0)
                unclamped = bpost[i] > lo + 1e-9 and bpost[i] < hi - 1e-9
                if not unclamped:
                    nclamp += 1; continue
                step = -(bpost[i] - bpre[i]) / ms
                nchk += 1
                if abs(step - s) > 1e-3:
                    nbad += 1
        chk(nchk > 0 and nbad == 0, "-(beta - beta_pre)/ms == sign(b2*mom_pre + (1-b2)*z_agg) on every unclamped coordinate",
            "%d checked, %d clamped" % (nchk, nclamp))

    print("\nR5  PROBE_TENSOR=1 WITHOUT PROBE is a no-op")
    bx, rx, ox, ptx = run_one(post, "scalar", net0, data, a.every, work, True, a.steps, probe=False)
    chk(rx is None and not ptx, "no probe.jsonl and no probe_tensor.json written")
    chk(any(l.startswith("PROBE_TENSOR: requested but") for l in ox), "the OFF line is printed")
    chk(bx == b1, "beta trajectory identical to the unpatched run")

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %r" % FAILED))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
