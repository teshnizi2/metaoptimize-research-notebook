"""Inertness AND decomposition of PATCH_PROBE_TENSOR on m=2 `sets:` (blockwise)
partitions -- the arms `ciso1` runs.

tests/test_probe_tensor.py (registered with ctd1, CORRECTIONS 182) proves R0-R5
for `scalar` and `layerwise` and is NOT edited (its spec tuple is literal).  The
`ciso1` batch runs three `sets:` m=2 partitions under PROBE_TENSOR=1, so the
same claims are re-proved HERE for those exact spec strings, by importing the
registered test's helpers (run_one, strip_pt, load_hf) unchanged:

  B2  INERTNESS, per spec: patched-with-PROBE_TENSOR-unset == unpatched (beta
      bit-identical at every step, probe.jsonl byte-identical), and
      patched-with-PROBE_TENSOR=1 == unpatched on beta at every step, every
      record equal once the appended keys are removed.
  B3  DECOMPOSITION, per spec: for each group k, sum_{i in group k} z_tensor[i]
      == z_agg[0][k] and sum m_tensor[i] == mom_pre[0][k] at every record.
  B4  THE APPLIED SIGN, per coordinate, on every unclamped coordinate.
  B5  the probe record's `beta` has exactly m entries and probe_tensor.json says
      stepsize_type == 'blockwise'.

RUN (needs torch; on the cluster against the real tree):
  python3 tests/test_probe_tensor_blockwise.py --pre /path/HF.py.pre_probe_tensor \\
        --post /path/Optimizers/HF.py --cifar-dir /path/cifar10 --work /path/scratch
The --pre file must carry PATCH_NAMESETS (it does: HF.py.pre_probe_tensor was
taken on 2026-09-08 21:09, after PATCH_NAMESETS on 2026-09-03).
"""
import argparse, copy, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_probe_tensor as T0   # the registered test, imported, not edited

SPECS = {
    "ISO": "sets:1-49,51-52,54-58,60-62/layer4.0.bn2.weight,layer4.0.shortcut.1.weight,layer4.1.bn2.weight",
    "CTRL": "sets:1-19,21-22,24-28,30-62/layer2.0.bn2.weight,layer2.0.shortcut.1.weight,layer2.1.bn2.weight",
    "ONE": "sets:1-49,51-62/layer4.0.bn2.weight",
}
FAILED = []


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)


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
    a = ap.parse_args()

    pre_src, post_src = open(a.pre).read(), open(a.post).read()
    print("=" * 78)
    print("B0  preconditions")
    chk("PATCH_PROBE_TENSOR" not in pre_src and "PATCH_PROBE_TENSOR" in post_src, "--pre unpatched, --post patched (PROBE_TENSOR)")
    chk("PATCH_NAMESETS" in pre_src and "PATCH_NAMESETS" in post_src, "BOTH files carry PATCH_NAMESETS (sets: is parseable on both)")

    import torch
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    work = a.work or os.path.join(os.getcwd(), "pt_bw_test_work")
    os.makedirs(work, exist_ok=True)
    pre, post = T0.load_hf(a.pre, "pre_bw"), T0.load_hf(a.post, "post_bw")
    cif = a.cifar_dir or os.getcwd()
    sys.path.insert(0, cif)
    cwd = os.getcwd()
    os.chdir(cif)
    from build_network import build_network
    torch.manual_seed(20260908)
    net0 = build_network(a.net, "cpu")
    os.chdir(cwd)
    nps = [(n, p.data.size()) for n, p in net0.named_parameters()]
    names = [n for n, _ in nps]
    Tn = len(nps)
    print("\nlive %s: %d parameter tensors" % (a.net, Tn))
    g = torch.Generator().manual_seed(7)
    data = [(torch.randn(a.batch, 3, 32, 32, generator=g),
             torch.randint(0, 100, (a.batch,), generator=g)) for _ in range(a.steps)]
    ms = T0.AMETA["meta_stepsize"]; mp = T0.AMETA["momentum_param"]; b2 = T0.AMETA["Lion_beta2"]
    lo, hi = [float(x) for x in T0.CLIP.split(":")]

    for arm, spec in SPECS.items():
        groups = post.HF.polish_the_stepsize_groups(None, spec, nps)
        gidx = [[names.index(n) for n in grp] for grp in groups]
        print("\nB2  INERTNESS -- %s  %s  sizes %s" % (arm, spec, [len(x) for x in gidx]))
        bp, rp, _, _ = T0.run_one(pre, spec, net0, data, a.every, work, False, a.steps)
        bu, ru, ou, ptu = T0.run_one(post, spec, net0, data, a.every, work, False, a.steps)
        bo, ro, oo, pto = T0.run_one(post, spec, net0, data, a.every, work, True, a.steps)
        chk(bp == bu, "(a) PROBE_TENSOR unset: beta identical to unpatched at every step",
            "%d steps x %d coords" % (len(bp), len(bp[0])))
        chk(rp is not None and rp == ru, "(a) PROBE_TENSOR unset: probe.jsonl byte-identical",
            "%d records" % (rp.count(b"\n") if rp else 0))
        chk(not ptu and not any("PROBE_TENSOR" in l for l in ou), "(a) no probe_tensor.json, no PROBE_TENSOR line")
        chk(bp == bo, "(b) PROBE_TENSOR=1: beta identical to unpatched at every step")
        lp = rp.decode().splitlines() if rp else []
        lo_ = ro.decode().splitlines() if ro else []
        chk(len(lp) == len(lo_) and all(T0.strip_pt(x) == y for x, y in zip(lo_, lp)) and lp,
            "(b) PROBE_TENSOR=1: every record == unpatched record once the appended keys are removed",
            "%d vs %d records" % (len(lp), len(lo_)))
        chk(pto and any(l.startswith("PROBE_TENSOR: on") and "type=blockwise" in l for l in oo),
            "(b) probe_tensor.json written and the ON line says type=blockwise")
        recs = [json.loads(x) for x in lo_]
        print("B3  DECOMPOSITION -- %s" % arm)
        bad_z = bad_m = 0; worst_z = worst_m = 0.0; n = 0
        for r in recs:
            zt, mt = r["z_tensor"], r["m_tensor"]
            za, ma = r["z_agg"][0], r["mom_pre"][0]
            chk(len(zt) == Tn and len(mt) == Tn and len(za) == len(gidx) and len(ma) == len(gidx) and len(r["beta"]) == len(gidx),
                "record step %d: %d per-tensor terms, %d group aggregates, %d betas" % (r["step"], len(zt), len(za), len(r["beta"])))
            gz = [sum(zt[i] for i in grp) for grp in gidx]
            gm = [sum(mt[i] for i in grp) for grp in gidx]
            sz = sum(abs(v) for v in zt) + 1e-30
            sm = sum(abs(v) for v in mt) + 1e-30
            dz = max(abs(p - q) for p, q in zip(gz, za)) / sz
            dm = max(abs(p - q) for p, q in zip(gm, ma)) / sm
            worst_z = max(worst_z, dz); worst_m = max(worst_m, dm)
            bad_z += dz > a.tol; bad_m += dm > a.tol
            n += 1
        chk(n > 0 and bad_z == 0, "group sums of z_tensor == the harness's per-group z at every record", "worst rel %.2e" % worst_z)
        chk(n > 0 and bad_m == 0, "group sums of m_tensor == the harness's per-group momentum at every record", "worst rel %.2e" % worst_m)
        print("B4  THE APPLIED SIGN -- %s" % arm)
        nchk = nbad = nclamp = 0
        for r in recs:
            bpre = r["beta_pre"][0]; bpost = r["beta"]
            za, ma = r["z_agg"][0], r["mom_pre"][0]
            for i in range(len(bpre)):
                L = b2 * ma[i] + (1 - b2) * za[i]
                s = 1 if L > 0 else (-1 if L < 0 else 0)
                if not (bpost[i] > lo + 1e-9 and bpost[i] < hi - 1e-9):
                    nclamp += 1; continue
                nchk += 1
                nbad += abs(-(bpost[i] - bpre[i]) / ms - s) > 1e-3
        chk(nchk > 0 and nbad == 0, "-(beta - beta_pre)/ms == sign(b2*mom_pre + (1-b2)*z_agg) on every unclamped coordinate",
            "%d checked, %d clamped" % (nchk, nclamp))
        hdr = json.load(open(os.path.join(work, "probe", "probe_tensor.json")))
        chk(hdr.get("stepsize_type") == "blockwise" and hdr.get("n_tensors") == Tn, "B5 probe_tensor.json: stepsize_type blockwise, %d tensors" % Tn)

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %r" % FAILED))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
