"""Structure, inertness, arithmetic and loudness of PATCH_VOTEWEIGHT (`VOTE_W=...`),
on the LIVE PlainNet18_c100 model of the isolated `cvt1` tree.  CORRECTIONS 227.

STRATEGY (the test_namesets / test_rednorm / test_probe_tensor pattern): compare the
PATCHED file against the UNPATCHED file, never against my expectation of it.

  V0  STRUCTURE.  PATCH_VOTEWEIGHT is absent from --pre and present in --post; the
      four inserted regions each appear exactly as registered; deleting them from
      --post reproduces --pre BYTE FOR BYTE.
  V1  OFF IS INERT.  For every spec cvt1 runs (scalar, sets:HEAD) and for the other
      PlainNet specs the campaign ran (layerwise, cpl1's ISO, cpl2's TWIN), with
      VOTE_W UNSET and with VOTE_W EMPTY: beta at EVERY step and probe.jsonl
      (PROBE and PROBE_TENSOR on) are IDENTICAL to the unpatched file's, and the run
      printed exactly one `VOTE_W: off` line.
  V2  IDENTITY IS INERT.  VOTE_W naming the arms' own tensors with weight 1 / 1.0
      (scalar: layer4.1.bn2.weight; scalar two items; sets:HEAD: layer4.1.bn1.weight)
      gives beta and probe.jsonl IDENTICAL to the unpatched file's.
  V3  THE INTERVENTION IS LIVE AND IS WHAT IT SAYS, for cvt1's three intervened
      arms (MUTE, DOSE, INJECT) with PROBE=1 and PROBE_TENSOR=1:
        (a) at every record the harness's z (z_agg) == sum_i w_i z_tensor_i per
            group, and mom_pre == sum_i w_i m_tensor_i (momentum is linear in z);
        (b) the UNWEIGHTED decomposition does NOT hold on the records where the
            weighted term is not negligible (non-vacuity);
        (c) the applied sign on every unclamped coordinate is
            sign(b2*mom_pre + (1-b2)*z_agg), as the registered B4 check reads it;
        (d) block_product itself, on random (u, v), equals an independent float64
            weighted sum, and with every weight 1 equals the ORIGINAL branch BITWISE.
  V4  THE GRAMMAR IS LOUD.  Unknown names, repeated names, signs, exponents, inf,
      nan, empty items, a layerwise spec, a nodewise spec, a named tensor alone in
      its group, and a `tn:` spec each RAISE at construction.
  V5  THE WITNESS LINES are exactly the literals the scorer's G-VOTEW gate expects.
  V6  RULE 20: every VOTE_W value cvt1 uses is one token of [A-Za-z0-9_.:/], no comma
      (it rides sbatch's comma-separated --export list).

RUN (needs torch; on the cluster against the real isolated tree):
  python3 tests/test_voteweight.py --pre  $WS/harness_cvt1/cifar10/Optimizers/HF.py.pre_voteweight \\
        --post $WS/harness_cvt1/cifar10/Optimizers/HF.py --cifar-dir $WS/harness_cvt1/cifar10 --work /tmp/x
"""
import argparse, ast, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_probe_tensor as T0   # registered, imported, NOT edited (load_hf, run_one)

FAILED = []

NET = "PlainNet18_c100"
SPEC_K01 = "scalar"
SPEC_HEAD = "sets:1-49,51-53/layer4.1.bn2.weight"
OTHER_SPECS = {"kL": "layerwise",
               "cpl1-ISO": "sets:1-43,45-49,51-53/layer4.0.bn2.weight,layer4.1.bn2.weight",
               "cpl2-TWIN": "sets:1-46,48-53/layer4.1.bn1.weight"}
VOTEW = {"MUTE": "layer4.1.bn2.weight:0", "DOSE": "layer4.1.bn2.weight:0.1", "INJECT": "layer4.1.bn1.weight:691"}
ARMSPEC = {"MUTE": SPEC_K01, "DOSE": SPEC_K01, "INJECT": SPEC_HEAD}
WEIGHTS = {"MUTE": {49: 0.0}, "DOSE": {49: 0.1}, "INJECT": {46: 691.0}}   # 0-based
WITNESS = {
    "off": "VOTE_W: off",
    "MUTE": "VOTE_W: on type=scalar items=50:layer4.1.bn2.weight:w=0.0:group=0:groupsize=53",
    "DOSE": "VOTE_W: on type=scalar items=50:layer4.1.bn2.weight:w=0.1:group=0:groupsize=53",
    "INJECT": "VOTE_W: on type=blockwise items=47:layer4.1.bn1.weight:w=691.0:group=0:groupsize=52",
}

GUARD = ("            # --- PATCH_VOTEWEIGHT ---\n"
         "            if getattr(self, '_vw_on', False):\n"
         "                return self._vw_block_product(u, v)\n"
         "            # --- end PATCH_VOTEWEIGHT ---\n")
CALL = "        self._vw_init(net_param_names_and_size)  # PATCH_VOTEWEIGHT\n"
M_HEAD = "    # ------------------------------------------------------ PATCH_VOTEWEIGHT\n"
M_TAIL = "    # -------------------------------------------------- end PATCH_VOTEWEIGHT\n\n"


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def structural(pre_src, post_src):
    print("=" * 78)
    print("V0  STRUCTURE: the two files differ ONLY by PATCH_VOTEWEIGHT's four regions")
    chk("PATCH_VOTEWEIGHT" not in pre_src and "VOTE_W" not in pre_src and "_vw_" not in pre_src,
        "--pre is UNPATCHED (no marker, no VOTE_W, no _vw_ name)")
    chk("PATCH_VOTEWEIGHT" in post_src, "--post IS patched")
    s = post_src
    chk(s.count(GUARD) == 2, "regions 1+2: the guarded early return appears exactly twice", "%d" % s.count(GUARD))
    chk(s.find("        if self.stepsize_type == 'scalar':\n" + GUARD) >= 0
        and s.find("        if self.stepsize_type == 'blockwise':\n" + GUARD) >= 0,
        "regions 1+2 sit at the top of block_product's scalar and blockwise branches")
    s = s.replace(GUARD, "")
    chk(s.count(CALL) == 1 and ("        self.len_beta_list = len(self.beta)\n" + CALL) in s,
        "region 3: ONE _vw_init call, right after init_meta's `self.len_beta_list = len(self.beta)`")
    s = s.replace(CALL, "")
    i0, i1 = s.find(M_HEAD), s.find(M_TAIL)
    chk(i0 >= 0 and i1 > i0 and s[i1 + len(M_TAIL):].startswith("    def check_required_attributes("),
        "region 4: the method block sits immediately before check_required_attributes")
    if i0 >= 0 and i1 > i0:
        s = s[:i0] + s[i1 + len(M_TAIL):]
    chk(s == pre_src, "deleting the four inserted regions reproduces --pre BYTE FOR BYTE",
        "" if s == pre_src else "(%d vs %d bytes)" % (len(s), len(pre_src)))
    tree = ast.parse(post_src)
    hf = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HF"]
    meth = [f.name for f in hf[0].body if isinstance(f, ast.FunctionDef)] if hf else []
    chk([m for m in meth if m.startswith("_vw_")] == ["_vw_parse", "_vw_init", "_vw_block_product"],
        "exactly three new methods, all `_vw_`-prefixed", str([m for m in meth if m.startswith("_vw_")]))
    pre_meth = [f.name for f in [n for n in ast.parse(pre_src).body if isinstance(n, ast.ClassDef)
                                 and n.name == "HF"][0].body if isinstance(f, ast.FunctionDef)]
    chk([m for m in meth if not m.startswith("_vw_")] == pre_meth, "every pre-existing method is present, in order")


def with_env(val, fn):
    old = os.environ.get("VOTE_W")
    if val is None:
        os.environ.pop("VOTE_W", None)
    else:
        os.environ["VOTE_W"] = val
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("VOTE_W", None)
        else:
            os.environ["VOTE_W"] = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--cifar-dir", required=True)
    ap.add_argument("--work", default="")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--every", type=int, default=7)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--structural-only", action="store_true")
    a = ap.parse_args()

    pre_src, post_src = open(a.pre).read(), open(a.post).read()
    structural(pre_src, post_src)
    if a.structural_only:
        print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %r" % FAILED))
        raise SystemExit(1 if FAILED else 0)

    import torch
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    work = a.work or os.path.join(os.getcwd(), "vw_test_work")
    os.makedirs(work, exist_ok=True)
    os.environ.pop("VOTE_W", None)
    pre, post = T0.load_hf(a.pre, "pre_vw"), T0.load_hf(a.post, "post_vw")
    sys.path.insert(0, a.cifar_dir)
    cwd = os.getcwd()
    os.chdir(a.cifar_dir)
    from build_network import build_network
    torch.manual_seed(20260916)
    net0 = build_network(NET, "cpu")
    os.chdir(cwd)
    nps = [(n, p.data.size()) for n, p in net0.named_parameters()]
    names = [n for n, _ in nps]
    Tn = len(nps)
    print("\nlive %s: %d parameter tensors" % (NET, Tn))
    chk(Tn == 53 and names[49] == "layer4.1.bn2.weight" and names[46] == "layer4.1.bn1.weight",
        "53 tensors; idx 50 layer4.1.bn2.weight, idx 47 layer4.1.bn1.weight (1-based, by NAME)")
    g = torch.Generator().manual_seed(11)
    data = [(torch.randn(a.batch, 3, 32, 32, generator=g),
             torch.randint(0, 100, (a.batch,), generator=g)) for _ in range(a.steps)]

    def run(mod, spec, vw, every, pt=True):
        return with_env(vw, lambda: T0.run_one(mod, spec, net0, data, every, work, pt, a.steps))

    print("\n" + "=" * 78)
    print("V1  OFF IS INERT (unset and empty), PROBE=%d and PROBE_TENSOR=1" % a.every)
    specs = dict(OTHER_SPECS, k01=SPEC_K01, HEAD=SPEC_HEAD)
    for lab, spec in specs.items():
        bp, rp, op, _ = run(pre, spec, None, a.every)
        for vw in (None, ""):
            bu, ru, ou, _ = run(post, spec, vw, a.every)
            chk(bp == bu and rp is not None and rp == ru,
                "%-9s VOTE_W %-7s beta at every step AND probe.jsonl bytes == unpatched" % (lab, "unset" if vw is None else "empty"),
                "%d steps, %d records" % (len(bp), rp.count(b"\n") if rp else 0))
            chk(sum(1 for ln in ou if ln.startswith("VOTE_W:")) == 1 and WITNESS["off"] in ou,
                "%-9s VOTE_W %-7s printed exactly one `VOTE_W: off`" % (lab, "unset" if vw is None else "empty"))
        chk(not any(ln.startswith("VOTE_W") for ln in op), "%-9s the unpatched file prints no VOTE_W line" % lab)

    print("\n" + "=" * 78)
    print("V2  IDENTITY (every named weight 1) IS INERT")
    for lab, spec, vw in (("k01", SPEC_K01, "layer4.1.bn2.weight:1"),
                          ("k01", SPEC_K01, "layer4.1.bn2.weight:1.0"),
                          ("k01", SPEC_K01, "layer4.1.bn2.weight:1/layer4.1.bn1.weight:1.0"),
                          ("HEAD", SPEC_HEAD, "layer4.1.bn1.weight:1"),
                          ("HEAD", SPEC_HEAD, "layer4.1.bn1.weight:1.0/linear.weight:1")):
        bp, rp, _, _ = run(pre, spec, None, a.every)
        bu, ru, ou, _ = run(post, spec, vw, a.every)
        chk(bp == bu and rp is not None and rp == ru, "%-4s VOTE_W=%s: beta and probe.jsonl == unpatched" % (lab, vw))
        chk(any(ln.startswith("VOTE_W: on") for ln in ou), "%-4s VOTE_W=%s: the switch WAS on (witness line printed)" % (lab, vw))

    print("\n" + "=" * 78)
    print("V3  THE INTERVENTION: MUTE, DOSE, INJECT (PROBE=1, PROBE_TENSOR=1)")
    ms = T0.AMETA["meta_stepsize"]; b2 = T0.AMETA["Lion_beta2"]
    lo, hi = [float(x) for x in T0.CLIP.split(":")]
    for arm in ("MUTE", "DOSE", "INJECT"):
        spec, W = ARMSPEC[arm], WEIGHTS[arm]
        groups = [list(range(Tn))] if spec == "scalar" else \
            [[names.index(n) for n in grp] for grp in post.HF.polish_the_stepsize_groups(None, spec, nps)]
        bo, ro, oo, _ = run(post, spec, VOTEW[arm], 1)
        bp, rp, _, _ = run(pre, spec, None, 1)
        recs = [json.loads(x) for x in ro.decode().splitlines()] if ro else []
        chk(WITNESS[arm] in oo, "%-6s witness line == the registered literal" % arm, WITNESS[arm])
        worst_w = worst_m = 0.0; bad_w = bad_m = 0; informative = 0; unweighted_ok_on_informative = 0
        nchk = nbad = 0
        for r in recs:
            zt, mt = r["z_tensor"], r["m_tensor"]
            za, ma = r["z_agg"][0], r["mom_pre"][0]
            sz = sum(abs(v) for v in zt) + 1e-30
            sm = sum(abs(v) for v in mt) + 1e-30
            wz = [sum(W.get(i, 1.0) * zt[i] for i in grp) for grp in groups]
            wm = [sum(W.get(i, 1.0) * mt[i] for i in grp) for grp in groups]
            uz = [sum(zt[i] for i in grp) for grp in groups]
            dz = max(abs(p - q) for p, q in zip(wz, za)) / sz
            dm = max(abs(p - q) for p, q in zip(wm, ma)) / sm
            worst_w, worst_m = max(worst_w, dz), max(worst_m, dm)
            bad_w += dz > a.tol
            bad_m += dm > a.tol
            gap = max(abs(p - q) for p, q in zip(wz, uz)) / sz
            if gap > 1e3 * a.tol:
                informative += 1
                unweighted_ok_on_informative += max(abs(p - q) for p, q in zip(uz, za)) / sz <= a.tol
            bpre, bpost = r["beta_pre"][0], r["beta"]
            for k in range(len(bpre)):
                L = b2 * ma[k] + (1 - b2) * za[k]
                s = 1 if L > 0 else (-1 if L < 0 else 0)
                if not (lo + 1e-9 < bpost[k] < hi - 1e-9):
                    continue
                nchk += 1
                nbad += abs(-(bpost[k] - bpre[k]) / ms - s) > 1e-3
        chk(recs and bad_w == 0, "%-6s (a) z_agg == sum_i w_i z_tensor_i per group at EVERY record" % arm,
            "%d records, worst rel %.2e" % (len(recs), worst_w))
        chk(recs and bad_m == 0, "%-6s (a) mom_pre == sum_i w_i m_tensor_i per group at EVERY record" % arm,
            "worst rel %.2e" % worst_m)
        chk(informative >= 5 and unweighted_ok_on_informative == 0,
            "%-6s (b) non-vacuous: on %d informative records the UNWEIGHTED sum does NOT match z_agg" % (arm, informative),
            "%d matched" % unweighted_ok_on_informative)
        chk(nchk > 0 and nbad == 0, "%-6s (c) applied sign == sign(b2*mom_pre + (1-b2)*z_agg) on every unclamped coord" % arm,
            "%d checked" % nchk)
        rp_recs = [json.loads(x) for x in rp.decode().splitlines()] if rp else []
        first = next((k for k, (x, y) in enumerate(zip(rp_recs, recs)) if x["z_agg"] != y["z_agg"]), None)
        chk(first is not None and first <= 2,
            "%-6s z differs from the unpatched run's by record %s (step 0 has h = 0, so every term is 0)" % (arm, first))

    print("\nV3d block_product on random (u, v): float64 weighted sum; weight 1 == ORIGINAL bitwise")
    gg = torch.Generator().manual_seed(3)
    u = [torch.randn(tuple(s), generator=gg) for _n, s in nps]
    v = [torch.randn(tuple(s), generator=gg) for _n, s in nps]
    for arm in ("MUTE", "DOSE", "INJECT"):
        spec, W = ARMSPEC[arm], WEIGHTS[arm]

        def mk(vw, mod=post):
            def f():
                import io, contextlib
                o = mod.HF.__new__(mod.HF)
                o.num_layers = Tn
                o._device = torch.device("cpu")
                with contextlib.redirect_stdout(io.StringIO()):
                    mod.HF.init_meta(o, spec, nps, 1e-6)
                return o
            return with_env(vw, f)
        on, one, orig = mk(VOTEW[arm]), mk(VOTEW[arm].rsplit(":", 1)[0] + ":1"), mk(None, pre)
        got = [float(x) for x in on.block_product(u, v)[0].reshape(-1)]
        groups = [list(range(Tn))] if spec == "scalar" else on.param_groups_indices
        ind = [sum(W.get(i, 1.0) * float((u[i].double() * v[i].double()).sum()) for i in grp) for grp in groups]
        chk(all(abs(p - q) <= 1e-4 * (1 + abs(q)) for p, q in zip(got, ind)),
            "%-6s weighted block_product == independent float64 sum" % arm, "%s vs %s" % (got[:2], ind[:2]))
        a1 = one.block_product(u, v)[0]
        a0 = orig.block_product(u, v)[0]
        chk(torch.equal(a1.reshape(-1), a0.reshape(-1)) and a1.dtype == a0.dtype,
            "%-6s the same arm at weight 1 == the ORIGINAL branch, BITWISE (dtype %s)" % (arm, a0.dtype))

    print("\n" + "=" * 78)
    print("V4  THE GRAMMAR IS LOUD")
    import io, contextlib

    def raises(spec, vw):
        def f():
            o = post.HF.__new__(post.HF)
            o.num_layers = Tn
            o._device = torch.device("cpu")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    post.HF.init_meta(o, spec, nps, 1e-6)
            except ValueError as e:
                return "PATCH_VOTEWEIGHT" in str(e)
            return False
        return with_env(vw, f)
    for spec, vw, why in (
            ("scalar", "layer9.bn1.weight:0", "unknown name"),
            ("scalar", "50:0", "an index, not a name"),
            ("scalar", "layer4.1.bn2.weight:0/layer4.1.bn2.weight:1", "repeated name"),
            ("scalar", "layer4.1.bn2.weight:-1", "a sign"),
            ("scalar", "layer4.1.bn2.weight:1e3", "an exponent"),
            ("scalar", "layer4.1.bn2.weight:inf", "inf"),
            ("scalar", "layer4.1.bn2.weight:nan", "nan"),
            ("scalar", "layer4.1.bn2.weight", "no weight"),
            ("scalar", "layer4.1.bn2.weight:0/", "an empty item"),
            ("scalar", "layer4.1.bn2.weight:0,layer4.1.bn1.weight:0", "a comma separator"),
            ("layerwise", "layer4.1.bn2.weight:0", "layerwise has no shared sum"),
            ("nodewise", "layer4.1.bn2.weight:0", "nodewise has no shared sum"),
            (SPEC_HEAD, "layer4.1.bn2.weight:0", "the named tensor is ALONE in its group"),
            ("tn:scalar", "layer4.1.bn2.weight:0", "combined with PATCH_REDNORM")):
        chk(raises(spec, vw), "raises: %-38s spec %-10s VOTE_W=%s" % (why, spec[:10], vw))

    print("\n" + "=" * 78)
    print("V5  THE WITNESS LITERALS AGREE WITH THE SCORER'S")
    sc = os.path.join(os.path.dirname(HERE), "analysis", "cVT1_voteweight_score.py")
    if os.path.exists(sc):
        txt = open(sc).read()
        m1 = re.search(r"^VOTEW = (\{.*?^\})", txt, re.S | re.M)
        m2 = re.search(r"^WITNESS = (\{.*?^\})", txt, re.S | re.M)
        sv = ast.literal_eval(m1.group(1)) if m1 else {}
        sw = ast.literal_eval(m2.group(1)) if m2 else {}
        chk(all(sv.get(k) == VOTEW[k] for k in VOTEW), "VOTE_W values == the scorer's VOTEW literals")
        chk(all(sw.get(k) == WITNESS[k] for k in ("MUTE", "DOSE", "INJECT")) and sw.get("k01") == WITNESS["off"]
            and sw.get("HEAD") == WITNESS["off"], "witness lines == the scorer's WITNESS literals (k01/HEAD: off)")
    else:
        print("  NOTE scorer not reachable at %s -- cross-check skipped" % sc)

    print("\nV6  RULE 20 token class")
    for k, v in VOTEW.items():
        chk(re.match(r"^[A-Za-z0-9_.:/]+$", v) is not None and "," not in v, "%-6s VOTE_W=%s is one token, no comma" % (k, v))

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %r" % FAILED))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
