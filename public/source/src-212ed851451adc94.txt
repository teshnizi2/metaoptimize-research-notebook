"""Structure, loudness, witness, schedule and meta-update semantics of PATCH_BETAHOLD (`BETA_HOLD=...`), on the LIVE
PlainNet18_c100 model of the isolated `cvt4` tree.  CORRECTIONS 237.

STRATEGY (the test_voteweight pattern): compare the PATCHED file against the UNPATCHED file (cvt1's HF.py), never
against my expectation of it; read every registered string and witness from the scorer.

  B0  STRUCTURE (no torch).  PATCH_BETAHOLD / BETA_HOLD / _bh_ absent from --pre, present in --post; the four inserted
      regions appear exactly as registered; deleting them from --post reproduces --pre BYTE FOR BYTE; exactly six new
      `_bh_` methods; every pre-existing method present, in order.
  B1  WITNESS AND INIT (torch, CPU).  For each registered arm string, HF.init_meta on the live model with HEAD's spec
      prints exactly ONE `BETA_HOLD:` line == the scorer's WITNESS_BH, byte for byte; the initial beta is
      [b0, b0] (tri, shared) or [b0, -15] (floor); VOTE_W stays off.  Unset and EMPTY print `BETA_HOLD: off`,
      set `_bh_on` False, and leave beta and every pre-existing attribute EQUAL to the unpatched file's.
  B2  THE GRAMMAR AND THE PREMISES ARE LOUD: 18 malformed values / specs / environments each RAISE at construction.
  B3  THE SCHEDULE.  The patch's own _bh_value(n), stored into float32, equals the scorer's v_of(n, P) at every n in
      0..50,001 for P_HIGH and P_HEAD, and -15 for floor.
  B4  THE META-UPDATE, WITH SYNTHETIC z (the momentum statement of the patch docstring).  Two HF objects, one from the
      UNPATCHED file and one from the PATCHED file with each hold, are driven through the harness's OWN
      Lion_meta_update + clamp (+ _bh_apply) for 3,000 updates with the same random z (the first z is 0, as in a
      run).  At every update: the complement's beta[0][0] and the WHOLE momentum_meta vector are BITWISE the
      unpatched object's (the hold touches neither; 50's momentum keeps its own recursion and is never read into an
      applied beta); beta[0][1] == the schedule (or == beta[0][0] for shared); _bh_nat == the value the harness's Lion
      wrote from the previous HELD value; _bh_active counts exactly the updates where they differ, and is > 0.
  B5  RULE 20: every registered BETA_HOLD value is one token of [A-Za-z0-9_.:], no comma.

RUN (needs torch; on the cluster against the real isolated tree):
  python3 tests/test_betahold.py --pre  $WS/harness_cvt4/cifar10/Optimizers/HF.py.pre_betahold \\
        --post $WS/harness_cvt4/cifar10/Optimizers/HF.py --cifar-dir $WS/harness_cvt4/cifar10
  python3 tests/test_betahold.py --pre ... --post ... --structural-only          # B0 only, no torch
"""
import argparse, ast, contextlib, importlib.util, io, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAILED = []

G_APPLY = ("            # --- PATCH_BETAHOLD ---\n"
           "            if getattr(self, '_bh_on', False):\n"
           "                self._bh_apply()\n"
           "            # --- end PATCH_BETAHOLD ---\n")
G_ATTACH = ("        # --- PATCH_BETAHOLD ---\n"
            "        if getattr(self, '_bh_on', False):\n"
            "            self._bh_attach(rec)\n"
            "        # --- end PATCH_BETAHOLD ---\n")
CALL = "        self._bh_init(net_param_names_and_size)  # PATCH_BETAHOLD\n"
M_HEAD = "    # ------------------------------------------------------ PATCH_BETAHOLD\n"
M_TAIL = "    # -------------------------------------------------- end PATCH_BETAHOLD\n\n"
NEW_METHODS = ["_bh_init", "_bh_value", "_bh_witness_peak", "_bh_set", "_bh_apply", "_bh_attach"]


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def _load(name, path):
    import importlib.machinery
    spec = importlib.util.spec_from_file_location(name, path, loader=importlib.machinery.SourceFileLoader(name, path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def structural(pre_src, post_src):
    print("=" * 78)
    print("B0  STRUCTURE: the two files differ ONLY by PATCH_BETAHOLD's four regions")
    chk("PATCH_BETAHOLD" not in pre_src and "BETA_HOLD" not in pre_src and "_bh_" not in pre_src,
        "--pre is UNPATCHED (no marker, no BETA_HOLD, no _bh_ name)")
    chk("PATCH_VOTEWEIGHT" in pre_src, "--pre is cvt1's PATCH_VOTEWEIGHT HF.py (the base this patch sits on)")
    chk("PATCH_BETAHOLD" in post_src, "--post IS patched")
    s = post_src
    anchor1 = ("                    self.beta[_i] = self.beta[_i].clamp(self._beta_lo, self._beta_hi)\n" + G_APPLY
               + "            self._probe(HtT_gradft)  # PATCH_PROBE\n")
    chk(s.count(G_APPLY) == 1 and s.count(anchor1) == 1,
        "region 1: ONE guarded _bh_apply, right after PATCH_CLIP's clamp and before _probe")
    s = s.replace(G_APPLY, "")
    chk(s.count(CALL) == 1 and ("        self._vw_init(net_param_names_and_size)  # PATCH_VOTEWEIGHT\n" + CALL) in s,
        "region 2: ONE _bh_init call, right after init_meta's _vw_init call")
    s = s.replace(CALL, "")
    chk(s.count(G_ATTACH) == 1 and (G_ATTACH + "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n") in s,
        "region 3: ONE guarded _bh_attach, right before _pt_attach in _probe")
    s = s.replace(G_ATTACH, "")
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
    chk([m for m in meth if m.startswith("_bh_")] == NEW_METHODS, "exactly six new methods, all `_bh_`-prefixed",
        str([m for m in meth if m.startswith("_bh_")]))
    pre_meth = [f.name for f in [n for n in ast.parse(pre_src).body if isinstance(n, ast.ClassDef)
                                 and n.name == "HF"][0].body if isinstance(f, ast.FunctionDef)]
    chk([m for m in meth if not m.startswith("_bh_")] == pre_meth, "every pre-existing method is present, in order")


@contextlib.contextmanager
def env(**kw):
    old = dict((k, os.environ.get(k)) for k in kw)
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def make(HF, torch, nps, spec, alg="Lion", clip=(-15.0, -2.3026), hier="none"):
    o = HF.__new__(HF)
    o.num_layers = len(nps)
    o._device = torch.device("cpu")
    o.args_meta = {"alg": alg, "meta_stepsize": 1e-3, "momentum_param": 0.99, "Lion_beta2": 0.9, "weight_decay": 0}
    o._beta_lo, o._beta_hi = clip if clip else (None, None)
    o._hier = hier
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        HF.init_meta(o, spec, nps, 1e-6)
    return o, buf.getvalue().splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--cifar-dir", default="")
    ap.add_argument("--structural-only", action="store_true")
    ap.add_argument("--updates", type=int, default=3000)
    a = ap.parse_args()
    pre_src, post_src = open(a.pre).read(), open(a.post).read()
    structural(pre_src, post_src)
    if a.structural_only:
        print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %d" % len(FAILED)))
        raise SystemExit(1 if FAILED else 0)

    import torch
    SC = _load("cvt4_scorer", os.path.join(REPO, "analysis", "cVT4_betahold_score.py"))
    tree = os.path.abspath(a.cifar_dir)
    sys.path.insert(0, tree)
    os.chdir(tree)
    from build_network import build_network
    net = build_network(SC.NET, "cpu")
    nps = [(n, p.data.size()) for n, p in net.named_parameters()]
    PRE = _load("hf_pre_betahold", a.pre).HF
    POST = _load("hf_post_betahold", a.post).HF

    print("\n" + "=" * 78)
    print("B1  WITNESS AND INIT on the live %s (%d tensors), HEAD's spec" % (SC.NET, len(nps)))
    chk([n for n, _s in nps] == SC.NAMES, "the live parameter names are the scorer's NAMES")
    for arm in SC.HELD:
        with env(BETA_HOLD=SC.BETAHOLD[arm], VOTE_W=None):
            o, out = make(POST, torch, nps, SC.HEADSPEC)
        bh = [ln for ln in out if ln.startswith("BETA_HOLD")]
        vw = [ln for ln in out if ln.startswith("VOTE_W")]
        chk(bh == [SC.WITNESS_BH[arm]] and vw == ["VOTE_W: off"],
            "B1 %-10s exactly one BETA_HOLD line == the scorer's witness; VOTE_W: off" % arm, (bh or ["(none)"])[0][:100])
        want1 = SC.LO if SC.MODE[arm][0] == "floor" else SC.B0
        chk(o._bh_on and o._bh_g == 1 and [float(x) for x in o.beta[0]] == [SC.B0, want1],
            "B1 %-10s _bh_on, group 1, initial beta [b0, %r]" % (arm, want1), str([float(x) for x in o.beta[0]]))
    for val in (None, ""):
        with env(BETA_HOLD=val, VOTE_W=None):
            o1, out1 = make(POST, torch, nps, SC.HEADSPEC)
            o0, _out0 = make(PRE, torch, nps, SC.HEADSPEC)
            k1, _ = make(POST, torch, nps, "scalar")
        chk([ln for ln in out1 if ln.startswith("BETA_HOLD")] == ["BETA_HOLD: off"] and o1._bh_on is False,
            "B1 BETA_HOLD %s: prints `BETA_HOLD: off`, _bh_on False" % ("unset" if val is None else "EMPTY"))
        same_attrs = sorted(k for k in vars(o1) if k != "_bh_on") == sorted(vars(o0))
        chk(same_attrs and torch.equal(o1.beta[0], o0.beta[0]) and o1.param_groups_indices == o0.param_groups_indices,
            "B1 BETA_HOLD %s: every attribute and beta EQUAL to the unpatched object's (only _bh_on added)"
            % ("unset" if val is None else "EMPTY"))
        chk(k1._bh_on is False, "B1 BETA_HOLD %s on the scalar spec is off too" % ("unset" if val is None else "EMPTY"))

    print("\n" + "=" * 78)
    print("B2  LOUDNESS: every malformed value, spec or environment RAISES at construction")
    n = SC.NAME_HEAD
    bad = [
        ("no mode", n, SC.HEADSPEC, {}), ("tri without P", n + ":tri", SC.HEADSPEC, {}),
        ("negative P", n + ":tri:-5", SC.HEADSPEC, {}), ("P = 0", n + ":tri:0", SC.HEADSPEC, {}),
        ("exponent P", n + ":tri:1e3", SC.HEADSPEC, {}), ("7-digit P", n + ":tri:1234567", SC.HEADSPEC, {}),
        ("unknown mode", n + ":fix:-15", SC.HEADSPEC, {}), ("trailing junk", n + ":floor,x", SC.HEADSPEC, {}),
        ("unknown name", "nosuch.weight:floor", SC.HEADSPEC, {}),
        ("name not alone in its group", "layer4.1.bn1.weight:floor", SC.HEADSPEC, {}),
        ("scalar spec", n + ":floor", "scalar", {}), ("layerwise spec", n + ":floor", "layerwise", {}),
        ("tn: spec", n + ":floor", "tn:" + SC.HEADSPEC, {}),
        ("shared with 3 groups", n + ":shared", "sets:1-48,51-53/49/layer4.1.bn2.weight", {}),
        ("VOTE_W on at the same time", n + ":floor", SC.HEADSPEC, {"VOTE_W": "layer4.1.bn1.weight:1"}),
        ("meta algorithm Adam", n + ":floor", SC.HEADSPEC, {"alg": "Adam"}),
        ("BETA_CLIP unset", n + ":tri:100", SC.HEADSPEC, {"clip": None}),
        ("HIER=twolevel", n + ":shared", SC.HEADSPEC, {"hier": "twolevel"}),
    ]
    for lab, val, spec, kw in bad:
        vw = kw.pop("VOTE_W", None)
        raised = False
        try:
            with env(BETA_HOLD=val, VOTE_W=vw):
                make(POST, torch, nps, spec, **kw)
        except ValueError as e:
            raised = "PATCH_BETAHOLD" in str(e)
        chk(raised, "B2 %-28s RAISES" % lab, repr(val)[:60])

    print("\n" + "=" * 78)
    print("B3  THE SCHEDULE: the patch's _bh_value, stored into float32, == the scorer's v_of at every update")
    for arm in ("HOLDHIGH", "HOLDHEAD", "HOLDLOW"):
        with env(BETA_HOLD=SC.BETAHOLD[arm], VOTE_W=None):
            o, _ = make(POST, torch, nps, SC.HEADSPEC)
        mism = 0
        t = torch.zeros(1)
        for k in range(0, 50002):
            t[0] = o._bh_value(k)
            want = SC.hold_value(arm, k)
            mism += float(t[0]) != want
        chk(mism == 0, "B3 %-10s float32(_bh_value(n)) == scorer hold_value(n) for n = 0..50001" % arm, "%d mismatches" % mism)

    print("\n" + "=" * 78)
    print("B4  THE META-UPDATE with synthetic z: complement and momentum untouched, 50 held, the patch's record exact")
    g = torch.Generator().manual_seed(237)
    for arm in SC.HELD + ("TRI100",):
        val = SC.BETAHOLD.get(arm) or (SC.NAME_HEAD + ":tri:100")
        with env(BETA_HOLD=None, VOTE_W=None):
            U, _ = make(PRE, torch, nps, SC.HEADSPEC)
        with env(BETA_HOLD=val, VOTE_W=None):
            H, _ = make(POST, torch, nps, SC.HEADSPEC)
        U.meta_update, H.meta_update = U.Lion_meta_update, H.Lion_meta_update
        bad_cmp = bad_mom = bad_hold = bad_nat = 0
        n_diff = 0
        mode = val.split(":", 1)[1]
        P = int(mode.split(":")[1]) if mode.startswith("tri") else None
        for t_ in range(1, a.updates + 1):
            z = torch.zeros(2) if t_ == 1 else torch.randn(2, generator=g) * torch.tensor([1.0, 3.0])
            U.Lion_meta_update([z.clone()])
            H.Lion_meta_update([z.clone()])
            for obj in (U, H):
                obj.beta[0] = obj.beta[0].clamp(obj._beta_lo, obj._beta_hi)
            wrote = float(H.beta[0][1])
            H._bh_apply()
            bad_cmp += float(H.beta[0][0]) != float(U.beta[0][0])
            bad_mom += not torch.equal(torch.as_tensor(H.momentum_meta[0]), torch.as_tensor(U.momentum_meta[0]))
            if mode == "shared":
                want = float(H.beta[0][0])
            elif mode == "floor":
                want = -15.0
            else:
                want = SC.v_of(t_, P)
            bad_hold += float(H.beta[0][1]) != want
            bad_nat += H._bh_nat != wrote
            n_diff += wrote != float(H.beta[0][1])
        chk(bad_cmp == 0, "B4 %-10s complement beta[0][0] BITWISE the unpatched object's at all %d updates" % (arm, a.updates))
        chk(bad_mom == 0, "B4 %-10s the WHOLE momentum_meta vector (50's entry included) BITWISE the unpatched object's" % arm)
        chk(bad_hold == 0, "B4 %-10s beta[0][1] == the registered hold after every update" % arm, "%d bad" % bad_hold)
        chk(bad_nat == 0 and H._bh_n == a.updates and H._bh_active == n_diff and n_diff > 0,
            "B4 %-10s _bh_nat == what Lion wrote; _bh_n == %d; _bh_active == %d updates where it differed (> 0)"
            % (arm, a.updates, n_diff))

    print("\n" + "=" * 78)
    print("B5  RULE 20")
    chk(all(re.match(r"^[A-Za-z0-9_.:]+$", SC.BETAHOLD[x]) and "," not in SC.BETAHOLD[x] for x in SC.HELD),
        "every registered BETA_HOLD value is one token of [A-Za-z0-9_.:], no comma")

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %d" % len(FAILED)))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
