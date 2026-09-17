"""Structure, loudness, witness, schedule and meta-update semantics of PATCH_WINDOWHOLD (`WINDOW_HOLD=...`), on the LIVE
PlainNet18_c100 model of the isolated `cvt9` tree.  CORRECTIONS 249.  (The test_comphold pattern: compare the PATCHED
file against the UNPATCHED file -- cvt6's HF.py -- and read every string, witness and schedule from the scorer.)

  W0  STRUCTURE (no torch).  PATCH_WINDOWHOLD / WINDOW_HOLD / _wh_ absent from --pre, present in --post; the four regions
      appear exactly as registered; deleting them from --post reproduces --pre BYTE FOR BYTE; exactly five new `_wh_`
      methods right after PATCH_COMPHOLD's five; every pre-existing method present, in order.
  W1  WITNESS AND INIT (torch, CPU).  For each registered arm, HF.init_meta with its BETA_HOLD / COMP_HOLD / WINDOW_HOLD
      prints exactly ONE line of each kind == the scorer's witnesses and `VOTE_W: off`; the initial beta is [b0, b0] (EARLY,
      the tri dose arms), [b0, -15] (LATE, LOWHEADPATH).  WINDOW_HOLD unset and EMPTY print `WINDOW_HOLD: off`, set
      `_wh_on` False, and leave beta and every pre-existing attribute EQUAL to the unpatched (cvt6) object's, on k01,
      LOWHEADPATH, HIGHHEADPATH, MIDDOSE and RESDOSE.
  W2  LOUDNESS: malformed values, n0 >= n1, BETA_HOLD off / floor / shared -- each RAISES at construction.
  W3  THE SCHEDULES.  float32(_wh_value(n)) == the scorer's hold_value at every n in 0..50,001 for EARLY and LATE.
  W4  THE META-UPDATE, WITH SYNTHETIC z, for EARLY, LATE and test-only windows 0:150 / 150:end: an object from the
      UNPATCHED file (same BETA_HOLD and COMP_HOLD, no WINDOW_HOLD) and one from the PATCHED file are driven through the
      harness's own Lion_meta_update + clamp + _bh_apply + _ch_apply (+ _wh_apply) for 3,000 updates with the same z.  At
      every update: the WHOLE momentum_meta vector is BITWISE the unpatched object's; beta[0][0] (the complement) BITWISE
      the unpatched object's; beta[0][1] == the windowed schedule; _wh_nat == what PATCH_BETAHOLD wrote; _wh_active
      counts exactly the differing updates.  DEAD STATE: a second patched object on a DIFFERENT z stream holds
      BITWISE the same beta at every update.
  W5  RULE 20: every registered WINDOW_HOLD value is one token of [A-Za-z0-9_.:], no comma.

  python3 tests/test_windowhold.py --pre $WS/harness_cvt9/cifar10/Optimizers/HF.py.pre_windowhold \\
        --post $WS/harness_cvt9/cifar10/Optimizers/HF.py --cifar-dir $WS/harness_cvt9/cifar10
  python3 tests/test_windowhold.py --pre ... --post ... --structural-only          # W0 only, no torch
"""
import argparse, ast, contextlib, importlib.machinery, importlib.util, io, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAILED = []

G_APPLY = ("            # --- PATCH_WINDOWHOLD ---\n"
           "            if getattr(self, '_wh_on', False):\n"
           "                self._wh_apply()\n"
           "            # --- end PATCH_WINDOWHOLD ---\n")
CH_APPLY_END = "            # --- end PATCH_COMPHOLD ---\n"
G_ATTACH = ("        # --- PATCH_WINDOWHOLD ---\n"
            "        if getattr(self, '_wh_on', False):\n"
            "            self._wh_attach(rec)\n"
            "        # --- end PATCH_WINDOWHOLD ---\n")
CALL = "        self._wh_init(net_param_names_and_size)  # PATCH_WINDOWHOLD\n"
M_HEAD = "    # ------------------------------------------------------ PATCH_WINDOWHOLD\n"
M_TAIL = "    # -------------------------------------------------- end PATCH_WINDOWHOLD\n\n"
NEW_METHODS = ["_wh_init", "_wh_value", "_wh_set", "_wh_apply", "_wh_attach"]


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path, loader=importlib.machinery.SourceFileLoader(name, path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def structural(pre_src, post_src):
    print("=" * 78)
    print("W0  STRUCTURE: the two files differ ONLY by PATCH_WINDOWHOLD's four regions")
    chk("PATCH_WINDOWHOLD" not in pre_src and "WINDOW_HOLD" not in pre_src and "_wh_" not in pre_src,
        "--pre is UNPATCHED (no marker, no WINDOW_HOLD, no _wh_ name)")
    chk("PATCH_COMPHOLD" in pre_src and "PATCH_BETAHOLD" in pre_src and "PATCH_VOTEWEIGHT" in pre_src,
        "--pre is cvt6's PATCH_COMPHOLD HF.py (the base this patch sits on)")
    chk("PATCH_WINDOWHOLD" in post_src, "--post IS patched")
    s = post_src
    anchor1 = CH_APPLY_END + G_APPLY + "            self._probe(HtT_gradft)  # PATCH_PROBE\n"
    chk(s.count(G_APPLY) == 1 and s.count(anchor1) == 1,
        "region 1: ONE guarded _wh_apply, right after PATCH_COMPHOLD's _ch_apply region and before _probe")
    s = s.replace(G_APPLY, "")
    chk(s.count(CALL) == 1 and ("        self._ch_init(net_param_names_and_size)  # PATCH_COMPHOLD\n" + CALL) in s,
        "region 2: ONE _wh_init call, right after init_meta's _ch_init call")
    s = s.replace(CALL, "")
    anchor3 = ("        # --- end PATCH_COMPHOLD ---\n" + G_ATTACH + "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")
    chk(s.count(G_ATTACH) == 1 and anchor3 in s,
        "region 3: ONE guarded _wh_attach, after PATCH_COMPHOLD's _ch_attach region and before _pt_attach in _probe")
    s = s.replace(G_ATTACH, "")
    i0, i1 = s.find(M_HEAD), s.find(M_TAIL)
    chk(i0 >= 0 and i1 > i0 and s[i1 + len(M_TAIL):].startswith("    def check_required_attributes(")
        and s[:i0].endswith("    # -------------------------------------------------- end PATCH_COMPHOLD\n\n"),
        "region 4: the method block sits right after PATCH_COMPHOLD's block, immediately before check_required_attributes")
    if i0 >= 0 and i1 > i0:
        s = s[:i0] + s[i1 + len(M_TAIL):]
    chk(s == pre_src, "deleting the four inserted regions reproduces --pre BYTE FOR BYTE",
        "" if s == pre_src else "(%d vs %d bytes)" % (len(s), len(pre_src)))
    tree = ast.parse(post_src)
    hf = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HF"]
    meth = [f.name for f in hf[0].body if isinstance(f, ast.FunctionDef)] if hf else []
    chk([m for m in meth if m.startswith("_wh_")] == NEW_METHODS, "exactly five new methods, all `_wh_`-prefixed",
        str([m for m in meth if m.startswith("_wh_")]))
    ch = [i for i, m in enumerate(meth) if m.startswith("_ch_")]
    wh = [i for i, m in enumerate(meth) if m.startswith("_wh_")]
    chk(ch and wh and wh[0] == ch[-1] + 1, "the _wh_ methods follow PATCH_COMPHOLD's _ch_ methods directly")
    pre_meth = [f.name for f in [n for n in ast.parse(pre_src).body if isinstance(n, ast.ClassDef)
                                 and n.name == "HF"][0].body if isinstance(f, ast.FunctionDef)]
    chk([m for m in meth if not m.startswith("_wh_")] == pre_meth, "every pre-existing method is present, in order")


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


def make(HF, torch, nps, spec):
    o = HF.__new__(HF)
    o.num_layers = len(nps)
    o._device = torch.device("cpu")
    o.args_meta = {"alg": "Lion", "meta_stepsize": 1e-3, "momentum_param": 0.99, "Lion_beta2": 0.9, "weight_decay": 0}
    o._beta_lo, o._beta_hi = -15.0, -2.3026
    o._hier = "none"
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
    SC = _load("cvt9_scorer", os.path.join(REPO, "analysis", "cVT9_dosewindow_score.py"))
    tree = os.path.abspath(a.cifar_dir)
    sys.path.insert(0, tree)
    os.chdir(tree)
    from build_network import build_network
    net = build_network(SC.NET, "cpu")
    nps = [(n, p.data.size()) for n, p in net.named_parameters()]
    PRE = _load("hf_pre_windowhold", a.pre).HF
    POST = _load("hf_post_windowhold", a.post).HF

    def envs(arm):
        return dict(BETA_HOLD=SC.BETAHOLD[arm] or None, COMP_HOLD=SC.COMPHOLD[arm] or None,
                    WINDOW_HOLD=SC.WINDOWHOLD[arm] or None, VOTE_W=None)

    print("\n" + "=" * 78)
    print("W1  WITNESS AND INIT on the live %s (%d tensors)" % (SC.NET, len(nps)))
    chk([n for n, _s in nps] == SC.NAMES, "the live parameter names are the scorer's NAMES")
    for arm in SC.ARMS:
        with env(**envs(arm)):
            o, out = make(POST, torch, nps, SC.SPEC[arm])
        got = dict((k, [ln for ln in out if ln.startswith(k)]) for k in ("BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD", "VOTE_W"))
        chk(got["BETA_HOLD"] == [SC.WITNESS_BH[arm]] and got["COMP_HOLD"] == [SC.WITNESS_CH[arm]]
            and got["WINDOW_HOLD"] == [SC.WITNESS_WH[arm]] and got["VOTE_W"] == ["VOTE_W: off"],
            "W1 %-12s exactly one BETA_HOLD, COMP_HOLD and WINDOW_HOLD line == the scorer's witnesses; VOTE_W: off" % arm,
            (got["WINDOW_HOLD"] or ["(none)"])[0][:90])
        if arm in SC.HELD:
            want1 = SC.hold_value(arm, 0)
            chk(o._bh_on and o._ch_on and bool(o._wh_on) == (arm in SC.WINDOWED)
                and [float(x) for x in o.beta[0]] == [SC.hold_c_value(0), want1],
                "W1 %-12s _bh_on, _ch_on, _wh_on %s; initial beta [b0, %r]" % (arm, arm in SC.WINDOWED, want1),
                str([float(x) for x in o.beta[0]]))
    for arm in ("k01", "LOWHEADPATH", "HIGHHEADPATH", "MIDDOSE", "RESDOSE"):
        for val in (None, ""):
            e = envs(arm)
            e["WINDOW_HOLD"] = val
            with env(**e):
                o1, out1 = make(POST, torch, nps, SC.SPEC[arm])
                o0, _ = make(PRE, torch, nps, SC.SPEC[arm])
            lab = "unset" if val is None else "EMPTY"
            chk([ln for ln in out1 if ln.startswith("WINDOW_HOLD")] == ["WINDOW_HOLD: off"] and o1._wh_on is False,
                "W1 %-12s WINDOW_HOLD %s: prints `WINDOW_HOLD: off`, _wh_on False" % (arm, lab))
            same_attrs = sorted(k for k in vars(o1) if k != "_wh_on") == sorted(vars(o0))
            same_other = all((torch.equal(v, vars(o0)[k]) if torch.is_tensor(v) else True)
                             for k, v in vars(o1).items() if k != "_wh_on")
            chk(same_attrs and same_other and all(torch.equal(x, y) for x, y in zip(o1.beta, o0.beta))
                and getattr(o1, "param_groups_indices", None) == getattr(o0, "param_groups_indices", None),
                "W1 %-12s WINDOW_HOLD %s: every attribute and beta EQUAL to cvt6's object (only _wh_on added)" % (arm, lab))

    print("\n" + "=" * 78)
    print("W2  LOUDNESS: every malformed value or premise RAISES at construction")
    tri, floor = SC.BETAHOLD["EARLY"], SC.BETAHOLD["LOWHEADPATH"]
    bad = [("no colon", "9429", tri), ("empty n0", ":9429", tri), ("negative", "-1:9429", tri), ("n0 == n1", "9429:9429", tri),
           ("n0 > n1", "9430:9429", tri), ("9-digit n1", "0:123456789", tri), ("exponent", "0:1e4", tri), ("END upper case", "0:END", tri),
           ("trailing junk", "0:9429,x", tri), ("three fields", "0:9429:end", tri), ("start end", "end:9429", tri),
           ("BETA_HOLD off", "0:9429", None), ("BETA_HOLD floor", "0:9429", floor), ("BETA_HOLD shared", "0:9429", SC.NAME_HEAD + ":shared")]
    for lab, val, bhv in bad:
        raised, why = False, ""
        try:
            with env(BETA_HOLD=bhv, COMP_HOLD=None, WINDOW_HOLD=val, VOTE_W=None):
                make(POST, torch, nps, SC.HEADSPEC)
        except ValueError as ex:
            raised = "PATCH_WINDOWHOLD" in str(ex) or (bhv is not None and "PATCH_BETAHOLD" in str(ex))
            why = str(ex)[:60]
        chk(raised, "W2 %-22s RAISES" % lab, "%r %s" % (val, why))

    print("\n" + "=" * 78)
    print("W3  THE SCHEDULES: float32(_wh_value(n)) == the scorer's hold_value(n) for n = 0..50001")
    for arm in SC.WINDOWED:
        with env(**envs(arm)):
            o, _ = make(POST, torch, nps, SC.HEADSPEC)
        t = torch.zeros(1)
        mism = 0
        for k in range(0, 50002):
            t[0] = o._wh_value(k)
            mism += float(t[0]) != SC.hold_value(arm, k)
        chk(mism == 0, "W3 %-6s float32(_wh_value(n)) == hold_value(n) at every n" % arm, "%d mismatches" % mism)

    print("\n" + "=" * 78)
    print("W4  THE META-UPDATE with synthetic z: momentum and the complement untouched, 50 windowed, dead state")
    g = torch.Generator().manual_seed(249)
    g2 = torch.Generator().manual_seed(9249)
    arms = [("EARLY", "0:9429"), ("LATE", "9429:end"), ("WIN0_150", "0:150"), ("WIN150_END", "150:end")]
    for lab, wv in arms:
        base = dict(BETA_HOLD=tri, COMP_HOLD=SC.COMPHOLD["EARLY"], VOTE_W=None)
        with env(WINDOW_HOLD=None, **base):
            U, _ = make(PRE, torch, nps, SC.HEADSPEC)
        with env(WINDOW_HOLD=wv, **base):
            H, _ = make(POST, torch, nps, SC.HEADSPEC)
            H2, _ = make(POST, torch, nps, SC.HEADSPEC)
        n0, n1 = wv.split(":")
        n0, n1 = int(n0), (None if n1 == "end" else int(n1))
        bad_mom = bad_c = bad_hold = bad_nat = bad_dead = 0
        n_diff = 0
        n_expect = 0
        for t_ in range(1, a.updates + 1):
            if t_ == 1:
                z, z2 = torch.zeros(2), torch.zeros(2)
            else:
                z = torch.randn(2, generator=g) * torch.tensor([1.0, 3.0])
                z2 = torch.randn(2, generator=g2) * torch.tensor([5.0, 0.3])
            U.Lion_meta_update([z.clone()])
            H.Lion_meta_update([z.clone()])
            H2.Lion_meta_update([z2.clone()])
            for o in (U, H, H2):
                o.beta[0] = o.beta[0].clamp(o._beta_lo, o._beta_hi)
                o._bh_apply()
                o._ch_apply()
            wrote = float(H.beta[0][1])
            H._wh_apply()
            H2._wh_apply()
            bad_mom += not torch.equal(torch.as_tensor(H.momentum_meta[0]), torch.as_tensor(U.momentum_meta[0]))
            bad_c += float(H.beta[0][0]) != float(U.beta[0][0])
            inw = t_ >= n0 and (n1 is None or t_ < n1)
            want = SC.v_of(t_, SC.P_HIGH) if inw else SC.f32(SC.LO)
            bad_hold += float(H.beta[0][1]) != want
            bad_nat += H._wh_nat != wrote or wrote != SC.v_of(t_, SC.P_HIGH)
            n_diff += wrote != float(H.beta[0][1])
            n_expect += want != SC.v_of(t_, SC.P_HIGH)
            bad_dead += not torch.equal(H.beta[0], H2.beta[0])
        chk(bad_mom == 0, "W4 %-10s the WHOLE momentum_meta vector BITWISE the unpatched (cvt6) object's at all %d updates" % (lab, a.updates))
        chk(bad_c == 0, "W4 %-10s the complement's beta[0][0] BITWISE the unpatched object's (the window writes nothing on group 0)" % lab)
        chk(bad_hold == 0, "W4 %-10s beta[0][1] == the windowed schedule after every update" % lab, "%d bad" % bad_hold)
        chk(bad_nat == 0 and H._wh_n == a.updates and H._wh_active == n_diff == n_expect,
            "W4 %-10s _wh_nat == what PATCH_BETAHOLD wrote (tri); _wh_n == %d; _wh_active == %d == the updates the window "
            "moves off the triangle (EARLY: 0 before update 9429)" % (lab, a.updates, n_diff))
        chk(bad_dead == 0, "W4 %-10s DEAD STATE: a second object on a DIFFERENT z stream holds BITWISE the same beta" % lab)

    print("\n" + "=" * 78)
    print("W5  RULE 20")
    chk(all(re.match(r"^[A-Za-z0-9_.:]*$", SC.WINDOWHOLD[x]) and "," not in SC.WINDOWHOLD[x] for x in SC.ARMS),
        "every registered WINDOW_HOLD value is one token of [A-Za-z0-9_.:], no comma")

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %d" % len(FAILED)))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
