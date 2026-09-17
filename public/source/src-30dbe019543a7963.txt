"""Structure, loudness, witness, schedule and meta-update semantics of PATCH_COMPHOLD (`COMP_HOLD=...`), on the LIVE
PlainNet18_c100 model of the isolated `cvt6` tree.  CORRECTIONS 242.

STRATEGY (the test_betahold pattern): compare the PATCHED file against the UNPATCHED file (cvt4's HF.py), never against
my expectation of it; read every registered string, witness and schedule from the scorer.

  C0  STRUCTURE (no torch).  PATCH_COMPHOLD / COMP_HOLD / _ch_ absent from --pre, present in --post; the four inserted
      regions appear exactly as registered; deleting them from --post reproduces --pre BYTE FOR BYTE; exactly five new
      `_ch_` methods, right after PATCH_BETAHOLD's six; every pre-existing method present, in order.
  C1  WITNESS AND INIT (torch, CPU).  For each registered arm, HF.init_meta on the live model with its BETA_HOLD and
      COMP_HOLD prints exactly ONE `BETA_HOLD` line == the scorer's WITNESS_BH, exactly ONE `COMP_HOLD` line ==
      WITNESS_CH, and `VOTE_W: off`; the initial beta is [b0, b0] / [b0, -15] as registered.  COMP_HOLD unset and EMPTY
      print `COMP_HOLD: off`, set `_ch_on` False, and leave beta and every pre-existing attribute EQUAL to the
      unpatched (cvt4) object's, on k01, HEAD, HOLDLOW and HOLDHIGH.
  C2  THE GRAMMAR AND THE PREMISES ARE LOUD: malformed values, a missing / malformed / out-of-clamp replay file, BETA_HOLD
      off or `shared`, a 3-group grouping -- each RAISES at construction.
  C3  THE SCHEDULES.  The patch's own _ch_value(n), stored into float32, equals the scorer's hold_c_value at every n in
      0..50,001 for tri:9428 and rec:cvt6_headpath; the tree's replay file is byte-identical to the scorer's.
  C4  THE META-UPDATE, WITH SYNTHETIC z.  For each forced arm, and test-only tri:100 / rec:cvt6_testdown: an object from
      the UNPATCHED file (cvt4's HF.py, same BETA_HOLD, no COMP_HOLD) and one from the PATCHED file are driven through
      the harness's OWN Lion_meta_update + clamp + _bh_apply (+ _ch_apply) for 3,000 updates with the same random z
      (the first z is 0, as in a run).  At every update: the WHOLE momentum_meta vector is BITWISE the unpatched
      object's; beta[0][1] (50) is BITWISE the unpatched object's (COMP_HOLD writes nothing on group 1); beta[0][0] ==
      the registered schedule; _ch_nat == the value Lion + clamp wrote; _ch_active counts exactly the differing updates.
      DEAD STATE: a SECOND patched object driven by a DIFFERENT z stream holds BITWISE the same beta at every update.
  C5  RULE 20: every registered COMP_HOLD value is one token of [A-Za-z0-9_.:], no comma.

RUN (needs torch; on the cluster against the real isolated tree):
  python3 tests/test_comphold.py --pre  $WS/harness_cvt6/cifar10/Optimizers/HF.py.pre_comphold \\
        --post $WS/harness_cvt6/cifar10/Optimizers/HF.py --cifar-dir $WS/harness_cvt6/cifar10
  python3 tests/test_comphold.py --pre ... --post ... --structural-only          # C0 only, no torch
"""
import argparse, ast, contextlib, hashlib, importlib.machinery, importlib.util, io, json, os, re, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAILED = []

G_APPLY = ("            # --- PATCH_COMPHOLD ---\n"
           "            if getattr(self, '_ch_on', False):\n"
           "                self._ch_apply()\n"
           "            # --- end PATCH_COMPHOLD ---\n")
BH_APPLY_END = "            # --- end PATCH_BETAHOLD ---\n"
G_ATTACH = ("        # --- PATCH_COMPHOLD ---\n"
            "        if getattr(self, '_ch_on', False):\n"
            "            self._ch_attach(rec)\n"
            "        # --- end PATCH_COMPHOLD ---\n")
CALL = "        self._ch_init(net_param_names_and_size)  # PATCH_COMPHOLD\n"
M_HEAD = "    # ------------------------------------------------------ PATCH_COMPHOLD\n"
M_TAIL = "    # -------------------------------------------------- end PATCH_COMPHOLD\n\n"
NEW_METHODS = ["_ch_init", "_ch_value", "_ch_set", "_ch_apply", "_ch_attach"]


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
    print("C0  STRUCTURE: the two files differ ONLY by PATCH_COMPHOLD's four regions")
    chk("PATCH_COMPHOLD" not in pre_src and "COMP_HOLD" not in pre_src and "_ch_" not in pre_src,
        "--pre is UNPATCHED (no marker, no COMP_HOLD, no _ch_ name)")
    chk("PATCH_BETAHOLD" in pre_src and "PATCH_VOTEWEIGHT" in pre_src,
        "--pre is cvt4's PATCH_BETAHOLD HF.py (the base this patch sits on)")
    chk("PATCH_COMPHOLD" in post_src, "--post IS patched")
    s = post_src
    anchor1 = BH_APPLY_END + G_APPLY + "            self._probe(HtT_gradft)  # PATCH_PROBE\n"
    chk(s.count(G_APPLY) == 1 and s.count(anchor1) == 1,
        "region 1: ONE guarded _ch_apply, right after PATCH_BETAHOLD's _bh_apply region and before _probe")
    s = s.replace(G_APPLY, "")
    chk(s.count(CALL) == 1 and ("        self._bh_init(net_param_names_and_size)  # PATCH_BETAHOLD\n" + CALL) in s,
        "region 2: ONE _ch_init call, right after init_meta's _bh_init call")
    s = s.replace(CALL, "")
    anchor3 = ("        # --- end PATCH_BETAHOLD ---\n" + G_ATTACH + "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")
    chk(s.count(G_ATTACH) == 1 and anchor3 in s,
        "region 3: ONE guarded _ch_attach, after PATCH_BETAHOLD's _bh_attach region and before _pt_attach in _probe")
    s = s.replace(G_ATTACH, "")
    i0, i1 = s.find(M_HEAD), s.find(M_TAIL)
    chk(i0 >= 0 and i1 > i0 and s[i1 + len(M_TAIL):].startswith("    def check_required_attributes(")
        and s[:i0].endswith("    # -------------------------------------------------- end PATCH_BETAHOLD\n\n"),
        "region 4: the method block sits right after PATCH_BETAHOLD's block, immediately before check_required_attributes")
    if i0 >= 0 and i1 > i0:
        s = s[:i0] + s[i1 + len(M_TAIL):]
    chk(s == pre_src, "deleting the four inserted regions reproduces --pre BYTE FOR BYTE",
        "" if s == pre_src else "(%d vs %d bytes)" % (len(s), len(pre_src)))
    tree = ast.parse(post_src)
    hf = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HF"]
    meth = [f.name for f in hf[0].body if isinstance(f, ast.FunctionDef)] if hf else []
    chk([m for m in meth if m.startswith("_ch_")] == NEW_METHODS, "exactly five new methods, all `_ch_`-prefixed",
        str([m for m in meth if m.startswith("_ch_")]))
    bh = [i for i, m in enumerate(meth) if m.startswith("_bh_")]
    ch = [i for i, m in enumerate(meth) if m.startswith("_ch_")]
    chk(bh and ch and ch[0] == bh[-1] + 1, "the _ch_ methods follow PATCH_BETAHOLD's _bh_ methods directly")
    pre_meth = [f.name for f in [n for n in ast.parse(pre_src).body if isinstance(n, ast.ClassDef)
                                 and n.name == "HF"][0].body if isinstance(f, ast.FunctionDef)]
    chk([m for m in meth if not m.startswith("_ch_")] == pre_meth, "every pre-existing method is present, in order")


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
    SC = _load("cvt6_scorer", os.path.join(REPO, "analysis", "cVT6_complementpath_score.py"))
    tree = os.path.abspath(a.cifar_dir)
    sys.path.insert(0, tree)
    os.chdir(tree)
    from build_network import build_network
    net = build_network(SC.NET, "cpu")
    nps = [(n, p.data.size()) for n, p in net.named_parameters()]
    PRE = _load("hf_pre_comphold", a.pre).HF
    POST = _load("hf_post_comphold", a.post).HF
    optdir = os.path.dirname(os.path.abspath(a.post))

    print("\n" + "=" * 78)
    print("C1  WITNESS AND INIT on the live %s (%d tensors)" % (SC.NET, len(nps)))
    chk([n for n, _s in nps] == SC.NAMES, "the live parameter names are the scorer's NAMES")
    for arm in SC.ARMS:
        with env(BETA_HOLD=SC.BETAHOLD[arm] or None, COMP_HOLD=SC.COMPHOLD[arm] or None, VOTE_W=None):
            o, out = make(POST, torch, nps, SC.SPEC[arm])
        bh = [ln for ln in out if ln.startswith("BETA_HOLD")]
        ch = [ln for ln in out if ln.startswith("COMP_HOLD")]
        vw = [ln for ln in out if ln.startswith("VOTE_W")]
        chk(bh == [SC.WITNESS_BH[arm]] and ch == [SC.WITNESS_CH[arm]] and vw == ["VOTE_W: off"],
            "C1 %-12s exactly one BETA_HOLD and one COMP_HOLD line == the scorer's witnesses; VOTE_W: off" % arm,
            (ch or ["(none)"])[0][:90])
        if arm in SC.HELD:
            want1 = SC.LO if SC.MODE[arm][0] == "floor" else SC.B0
            chk(o._bh_on and o._bh_g == 1 and [float(x) for x in o.beta[0]] == [SC.B0, want1]
                and bool(o._ch_on) == (arm in SC.FORCED) and (arm not in SC.FORCED or o._ch_g == 0),
                "C1 %-12s _bh_on, group 1; _ch_on %s%s; initial beta [b0, %r]" % (
                    arm, arm in SC.FORCED, ", group 0" if arm in SC.FORCED else "", want1), str([float(x) for x in o.beta[0]]))
    for spec_arm in ("k01", "HEAD", "HOLDLOW", "HOLDHIGH"):
        for val in (None, ""):
            with env(BETA_HOLD=SC.BETAHOLD[spec_arm] or None, COMP_HOLD=val, VOTE_W=None):
                o1, out1 = make(POST, torch, nps, SC.SPEC[spec_arm])
                o0, _out0 = make(PRE, torch, nps, SC.SPEC[spec_arm])
            lab = "unset" if val is None else "EMPTY"
            chk([ln for ln in out1 if ln.startswith("COMP_HOLD")] == ["COMP_HOLD: off"] and o1._ch_on is False,
                "C1 %-8s COMP_HOLD %s: prints `COMP_HOLD: off`, _ch_on False" % (spec_arm, lab))
            same_attrs = sorted(k for k in vars(o1) if k != "_ch_on") == sorted(vars(o0))
            same_other = all((torch.equal(v, vars(o0)[k]) if torch.is_tensor(v) else True)
                             for k, v in vars(o1).items() if k != "_ch_on")
            chk(same_attrs and same_other and all(torch.equal(x, y) for x, y in zip(o1.beta, o0.beta))
                and getattr(o1, "param_groups_indices", None) == getattr(o0, "param_groups_indices", None),
                "C1 %-8s COMP_HOLD %s: every attribute and beta EQUAL to cvt4's object (only _ch_on added)" % (spec_arm, lab))

    print("\n" + "=" * 78)
    print("C2  LOUDNESS: every malformed value, file, spec or environment RAISES at construction")
    tmpd = tempfile.mkdtemp(prefix="cvt6_c2_")
    bad_files = {"cvt6_zz_badjson": "{not json", "cvt6_zz_decreasing": json.dumps({"id": "cvt6_zz_decreasing", "n": [2, 1], "v": [-14.0, -14.0]}),
                 "cvt6_zz_outside": json.dumps({"id": "cvt6_zz_outside", "n": [2, 102], "v": [-14.0, -1.0]}),
                 "cvt6_zz_wrongid": json.dumps({"id": "something_else", "n": [2, 102], "v": [-14.0, -14.5]}),
                 "cvt6_zz_onepoint": json.dumps({"id": "cvt6_zz_onepoint", "n": [2], "v": [-14.0]}),
                 "cvt6_zz_floatn": json.dumps({"id": "cvt6_zz_floatn", "n": [2.0, 102.0], "v": [-14.0, -14.5]}),
                 "cvt6_zz_nan": '{"id": "cvt6_zz_nan", "n": [2, 102], "v": [-14.0, NaN]}'}
    # the bad replay files live in a TEMP COPY of the patched HF.py's directory -- the tree is never written
    shutil.copy(os.path.abspath(a.post), os.path.join(tmpd, "HF.py"))
    for k, body in bad_files.items():
        open(os.path.join(tmpd, k + ".json"), "w").write(body)
    POSTC2 = _load("hf_post_comphold_c2", os.path.join(tmpd, "HF.py")).HF
    try:
        hs = SC.HEADSPEC
        floor, tri = SC.BETAHOLD["HOLDLOW"], SC.BETAHOLD["HOLDHIGH"]
        bad = [
            ("no mode", "9428", floor, hs), ("tri without P", "tri", floor, hs), ("negative P", "tri:-5", floor, hs),
            ("P = 0", "tri:0", floor, hs), ("7-digit P", "tri:1234567", floor, hs), ("exponent P", "tri:1e3", floor, hs),
            ("unknown mode", "floor", floor, hs), ("trailing junk", "tri:9428,x", floor, hs),
            ("rec id with a dot", "rec:cvt6_headpath.json", floor, hs), ("rec id with a slash", "rec:../x", floor, hs),
            ("rec file missing", "rec:cvt6_no_such_file", floor, hs), ("rec file not json", "rec:cvt6_zz_badjson", floor, hs),
            ("rec n decreasing", "rec:cvt6_zz_decreasing", floor, hs), ("rec v outside the clamp", "rec:cvt6_zz_outside", floor, hs),
            ("rec id mismatch", "rec:cvt6_zz_wrongid", floor, hs), ("rec one knot", "rec:cvt6_zz_onepoint", floor, hs),
            ("rec float n", "rec:cvt6_zz_floatn", floor, hs), ("rec NaN knot", "rec:cvt6_zz_nan", floor, hs),
            ("BETA_HOLD off", "tri:9428", None, hs), ("BETA_HOLD shared", "tri:9428", SC.NAME_HEAD + ":shared", hs),
            ("3 groups", "tri:9428", floor, "sets:1-48,51-53/49/layer4.1.bn2.weight"),
            ("scalar spec with BETA_HOLD off", "tri:9428", None, "scalar"),
        ]
        shutil.copy(os.path.join(optdir, SC.HEADPATH_ID + ".json"), os.path.join(tmpd, SC.HEADPATH_ID + ".json"))
        with env(BETA_HOLD=floor, COMP_HOLD=SC.COMPHOLD["LOWHEADPATH"], VOTE_W=None):
            _o, _out = make(POSTC2, torch, nps, hs)
        chk([ln for ln in _out if ln.startswith("COMP_HOLD")] == [SC.WITNESS_CH["LOWHEADPATH"]],
            "C2 control: the temp copy of HF.py accepts the registered replay file (so each raise below is the bad input)")
        for lab, val, bhv, spec in bad:
            raised, why = False, ""
            try:
                with env(BETA_HOLD=bhv, COMP_HOLD=val, VOTE_W=None):
                    make(POSTC2, torch, nps, spec)
            except ValueError as e:
                raised = "PATCH_COMPHOLD" in str(e) or (bhv is not None and "PATCH_BETAHOLD" in str(e))
                why = str(e)[:60]
            chk(raised, "C2 %-30s RAISES" % lab, "%r %s" % (val, why))
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    print("\n" + "=" * 78)
    print("C3  THE SCHEDULES: the patch's _ch_value, stored into float32, == the scorer's hold_c_value at every update")
    tree_json = os.path.join(optdir, SC.HEADPATH_ID + ".json")
    chk(os.path.exists(tree_json) and open(tree_json, "rb").read() == open(SC.HEADPATH_FILE, "rb").read()
        and hashlib.sha256(open(tree_json, "rb").read()).hexdigest() == SC.HEADPATH_SHA,
        "the tree's %s.json is BYTE-IDENTICAL to the repo's (sha HEADPATH_SHA)" % SC.HEADPATH_ID)
    for arm in ("LOWMUTEPATH", "LOWHEADPATH", "HIGHHEADPATH"):
        with env(BETA_HOLD=SC.BETAHOLD[arm], COMP_HOLD=SC.COMPHOLD[arm], VOTE_W=None):
            o, _ = make(POST, torch, nps, SC.HEADSPEC)
        mism = 0
        t = torch.zeros(1)
        for k in range(0, 50002):
            t[0] = o._ch_value(k)
            mism += float(t[0]) != SC.hold_c_value(arm, k)
        chk(mism == 0, "C3 %-12s float32(_ch_value(n)) == scorer hold_c_value(n) for n = 0..50001" % arm, "%d mismatches" % mism)

    print("\n" + "=" * 78)
    print("C4  THE META-UPDATE with synthetic z: momentum and 50 untouched, the complement held, dead state")
    g = torch.Generator().manual_seed(242)
    g2 = torch.Generator().manual_seed(9242)
    arms = [(x, SC.BETAHOLD[x], SC.COMPHOLD[x]) for x in SC.FORCED]
    arms += [("TRI100", SC.BETAHOLD["HOLDLOW"], "tri:100"), ("TESTDOWN", SC.BETAHOLD["HOLDHIGH"], "rec:" + SC.TESTDOWN_ID)]
    for lab, bhv, chv in arms:
        with env(BETA_HOLD=bhv, COMP_HOLD=None, VOTE_W=None):
            U, _ = make(PRE, torch, nps, SC.HEADSPEC)
        with env(BETA_HOLD=bhv, COMP_HOLD=chv, VOTE_W=None):
            H, _ = make(POST, torch, nps, SC.HEADSPEC)
            H2, _ = make(POST, torch, nps, SC.HEADSPEC)
        for o in (U, H, H2):
            o.meta_update = o.Lion_meta_update
        bad_mom = bad_50 = bad_hold = bad_nat = bad_dead = 0
        n_diff = 0
        for t_ in range(1, a.updates + 1):
            if t_ == 1:
                z = torch.zeros(2)
                z2 = torch.zeros(2)
            else:
                z = torch.randn(2, generator=g) * torch.tensor([1.0, 3.0])
                z2 = torch.randn(2, generator=g2) * torch.tensor([5.0, 0.3])
            U.Lion_meta_update([z.clone()])
            H.Lion_meta_update([z.clone()])
            H2.Lion_meta_update([z2.clone()])
            for o in (U, H, H2):
                o.beta[0] = o.beta[0].clamp(o._beta_lo, o._beta_hi)
                o._bh_apply()
            wrote = float(H.beta[0][0])
            H._ch_apply()
            H2._ch_apply()
            bad_mom += not torch.equal(torch.as_tensor(H.momentum_meta[0]), torch.as_tensor(U.momentum_meta[0]))
            bad_50 += float(H.beta[0][1]) != float(U.beta[0][1])
            if chv.startswith("tri:"):
                want = SC.v_of(t_, int(chv.split(":")[1]))
            else:
                K = json.load(open(os.path.join(optdir, chv.split(":")[1] + ".json")))
                want = SC.rec_value(t_, K["n"], K["v"]) if lab == "TESTDOWN" else SC.hold_c_value(lab, t_)
            bad_hold += float(H.beta[0][0]) != want
            bad_nat += H._ch_nat != wrote
            n_diff += wrote != float(H.beta[0][0])
            bad_dead += not torch.equal(H.beta[0], H2.beta[0])
        chk(bad_mom == 0, "C4 %-12s the WHOLE momentum_meta vector BITWISE the unpatched (cvt4) object's at all %d updates" % (lab, a.updates))
        chk(bad_50 == 0, "C4 %-12s 50's beta[0][1] BITWISE the unpatched object's (COMP_HOLD writes nothing on group 1)" % lab)
        chk(bad_hold == 0, "C4 %-12s beta[0][0] == the registered schedule after every update" % lab, "%d bad" % bad_hold)
        chk(bad_nat == 0 and H._ch_n == a.updates and H._ch_active == n_diff and n_diff > 0,
            "C4 %-12s _ch_nat == what Lion wrote; _ch_n == %d; _ch_active == %d updates where it differed (> 0)"
            % (lab, a.updates, n_diff))
        chk(bad_dead == 0, "C4 %-12s DEAD STATE: a second object driven by a DIFFERENT z stream holds BITWISE the same beta "
                           "at every update" % lab)

    print("\n" + "=" * 78)
    print("C5  RULE 20")
    chk(all(re.match(r"^[A-Za-z0-9_.:]*$", SC.COMPHOLD[x]) and "," not in SC.COMPHOLD[x] for x in SC.ARMS),
        "every registered COMP_HOLD value is one token of [A-Za-z0-9_.:], no comma")

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %d" % len(FAILED)))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
