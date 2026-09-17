"""Structure, loudness, witness, schedule and meta-update semantics of PATCH_RESTHOLD (`REST_HOLD=...`), on the LIVE
ResNet18_c100 model of the isolated `cvt8` tree.  CORRECTIONS 248.

STRATEGY (the test_betahold / test_comphold pattern): compare the PATCHED file against the UNPATCHED file (cvt7's HF.py),
never against my expectation of it; read every registered string, witness and schedule from the scorer.

  R0  STRUCTURE (no torch).  PATCH_RESTHOLD / REST_HOLD / _rh_ absent from --pre, present in --post; the four inserted
      regions appear exactly as registered; deleting them from --post reproduces --pre BYTE FOR BYTE; exactly five new
      `_rh_` methods, right after PATCH_GROUPHOLD's five; every pre-existing method present, in order.
  R1  WITNESS AND INIT (torch, CPU).  For each registered arm, HF.init_meta on the live model with its GROUP_HOLD and
      REST_HOLD prints exactly ONE GROUP_HOLD line == the scorer's WITNESS_GH, ONE REST_HOLD line == WITNESS_RH,
      `BETA_HOLD: off` and `VOTE_W: off`; the initial beta is as registered.  REST_HOLD unset and EMPTY print
      `REST_HOLD: off`, set `_rh_on` False, and leave beta and every pre-existing attribute EQUAL to the unpatched (cvt7)
      object's, on k01, ISO, and GROUP_HOLD floor / tri:8609 / tri:9428.
  R2  LOUDNESS: malformed values, a missing / malformed / out-of-clamp / wrong-id replay file, GROUP_HOLD off, a 4-group
      grouping -- each RAISES at construction (bad files in a TEMP COPY of HF.py's directory; the tree is never written).
  R3  THE SCHEDULES.  The patch's own _rh_value(n), stored into float32, equals the scorer's hold_r_value at every n in
      0..50,001 for the three forced arms; the tree's replay file is byte-identical to the repo's.
  R4  THE META-UPDATE, WITH SYNTHETIC z.  For each forced arm, and test-only tri:100 / rec:cvt8_testdown: an object from
      the UNPATCHED file (cvt7's HF.py, same GROUP_HOLD, no REST_HOLD) and one from the PATCHED file are driven through the
      harness's OWN Lion_meta_update + clamp + _gh_apply (+ _rh_apply) for 3,000 updates with the same random z.  At every
      update: the WHOLE momentum_meta vector is BITWISE the unpatched object's; the carrier group's beta[0][1] is BITWISE
      the unpatched object's; beta[0][0] == the registered schedule; _rh_nat == what Lion + clamp wrote; _rh_active counts
      exactly the differing updates.  DEAD STATE: a SECOND patched object driven by a DIFFERENT z stream holds BITWISE the
      same beta at every update.
  R5  RULE 20 and the reader prefixes.

RUN (needs torch):
  python3 tests/test_resthold.py --pre $WS/harness_cvt8/cifar10/Optimizers/HF.py.pre_resthold \\
        --post $WS/harness_cvt8/cifar10/Optimizers/HF.py --cifar-dir $WS/harness_cvt8/cifar10
  python3 tests/test_resthold.py --pre ... --post ... --structural-only          # R0 only, no torch
"""
import argparse, ast, contextlib, hashlib, importlib.machinery, importlib.util, io, json, os, re, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAILED = []

R_APPLY = ("            # --- PATCH_RESTHOLD ---\n"
           "            if getattr(self, '_rh_on', False):\n"
           "                self._rh_apply()\n"
           "            # --- end PATCH_RESTHOLD ---\n")
GH_APPLY_END = "                self._gh_apply()\n            # --- end PATCH_GROUPHOLD ---\n"
R_ATTACH = ("        # --- PATCH_RESTHOLD ---\n"
            "        if getattr(self, '_rh_on', False):\n"
            "            self._rh_attach(rec)\n"
            "        # --- end PATCH_RESTHOLD ---\n")
GH_ATTACH_END = "            self._gh_attach(rec)\n        # --- end PATCH_GROUPHOLD ---\n"
CALL = "        self._rh_init(net_param_names_and_size)  # PATCH_RESTHOLD\n"
M_HEAD = "    # ---------------------------------------------------- PATCH_RESTHOLD\n"
M_TAIL = "    # ------------------------------------------------ end PATCH_RESTHOLD\n\n"
NEW_METHODS = ["_rh_init", "_rh_value", "_rh_set", "_rh_apply", "_rh_attach"]


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
    print("R0  STRUCTURE: the two files differ ONLY by PATCH_RESTHOLD's four regions")
    chk("PATCH_RESTHOLD" not in pre_src and "REST_HOLD" not in pre_src and "_rh_" not in pre_src,
        "--pre is UNPATCHED (no marker, no REST_HOLD, no _rh_ name)")
    chk("PATCH_GROUPHOLD" in pre_src and "PATCH_BETAHOLD" in pre_src and "PATCH_VOTEWEIGHT" in pre_src and "COMP_HOLD" not in pre_src,
        "--pre is cvt7's PATCH_GROUPHOLD HF.py (the base this patch sits on), with no PATCH_COMPHOLD")
    chk("PATCH_RESTHOLD" in post_src, "--post IS patched")
    s = post_src
    anchor1 = GH_APPLY_END + R_APPLY + "            self._probe(HtT_gradft)  # PATCH_PROBE\n"
    chk(s.count(R_APPLY) == 1 and s.count(anchor1) == 1,
        "region 1: ONE guarded _rh_apply, right after PATCH_GROUPHOLD's _gh_apply region and before _probe")
    s = s.replace(R_APPLY, "")
    chk(s.count(CALL) == 1 and ("        self._gh_init(net_param_names_and_size)  # PATCH_GROUPHOLD\n" + CALL) in s,
        "region 2: ONE _rh_init call, right after init_meta's _gh_init call")
    s = s.replace(CALL, "")
    chk(s.count(R_ATTACH) == 1 and (GH_ATTACH_END + R_ATTACH + "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n") in s,
        "region 3: ONE guarded _rh_attach, after PATCH_GROUPHOLD's _gh_attach region and before _pt_attach in _probe")
    s = s.replace(R_ATTACH, "")
    i0, i1 = s.find(M_HEAD), s.find(M_TAIL)
    chk(i0 >= 0 and i1 > i0 and s[i1 + len(M_TAIL):].startswith("    def check_required_attributes(")
        and s[:i0].endswith("    # ----------------------------------------------- end PATCH_GROUPHOLD\n\n"),
        "region 4: the method block sits right after PATCH_GROUPHOLD's block, immediately before check_required_attributes")
    if i0 >= 0 and i1 > i0:
        s = s[:i0] + s[i1 + len(M_TAIL):]
    chk(s == pre_src, "deleting the four inserted regions reproduces --pre BYTE FOR BYTE",
        "" if s == pre_src else "(%d vs %d bytes)" % (len(s), len(pre_src)))
    tree = ast.parse(post_src)
    hf = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HF"]
    meth = [f.name for f in hf[0].body if isinstance(f, ast.FunctionDef)] if hf else []
    chk([m for m in meth if m.startswith("_rh_")] == NEW_METHODS, "exactly five new methods, all `_rh_`-prefixed",
        str([m for m in meth if m.startswith("_rh_")]))
    gh = [i for i, m in enumerate(meth) if m.startswith("_gh_")]
    rh = [i for i, m in enumerate(meth) if m.startswith("_rh_")]
    chk(gh and rh and rh[0] == gh[-1] + 1, "the _rh_ methods follow PATCH_GROUPHOLD's _gh_ methods directly")
    pre_meth = [f.name for f in [n for n in ast.parse(pre_src).body if isinstance(n, ast.ClassDef)
                                 and n.name == "HF"][0].body if isinstance(f, ast.FunctionDef)]
    chk([m for m in meth if not m.startswith("_rh_")] == pre_meth, "every pre-existing method is present, in order")


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
    SC = _load("cvt8_scorer", os.path.join(REPO, "analysis", "cVT8_doseroute_score.py"))
    tree = os.path.abspath(a.cifar_dir)
    sys.path.insert(0, tree)
    os.chdir(tree)
    from build_network import build_network
    net = build_network(SC.NET, "cpu")
    nps = [(n, p.data.size()) for n, p in net.named_parameters()]
    PRE = _load("hf_pre_resthold", a.pre).HF
    POST = _load("hf_post_resthold", a.post).HF
    optdir = os.path.dirname(os.path.abspath(a.post))
    CLEAN = dict(VOTE_W=None, BETA_HOLD=None, COMP_HOLD=None, GROUP_HOLD=None, REST_HOLD=None)

    print("\n" + "=" * 78)
    print("R1  WITNESS AND INIT on the live %s (%d tensors)" % (SC.NET, len(nps)))
    chk([n for n, _s in nps] == SC.NAMES, "the live parameter names are the scorer's NAMES")
    for arm in SC.ARMS:
        with env(**dict(CLEAN, GROUP_HOLD=SC.GROUPHOLD[arm] or None, REST_HOLD=SC.RESTHOLD[arm] or None)):
            o, out = make(POST, torch, nps, SC.SPEC[arm])
        gh = [ln for ln in out if ln.startswith("GROUP_HOLD")]
        rh = [ln for ln in out if ln.startswith("REST_HOLD")]
        bh = [ln for ln in out if ln.startswith("BETA_HOLD")]
        vw = [ln for ln in out if ln.startswith("VOTE_W")]
        chk(gh == [SC.WITNESS_GH[arm]] and rh == [SC.WITNESS_RH[arm]] and bh == ["BETA_HOLD: off"] and vw == ["VOTE_W: off"]
            and not [ln for ln in out if ln.startswith("COMP_HOLD")],
            "R1 %-12s exactly one GROUP_HOLD and one REST_HOLD line == the scorer's witnesses; BETA_HOLD: off; VOTE_W: off" % arm,
            (rh or ["(none)"])[0][:90])
        if arm in SC.HELD:
            want1 = SC.LO if SC.MODE[arm][0] == "floor" else SC.B0
            want0 = SC.hold_r_value(arm, 0) if arm in SC.FORCED else SC.B0
            chk(o._gh_on and o._gh_g == 1 and [float(x) for x in o.beta[0]] == [want0, want1]
                and bool(o._rh_on) == (arm in SC.FORCED) and (arm not in SC.FORCED or o._rh_g == 0),
                "R1 %-12s _gh_on group 1; _rh_on %s; initial beta [%r, %r]" % (arm, arm in SC.FORCED, want0, want1),
                str([float(x) for x in o.beta[0]]))
    for lab_arm, spec, ghv in (("k01", "scalar", None), ("ISO", SC.ISOSPEC, None), ("floor", SC.ISOSPEC, SC.GROUPHOLD["LOWISOPATH"]),
                               ("tri8609", SC.ISOSPEC, SC.GROUPHOLD["HOLDHIGH"]), ("tri9428", SC.ISOSPEC, SC.GROUPHOLD["HOLDBIG"])):
        for val in (None, ""):
            with env(**dict(CLEAN, GROUP_HOLD=ghv, REST_HOLD=val)):
                o1, out1 = make(POST, torch, nps, spec)
                o0, _out0 = make(PRE, torch, nps, spec)
            lab = "unset" if val is None else "EMPTY"
            chk([ln for ln in out1 if ln.startswith("REST_HOLD")] == ["REST_HOLD: off"] and o1._rh_on is False,
                "R1 %-8s REST_HOLD %s: prints `REST_HOLD: off`, _rh_on False" % (lab_arm, lab))
            same_attrs = sorted(k for k in vars(o1) if k != "_rh_on") == sorted(vars(o0))
            same_other = all((torch.equal(v, vars(o0)[k]) if torch.is_tensor(v) else True) for k, v in vars(o1).items() if k != "_rh_on")
            chk(same_attrs and same_other and all(torch.equal(x, y) for x, y in zip(o1.beta, o0.beta))
                and getattr(o1, "param_groups_indices", None) == getattr(o0, "param_groups_indices", None),
                "R1 %-8s REST_HOLD %s: every attribute and beta EQUAL to cvt7's object (only _rh_on added)" % (lab_arm, lab))

    print("\n" + "=" * 78)
    print("R2  LOUDNESS: every malformed value, file, spec or environment RAISES at construction")
    tmpd = tempfile.mkdtemp(prefix="cvt8_r2_")
    bad_files = {"cvt8_zz_badjson": "{not json", "cvt8_zz_decreasing": json.dumps({"id": "cvt8_zz_decreasing", "n": [2, 1], "v": [-14.0, -14.0]}),
                 "cvt8_zz_outside": json.dumps({"id": "cvt8_zz_outside", "n": [2, 102], "v": [-14.0, -1.0]}),
                 "cvt8_zz_wrongid": json.dumps({"id": "something_else", "n": [2, 102], "v": [-14.0, -14.5]}),
                 "cvt8_zz_onepoint": json.dumps({"id": "cvt8_zz_onepoint", "n": [2], "v": [-14.0]}),
                 "cvt8_zz_floatn": json.dumps({"id": "cvt8_zz_floatn", "n": [2.0, 102.0], "v": [-14.0, -14.5]}),
                 "cvt8_zz_nan": '{"id": "cvt8_zz_nan", "n": [2, 102], "v": [-14.0, NaN]}'}
    shutil.copy(os.path.abspath(a.post), os.path.join(tmpd, "HF.py"))
    for k, body in bad_files.items():
        open(os.path.join(tmpd, k + ".json"), "w").write(body)
    POSTR2 = _load("hf_post_resthold_r2", os.path.join(tmpd, "HF.py")).HF
    try:
        iso = SC.ISOSPEC
        fl, tri = SC.GROUPHOLD["LOWISOPATH"], SC.GROUPHOLD["HOLDBIG"]
        bad = [
            ("no mode", "9428", fl, iso), ("tri without P", "tri", fl, iso), ("negative P", "tri:-5", fl, iso),
            ("P = 0", "tri:0", fl, iso), ("7-digit P", "tri:1234567", fl, iso), ("exponent P", "tri:1e3", fl, iso),
            ("unknown mode", "floor", fl, iso), ("trailing junk", "rec:cvt8_isopath,x", fl, iso),
            ("rec id with a dot", "rec:cvt8_isopath.json", fl, iso), ("rec id with a slash", "rec:../x", fl, iso),
            ("rec file missing", "rec:cvt8_no_such_file", fl, iso), ("rec file not json", "rec:cvt8_zz_badjson", fl, iso),
            ("rec n decreasing", "rec:cvt8_zz_decreasing", fl, iso), ("rec v outside the clamp", "rec:cvt8_zz_outside", fl, iso),
            ("rec id mismatch", "rec:cvt8_zz_wrongid", fl, iso), ("rec one knot", "rec:cvt8_zz_onepoint", fl, iso),
            ("rec float n", "rec:cvt8_zz_floatn", fl, iso), ("rec NaN knot", "rec:cvt8_zz_nan", fl, iso),
            ("GROUP_HOLD off", "rec:cvt8_isopath", None, iso), ("GROUP_HOLD off, tri", "tri:9428", None, iso),
            ("scalar spec with GROUP_HOLD off", "rec:cvt8_isopath", None, "scalar"),
            ("4 groups", "tri:9428", tri, "sets:1-24/25-49,51-52/54-58,60-62/layer4.0.bn2.weight,layer4.0.shortcut.1.weight,layer4.1.bn2.weight"),
        ]
        shutil.copy(os.path.join(optdir, SC.ISOPATH_ID + ".json"), os.path.join(tmpd, SC.ISOPATH_ID + ".json"))
        with env(**dict(CLEAN, GROUP_HOLD=fl, REST_HOLD=SC.RESTHOLD["LOWISOPATH"])):
            _o, _out = make(POSTR2, torch, nps, iso)
        chk([ln for ln in _out if ln.startswith("REST_HOLD")] == [SC.WITNESS_RH["LOWISOPATH"]],
            "R2 control: the temp copy of HF.py accepts the registered replay file (so each raise below is the bad input)")
        for lab, val, ghv, spec in bad:
            raised, why = False, ""
            try:
                with env(**dict(CLEAN, GROUP_HOLD=ghv, REST_HOLD=val)):
                    make(POSTR2, torch, nps, spec)
            except ValueError as e:
                raised = "PATCH_RESTHOLD" in str(e) or (ghv is not None and "PATCH_GROUPHOLD" in str(e))
                why = str(e)[:60]
            chk(raised, "R2 %-32s RAISES" % lab, "%r %s" % (val, why))
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

    print("\n" + "=" * 78)
    print("R3  THE SCHEDULES: the patch's _rh_value, stored into float32, == the scorer's hold_r_value at every update")
    tree_json = os.path.join(optdir, SC.ISOPATH_ID + ".json")
    chk(os.path.exists(tree_json) and open(tree_json, "rb").read() == open(SC.ISOPATH_FILE, "rb").read()
        and hashlib.sha256(open(tree_json, "rb").read()).hexdigest() == SC.ISOPATH_SHA,
        "the tree's %s.json is BYTE-IDENTICAL to the repo's (sha ISOPATH_SHA)" % SC.ISOPATH_ID)
    for arm in SC.FORCED:
        with env(**dict(CLEAN, GROUP_HOLD=SC.GROUPHOLD[arm], REST_HOLD=SC.RESTHOLD[arm])):
            o, _ = make(POST, torch, nps, SC.ISOSPEC)
        mism = 0
        t = torch.zeros(1)
        for k in range(0, 50002):
            t[0] = o._rh_value(k)
            mism += float(t[0]) != SC.hold_r_value(arm, k)
        chk(mism == 0, "R3 %-12s float32(_rh_value(n)) == scorer hold_r_value(n) for n = 0..50001" % arm, "%d mismatches" % mism)

    print("\n" + "=" * 78)
    print("R4  THE META-UPDATE with synthetic z: momentum and the carrier group untouched, the complement held, dead state")
    g = torch.Generator().manual_seed(248)
    g2 = torch.Generator().manual_seed(9248)
    arms = [(x, SC.GROUPHOLD[x], SC.RESTHOLD[x]) for x in SC.FORCED]
    arms += [("TRI100", SC.GROUPHOLD["LOWISOPATH"], "tri:100"), ("TESTDOWN", SC.GROUPHOLD["HOLDBIG"], "rec:" + SC.TESTDOWN_ID)]
    for lab, ghv, rhv in arms:
        with env(**dict(CLEAN, GROUP_HOLD=ghv)):
            U, _ = make(PRE, torch, nps, SC.ISOSPEC)
        with env(**dict(CLEAN, GROUP_HOLD=ghv, REST_HOLD=rhv)):
            H, _ = make(POST, torch, nps, SC.ISOSPEC)
            H2, _ = make(POST, torch, nps, SC.ISOSPEC)
        bad_mom = bad_car = bad_hold = bad_nat = bad_dead = 0
        n_diff = 0
        K = json.load(open(os.path.join(optdir, rhv.split(":")[1] + ".json"))) if rhv.startswith("rec:") else None
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
                o._gh_apply()
            wrote = float(H.beta[0][0])
            H._rh_apply()
            H2._rh_apply()
            bad_mom += not torch.equal(torch.as_tensor(H.momentum_meta[0]), torch.as_tensor(U.momentum_meta[0]))
            bad_car += float(H.beta[0][1]) != float(U.beta[0][1])
            want = SC.v_of(t_, 100) if rhv == "tri:100" else SC.rec_value(t_, K["n"], K["v"])
            bad_hold += float(H.beta[0][0]) != want
            bad_nat += H._rh_nat != wrote
            n_diff += wrote != float(H.beta[0][0])
            bad_dead += not torch.equal(H.beta[0], H2.beta[0])
        chk(bad_mom == 0, "R4 %-12s the WHOLE momentum_meta vector BITWISE the unpatched (cvt7) object's at all %d updates" % (lab, a.updates))
        chk(bad_car == 0, "R4 %-12s the carrier group's beta[0][1] BITWISE the unpatched object's (REST_HOLD writes nothing on group 1)" % lab)
        chk(bad_hold == 0, "R4 %-12s beta[0][0] == the registered schedule after every update" % lab, "%d bad" % bad_hold)
        chk(bad_nat == 0 and H._rh_n == a.updates and H._rh_active == n_diff and n_diff > 0,
            "R4 %-12s _rh_nat == what Lion wrote; _rh_n == %d; _rh_active == %d updates where it differed (> 0)" % (lab, a.updates, n_diff))
        chk(bad_dead == 0, "R4 %-12s DEAD STATE: a second object driven by a DIFFERENT z stream holds BITWISE the same beta at every update" % lab)

    print("\n" + "=" * 78)
    print("R5  RULE 20 and the reader prefixes")
    chk(all(re.match(r"^[A-Za-z0-9_.:]*$", SC.RESTHOLD[x]) and "," not in SC.RESTHOLD[x] for x in SC.ARMS),
        "every registered REST_HOLD value is one token of [A-Za-z0-9_.:], no comma")
    pf = ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "COMP_HOLD", "REST_HOLD")
    chk(not any(p != q and p.startswith(q) for p in pf for q in pf), "no witness prefix is a prefix of another (V/B/G/C/R)")

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %d" % len(FAILED)))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
