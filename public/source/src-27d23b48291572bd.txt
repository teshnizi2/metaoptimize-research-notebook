"""Structure, loudness, witness, schedule and meta-update semantics of PATCH_GROUPHOLD (`GROUP_HOLD=...`), on the LIVE
ResNet18_c100 model of the isolated `cvt7` tree.  CORRECTIONS 243.

STRATEGY (the test_betahold pattern): compare the PATCHED file against the UNPATCHED file (cvt4's HF.py), never against
my expectation of it; read every registered string and witness from the scorer.

  G0  STRUCTURE (no torch).  PATCH_GROUPHOLD / GROUP_HOLD / _gh_ absent from --pre, present in --post; the four inserted
      regions appear exactly as registered; deleting them from --post reproduces --pre BYTE FOR BYTE; exactly five new
      `_gh_` methods; every pre-existing method present, in order.
  G1  WITNESS AND INIT (torch, CPU).  For each registered arm string, HF.init_meta on the live model with ISO's spec
      prints exactly ONE `GROUP_HOLD:` line == the scorer's WITNESS_GH, byte for byte, plus `BETA_HOLD: off` and
      `VOTE_W: off`; the group index is 1 and the initial beta is [b0, b0] (tri) or [b0, -15] (floor).  Unset and EMPTY
      print `GROUP_HOLD: off`, set `_gh_on` False, and leave beta and every pre-existing attribute EQUAL to the
      unpatched file's (only `_gh_on` added).
  G2  THE GRAMMAR AND THE PREMISES ARE LOUD: malformed values, a partial set, a superset, names out of model order, a
      repeated name, the complement named, specs and environments that the patch refuses -- each RAISES.
  G3  THE SCHEDULE.  The patch's own _gh_value(n), stored into float32, equals the scorer's hold_value at every n in
      0..50,001 for the three held arms.
  G4  THE META-UPDATE, WITH SYNTHETIC z.  Two HF objects, one from the UNPATCHED file and one from the PATCHED file with
      each hold, are driven through the harness's OWN Lion_meta_update + clamp (+ _bh_apply guard, off, + _gh_apply)
      for 3,000 updates with the same random z (the first z is 0, as in a run).  At every update: the complement's
      beta[0][0] and the WHOLE momentum_meta vector are BITWISE the unpatched object's; beta[0][1] == the schedule;
      _gh_nat == the value the harness's Lion wrote from the previous HELD value; _gh_active counts exactly the updates
      where they differ, and is > 0.
  G5  RULE 20: every registered GROUP_HOLD value is one token of [A-Za-z0-9_.:+], no comma; no witness starts with a
      registered reader's prefix (BETA_HOLD, VOTE_W).

RUN (needs torch; on the cluster against the real isolated tree):
  python3 tests/test_grouphold.py --pre  $WS/harness_cvt7/cifar10/Optimizers/HF.py.pre_grouphold \\
        --post $WS/harness_cvt7/cifar10/Optimizers/HF.py --cifar-dir $WS/harness_cvt7/cifar10
  python3 tests/test_grouphold.py --pre ... --post ... --structural-only          # G0 only, no torch
"""
import argparse, ast, contextlib, importlib.machinery, importlib.util, io, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAILED = []

G_APPLY = ("            # --- PATCH_GROUPHOLD ---\n"
           "            if getattr(self, '_gh_on', False):\n"
           "                self._gh_apply()\n"
           "            # --- end PATCH_GROUPHOLD ---\n")
G_ATTACH = ("        # --- PATCH_GROUPHOLD ---\n"
            "        if getattr(self, '_gh_on', False):\n"
            "            self._gh_attach(rec)\n"
            "        # --- end PATCH_GROUPHOLD ---\n")
CALL = "        self._gh_init(net_param_names_and_size)  # PATCH_GROUPHOLD\n"
M_HEAD = "    # --------------------------------------------------- PATCH_GROUPHOLD\n"
M_TAIL = "    # ----------------------------------------------- end PATCH_GROUPHOLD\n\n"
NEW_METHODS = ["_gh_init", "_gh_value", "_gh_set", "_gh_apply", "_gh_attach"]
BH_APPLY_END = "                self._bh_apply()\n            # --- end PATCH_BETAHOLD ---\n"
BH_ATTACH_END = "            self._bh_attach(rec)\n        # --- end PATCH_BETAHOLD ---\n"


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
    print("G0  STRUCTURE: the two files differ ONLY by PATCH_GROUPHOLD's four regions")
    chk("PATCH_GROUPHOLD" not in pre_src and "GROUP_HOLD" not in pre_src and "_gh_" not in pre_src,
        "--pre is UNPATCHED (no marker, no GROUP_HOLD, no _gh_ name)")
    chk("PATCH_BETAHOLD" in pre_src and "PATCH_VOTEWEIGHT" in pre_src,
        "--pre is cvt4's PATCH_BETAHOLD HF.py (the base this patch sits on)")
    chk("PATCH_GROUPHOLD" in post_src, "--post IS patched")
    s = post_src
    anchor1 = BH_APPLY_END + G_APPLY + "            self._probe(HtT_gradft)  # PATCH_PROBE\n"
    chk(s.count(G_APPLY) == 1 and s.count(anchor1) == 1,
        "region 1: ONE guarded _gh_apply, right after PATCH_BETAHOLD's guarded apply and before _probe")
    s = s.replace(G_APPLY, "")
    chk(s.count(CALL) == 1 and ("        self._bh_init(net_param_names_and_size)  # PATCH_BETAHOLD\n" + CALL) in s,
        "region 2: ONE _gh_init call, right after init_meta's _bh_init call")
    s = s.replace(CALL, "")
    chk(s.count(G_ATTACH) == 1 and (BH_ATTACH_END + G_ATTACH + "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n") in s,
        "region 3: ONE guarded _gh_attach, after PATCH_BETAHOLD's attach and right before _pt_attach in _probe")
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
    chk([m for m in meth if m.startswith("_gh_")] == NEW_METHODS, "exactly five new methods, all `_gh_`-prefixed",
        str([m for m in meth if m.startswith("_gh_")]))
    pre_meth = [f.name for f in [n for n in ast.parse(pre_src).body if isinstance(n, ast.ClassDef)
                                 and n.name == "HF"][0].body if isinstance(f, ast.FunctionDef)]
    chk([m for m in meth if not m.startswith("_gh_")] == pre_meth, "every pre-existing method is present, in order")


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
    SC = _load("cvt7_scorer", os.path.join(REPO, "analysis", "cVT7_grouphold_score.py"))
    tree = os.path.abspath(a.cifar_dir)
    sys.path.insert(0, tree)
    os.chdir(tree)
    from build_network import build_network
    net = build_network(SC.NET, "cpu")
    nps = [(n, p.data.size()) for n, p in net.named_parameters()]
    PRE = _load("hf_pre_grouphold", a.pre).HF
    POST = _load("hf_post_grouphold", a.post).HF
    CLEAN = dict(GROUP_HOLD=None, BETA_HOLD=None, VOTE_W=None)

    print("\n" + "=" * 78)
    print("G1  WITNESS AND INIT on the live %s (%d tensors), ISO's spec" % (SC.NET, len(nps)))
    chk([n for n, _s in nps] == SC.NAMES, "the live parameter names are the scorer's NAMES")
    for arm in SC.HELD:
        with env(**dict(CLEAN, GROUP_HOLD=SC.GROUPHOLD[arm])):
            o, out = make(POST, torch, nps, SC.ISOSPEC)
        gh = [ln for ln in out if ln.startswith("GROUP_HOLD")]
        bh = [ln for ln in out if ln.startswith("BETA_HOLD")]
        vw = [ln for ln in out if ln.startswith("VOTE_W")]
        chk(gh == [SC.WITNESS_GH[arm]] and bh == ["BETA_HOLD: off"] and vw == ["VOTE_W: off"],
            "G1 %-9s exactly one GROUP_HOLD line == the scorer's witness; BETA_HOLD: off; VOTE_W: off" % arm,
            (gh or ["(none)"])[0][:110])
        want1 = SC.LO if SC.MODE[arm][0] == "floor" else SC.B0
        chk(o._gh_on and o._gh_g == 1 and o._bh_on is False and [float(x) for x in o.beta[0]] == [SC.B0, want1]
            and [len(x) for x in o.param_groups_indices] == SC.ISO_SIZES,
            "G1 %-9s _gh_on, group 1 of [59,3], initial beta [b0, %r]" % (arm, want1), str([float(x) for x in o.beta[0]]))
    for val in (None, ""):
        lab = "unset" if val is None else "EMPTY"
        with env(**dict(CLEAN, GROUP_HOLD=val)):
            o1, out1 = make(POST, torch, nps, SC.ISOSPEC)
            o0, _out0 = make(PRE, torch, nps, SC.ISOSPEC)
            k1, _ = make(POST, torch, nps, "scalar")
        chk([ln for ln in out1 if ln.startswith("GROUP_HOLD")] == ["GROUP_HOLD: off"] and o1._gh_on is False,
            "G1 GROUP_HOLD %s: prints `GROUP_HOLD: off`, _gh_on False" % lab)
        same_attrs = sorted(k for k in vars(o1) if k != "_gh_on") == sorted(vars(o0))
        chk(same_attrs and torch.equal(o1.beta[0], o0.beta[0]) and o1.param_groups_indices == o0.param_groups_indices,
            "G1 GROUP_HOLD %s: every attribute and beta EQUAL to the unpatched object's (only _gh_on added)" % lab)
        chk(k1._gh_on is False, "G1 GROUP_HOLD %s on the scalar spec is off too" % lab)

    print("\n" + "=" * 78)
    print("G2  LOUDNESS: every malformed value, set, spec or environment RAISES at construction")
    c = list(SC.CARRIERS)
    full = "+".join(c)
    bad = [
        ("no mode", full, SC.ISOSPEC, {}), ("tri without P", full + ":tri", SC.ISOSPEC, {}),
        ("negative P", full + ":tri:-5", SC.ISOSPEC, {}), ("P = 0", full + ":tri:0", SC.ISOSPEC, {}),
        ("exponent P", full + ":tri:1e3", SC.ISOSPEC, {}), ("7-digit P", full + ":tri:1234567", SC.ISOSPEC, {}),
        ("unknown mode", full + ":shared", SC.ISOSPEC, {}), ("trailing junk", full + ":floor,x", SC.ISOSPEC, {}),
        ("comma-separated names", ",".join(c) + ":floor", SC.ISOSPEC, {}),
        ("empty name", "+" + full + ":floor", SC.ISOSPEC, {}),
        ("unknown name", "nosuch.weight:floor", SC.ISOSPEC, {}),
        ("PARTIAL set (2 of 3)", "+".join(c[:2]) + ":floor", SC.ISOSPEC, {}),
        ("ONE carrier alone", c[2] + ":floor", SC.ISOSPEC, {}),
        ("SUPERSET (+ layer4.1.bn1.weight)", "+".join(c[:2] + ["layer4.1.bn1.weight"] + c[2:]) + ":floor", SC.ISOSPEC, {}),
        ("names out of model order", "+".join([c[1], c[0], c[2]]) + ":floor", SC.ISOSPEC, {}),
        ("repeated name", "+".join(c + [c[2]]) + ":floor", SC.ISOSPEC, {}),
        ("the complement's first member", "conv1.weight:floor", SC.ISOSPEC, {}),
        ("scalar spec", full + ":floor", "scalar", {}), ("layerwise spec", full + ":floor", "layerwise", {}),
        ("tn: spec", full + ":floor", "tn:" + SC.ISOSPEC, {}),
        ("VOTE_W on at the same time", full + ":floor", SC.ISOSPEC, {"VOTE_W": "layer4.0.bn1.weight:1"}),
        ("BETA_HOLD on at the same time", "layer4.0.bn2.weight:floor", "sets:1-49,51-52,54-58,60-62/layer4.0.bn2.weight/layer4.0.shortcut.1.weight/layer4.1.bn2.weight",
         {"BETA_HOLD": "layer4.0.bn2.weight:floor", "gh": "layer4.0.shortcut.1.weight:floor"}),
        ("meta algorithm Adam", full + ":floor", SC.ISOSPEC, {"alg": "Adam"}),
        ("BETA_CLIP unset", full + ":tri:100", SC.ISOSPEC, {"clip": None}),
        ("HIER=twolevel", full + ":floor", SC.ISOSPEC, {"hier": "twolevel"}),
    ]
    for lab, val, spec, kw in bad:
        vw = kw.pop("VOTE_W", None)
        bhv = kw.pop("BETA_HOLD", None)
        ghv = kw.pop("gh", None) or val
        raised = False
        try:
            with env(GROUP_HOLD=ghv, VOTE_W=vw, BETA_HOLD=bhv):
                make(POST, torch, nps, spec, **kw)
        except ValueError as e:
            raised = "PATCH_GROUPHOLD" in str(e)
        chk(raised, "G2 %-34s RAISES" % lab, repr(ghv)[:60])

    print("\n" + "=" * 78)
    print("G3  THE SCHEDULE: the patch's _gh_value, stored into float32, == the scorer's hold_value at every update")
    for arm in SC.HELD:
        with env(**dict(CLEAN, GROUP_HOLD=SC.GROUPHOLD[arm])):
            o, _ = make(POST, torch, nps, SC.ISOSPEC)
        mism = 0
        t = torch.zeros(1)
        for k in range(0, 50002):
            t[0] = o._gh_value(k)
            mism += float(t[0]) != SC.hold_value(arm, k)
        chk(mism == 0, "G3 %-9s float32(_gh_value(n)) == scorer hold_value(n) for n = 0..50001" % arm, "%d mismatches" % mism)

    print("\n" + "=" * 78)
    print("G4  THE META-UPDATE with synthetic z: complement and momentum untouched, the carrier group held, the record exact")
    g = torch.Generator().manual_seed(243)
    for arm in SC.HELD + ("TRI100",):
        val = SC.GROUPHOLD.get(arm) or ("+".join(SC.CARRIERS) + ":tri:100")
        with env(**CLEAN):
            U, _ = make(PRE, torch, nps, SC.ISOSPEC)
        with env(**dict(CLEAN, GROUP_HOLD=val)):
            H, _ = make(POST, torch, nps, SC.ISOSPEC)
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
            if getattr(H, "_bh_on", False):
                H._bh_apply()
            H._gh_apply()
            bad_cmp += float(H.beta[0][0]) != float(U.beta[0][0])
            bad_mom += not torch.equal(torch.as_tensor(H.momentum_meta[0]), torch.as_tensor(U.momentum_meta[0]))
            want = -15.0 if mode == "floor" else SC.v_of(t_, P)
            bad_hold += float(H.beta[0][1]) != want
            bad_nat += H._gh_nat != wrote
            n_diff += wrote != float(H.beta[0][1])
        chk(bad_cmp == 0, "G4 %-9s complement beta[0][0] BITWISE the unpatched object's at all %d updates" % (arm, a.updates))
        chk(bad_mom == 0, "G4 %-9s the WHOLE momentum_meta vector (the carrier group's entry included) BITWISE the unpatched object's" % arm)
        chk(bad_hold == 0, "G4 %-9s beta[0][1] == the registered hold after every update" % arm, "%d bad" % bad_hold)
        chk(bad_nat == 0 and H._gh_n == a.updates and H._gh_active == n_diff and n_diff > 0,
            "G4 %-9s _gh_nat == what Lion wrote; _gh_n == %d; _gh_active == %d updates where it differed (> 0)"
            % (arm, a.updates, n_diff))

    print("\n" + "=" * 78)
    print("G5  RULE 20 and the reader prefixes")
    chk(all(re.match(r"^[A-Za-z0-9_.:+]+$", SC.GROUPHOLD[x]) and "," not in SC.GROUPHOLD[x] for x in SC.HELD),
        "every registered GROUP_HOLD value is one token of [A-Za-z0-9_.:+], no comma")
    chk(not any(w.startswith("BETA_HOLD") or w.startswith("VOTE_W") for w in SC.WITNESS_GH.values()),
        "no GROUP_HOLD witness starts with BETA_HOLD or VOTE_W")

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %d" % len(FAILED)))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
