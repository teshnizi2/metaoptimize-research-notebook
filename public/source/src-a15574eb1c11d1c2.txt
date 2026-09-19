"""Structure, loudness, witness and step semantics of PATCH_SHADOWVOTE (`SHADOW_VOTE=...`), on the isolated `csv1` tree.
CORRECTIONS 262.  (The test_windowhold pattern: compare the PATCHED file against the UNPATCHED file -- cvt9's HF.py -- and
read every string and witness from the scorer.)

  S0  STRUCTURE (no torch).  PATCH_SHADOWVOTE / SHADOW_VOTE / _sv_ absent from --pre, present in --post; the five regions
      appear exactly as registered; deleting them from --post reproduces --pre BYTE FOR BYTE; exactly five new `_sv_`
      methods right after PATCH_WINDOWHOLD's five; every pre-existing method present, in order.
  S1  WITNESS AND INIT (torch, CPU, the LIVE PlainNet18_c100, init_meta only).  Each registered arm prints exactly ONE
      SHADOW_VOTE line and ONE VOTE_W line == the scorer's witnesses, and BETA_HOLD / COMP_HOLD / WINDOW_HOLD: off.
      SHADOW_VOTE unset and EMPTY print `SHADOW_VOTE: off`, set `_sv_on` False, and leave beta and every pre-existing
      attribute EQUAL to the unpatched (cvt9) object's, on k01, MUTE and HEAD.
  S2  LOUDNESS: malformed values and every refused premise RAISE at construction.
  S3  STEP SEMANTICS on a tiny BatchNorm MLP through the REAL HF.__init__ and HF.step (CPU), 150 steps, alpha0 1e-2 so the
      named tensor moves visibly, with SCHED '' AND SCHED 'none' (the batch's; alpha then float64):
      (a) OFF (unset, empty): beta, loss, every weight, every h_condenced and momentum_base BITWISE the unpatched object's;
      (b) INERT on-paths shadow:shared and natural:shared: the same, BITWISE;
      (c) shadow:floor: at every step the named weight moved by EXACTLY the floor step (w - a_fl*(m + wd*w), bitwise), its
          h_condenced slot == the SHADOW recursion at the shared step (bitwise, from the test's own pre-step copies), the
          kept natural trace == the NATURAL recursion at the floor, every other tensor's h == the unmodified recursion at the
          shared step, and z handed to the meta update == sum_j <h_j, g_j> over the slots the patch left (the shadow term);
      (d) natural:floor: the same with the roles swapped (the slot holds the natural trace, the kept trace the shadow).
  S4  RULE 20: every registered SHADOW_VOTE value is one token of [A-Za-z0-9_.:/], no comma.

  python3 tests/test_shadowvote.py --pre $WS/harness_csv1/cifar10/Optimizers/HF.py.pre_shadowvote \\
        --post $WS/harness_csv1/cifar10/Optimizers/HF.py --cifar-dir $WS/harness_csv1/cifar10
  python3 tests/test_shadowvote.py --pre ... --post ... --structural-only          # S0 only, no torch
"""
import argparse, ast, contextlib, importlib.machinery, importlib.util, io, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAILED = []

G_PRE = ("            # --- PATCH_SHADOWVOTE ---\n"
         "            if getattr(self, '_sv_on', False):\n"
         "                self._sv_pre(net, g)\n"
         "            # --- end PATCH_SHADOWVOTE ---\n")
G_POST = ("            # --- PATCH_SHADOWVOTE ---\n"
          "            if getattr(self, '_sv_on', False):\n"
          "                self._sv_post(net)\n"
          "            # --- end PATCH_SHADOWVOTE ---\n")
PT_CAP = "            self._pt_capture(self.h_condenced, g, HtT_gradft)  # PATCH_PROBE_TENSOR\n"
BASE = "            self.base_update(net,g)\n"
G_ATTACH = ("        # --- PATCH_SHADOWVOTE ---\n"
            "        if getattr(self, '_sv_on', False):\n"
            "            self._sv_attach(rec)\n"
            "        # --- end PATCH_SHADOWVOTE ---\n")
CALL = "        self._sv_init(net_param_names_and_size)  # PATCH_SHADOWVOTE\n"
M_HEAD = "    # ------------------------------------------------------ PATCH_SHADOWVOTE\n"
M_TAIL = "    # -------------------------------------------------- end PATCH_SHADOWVOTE\n\n"
NEW_METHODS = ["_sv_init", "_sv_pre", "_sv_post", "_sv_norms", "_sv_attach"]


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
    print("S0  STRUCTURE: the two files differ ONLY by PATCH_SHADOWVOTE's five regions")
    chk("PATCH_SHADOWVOTE" not in pre_src and "SHADOW_VOTE" not in pre_src and "_sv_" not in pre_src,
        "--pre is UNPATCHED (no marker, no SHADOW_VOTE, no _sv_ name)")
    chk("PATCH_WINDOWHOLD" in pre_src and "PATCH_COMPHOLD" in pre_src and "PATCH_BETAHOLD" in pre_src and "PATCH_VOTEWEIGHT" in pre_src,
        "--pre is cvt9's PATCH_WINDOWHOLD HF.py (the base this patch sits on)")
    chk("PATCH_SHADOWVOTE" in post_src, "--post IS patched")
    s = post_src
    chk(s.count(G_PRE) == 1 and s.count(PT_CAP + G_PRE + "            \n" + BASE) == 1,
        "region 1: ONE guarded _sv_pre, right after the PROBE_TENSOR capture and before base_update")
    s = s.replace(G_PRE, "")
    chk(s.count(G_POST) == 1 and s.count(BASE + G_POST) == 1, "region 2: ONE guarded _sv_post, right after base_update")
    s = s.replace(G_POST, "")
    chk(s.count(CALL) == 1 and ("        self._wh_init(net_param_names_and_size)  # PATCH_WINDOWHOLD\n" + CALL) in s,
        "region 3: ONE _sv_init call, right after init_meta's _wh_init call")
    s = s.replace(CALL, "")
    anchor4 = ("        # --- end PATCH_WINDOWHOLD ---\n" + G_ATTACH + "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")
    chk(s.count(G_ATTACH) == 1 and anchor4 in s,
        "region 4: ONE guarded _sv_attach, after PATCH_WINDOWHOLD's _wh_attach region and before _pt_attach in _probe")
    s = s.replace(G_ATTACH, "")
    i0, i1 = s.find(M_HEAD), s.find(M_TAIL)
    chk(i0 >= 0 and i1 > i0 and s[i1 + len(M_TAIL):].startswith("    def check_required_attributes(")
        and s[:i0].endswith("    # -------------------------------------------------- end PATCH_WINDOWHOLD\n\n"),
        "region 5: the method block sits right after PATCH_WINDOWHOLD's block, immediately before check_required_attributes")
    if i0 >= 0 and i1 > i0:
        s = s[:i0] + s[i1 + len(M_TAIL):]
    chk(s == pre_src, "deleting the five inserted regions reproduces --pre BYTE FOR BYTE",
        "" if s == pre_src else "(%d vs %d bytes)" % (len(s), len(pre_src)))
    tree = ast.parse(post_src)
    hf = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HF"]
    meth = [f.name for f in hf[0].body if isinstance(f, ast.FunctionDef)] if hf else []
    chk([m for m in meth if m.startswith("_sv_")] == NEW_METHODS, "exactly five new methods, all `_sv_`-prefixed",
        str([m for m in meth if m.startswith("_sv_")]))
    wh = [i for i, m in enumerate(meth) if m.startswith("_wh_")]
    sv = [i for i, m in enumerate(meth) if m.startswith("_sv_")]
    chk(wh and sv and sv[0] == wh[-1] + 1, "the _sv_ methods follow PATCH_WINDOWHOLD's _wh_ methods directly")
    pre_meth = [f.name for f in [n for n in ast.parse(pre_src).body if isinstance(n, ast.ClassDef)
                                 and n.name == "HF"][0].body if isinstance(f, ast.FunctionDef)]
    chk([m for m in meth if not m.startswith("_sv_")] == pre_meth, "every pre-existing method is present, in order")


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


KEYS = ("SHADOW_VOTE", "VOTE_W", "BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD")


def make(HF, torch, nps, spec, base_alg="SGDm", hier="none", sched="none", clip=True):
    o = HF.__new__(HF)
    o.num_layers = len(nps)
    o._device = torch.device("cpu")
    o.args_base = {"alg": base_alg, "weight_decay": 0.1, "momentum_param": 0.99}
    o.args_meta = {"alg": "Lion", "meta_stepsize": 1e-3, "momentum_param": 0.99, "Lion_beta2": 0.9, "weight_decay": 0}
    o._beta_lo, o._beta_hi = (-15.0, -2.3026) if clip else (None, None)
    o._hier = hier
    o._sched = sched
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        HF.init_meta(o, spec, nps, 1e-6)
    return o, buf.getvalue().splitlines()


class _W:
    def add_scalar(self, *a, **k):
        pass


def run_tiny(HF, torch, nn, mode, sched, steps, check=None):
    """the REAL HF.__init__ / HF.step on a tiny BatchNorm MLP.  -> dict of trajectories; check(o, net, pre) per step."""
    torch.manual_seed(262)
    net = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4))
    X = torch.randn(64, 8, generator=torch.Generator().manual_seed(1))
    Y = torch.randint(0, 4, (64,), generator=torch.Generator().manual_seed(2))
    crit = nn.CrossEntropyLoss()
    e = dict((k, None) for k in KEYS)
    e.update({"BETA_CLIP": "-15:-2.3026", "SCHED": sched, "HIER": "none", "PROBE": None, "PROBE_TENSOR": None})
    if mode is not None:
        e["SHADOW_VOTE"] = mode
    out = {"beta": [], "loss": []}
    with env(**e):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            o = HF(net, "scalar", 1e-2, {"alg": "SGDm", "weight_decay": 0.1, "momentum_param": 0.99},
                   {"alg": "Lion", "meta_stepsize": 1e-3, "momentum_param": 0.99, "Lion_beta2": 0.9, "weight_decay": 0}, 1, _W())
        out["stdout"] = buf.getvalue().splitlines()
        seen_z = []
        orig_meta = o.meta_update

        def spy_meta(z):
            seen_z.append(float(z[0]))
            return orig_meta(z)
        o.meta_update = spy_meta
        for t in range(steps):
            idx = torch.arange(t % 64, t % 64 + 16) % 64
            loss = crit(net(X[idx]), Y[idx])
            pre = None
            if check is not None:
                import numpy as np
                pre = {"h": [h.clone() if torch.is_tensor(h) else h for h in o.h_condenced],
                       "m": [m.clone() if torch.is_tensor(m) else m for m in o.momentum_base],
                       "w": [p.data.clone() for p in net.parameters()],
                       "beta": o.beta[0].clone(), "other": None if getattr(o, "_sv_other", None) is None
                       else dict((k, v.clone()) for k, v in o._sv_other.items())}
                a = np.exp(pre["beta"].cpu().numpy())
                pre["a_sh"] = a * 1.0 if sched else a
            nz = len(seen_z)
            with contextlib.redirect_stdout(io.StringIO()):
                o.step(net, loss)
            out["beta"].append(float(o.beta[0]))
            out["loss"].append(float(loss))
            if check is not None:
                pre["z"] = seen_z[nz]
                check(o, net, pre, t)
        out["w"] = [p.data.clone() for p in net.parameters()]
        out["h"] = [h.clone() if torch.is_tensor(h) else h for h in o.h_condenced]
        out["m"] = [m.clone() if torch.is_tensor(m) else m for m in o.momentum_base]
        out["obj"] = o
    return out


def same_traj(torch, A, B):
    eq = lambda xs, ys: len(xs) == len(ys) and all((torch.equal(x, y) if torch.is_tensor(x) else x == y) for x, y in zip(xs, ys))
    return A["beta"] == B["beta"] and A["loss"] == B["loss"] and eq(A["w"], B["w"]) and eq(A["h"], B["h"]) and eq(A["m"], B["m"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--cifar-dir", default="")
    ap.add_argument("--structural-only", action="store_true")
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()
    pre_src, post_src = open(a.pre).read(), open(a.post).read()
    structural(pre_src, post_src)
    if a.structural_only:
        print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %d" % len(FAILED)))
        raise SystemExit(1 if FAILED else 0)

    import numpy as np
    import torch
    import torch.nn as nn
    SC = _load("csv1_scorer", os.path.join(REPO, "analysis", "cSV1_shadowvote_score.py"))
    tree = os.path.abspath(a.cifar_dir)
    sys.path.insert(0, tree)
    os.chdir(tree)
    from build_network import build_network
    net = build_network(SC.NET, "cpu")
    nps = [(n, p.data.size()) for n, p in net.named_parameters()]
    PRE = _load("hf_pre_shadowvote", a.pre).HF
    POST = _load("hf_post_shadowvote", a.post).HF

    def envs(arm):
        return dict(SHADOW_VOTE=SC.SHADOWVOTE[arm] or None, VOTE_W=SC.VOTEW[arm] or None, BETA_HOLD=None, COMP_HOLD=None, WINDOW_HOLD=None)

    print("\n" + "=" * 78)
    print("S1  WITNESS AND INIT on the live %s (%d tensors)" % (SC.NET, len(nps)))
    chk([n for n, _s in nps] == SC.NAMES, "the live parameter names are the scorer's NAMES")
    for arm in SC.ARMS:
        with env(**envs(arm)):
            o, out = make(POST, torch, nps, SC.SPEC[arm])
        got = dict((k, [ln for ln in out if ln.startswith(k)]) for k in KEYS)
        chk(got["SHADOW_VOTE"] == [SC.WITNESS_SV[arm]] and got["VOTE_W"] == [SC.WITNESS_VW[arm]]
            and got["BETA_HOLD"] == ["BETA_HOLD: off"] and got["COMP_HOLD"] == ["COMP_HOLD: off"] and got["WINDOW_HOLD"] == ["WINDOW_HOLD: off"],
            "S1 %-10s exactly one SHADOW_VOTE and VOTE_W line == the scorer's witnesses; holds off" % arm, (got["SHADOW_VOTE"] or ["(none)"])[0][:100])
        chk(bool(o._sv_on) == (arm in SC.SV_ARMS) and (arm not in SC.SV_ARMS or (o._sv_idx == [SC.IDX_HEAD - 1]
            and (o._sv_vote, o._sv_applied) == SC.MODE[arm]
            and (o._sv_floor_alpha is None if SC.MODE[arm][1] == "shared" else float(o._sv_floor_alpha) == SC.FLOOR_ALPHA))),
            "S1 %-10s _sv_on %s; index, mode and floor step size as registered (FLOOR_ALPHA %r)" % (arm, arm in SC.SV_ARMS, SC.FLOOR_ALPHA))
    for arm in ("k01", "MUTE", "HEAD"):
        for val in (None, ""):
            e = envs(arm)
            e["SHADOW_VOTE"] = val
            with env(**e):
                o1, out1 = make(POST, torch, nps, SC.SPEC[arm])
                o0, _ = make(PRE, torch, nps, SC.SPEC[arm])
            lab = "unset" if val is None else "EMPTY"
            chk([ln for ln in out1 if ln.startswith("SHADOW_VOTE")] == ["SHADOW_VOTE: off"] and o1._sv_on is False,
                "S1 %-10s SHADOW_VOTE %s: prints `SHADOW_VOTE: off`, _sv_on False" % (arm, lab))
            same_attrs = sorted(k for k in vars(o1) if k != "_sv_on") == sorted(vars(o0))
            same_other = all((torch.equal(v, vars(o0)[k]) if torch.is_tensor(v) else True) for k, v in vars(o1).items() if k != "_sv_on")
            chk(same_attrs and same_other and all(torch.equal(x, y) for x, y in zip(o1.beta, o0.beta)),
                "S1 %-10s SHADOW_VOTE %s: every attribute and beta EQUAL to cvt9's object (only _sv_on added)" % (arm, lab))

    print("\n" + "=" * 78)
    print("S2  LOUDNESS: every malformed value or refused premise RAISES at construction")
    nm = SC.NAME_HEAD
    bad = [("no name", "shadow:floor", {}), ("unknown applied", "shadow:low:" + nm, {}), ("upper case", "Shadow:floor:" + nm, {}),
           ("empty name", "shadow:floor:", {}), ("comma list", "shadow:floor:%s,layer4.1.bn1.weight" % nm, {}),
           ("unknown name", "shadow:floor:layer9.bn.weight", {}), ("repeated name", "shadow:floor:%s/%s" % (nm, nm), {}),
           ("trailing slash", "shadow:floor:%s/" % nm, {}), ("blockwise spec", "shadow:floor:" + nm, {"spec": SC.HEADSPEC}),
           ("base AdamW", "shadow:floor:" + nm, {"base_alg": "AdamW"}), ("with VOTE_W", "shadow:floor:" + nm, {"vw": "layer4.1.bn1.weight:0"}),
           ("HIER zpool", "shadow:floor:" + nm, {"hier": "zpool"}), ("SCHED cosine", "shadow:floor:" + nm, {"sched": "cosine"}),
           ("floor without BETA_CLIP", "shadow:floor:" + nm, {"clip": False}), ("with BETA_HOLD", "shadow:floor:" + nm, {"bh": nm + ":floor"})]
    for lab, val, kw in bad:
        raised, why = False, ""
        try:
            with env(SHADOW_VOTE=val, VOTE_W=kw.get("vw"), BETA_HOLD=kw.get("bh"), COMP_HOLD=None, WINDOW_HOLD=None):
                make(POST, torch, nps, kw.get("spec", "scalar"), base_alg=kw.get("base_alg", "SGDm"), hier=kw.get("hier", "none"),
                     sched=kw.get("sched", "none"), clip=kw.get("clip", True))
        except ValueError as ex:
            raised = "PATCH_SHADOWVOTE" in str(ex) or ("bh" in kw and "PATCH_BETAHOLD" in str(ex))
            why = str(ex)[:70]
        chk(raised, "S2 %-24s RAISES" % lab, "%r %s" % (val[:40], why))
    with env(SHADOW_VOTE="shadow:shared:%s/layer4.1.bn1.weight" % nm, VOTE_W=None, BETA_HOLD=None, COMP_HOLD=None, WINDOW_HOLD=None):
        o, out = make(POST, torch, nps, "scalar", clip=False)
    chk(o._sv_on and o._sv_idx == [49, 46] and [ln for ln in out if ln.startswith("SHADOW_VOTE")] ==
        ["SHADOW_VOTE: on type=scalar base=SGDm vote=shadow applied=shared floor=na items=50:%s:numel=512,47:layer4.1.bn1.weight:numel=512" % nm],
        "S2 a two-name list constructs (list-capable), applied=shared needs no BETA_CLIP")

    print("\n" + "=" * 78)
    print("S3  STEP SEMANTICS through the REAL HF.__init__ / HF.step on a tiny BatchNorm MLP (%d steps, alpha0 1e-2)" % a.steps)
    I = 2   # the tiny net's BatchNorm weight: 0.weight, 0.bias, 1.weight (index 2), 1.bias, 3.weight, 3.bias
    NMT = "1.weight"
    for sched in ("", "none"):
        ref = run_tiny(PRE, torch, nn, None, sched, a.steps)
        for lab, mode in (("OFF unset", None), ("OFF empty", ""), ("INERT shadow:shared", "shadow:shared:" + NMT),
                          ("INERT natural:shared", "natural:shared:" + NMT)):
            R = run_tiny(POST, torch, nn, mode, sched, a.steps)
            chk(same_traj(torch, ref, R), "S3 SCHED=%-5r %-22s beta, loss, every weight, h_condenced and momentum_base BITWISE cvt9's"
                % (sched, lab), "beta %.4f -> %.4f" % (ref["beta"][0], ref["beta"][-1]))
        for vote in ("shadow", "natural"):
            bad = {"w": 0, "slot": 0, "kept": 0, "others": 0, "z": 0, "alpha": 0}
            wd, gm = 0.1, 1

            def check(o, net, pre, t, vote=vote, bad=bad):
                fl = o._sv_floor_alpha * 1.0 if sched else o._sv_floor_alpha
                a_sh = pre["a_sh"]
                ps = [p.data for p in net.parameters()]
                bad["alpha"] += not (float(o._sv_save[0][1]) == float(a_sh) and float(o._sv_save[0][2]) == float(fl))
                w_exp = pre["w"][I] - fl * (pre["m"][I] + wd * pre["w"][I])
                bad["w"] += not torch.equal(ps[I], w_exp)
                sh = gm * (1 - wd * a_sh) * pre["h"][I] - a_sh * (pre["m"][I] + wd * pre["w"][I])
                oth_prev = pre["other"][I] if pre["other"] is not None else torch.zeros_like(pre["h"][I])
                if vote == "shadow":
                    slot_exp = sh
                    kept_exp = gm * (1 - wd * fl) * oth_prev - fl * (pre["m"][I] + wd * pre["w"][I])
                else:
                    slot_exp = gm * (1 - wd * fl) * pre["h"][I] - fl * (pre["m"][I] + wd * pre["w"][I])
                    kept_exp = gm * (1 - wd * a_sh) * oth_prev - a_sh * (pre["m"][I] + wd * pre["w"][I])
                bad["slot"] += not torch.equal(o.h_condenced[I], slot_exp)
                bad["kept"] += not torch.equal(o._sv_other[I], kept_exp)
                for j in range(len(ps)):
                    if j == I:
                        continue
                    hj = gm * (1 - wd * a_sh) * pre["h"][j] - a_sh * (pre["m"][j] + wd * pre["w"][j])
                    bad["others"] += not torch.equal(o.h_condenced[j], hj)
                # z handed to the meta update: sum_j <h_j(before), g_j> over the slots as they were BEFORE this step
                # (recomputed from the step's gradient is not possible here; check instead that it equals the sum over
                # the pre-step slots using the momentum identity m_new = 0.99 m + 0.01 g  ->  g = (m_new - 0.99 m) / 0.01)
                gs = [(o.momentum_base[j] - 0.99 * pre["m"][j]) / 0.01 for j in range(len(ps))]
                zexp = float(sum((pre["h"][j] * gs[j]).sum() for j in range(len(ps))))
                bad["z"] += abs(zexp - pre["z"]) > 1e-4 * (abs(zexp) + abs(pre["z"])) + 1e-12
            R = run_tiny(POST, torch, nn, "%s:floor:%s" % (vote, NMT), sched, a.steps, check=check)
            chk(not any(bad.values()), "S3 SCHED=%-5r %-7s:floor  every step: the named weight moved by the FLOOR step (bitwise); the slot "
                "== the %s recursion, the kept trace == the %s one (bitwise); every other h == the shared recursion; z == sum <h, g>"
                % (sched, vote, "SHADOW (shared step)" if vote == "shadow" else "NATURAL (floor)",
                   "NATURAL" if vote == "shadow" else "SHADOW"), str(bad))
            hv, ho = float(R["obj"].h_condenced[I].norm()), float(R["obj"]._sv_other[I].norm())
            chk((hv > 2 * ho) if vote == "shadow" else (ho > 2 * hv), "S3 SCHED=%-5r %-7s:floor  non-vacuity: the two traces differ (||slot|| %.3e, ||kept|| %.3e)"
                % (sched, vote, hv, ho))
            chk(not same_traj(torch, ref, R), "S3 SCHED=%-5r %-7s:floor  departs from cvt9's run (the switch bites)" % (sched, vote))

    print("\n" + "=" * 78)
    print("S4  RULE 20")
    chk(all(re.match(r"^[A-Za-z0-9_.:/]*$", SC.SHADOWVOTE[x]) and "," not in SC.SHADOWVOTE[x] for x in SC.ARMS),
        "every registered SHADOW_VOTE value is one token of [A-Za-z0-9_.:/], no comma")

    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %d" % len(FAILED)))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
