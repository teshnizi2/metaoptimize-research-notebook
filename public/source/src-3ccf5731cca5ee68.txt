"""STRUCTURE, LOUDNESS AND ARITHMETIC OF PATCH_DECAYMASK.  CORRECTIONS 260 / 261.

  D0  STRUCTURE: the post HF.py carries exactly the three registered regions; DELETING THEM REPRODUCES THE PRE HF.py BYTE
      FOR BYTE; the four _dm_ methods sit right before check_required_attributes; SGDm_base_update is unchanged; every
      pre-existing method is present in order.  (--structural-only stops here; no torch.)
  D1  WITNESS / INIT on the LIVE model of --cifar-dir (build_network on CPU): DECAY_MASK unset and empty leave every
      attribute of an HF object built by init_meta EQUAL to the pre-patch object's (plus `_dm_on` False), base_update NOT
      rebound, print `DECAY_MASK: off`; each registered string prints exactly cwd_design's witness; `normscale` == the
      set of BatchNorm2d/GroupNorm `.weight` tensors by MODULE TYPE on the live model.
  D2  LOUD: 12 malformed / illegal values or premises raise ValueError.
  D3  ARITHMETIC over 12 synthetic SGDm updates on the live model's tensor shapes (random grads, alpha a mix of scalar
      and per-tensor values, gamma 1): (a) the ON update with an all-False mask is BITWISE the unpatched
      SGDm_base_update (weights, momentum, h); (b) the ON update with the registered mask gives, for masked tensors,
      w' == w - a*m, h' == gamma*h - a*m BITWISE, and for unmasked tensors BITWISE the unpatched update; (c) the masked
      tensors DIFFER from the unpatched update (non-vacuity, alpha large); (d) the probe attach adds only dm_* keys with
      dm_n == updates, dm_skipped == updates * k, dm_wdterm > 0, len(dm_norm) == k.

  python3 tests/test_decaymask.py --pre <HF.py.pre_decaymask> --post <HF.py> [--structural-only] \\
      [--cifar-dir <tree>/cifar10 --batch cwd1|cwd2]
"""
import argparse
import contextlib
import importlib.util
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAILED = []


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def _load(name, path):
    import importlib.machinery
    spec = importlib.util.spec_from_file_location(name, path, loader=importlib.machinery.SourceFileLoader(name, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R1 = "        self._dm_init(net_param_names_and_size)  # PATCH_DECAYMASK\n"
R2 = ("        # --- PATCH_DECAYMASK ---\n"
      "        if getattr(self, '_dm_on', False):\n"
      "            self._dm_attach(rec)\n"
      "        # --- end PATCH_DECAYMASK ---\n")
R3_HEAD = "    # ------------------------------------------------------ PATCH_DECAYMASK\n"
R3_TAIL = "    # -------------------------------------------------- end PATCH_DECAYMASK\n\n"


def structure(pre, post):
    print("D0 STRUCTURE")
    chk(pre.count("PATCH_DECAYMASK") == 0 and "_dm_" not in pre, "D0 the pre file carries no PATCH_DECAYMASK / _dm_ name")
    chk(post.count(R1) == 1, "D0 region 1 (the _dm_init call) present once")
    chk(post.count(R2) == 1, "D0 region 2 (the guarded _dm_attach) present once")
    i3, j3 = post.find(R3_HEAD), post.find(R3_TAIL)
    chk(i3 > 0 and j3 > i3 and post.count(R3_HEAD) == 1 and post.count(R3_TAIL) == 1, "D0 region 3 (the method block) present once")
    if FAILED:
        return
    chk(post.index(R1) + len(R1) == post.index("        self.trace_meta = [0.0 for _ in  range(self.len_beta_list)]\n"),
        "D0 region 1 sits IMMEDIATELY before `self.trace_meta = ...` in init_meta")
    init_meta = post[post.index("    def init_meta("):post.index("        self.trace_meta = [0.0 for _ in  range(self.len_beta_list)]\n")]
    last_hold = max(init_meta.find("_rh_init(net_param_names_and_size)"), init_meta.find("_wh_init(net_param_names_and_size)"))
    chk(last_hold > 0 and init_meta.find("_dm_init(") > last_hold, "D0 region 1 runs AFTER the last hold patch's init call")
    chk(post.index(R2) + len(R2) == post.index("        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n"),
        "D0 region 2 sits IMMEDIATELY before `self._pt_attach(rec)  # PATCH_PROBE_TENSOR`")
    chk(j3 + len(R3_TAIL) == post.index("    def check_required_attributes("), "D0 region 3 sits IMMEDIATELY before check_required_attributes")
    stripped = post.replace(R1, "", 1).replace(R2, "", 1)
    k3, l3 = stripped.find(R3_HEAD), stripped.find(R3_TAIL)
    stripped = stripped[:k3] + stripped[l3 + len(R3_TAIL):]
    chk(stripped == pre, "D0 DELETING THE THREE REGIONS REPRODUCES THE PRE HF.py BYTE FOR BYTE", "%d vs %d bytes" % (len(stripped), len(pre)))
    block = post[i3:j3]
    meths = re.findall(r"^    def (\w+)\(", block, re.M)
    chk(meths == ["_dm_init", "_dm_resolve", "_dm_SGDm_base_update", "_dm_attach"], "D0 the block defines exactly the four _dm_ methods, in order", str(meths))
    pm = re.findall(r"^    def (\w+)\(", pre, re.M)
    qm = [m for m in re.findall(r"^    def (\w+)\(", post, re.M) if not m.startswith("_dm_")]
    chk(pm == qm, "D0 every pre-existing method present, in order (%d)" % len(pm))
    orig = ("    def SGDm_base_update(self,net,g):\n        #Base update\n"
            "        for w, grad, a ,i in zip(net.parameters(), g, self.alpha, range(self.num_layers)):\n"
            "            delta = a * (self.momentum_base[i] + self.args_base['weight_decay']*w.data)\n"
            "            w.data = w.data - delta\n")
    chk(pre.count(orig) == 1 and post.count(orig) == 1, "D0 SGDm_base_update is the registered text in pre and post")
    ub = block[block.index("    def _dm_SGDm_base_update("):block.index("    def _dm_attach(")]
    chk(ub.count("            delta = a * (self.momentum_base[i] + self.args_base['weight_decay']*w.data)\n") == 1
        and ub.count("            self.h_condenced[i] = self.gamma*(1-self.args_base['weight_decay']*a)*self.h_condenced[i] - delta\n") == 1
        and ub.count("                delta = a * self.momentum_base[i]\n") == 1
        and ub.count("                self.h_condenced[i] = self.gamma*self.h_condenced[i] - delta\n") == 1,
        "D0 the copy's unmasked branch is the original lines verbatim; the masked branch drops wd in delta AND in h")


def build_obj(HF, torch, nps, spec, env, args_base):
    for k in ("DECAY_MASK",):
        os.environ.pop(k, None)
    if env is not None:
        os.environ["DECAY_MASK"] = env
    o = HF.__new__(HF)
    o.num_layers = len(nps)
    o._device = torch.device("cpu")
    o.args_meta = {"alg": "Lion", "meta_stepsize": 1e-3, "momentum_param": 0.99, "Lion_beta2": 0.9, "weight_decay": 0}
    o.args_base = dict(args_base)
    o._beta_lo, o._beta_hi = -15.0, -2.3026
    o._hier = "none"
    o.base_update = o.SGDm_base_update
    buf = io.StringIO()
    err = None
    try:
        with contextlib.redirect_stdout(buf):
            HF.init_meta(o, spec, nps, 1e-6)
    except ValueError as ex:
        err = ex
    finally:
        os.environ.pop("DECAY_MASK", None)
    return o, buf.getvalue().splitlines(), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--structural-only", action="store_true")
    ap.add_argument("--cifar-dir", default="")
    ap.add_argument("--batch", default="")
    a = ap.parse_args()
    pre, post = open(a.pre).read(), open(a.post).read()
    structure(pre, post)
    if a.structural_only or FAILED:
        print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %r" % FAILED))
        raise SystemExit(1 if FAILED else 0)

    D = _load("cwd_design", os.path.join(REPO, "analysis", "cwd_design.py"))
    B = {"cwd1": D.CWD1, "cwd2": D.CWD2}[a.batch]
    tree = os.path.abspath(a.cifar_dir)
    os.chdir(tree)
    sys.path.insert(0, tree)
    import torch
    import torch.nn as nn
    from build_network import build_network
    HFpost = _load("hf_post", a.post).HF
    HFpre = _load("hf_pre", a.pre).HF
    net = build_network(B.NET, "cpu")
    nps = [(n, p.data.size()) for n, p in net.named_parameters()]
    names = [n for n, _ in nps]
    chk(names == B.NAMES and len(nps) == B.NTENS, "D1 the live %s from %s has the design's %d tensor names in order" % (B.NET, tree, B.NTENS))
    own = {}
    for mn, m in net.named_modules():
        for pn, _p in m.named_parameters(recurse=False):
            own[(mn + "." if mn else "") + pn] = m
    bnw = [i for i, n in enumerate(names) if n.endswith(".weight") and isinstance(own[n], (nn.BatchNorm2d, nn.GroupNorm))]
    AB = {"alg": "SGDm", "weight_decay": 0.1, "momentum_param": 0.99}

    print("\nD1 WITNESS / INIT")
    for spec in sorted(set(B.SPEC.values())):
        for env in (None, ""):
            o_pre = HFpre.__new__(HFpre)
            o_pre.num_layers = len(nps); o_pre._device = torch.device("cpu")
            o_pre.args_meta = {"alg": "Lion", "meta_stepsize": 1e-3, "momentum_param": 0.99, "Lion_beta2": 0.9, "weight_decay": 0}
            o_pre.args_base = dict(AB); o_pre._beta_lo, o_pre._beta_hi = -15.0, -2.3026; o_pre._hier = "none"
            o_pre.base_update = o_pre.SGDm_base_update
            os.environ.pop("DECAY_MASK", None)
            with contextlib.redirect_stdout(io.StringIO()):
                HFpre.init_meta(o_pre, spec, nps, 1e-6)
            o, out, err = build_obj(HFpost, torch, nps, spec, env, AB)
            kp = set(vars(o_pre)); ko = set(vars(o))
            same = (err is None and ko - kp == {"_dm_on"} and kp - ko == set() and o._dm_on is False
                    and all(k == "base_update" or (not torch.is_tensor(vars(o_pre)[k]) and not isinstance(vars(o_pre)[k], list) and vars(o_pre)[k] == vars(o)[k])
                            or (isinstance(vars(o_pre)[k], list) and len(vars(o_pre)[k]) == len(vars(o)[k])
                                and all((torch.equal(x, y) if torch.is_tensor(x) else x == y) for x, y in zip(vars(o_pre)[k], vars(o)[k])))
                            or (torch.is_tensor(vars(o_pre)[k]) and torch.equal(vars(o_pre)[k], vars(o)[k])) for k in kp)
                    and o.base_update.__func__ is HFpost.SGDm_base_update)
            chk(same and [ln for ln in out if ln.startswith("DECAY_MASK")] == ["DECAY_MASK: off"],
                "D1 spec %-38s DECAY_MASK %-5s: attributes == the pre object's (+ _dm_on False), base_update NOT rebound, `DECAY_MASK: off`"
                % (spec, "unset" if env is None else "empty"))
    for arm in B.ARMS:
        o, out, err = build_obj(HFpost, torch, nps, B.SPEC[arm], B.DMASK[arm] or None, AB)
        dl = [ln for ln in out if ln.startswith("DECAY_MASK")]
        ok = err is None and dl == [B.WITNESS_DM[arm]]
        if B.DMASK[arm]:
            ok = ok and o._dm_on and o.base_update.__func__ is HFpost._dm_SGDm_base_update and o._dm_idx == D.dm_indices(B.DMASK[arm], B.TENSORS)
        chk(ok, "D1 %-12s prints exactly the design's DECAY_MASK witness and %s" % (arm, "rebinds base_update" if B.DMASK[arm] else "does not rebind"),
            "" if ok else "%r %r" % (dl, err))
    o, out, err = build_obj(HFpost, torch, nps, "scalar", "normscale", AB)
    chk(err is None and o._dm_idx == bnw and len(bnw) > 0,
        "D1 normscale == the BatchNorm2d/GroupNorm .weight tensors BY MODULE TYPE on the live model (%d tensors)" % len(bnw))

    print("\nD2 LOUD")
    bad_vals = ["normscale+", "+layer4.1.bn2.weight", "layer4.1.bn2.weight+layer4.1.bn2.weight", "nosuch.weight",
                "layer4.1.bn2.weight,bn1.weight", "layer4.1.bn2.weight:0", " normscale", "Normscale"]
    for v in bad_vals:
        _o, _out, err = build_obj(HFpost, torch, nps, "scalar", v, AB)
        chk(isinstance(err, ValueError), "D2 DECAY_MASK=%r raises ValueError" % v, repr(err)[:80])
    for ab, lab in ((dict(AB, alg="AdamW"), "alg AdamW"), (dict(AB, weight_decay=0.0), "wd 0"), (dict(AB, weight_decay=None), "wd None"), ({}, "no args_base")):
        _o, _out, err = build_obj(HFpost, torch, nps, "scalar", "normscale", ab)
        chk(isinstance(err, ValueError), "D2 %s raises ValueError" % lab, repr(err)[:80])

    print("\nD3 ARITHMETIC (12 synthetic SGDm updates on the live shapes)")
    spec_masked = [arm for arm in B.ARMS if B.DMASK[arm]][0]
    mask_str = B.DMASK[spec_masked]
    g_ = torch.Generator().manual_seed(7)

    class FakeNet:
        def __init__(self, ts):
            self.ps = [nn.Parameter(t.clone()) for t in ts]

        def parameters(self):
            return iter(self.ps)

    init = [torch.randn(tuple(s), generator=g_) for _n, s in nps]
    NUPD = 12
    grads = [[torch.randn(tuple(s), generator=g_) for _n, s in nps] for _ in range(NUPD)]

    def run(obj, alpha_big, force_all_false=False, attach=False, verify=False):
        """-> (net, obj, rec, bad_masked, bad_unmasked, first_step_weights_of_unmasked0).  Checks inline (no history kept)."""
        netf = FakeNet(init)
        obj.momentum_base = [torch.zeros(tuple(s)) for _n, s in nps]
        obj.h_condenced = [torch.zeros(tuple(s)) for _n, s in nps]
        obj.gamma = 1
        if force_all_false:
            obj._dm_mask = [False] * len(nps)
        bm = bu = 0
        first = None
        for t in range(NUPD):
            obj.alpha = [torch.tensor(alpha_big * (1 + 0.1 * ((i + t) % 3)), dtype=torch.float32) for i in range(len(nps))]
            if verify:
                wb = [p.data.clone() for p in netf.ps]
                mb = [m.clone() for m in obj.momentum_base]
                hb = [h.clone() for h in obj.h_condenced]
            obj.base_update(netf, grads[t])
            if verify:
                for i in range(len(nps)):
                    a_ = obj.alpha[i]
                    if obj._dm_mask[i]:
                        bm += not (torch.equal(netf.ps[i].data, wb[i] - a_ * mb[i]) and torch.equal(obj.h_condenced[i], 1 * hb[i] - a_ * mb[i]))
                    else:
                        d_ = a_ * (mb[i] + 0.1 * wb[i])
                        bu += not (torch.equal(netf.ps[i].data, wb[i] - d_) and torch.equal(obj.h_condenced[i], 1 * (1 - 0.1 * a_) * hb[i] - d_))
                del wb, mb, hb
        rec = {}
        if attach:
            obj._dm_attach(rec)
        return netf, obj, rec, bm, bu

    ref_o, _o1, _e = build_obj(HFpre, torch, nps, "scalar", None, AB)
    ref_net, ref_obj, _r, _b1, _b2 = run(ref_o, 0.05)
    on_o, _o2, e2 = build_obj(HFpost, torch, nps, "scalar", mask_str, AB)
    chk(e2 is None, "D3 the ON object constructs with DECAY_MASK=%s" % mask_str)
    z_net, z_obj, _zr, _b3, _b4 = run(on_o, 0.05, force_all_false=True)
    chk(all(torch.equal(p.data, q.data) for p, q in zip(ref_net.ps, z_net.ps))
        and all(torch.equal(x, y) for x, y in zip(ref_obj.momentum_base, z_obj.momentum_base))
        and all(torch.equal(x, y) for x, y in zip(ref_obj.h_condenced, z_obj.h_condenced)),
        "D3(a) ON with an ALL-FALSE mask == the unpatched SGDm_base_update BITWISE (weights, momentum, h) after %d updates" % NUPD)
    del z_net, z_obj
    on_o, _o3, _e3 = build_obj(HFpost, torch, nps, "scalar", mask_str, AB)
    m_net, m_obj, rec, bad_m, bad_u = run(on_o, 0.05, attach=True, verify=True)
    k = len(on_o._dm_idx)
    chk(bad_m == 0 and bad_u == 0 and k > 0,
        "D3(b) masked tensors: w' == w - a*m and h' == h - a*m BITWISE; unmasked: the original update BITWISE (%d updates x %d tensors, %d masked)" % (NUPD, len(nps), k),
        "bad masked %d, unmasked %d" % (bad_m, bad_u))
    diff_m = min(float((m_net.ps[i].data - ref_net.ps[i].data).abs().max()) for i in on_o._dm_idx)
    chk(diff_m > 0, "D3(c) every masked tensor DIFFERS from the unpatched update after %d updates (min over masked of max|diff| %.3e)" % (NUPD, diff_m))
    keys = sorted(rec)
    chk(keys == sorted(["dm_n", "dm_masked", "dm_skipped", "dm_wdterm", "dm_norm", "dm_absmin", "dm_small"])
        and rec["dm_n"] == NUPD and rec["dm_skipped"] == NUPD * k and rec["dm_masked"] == k and rec["dm_wdterm"] > 0
        and len(rec["dm_norm"]) == k and rec["dm_absmin"] >= 0 and isinstance(rec["dm_small"], int),
        "D3(d) _dm_attach adds only dm_* keys: dm_n == updates, dm_skipped == updates*k, dm_wdterm > 0, one norm per masked tensor",
        "%s" % {kk: (rec[kk] if kk != "dm_norm" else len(rec[kk])) for kk in keys})
    print("\n%s" % ("ALL PASS" if not FAILED else "FAILURES: %r" % FAILED))
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
