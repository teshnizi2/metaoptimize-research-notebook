# =====================================================================================
# test_resnet_gn_c100.py -- correctness AND backward-compatibility suite for
#   PATCH_RESNET_GN_C100 (`--NN-name ResNet18_gn_c100`).  Registered for `cgn1`,
#   CORRECTIONS 212.  Same strategy as tests/test_vggbn2.py: the patched file is
#   compared against the UNPATCHED file on every network the corpus has run, not
#   against my reading of the diff.
#
#   R0  BYTE IDENTITY.  `PATCH_RESNET_GN_C100` absent from --pre, present in --post;
#       deleting the ONE inserted region from --post reproduces --pre BYTE FOR BYTE;
#       and the patch's own pure function patch_text(--pre) reproduces --post exactly.
#   R1  BACKWARD COMPATIBILITY, EVERY CORPUS ARCHITECTURE.  For each `network` value
#       in the corpus (read live from --csv, `cgn1-` rows excluded, the patch's OWN
#       architecture ResNet18_gn_c100 excluded BY ARCHITECTURE -- the 207 lesson) plus
#       a frozen list and extras: identical ordered named_parameters() (name AND
#       shape), identical parameter count, and BITWISE-EQUAL INITIAL VALUES under a
#       fixed torch seed; a name that raises must raise the identical exception type.
#   R2  THE NEW NAME.  Builds; 62 tensors; 11,220,132 parameters; its (name, shape)
#       list is ELEMENTWISE EQUAL to ResNet18_c100's; its initial parameters under a
#       fixed seed are BITWISE EQUAL to ResNet18_c100's; 20 GroupNorm / 0 BatchNorm2d
#       (and the reverse on ResNet18_c100 -- anti-vacuity); every GN has 32 groups over
#       a channel count divisible by 32; 0 buffers (ResNet18_c100: 60); the
#       (conv, norm.weight, norm.bias) triple at every index 1..60 -- class == index
#       mod 3 -- with every 1-D tensor OWNED BY a GroupNorm; tensors 61/62 are
#       linear.weight (100, 512) / linear.bias (100,); 41 one-D tensors; the 512-wide
#       1-D *.weight tensors are exactly 47/50/53/56/59 by name.  Manifest printed.
#   R3  END TO END THROUGH THE HARNESS's HF.init_meta: scalar, layerwise and a sets:
#       two-group spec compose IDENTICALLY on ResNet18_gn_c100 and ResNet18_c100.
#   R4  RULE 20 SAFETY of the new --NN-name string.
#   R5  IT RUNS: one CPU forward/backward, (2,100) logits, finite loss, finite and
#       non-zero gradient on EVERY tensor, and a train-mode output that DIFFERS from
#       ResNet18_c100's at the same seed and input (the normaliser really changed).
#   R6  SCOPE: --pre already carries PATCH_RESNET_GN (the `norm` kwarg the region
#       relies on); HF.py carries no PATCH_RESNET_GN_C100 marker.
#
# RUN (needs torch; CPU):
#   python3 tests/test_resnet_gn_c100.py --pre <tree>/build_network.py.pre_gn_c100 \
#       --post <tree>/build_network.py --hf <tree>/Optimizers/HF.py [--csv results/all_runs.csv]
# =====================================================================================
import sys, os, re, argparse, importlib.util, importlib.machinery, traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = "cgn1"
NEWNAME = "ResNet18_gn_c100"
REF = "ResNet18_c100"
NEW_NETS = [NEWNAME]

# the region, written once here and once in the patch; R0 also asserts the two agree
REGION_HEAD = "\n    # --- PATCH_RESNET_GN_C100: the ONLY new name."
REGION_TAIL = "        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100, norm='gn').to(device)\n"
MARK = "PATCH_RESNET_GN_C100"

CORPUS_NETS = ["ResNet18", "ResNet18_c100", "ResNet34", "ResNet10", "ResNet50", "resnet18",
               "VGG11_bn_c100", "ResNet18_gn", "ResNet10_c100", "ResNet18_tin",
               "ResNet34_c100", "ResNet101"]
EXTRA_NETS = ["M1", "M2", "VGG11_bn", "ResNet18_soft", "ResNet152", "NoSuchNet_xyz", ""]

NTENS = 62
TOTPAR = 11220132
W512 = [(47, "layer4.0.bn1.weight"), (50, "layer4.0.bn2.weight"),
        (53, "layer4.0.shortcut.1.weight"), (56, "layer4.1.bn1.weight"),
        (59, "layer4.1.bn2.weight")]
SETS = "sets:1-49,51-52,54-58,60-62/50,53,59"

FAILED = []


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def load_mod(path, name):
    d = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, d)
    try:
        loader = importlib.machinery.SourceFileLoader(name, os.path.abspath(path))
        spec = importlib.util.spec_from_loader(name, loader)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        loader.exec_module(m)
        return m
    finally:
        sys.path.remove(d)


def manifest(net):
    nps = [(n, tuple(p.shape)) for n, p in net.named_parameters()]
    tot = sum(int(p.numel()) for p in net.parameters())
    return nps, tot


def build_or_exc(mod, name, dev):
    try:
        return ("ok", mod.build_network(name, dev))
    except BaseException as e:
        return ("exc", type(e).__name__)


def owners(net):
    own = {}
    for mn, m in net.named_modules():
        for pn, _p in m.named_parameters(recurse=False):
            own[(mn + "." if mn else "") + pn] = type(m).__name__
    return own


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--hf", default="")
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    a = ap.parse_args()

    import torch
    import torch.nn as nn
    dev = torch.device("cpu")
    print("torch %s   pre=%s   post=%s" % (torch.__version__, a.pre, a.post))
    pre_src = open(a.pre).read()
    post_src = open(a.post).read()

    # ---- R0 ------------------------------------------------------------------------
    print("\nR0  the two files differ ONLY by the one inserted region")
    chk(MARK not in pre_src, "R0a --pre carries no %s marker" % MARK)
    chk(MARK in post_src, "R0b --post carries the %s marker" % MARK)
    chk(post_src.count(REGION_HEAD) == 1, "R0c region head found exactly once",
        str(post_src.count(REGION_HEAD)))
    chk(post_src.count(REGION_TAIL) == 1, "R0d region tail found exactly once",
        str(post_src.count(REGION_TAIL)))
    stripped = post_src
    i = post_src.find(REGION_HEAD)
    j = post_src.find(REGION_TAIL, max(i, 0))
    if 0 <= i < j:
        stripped = post_src[:i] + post_src[j + len(REGION_TAIL):]
    chk(MARK not in stripped, "R0e no %s marker survives the deletion" % MARK)
    chk(stripped == pre_src, "R0f deleting the region reproduces --pre BYTE FOR BYTE",
        "stripped %d bytes, pre %d bytes" % (len(stripped), len(pre_src)))
    pp = os.path.join(REPO, "patches", "patch_resnet_gn_c100.py")
    if os.path.exists(pp):
        spec = importlib.util.spec_from_file_location("_pgn100", pp)
        mp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mp)
        chk(mp.REGION_HEAD == REGION_HEAD and mp.REGION_TAIL == REGION_TAIL and mp.MARK == MARK,
            "R0g the patch's region constants equal this suite's")
        try:
            out = mp.patch_text(pre_src)
            chk(out == post_src, "R0h the patch's patch_text(--pre) reproduces --post EXACTLY",
                "%d vs %d bytes" % (len(out), len(post_src)))
        except Exception as ex:
            chk(False, "R0h the patch's patch_text(--pre) reproduces --post EXACTLY", repr(ex))
    else:
        chk(False, "R0g patches/patch_resnet_gn_c100.py present", pp)

    m_pre = load_mod(a.pre, "_gn100_pre")
    m_post = load_mod(a.post, "_gn100_post")

    # ---- R1 ------------------------------------------------------------------------
    nets = list(CORPUS_NETS)
    if os.path.exists(a.csv):
        import csv as _csv
        seen = []
        for r in _csv.DictReader(open(a.csv)):
            if r.get("run", "").startswith(PREFIX + "-"):
                continue
            v = r.get("network", "")
            if v in NEW_NETS:
                continue
            if v and v not in seen:
                seen.append(v)
        print("\nR1  corpus `network` column (%s- excluded; %s excluded BY ARCHITECTURE): "
              "%d distinct: %s" % (PREFIX, NEWNAME, len(seen), ", ".join(seen)))
        nets = seen + [x for x in CORPUS_NETS if x not in seen]
    else:
        print("\nR1  NOTE: %s not readable; using the frozen list" % a.csv)
    for n in nets + [x for x in EXTRA_NETS if x not in nets]:
        torch.manual_seed(1234)
        k1, v1 = build_or_exc(m_pre, n, dev)
        torch.manual_seed(1234)
        k2, v2 = build_or_exc(m_post, n, dev)
        if k1 == "exc" or k2 == "exc":
            chk(k1 == k2 and v1 == v2, "R1 %-18s raises identically" % repr(n), "%s / %s" % (v1, v2))
            continue
        mp1, t1 = manifest(v1)
        mp2, t2 = manifest(v2)
        same_val = (len(mp1) == len(mp2)
                    and all(torch.equal(x, y) for x, y in zip(v1.parameters(), v2.parameters())))
        chk(mp1 == mp2 and t1 == t2 and same_val,
            "R1 %-18s identical names/shapes/count AND bitwise-equal init" % repr(n),
            "%d tensors, %d params" % (len(mp1), t1))
        del v1, v2

    # ---- R2 ------------------------------------------------------------------------
    print("\nR2  %s builds; LIVE MANIFEST printed in full" % NEWNAME)
    k, g = build_or_exc(m_post, NEWNAME, dev)
    k1, _ = build_or_exc(m_pre, NEWNAME, dev)
    chk(k1 == "exc", "R2a %s does NOT build on --pre (it is new)" % NEWNAME)
    if not chk(k == "ok", "R2b %s builds on --post" % NEWNAME, "" if k == "ok" else g):
        return finish()
    b = m_post.build_network(REF, dev)
    nps, tot = manifest(g)
    rps, rtot = manifest(b)
    own = owners(g)
    names = [n for n, _ in nps]
    for i, (n, s) in enumerate(nps, 1):
        q = 1
        for d in s:
            q *= d
        print("    %3d  %-28s %-18s %9d  %s" % (i, n, s, q, own[n]))
    chk(len(nps) == NTENS, "R2c %d parameter tensors" % NTENS, str(len(nps)))
    chk(tot == TOTPAR, "R2d %d parameters" % TOTPAR, str(tot))
    chk(nps == rps, "R2e (name, shape) list ELEMENTWISE EQUAL to %s's -- same partition object" % REF)
    torch.manual_seed(7)
    pg = [p.detach().clone() for p in m_post.build_network(NEWNAME, dev).parameters()]
    torch.manual_seed(7)
    pb = [p.detach().clone() for p in m_post.build_network(REF, dev).parameters()]
    chk(len(pg) == len(pb) and all(torch.equal(x, y) for x, y in zip(pg, pb)),
        "R2f initial parameters BITWISE EQUAL to %s's under the same seed" % REF)
    ngn = sum(isinstance(m, nn.GroupNorm) for m in g.modules())
    nbn = sum(isinstance(m, nn.BatchNorm2d) for m in g.modules())
    rgn = sum(isinstance(m, nn.GroupNorm) for m in b.modules())
    rbn = sum(isinstance(m, nn.BatchNorm2d) for m in b.modules())
    chk(ngn == 20 and nbn == 0, "R2g %s: 20 GroupNorm, 0 BatchNorm2d" % NEWNAME, "%d / %d" % (ngn, nbn))
    chk(rgn == 0 and rbn == 20, "R2h (anti-vacuity) %s: 0 GroupNorm, 20 BatchNorm2d" % REF,
        "%d / %d" % (rgn, rbn))
    grp = sorted({(m.num_groups, m.num_channels) for m in g.modules() if isinstance(m, nn.GroupNorm)})
    chk(all(ng == 32 and c % 32 == 0 for ng, c in grp)
        and grp == [(32, 64), (32, 128), (32, 256), (32, 512)],
        "R2i every GroupNorm is GroupNorm(32, C), C in 64/128/256/512", str(grp))
    chk(len(list(g.buffers())) == 0 and len(list(b.buffers())) == 60,
        "R2j buffers: %s 0, %s 60 (BN running stats; never parameters)" % (NEWNAME, REF),
        "%d / %d" % (len(list(g.buffers())), len(list(b.buffers()))))
    bad = []
    for i in range(1, 61):
        n, s = nps[i - 1]
        if i % 3 == 1:
            ok = len(s) == 4 and own[n] == "Conv2d" and n.endswith(".weight")
        elif i % 3 == 2:
            ok = len(s) == 1 and own[n] == "GroupNorm" and n.endswith(".weight")
        else:
            ok = len(s) == 1 and own[n] == "GroupNorm" and n.endswith(".bias")
        if not ok:
            bad.append(i)
    chk(not bad, "R2k class == index mod 3 at every index 1..60 (conv / GN scale / GN shift)",
        "violations %s" % bad[:6])
    chk(nps[60] == ("linear.weight", (100, 512)) and nps[61] == ("linear.bias", (100,)),
        "R2l tensors 61, 62 are linear.weight (100,512), linear.bias (100,)", str(nps[60:]))
    n1d = [i for i, (_n, s) in enumerate(nps, 1) if len(s) == 1]
    chk(len(n1d) == 41, "R2m 41 one-dimensional tensors", str(len(n1d)))
    w512 = [(i, n) for i, (n, s) in enumerate(nps, 1) if len(s) == 1 and s[0] == 512
            and n.endswith(".weight")]
    chk(w512 == W512, "R2n 512-wide 1-D *.weight tensors are 47/50/53/56/59 by name", str(w512))

    # ---- R3 ------------------------------------------------------------------------
    print("\nR3  init_meta composes IDENTICALLY on %s and %s" % (NEWNAME, REF))
    hf = a.hf or os.path.join(os.path.dirname(os.path.abspath(a.post)), "Optimizers", "HF.py")
    if not os.path.exists(hf):
        chk(False, "R3 HF.py present", hf)
    else:
        chk(MARK not in open(hf).read(), "R6b HF.py carries no %s marker (optimiser untouched)" % MARK)
        d = os.path.dirname(os.path.dirname(os.path.abspath(hf)))
        sys.path.insert(0, d)
        spec = importlib.util.spec_from_file_location("_gn100_hf", hf)
        mh = importlib.util.module_from_spec(spec)
        sys.modules["_gn100_hf"] = mh
        spec.loader.exec_module(mh)
        HF = mh.HF

        def compose(net, s):
            T = len(list(net.parameters()))
            o = HF.__new__(HF)
            o.num_layers = T
            o._device = dev
            HF.init_meta(o, s, [(n, p.data.size()) for n, p in net.named_parameters()], 1e-6)
            gi = getattr(o, "param_groups_indices", None)
            return (o.stepsize_type, [tuple(x.shape) for x in o.beta], o.len_beta_list,
                    None if gi is None else [list(x) for x in gi])

        for s, typ in (("scalar", "scalar"), ("layerwise", "layerwise"), (SETS, "blockwise")):
            try:
                cg, cb = compose(g, s), compose(b, s)
                ok = cg == cb and cg[0] == typ
                if s == "scalar":
                    ok = ok and cg[1] == [()]
                elif s == "layerwise":
                    ok = ok and cg[1] == [(NTENS,)]
                else:
                    ok = ok and [len(x) for x in cg[3]] == [59, 3] and cg[1] == [(2,)]
                chk(ok, "R3 %-40s composes identically" % s[:40],
                    "type=%s beta=%s len=%d" % (cg[0], cg[1], cg[2]))
            except BaseException as ex:
                chk(False, "R3 %-40s composes identically" % s[:40], "%s: %s" % (type(ex).__name__, ex))

    # ---- R4 ------------------------------------------------------------------------
    print("\nR4  RULE 20 safety of the new --NN-name")
    chk(re.match(r"^[A-Za-z0-9_.]+$", NEWNAME) is not None and not NEWNAME.startswith("-"),
        "R4 %s is one safe shell token" % NEWNAME)

    # ---- R5 ------------------------------------------------------------------------
    print("\nR5  one CPU forward/backward")
    torch.manual_seed(3)
    net = m_post.build_network(NEWNAME, dev)
    torch.manual_seed(3)
    refn = m_post.build_network(REF, dev)
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 100, (4,))
    out = net(x)
    chk(tuple(out.shape) == (4, 100), "R5a output shape (4,100)", str(tuple(out.shape)))
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    gs = [(nm, p.grad) for nm, p in net.named_parameters()]
    chk(all(gg is not None and torch.isfinite(gg).all() for _nm, gg in gs) and torch.isfinite(loss).all(),
        "R5b loss and every gradient finite", "loss %.4f" % float(loss))
    dead = [nm for nm, gg in gs if gg is None or float(gg.abs().sum()) == 0.0]
    chk(not dead, "R5c no tensor has an identically-zero gradient", str(dead[:5]))
    with torch.no_grad():
        o2 = refn(x)
    chk(not torch.allclose(out.detach(), o2), "R5d (anti-vacuity) output DIFFERS from %s's" % REF)

    # ---- R6 ------------------------------------------------------------------------
    print("\nR6  scope")
    chk("PATCH_RESNET_GN" in pre_src and "def _norm2d(norm, c):" in pre_src,
        "R6a --pre already carries PATCH_RESNET_GN (the norm kwarg the region relies on)")
    return finish()


def finish():
    print("\n" + ("ALL PASS" if not FAILED else "FAILURES: %d" % len(FAILED)))
    for f in FAILED:
        print("   FAIL", f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(3)
