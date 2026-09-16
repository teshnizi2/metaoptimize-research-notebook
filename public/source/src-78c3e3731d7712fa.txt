"""Correctness AND BACKWARD-COMPATIBILITY tests for PATCH_PLAINNET
(`--NN-name PlainNet18_c100`: ResNet18_c100 with its residual additions removed).

STRATEGY, copied deliberately from tests/test_vggbn.py / tests/test_namesets.py.  The
patch's whole claim is that it is ADDITIVE and that the new network is ResNet18_c100
MINUS ONE VARIABLE.  So nothing here compares the patched code with my expectation of
it: backward compatibility is checked against the UNPATCHED file on every network the
corpus has actually run, and the one-variable claim is checked FUNCTIONALLY against
ResNet18_c100's own modules.

  P0  BYTE IDENTITY (additivity).  `PATCH_PLAINNET` is absent from --pre and present in
      --post exactly once; --post STARTS WITH --pre (a pure append); and DELETING THE
      ONE REGION -- from its marker line to end of file -- reproduces --pre BYTE FOR
      BYTE.  The deletion is done here, by the patch's own marker.
  P1  BACKWARD COMPATIBILITY, EVERY `network` VALUE IN THE CORPUS (read live from
      --csv; the patch's OWN architecture excluded BY ARCHITECTURE, the 207 lesson)
      plus M1, M2, ResNet18_soft, ResNet34, ResNet152, VGG11_bn, near-miss spellings
      and a typo.  Under an identical torch seed, --pre and --post give:
        - the full named_parameters() list, NAME AND SHAPE, in order (HF.init_meta
          consumes it POSITIONALLY);
        - total parameter count;
        - BITWISE-equal initial parameters AND buffers;
        - an identical torch RNG state after the build (no inserted draw);
        - an identical module repr;
      or, for a name that raises, the IDENTICAL exception type.
  P2  THE NEW MODEL'S MANIFEST, printed in full and asserted: 53 tensors; its ordered
      (name, shape) list IS ResNet18_c100's with exactly the 9 projection-shortcut
      tensors deleted (198); the triple identity `class == index mod 3` at 1..51;
      linear.weight/linear.bias at 52/53; the four 512-wide BN scales at 41/44/47/50;
      and the map from ResNet18_c100's carrier indices 50/53/59 to PlainNet's.
  P3  NO RESIDUAL ADDITION, and ONE VARIABLE ONLY.
      P3a-c  source and module tree: no `+=`/`+` in PlainBlock.forward, no module,
             parameter or buffer named `shortcut`.
      P3d    torch.fx trace: ZERO add ops on PlainNet18_c100; EIGHT on ResNet18_c100
             (non-vacuity -- one per BasicBlock).
      P3e    FUNCTIONAL: load ResNet18_c100's own weights and buffers into
             PlainNet18_c100; its output is BITWISE equal to a reference forward that
             walks ResNet18_c100's modules with every `out += shortcut(x)` removed --
             in eval AND train mode -- while the same walker WITH the addition
             reproduces ResNet18_c100 bitwise (the walker is validated), and the
             PlainNet output DIFFERS from ResNet18_c100's (the addition mattered).
  P4  GRADIENTS.  One CPU step: finite loss; every one of the 53 tensors gets a
      finite, non-None, not-identically-zero gradient.  NEGATIVE CONTROL (198's
      pricing, measured): a naive block that deletes ONLY the addition, keeping the
      shortcut modules, leaves EXACTLY the 9 shortcut tensors with grad None.
  P5  END TO END THROUGH THE LIVE HF.py: `scalar`, `layerwise` and the batch's three
      `sets:` specs compose, with the group sizes, isolated names and beta shapes the
      design requires; `resnet18_blocks`, `resnet50_blocks` and ciso1's/cdep1's ResNet
      ISO string (which names `layer4.0.shortcut.1.weight`) RAISE on the 53-tensor
      model -- loudly, not silently mis-partition.  The three specs are cross-checked
      against the registered scorer's SPEC table when it is reachable.
  P6  RULE 20 SAFETY of the new NN-name and the three spec strings.
  P7  THE DISPATCH: --pre raises ZeroDivisionError on `PlainNet18_c100`; --post
      builds a `ResNet` whose eight blocks are all `PlainBlock`; --post's
      `build_network` is the wrapper and `_build_network_pre_plainnet` is the
      original function object.

Any failure means the patched build_network is not additive, or the new network is
not ResNet18_c100 minus the residual addition, and every number from a
PlainNet18_c100 batch is void.

RUN (needs torch; run it on the cluster against the real tree):
  python3 tests/test_plainnet.py --pre  /path/to/build_network.py.pre_plainnet \\
                                 --post /path/to/build_network.py \\
                                 [--hf /path/to/Optimizers/HF.py] [--csv results/all_runs.csv]
"""
import argparse
import ast
import importlib.machinery
import importlib.util
import operator
import os
import re
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARK = "PATCH_PLAINNET"
REGION_HEAD = "\n\n# --- PATCH_PLAINNET ---"
NEW_NET = "PlainNet18_c100"
PARENT = "ResNet18_c100"
PREFIX = "cpl1-"

# registered corpus list, used only if --csv is unreadable
CORPUS_NETS = [
    "ResNet18", "ResNet18_c100", "ResNet34", "ResNet10", "ResNet50", "resnet18",
    "ResNet18_gn", "ResNet10_c100", "ResNet18_tin", "ResNet34_c100", "ResNet101",
    "VGG11_bn_c100",
]
EXTRA_NETS = ["M1", "M2", "ResNet18_soft", "ResNet34", "ResNet152", "VGG11_bn",
              "PlainNet18", "plainnet18_c100", "PlainNet18_c10", "NoSuchNet_xyz", ""]

SHORTCUT_9 = ["layer%d.0.shortcut.%s" % (s, k)
              for s in (2, 3, 4) for k in ("0.weight", "1.weight", "1.bias")]

# the batch's three m=2 specs -- LITERAL here, in the scorer's SPEC table, in the
# launcher and in tests/test_probe_tensor_blockwise_cpl1.py; P5 cross-checks the scorer.
SPECS = {
    "ISO": "sets:1-43,45-49,51-53/layer4.0.bn2.weight,layer4.1.bn2.weight",
    "CTL": "sets:1-40,42-46,48-53/layer4.0.bn1.weight,layer4.1.bn1.weight",
    "ONE": "sets:1-43,45-53/layer4.0.bn2.weight",
}
WANT_ISOLATED = {
    "ISO": ["layer4.0.bn2.weight", "layer4.1.bn2.weight"],
    "CTL": ["layer4.0.bn1.weight", "layer4.1.bn1.weight"],
    "ONE": ["layer4.0.bn2.weight"],
}
WANT_SIZES = {"ISO": [51, 2], "CTL": [51, 2], "ONE": [52, 1]}
BN512 = [(41, "layer4.0.bn1.weight"), (44, "layer4.0.bn2.weight"),
         (47, "layer4.1.bn1.weight"), (50, "layer4.1.bn2.weight")]
# ResNet18_c100 1-based index -> (name, PlainNet18_c100 1-based index or None)
CARRIER_MAP = {50: ("layer4.0.bn2.weight", 44), 53: ("layer4.0.shortcut.1.weight", None),
               59: ("layer4.1.bn2.weight", 50), 47: ("layer4.0.bn1.weight", 41),
               56: ("layer4.1.bn1.weight", 47)}
RESNET_ISO_SPECS = [
    "sets:1-49,51-52,54-58,60-62/layer4.0.bn2.weight,layer4.0.shortcut.1.weight,layer4.1.bn2.weight",
    "sets:1-46,48-55,57-62/layer4.0.bn1.weight,layer4.1.bn1.weight",
]
SCORER = os.path.join(REPO, "analysis", "cPL1_plainnet_residual_score.py")

FAILED = []


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label,
                           ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)
    return bool(cond)


def load_mod(path, name):
    """Import a build_network.py under a private module name (SourceFileLoader, because
    --pre is conventionally named `build_network.py.pre_plainnet`)."""
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


def numel(shape):
    k = 1
    for d in shape:
        k *= int(d)
    return k


def manifest(net):
    nps = [(n, tuple(p.shape)) for n, p in net.named_parameters()]
    return nps, sum(numel(s) for _n, s in nps)


def build_or_exc(mod, name, dev):
    try:
        return ("ok", mod.build_network(name, dev))
    except BaseException as e:               # the file's idiom is 0/0 -> ZeroDivisionError
        return ("exc", type(e).__name__)


def ref_forward(res, x, with_add):
    """ResNet18_c100's forward, written out module by module, with or without the
    residual addition.  Used ONLY to prove PlainNet18_c100 == ResNet18_c100 - addition."""
    import torch.nn.functional as F
    out = F.relu(res.bn1(res.conv1(x)))
    for layer in (res.layer1, res.layer2, res.layer3, res.layer4):
        for blk in layer:
            o = F.relu(blk.bn1(blk.conv1(out)))
            o = blk.bn2(blk.conv2(o))
            if with_add:
                o += blk.shortcut(out)
            out = F.relu(o)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    return res.linear(out)


def count_adds(net):
    import torch
    import torch.fx
    gm = torch.fx.symbolic_trace(net)
    n = 0
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in (operator.add, operator.iadd,
                                                          torch.add):
            n += 1
        elif node.op == "call_method" and node.target in ("add", "add_", "__add__",
                                                          "__iadd__"):
            n += 1
    return n


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="build_network.py WITHOUT the patch")
    ap.add_argument("--post", required=True, help="build_network.py WITH the patch")
    ap.add_argument("--hf", default="", help="live Optimizers/HF.py (for P5)")
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    a = ap.parse_args()

    import torch
    dev = torch.device("cpu")
    print("torch %s   pre=%s   post=%s" % (torch.__version__, a.pre, a.post))
    pre_src = open(a.pre).read()
    post_src = open(a.post).read()

    # ---- P0 BYTE IDENTITY -----------------------------------------------------------
    print("\nP0  the patch is ONE pure append; deleting it reproduces --pre byte for byte")
    chk(MARK not in pre_src, "P0a --pre carries no PATCH_PLAINNET marker")
    chk(post_src.count(REGION_HEAD) == 1, "P0b --post carries the region marker EXACTLY once",
        "count %d" % post_src.count(REGION_HEAD))
    chk(post_src.startswith(pre_src), "P0c --post STARTS WITH --pre (nothing above EOF changed)",
        "pre %d bytes, post %d bytes" % (len(pre_src), len(post_src)))
    i = post_src.find(REGION_HEAD)
    stripped = post_src[:i] if i >= 0 else post_src
    chk(MARK not in stripped, "P0d no PATCH_PLAINNET marker survives the deletion")
    chk(stripped == pre_src, "P0e deleting the region reproduces --pre BYTE FOR BYTE",
        "stripped %d bytes, pre %d bytes" % (len(stripped), len(pre_src)))
    if stripped != pre_src:
        for k in range(min(len(stripped), len(pre_src))):
            if stripped[k] != pre_src[k]:
                print("      first difference at byte %d: %r vs %r"
                      % (k, stripped[k - 40:k + 40], pre_src[k - 40:k + 40]))
                break
    region = post_src[i:] if i >= 0 else ""
    code = "\n".join(re.sub(r"#.*", "", ln) for ln in region.splitlines())
    chk("0/0" not in code and "build_network" in code,
        "P0f the region's CODE (comments stripped) adds no new failure path (no 0/0)")

    m_pre = load_mod(a.pre, "_pn_pre")
    m_post = load_mod(a.post, "_pn_post")

    # ---- P1 BACKWARD COMPATIBILITY -------------------------------------------------
    nets = list(CORPUS_NETS)
    if os.path.exists(a.csv):
        import csv as _csv
        seen = []
        for r in _csv.DictReader(open(a.csv)):
            if r.get("run", "").startswith(PREFIX):
                continue                               # our own batch, by prefix
            v = r.get("network", "")
            if v == NEW_NET:
                continue                               # the patch's OWN architecture (207)
            if v and v not in seen:
                seen.append(v)
        print("\nP1  corpus `network` column (%s excluded, %s excluded by ARCHITECTURE): "
              "%d distinct: %s" % (PREFIX, NEW_NET, len(seen), ", ".join(seen)))
        nets = seen
    else:
        print("\nP1  NOTE: %s not readable; using the registered list" % a.csv)
    todo = nets + [x for x in EXTRA_NETS if x not in nets]
    for n in todo:
        torch.manual_seed(1234)
        k_pre, v_pre = build_or_exc(m_pre, n, dev)
        r_pre = torch.rand(4)
        torch.manual_seed(1234)
        k_post, v_post = build_or_exc(m_post, n, dev)
        r_post = torch.rand(4)
        if k_pre == "exc" or k_post == "exc":
            chk(k_pre == k_post and v_pre == v_post,
                "P1 %-18s raises identically" % repr(n), "%s / %s" % (v_pre, v_post))
            continue
        mp, tp = manifest(v_pre)
        mq, tq = manifest(v_post)
        same_p = all(torch.equal(x, y) for x, y in zip(v_pre.parameters(), v_post.parameters()))
        bp = list(v_pre.named_buffers())
        bq = list(v_post.named_buffers())
        same_b = ([k for k, _ in bp] == [k for k, _ in bq]
                  and all(torch.equal(x, y) for (_k, x), (_l, y) in zip(bp, bq)))
        chk(mp == mq and tp == tq and same_p and same_b and torch.equal(r_pre, r_post)
            and repr(v_pre) == repr(v_post),
            "P1 %-18s identical names/shapes/order, count, BITWISE params+buffers, RNG state, repr"
            % repr(n), "%d tensors, %d params" % (len(mp), tp))
        del v_pre, v_post

    # ---- P2 THE NEW MODEL ------------------------------------------------------------
    print("\nP2  %s: LIVE MANIFEST printed in full" % NEW_NET)
    net = m_post.build_network(NEW_NET, dev)
    par = m_post.build_network(PARENT, dev)
    nps, tot = manifest(net)
    pps, ptot = manifest(par)
    names = [x for x, _ in nps]
    pnames = [x for x, _ in pps]
    for k, ((nm, s)) in enumerate(nps, 1):
        print("    %3d  %-28s %-20s %9d" % (k, nm, s, numel(s)))
    print("    --- %s: %d parameter tensors, %d parameters   (%s: %d / %d)"
          % (NEW_NET, len(nps), tot, PARENT, len(pps), ptot))
    chk(len(nps) == 53, "P2a %s has 53 parameter tensors" % NEW_NET, str(len(nps)))
    removed = [x for x in pnames if x not in names]
    chk(len(pps) == 62 and removed == SHORTCUT_9 and all(x in pnames for x in names),
        "P2b the tensors ResNet18_c100 has and PlainNet lacks are EXACTLY the 9 shortcut tensors",
        str(removed))
    chk(nps == [e for e in pps if e[0] not in SHORTCUT_9],
        "P2c PlainNet's ordered (name, shape) list == ResNet18_c100's with the 9 deleted")
    sc_par = sum(numel(s) for nm, s in pps if nm in SHORTCUT_9)
    chk(tot == ptot - sc_par, "P2d total = %d - %d (the 9 shortcut tensors) = %d"
        % (ptot, sc_par, ptot - sc_par), str(tot))
    pat = {1: r"^(conv1|layer\d\.\d\.conv[12])\.weight$",
           2: r"^(bn1|layer\d\.\d\.bn[12])\.weight$",
           0: r"^(bn1|layer\d\.\d\.bn[12])\.bias$"}
    bad = [k for k in range(1, 52) if not re.match(pat[k % 3], names[k - 1])]
    chk(not bad, "P2e class == index mod 3 at EVERY index 1..51 (conv, bn.weight, bn.bias)",
        "violations %s" % bad[:6])
    chk(names[51:] == ["linear.weight", "linear.bias"],
        "P2f tensors 52, 53 are linear.weight, linear.bias", str(names[51:]))
    w512 = [(k, names[k - 1]) for k, (_n, s) in enumerate(nps, 1)
            if len(s) == 1 and s[0] == 512 and names[k - 1].endswith(".weight")]
    chk(w512 == BN512, "P2g the four 512-wide BN scales are at 41/44/47/50", str(w512))
    for ri, (nm, pi) in sorted(CARRIER_MAP.items()):
        ok = pnames[ri - 1] == nm and ((pi is None and nm not in names)
                                       or (pi is not None and names.index(nm) + 1 == pi))
        chk(ok, "P2h ResNet18_c100 idx %d %-27s -> PlainNet idx %s" % (ri, nm, pi))
    chk(not any(re.match(r"^conv\d+\.bias$|\.conv\d\.bias$", x) for x in names),
        "P2i no conv carries a bias (the 1-D tensors are BN params + linear.bias)")

    # ---- P3 NO RESIDUAL ADDITION, ONE VARIABLE ONLY ---------------------------------
    print("\nP3  NO residual addition, and PlainNet18_c100 == ResNet18_c100 minus the addition")
    import inspect
    fsrc = re.sub(r"#.*", "", inspect.getsource(m_post.PlainBlock.forward))
    csrc = re.sub(r"#.*", "", inspect.getsource(m_post.PlainBlock))
    chk("+=" not in fsrc and not re.search(r"\+", fsrc), "P3a PlainBlock.forward has no `+=` and no `+`")
    chk("shortcut" not in csrc, "P3b PlainBlock (comments stripped) never mentions `shortcut`")
    chk(not any("shortcut" in x for x, _ in net.named_modules())
        and not any("shortcut" in x for x, _ in net.named_parameters())
        and not any("shortcut" in x for x, _ in net.named_buffers()),
        "P3c no module, parameter or buffer name contains `shortcut`")
    blocks = [b for L in (net.layer1, net.layer2, net.layer3, net.layer4) for b in L]
    chk(len(blocks) == 8 and all(type(b).__name__ == "PlainBlock" for b in blocks)
        and not any(hasattr(b, "shortcut") for b in blocks),
        "P3c' eight blocks, all PlainBlock, none has a `shortcut` attribute")
    try:
        na, nr = count_adds(net), count_adds(par)
        chk(na == 0, "P3d torch.fx: ZERO add ops in the traced %s graph" % NEW_NET, str(na))
        chk(nr == 8, "P3d' torch.fx: EIGHT add ops in the traced %s graph (non-vacuity)" % PARENT,
            str(nr))
    except Exception as e:
        chk(False, "P3d torch.fx trace", "%s: %s" % (type(e).__name__, e))
    rsrc = inspect.getsource(m_post.BasicBlock.forward)
    chk("+=" in rsrc, "P3d'' (non-vacuity) BasicBlock.forward DOES contain `+=`")
    # functional one-variable proof
    torch.manual_seed(5)
    res = m_post.build_network(PARENT, dev)
    pln = m_post.build_network(NEW_NET, dev)
    rsd = res.state_dict()
    psd = pln.state_dict()
    extra = sorted(set(rsd) - set(psd))
    chk(set(psd) <= set(rsd) and all(("shortcut" in k) for k in extra),
        "P3e every PlainNet state entry exists in ResNet18_c100; the rest are shortcut entries",
        "%d extra, all shortcut" % len(extra))
    pln.load_state_dict(dict((k, rsd[k].clone()) for k in psd), strict=True)
    g = torch.Generator().manual_seed(11)
    x = torch.randn(6, 3, 32, 32, generator=g)
    for mode in ("eval", "train"):
        # fresh copies so train-mode running-stat updates cannot leak between calls
        import copy
        r1, r2 = copy.deepcopy(res), copy.deepcopy(res)
        p1 = copy.deepcopy(pln)
        for m in (r1, r2, p1):
            m.train(mode == "train")
        with torch.no_grad():
            y_res = r1(x)
            y_ref_add = ref_forward(r2, x, True)
            y_pln = p1(x)
            r3 = copy.deepcopy(res)
            r3.train(mode == "train")
            y_ref_noadd = ref_forward(r3, x, False)
        chk(torch.equal(y_res, y_ref_add),
            "P3f [%s] the reference walker WITH the addition reproduces ResNet18_c100 bitwise" % mode)
        chk(torch.equal(y_pln, y_ref_noadd),
            "P3g [%s] PlainNet18_c100 == ResNet18_c100's modules with the addition removed, BITWISE"
            % mode, "max|d| %.3e" % float((y_pln - y_ref_noadd).abs().max()))
        chk(not torch.equal(y_pln, y_res),
            "P3h [%s] ... and differs from ResNet18_c100 (the addition mattered)" % mode,
            "max|d| %.3e" % float((y_pln - y_res).abs().max()))

    # ---- P4 GRADIENTS ------------------------------------------------------------------
    print("\nP4  one CPU forward/backward: every tensor gets a real gradient; 198's naive "
          "deletion measured")
    torch.manual_seed(3)
    net = m_post.build_network(NEW_NET, dev)
    xb = torch.randn(4, 3, 32, 32)
    yb = torch.randint(0, 100, (4,))
    out = net(xb)
    chk(tuple(out.shape) == (4, 100), "P4a output shape (4,100)", str(tuple(out.shape)))
    loss = torch.nn.functional.cross_entropy(out, yb)
    loss.backward()
    gs = [(nm, p.grad) for nm, p in net.named_parameters()]
    chk(all(gr is not None for _n, gr in gs), "P4b all 53 tensors received a gradient")
    chk(torch.isfinite(loss).all() and all(gr is not None and torch.isfinite(gr).all()
                                           for _n, gr in gs),
        "P4c loss and all gradients finite", "loss %.4f" % float(loss))
    dead = [nm for nm, gr in gs if gr is not None and float(gr.abs().sum()) == 0.0]
    chk(not dead, "P4d no tensor has an identically-zero gradient", str(dead[:5]))

    class NaiveBlock(m_post.BasicBlock):
        """BasicBlock with ONLY the addition deleted -- the object 198 priced as wrong."""
        def forward(self, x):
            import torch.nn.functional as F
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out)

    torch.manual_seed(3)
    naive = m_post.ResNet(NaiveBlock, [2, 2, 2, 2], num_classes=100)
    torch.nn.functional.cross_entropy(naive(xb), yb).backward()
    nog = [nm for nm, p in naive.named_parameters() if p.grad is None]
    chk(len(list(naive.parameters())) == 62 and nog == SHORTCUT_9,
        "P4e NEGATIVE CONTROL: deleting only the addition leaves EXACTLY the 9 shortcut "
        "tensors with NO gradient (62 tensors, 9 dead) -- why the modules must not exist",
        str(nog))

    # ---- P5 END TO END THROUGH HF.init_meta --------------------------------------------
    print("\nP5  the batch's specs compose on %s through the LIVE HF.py" % NEW_NET)
    hf = a.hf or os.path.join(os.path.dirname(os.path.abspath(a.post)), "Optimizers", "HF.py")
    if not os.path.exists(hf):
        chk(False, "P5 HF.py reachable", hf)
    else:
        d = os.path.dirname(os.path.dirname(os.path.abspath(hf)))
        sys.path.insert(0, d)
        spec = importlib.util.spec_from_file_location("_pn_hf", hf)
        mh = importlib.util.module_from_spec(spec)
        sys.modules["_pn_hf"] = mh
        spec.loader.exec_module(mh)
        HF = mh.HF
        net = m_post.build_network(NEW_NET, dev)
        nps = [(n, p.data.size()) for n, p in net.named_parameters()]
        names = [n for n, _ in nps]
        T = len(nps)
        nu = [int(p.numel()) for _n, p in net.named_parameters()]

        def meta(s):
            o = HF.__new__(HF)
            o.num_layers = T
            o._device = dev
            HF.init_meta(o, s, nps, 1e-6)
            return o
        o = meta("scalar")
        chk(o.stepsize_type == "scalar" and len(o.beta) == 1 and o.beta[0].dim() == 0
            and not getattr(o, "_rednorm", False), "P5a scalar -> one 0-dim beta, no rednorm")
        o = meta("layerwise")
        chk(o.stepsize_type == "layerwise" and [tuple(b.shape) for b in o.beta] == [(T,)],
            "P5b layerwise -> one beta of shape (%d,)" % T)
        for arm, s in SPECS.items():
            try:
                grp = HF.polish_the_stepsize_groups(None, s, nps)
                o = meta(s)
                sizes = [len(x) for x in grp]
                iso = list(grp[1]) if len(grp) == 2 else []
                pc = [sum(nu[names.index(n)] for n in x) for x in grp]
                chk(len(grp) == 2 and sizes == WANT_SIZES[arm] and iso == WANT_ISOLATED[arm]
                    and sorted(grp[0] + grp[1]) == sorted(names)
                    and o.stepsize_type == "blockwise"
                    and [len(x) for x in o.param_groups_indices] == sizes
                    and [tuple(b.shape) for b in o.beta] == [(2,)]
                    and not getattr(o, "_rednorm", False),
                    "P5c %-3s composes: sizes %s, isolated %s, params %s"
                    % (arm, sizes, [names.index(n) + 1 for n in iso], pc))
            except BaseException as e:
                chk(False, "P5c %s composes" % arm, "%s: %s" % (type(e).__name__, e))
        for s in ["resnet18_blocks", "resnet50_blocks"] + RESNET_ISO_SPECS:
            try:
                meta(s)
                chk(False, "P5d %-40s RAISES on a 53-tensor model" % s[:40], "it did not")
            except BaseException as e:
                chk(True, "P5d %-40s RAISES on a 53-tensor model" % s[:40], type(e).__name__)
    if os.path.exists(SCORER):
        m = re.search(r"^SPEC = (\{.*?^\})", open(SCORER).read(), re.S | re.M)
        theirs = ast.literal_eval(m.group(1)) if m else {}
        chk(all(theirs.get(k) == v for k, v in SPECS.items()),
            "P5e the three specs are byte-identical to the registered scorer's SPEC table",
            os.path.basename(SCORER))
    else:
        print("  NOTE P5e scorer not reachable at %s -- cross-check skipped" % SCORER)

    # ---- P6 RULE 20 ------------------------------------------------------------------
    print("\nP6  RULE 20 safety")
    for s in [NEW_NET] + list(SPECS.values()):
        chk(re.match(r"^[A-Za-z0-9_.,:/-]+$", s) is not None and not s.startswith("-")
            and "=" not in s, "P6 %-40s is one safe shell token" % s[:40])

    # ---- P7 THE DISPATCH ---------------------------------------------------------------
    print("\nP7  the dispatch")
    k, v = build_or_exc(m_pre, NEW_NET, dev)
    chk(k == "exc" and v == "ZeroDivisionError",
        "P7a --pre raises ZeroDivisionError on %s (the name is new)" % NEW_NET, str(v))
    k, v = build_or_exc(m_post, NEW_NET, dev)
    chk(k == "ok" and type(v).__name__ == "ResNet", "P7b --post builds a ResNet instance", str(type(v)))
    chk(getattr(m_post, "_build_network_pre_plainnet", None) is not None
        and m_post.build_network is not m_post._build_network_pre_plainnet
        and m_post._build_network_pre_plainnet.__code__.co_firstlineno
        == m_pre.build_network.__code__.co_firstlineno,
        "P7c --post's build_network is the wrapper; the original is kept, same source line")

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
