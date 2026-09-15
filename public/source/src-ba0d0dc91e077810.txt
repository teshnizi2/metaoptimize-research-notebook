# =====================================================================================
# test_vggbn2.py -- REGISTERED SUCCESSOR to tests/test_vggbn.py (sha c908b769..., FROZEN,
#   committed 5741218's tree; not edited -- RULE 16).  CORRECTIONS 207.
#
# THE ONE CHANGE.  V1 (the pre/post identity check over every `network` value in the
# corpus) now EXCLUDES THE PATCH'S OWN ARCHITECTURES -- NEW_NETS = VGG11_bn,
# VGG11_bn_c100 -- BY ARCHITECTURE.  Those names exist ONLY post-patch BY CONSTRUCTION
# (the patch is what adds them), so pre-patch raises on them and post-patch builds them;
# V1 comparing the two is a check that must fail by design.  The frozen file excluded
# them only by BATCH PREFIX (`cvg1-`), so the first other VGG batch to be ingested
# (cvi1, b9e836e, 2,797 -> 2,809) put VGG11_bn_c100 into V1's list and broke it
# (CORRECTIONS 205.3).  Excluding by architecture means NO future VGG batch can break it.
# The `cvg1-` prefix exclusion is KEPT (now redundant; every cvg1 row is VGG11_bn_c100).
# The new names are still covered, post-patch, by V2 (build + full manifest), V4, V5, V6.
# EVERY OTHER CHECK IS BYTE-IDENTICAL to the frozen file.
# =====================================================================================
"""Correctness AND BACKWARD-COMPATIBILITY tests for PATCH_VGGBN
(`--NN-name VGG11_bn` / `VGG11_bn_c100`).

STRATEGY, copied deliberately from tests/test_namesets.py.  The patch's whole claim is
that it is ADDITIVE.  So the test does not compare the patched code against my
expectation of it; it compares the patched code against the UNPATCHED code, on the
network names and the step-size specs the corpus itself has actually used.

  V0  BYTE IDENTITY.  `PATCH_VGGBN` is absent from --pre and present in --post, and
      DELETING THE TWO INSERTED REGIONS FROM --post REPRODUCES --pre BYTE FOR BYTE.
      The deletion is performed by this file, by the patch's own region markers, so
      "additive" is proved rather than asserted.  Both regions must be found exactly
      once, and no `PATCH_VGGBN` marker may survive the deletion.
  V1  BACKWARD COMPATIBILITY, EVERY ARCHITECTURE THE CORPUS HAS RUN.  For each of the
      eleven `network` values in results/all_runs.csv (plus M1, M2, ResNet18_soft,
      ResNet101/152 and a deliberate typo), build_network is called on --pre and on
      --post and the results are compared as:
        - the full named_parameters() list, NAME AND SHAPE, elementwise, in order;
        - total parameter count;
        - or, for a name that raises, the IDENTICAL exception type.
      Name/shape order is what HF.init_meta consumes POSITIONALLY, so an ordering
      change would silently re-map every beta.  This is the load-bearing check.
  V1b BITWISE INITIAL WEIGHTS.  For ResNet18_c100 under a fixed torch seed, --pre and
      --post produce bitwise-equal initial parameter tensors.  V1 compares shapes;
      this compares VALUES, so an inserted RNG draw could not hide.
  V2  THE NEW NAMES BUILD, and their live manifest is PRINTED IN FULL and asserted:
      26 tensors; the (conv, bn.weight, bn.bias) triple identity `class == index mod 3`
      at every index 1..24; 17 one-dimensional tensors; total parameter counts;
      and AT LEAST THREE 512-wide 1-D `*.weight` tensors, without which a
      carrier-isolation contrast is not statable on this architecture.
  V3  NO RESIDUAL ADDITION.  The source of VGG_bn.forward contains no `+=`, no `+`
      between tensors, and no `shortcut`; the model has no sub-module named
      `shortcut`; and no parameter name contains `shortcut`.  This is the structural
      claim the whole batch rests on, so it is asserted on the class, not argued.
  V4  END TO END THROUGH init_meta, on the LIVE HF.py: `scalar`, `layerwise`,
      `nodewise`, `nodewise1d`, `weightwise`, `chunk777` and a `sets:` two-group spec
      all compose on VGG11_bn_c100, with the beta shapes and group sizes each
      granularity is defined to produce.  `resnet18_blocks` and `[49,13]` must RAISE
      on a 26-tensor model -- loudly, not silently mis-partition.
  V5  RULE 20 SAFETY of the two new NN-name strings: single shell token, no
      whitespace, no quote character, no `=`, does not begin with `-`.
  V6  FORWARD/BACKWARD actually runs: one CPU step on a 2x3x32x32 batch produces a
      finite loss and a finite gradient on EVERY parameter tensor (no dead tensor,
      which under a meta-learned step size would receive a garbage beta).

Any failure means the patched build_network is not additive, or the new network is not
what the batch claims, and every number taken from a VGG11_bn batch is void.

RUN (needs torch; run it on the cluster against the real tree):
  python3 tests/test_vggbn.py --pre  /path/to/build_network.py.pre_vggbn \\
                              --post /path/to/build_network.py \\
                              [--hf /path/to/Optimizers/HF.py] [--csv results/all_runs.csv]
"""
import sys, os, re, argparse, importlib.util, traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- the network names the CORPUS has actually run.  Drawn from the `network` column
# ---- at registration; --csv overrides with a live read.  `resnet18` (lowercase) is in
# ---- the corpus as a legacy label and is NOT a build_network name -- it must raise on
# ---- BOTH sides, which is itself a backward-compatibility fact worth pinning.
CORPUS_NETS = [
    "ResNet18", "ResNet18_c100", "ResNet34", "ResNet10", "ResNet50", "resnet18",
    "ResNet18_gn", "ResNet10_c100", "ResNet18_tin", "ResNet34_c100", "ResNet101",
]
EXTRA_NETS = ["M1", "M2", "ResNet18_soft", "ResNet152", "ResNet34", "NoSuchNet_xyz", ""]

NEW_NETS = ["VGG11_bn", "VGG11_bn_c100"]

# ---- the two inserted regions, written once here and once in the patch; V0 proves the
# ---- deletion of exactly these reproduces --pre.
REGION_A_HEAD = "\n    # --- PATCH_VGGBN: the ONLY new names."
REGION_A_TAIL = "        return VGG_bn(VGG_CFG11, num_classes=100).to(device)\n"
REGION_B_HEAD = "\n\n# --- PATCH_VGGBN ---------------------"

FAILED = []


def _prod(shape):
    k = 1
    for d in shape:
        k *= int(d)
    return k


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label,
                           ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)


def load_mod(path, name):
    """Import a build_network.py under a private module name.

    An explicit SourceFileLoader is required because --pre is conventionally named
    `build_network.py.pre_vggbn`, and spec_from_file_location cannot infer a loader
    from a suffix that is not `.py`.
    """
    import importlib.machinery
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
    """(name, shape) list and total, exactly as HF.init_meta consumes it."""
    nps = [(n, tuple(p.shape)) for n, p in net.named_parameters()]
    tot = 0
    for _n, s in nps:
        k = 1
        for d in s:
            k *= d
        tot += k
    return nps, tot


def build_or_exc(mod, name, dev):
    try:
        return ("ok", mod.build_network(name, dev))
    except BaseException as e:              # the file's idiom is 0/0 -> ZeroDivisionError
        return ("exc", type(e).__name__)


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="build_network.py WITHOUT the patch")
    ap.add_argument("--post", required=True, help="build_network.py WITH the patch")
    ap.add_argument("--hf", default="", help="live Optimizers/HF.py (for V4)")
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    a = ap.parse_args()

    import torch
    dev = torch.device("cpu")
    print("torch %s   pre=%s   post=%s" % (torch.__version__, a.pre, a.post))

    pre_src = open(a.pre).read()
    post_src = open(a.post).read()

    # ---- V0 BYTE IDENTITY -----------------------------------------------------------
    print("\nV0  the two files differ ONLY by the two inserted regions")
    chk("PATCH_VGGBN" not in pre_src, "V0a --pre carries no PATCH_VGGBN marker")
    chk("PATCH_VGGBN" in post_src, "V0b --post carries the PATCH_VGGBN marker")
    stripped = post_src
    iA = stripped.find(REGION_A_HEAD)
    jA = stripped.find(REGION_A_TAIL)
    okA = 0 <= iA < jA
    chk(okA, "V0c region A found exactly once", "at %d..%d" % (iA, jA))
    if okA:
        stripped = stripped[:iA] + stripped[jA + len(REGION_A_TAIL):]
    iB = stripped.find(REGION_B_HEAD)
    chk(iB >= 0, "V0d region B found", "at %d" % iB)
    if iB >= 0:
        # region B is appended at EOF and BEGINS with the two newlines that separate it
        # from the pre-patch file's own trailing newline, so the deletion is a pure
        # truncation -- nothing is re-added.
        stripped = stripped[:iB]
    chk("PATCH_VGGBN" not in stripped, "V0e no PATCH_VGGBN marker survives the deletion")
    chk(stripped == pre_src,
        "V0f deleting the two regions reproduces --pre BYTE FOR BYTE",
        "stripped %d bytes, pre %d bytes" % (len(stripped), len(pre_src)))
    if stripped != pre_src:
        for k in range(min(len(stripped), len(pre_src))):
            if stripped[k] != pre_src[k]:
                print("      first difference at byte %d: %r vs %r"
                      % (k, stripped[k - 40:k + 40], pre_src[k - 40:k + 40]))
                break

    m_pre = load_mod(a.pre, "_bn_pre")
    m_post = load_mod(a.post, "_bn_post")

    # ---- V1 BACKWARD COMPATIBILITY, EVERY CORPUS ARCHITECTURE -----------------------
    nets = list(CORPUS_NETS)
    if os.path.exists(a.csv):
        import csv as _csv
        seen = []
        for r in _csv.DictReader(open(a.csv)):
            if r.get("run", "").startswith("cvg1-"):
                continue                      # exclude our own batch from every reader
            v = r.get("network", "")
            if v in NEW_NETS:
                continue                      # the patch's OWN architectures, BY ARCHITECTURE
            if v and v not in seen:
                seen.append(v)
        print("\nV1  corpus `network` column (cvg1- excluded; the patch's own %s excluded"
              " BY ARCHITECTURE): %d distinct: %s" % ("/".join(NEW_NETS), len(seen), ", ".join(seen)))
        nets = seen
    else:
        print("\nV1  NOTE: %s not readable; using the registered list" % a.csv)
    for n in nets + [x for x in EXTRA_NETS if x not in nets]:
        torch.manual_seed(1234)
        k_pre, v_pre = build_or_exc(m_pre, n, dev)
        torch.manual_seed(1234)
        k_post, v_post = build_or_exc(m_post, n, dev)
        if k_pre == "exc" or k_post == "exc":
            chk(k_pre == k_post and v_pre == v_post,
                "V1 %-16s raises identically" % repr(n), "%s / %s" % (v_pre, v_post))
            continue
        mp, tp = manifest(v_pre)
        mq, tq = manifest(v_post)
        chk(mp == mq and tp == tq,
            "V1 %-16s identical named_parameters() and param count" % repr(n),
            "%d tensors, %d params" % (len(mp), tp))

    # ---- V1b BITWISE INITIAL WEIGHTS -------------------------------------------------
    print("\nV1b bitwise-equal initial weights (an inserted RNG draw could not hide)")
    for n in ("ResNet18_c100", "ResNet18"):
        torch.manual_seed(7)
        p1 = [p.detach().clone() for p in m_pre.build_network(n, dev).parameters()]
        torch.manual_seed(7)
        p2 = [p.detach().clone() for p in m_post.build_network(n, dev).parameters()]
        same = len(p1) == len(p2) and all(torch.equal(x, y) for x, y in zip(p1, p2))
        chk(same, "V1b %-16s initial weights bitwise equal" % repr(n),
            "%d tensors" % len(p1))

    # ---- V2 THE NEW NAMES, LIVE MANIFEST --------------------------------------------
    print("\nV2  the new names build; LIVE MANIFEST printed in full")
    exp_tot = {"VGG11_bn": None, "VGG11_bn_c100": None}
    for n in NEW_NETS:
        k, v = build_or_exc(m_post, n, dev)
        if k != "ok":
            chk(False, "V2 %s builds" % n, str(v))
            continue
        nps, tot = manifest(v)
        exp_tot[n] = tot
        names = [x for x, _ in nps]
        numel = []
        for _x, s in nps:
            q = 1
            for d in s:
                q *= d
            numel.append(q)
        print("    --- %s: %d parameter tensors, %d parameters" % (n, len(nps), tot))
        for i, ((nm, s), q) in enumerate(zip(nps, numel), 1):
            print("    %3d  %-18s %-20s %9d" % (i, nm, s, q))
        chk(len(nps) == 26, "V2 %-14s has 26 parameter tensors" % n, str(len(nps)))
        # the (conv, bn.weight, bn.bias) triple identity, indices 1..24
        cls = {1: "conv", 2: "bnw", 0: "bnb"}
        bad = [i for i in range(1, 25)
               if not ((cls[i % 3] == "conv" and re.match(r"^conv\d+\.weight$", names[i - 1]))
                       or (cls[i % 3] == "bnw" and re.match(r"^bn\d+\.weight$", names[i - 1]))
                       or (cls[i % 3] == "bnb" and re.match(r"^bn\d+\.bias$", names[i - 1])))]
        chk(not bad, "V2 %-14s class == index mod 3 at every index 1..24" % n,
            "violations %s" % bad[:6])
        chk(names[24:] == ["linear.weight", "linear.bias"],
            "V2 %-14s tensors 25,26 are linear.weight, linear.bias" % n, str(names[24:]))
        n1d = [i for i, (_x, s) in enumerate(nps, 1) if len(s) == 1]
        chk(len(n1d) == 17, "V2 %-14s has 17 one-dimensional tensors" % n, str(len(n1d)))
        w512 = [(i, names[i - 1]) for i, (_x, s) in enumerate(nps, 1)
                if len(s) == 1 and s[0] == 512 and names[i - 1].endswith(".weight")]
        chk(len(w512) >= 3,
            "V2 %-14s has >= 3 512-wide 1-D *.weight tensors (isolation is statable)" % n,
            str(w512))
        chk(all(q == 512 for i, _nm in w512 for q in [numel[i - 1]]),
            "V2 %-14s each of those is exactly 512 parameters" % n)
        # convs carry no bias -- so the 1-D tensors are exactly BN params + linear.bias
        chk(not any(re.match(r"^conv\d+\.bias$", x) for x in names),
            "V2 %-14s no conv carries a bias (matches ResNet18)" % n)
    chk(exp_tot["VGG11_bn_c100"] == 9274532,
        "V2 VGG11_bn_c100 total is 9,274,532 parameters", str(exp_tot["VGG11_bn_c100"]))
    chk(exp_tot["VGG11_bn"] == 9228362,
        "V2 VGG11_bn total is 9,228,362 parameters", str(exp_tot["VGG11_bn"]))

    # ---- V3 NO RESIDUAL ADDITION -----------------------------------------------------
    print("\nV3  the structural claim: NO residual addition anywhere")
    import inspect
    fsrc = inspect.getsource(m_post.VGG_bn.forward)
    csrc = inspect.getsource(m_post.VGG_bn)
    chk("+=" not in fsrc, "V3a VGG_bn.forward contains no `+=`")
    chk("shortcut" not in csrc, "V3b the class mentions no `shortcut`")
    body = re.sub(r"#.*", "", fsrc)
    chk(not re.search(r"out\s*=\s*out\s*\+", body) and not re.search(r"\+\s*self\.", body),
        "V3c no tensor addition in forward")
    net = m_post.build_network("VGG11_bn_c100", dev)
    chk(not any("shortcut" in nm for nm, _ in net.named_modules()),
        "V3d no sub-module named shortcut")
    chk(not any("shortcut" in nm for nm, _ in net.named_parameters()),
        "V3e no parameter name contains shortcut")
    # and the contrast: ResNet18 DOES have all of those, so V3 is not vacuous
    rsrc = inspect.getsource(m_post.BasicBlock.forward)
    chk("+=" in rsrc, "V3f (non-vacuity) BasicBlock.forward DOES contain `+=`")

    # ---- V4 END TO END THROUGH init_meta ---------------------------------------------
    print("\nV4  every granularity composes on VGG11_bn_c100 through the LIVE HF.py")
    hf = a.hf or os.path.join(os.path.dirname(os.path.abspath(a.post)), "Optimizers", "HF.py")
    if not os.path.exists(hf):
        print("    SKIP: no HF.py at %s" % hf)
    else:
        d = os.path.dirname(os.path.dirname(os.path.abspath(hf)))
        sys.path.insert(0, d)
        spec = importlib.util.spec_from_file_location("_vgg_hf", hf)
        mh = importlib.util.module_from_spec(spec)
        sys.modules["_vgg_hf"] = mh
        spec.loader.exec_module(mh)
        HF = mh.HF
        net = m_post.build_network("VGG11_bn_c100", dev)
        nps = [(n, p.data.size()) for n, p in net.named_parameters()]
        names = [n for n, _ in nps]
        T = len(nps)
        SETS = ("sets:1-13,15-16,18-19,21-22,24-26/"
                "bn5.weight,bn6.weight,bn7.weight,bn8.weight")
        # expectations DERIVED from the live manifest, not hard-coded, so the check is
        # "init_meta agrees with the granularity's definition on this model".
        exp_n1d = sum(1 if len(s) == 1 else int(s[0]) for _n, s in nps)
        exp_node = sum(int(s[0]) for _n, s in nps)
        exp_ww = sum(int(_prod(s)) for _n, s in nps)
        print("    derived on the live model: T=%d  nodewise groups=%d  "
              "nodewise1d groups=%d  weightwise groups=%d"
              % (T, exp_node, exp_n1d, exp_ww))
        want = {
            "scalar":     "scalar",
            "layerwise":  "layerwise",
            "nodewise":   "nodewise",
            "nodewise1d": "nodewise1d",
            "weightwise": "weightwise",
            "chunk777":   "chunkwise",
            SETS:         "blockwise",
        }
        for spec_s, typ in want.items():
            try:
                o = HF.__new__(HF)
                o.num_layers = T
                o._device = dev
                HF.init_meta(o, spec_s, nps, 1e-6)
                shp = [tuple(b.shape) for b in o.beta]
                lab = spec_s if len(spec_s) < 30 else spec_s[:27] + "..."
                ok = (o.stepsize_type == typ)
                extra = "type=%s len_beta_list=%d" % (o.stepsize_type, o.len_beta_list)
                if spec_s == "scalar":
                    ok = ok and len(o.beta) == 1 and o.beta[0].dim() == 0
                elif spec_s == "layerwise":
                    ok = ok and shp == [(T,)]
                elif spec_s == "nodewise1d":
                    got = sum(int(b.numel()) for b in o.beta)
                    ok = ok and len(o.beta) == T and got == exp_n1d
                    extra += " groups=%d (derived %d)" % (got, exp_n1d)
                elif spec_s == "nodewise":
                    got = sum(int(b.numel()) for b in o.beta)
                    ok = ok and len(o.beta) == T and got == exp_node
                    extra += " groups=%d (derived %d)" % (got, exp_node)
                elif spec_s == "weightwise":
                    got = sum(int(b.numel()) for b in o.beta)
                    ok = ok and got == exp_ww
                    extra += " groups=%d (derived %d)" % (got, exp_ww)
                elif spec_s == SETS:
                    sizes = [len(x) for x in o.param_groups_indices]
                    ok = ok and sizes == [22, 4] and shp == [(2,)]
                    extra += " sizes=%s" % sizes
                chk(ok, "V4 %-32s composes" % lab, extra)
            except BaseException as e:
                chk(False, "V4 %-32s composes" % spec_s[:32],
                    "%s: %s" % (type(e).__name__, e))
        for spec_s in ("resnet18_blocks", "[49,13]", "resnet50_blocks"):
            try:
                o = HF.__new__(HF)
                o.num_layers = T
                o._device = dev
                HF.init_meta(o, spec_s, nps, 1e-6)
                chk(False, "V4 %-32s RAISES on a 26-tensor model" % spec_s, "it did not")
            except BaseException as e:
                chk(True, "V4 %-32s RAISES on a 26-tensor model" % spec_s,
                    type(e).__name__)
        # the sets: group really is the four 512-wide BN scales
        g = HF.polish_the_stepsize_groups(None, SETS, nps)
        idx = tuple(names.index(n) + 1 for n in g[1])
        chk(sorted(g[0] + g[1]) == sorted(names) and len(g[1]) == 4,
            "V4 the sets: spec partitions all 26 tensors, 4 isolated", str(idx))

    # ---- V5 RULE 20 SAFETY ------------------------------------------------------------
    print("\nV5  RULE 20 safety of the two new --NN-name strings")
    for n in NEW_NETS:
        chk(re.match(r"^[A-Za-z0-9_.]+$", n) is not None and not n.startswith("-")
            and "=" not in n and '"' not in n and "'" not in n,
            "V5 %-14s is one safe shell token" % n)

    # ---- V6 IT ACTUALLY RUNS -----------------------------------------------------------
    print("\nV6  one CPU forward/backward: finite loss, finite grad on EVERY tensor")
    for n in NEW_NETS:
        torch.manual_seed(3)
        net = m_post.build_network(n, dev)
        ncls = 10 if n == "VGG11_bn" else 100
        x = torch.randn(2, 3, 32, 32)
        y = torch.randint(0, ncls, (2,))
        out = net(x)
        chk(tuple(out.shape) == (2, ncls), "V6 %-14s output shape (2,%d)" % (n, ncls),
            str(tuple(out.shape)))
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        gs = [(nm, p.grad) for nm, p in net.named_parameters()]
        chk(all(g is not None for _nm, g in gs),
            "V6 %-14s every tensor received a gradient" % n)
        chk(all(g is not None and torch.isfinite(g).all() for _nm, g in gs)
            and torch.isfinite(loss).all(),
            "V6 %-14s loss and all gradients finite" % n, "loss %.4f" % float(loss))
        dead = [nm for nm, g in gs if g is not None and float(g.abs().sum()) == 0.0]
        chk(not dead, "V6 %-14s no tensor has an identically-zero gradient" % n,
            str(dead[:5]))

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
