"""Correctness AND BACKWARD-COMPATIBILITY tests for PATCH_NAMESETS
(`--stepsize-groups sets:<g1>/<g2>/...`).

STRATEGY.  The patch's whole claim is that it is ADDITIVE.  So the test does not
compare the patched code against my expectation of it; it compares the patched code
against the UNPATCHED code, on the specs the corpus itself has actually used.

  N0  the two HF.py files under test differ ONLY by the patch: `PATCH_NAMESETS` is
      absent from the pre file and present in the post file, and deleting the
      inserted regions from the post file reproduces the pre file BYTE FOR BYTE.
  N1  BACKWARD COMPATIBILITY.  For every spec drawn from the corpus's own
      `granularity` column (plus the two `*_blocks` names and a hand-made
      list-of-lists), `polish_the_stepsize_groups` returns the IDENTICAL object
      before and after the patch -- or raises the identical exception type before
      and after.  Group membership is compared as nested lists of strings, so the
      comparison is byte-level on every name.
  N2  THE INERTNESS EQUALITY.  `sets:1-49/50-62` on the PATCHED code equals
      `[49,13]` on the UNPATCHED code, exactly.  Same for `sets:1-31/32-62` vs
      `[31,31]` and `sets:1-2/3-62` vs `[2,60]`.  This is what licenses running a
      prefix arm through the new grammar and calling it the same partition.
  N3  END TO END THROUGH init_meta.  For that same pair, `param_groups_indices`,
      `map_layers_to_blocks`, `stepsize_type` and `beta` shape are identical.  N2
      makes this a formality, but the formality is the thing the optimizer actually
      consumes, so it is asserted rather than argued.
  N4  THE SWAP ARMS.  The three `cpr1` specs compose, on the LIVE model, to the
      partitions the batch claims: group sizes [49,13] in all three, and coarse-set
      symmetric differences against the prefix arm of EXACTLY the intended pair.
  N5  THE GRAMMAR IS TOTAL AND LOUD.  Every malformed spec raises ValueError:
      duplicate tensor, missing tensor, out-of-range index, inverted range, unknown
      name, empty group, empty item, empty spec.
  N6  NO NAME/INDEX AMBIGUITY on this model: no parameter name matches `\\d+` or
      `\\d+-\\d+`, so the parser's range-then-int-then-name order is well defined.
  N7  RULE 20 SAFETY of the literal spec strings: single shell token, no whitespace,
      no quote character, no `=`, does not begin with `-`, and round-trips through
      analysis/argsline_guard.py's own tokenizer as ONE value.

Any failure means a partition is not what its spec says, and every number taken from
a `sets:` batch is void.

RUN (needs torch; run it on the cluster against the real tree):
  python3 tests/test_namesets.py --pre  /path/to/HF.py.pre_namesets \\
                                --post /path/to/Optimizers/HF.py \\
                                [--cifar-dir /path/to/cifar10] \\
                                [--specs-file specs.txt]
"""
import sys, os, re, argparse, importlib.util, traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- the specs the CORPUS has actually used.  Drawn from results/all_runs.csv's
# `granularity` column at registration time; --specs-file overrides with a live read.
CORPUS_SPECS = [
    "scalar", "layerwise", "nodewise", "weightwise", "nodewise1d",
    "resnet18_blocks", "resnet50_blocks",
    "chunk1", "chunk2", "chunk16", "chunk128", "chunk295", "chunk771", "chunk777",
    "chunk835", "chunk884", "chunk1024", "chunk2293", "chunk2325", "chunk2500",
    "chunk8192", "chunk65536",
    "permnode0", "permnode1", "permnode2", "permnode101", "permnode202", "permnode303",
    "[2,60]", "[16,16,15,15]", "[17,45]", "[24,38]", "[31,31]", "[38,24]", "[42,20]",
    "[45,17]", "[46,16]", "[47,15]", "[48,14]", "[49,13]", "[50,12]", "[51,11]",
    "[52,10]", "[53,9]", "[54,8]", "[55,7]", "[60,2]",
    "[8,8,8,8,8,8,7,7]", "[4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3]",
    "[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1]",
]

# ---- cpr1's own arms, written once here and once in the launcher; N4 proves them.
SPEC_PREFIX_LEGACY = "[49,13]"
SPEC_PREFIX_NEW = "sets:1-49/50-62"
SPEC_SWAP = "sets:1-48,layer4.0.bn2.weight/layer4.0.conv2.weight,51-62"
SPEC_CTRL = "sets:1-48,layer4.0.shortcut.0.weight/layer4.0.conv2.weight,50-51,53-62"

FAILED = []


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label,
                           ("   " + extra) if extra else ""))
    if not cond:
        FAILED.append(label)


def load_hf(path, tag):
    spec = importlib.util.spec_from_file_location("hf_" + tag, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hf_" + tag] = mod
    spec.loader.exec_module(mod)
    return mod


def polish(mod, spec, nps):
    """Return ('ok', value) or ('exc', ExceptionClassName)."""
    try:
        return ("ok", mod.HF.polish_the_stepsize_groups(None, spec, nps))
    except Exception as e:               # noqa: BLE001 -- the comparison IS the test
        return ("exc", type(e).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="UNPATCHED HF.py")
    ap.add_argument("--post", required=True, help="PATCHED HF.py")
    ap.add_argument("--cifar-dir", default=os.environ.get("CIFAR10_DIR", ""))
    ap.add_argument("--specs-file", default="",
                    help="one spec per line; overrides the built-in corpus list")
    ap.add_argument("--net", default="ResNet18_c100")
    ap.add_argument("--argsline-guard", default="",
                    help="path to analysis/argsline_guard.py (N7)")
    a = ap.parse_args()

    specs = list(CORPUS_SPECS)
    if a.specs_file:
        specs = [l.strip() for l in open(a.specs_file) if l.strip()]
        print("specs read from %s (%d)" % (a.specs_file, len(specs)))

    pre_src, post_src = open(a.pre).read(), open(a.post).read()

    print("=" * 78)
    print("N0  the two files differ ONLY by PATCH_NAMESETS")
    chk("PATCH_NAMESETS" not in pre_src, "the --pre file is UNPATCHED")
    chk("PATCH_NAMESETS" in post_src, "the --post file IS patched")
    # strip the two inserted regions and demand the pre file back, byte for byte
    stripped = post_src
    i = stripped.find("# --- PATCH_NAMESETS: `sets:")
    j = stripped.find("class HF():\n")
    if 0 <= i < j:
        stripped = stripped[:i] + stripped[j:]
    ins2 = ("        # --- PATCH_NAMESETS: the ONLY new branch.  Every pre-existing spec form\n"
            "        # fails this test and reaches the original body byte-unchanged. ---\n"
            "        if isinstance(stepsize_groups, str) and stepsize_groups.startswith(_NS_PREFIX):\n"
            "            return _namesets_parse(stepsize_groups, net_param_names_and_size)\n")
    chk(stripped.count(ins2) == 1, "the dispatch block appears exactly once")
    stripped = stripped.replace(ins2, "", 1)
    chk(stripped == pre_src,
        "deleting the two inserted regions reproduces --pre BYTE FOR BYTE",
        "" if stripped == pre_src else "(%d vs %d bytes)" % (len(stripped), len(pre_src)))

    pre, post = load_hf(a.pre, "pre"), load_hf(a.post, "post")

    cif = a.cifar_dir or os.getcwd()
    sys.path.insert(0, cif)
    cwd = os.getcwd()
    os.chdir(cif)
    from build_network import build_network
    net = build_network(a.net, "cpu")
    os.chdir(cwd)
    nps = [(n, p.data.size()) for n, p in net.named_parameters()]
    names = [n for n, _ in nps]
    numel = [int(p.numel()) for _, p in net.named_parameters()]
    T = len(nps)
    print("\nlive %s: %d parameter tensors, %d parameters" % (a.net, T, sum(numel)))

    print("\nN6  no parameter name is spelled like an index or a range")
    amb = [n for n in names if re.match(r"^\d+$", n) or re.match(r"^(\d+)-(\d+)$", n)]
    chk(not amb, "no ambiguous parameter name on this model", str(amb[:5]))

    print("\nN1  BACKWARD COMPATIBILITY over %d corpus specs" % len(specs))
    same = diff = 0
    for s in specs:
        rp, rq = polish(pre, s, nps), polish(post, s, nps)
        if rp == rq:
            same += 1
        else:
            diff += 1
            print("  FAIL %-20s pre=%r post=%r" % (s, rp, rq))
    chk(diff == 0, "%d/%d specs give byte-identical results before and after"
        % (same, len(specs)))
    # and the test must not be vacuously passing: at least some specs must actually
    # have produced a partition rather than an exception on both sides.
    built = sum(1 for s in specs if polish(pre, s, nps)[0] == "ok")
    chk(built >= 20, "at least 20 of the corpus specs really composed a partition",
        "%d composed, %d raised on BOTH sides" % (built, len(specs) - built))

    print("\nN2  THE INERTNESS EQUALITY -- new grammar == old grammar, exactly")
    for new, old in ((SPEC_PREFIX_NEW, SPEC_PREFIX_LEGACY),
                     ("sets:1-31/32-62", "[31,31]"),
                     ("sets:1-2/3-62", "[2,60]"),
                     ("sets:1-16/17-32/33-47/48-62", "[16,16,15,15]")):
        gn = polish(post, new, nps)
        go = polish(pre, old, nps)
        chk(gn[0] == "ok" and gn == go,
            "%-32s == %-16s (unpatched)" % (new, old),
            "" if gn == go else "%r vs %r" % (gn, go))

    print("\nN3  END TO END THROUGH init_meta")
    try:
        import torch
        def im(mod, spec):
            o = mod.HF.__new__(mod.HF)
            o.num_layers = T
            o._device = torch.device("cpu")
            mod.HF.init_meta(o, spec, nps, 1e-6)
            return (o.stepsize_type, o.param_groups_indices, o.map_layers_to_blocks,
                    [tuple(b.shape) for b in o.beta])
        A = im(pre, SPEC_PREFIX_LEGACY)
        B = im(post, SPEC_PREFIX_NEW)
        chk(A == B, "init_meta state identical for %s (pre) and %s (post)"
            % (SPEC_PREFIX_LEGACY, SPEC_PREFIX_NEW))
        chk(A[0] == "blockwise", "both route to stepsize_type 'blockwise'", A[0])
        chk(len(A[1]) == 2 and [len(x) for x in A[1]] == [49, 13],
            "param_groups_indices sizes", str([len(x) for x in A[1]]))
    except Exception:
        chk(False, "init_meta comparison ran", traceback.format_exc().splitlines()[-1])

    print("\nN4  cpr1's ARMS ON THE LIVE MODEL")
    gp = polish(post, SPEC_PREFIX_NEW, nps)
    arms = {"kP": SPEC_PREFIX_NEW, "kS": SPEC_SWAP, "kC": SPEC_CTRL}
    coarse = {}
    for tag in ("kP", "kS", "kC"):
        r = polish(post, arms[tag], nps)
        chk(r[0] == "ok", "%s composes" % tag, "" if r[0] == "ok" else str(r))
        if r[0] != "ok":
            continue
        g = r[1]
        chk(len(g) == 2, "%s is m=2" % tag, str(len(g)))
        chk([len(x) for x in g] == [49, 13], "%s group sizes are [49,13]" % tag,
            str([len(x) for x in g]))
        chk(sorted(g[0] + g[1]) == sorted(names), "%s covers every tensor once" % tag)
        coarse[tag] = set(g[0])
        pc = [sum(numel[names.index(n)] for n in x) for x in g]
        print("       %s  coarse %2d tensors %9d params | fine %2d tensors %9d params"
              % (tag, len(g[0]), pc[0], len(g[1]), pc[1]))
    if len(coarse) == 3:
        dS = coarse["kP"] ^ coarse["kS"]
        dC = coarse["kP"] ^ coarse["kC"]
        chk(dS == {"layer4.0.conv2.weight", "layer4.0.bn2.weight"},
            "kS differs from kP in EXACTLY the conv2 <-> bn2.weight swap", str(sorted(dS)))
        chk(dC == {"layer4.0.conv2.weight", "layer4.0.shortcut.0.weight"},
            "kC differs from kP in EXACTLY the conv2 <-> shortcut.0.weight swap",
            str(sorted(dC)))
        chk(len(coarse["kS"]) == len(coarse["kP"]) == len(coarse["kC"]) == 49,
            "all three coarse groups have the SAME SIZE (49)")

    print("\nN5  THE GRAMMAR IS TOTAL AND LOUD")
    bad = [
        ("sets:", "empty spec"),
        ("sets:1-62/", "empty trailing group"),
        ("sets:1-61", "a tensor is in no group"),
        ("sets:1-49,49/50-62", "a tensor claimed twice in one group"),
        ("sets:1-49/49-62", "a tensor claimed by two groups"),
        ("sets:1-63/2-62", "index above T"),
        ("sets:0-49/50-62", "index below 1"),
        ("sets:49-1/50-62", "inverted range"),
        ("sets:1-48,no.such.tensor/49-62", "unknown parameter name"),
        ("sets:1-48,,50/49,51-62", "empty item"),
        ("sets:1-49/50-61", "a tensor is in no group (tail)"),
    ]
    for s, why in bad:
        r = polish(post, s, nps)
        chk(r == ("exc", "ValueError"), "rejects %-34s (%s)" % (s, why), str(r))

    print("\nN7  RULE 20 SAFETY of the literal spec strings")
    # `[` and `]` are bash GLOB characters and the legacy `[k,62-k]` form therefore
    # has always had to be single-quoted in the launcher; they are permitted here so
    # the legacy form is held to the same standard it already meets.  The NEW forms
    # are additionally checked to need no quoting at all.
    tok_ok = re.compile(r"^[A-Za-z0-9_.,:/\[\]-]+$")
    noglob = re.compile(r"^[A-Za-z0-9_.,:/-]+$")
    for s in (SPEC_PREFIX_LEGACY, SPEC_PREFIX_NEW, SPEC_SWAP, SPEC_CTRL, "scalar"):
        chk(bool(tok_ok.match(s)) and not s.startswith("-") and "=" not in s
            and not any(c.isspace() for c in s) and "'" not in s and '"' not in s,
            "%-66s is one safe shell token" % s)
    for s in (SPEC_PREFIX_NEW, SPEC_SWAP, SPEC_CTRL):
        chk(bool(noglob.match(s)),
            "%-66s needs no shell quoting (no glob char)" % s)
    ag = a.argsline_guard or os.environ.get("ARGSGUARD_PY") or os.path.join(
        REPO, "analysis", "argsline_guard.py")
    if os.path.exists(ag):
        sys.path.insert(0, os.path.dirname(ag))
        import argsline_guard as AG
        for s in (SPEC_PREFIX_LEGACY, SPEC_PREFIX_NEW, SPEC_SWAP, SPEC_CTRL):
            line = "--dataset CIFAR100 --stepsize-groups %s --seed 9" % s
            occ = AG.parse_flags(AG.tokenize(line))
            d = dict((f, v) for f, v, _ in occ)
            chk(d.get("stepsize-groups") == s,
                "argsline_guard round-trips %-66s" % s, repr(d.get("stepsize-groups")))
            chk(len(occ) == 3, "argsline_guard sees exactly 3 flags on that line",
                str([f for f, _, _ in occ]))
    else:
        chk(False, "analysis/argsline_guard.py found", ag)

    print("=" * 78)
    if FAILED:
        print("FAILED %d check(s):" % len(FAILED))
        for f in FAILED:
            print("   -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
