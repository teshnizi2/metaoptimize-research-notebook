"""Inertness AND decomposition of PATCH_PROBE_TENSOR on the THREE m=2 `sets:`
(blockwise) partitions `cpl1` runs -- ISO, CTL and ONE -- ON `PlainNet18_c100`.

WHY THIS FILE EXISTS AND WHY THE REGISTERED TEST IS NOT EDITED.
tests/test_probe_tensor_blockwise.py was registered with `ciso1` (CORRECTIONS 187)
and its SPECS dict is a LITERAL holding that batch's three ResNet18_c100 arms.  On
the 53-tensor PlainNet18_c100 those literals raise (`layer4.0.shortcut.1.weight`
does not exist; `1-49,51-52,...` runs past 53) -- the parser being correctly loud.
STANDING RULE 16 forbids editing a registered artefact, so it is FROZEN and this
successor re-binds TWO module attributes, `SPECS` and the `--net` default, before
calling the registered `main()` -- the mechanism of
tests/test_probe_tensor_blockwise_cdep1.py (193) and
tests/test_probe_tensor_blockwise_cvi1.py (198), reused verbatim.  Every check that
runs (B0 preconditions, B2 inertness, B3 decomposition, B4 the applied sign, B5 the
header) is the registered test's own code, byte for byte.

WHAT IS NEW.  cpl1 is the first blockwise batch on a 53-tensor model, so no existing
inertness log covers it.  `--net PlainNet18_c100` is passed explicitly; it is a
DOCUMENTED argument of the registered test, not an edit.  The --cifar-dir must hold
a build_network.py that carries PATCH_PLAINNET: the isolated tree
$METAOPT_WS/harness_cpl1/cifar10 built by bin/cPL1_stage_harness.sh (the live tree
never carries it), or a scratch copy for a dry run.

RUN (needs torch; on the cluster):
  python3 tests/test_probe_tensor_blockwise_cpl1.py \\
        --pre /path/Optimizers/HF.py.pre_probe_tensor \\
        --post /path/Optimizers/HF.py --net PlainNet18_c100 \\
        --cifar-dir /path/cifar10 --work /path/scratch
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_probe_tensor_blockwise as TB    # the registered test, imported, NOT edited

# cpl1's three m=2 arms on PlainNet18_c100 (53 tensors).  Byte-identical to the SPEC
# table in analysis/cPL1_plainnet_residual_score.py, to the G_* variables in
# bin/cPL1_plainnet_residual.sh and to tests/test_plainnet.py's SPECS; the assertion
# below proves this file agrees with the scorer.
SPECS_CPL1 = {
    "ISO": "sets:1-43,45-49,51-53/layer4.0.bn2.weight,layer4.1.bn2.weight",
    "CTL": "sets:1-40,42-46,48-53/layer4.0.bn1.weight,layer4.1.bn1.weight",
    "ONE": "sets:1-43,45-53/layer4.0.bn2.weight",
}
NET_CPL1 = "PlainNet18_c100"

if __name__ == "__main__":
    reg = os.path.join(HERE, "test_probe_tensor_blockwise.py")
    print("REGISTERED_TEST %s" % reg)
    print("REGISTERED_TEST_SHA256 %s" % hashlib.sha256(open(reg, "rb").read()).hexdigest())
    print("REGISTERED_TEST_SPECS_REPLACED_AT_RUNTIME %s -> %s"
          % (sorted(TB.SPECS), sorted(SPECS_CPL1)))
    sc = os.path.join(os.path.dirname(HERE), "analysis", "cPL1_plainnet_residual_score.py")
    if os.path.exists(sc):
        import ast
        import re
        m = re.search(r"^SPEC = (\{.*?^\})", open(sc).read(), re.S | re.M)
        theirs = ast.literal_eval(m.group(1)) if m else {}
        bad = [k for k in SPECS_CPL1 if theirs.get(k) != SPECS_CPL1[k]]
        if bad:
            print("!!! SPEC DRIFT against %s: %s" % (sc, bad))
            raise SystemExit(2)
        print("SCORER_SPECS_AGREE %s (%s)" % (sorted(SPECS_CPL1), os.path.basename(sc)))
    else:
        print("SCORER_NOT_REACHABLE %s -- cross-check skipped" % sc)
    if "--net" not in sys.argv:
        sys.argv += ["--net", NET_CPL1]
    TB.SPECS = SPECS_CPL1
    TB.main()
