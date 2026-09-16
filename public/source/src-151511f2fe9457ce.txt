"""Inertness AND decomposition of PATCH_PROBE_TENSOR on the THREE m=2 `sets:`
(blockwise) partitions `cpl2` runs -- ISO, HEAD and TWIN -- ON `PlainNet18_c100`.

WHY THIS FILE EXISTS AND WHY THE REGISTERED TEST IS NOT EDITED.
tests/test_probe_tensor_blockwise.py was registered with `ciso1` (CORRECTIONS 187);
its SPECS dict is a LITERAL.  STANDING RULE 16 forbids editing a registered
artefact, and tests/test_probe_tensor_blockwise_cpl1.py (215) is registered too
(its SPECS are cpl1's).  So this successor re-binds ONE module attribute of the
registered test, `SPECS`, and passes the DOCUMENTED `--net PlainNet18_c100`
argument before calling the registered `main()` -- the mechanism of the cdep1 /
cvi1 / cpl1 successors, reused.  Every check that runs (B0 preconditions, B2
inertness, B3 decomposition, B4 the applied sign, B5 the header) is the registered
test's own code, byte for byte.

WHAT IS NEW.  cpl2 runs the first [52,1] SINGLETON partitions that isolate
layer4.1.bn2.weight (HEAD, PlainNet idx 50) and layer4.1.bn1.weight (TWIN, idx 47);
cpl1's log covers ISO but not these two.  INDICES ARE PlainNet's (53 tensors), NOT
ResNet's.  The --cifar-dir must hold a build_network.py carrying PATCH_PLAINNET:
the isolated tree $METAOPT_WS/harness_cpl1/cifar10 built by
bin/cPL1_stage_harness.sh (the live tree never carries it).

RUN (needs torch; on the cluster):
  python3 tests/test_probe_tensor_blockwise_cpl2.py \\
        --pre  /path/harness_cpl1/cifar10/Optimizers/HF.py.pre_probe_tensor \\
        --post /path/harness_cpl1/cifar10/Optimizers/HF.py \\
        --cifar-dir /path/harness_cpl1/cifar10 --work /path/scratch
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_probe_tensor_blockwise as TB    # the registered test, imported, NOT edited

# cpl2's three m=2 arms on PlainNet18_c100 (53 tensors).  Byte-identical to the SPEC
# table in analysis/cPL2_plainnet_head_score.py and to the G_* variables in
# bin/cPL2_plainnet_head.sh; the assertion below proves this file agrees with the scorer.
SPECS_CPL2 = {
    "ISO": "sets:1-43,45-49,51-53/layer4.0.bn2.weight,layer4.1.bn2.weight",
    "HEAD": "sets:1-49,51-53/layer4.1.bn2.weight",
    "TWIN": "sets:1-46,48-53/layer4.1.bn1.weight",
}
NET_CPL2 = "PlainNet18_c100"

if __name__ == "__main__":
    reg = os.path.join(HERE, "test_probe_tensor_blockwise.py")
    print("REGISTERED_TEST %s" % reg)
    print("REGISTERED_TEST_SHA256 %s" % hashlib.sha256(open(reg, "rb").read()).hexdigest())
    print("REGISTERED_TEST_SPECS_REPLACED_AT_RUNTIME %s -> %s" % (sorted(TB.SPECS), sorted(SPECS_CPL2)))
    sc = os.path.join(os.path.dirname(HERE), "analysis", "cPL2_plainnet_head_score.py")
    if os.path.exists(sc):
        import ast
        import re
        m = re.search(r"^SPEC = (\{.*?^\})", open(sc).read(), re.S | re.M)
        theirs = ast.literal_eval(m.group(1)) if m else {}
        bad = [k for k in SPECS_CPL2 if theirs.get(k) != SPECS_CPL2[k]]
        if bad:
            print("!!! SPEC DRIFT against %s: %s" % (sc, bad))
            raise SystemExit(2)
        print("SCORER_SPECS_AGREE %s (%s)" % (sorted(SPECS_CPL2), os.path.basename(sc)))
    else:
        print("SCORER_NOT_REACHABLE %s -- cross-check skipped" % sc)
    if "--net" not in sys.argv:
        sys.argv += ["--net", NET_CPL2]
    TB.SPECS = SPECS_CPL2
    TB.main()
