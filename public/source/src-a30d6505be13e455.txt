"""Inertness AND decomposition of PATCH_PROBE_TENSOR on the THREE m=2 `sets:`
(blockwise) partitions `cgn2` runs -- ISO, CTL and ONE -- ON `ResNet18_gn_c100`.

WHY THIS FILE EXISTS AND WHY THE REGISTERED TEST IS NOT EDITED.
tests/test_probe_tensor_blockwise.py was registered with `ciso1` (CORRECTIONS 187);
its SPECS dict is a LITERAL and its --net defaults to ResNet18_c100.  STANDING RULE
16 forbids editing a registered artefact, so this successor re-binds ONE module
attribute, `SPECS`, and passes the DOCUMENTED `--net ResNet18_gn_c100` argument
before calling the registered `main()` -- the mechanism of
tests/test_probe_tensor_blockwise_cdep1.py (193) and _cpl1.py (215), reused.  Every
check that runs (B0 preconditions, B2 inertness, B3 decomposition, B4 the applied
sign, B5 the header) is the registered test's own code, byte for byte.

WHAT IS NEW.  No blockwise inertness log exists on a GroupNorm network: cgn1 ran
only scalar and layerwise.  The three spec strings are byte-identical to the
BatchNorm batches' (ciso1 ISO / ONE, cdep1 DEPTH) because ResNet18_gn_c100 has the
same 62-tensor list and names; this test proves the partitions and the probe are
inert and decompose on the GN MODEL.  The --cifar-dir must hold a build_network.py
carrying PATCH_RESNET_GN_C100: the isolated tree $METAOPT_WS/harness_cgn1/cifar10
built by bin/cGN1_stage_harness.sh (the live tree never carries it).  --pre is the
LIVE tree's Optimizers/HF.py.pre_probe_tensor (the cgn1 tree's HF.py is
byte-identical to the live one, stage check V-a).

RUN (needs torch; on the cluster):
  python3 tests/test_probe_tensor_blockwise_cgn2.py \\
        --pre  /path/live/Optimizers/HF.py.pre_probe_tensor \\
        --post /path/harness_cgn1/cifar10/Optimizers/HF.py \\
        --cifar-dir /path/harness_cgn1/cifar10 --work /path/scratch
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_probe_tensor_blockwise as TB    # the registered test, imported, NOT edited

# cgn2's three m=2 arms on ResNet18_gn_c100 (62 tensors).  Byte-identical to the SPEC
# table in analysis/cGN2_gn_isolation_score.py and to the G_* variables in
# bin/cGN2_gn_isolation.sh; the assertion below proves this file agrees with the scorer.
SPECS_CGN2 = {
    "ISO": "sets:1-49,51-52,54-58,60-62/layer4.0.bn2.weight,layer4.0.shortcut.1.weight,layer4.1.bn2.weight",
    "CTL": "sets:1-46,49-55,57-62/layer4.0.bn1.weight,layer4.0.bn1.bias,layer4.1.bn1.weight",
    "ONE": "sets:1-49,51-62/layer4.0.bn2.weight",
}
NET_CGN2 = "ResNet18_gn_c100"

if __name__ == "__main__":
    reg = os.path.join(HERE, "test_probe_tensor_blockwise.py")
    print("REGISTERED_TEST %s" % reg)
    print("REGISTERED_TEST_SHA256 %s" % hashlib.sha256(open(reg, "rb").read()).hexdigest())
    print("REGISTERED_TEST_SPECS_REPLACED_AT_RUNTIME %s -> %s" % (sorted(TB.SPECS), sorted(SPECS_CGN2)))
    sc = os.path.join(os.path.dirname(HERE), "analysis", "cGN2_gn_isolation_score.py")
    if os.path.exists(sc):
        import ast
        import re
        m = re.search(r"^SPEC = (\{.*?^\})", open(sc).read(), re.S | re.M)
        theirs = ast.literal_eval(m.group(1)) if m else {}
        bad = [k for k in SPECS_CGN2 if theirs.get(k) != SPECS_CGN2[k]]
        if bad:
            print("!!! SPEC DRIFT against %s: %s" % (sc, bad))
            raise SystemExit(2)
        print("SCORER_SPECS_AGREE %s (%s)" % (sorted(SPECS_CGN2), os.path.basename(sc)))
    else:
        print("SCORER_NOT_REACHABLE %s -- cross-check skipped" % sc)
    if "--net" not in sys.argv:
        sys.argv += ["--net", NET_CGN2]
    TB.SPECS = SPECS_CGN2
    TB.main()
