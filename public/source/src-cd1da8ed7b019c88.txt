"""Inertness AND decomposition of PATCH_PROBE_TENSOR on the TWO m=2 `sets:`
(blockwise) partitions `cvi1` runs -- ISO and CTL -- ON `VGG11_bn_c100`.

WHY THIS FILE EXISTS AND WHY THE REGISTERED TEST IS NOT EDITED.
tests/test_probe_tensor_blockwise.py was registered with `ciso1` (CORRECTIONS
187) and its SPECS dict is a LITERAL holding that batch's three ResNet arms.
Its `--net` flag alone is NOT enough to move it to another architecture: on a
26-tensor model those literals raise
    ValueError: PATCH_NAMESETS: range '1-49' is not inside 1..26
-- which is the parser being correctly loud, and is exactly the failure this
file was written in response to.  STANDING RULE 16 forbids editing a registered
artefact once its data exist: it is FROZEN and a successor is registered instead
(precedent cN1/cN2 at 149, cdn1/cdn2 at 175, and
tests/test_probe_tensor_blockwise_cdep1.py at 193, whose mechanism this file
reuses verbatim).

This file does NOT copy the registered test's logic and does NOT modify the file
on disk.  It IMPORTS the module and re-binds TWO module attributes, `SPECS` and
the `--net` default, before calling the registered `main()`.  Every check that
runs (B0 preconditions, B2 inertness, B3 decomposition, B4 the applied sign, B5
the header) is the registered test's own code, byte for byte, executed against
`cvi1`'s two spec strings on `VGG11_bn_c100`.  The registered file's sha256 is
printed below so the CORRECTIONS entry can quote it and anyone can verify it is
unchanged.

WHAT IS NEW HERE BEYOND cdep1's SUCCESSOR.  cvi1 is the FIRST blockwise (`sets:`)
batch this campaign has ever run on a NON-ResNet architecture, so no existing
inertness log covers the blockwise path on a 26-tensor model.  `--net` is passed
explicitly on the command line below; it is a DOCUMENTED argument of the
registered test, not an edit.

RUN (needs torch; on the cluster against the real tree):
  python3 tests/test_probe_tensor_blockwise_cvi1.py \\
        --pre /path/Optimizers/HF.py.pre_probe_tensor \\
        --post /path/Optimizers/HF.py --net VGG11_bn_c100 \\
        --cifar-dir /path/cifar10 --work /path/scratch
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_probe_tensor_blockwise as TB    # the registered test, imported, NOT edited

# cvi1's two m=2 arms on VGG11_bn_c100 (26 tensors, 9,274,532 parameters).
# These strings are byte-identical to the SPEC table in
# analysis/cVI1_vggiso_identity_score.py and to the G_* variables in
# bin/cVI1_vgg_isolate.sh; the launcher's guard 4c' proves the scorer and the
# launcher agree, and the assertion below proves this file agrees with the
# scorer too, so all three cannot drift.
SPECS_CVI1 = {
    "ISO": "sets:1-22,24-26/bn8.weight",
    "CTL": "sets:1-19,21-26/bn7.weight",
}
NET_CVI1 = "VGG11_bn_c100"

if __name__ == "__main__":
    reg = os.path.join(HERE, "test_probe_tensor_blockwise.py")
    print("REGISTERED_TEST %s" % reg)
    print("REGISTERED_TEST_SHA256 %s" % hashlib.sha256(open(reg, "rb").read()).hexdigest())
    print("REGISTERED_TEST_SPECS_REPLACED_AT_RUNTIME %s -> %s"
          % (sorted(TB.SPECS), sorted(SPECS_CVI1)))
    # cross-check against the registered SCORER so this file cannot drift from it
    sc = os.path.join(os.path.dirname(HERE), "analysis", "cVI1_vggiso_identity_score.py")
    if os.path.exists(sc):
        import ast
        import re
        m = re.search(r"^SPEC = (\{.*?^\})", open(sc).read(), re.S | re.M)
        theirs = ast.literal_eval(m.group(1)) if m else {}
        bad = [k for k in SPECS_CVI1 if theirs.get(k) != SPECS_CVI1[k]]
        if bad:
            print("!!! SPEC DRIFT against %s: %s" % (sc, bad))
            raise SystemExit(2)
        print("SCORER_SPECS_AGREE %s (%s)" % (sorted(SPECS_CVI1), os.path.basename(sc)))
    else:
        print("SCORER_NOT_REACHABLE %s -- cross-check skipped" % sc)
    if "--net" not in sys.argv:
        sys.argv += ["--net", NET_CVI1]
    TB.SPECS = SPECS_CVI1
    TB.main()
