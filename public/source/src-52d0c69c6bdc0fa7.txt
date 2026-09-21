"""BITE OF PATCH_DECAYMASK's NAME-LIST GRAMMAR ON cwd1's UNCHANGED TREE, FOR cwd4's SIX EXACT STRINGS.  CORRECTIONS 280.

`cwd4` adds NO harness code: it runs from cwd1's tree ($WS/harness_cwd1/cifar10, HF.py 94aedc33..., proved inert and
biting by `cwdinert1` 5026046, 80 PASS / 0 FAIL, CORRECTIONS 260, and used unchanged by `cwd3`, 275 / 278) with
PATCH_DECAYMASK's REGISTERED name-list form `DECAY_MASK=<name>(+<name>)*`.  `cwd3` proved the k=3, k=2 and k=20 forms
on ResNet18_c100's real GPU path; `cwd4` needs, IN ADDITION:

  * the k=1 form on THIS network (cwd2 ran a one-name list on PlainNet's tree, not this one), on THREE different names;
  * A NEW k=2 STRING, {50,53}, which `cwd3` never ran;
  * and -- THE POINT OF THIS DRIVER -- SET DISCRIMINATION BETWEEN ARMS WHOSE `dm_masked` k IS EQUAL.  `cwd4` has TWO
    arms at k=2 (TWOWD0 {50,53} and CTL2WD0 {47,56}) and THREE at k=1 (ONE50 / ONE53 / ONE59).  G-BITE's record audit
    CANNOT tell them apart -- k is the same.  The scorer declares this (G-SETSEP, 280.4) and leans on G-WITNESS plus
    THIS proof: RR4 below shows, on the real GPU path at a test-only alpha0, that each string masks EXACTLY its own
    tensors and FAILS the set it must be told apart from.

It IMPORTS, UNEDITED:
  tests/test_decaymask_realrun.py   (cwd1 / cwd2's driver): run_verify and its --child-verify child (the verifier around
                                    optimizer.base_update that checks w', h', m' against the registered formula per tensor)
  tests/test_grouphold_realrun.py   (cvt7's registered real-run test): run_variant, same, chk
  analysis/cwd4_design.py           strings, witnesses, indices, DISCRIMINATE

  RR0  DETERMINISM CONTROL: the parent tree (cvt8's) run twice, scalar -> identical beta, loss, probe bytes, final weights.
  RR1  OFF on THIS tree: DECAY_MASK unset and empty, scalar == the parent tree BITWISE; `DECAY_MASK: off` printed.
  RR2  THE EXACT MASKED STRINGS (run_variant): exactly the design's witness line; at every probe record dm_n == step+2,
       dm_skipped == dm_n * k, dm_masked == k, dm_wdterm > 0, k finite norms (k = 3 / 2 / 2 / 1 / 1 / 1).
  RR3  UPDATE IDENTITY (run_verify, the registered child): at EVERY step, for EVERY tensor, w' h' m' BITWISE the formula
       with wd 0 on the design's indices and 0.1 elsewhere -- for all six masked strings; the parent tree with an EMPTY
       mask as the control.
  RR4  SET DISCRIMINATION (NON-VACUITY, test-only alpha0 1e-2 so the WD term is visible at float resolution): for each
       ordered pair in cwd4_design.DISCRIMINATE, the run matches its OWN index set and FAILS the other's --
       TWOWD0 vs CTL2WD0 and CTL2WD0 vs TWOWD0 (THE k=2 COLLISION), CARWD0 vs TWOWD0 (a proper SUPERSET, one tensor
       apart), and the three k=1 arms cyclically (ONE50 vs ONE53, ONE53 vs ONE59, ONE59 vs ONE50).  Every masked run's
       final weights differ from the OFF run's.
  RR5  LOUDNESS on THIS tree (CPU, the real HF.init_meta): a misspelt name, a duplicate, an empty `+` token, a name
       outside the model and a bad character each raise ValueError at construction -- a typo in a name list can never
       run silently unmasked.  A ONE-NAME list is checked by the same code path (`raw.split("+")` over one token).

  python3 tests/test_decaymask_realrun_cwd4.py --tree-pre $WS/harness_cvt8/cifar10 --tree-post $WS/harness_cwd1/cifar10 \\
      --steps 300 --every 10 --seed 143 --work /tmp/cwd4_realrun
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

REUSED_DRIVER_SHA = "0c64e50ffaac6a2a483f68f81e1b9634199ef7a2a82545964351cad3cfb396bb"   # tests/test_decaymask_realrun.py


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def child_loud(tree):
    """RR5 in a clean process: construct HF.init_meta on the live model with bad DECAY_MASK values; print one line each."""
    os.chdir(tree)
    sys.path.insert(0, tree)
    import io, contextlib
    import torch
    from build_network import build_network
    from Optimizers.HF import HF
    net = build_network("ResNet18_c100", "cpu")
    nps = [(n, p.data.size()) for n, p in net.named_parameters()]
    cases = [("misspelt-one", "layer4.0.bn2.weigth"),
             ("misspelt-pair", "layer4.0.bn2.weight+layer4.0.shortcut.1.weigth"),
             ("duplicate", "layer4.0.bn2.weight+layer4.0.bn2.weight"),
             ("empty-token", "layer4.0.bn2.weight++layer4.0.shortcut.1.weight"),
             ("not-a-name", "layer5.0.bn2.weight"),
             ("bad-char", "layer4.0.bn2.weight,layer4.0.shortcut.1.weight")]
    for lab, v in cases:
        for k in ("VOTE_W", "BETA_HOLD", "COMP_HOLD", "WINDOW_HOLD", "GROUP_HOLD", "REST_HOLD", "DECAY_MASK"):
            os.environ.pop(k, None)
        os.environ["DECAY_MASK"] = v
        o = HF.__new__(HF)
        o.num_layers = len(nps)
        o._device = torch.device("cpu")
        o.args_meta = {"alg": "Lion", "meta_stepsize": 1e-3, "momentum_param": 0.99, "Lion_beta2": 0.9, "weight_decay": 0}
        o.args_base = {"alg": "SGDm", "weight_decay": 0.1, "momentum_param": 0.99}
        o._beta_lo, o._beta_hi = -15.0, -2.3026
        o._hier = "none"
        o.base_update = o.SGDm_base_update
        raised = "none"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                HF.init_meta(o, "scalar", nps, 1e-6)
        except ValueError as ex:
            raised = "ValueError: %s" % str(ex)[:90]
        except Exception as ex:
            raised = "OTHER %s" % type(ex).__name__
        print("LOUD %s %s" % (lab, raised))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child-loud", action="store_true")
    ap.add_argument("--tree", default="")
    ap.add_argument("--tree-pre", default="")
    ap.add_argument("--tree-post", default="")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=143)
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    if a.child_loud:
        child_loud(a.tree)
        return

    D = _load("cwd4_design", os.path.join(REPO, "analysis", "cwd4_design.py"))
    B = D.CWD4
    TD = _load("test_decaymask_realrun", os.path.join(HERE, "test_decaymask_realrun.py"))
    RR = _load("test_grouphold_realrun", os.path.join(HERE, "test_grouphold_realrun.py"))
    chk, same = RR.chk, RR.same
    a.net = B.NET
    pre, post = os.path.abspath(a.tree_pre), os.path.abspath(a.tree_post)
    os.makedirs(a.work, exist_ok=True)
    for k in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "COMP_HOLD", "REST_HOLD", "WINDOW_HOLD", "DECAY_MASK"):
        os.environ.pop(k, None)

    def sh(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    print("BATCH cwd4")
    print("DRIVER_SHA256 %s" % sh(os.path.abspath(__file__)))
    print("REUSED_DRIVER_SHA256 %s" % sh(TD.__file__))
    print("REGISTERED_TEST_SHA256 %s" % sh(RR.__file__))
    print("DESIGN_SHA256 %s" % sh(D.__file__))
    print("CWD_DESIGN_SHA256 %s" % sh(D.W.__file__))
    print("PATCH_SHA256 %s" % sh(os.path.join(REPO, "patches", "patch_decaymask.py")))
    for t, lab in ((pre, "pre"), (post, "post")):
        print("TREE_%s %s HF %s build_network %s" % (lab.upper(), t, sh(os.path.join(t, "Optimizers", "HF.py")), sh(os.path.join(t, "build_network.py"))))
    chk(sh(os.path.join(post, "Optimizers", "HF.py")) == B.HF_POST_SHA, "--tree-post's HF.py is cwd1's PATCH_DECAYMASK tree, UNCHANGED (94aedc33...)")
    hpre = open(os.path.join(pre, "Optimizers", "HF.py"), "rb").read()
    chk(hashlib.sha256(hpre).hexdigest() == B.HF_PARENT_SHA and b"PATCH_DECAYMASK" not in hpre,
        "--tree-pre's HF.py is cvt8's and carries no PATCH_DECAYMASK")
    wpre = os.path.join(post, "Optimizers", "HF.py.pre_decaymask")
    chk(os.path.exists(wpre) and open(wpre, "rb").read() == hpre, "--tree-pre's HF.py is BYTE-IDENTICAL to --tree-post's HF.py.pre_decaymask")
    chk(sh(os.path.join(pre, "build_network.py")) == sh(os.path.join(post, "build_network.py")) == B.BN_SHA,
        "the two trees' build_network.py are BYTE-IDENTICAL and == the design's BN_SHA")
    chk(sh(TD.__file__) == REUSED_DRIVER_SHA and sh(D.W.__file__) == D.CWD_DESIGN_SHA
        and sh(os.path.join(REPO, "patches", "patch_decaymask.py")) == D.PATCH_DECAYMASK_SHA,
        "the reused driver, cwd_design.py and patch_decaymask.py are cwd1's registered files, UNEDITED (RULE 16)")
    print("\nsteps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026, net %s, every arm scalar"
          % (a.steps, a.every, a.seed, B.NET))
    print("k per masked arm: %s" % ", ".join("%s=%d" % (x, B.K_MASKED[x]) for x in B.MASKED))

    def run_arm(tag, tree, dm):
        old = os.environ.pop("DECAY_MASK", None)
        if dm is not None:
            os.environ["DECAY_MASK"] = dm
        try:
            return RR.run_variant(a, tag, tree, "scalar", None)
        finally:
            os.environ.pop("DECAY_MASK", None)
            if old is not None:
                os.environ["DECAY_MASK"] = old

    def lines(R, prefix):
        return [ln for ln in (R["stdout"] if R else []) if ln.startswith(prefix)]

    def idx_of(arm):
        return ",".join("%d" % (i + 1) for i in D.dm_indices(B.DMASK[arm], B.TENSORS))

    print("\nRR0 DETERMINISM CONTROL (the parent tree, twice)")
    A1 = run_arm("pre_k01_a", pre, None)
    A2 = run_arm("pre_k01_b", pre, None)
    same(A1, A2, "RR0 parent k01 vs itself")

    print("\nRR1 OFF (DECAY_MASK unset / empty on cwd1's tree == the parent tree, bitwise)")
    for val in (None, ""):
        vl = "unset" if val is None else "empty"
        R = run_arm("post_k01_%s" % vl, post, val)
        same(A1, R, "RR1 k01 DECAY_MASK %s" % vl)
        chk(lines(R, "DECAY_MASK") == ["DECAY_MASK: off"] and all(lines(R, k.split(":")[0]) == [k] for k in B.OFF_LINES),
            "RR1 k01 DECAY_MASK %s printed `DECAY_MASK: off` and the parent's own witness lines" % vl)
        if R and R["probe_bytes"]:
            chk(b'"dm_' not in R["probe_bytes"], "RR1 k01 DECAY_MASK %s: no dm_* key in probe.jsonl" % vl)

    print("\nRR2 THE EXACT MASKED STRINGS ON THE REAL PATH (k = 3 / 2 / 2 / 1 / 1 / 1)")
    for arm in B.MASKED:
        R = run_arm("post_%s" % arm.lower(), post, B.DMASK[arm])
        if R is None:
            continue
        k = B.K_MASKED[arm]
        chk(lines(R, "DECAY_MASK") == [B.WITNESS_DM[arm]] and all(lines(R, kk.split(":")[0]) == [kk] for kk in B.OFF_LINES),
            "RR2 %-8s printed exactly the design's DECAY_MASK witness (k=%d, idx %s)" % (arm, k, idx_of(arm)),
            (lines(R, "DECAY_MASK") or ["(none)"])[0][:130])
        recs = [json.loads(x) for x in R["probe_bytes"].decode().splitlines()]
        bad = 0
        for r in recs:
            nrm = r.get("dm_norm")
            bad += not (r.get("dm_n") == r["step"] + 2 and r.get("dm_skipped") == r.get("dm_n", -1) * k and r.get("dm_masked") == k
                        and isinstance(r.get("dm_wdterm"), float) and r["dm_wdterm"] > 0 and isinstance(nrm, list) and len(nrm) == k
                        and all(isinstance(x, float) and x == x and abs(x) < 1e30 for x in nrm)
                        and isinstance(r.get("dm_small"), int) and isinstance(r.get("dm_absmin"), float))
        chk(len(recs) == a.steps // a.every and bad == 0,
            "RR2 %-8s at every probe record dm_n == step+2, dm_skipped == dm_n*%d, dm_masked == %d, dm_wdterm > 0, %d finite norm(s)" % (arm, k, k, k),
            "%d records, %d bad; last %s" % (len(recs), bad, {kk: recs[-1].get(kk) for kk in ("dm_n", "dm_skipped", "dm_wdterm", "dm_absmin")} if recs else ""))

    print("\nRR3 UPDATE IDENTITY ON THE REAL PATH (the registered verifier around base_update)")
    V0 = TD.run_verify(a, "verify_pre_k01", pre, "scalar", {}, "1e-6", "", "")
    chk(V0 is not None and V0["steps"] == a.steps and V0["stats"]["expect"] == [0, 0, 0],
        "RR3 CONTROL the parent tree (OFF) matches the registered formula with an EMPTY mask at every step, every tensor",
        str(V0["stats"] if V0 else None))
    for arm in B.MASKED:
        idx = idx_of(arm)
        V = TD.run_verify(a, "verify_%s" % arm.lower(), post, "scalar", {"DECAY_MASK": B.DMASK[arm]}, "1e-6", idx, "")
        chk(V is not None and V["steps"] == a.steps and V["stats"]["expect"] == [0, 0, 0]
            and [ln for ln in V["stdout"] if ln.startswith("DECAY_MASK")] == [B.WITNESS_DM[arm]],
            "RR3 %-8s w', h', m' BITWISE the formula with wd 0 on idx {%s} and 0.1 elsewhere, at every one of %d steps"
            % (arm, idx, a.steps), "%s; bites at float resolution %s" % (V["stats"] if V else None, V["bites"] if V else None))

    print("\nRR4 SET DISCRIMINATION (test-only alpha0 1e-2: the WD term visible at float resolution).")
    print("    THIS IS WHAT SEPARATES THE ARMS WHOSE k COLLIDES -- G-BITE's record audit cannot (280.4).")
    VO = TD.run_verify(a, "verify_big_off", post, "scalar", {}, "1e-2", "", "")
    chk(VO is not None and VO["stats"]["expect"] == [0, 0, 0], "RR4 the test-only alpha0 1e-2 OFF run matches the EMPTY-mask formula at every step")
    for arm, other in D.DISCRIMINATE:
        V = TD.run_verify(a, "verify_big_%s_vs_%s" % (arm.lower(), other.lower()), post, "scalar",
                          {"DECAY_MASK": B.DMASK[arm]}, "1e-2", idx_of(arm), idx_of(other))
        chk(V is not None and V["stats"]["expect"] == [0, 0, 0] and V["bites"] > 0 and sum(V["stats"]["wrong"][:2]) > 0
            and VO is not None and V["weights_sha256"] != VO["weights_sha256"],
            "RR4 %-8s (k=%d) matches ITS OWN set {%s}, the WD term bites, the SAME run FAILS %s's set {%s}%s, and its"
            " weights differ from OFF"
            % (arm, B.K_MASKED[arm], idx_of(arm), other, idx_of(other),
               "  [SAME k -- THE COLLISION]" if B.K_MASKED[arm] == B.K_MASKED[other] else ""),
            "stats %s bites %s" % (V["stats"] if V else None, V["bites"] if V else None))

    print("\nRR5 LOUDNESS ON cwd1's TREE (CPU, the real HF.init_meta), INCLUDING THE ONE-NAME FORM")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child-loud", "--tree", post],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
    loud = [ln for ln in p.stdout.splitlines() if ln.startswith("LOUD ")]
    for ln in loud:
        print("  " + ln)
    chk(p.returncode == 0 and len(loud) == 6 and all(" ValueError: PATCH_DECAYMASK" in ln for ln in loud),
        "RR5 a misspelt ONE-name list, a misspelt pair, a duplicate, an empty + token, a non-name and a bad character"
        " EACH raise ValueError at construction")

    fails = list(RR.FAILED)
    print("\n%s" % ("ALL PASS" if not fails else "FAILURES: %r" % fails))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
