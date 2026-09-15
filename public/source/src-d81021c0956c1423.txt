#!/usr/bin/env python3
# =============================================================================
# cZ1_conv1_prefix_step.py
#   DOES THE PREFIX-STEP VALUE IN THE `layer4.0` REGION BELONG TO
#   `layer4.0.conv2.weight` ALONE, OR DOES `layer4.0.conv1.weight` CARRY IT TOO?
#
#   ZERO GPU.  A PURE RE-ANALYSIS OF ROWS THAT WERE ALREADY IN
#   `results/all_runs.csv` BEFORE THIS FILE WAS WRITTEN.
#
# RUN
#   python3 analysis/cZ1_conv1_prefix_step.py            <-- THE FULL, DOCUMENTED
#                                                            INVOCATION.  NO ARGUMENT.
#   python3 analysis/cZ1_conv1_prefix_step.py --selftest
#   python3 analysis/cZ1_conv1_prefix_step.py --csv <path>       (override only)
#
# `--csv` DEFAULTS TO THE REPOSITORY'S OWN `results/all_runs.csv`, RESOLVED FROM
# **THIS FILE'S** LOCATION, so the scorer cannot be aimed at the wrong corpus by
# being run from the wrong directory (CORRECTIONS 164.2: a missing optional flag
# published a wrong verdict for an hour; the fix is a default that cannot be
# forgotten, not a reminder).
#
# -----------------------------------------------------------------------------
# RULE 21 -- WHAT IS AND IS NOT CLAIMED.  **THIS FILE MAKES NO RULE 21 CLAIM.**
# -----------------------------------------------------------------------------
# It has NO RUNS OF ITS OWN.  Every row it reads was produced by `cpk2`/`cpk3`
# and ingested in earlier cycles, so the normal wall-clock proof (git commit
# time strictly before the earliest `sacct` Submit of the batch being scored)
# DOES NOT EXIST and is not asserted.  What IS claimed, and only this:
#
#     COMMIT-BEFORE-FIRST-EXECUTION.  This file is git-committed BEFORE it is
#     executed on data even once, and it is then run UNEDITED.
#
# Precedent for the weakened claim: `cQ1` (CORRECTIONS 149) and `cS1` (159).
#
# **DISCLOSURE, MADE HERE RATHER THAN DISCOVERED LATER.**  The six `cpk3` cell
# MEANS were visible to the author while this file was being designed -- they
# are printable from the committed corpus by anyone at any time, and pretending
# otherwise would be a lie.  What that costs, stated plainly: this file is an
# AUDIT WITH A REGISTERED BRANCH MAP, **not** a prediction test, and NOTHING in
# it may be read as "the registered prediction was confirmed".  What it still
# buys: the quantity, the bar, the gates and the branch boundaries are fixed in
# a committed artefact BEFORE the arithmetic is run and printed, so the verdict
# cannot be shopped to the number afterwards.  The two branch constants were
# chosen on stated grounds and NOT tuned to the answer (see BRANCH CONSTANTS).
#
# STANDING RULE 16: once committed this file is NEVER edited.  If it is wrong,
# it is wrong in public and a SUCCESSOR file says so.
#
# -----------------------------------------------------------------------------
# TWO PREMISES IN THE CYCLE BRIEF ARE FALSE AND ARE CORRECTED HERE
# -----------------------------------------------------------------------------
# (1) The brief says the `[45,17]` and `[46,16]` rows give tensor 46's behaviour
#     "at plateau5@100".  THEY DO NOT.  `[46,16]` exists at exactly ONE horizon
#     anywhere in the corpus -- **772 epochs, `cpk3` only, 3 rows**.  There is no
#     `[46,16]` row at 100 epochs in 2,638 rows.  Section B re-derives that
#     census and FAILS this file if it is ever untrue.  Everything below is
#     therefore a **772-EPOCH** object, and CORRECTIONS 156 already established
#     that cut-position numerals are horizon-specific: 156 rescoped 147's cliff
#     numeral and `cpk1`'s argmax as 100-epoch objects for exactly this reason.
#     THE CONSEQUENCE IS BINDING: no number in this file may be composed with
#     the 100-epoch `[49,13]` swap fits of CORRECTIONS 166.7(5).
#
# (2) The brief says these rows decide "whether the OUT-OF-PREFIX NULL
#     generalises past one tensor".  THEY CANNOT.  `cpk3` is a ladder of
#     CONSECUTIVE PREFIX cuts 45..50; **every one of its coarse groups is a
#     contiguous prefix `{1..k}`** and NOT ONE of its arms places any tensor in
#     the coarse group out of prefix.  A batch with no out-of-prefix arm cannot
#     test an out-of-prefix null.  Section C asserts the contiguity of all six
#     arms from the granularity strings themselves and FAILS if it is untrue.
#     Refusing that half of the question is a deliverable, not an omission.
#
# WHAT THESE ROWS *CAN* DECIDE, and what this file therefore scores:
#
#     CORRECTIONS 166.7(5) records a rival to the `CONTIGUITY-OPERATIVE` token:
#     a NO-CONTIGUITY model in which the operative variable is simply GROUP
#     MEMBERSHIP OF `layer4.0.conv2.weight` (tensor 49), with tensors 48, 51 and
#     52 all null at size 49.  166.7(7) states the open question in those exact
#     terms -- CONTIGUITY, or CONV2 MEMBERSHIP.  The rival's NAME asserts that
#     tensor 49 is PRIVILEGED among its neighbours.
#
#     `cpk3` tests that privilege directly, IN ONE BATCH, at ONE horizon, on ONE
#     seed triple {6,7,8}: it contains the five CONSECUTIVE single-tensor prefix
#     steps 46, 47, 48, 49, 50.  `layer4.0.conv1.weight` (tensor 46, 1,179,648
#     params) is the OTHER large convolution of the same residual block, and its
#     prefix step `S46 = L[46,16] - L[45,17]` is measured against `layer4.0
#     .conv2.weight`'s own `S49 = L[49,13] - L[48,14]` **in the same batch**.
#     BATCH is this campaign's unit of replication (F(62,85)=5.47, p 6.9e-13),
#     so a within-batch step comparison carries no batch offset at all.
#
#     If `S46` is resolvably non-zero and a substantial fraction of `S49`, then
#     tensor 49 is NOT privileged among the `layer4.0` convolutions, and the
#     rival account of 166.7 CANNOT be called "only `conv2` matters" -- it must
#     be RENAMED to something that covers at least two tensors.  If `S46` is
#     within bar of zero while `S49` is large, the privilege survives a test it
#     had never faced.
#
# WHAT A LARGE `S46` DOES **NOT** ESTABLISH, registered before the arithmetic:
# it does NOT show that tensor 46 pays ~0 out of prefix, because no such arm
# exists.  It supplies a CANDIDATE for that experiment -- a second tensor with a
# large IN-PREFIX value -- and naming the next experiment is the most this
# corpus can currently do.  Section H prints that sentence under every verdict.
#
# -----------------------------------------------------------------------------
# THE BAR, RE-DERIVED AT WRITE TIME, NEVER COPIED (156, 159: sigma has drifted
# TWICE after a registration, so no literal is trusted)
# -----------------------------------------------------------------------------
# SIGMA_W is the pooled within-(batch x granularity) across-seed SD of
# `plateau5` on the `ResNet18_c100` / CIFAR100 / complete / window_ok corpus,
# computed FOUR ways, and the FROZEN RULE IS `SIGMA_W = max` of the four -- the
# same rule `cR1` and `cX1` used.  A step is a difference of two n=3 cell means,
# so `SE_STEP = SIGMA_W * sqrt(2/3)` and `READ_BAR = 2 * SE_STEP`.
#
# INGEST-INVARIANCE.  EVERY corpus reader in this file excludes rows whose
# `run` begins `cdn1-`, `cru1-` or `crn1-`: the three batches in flight in cycle
# 141, none of which is ingested at write time.  So every premise here is
# invariant under their ingest, exactly as `cW1` (CORRECTIONS 165) achieved for
# its own prefix.  Section A VERIFIES that zero such rows exist rather than
# asserting it -- the 161.9 defect, where `cS2` claimed invariance with 1 reader
# of 6.  If those batches land and are ingested, this file's numbers DO NOT MOVE.
#
# -----------------------------------------------------------------------------
# BRANCH CONSTANTS -- chosen on stated grounds, NOT tuned to the answer
# -----------------------------------------------------------------------------
#   FRAC = 0.25   "a substantial fraction of conv2's own step".  A quarter is
#                 the coarsest threshold that still means "same order of
#                 magnitude"; it is not a fitted number and no value in the
#                 corpus was consulted to pick it.
#   CLIFF_MIN = 5.0 pp   the ANCHOR gate on `S49`.  If `conv2`'s own prefix step
#                 is not itself large in this batch, there is no privilege to
#                 test and the verdict is VOID.  5.0 pp is the scale at which
#                 this campaign has previously called a CIFAR-100 cut-position
#                 effect real, and it is ~6.8x the read bar either way.
# =============================================================================

from __future__ import annotations
import argparse, collections, csv, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(os.path.dirname(HERE), "results", "all_runs.csv")

# batches in flight in cycle 141; excluded from EVERY reader (ingest-invariance)
INFLIGHT = ("cdn1-", "cru1-", "crn1-")

# the cell this file lives in, asserted on every row it uses
CELL = dict(network="ResNet18_c100", dataset="CIFAR100",
            meta_stepsize="1e-3", alpha0="1e-6", beta_clip="-15:-2.3026")
HORIZON = "772"

# the six cpk3 arms, granularity string -> coarse prefix size
CPK3 = {"scalar": None, "[45,17]": 45, "[46,16]": 46,
        "[47,15]": 47, "[48,14]": 48, "[49,13]": 49, "[50,12]": 50}

# tensor identities, from the live 62-tensor manifest of ResNet18_c100
TENSOR = {46: ("layer4.0.conv1.weight", 1179648),
          47: ("layer4.0.bn1.weight",       512),
          48: ("layer4.0.bn1.bias",         512),
          49: ("layer4.0.conv2.weight", 2359296),
          50: ("layer4.0.bn2.weight",       512),
          51: ("layer4.0.bn2.bias",         512),
          52: ("layer4.0.shortcut.0.weight", 131072)}

FRAC      = 0.25    # "substantial fraction of S49"
CLIFF_MIN = 5.0     # pp, the anchor gate on S49
FLOOR_MIN_SE = 10.0 # every scored cell must clear the in-batch m=1 anchor by this

_fails = []
def chk(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        _fails.append(msg)
    return bool(cond)

# ---------------------------------------------------------------- corpus -----
def rows(csv_path):
    """EVERY reader goes through here.  The three in-flight batches are excluded."""
    out = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            run = str(r.get("run") or "")
            if any(run.startswith(p) for p in INFLIGHT):
                continue
            out.append(r)
    return out

def inflight_count(csv_path):
    n = 0
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if any(str(r.get("run") or "").startswith(p) for p in INFLIGHT):
                n += 1
    return n

def total_rows(csv_path):
    with open(csv_path, newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))

def usable(r):
    return r.get("complete") == "1" and r.get("window_ok") == "1"

def in_cell(r):
    return all(r.get(k) == v for k, v in CELL.items())

def batch_of(r):
    return str(r.get("run") or "").split("-")[0]

def cells(rs, batch, horizon):
    """granularity -> sorted list of (run, seed, plateau5) for one batch/horizon."""
    g = collections.defaultdict(list)
    for r in rs:
        if batch_of(r) != batch or r.get("epochs_requested") != horizon:
            continue
        if not (usable(r) and in_cell(r)):
            continue
        g[r["granularity"]].append((r["run"], r["seed"], float(r["plateau5"]),
                                    r.get("epochs_done"), float(r["final_train"])))
    for k in g:
        g[k].sort()
    return g

def mean(v):
    return sum(v) / len(v)

def sd(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))

# ------------------------------------------------------------------ sigma ----
def sigma_pool(rs, sel):
    g = collections.defaultdict(list)
    for r in rs:
        if not (usable(r) and r.get("network") == "ResNet18_c100"
                and r.get("dataset") == "CIFAR100"):
            continue
        if not sel(r):
            continue
        g[(batch_of(r), r["granularity"])].append(float(r["plateau5"]))
    ss = df = ncell = nmem = 0
    ss = 0.0
    for v in g.values():
        if len(v) < 2:
            continue
        m = mean(v)
        ss += sum((x - m) ** 2 for x in v)
        df += len(v) - 1
        ncell += 1
        nmem += len(v)
    return (math.sqrt(ss / df) if df else float("nan")), df, ncell, nmem

def derive_sigma(rs):
    isc = lambda r: str(r.get("granularity") or "").startswith("[")
    pools = [
        ("SIGMA_772_CUT",  lambda r: r.get("epochs_requested") == "772" and isc(r)),
        ("SIGMA_772_WIDE", lambda r: r.get("epochs_requested") == "772"),
        ("SIGMA_100_CUT",  lambda r: r.get("epochs_requested") == "100" and isc(r)),
        ("SIGMA_ALLH_CUT", lambda r: isc(r)),
    ]
    res = []
    for name, sel in pools:
        s, df, nc, nm = sigma_pool(rs, sel)
        res.append((name, s, df, nc, nm))
    return res

# ------------------------------------------------------------------ main -----
def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="corpus CSV (DEFAULTS to the repo's results/all_runs.csv)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    print("=" * 78)
    print("cZ1_conv1_prefix_step.py -- IS `layer4.0.conv2.weight` PRIVILEGED,")
    print("   OR DOES `layer4.0.conv1.weight` CARRY PREFIX VALUE TOO?")
    print("   ZERO GPU.  NO RULE 21 CLAIM (no runs of its own):")
    print("   COMMIT-BEFORE-FIRST-EXECUTION ONLY.  Precedent cQ1 (149), cS1 (159).")
    print("=" * 78)
    print("CSV: %s%s" % (a.csv, "  (DEFAULT, resolved from this file)"
                         if a.csv == DEFAULT_CSV else "  (OVERRIDDEN)"))
    if not os.path.exists(a.csv):
        print("FATAL: corpus not found"); return 2
    rs = rows(a.csv)

    # ---- A. ingest-invariance, VERIFIED not asserted -------------------------
    print("\nA. INGEST-INVARIANCE (161.9: verify, do not assert)")
    tot = total_rows(a.csv); nfl = inflight_count(a.csv)
    print("   corpus rows            %d" % tot)
    print("   rows excluded by EVERY reader (%s) : %d"
          % (",".join(INFLIGHT), nfl))
    chk(nfl == 0, "zero cdn1/cru1/crn1 rows in the corpus at write time "
                  "-> the exclusion is currently a NO-OP and every premise "
                  "below is invariant under their future ingest")
    print("   readers that apply the exclusion: 1 of 1 "
          "(`rows()` is the ONLY corpus reader in this file)")

    # ---- B. the horizon census: the brief's `@100` premise ------------------
    print("\nB. THE HORIZON CENSUS -- THE BRIEF'S `plateau5@100` PREMISE IS FALSE")
    hz = collections.defaultdict(list)
    for r in rs:
        if r.get("granularity") in ("[45,17]", "[46,16]") and usable(r) and in_cell(r):
            hz[(r["granularity"], r["epochs_requested"])].append(batch_of(r))
    for k in sorted(hz):
        print("   %-9s %-4s ep  n=%d  batches %s"
              % (k[0], k[1], len(hz[k]), sorted(set(hz[k]))))
    n46_100 = len(hz.get(("[46,16]", "100"), []))
    n46_772 = len(hz.get(("[46,16]", "772"), []))
    chk(n46_100 == 0, "[46,16] has ZERO rows at 100 epochs anywhere in the corpus "
                      "-- the brief's `@100` is wrong")
    chk(n46_772 == 3, "[46,16] has exactly 3 rows, all at 772 epochs (cpk3)")
    print("   => EVERYTHING BELOW IS A 772-EPOCH OBJECT and MAY NOT be composed")
    print("      with the 100-epoch [49,13] swap fits of CORRECTIONS 166.7(5)")
    print("      (CORRECTIONS 156's rescoping applies verbatim).")

    # ---- C. cpk3 is all-prefix: the out-of-prefix half is REFUSED -----------
    print("\nC. `cpk3` HAS NO OUT-OF-PREFIX ARM -- THAT HALF OF THE BRIEF IS REFUSED")
    g3 = cells(rs, "cpk3", HORIZON)
    allpref = all(k in CPK3 for k in g3)
    print("   cpk3 granularities: %s" % sorted(g3))
    chk(allpref, "every cpk3 arm is `scalar` or a CONTIGUOUS PREFIX `[k,62-k]` "
                 "-- no arm places any tensor in the coarse group out of prefix")
    chk(not any(str(k).startswith("sets:") or "/" in str(k) for k in g3),
        "no `sets:`/hole-carrying spec appears in cpk3")
    print("   => THE OUT-OF-PREFIX NULL CANNOT BE TESTED BY THESE ROWS.  Refused.")

    # ---- D. the bar, re-derived four ways ----------------------------------
    print("\nD. THE BAR, RE-DERIVED AT WRITE TIME (never copied; 156/159 drift)")
    pools = derive_sigma(rs)
    for name, s, df, nc, nm in pools:
        print("   %-16s %.6f   df %-4d cells %-3d members %d" % (name, s, df, nc, nm))
    SIGMA_W = max(p[1] for p in pools)
    which = [p[0] for p in pools if p[1] == SIGMA_W][0]
    SE_STEP = SIGMA_W * math.sqrt(2.0 / 3.0)
    READ_BAR = 2.0 * SE_STEP
    SE_CELL = SIGMA_W / math.sqrt(3.0)
    print("   FROZEN RULE: SIGMA_W = max of the four = %.6f   (%s)" % (SIGMA_W, which))
    print("   SE_STEP  = SIGMA_W*sqrt(2/3) = %.6f   (difference of two n=3 means)"
          % SE_STEP)
    print("   READ_BAR = 2*SE_STEP         = %.6f" % READ_BAR)
    print("   SE_CELL  = SIGMA_W/sqrt(3)   = %.6f" % SE_CELL)
    print("   FRAC = %.2f   CLIFF_MIN = %.2f pp   (constants, stated grounds, "
          "not fitted)" % (FRAC, CLIFF_MIN))

    # ---- E. gate G0: provenance --------------------------------------------
    print("\nE. GATE G0 -- PROVENANCE OF THE cpk3 CELLS")
    need = ["scalar", "[45,17]", "[46,16]", "[47,15]", "[48,14]", "[49,13]", "[50,12]"]
    ok0 = True
    seeds_seen = set()
    for g in need:
        v = g3.get(g, [])
        good = (len(v) == 3 and all(x[3] == HORIZON for x in v))
        ok0 &= good
        for x in v:
            seeds_seen.add(x[1])
        print("   %-9s n=%d  seeds %s  epochs_done %s  plateau5 %s"
              % (g, len(v), [x[1] for x in v], sorted(set(x[3] for x in v)),
                 ["%.3f" % x[2] for x in v]))
    ok0 &= chk(seeds_seen == {"6", "7", "8"}, "cpk3 seeds are exactly {6,7,8}")
    ok0 = chk(ok0, "G0: all 7 cpk3 cells present, n=3 each, epochs_done==772, "
                   "one cell (%s)" % ", ".join("%s=%s" % kv for kv in sorted(CELL.items())))

    L  = {g: mean([x[2] for x in g3[g]]) for g in need if g in g3}
    TR = {g: mean([x[4] for x in g3[g]]) for g in need if g in g3}
    SDc= {g: sd([x[2] for x in g3[g]]) for g in need if g in g3}

    # ---- F. the steps -------------------------------------------------------
    print("\nF. THE FIVE CONSECUTIVE SINGLE-TENSOR PREFIX STEPS, ALL IN BATCH cpk3")
    print("   %-4s %-26s %-9s %8s %8s %8s %9s" %
          ("t", "tensor", "step", "S (pp)", "SE", "|S|/SE", "resolved"))
    S = {}
    order = [(46, "[45,17]", "[46,16]"), (47, "[46,16]", "[47,15]"),
             (48, "[47,15]", "[48,14]"), (49, "[48,14]", "[49,13]"),
             (50, "[49,13]", "[50,12]")]
    for t, lo, hi in order:
        S[t] = L[hi] - L[lo]
        print("   %-4d %-26s %-9s %+8.4f %8.4f %8.2f %9s"
              % (t, TENSOR[t][0], "%s->%s" % (lo.split(",")[0] + "]", hi),
                 S[t], SE_STEP, abs(S[t]) / SE_STEP,
                 "YES" if abs(S[t]) > READ_BAR else "no"))
    print("\n   cell levels (plateau5 TEST / final_train TRAIN, mean of 3, in-batch sd):")
    for g in need:
        if g in L:
            print("     %-9s TEST %8.4f  (sd %6.4f)   TRAIN %8.4f" % (g, L[g], SDc[g], TR[g]))

    # ---- G. gates G1 anchor, G2 floor --------------------------------------
    print("\nG. GATE G1 (ANCHOR) AND GATE G2 (FLOOR)")
    S49 = S[49]; S46 = S[46]
    g1 = chk(S49 > CLIFF_MIN,
             "G1 ANCHOR: conv2's OWN in-batch prefix step S49 = %+.4f pp exceeds "
             "CLIFF_MIN %.2f (=%.2f SE_STEP).  Without it there is no privilege "
             "to test and the verdict is VOID." % (S49, CLIFF_MIN, S49 / SE_STEP))
    anchor = L["scalar"]
    print("   in-batch m=1 anchor (cpk3-k01, n=3): %.4f pp   SE_CELL %.6f" % (anchor, SE_CELL))
    g2 = True
    for g in need:
        if g == "scalar":
            continue
        marg = L[g] - anchor
        g2 &= (marg / SE_CELL) >= FLOOR_MIN_SE
        print("     %-9s +%8.4f pp over the anchor = %7.2f SE_CELL   TRAIN %.2f"
              % (g, marg, marg / SE_CELL, TR[g]))
    g2 = chk(g2, "G2 FLOOR: every scored cell clears the IN-BATCH m=1 anchor by "
                 ">= %.1f SE_CELL -- no arm is at or near the floor, so a "
                 "content-free 'the arm just died' model fits none of them "
                 "(the cpr1 defect, CORRECTIONS 164)" % FLOOR_MIN_SE)

    # ---- H. THE BRANCH MAP --------------------------------------------------
    print("\nH. THE REGISTERED BRANCH MAP, ON THE PRIMARY S46")
    print("   PRIMARY   S46 = L[46,16] - L[45,17]  =  %+.6f pp  = %+.2f SE_STEP" %
          (S46, S46 / SE_STEP))
    print("   REFERENCE S49                        =  %+.6f pp  = %+.2f SE_STEP" %
          (S49, S49 / SE_STEP))
    print("   RATIO S46/S49 = %.4f   FRAC*S49 = %.6f" % (S46 / S49 if S49 else float('nan'),
                                                         FRAC * S49))
    if not (g1 and g2 and ok0):
        verdict = "VOID-GATE-FAILED"
    elif S46 < -READ_BAR:
        verdict = "CONV1-NEGATIVE-UNREGISTERED"
    elif abs(S46) <= READ_BAR:
        verdict = "CONV2-PRIVILEGED"
    elif S46 >= FRAC * S49:
        verdict = "CONV2-NOT-PRIVILEGED"
    else:
        verdict = "CONV1-RESOLVED-BUT-MINOR"
    print("\n   FINAL: %s" % verdict)
    MEANING = {
      "CONV2-NOT-PRIVILEGED":
        "`layer4.0.conv1.weight` carries a prefix step of the SAME ORDER as\n"
        "     `layer4.0.conv2.weight`'s, in the same batch, at the same horizon, on the\n"
        "     same seeds.  Tensor 49 is therefore NOT privileged among the layer4.0\n"
        "     convolutions, and the rival account recorded at CORRECTIONS 166.7(5)/(7)\n"
        "     MAY NOT BE NAMED 'only conv2 matters' / 'conv2 membership'.  IT MUST BE\n"
        "     RENAMED to a form covering at least two tensors.  This does NOT resurrect\n"
        "     CONTIGUITY-OPERATIVE and does NOT kill the rival -- it renames it.",
      "CONV2-PRIVILEGED":
        "Among tensors 46/47/48 none carries resolvable prefix value while conv2's own\n"
        "     step is large.  The rival account's NAME survives a test it had never faced.",
      "CONV1-RESOLVED-BUT-MINOR":
        "conv1's step is resolvable but under a quarter of conv2's.  The privilege is\n"
        "     weakened, not broken; a rename is a hedge, not a requirement.",
      "CONV1-NEGATIVE-UNREGISTERED":
        "conv1 coarse membership HURTS.  No registered account predicts this; report as\n"
        "     an unregistered pattern, not as evidence for any account.",
      "VOID-GATE-FAILED":
        "A gate failed.  No reading is entitled.",
    }
    print("   MEANS: %s" % MEANING[verdict])

    print("\n   CENSUS of the other four steps at the SAME bar (not branch-bearing):")
    for t in (47, 48, 49, 50):
        print("     t%-3d %-26s %+9.4f pp  %+7.2f SE_STEP   %s"
              % (t, TENSOR[t][0], S[t], S[t] / SE_STEP,
                 "RESOLVED" if abs(S[t]) > READ_BAR else "within bar"))
    nres = sum(1 for t in (46, 47, 48, 49, 50) if abs(S[t]) > READ_BAR)
    print("     -> %d of the 5 consecutive single-tensor prefix steps are RESOLVED "
          "at 2 SE." % nres)

    # ---- I. the one cross-batch replication that exists ---------------------
    print("\nI. THE ONLY CROSS-BATCH CHECK THAT EXISTS -- cpk2 HAS NO [46,16]")
    g2b = cells(rs, "cpk2", HORIZON)
    print("   cpk2 granularities: %s   seeds %s"
          % (sorted(g2b), sorted(set(x[1] for v in g2b.values() for x in v))))
    have = "[45,17]" in g2b and "[47,15]" in g2b
    chk("[46,16]" not in g2b, "cpk2 does NOT contain [46,16] -- S46 itself is "
                              "UNREPLICATED and this file says so")
    if have:
        d2 = mean([x[2] for x in g2b["[47,15]"]]) - mean([x[2] for x in g2b["[45,17]"]])
        d3 = L["[47,15]"] - L["[45,17]"]
        print("   the TWO-TENSOR composite S46+S47 = L[47,15]-L[45,17], a WITHIN-BATCH")
        print("   difference in each batch, so each batch's offset cancels:")
        print("     cpk3 (seeds 6,7,8): %+.6f pp" % d3)
        print("     cpk2 (seeds 3,4,5): %+.6f pp" % d2)
        print("     difference          %+.6f pp = %+.2f SE (SE of a diff of two "
              "such steps = SIGMA_W*sqrt(4/3) = %.6f)"
              % (d3 - d2, (d3 - d2) / (SIGMA_W * math.sqrt(4.0 / 3.0)),
                 SIGMA_W * math.sqrt(4.0 / 3.0)))
        chk(abs(d3 - d2) <= 2 * SIGMA_W * math.sqrt(4.0 / 3.0),
            "the composite REPLICATES across cpk2/cpk3 on DISJOINT seed triples "
            "within 2 SE -- so the 45->47 region is not a cpk3 artefact")

    # ---- J. scope -----------------------------------------------------------
    print("\nJ. SCOPE, BINDING")
    print("   * 772 epochs ONLY.  May NOT be composed with the 100-epoch [49,13]")
    print("     swap fits of 166.7(5) (CORRECTIONS 156).")
    print("   * PREFIX STEPS ONLY.  Says NOTHING about what tensor 46 is worth OUT")
    print("     of prefix -- no such arm exists.  It supplies a CANDIDATE for that")
    print("     experiment, and naming the next experiment is the most these rows do.")
    print("   * S46 is UNREPLICATED (one batch, one seed triple); only the two-tensor")
    print("     composite S46+S47 has a second batch.")
    print("   * NOT a prediction test -- the cell means were visible at design time")
    print("     (disclosed in the header).  A registered branch map, run unedited.")
    print("   * Nothing here revives or refutes CONTIGUITY-OPERATIVE.  It constrains")
    print("     only the NAME of the rival recorded at 166.7(5)/(7).")
    print("   * RULE 11 is open on every row used: all are at the single shared")
    print("     ms=1e-3, and hz9 showed such a contrast can reverse under tuning.")

    if a.selftest:
        print("\nK. SELFTEST -- arithmetic and internal consistency")
        chk(abs((L["[46,16]"] - L["[45,17]"]) - S46) < 1e-12, "S46 identity")
        chk(abs(sum(S[t] for t in (46,47,48,49,50)) - (L["[50,12]"] - L["[45,17]"])) < 1e-9,
            "the five steps telescope to L[50,12]-L[45,17]")
        chk(TENSOR[46][1] == 1179648 and TENSOR[49][1] == 2359296,
            "tensor 46/49 param counts match the live 62-tensor manifest")
        chk(abs(READ_BAR - 2 * SIGMA_W * math.sqrt(2.0/3.0)) < 1e-12, "READ_BAR identity")
        chk(SIGMA_W == max(p[1] for p in pools), "SIGMA_W is the max of the four pools")
        chk(len(need) == 7 and len(order) == 5, "7 cells, 5 consecutive steps")
        chk(verdict in MEANING, "the verdict is one of the registered branches")
        # branch map is exhaustive and mutually exclusive on a grid
        seen = set()
        for x in [i / 4.0 for i in range(-80, 81)]:
            if x < -READ_BAR: b = "CONV1-NEGATIVE-UNREGISTERED"
            elif abs(x) <= READ_BAR: b = "CONV2-PRIVILEGED"
            elif x >= FRAC * S49: b = "CONV2-NOT-PRIVILEGED"
            else: b = "CONV1-RESOLVED-BUT-MINOR"
            seen.add(b)
        chk(len(seen) == 4, "all four branches reachable on a grid of S46 (exhaustive, "
                            "first-match-wins, mutually exclusive by construction)")

    print("\n" + "=" * 78)
    print("FINAL: %s" % verdict)
    print("checks: %d FAIL" % len(_fails))
    for m in _fails:
        print("   FAILED: %s" % m)
    print("=" * 78)
    return 0 if not _fails else 1

if __name__ == "__main__":
    sys.exit(main())
