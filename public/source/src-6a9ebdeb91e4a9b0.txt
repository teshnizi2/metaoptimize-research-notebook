#!/usr/bin/env python3
"""
c98b_reproduce.py -- SUCCESSOR to analysis/c98_reproduce.py (CORRECTIONS 224).

WHY IT EXISTS.  c98_reproduce.py is registered and may not be edited.  Since the
corpus grew past the 2,177 rows the paper was written against it has exited 1 on
every run, on ten corpus-count assertions that move with each ingest.  A permanently
non-zero audit hides a real regression in the noise.  This successor runs c98's OWN
code -- it imports c98_reproduce and executes its sections, its chk(), its main(),
unmodified -- and changes exactly one thing: the outcome of a fixed, named set of
DRIFT assertions no longer decides the exit status.

    python3 analysis/c98b_reproduce.py            # same flags as c98_reproduce.py
    python3 analysis/c98b_reproduce.py --ledger   # print the classified assertion ledger

WHAT IS DRIFT, BY RULE (not "whatever fails today").  An assertion is DRIFT iff its
derived value is an aggregate over the WHOLE corpus that no batch prefix bounds, so
that appending rows for an unrelated batch can move it.  Applied to c98's 636
assertion sites, the rule selects all 18 sites of three sections and nothing else:
  [1]  corpus          -- 8 whole-corpus counts/sums (rows, admissible, wallclock,
                          GPU-hours, nodes, and the three attrition counts)
  [8]  competitiveness -- 4 corpus-wide maxima (best MetaOptimize arm, best tuned
                          SGD baseline and its se, and their difference)
  [15] metacensus      -- 6 whole-corpus family counts
Every other site (618) is a SCIENCE check: a within-batch contrast, a fixed cell
list, a registered constant, or the manuscript census.  Those run through c98's
chk() byte-for-byte and still fail the audit exactly as they do in c98.

The 18 drift sites are NOT silenced: each prints expected (the paper's numeral) vs
live, and two DRIFT GUARDS still fail the audit, because they would signal a real
regression rather than growth:
  G1  an append-only count fell BELOW the paper's value (the corpus lost rows or a
      row lost admissibility -- ingests are append-only, "removed 0");
  G2  the competitiveness deficit is no longer positive (the paper's scope limit,
      abstract (ii), would have reversed, not drifted).

THE PAPER'S NUMERALS REMAIN STALE.  This script does not make 2177, 1735, 93.317,
1.807, 431, 241 ... correct; it stops them from masking science regressions.  Whether
and when paper/ is refreshed is the author's decision.

Exit status: 0 iff every SCIENCE check passes, no DRIFT GUARD fires, and every
declared drift site that should have run was matched exactly once.  2 if the drift
declaration no longer matches c98's assertion sites (the classification is stale).
"""
import contextlib, io, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c98_reproduce as R                                   # noqa: E402

# (section, exact chk name) -> kind.  "count" sites are append-only monotone and are
# covered by guard G1; "level" sites are not monotone and carry no G1 guard.
DRIFT = {
    ("corpus", "rows in results/all_runs.csv"): "count",
    ("corpus", "admissible rows"): "count",
    ("corpus", "runs carrying a wallclock"): "count",
    ("corpus", "GPU-hours"): "count",
    ("corpus", "distinct nodes"): "count",
    ("corpus", "rows failing window_ok"): "count",
    ("corpus", "rows failing complete"): "count",
    ("corpus", "rows with no plateau5"): "count",
    ("competitiveness", "best ResNet-18/C10 MetaOptimize arm"): "level",
    ("competitiveness", "tuned SGD+cosine baseline, ResNet-18/C10"): "level",
    ("competitiveness", "   se"): "level",
    ("competitiveness", "deficit"): "deficit",
    ("metacensus", "admissible runs in the partition families"): "count",
    ("metacensus", "   ...with meta = Lion"): "count",
    ("metacensus", "   ...with meta = RMSProp (all twelve are sm4)"): "count",
    ("metacensus", "   distinct meta-optimisers on that family"): "count",
    ("metacensus", "count-matched-family rows outside the in-flight rp1 batch"): "count",
    ("metacensus", "   ...of which admissible"): "count",
}
DRIFT_SECTIONS = sorted({s for s, _ in DRIFT})

LEDGER = []        # (section, name, where, fmt, paper, got, ok, klass) for every asserted site
DRIFT_SEEN = {}    # key -> times matched
GUARD_FAILS = []   # (guard, name, got, paper, why)
_STATE = {"section": None}
_ORIG_CHK = R.chk


def classify(section, name):
    return "DRIFT" if (section, name) in DRIFT else "SCIENCE"


def chk(name, got, paper, where, fmt="%+.3f"):
    """c98's chk(), called unmodified; only a DRIFT site's FAIL is moved out of FAILS."""
    section = _STATE["section"]
    klass = classify(section, name)
    n_fail = len(R.FAILS)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _ORIG_CHK(name, got, paper, where, fmt)
    line = buf.getvalue()
    if paper is None:                       # c98 asserts nothing here; neither do we
        sys.stdout.write(line)
        return
    ok = len(R.FAILS) == n_fail
    LEDGER.append((section, name, where, fmt, paper, got, ok, klass))
    if klass == "SCIENCE":
        sys.stdout.write(line)
        return
    key = (section, name)
    DRIFT_SEEN[key] = DRIFT_SEEN.get(key, 0) + 1
    if not ok:
        R.FAILS.pop()                       # _ORIG_CHK appended exactly one entry
    sys.stdout.write(line.replace("**FAIL**", "DRIFT   ").rstrip("\n") + "   [DRIFT]\n")
    kind = DRIFT[key]
    if kind == "count" and got < paper - 1e-9:
        GUARD_FAILS.append(("G1", name, got, paper,
                            "append-only count fell below the paper's value"))
    if kind == "deficit" and not got > 0:
        GUARD_FAILS.append(("G2", name, got, paper,
                            "the competitiveness deficit is no longer positive"))


def _wrap(section, fn):
    def run(*a, **k):
        _STATE["section"] = section
        try:
            return fn(*a, **k)
        finally:
            _STATE["section"] = None
    run.__name__ = fn.__name__
    return run


def install():
    """Route c98's sections through the classifier.  Idempotent."""
    if getattr(R, "_c98b_installed", False):
        return
    R.chk = chk
    R.SECTIONS = [(n, _wrap(n, f)) for n, f in R.SECTIONS]
    setattr(R, "_c98b_installed", True)


def _sections_requested(argv):
    names = [n for n, _ in R.SECTIONS]
    want = [n for n in names if "--" + n in argv]
    return want or names


def drift_report(argv):
    ran = set(_sections_requested(argv))
    if "--census" in argv:                 # census mode runs every section silently
        ran = {n for n, _ in R.SECTIONS}
    bad_decl = []
    for key in DRIFT:
        seen = DRIFT_SEEN.get(key, 0)
        if seen > 1 or (key[0] in ran and seen != 1):
            bad_decl.append((key, seen))
    drift_rows = [x for x in LEDGER if x[7] == "DRIFT"]
    print("\n" + "=" * 78)
    print("DRIFT REPORT (c98b) -- whole-corpus numerals that move with each ingest.")
    print("NOT counted as failures.  expected = the paper's numeral; live = this corpus.")
    print("THE PAPER'S NUMERALS BELOW REMAIN STALE; refreshing paper/ is the author's call.")
    print("=" * 78)
    print("  %-16s %-58s %12s %12s %10s  %s" % ("section", "assertion", "expected",
                                                "live", "delta", "state"))
    for section, name, where, fmt, paper, got, ok, _ in drift_rows:
        print("  %-16s %-58s %12s %12s %+10.4g  %s"
              % (section, name.strip(), fmt % paper, fmt % got, got - paper,
                 "agrees" if ok else "DIFFERS"))
    nd = sum(1 for x in drift_rows if not x[6])
    print("  %d drift site(s) ran, %d differ from the paper's numeral." % (len(drift_rows), nd))
    if GUARD_FAILS:
        print("\n%d DRIFT GUARD(S) FIRED -- these are regressions, not drift:" % len(GUARD_FAILS))
        for g, name, got, paper, why in GUARD_FAILS:
            print("   %s  %-50s live %s vs paper %s  (%s)" % (g, name.strip(), got, paper, why))
    if bad_decl:
        print("\nDRIFT DECLARATION IS STALE (c98's assertion sites changed):")
        for key, seen in bad_decl:
            print("   %-60s matched %d time(s)" % (repr(key), seen))
    return drift_rows, bad_decl


def run(argv):
    """Run the audit; returns the exit status.  argv excludes the program name."""
    install()
    ledger_only = "--ledger" in argv
    argv = [a for a in argv if a != "--ledger"]
    old = sys.argv
    sys.argv = ["c98b_reproduce.py"] + argv
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_c98 = R.main()
    finally:
        sys.argv = old
    out = buf.getvalue()
    n_sci = sum(1 for x in LEDGER if x[7] == "SCIENCE")
    n_dr = sum(1 for x in LEDGER if x[7] == "DRIFT")
    # c98's footer counts every assertion site; say what c98b actually gates on.
    out = out.replace("ALL %d CHECKS PASS." % len(R.ASSERTED),
                      "ALL %d SCIENCE CHECKS PASS  (%d assertion sites = %d science + %d "
                      "drift; drift reported below, not gated)."
                      % (n_sci, len(R.ASSERTED), n_sci, n_dr))
    if not ledger_only:
        sys.stdout.write(out)
    else:
        for section, name, where, fmt, paper, got, ok, klass in LEDGER:
            print("%-7s %-16s %-60s paper %-10s live %-12s %s"
                  % (klass, section, name.strip(), fmt % paper, fmt % got,
                     "PASS" if ok else "FAIL"))
    _, bad_decl = drift_report(argv)
    sci_fail = sum(1 for x in LEDGER if x[7] == "SCIENCE" and not x[6])
    print("\n" + "=" * 78)
    print("c98b VERDICT: science %d/%d PASS | drift %d site(s), %d differ (not gated) | "
          "guards fired %d | declaration %s"
          % (n_sci - sci_fail, n_sci, n_dr,
             sum(1 for x in LEDGER if x[7] == "DRIFT" and not x[6]),
             len(GUARD_FAILS), "STALE" if bad_decl else "OK"))
    print("=" * 78)
    if bad_decl:
        return 2
    if rc_c98 or GUARD_FAILS:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
