#!/usr/bin/env python3
"""corpus_exclusions.py -- read and verify results/CORPUS-EXCLUSIONS.tsv (CORRECTIONS 230).

results/all_runs.csv has no column for a harness intervention the ARGS line cannot carry (cvt1's
VOTE_W).  Such rows carry a plain arm's cell key.  The exclusion list names them by (run, job_id);
every corpus reader that pools by cell drops them first.  Stdlib only; the CSV schema is unchanged.

As a library (copy the three lines into a registration script, or import this module):

    from corpus_exclusions import load, filter_rows
    rows = filter_rows(csv.DictReader(open("results/all_runs.csv")))

From the command line:

    python3 analysis/corpus_exclusions.py --check [--runs <dir> ...]

  --check  every listed (run, job_id) is present EXACTLY once in results/all_runs.csv; no key is listed
           twice; with --runs, the run's own raw .out carries exactly the listed witness line (of the
           intervention kind the witness names: VOTE_W, BETA_HOLD, GROUP_HOLD or COMP_HOLD, see KINDS), and
           every OTHER .out of a listed batch carries that kind's `off` line (so the list is complete for that
           batch);
           COMPLETENESS (CORRECTIONS 239): every .out found under --runs that is a CSV row and prints an ON
           line of any kind is listed, with a witness of that kind (ON runs not yet in the CSV are counted,
           not required; logs not under --runs are not read, and without --runs no log is read at all);
           TWO-KIND RUNS (CORRECTIONS 245): a run whose registered design turns on two kinds (cvt6's forced
           arms: BETA_HOLD + COMP_HOLD) is listed ONCE, by either ON line, and must print exactly the lines
           MULTI_KIND registers for its (batch, arm) -- so its other ON line is verified, not skipped;
           then prints the noise-floor demonstration: the registered cvt1 sigmas (227.6) re-derived on
           the current corpus three ways -- as 227 did (every `cvt1-` row dropped), as a future
           registration should (only the listed rows dropped), and naively (nothing dropped).
Exit 0 all checks pass, 1 any check fails.
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_TSV = os.path.join(REPO, "results", "CORPUS-EXCLUSIONS.tsv")
DEFAULT_CSV = os.path.join(REPO, "results", "all_runs.csv")


def load(path=DEFAULT_TSV):
    """-> list of dicts, one per listed row (comment lines skipped)."""
    lines = [ln.rstrip("\n") for ln in open(path) if ln.strip() and not ln.startswith("#")]
    head = lines[0].split("\t")
    out = []
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) != len(head):
            raise ValueError("bad arity in %s: %r" % (path, ln))
        out.append(dict(zip(head, f)))
    return out


def keys(path=DEFAULT_TSV):
    return set((e["run"], e["job_id"]) for e in load(path))


def is_excluded(row, _cache={}):
    if "k" not in _cache:
        _cache["k"] = keys()
    return (row.get("run"), row.get("job_id")) in _cache["k"]


def filter_rows(rows, path=DEFAULT_TSV):
    k = keys(path)
    return [r for r in rows if (r.get("run"), r.get("job_id")) not in k]


# ---- the 227.6 sigma definitions, re-typed (cVT1_voteweight_score.py is NOT imported) -------------
CELLKEYS = ["network", "dataset", "granularity", "base", "meta", "meta_stepsize", "alpha0", "gamma",
            "augment", "beta_clip", "batch_size", "epochs_requested", "hier", "lam", "eta_ratio"]


def _pooled(rows, net):
    cells = {}
    for r in rows:
        if not (r.get("superseded") == "0" and r.get("collapsed") == "0" and r.get("complete") == "1"
                and (r.get("plateau5") or "").strip()):
            continue
        if not (r.get("epochs_requested") == "100" and r.get("augment") == "1" and r.get("beta_clip") == "-15:-2.3026"
                and r.get("meta_stepsize") == "1e-3" and r.get("alpha0") == "1e-6" and r.get("batch_size") == "100"
                and r.get("network") == net and r.get("dataset") == "CIFAR100"):
            continue
        cells.setdefault(tuple(r.get(c, "") for c in CELLKEYS), []).append(float(r["plateau5"]))
    ss, df, nc = 0.0, 0, 0
    for v in cells.values():
        if len(v) < 2:
            continue
        m = sum(v) / len(v)
        ss += sum((x - m) ** 2 for x in v)
        df += len(v) - 1
        nc += 1
    return (math.sqrt(ss / df) if df else float("nan")), df, nc


# ---- --check only: the registered intervention KINDS (CORRECTIONS 239) -------------------------------------
# One entry per harness intervention the ARGS line cannot carry: the prefix of the witness line its patched tree
# prints on EVERY run, and the exact line it prints when off.  A listed row's kind is the entry its `witness`
# starts with (`<prefix>:`).  A future kind is one line here.  load / keys / is_excluded / filter_rows ignore it.
KINDS = [
    ("VOTE_W", "VOTE_W: off"),        # patches/patch_voteweight.py, CORRECTIONS 227 (cvt1, cvt2, cvt3, cvt5)
    ("BETA_HOLD", "BETA_HOLD: off"),  # patches/patch_betahold.py, CORRECTIONS 237 (cvt4, cvt6)
    ("GROUP_HOLD", "GROUP_HOLD: off"),  # patches/patch_grouphold.py, CORRECTIONS 243 (cvt7)
    ("COMP_HOLD", "COMP_HOLD: off"),  # patches/patch_comphold.py, CORRECTIONS 242 (cvt6)
]
# No prefix above is a prefix of another (their first letters V / B / G / C differ), so no line starts with two of them
# and every `startswith` reader selects each line for ONE kind (CORRECTIONS 245); check() FAILs if an entry breaks it.

# ---- --check only: runs whose registered design turns ON more than one kind (CORRECTIONS 245) ----------------------
# A TSV row carries ONE witness.  Such a run is listed ONCE, by any one of its ON lines; each (batch, arm) below must
# print EXACTLY these lines, one per kind -- so the ON line the row does not carry is verified too, and a wrong or
# missing one FAILs.  Re-typed from the registered scorer (as 227.6's sigmas are; the scorer is NOT imported):
# analysis/cVT6_complementpath_score.py WITNESS_BH / WITNESS_CH for FORCED (242.4).  Any other listed run printing an
# ON line of a kind its witness does not name FAILs, as before.  A batch here holds its unlisted runs to that kind's
# `off` line too.
_CVT6_BH_TRI = ("BETA_HOLD: on type=blockwise group=1 groupsize=1 name=layer4.1.bn2.weight mode=tri P=9428 "
                "b0=-13.815510749816895 ms=0.001 lo=-15.0 hi=-2.3026 peak=-4.387510749816894")
_CVT6_BH_FLOOR = "BETA_HOLD: on type=blockwise group=1 groupsize=1 name=layer4.1.bn2.weight mode=floor value=-15.0"
_CVT6_CH_REC = ("COMP_HOLD: on type=blockwise group=0 groupsize=52 mode=rec id=cvt6_headpath "
                "sha256=74be71fa524ad0122d1408e01dd2b633b004b2e27228393fe6e593f494358a5d knots=500 n0=2 n1=49902 "
                "b0=-13.815510749816895 lo=-15.0 hi=-2.3026 vmax=-4.852388381958008 vlast=-15.0")
_CVT6_CH_TRI = ("COMP_HOLD: on type=blockwise group=0 groupsize=52 mode=tri P=9428 b0=-13.815510749816895 ms=0.001 "
                "lo=-15.0 hi=-2.3026 peak=-4.387510749816894")
MULTI_KIND = {
    ("cvt6", "HIGHHEADPATH"): {"BETA_HOLD": _CVT6_BH_TRI, "COMP_HOLD": _CVT6_CH_REC},
    ("cvt6", "LOWMUTEPATH"): {"BETA_HOLD": _CVT6_BH_FLOOR, "COMP_HOLD": _CVT6_CH_TRI},
    ("cvt6", "LOWHEADPATH"): {"BETA_HOLD": _CVT6_BH_FLOOR, "COMP_HOLD": _CVT6_CH_REC},
}


def multi_kind_of(fn):
    """-> the registered {kind: line} of a `<batch>-<arm>-...` run name or .out file name, or None."""
    return MULTI_KIND.get(tuple(fn.split("-")[:2]))


def kind_of(witness):
    for k, _off in KINDS:
        if witness.startswith(k + ":"):
            return k
    return None


def witness_lines(path):
    """-> {prefix: [every line starting with it, in order]} for one raw .out."""
    got = dict((k, []) for k, _off in KINDS)
    for ln in open(path, errors="replace"):
        for k, _off in KINDS:
            if ln.startswith(k):
                got[k].append(ln.rstrip("\n"))
    return got


def check(runs_dirs):
    bad = []
    ents = load()
    rows = list(csv.DictReader(open(DEFAULT_CSV)))
    print("exclusion list: %d rows, batches %s" % (len(ents), sorted(set(e["batch"] for e in ents))))
    ks = [(e["run"], e["job_id"]) for e in ents]
    if len(set(ks)) != len(ks):
        bad.append("duplicate key in the list")
    # CORRECTIONS 245: a prefix of another prefix would let one line be read as two kinds
    collide = [(ka, kb) for ka, _o in KINDS for kb, _p in KINDS if ka != kb and ka.startswith(kb)]
    for ka, kb in collide:
        bad.append("KINDS prefix %r starts with %r: a %s line would also be read as %s" % (ka, kb, ka, kb))
    cnt = {}
    for r in rows:
        cnt[(r["run"], r["job_id"])] = cnt.get((r["run"], r["job_id"]), 0) + 1
    for k in ks:
        if cnt.get(k, 0) != 1:
            bad.append("%s/%s occurs %d times in the CSV" % (k[0], k[1], cnt.get(k, 0)))
    print("  every listed key present exactly once in the CSV (%d rows): %s" % (len(rows), not bad))
    if runs_dirs:
        outs = {}
        for d in runs_dirs:
            for root, _ds, fs in os.walk(d):
                for fn in fs:
                    if fn.endswith(".out"):
                        outs.setdefault(fn, os.path.join(root, fn))
        batches = set(e["batch"] for e in ents)
        listed = dict(("%s-%s.out" % (e["run"], e["job_id"]), e) for e in ents)
        bkinds = {}
        for e in ents:
            if kind_of(e["witness"]) is None:
                bad.append("%s/%s witness %r names no registered intervention kind (KINDS)"
                           % (e["run"], e["job_id"], e["witness"][:60]))
            else:
                bkinds.setdefault(e["batch"], set()).add(kind_of(e["witness"]))
        for (mb, _ma), des in MULTI_KIND.items():  # CORRECTIONS 245: a listed two-kind batch uses both kinds
            if mb in bkinds:
                bkinds[mb].update(des)
        wl = {}
        n_w = n_off = 0
        for fn, p in sorted(outs.items()):
            if fn.split("-")[0] not in batches:
                continue
            wl[fn] = witness_lines(p)
            if fn in listed:
                kd = kind_of(listed[fn]["witness"])
                if kd is not None and wl[fn][kd] != [listed[fn]["witness"]]:
                    bad.append("%s witness %r != listed" % (fn, wl[fn][kd]))
                n_w += 1
            else:
                for kd, off in KINDS:
                    if kd in bkinds.get(fn.split("-")[0], ()) and wl[fn][kd] != [off]:
                        bad.append("%s is NOT listed but its witness is %r" % (fn, wl[fn][kd]))
                n_off += 1
        if n_w != len(ents):
            bad.append("found %d of %d listed .out files" % (n_w, len(ents)))
        print("  raw .out witness: %d listed runs carry their listed line; %d unlisted runs of the same batch print %s"
              % (n_w, n_off, " / ".join("`%s`" % off for kd, off in KINDS if any(kd in s for s in bkinds.values()))))
        # COMPLETENESS (CORRECTIONS 239): every .out found that is a CSV row and prints an ON line of any kind is
        # listed with a witness of that kind.  A run not yet in the CSV is counted, not required (listing it would
        # fail the present-exactly-once check above); a corpus row whose .out is not under --runs is not seen.
        nb = len(bad)
        n_on = n_csv = 0
        used = set(k for s in bkinds.values() for k in s)
        for fn, p in sorted(outs.items()):
            got = wl[fn] if fn in wl else witness_lines(p)
            on = [kd for kd, off in KINDS if any(ln != off for ln in got[kd])]
            if not on:
                continue
            used.update(on)
            n_on += 1
            run, _sep, jid = fn[:-len(".out")].rpartition("-")
            if (run, jid) not in cnt:
                continue
            n_csv += 1
            for kd in on:
                if fn not in listed:
                    bad.append("%s is a CSV row printing an ON %s line but is NOT listed" % (fn, kd))
                elif kind_of(listed[fn]["witness"]) != kd and kd not in (multi_kind_of(fn) or {}):
                    bad.append("%s prints an ON %s line but is listed with a %s witness"
                               % (fn, kd, kind_of(listed[fn]["witness"])))
        # the label names the kinds in use -- a listed witness's kind or an ON line found (CORRECTIONS 245; every KINDS
        # entry is scanned, see the next line)
        print("  completeness (%s): %d .out files print an ON line; %d are CSV rows, every one listed with its kind: %s;"
              " %d not in the CSV (not ingested, not required)"
              % (" / ".join(kd for kd, _off in KINDS if kd in used), n_on, n_csv, len(bad) == nb, n_on - n_csv))
        print("  kinds scanned (CORRECTIONS 245): %s; no prefix is a prefix of another, so no line is read as two kinds: %s"
              % (" / ".join(kd for kd, _off in KINDS), not collide))
        # TWO-KIND RUNS (CORRECTIONS 245): a listed run of a MULTI_KIND (batch, arm) prints exactly its registered line of
        # every kind registered there -- the kind its witness names and the one the TSV row cannot carry.
        nb = len(bad)
        n_two = 0
        for fn in sorted(listed):
            des = multi_kind_of(fn)
            if des is None or fn not in wl:
                continue
            n_two += 1
            for kd, _off in KINDS:
                if kd in des and wl[fn][kd] != [des[kd]]:
                    bad.append("%s is registered with %s ON but prints %r, not the registered line (MULTI_KIND)"
                               % (fn, kd, wl[fn][kd]))
        print("  two-kind runs (CORRECTIONS 245): %d listed runs of a registered two-kind (batch, arm), every one printing"
              " exactly its registered ON line of each kind: %s" % (n_two, len(bad) == nb))
    print("\nnoise-floor demonstration (227.6's definitions; cell = 15 CELLKEYS; complete, unsuperseded, std cell)")
    k = set(ks)
    variants = [("as 227 registered it: every cvt1- row dropped", [r for r in rows if not r["run"].startswith("cvt1-")]),
                ("FUTURE READER: listed rows dropped (cvt1 k01/HEAD kept)", [r for r in rows if (r["run"], r["job_id"]) not in k]),
                ("NAIVE: nothing dropped (the trap)", rows)]
    for lab, rr in variants:
        sp = _pooled(rr, "PlainNet18_c100")
        sr = _pooled(rr, "ResNet18_c100")
        print("  %-58s SIGMA_PLAIN %.6f (df %d, %d cells)  SIGMA_R18ALL %.6f (df %d)" % (lab, sp[0], sp[1], sp[2], sr[0], sr[1]))
    print("\nVERDICT: %s" % ("PASS" if not bad else "FAIL"))
    for b in bad:
        print("  FAIL " + b)
    return 0 if not bad else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--check" not in a:
        print(__doc__)
        sys.exit(2)
    rd = []
    if "--runs" in a:
        rd = [x for x in a[a.index("--runs") + 1:] if not x.startswith("--")]
    sys.exit(check(rd))
