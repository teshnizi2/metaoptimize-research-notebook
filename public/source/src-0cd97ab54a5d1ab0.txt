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
           intervention kind the witness names: VOTE_W or BETA_HOLD, see KINDS), and every OTHER .out of a
           listed batch carries that kind's `off` line (so the list is complete for that batch);
           COMPLETENESS (CORRECTIONS 239): every .out found under --runs that is a CSV row and prints an ON
           line of any kind is listed, with a witness of that kind (ON runs not yet in the CSV are counted,
           not required; logs not under --runs are not read, and without --runs no log is read at all);
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
    ("BETA_HOLD", "BETA_HOLD: off"),  # patches/patch_betahold.py, CORRECTIONS 237 (cvt4)
]


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
        for fn, p in sorted(outs.items()):
            got = wl[fn] if fn in wl else witness_lines(p)
            on = [kd for kd, off in KINDS if any(ln != off for ln in got[kd])]
            if not on:
                continue
            n_on += 1
            run, _sep, jid = fn[:-len(".out")].rpartition("-")
            if (run, jid) not in cnt:
                continue
            n_csv += 1
            for kd in on:
                if fn not in listed:
                    bad.append("%s is a CSV row printing an ON %s line but is NOT listed" % (fn, kd))
                elif kind_of(listed[fn]["witness"]) != kd:
                    bad.append("%s prints an ON %s line but is listed with a %s witness"
                               % (fn, kd, kind_of(listed[fn]["witness"])))
        print("  completeness (%s): %d .out files print an ON line; %d are CSV rows, every one listed with its kind: %s;"
              " %d not in the CSV (not ingested, not required)"
              % (" / ".join(kd for kd, _off in KINDS), n_on, n_csv, len(bad) == nb, n_on - n_csv))
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
