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
           twice; with --runs, the run's own raw .out carries exactly the listed witness line, and every
           OTHER .out of a listed batch carries `VOTE_W: off` (so the list is complete for that batch);
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
        n_w = n_off = 0
        for fn, p in sorted(outs.items()):
            if fn.split("-")[0] not in batches:
                continue
            vw = [ln.rstrip("\n") for ln in open(p, errors="replace") if ln.startswith("VOTE_W")]
            if fn in listed:
                if vw != [listed[fn]["witness"]]:
                    bad.append("%s witness %r != listed" % (fn, vw))
                n_w += 1
            else:
                if vw != ["VOTE_W: off"]:
                    bad.append("%s is NOT listed but its witness is %r" % (fn, vw))
                n_off += 1
        if n_w != len(ents):
            bad.append("found %d of %d listed .out files" % (n_w, len(ents)))
        print("  raw .out witness: %d listed runs carry their listed line; %d unlisted runs of the same batch print `VOTE_W: off`"
              % (n_w, n_off))
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
