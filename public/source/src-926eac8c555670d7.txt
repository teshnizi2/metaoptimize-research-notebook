#!/usr/bin/env python3
"""cvt3 REGISTRATION-TIME DERIVATIONS (CORRECTIONS 232).  Reads ONLY cvt1's landed probe records and
manifest (and, for S5, the corpus through analysis/corpus_exclusions.filter_rows).  The two tensor sets
it prints are copied into analysis/cVT3_downcoalition_score.py as LITERALS; that scorer's selftest
re-runs select_sets() on the host's cvt1 records and requires equality.  Nothing here reads a cvt3 run.

THE RULES, WRITTEN BEFORE ANY cvt3 RUN (232.3).  Every number below is computed mechanically from them.

  Records.  cvt1's MUTE arm (scalar grouping, VOTE_W=layer4.1.bn2.weight:0), seeds 78/79/80, every
            probe record with step >= 36 * 500 (epoch >= 36.0, i.e. after k01's pin at 36.2-36.6 and
            MUTE's own at 40.2-40.6): 320 records per seed, 960 pooled.
  Vote.     L_i = w_i * (b2 * m_tensor_i + (1 - b2) * z_tensor_i), b2 = the record's pt_b2, w = MUTE's
            registered weights (w_50 = 0, all others 1).  Lion moves beta by -ms * sign(sum_i L_i), so a
            tensor votes DOWN on a record iff L_i > 0.  share_i = |L_i| / sum_j |L_j| on that record.
  RULE C    (the DOWN coalition).  ELIGIBLE: i != 50 and DOWN on >= 0.90 of the 960 pooled records.
            RANK by the pooled mean share, descending (ties: lower index first).  CUMULATIVE MASS CUT
            = 1.00 of the eligible set's summed share: EVERY eligible tensor is kept.  (Why 1.00 and not
            0.90 / 0.95: cvt1's MUTE failed because the DOWN vote was RE-CARRIED by the tensors that were
            left voting.  A cut would re-create that trap by design.  Where 0.90 and 0.95 would have
            cut is printed, descriptive.)
            MUTE-DOWN silences {50} U C: every weight 0, scalar grouping.
  RULE K    (the count-matched NON-DOWN control).  POOL: i != 50, i not in C, DOWN on <= 0.50 of the 960
            pooled records (a majority of its records do NOT vote DOWN).  For each member c of C in RANK
            order, pick the unused pool tensor with the SAME module class (manifest column 4: Conv2d /
            BatchNorm2d / Linear) minimising |ln(numel_c / numel_i)|; ties -> smallest |i - c|; then the
            lower index.  If any member finds no candidate, the rule FAILS LOUDLY (no partial control).
            MUTE-CTL silences {50} U K: every weight 0, scalar grouping.
  VOTE_W    items in ascending index order, `<name>:0` joined by `/`; the patch prints its witness items
            in the same order as `<idx>:<name>:w=0.0:group=0:groupsize=53` joined by `,`.

Sections:
  S1  the vote table on the rule's records (every tensor): class, numel, pooled + per-seed DOWN fraction,
      pooled + per-seed mean share
  S2  RULE C: eligible set, rank, cumulative share, the 0.90/0.95 cut positions; the resulting set
  S3  RULE K: the pool, the matched pairs, the resulting set; count, class and numel disclosure; mass
  S4  VOTE_W strings and witness lines for MUTE50 / MUTEDOWN / MUTECTL
  S5  descriptive prior (NON-GATING): on cvt1's k01, MUTE and HEAD records, by phase, the fraction of
      records whose group-0 sum votes DOWN as run, and the fraction that would vote UP (L sum < 0) with
      {50} U C and with {50} U K silenced -- a record-level counterfactual on cvt1's trajectories, NOT a
      prediction of cvt3's; plus the same vote table restricted to epochs 17-36 (does RULE C pick the
      same set on the decline?)
  S6  the noise floor, 227.6's definitions, read through corpus_exclusions.filter_rows (OPERATIONS 36):
      SIGMA_R18ALL, SIGMA_PLAIN with the filter and naively; SIGMA_PRIOR = max of the filtered pair

  python3 analysis/cvt3_registration_derivations.py <runs dir holding cvt1/> [<all_runs.csv>]
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS_CVT1 = (78, 79, 80)
IDX_HEAD = 50
RULE_EPOCH_LO = 36.0
STEPS_PER_EPOCH = 500
DOWN_ELIGIBLE = 0.90
MASS_CUT = 1.00
CTL_POOL_MAX_DOWN = 0.50
PHASES = ((0.0, 5.0), (5.0, 17.0), (17.0, 36.0), (36.0, 100.01))
CELLKEYS = ["network", "dataset", "granularity", "base", "meta", "meta_stepsize", "alpha0", "gamma",
            "augment", "beta_clip", "batch_size", "epochs_requested", "hier", "lam", "eta_ratio"]


def manifest(cvt1dir):
    names, numel, cls = [], [], []
    for ln in open(os.path.join(cvt1dir, "PARTITION-MANIFEST.txt")):
        p = ln.split()
        if p and p[0] == "TENSOR":
            if int(p[1]) != len(names) + 1:
                raise ValueError("manifest TENSOR lines out of order")
            names.append(p[2])
            numel.append(int(p[3]))
            cls.append(p[4])
    if len(names) != 53 or names[IDX_HEAD - 1] != "layer4.1.bn2.weight":
        raise ValueError("cvt1 manifest is not the 53-tensor PlainNet with idx 50 = layer4.1.bn2.weight")
    return names, numel, cls


def records(cvt1dir, arm, seed):
    return [json.loads(x) for x in open(os.path.join(cvt1dir, "probe_cvt1-%s-s%d" % (arm, seed), "probe.jsonl"))
            if x.strip()]


def vote_terms(r, weights, group):
    b2 = r["pt_b2"]
    return dict((i, weights.get(i, 1.0) * (b2 * r["m_tensor"][i - 1] + (1.0 - b2) * r["z_tensor"][i - 1]))
                for i in group)


def vote_table(cvt1dir, arm, weights, lo, hi):
    """pooled + per-seed DOWN fraction and mean share over records with lo <= epoch < hi (scalar arm)."""
    group = list(range(1, 54))
    acc = dict((i, {"down": {}, "share": {}, "n": {}}) for i in group)
    for s in SEEDS_CVT1:
        for r in records(cvt1dir, arm, s):
            e = r["step"] / float(STEPS_PER_EPOCH)
            if not (lo <= e < hi):
                continue
            L = vote_terms(r, weights, group)
            tot = math.fsum(abs(v) for v in L.values())
            for i in group:
                a = acc[i]
                a["down"][s] = a["down"].get(s, 0) + (L[i] > 0)
                a["share"][s] = a["share"].get(s, 0.0) + (abs(L[i]) / tot if tot > 0 else 0.0)
                a["n"][s] = a["n"].get(s, 0) + 1
    out = {}
    for i in group:
        a = acc[i]
        n = sum(a["n"].values())
        out[i] = {"n": n, "n_seed": dict(a["n"]),
                  "down": sum(a["down"].values()) / float(n),
                  "down_seed": dict((s, a["down"][s] / float(a["n"][s])) for s in SEEDS_CVT1),
                  "share": math.fsum(a["share"].values()) / float(n),
                  "share_seed": dict((s, a["share"][s] / float(a["n"][s])) for s in SEEDS_CVT1)}
    return out


def rule_c(tab):
    elig = [i for i in sorted(tab) if i != IDX_HEAD and tab[i]["down"] >= DOWN_ELIGIBLE]
    ranked = sorted(elig, key=lambda i: (-tab[i]["share"], i))
    total = math.fsum(tab[i]["share"] for i in ranked)
    cum, rows, kept = 0.0, [], []
    for i in ranked:
        before = cum
        cum += tab[i]["share"]
        rows.append((i, tab[i]["share"], cum, cum / total))
        if before / total < MASS_CUT - 1e-12:
            kept.append(i)
    return ranked, rows, total, kept


def rule_k(tab, coal_ranked, names, numel, cls):
    pool = [i for i in sorted(tab) if i != IDX_HEAD and i not in coal_ranked and tab[i]["down"] <= CTL_POOL_MAX_DOWN]
    used, pairs = set(), []
    for c in coal_ranked:
        cand = [i for i in pool if i not in used and cls[i - 1] == cls[c - 1]]
        if not cand:
            raise SystemExit("RULE K FAILS: no unused %s in the NON-DOWN pool for %d %s" % (cls[c - 1], c, names[c - 1]))
        pick = min(cand, key=lambda i: (abs(math.log(float(numel[c - 1]) / numel[i - 1])), abs(i - c), i))
        used.add(pick)
        pairs.append((c, pick))
    return pool, pairs


def select_sets(cvt1dir):
    """-> (coalition in RANK order, control in PAIR order) as 1-based indices.  The scorer's selftest calls this."""
    names, numel, cls = manifest(cvt1dir)
    tab = vote_table(cvt1dir, "MUTE", {IDX_HEAD: 0.0}, RULE_EPOCH_LO, 1e9)
    _ranked, _rows, _tot, kept = rule_c(tab)
    _pool, pairs = rule_k(tab, kept, names, numel, cls)
    return kept, [k for _c, k in pairs]


def votew_string(idxs, names):
    return "/".join("%s:0" % names[i - 1] for i in sorted(idxs))


def witness_line(idxs, names):
    return "VOTE_W: on type=scalar items=" + ",".join(
        "%d:%s:w=0.0:group=0:groupsize=53" % (i, names[i - 1]) for i in sorted(idxs))


def pooled_sigma(rows, net):
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
    return (math.sqrt(ss / df) if df else None), df, nc


def main():
    runs = sys.argv[1]
    cvt1dir = os.path.join(runs, "cvt1")
    csvp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(HERE), "results", "all_runs.csv")
    names, numel, cls = manifest(cvt1dir)

    print("S1  THE VOTE TABLE -- cvt1 MUTE (w_50 = 0), seeds 78/79/80, records with epoch >= %.1f" % RULE_EPOCH_LO)
    tab = vote_table(cvt1dir, "MUTE", {IDX_HEAD: 0.0}, RULE_EPOCH_LO, 1e9)
    print("    records per seed %s, pooled %d" % ("/".join(str(tab[1]["n_seed"][s]) for s in SEEDS_CVT1), tab[1]["n"]))
    print("    idx name                      class        numel    DOWN pooled (s78/s79/s80)        share pooled (s78/s79/s80)")
    for i in range(1, 54):
        t = tab[i]
        print("    %2d  %-25s %-11s %8d  %.4f (%s)  %.6f (%s)" % (
            i, names[i - 1], cls[i - 1], numel[i - 1], t["down"], "/".join("%.3f" % t["down_seed"][s] for s in SEEDS_CVT1),
            t["share"], "/".join("%.5f" % t["share_seed"][s] for s in SEEDS_CVT1)))

    print("\nS2  RULE C -- eligible: i != 50 and pooled DOWN >= %.2f; rank by pooled share; cumulative mass cut %.2f" % (DOWN_ELIGIBLE, MASS_CUT))
    ranked, rows, total, kept = rule_c(tab)
    for k, (i, sh, cum, frac) in enumerate(rows, 1):
        print("    rank %2d  %2d %-25s %-11s %8d  DOWN %.4f  share %.6f  cumulative %.6f = %.4f of eligible"
              % (k, i, names[i - 1], cls[i - 1], numel[i - 1], tab[i]["down"], sh, cum, frac))
    for cut in (0.90, 0.95):
        n_ = next(k for k, (_i, _s, _c, f) in enumerate(rows, 1) if f >= cut - 1e-12)
        print("    (descriptive) a %.2f cut would keep the first %d of %d" % (cut, n_, len(rows)))
    print("    eligible %d, summed share %.6f of the mean |L| mass; KEPT at cut %.2f: %d" % (len(rows), total, MASS_CUT, len(kept)))
    print("    near the eligibility bar (0.80 <= DOWN < 0.90, NOT eligible): %s" % (
        ", ".join("%d %s %.4f" % (i, names[i - 1], tab[i]["down"]) for i in range(1, 54)
                  if i != IDX_HEAD and 0.80 <= tab[i]["down"] < DOWN_ELIGIBLE) or "none"))
    print("    COALITION C (rank order): %s" % " ".join(str(i) for i in kept))
    print("    classes: %s" % ", ".join("%s %d" % (c, sum(cls[i - 1] == c for i in kept)) for c in ("BatchNorm2d", "Conv2d", "Linear")))

    print("\nS3  RULE K -- pool: i != 50, i not in C, pooled DOWN <= %.2f; same module class; nearest |ln numel ratio|; ties |i-c|, lower i" % CTL_POOL_MAX_DOWN)
    pool, pairs = rule_k(tab, kept, names, numel, cls)
    print("    pool (%d): %s" % (len(pool), " ".join("%d" % i for i in pool)))
    excl = [i for i in range(1, 54) if i != IDX_HEAD and i not in kept and i not in pool]
    print("    neither C nor pool (0.50 < DOWN < 0.90): %s" % ", ".join("%d %s %.3f" % (i, names[i - 1], tab[i]["down"]) for i in excl))
    for c, k in pairs:
        print("    %2d %-24s %-11s %8d share %.6f DOWN %.3f  <->  %2d %-24s %-11s %8d share %.6f DOWN %.3f  numel ratio %.1f"
              % (c, names[c - 1], cls[c - 1], numel[c - 1], tab[c]["share"], tab[c]["down"],
                 k, names[k - 1], cls[k - 1], numel[k - 1], tab[k]["share"], tab[k]["down"],
                 float(numel[c - 1]) / numel[k - 1]))
    ctl = [k for _c, k in pairs]
    print("    CONTROL K (pair order): %s" % " ".join(str(i) for i in ctl))
    print("    count C %d == K %d; classes C %s | K %s" % (
        len(kept), len(ctl),
        ", ".join("%s %d" % (c, sum(cls[i - 1] == c for i in kept)) for c in ("BatchNorm2d", "Conv2d", "Linear")),
        ", ".join("%s %d" % (c, sum(cls[i - 1] == c for i in ctl)) for c in ("BatchNorm2d", "Conv2d", "Linear"))))
    print("    params C %d | K %d (ratio %.2f); BatchNorm2d pairs numel-equal: %d of %d; Conv2d params C %d | K %d (ratio %.1f)" % (
        sum(numel[i - 1] for i in kept), sum(numel[i - 1] for i in ctl),
        float(sum(numel[i - 1] for i in kept)) / sum(numel[i - 1] for i in ctl),
        sum(numel[c - 1] == numel[k - 1] for c, k in pairs if cls[c - 1] == "BatchNorm2d"),
        sum(cls[c - 1] == "BatchNorm2d" for c in kept),
        sum(numel[i - 1] for i in kept if cls[i - 1] == "Conv2d"), sum(numel[i - 1] for i in ctl if cls[i - 1] == "Conv2d"),
        float(sum(numel[i - 1] for i in kept if cls[i - 1] == "Conv2d")) / max(1, sum(numel[i - 1] for i in ctl if cls[i - 1] == "Conv2d"))))
    sc = math.fsum(tab[i]["share"] for i in kept)
    sk = math.fsum(tab[i]["share"] for i in ctl)
    snd = math.fsum(tab[i]["share"] for i in range(1, 54) if i != IDX_HEAD and i not in kept)
    print("    MASS on the rule's records: C %.6f | K %.6f (C/K %.1f) | every non-C tensor together %.6f -- a mass-matched"
          % (sc, sk, sc / sk, snd))
    print("    NON-DOWN control does not exist on this net: K is matched on COUNT and CLASS, not on mass (disclosed).")

    print("\nS4  VOTE_W STRINGS AND WITNESS LINES (ascending index)")
    sets = {"MUTE50": [IDX_HEAD], "MUTEDOWN": sorted([IDX_HEAD] + kept), "MUTECTL": sorted([IDX_HEAD] + ctl)}
    for arm in ("MUTE50", "MUTEDOWN", "MUTECTL"):
        v = votew_string(sets[arm], names)
        print("    %-8s items %d  VOTE_W (%d chars) %s" % (arm, len(sets[arm]), len(v), v))
        print("    %-8s WITNESS %s" % (arm, witness_line(sets[arm], names)))

    print("\nS5  DESCRIPTIVE PRIOR (NON-GATING): record-level counterfactual on cvt1's own trajectories, group 0")
    print("    DOWN-as-run = fraction of records whose weighted group-0 sum > 0 (beta moved DOWN);")
    print("    UP-if-C / UP-if-K = fraction whose sum < 0 after ALSO silencing {50} U C / {50} U K.")
    sil_c = set([IDX_HEAD] + kept)
    sil_k = set([IDX_HEAD] + ctl)
    for arm, w, grp in (("k01", {}, list(range(1, 54))), ("MUTE", {IDX_HEAD: 0.0}, list(range(1, 54))),
                        ("HEAD", {}, [i for i in range(1, 54) if i != IDX_HEAD])):
        for s in SEEDS_CVT1:
            recs = records(cvt1dir, arm, s)
            cells = []
            for lo, hi in PHASES:
                ph = [r for r in recs if lo <= r["step"] / float(STEPS_PER_EPOCH) < hi]
                n = len(ph)
                d_run = u_c = u_k = 0
                for r in ph:
                    L = vote_terms(r, w, grp)
                    d_run += math.fsum(L.values()) > 0
                    u_c += math.fsum(v for i, v in L.items() if i not in sil_c) < 0
                    u_k += math.fsum(v for i, v in L.items() if i not in sil_k) < 0
                cells.append("ep %g-%g n %d: DOWN-as-run %.3f UP-if-C %.3f UP-if-K %.3f"
                             % (lo, min(hi, 100), n, d_run / float(n), u_c / float(n), u_k / float(n)))
            bmax = max(r["beta"][0] for r in recs)
            print("    %-4s s%d  beta[0] max %.3f | %s" % (arm, s, bmax, " | ".join(cells)))
    tab2 = vote_table(cvt1dir, "MUTE", {IDX_HEAD: 0.0}, 17.0, 36.0)
    _r2, _rows2, _t2, kept2 = rule_c(tab2)
    print("    RULE C applied to MUTE's epoch 17-36 records instead (descriptive): %d tensors %s; same set as registered: %s; "
          "in registered C only: %s; in this one only: %s"
          % (len(kept2), " ".join(str(i) for i in kept2), set(kept2) == set(kept),
             sorted(set(kept) - set(kept2)), sorted(set(kept2) - set(kept))))

    print("\nS6  THE NOISE FLOOR (227.6's definitions) READ THROUGH corpus_exclusions.filter_rows (OPERATIONS 36)")
    sys.path.insert(0, HERE)
    import corpus_exclusions
    raw = [r for r in csv.DictReader(open(csvp)) if not r.get("run", "").startswith("cvt3-")]
    filt = corpus_exclusions.filter_rows(raw)
    print("    corpus %d rows (cvt3- excluded: 0 exist), filtered %d (dropped %d listed rows)" % (len(raw), len(filt), len(raw) - len(filt)))
    res = {}
    for lab, rows_ in (("FILTERED", filt), ("NAIVE", raw)):
        for net, key in (("ResNet18_c100", "SIGMA_R18ALL"), ("PlainNet18_c100", "SIGMA_PLAIN")):
            s_, df_, nc_ = pooled_sigma(rows_, net)
            res[(lab, key)] = s_
            print("    %-8s %-13s %r  (df %d, %d cells)" % (lab, key, s_, df_, nc_))
    prior = max(res[("FILTERED", "SIGMA_R18ALL")], res[("FILTERED", "SIGMA_PLAIN")])
    print("    SIGMA_PRIOR = max(FILTERED SIGMA_R18ALL, FILTERED SIGMA_PLAIN) = %r;  SE = %.6f" % (prior, prior * math.sqrt(2.0 / 3.0)))
    print("    (NAIVE SIGMA_PLAIN is the trap OPERATIONS 36 names; it feeds nothing)")


if __name__ == "__main__":
    main()
