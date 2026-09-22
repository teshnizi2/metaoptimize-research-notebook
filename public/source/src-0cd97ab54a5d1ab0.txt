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
           intervention kind the witness names: VOTE_W, BETA_HOLD, GROUP_HOLD, COMP_HOLD, REST_HOLD, WINDOW_HOLD,
           DECAY_MASK or SHADOW_VOTE, see
           KINDS), and
           every OTHER .out of a listed batch carries that kind's `off` line (so the list is complete for that
           batch);
           COMPLETENESS (CORRECTIONS 239): every .out found under --runs that is a CSV row and prints an ON
           line of any kind is listed, with a witness of that kind (ON runs not yet in the CSV are counted,
           not required; logs not under --runs are not read, and without --runs no log is read at all);
           TWO-KIND RUNS (CORRECTIONS 245): a run whose registered design turns on two kinds (cvt6's forced
           arms: BETA_HOLD + COMP_HOLD) is listed ONCE, by either ON line, and must print exactly the lines
           MULTI_KIND registers for its (batch, arm) -- so its other ON line is verified, not skipped;
           CORRECTIONS 251: any number of kinds, e.g. cvt8's forced arms (GROUP_HOLD + REST_HOLD), cvt9's held arms
           (BETA_HOLD + COMP_HOLD) and its EARLY / LATE (+ WINDOW_HOLD, three kinds);
           CORRECTIONS 269: cwd2's HIGHWD0 / LOWWD0 (BETA_HOLD + COMP_HOLD + DECAY_MASK) and its HIGHHEADPATH (two);
           cwd1's masked arms and csv1's switch arms turn on ONE kind each and register nothing there;
           ARGS-VALUE KINDS (CORRECTIONS 263): a run may deviate from the standard cell in a CLI FLAG alone
           (cmo1's `--momentum-param-base 0.9`, `--weight-decay-base 0`), which no CSV column carries and no patch
           line announces.  Such a row is listed with an ARGS witness `<KIND>: <flag>=<value>` (ARGS_KINDS);
           `--check --runs` verifies it against the run's OWN `ARGS:` line read with argparse semantics, requires
           every INGESTED standard-cell run whose ARGS deviates to be listed, holds the unlisted runs of a listed
           batch to the standard values, and FAILs if one 15-key cell pools unlisted rows with different values;
           TWO-AXIS RUNS (CORRECTIONS 284): a run may do BOTH -- cwd5's CARW2 prints an ON DECAY_MASK line AND runs
           at a non-standard weight decay -- and a TSV row carries ONE witness.  Such a run is listed with its ARGS
           witness and its ON kinds are registered in MULTI_KIND: the only listing both readers accept, since the
           completeness reader has that escape and the ARGS-value reader has none.  Each ON line is still held to
           its registered string, and the reverse listing still FAILs;
           TWO-ARGS RUNS (CORRECTIONS 294): a run may deviate on TWO ARGS kinds at once -- caw2's XS / XL run the
           AdamW base at `--momentum-param-base 0.9` AND the dose `--weight-decay-base 1.0`.  Such a run is listed
           ONCE, by the ARGS witness of any one of its deviating kinds, and MULTI_ARGS registers the witness of EVERY
           ARGS kind its (batch, arm) deviates on; each registered value is held to the run's own ARGS line, and a
           two-ARGS run of an arm with no MULTI_ARGS entry still FAILs;
           then prints the noise-floor demonstration: the registered cvt1 sigmas (227.6) re-derived on
           the current corpus three ways -- as 227 did (every `cvt1-` row dropped), as a future
           registration should (only the listed rows dropped), and naively (nothing dropped).
Exit 0 all checks pass, 1 any check fails.
"""
import csv
import math
import os
import re
import shlex
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
    ("REST_HOLD", "REST_HOLD: off"),  # patches/patch_resthold.py, CORRECTIONS 248 (cvt8); added at 251
    ("WINDOW_HOLD", "WINDOW_HOLD: off"),  # patches/patch_windowhold.py, CORRECTIONS 249 (cvt9); added at 251
    ("DECAY_MASK", "DECAY_MASK: off"),  # patches/patch_decaymask.py, CORRECTIONS 260 / 261 (cwd1, cwd2); added at 269
    ("SHADOW_VOTE", "SHADOW_VOTE: off"),  # patches/patch_shadowvote.py, CORRECTIONS 262 (csv1); added at 269
]
# No prefix above is a prefix of another (their first letters V / B / G / C / R / W / D / S differ), so no line starts with two of
# them and every `startswith` reader selects each line for ONE kind (CORRECTIONS 245); check() FAILs if an entry breaks it.
KINDS_AT_251 = 6   # CORRECTIONS 269: 251's `kinds scanned` line is frozen over the first six entries, so it stays byte-identical

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
# CORRECTIONS 251: cvt8 (248) and cvt9 (249), appended; the entries above are unchanged.  A (batch, arm) may register ANY
# number of kinds -- check() already reads MULTI_KIND per kind -- and cvt9's EARLY / LATE register three.  Re-typed from
# the registered scorers (NOT imported): analysis/cVT8_doseroute_score.py WITNESS_GH / WITNESS_RH for FORCED, and
# analysis/cVT9_dosewindow_score.py WITNESS_BH / WITNESS_CH for FORCED (= HELD) + WITNESS_WH for WINDOWED.  cvt9's
# BETA_HOLD tri:9428 / floor lines and its COMP_HOLD line are byte-identical to cvt6's (cVT9 selftest B), so they are reused.
# tests/test_corpus_exclusions_check.py C23 pins every entry to the scorers' tables.
_CVT8_GH = "GROUP_HOLD: on type=blockwise group=1 groupsize=3 names=layer4.0.bn2.weight+layer4.0.shortcut.1.weight+layer4.1.bn2.weight"
_CVT8_GH_TRI_8609 = _CVT8_GH + " mode=tri P=8609 b0=-13.815510749816895 ms=0.001 lo=-15.0 hi=-2.3026 peak=-5.2065107498168945"
_CVT8_GH_TRI_9428 = _CVT8_GH + " mode=tri P=9428 b0=-13.815510749816895 ms=0.001 lo=-15.0 hi=-2.3026 peak=-4.387510749816894"
_CVT8_GH_FLOOR = _CVT8_GH + " mode=floor value=-15.0"
_CVT8_RH_REC = ("REST_HOLD: on type=blockwise group=0 groupsize=59 mode=rec id=cvt8_isopath "
                "sha256=08ab25f3a329cb260bb39fb72f3299c021e7169bf612fa27d539166296e28e70 knots=500 n0=2 n1=49902 "
                "b0=-13.815510749816895 lo=-15.0 hi=-2.3026 vmax=-4.985378742218018 vlast=-14.924964427947998")
_CVT9_BH_TRI_7235 = ("BETA_HOLD: on type=blockwise group=1 groupsize=1 name=layer4.1.bn2.weight mode=tri P=7235 "
                     "b0=-13.815510749816895 ms=0.001 lo=-15.0 hi=-2.3026 peak=-6.580510749816894")
_CVT9_BH_TRI_8609 = ("BETA_HOLD: on type=blockwise group=1 groupsize=1 name=layer4.1.bn2.weight mode=tri P=8609 "
                     "b0=-13.815510749816895 ms=0.001 lo=-15.0 hi=-2.3026 peak=-5.2065107498168945")
_CVT9_WH = "WINDOW_HOLD: on type=blockwise group=1 name=layer4.1.bn2.weight base=tri P=9428"
_CVT9_WH_EARLY = _CVT9_WH + " n0=0 n1=9429 outside=floor value=-15.0"
_CVT9_WH_LATE = _CVT9_WH + " n0=9429 n1=end outside=floor value=-15.0"
MULTI_KIND.update({
    ("cvt8", "HIGHISOPATH"): {"GROUP_HOLD": _CVT8_GH_TRI_8609, "REST_HOLD": _CVT8_RH_REC},
    ("cvt8", "BIGISOPATH"): {"GROUP_HOLD": _CVT8_GH_TRI_9428, "REST_HOLD": _CVT8_RH_REC},
    ("cvt8", "LOWISOPATH"): {"GROUP_HOLD": _CVT8_GH_FLOOR, "REST_HOLD": _CVT8_RH_REC},
    ("cvt9", "LOWHEADPATH"): {"BETA_HOLD": _CVT6_BH_FLOOR, "COMP_HOLD": _CVT6_CH_REC},
    ("cvt9", "HIGHHEADPATH"): {"BETA_HOLD": _CVT6_BH_TRI, "COMP_HOLD": _CVT6_CH_REC},
    ("cvt9", "MIDDOSE"): {"BETA_HOLD": _CVT9_BH_TRI_7235, "COMP_HOLD": _CVT6_CH_REC},
    ("cvt9", "RESDOSE"): {"BETA_HOLD": _CVT9_BH_TRI_8609, "COMP_HOLD": _CVT6_CH_REC},
    ("cvt9", "EARLY"): {"BETA_HOLD": _CVT6_BH_TRI, "COMP_HOLD": _CVT6_CH_REC, "WINDOW_HOLD": _CVT9_WH_EARLY},
    ("cvt9", "LATE"): {"BETA_HOLD": _CVT6_BH_TRI, "COMP_HOLD": _CVT6_CH_REC, "WINDOW_HOLD": _CVT9_WH_LATE},
})
# CORRECTIONS 269: cwd2 (261), appended; every entry above is unchanged.  cwd2 runs cvt9's HEADPATH schedules with
# PATCH_DECAYMASK on top, so its HIGHWD0 / LOWWD0 print THREE ON lines (BETA_HOLD + COMP_HOLD + DECAY_MASK) and its
# HIGHHEADPATH the same TWO cvt9's HIGHHEADPATH prints.  Re-typed from the registered tables (NOT imported):
# analysis/cwd_design.py CWD2.WITNESS_BH / WITNESS_CH (= cvt9's, by construction: 261.5 loads the sha-pinned cvt9
# scorer) and CWD2.WITNESS_DM, which analysis/cWD2_carrierwd_score.py imports UNEDITED.  cwd2's BETA_HOLD tri:9428 /
# floor lines and its COMP_HOLD line are byte-identical to cvt6's / cvt9's, so they are reused.  `cwd1` (260) and
# `csv1` (262) register NO entry here: no arm of either turns on two kinds -- cwd1's masked arms print only
# DECAY_MASK (VOTE_W / BETA_HOLD / GROUP_HOLD / REST_HOLD off) and csv1's switch arms only SHADOW_VOTE, while its
# MUTE arm is a plain VOTE_W row of the kind 227 already knows.
# tests/test_corpus_exclusions_check.py C33 pins every entry to those tables and proves the three batches' arm sets.
_CWD2_DM = ("DECAY_MASK: on base=SGDm wd=0.1 spec=layer4.1.bn2.weight masked=1 of=53 numel=512 idx=50 "
            "names=layer4.1.bn2.weight")
MULTI_KIND.update({
    ("cwd2", "HIGHHEADPATH"): {"BETA_HOLD": _CVT6_BH_TRI, "COMP_HOLD": _CVT6_CH_REC},
    ("cwd2", "HIGHWD0"): {"BETA_HOLD": _CVT6_BH_TRI, "COMP_HOLD": _CVT6_CH_REC, "DECAY_MASK": _CWD2_DM},
    ("cwd2", "LOWWD0"): {"BETA_HOLD": _CVT6_BH_FLOOR, "COMP_HOLD": _CVT6_CH_REC, "DECAY_MASK": _CWD2_DM},
})
# ---- CORRECTIONS 284: TWO-AXIS RUNS -- an ON `<KIND>` line AND a deviating ARGS value on the SAME run --------------
# `cwd5`'s `CARW2` (CORRECTIONS 281) is the first run of the campaign that deviates on BOTH axes this module reads: it
# prints an ON `DECAY_MASK` line (the three `ctd1` carriers masked) AND runs at `--weight-decay-base 1e-2` instead of
# the standard cell's `0.1` (it sits on the ladder's rung W2).  A TSV row carries ONE witness, and 281.12 proved on an
# isolated copy that BOTH listings FAILed the real check(): the KINDS reader (completeness, below) refuses an ON line
# whose kind the witness does not name, and the ARGS-VALUE reader refuses a deviating run listed with another kind.
#
# THE RULE, and it is FORCED, not preferred: such a run is listed with its **ARGS** witness, and every kind it turns ON
# is registered here.  The completeness reader has a MULTI_KIND escape and the ARGS-value reader has none, so the ARGS
# listing is the ONLY one either reader can accept; the reverse listing still FAILs at "deviates on <KIND> but is
# listed with a <KIND> witness", so the module enforces the rule rather than documenting it (C35a).  Nothing is
# weakened: the MULTI_KIND loop below then holds the run's ON line to the registered string byte for byte, the batch
# rule holds the batch's unlisted runs to `DECAY_MASK: off`, and an ON line of a kind NOT registered here still FAILs.
# An entry may register ONE kind for this reason, where 245's entries register two or more.
#
# Re-typed (NOT imported) from the frozen table `analysis/cwd5_design.py` `CWD5.WITNESS_DM["CARW2"]`, which the
# registered `analysis/cWD5_wdladder_score.py` imports UNEDITED; the `wd=0.01` token is the ARM'S OWN rung, the string
# PATCH_DECAYMASK had never printed before this batch (281.6 RW3).  tests/test_corpus_exclusions_check.py C36 pins the
# literal to that table; C34 / C35 run the rule and its corruptions.
_CWD5_DM_CARW2 = ("DECAY_MASK: on base=SGDm wd=0.01 spec=layer4.0.bn2.weight+layer4.0.shortcut.1.weight+"
                  "layer4.1.bn2.weight masked=3 of=62 numel=1536 idx=50,53,59 "
                  "names=layer4.0.bn2.weight,layer4.0.shortcut.1.weight,layer4.1.bn2.weight")
MULTI_KIND.update({
    ("cwd5", "CARW2"): {"DECAY_MASK": _CWD5_DM_CARW2},
})
MULTI_KINDS_AT_251 = 2  # 251's `multi-kind runs` line is computed over the entries registering 2+ kinds, so it stays
#                         byte-identical as one-kind two-axis entries are added; the added ones get their own line.


# ---- --check only: ARGS-VALUE witness kinds (CORRECTIONS 263) ---------------------------------------------------
# `cmo1` (CORRECTIONS 255) changes the BASE optimiser through CLI FLAGS, not a patched tree: its M9* arms run
# `--momentum-param-base 0.9` and its W0* arms `--weight-decay-base 0`, everything else at the standard cell.
# Neither flag is a column of results/all_runs.csv, so aggregate.py writes those rows with the PLAIN arm's cell key
# (255.7) and they would pool into the 0.99 / 0.1 cells.  There is no `<KIND>: on` line to read here -- the witness
# is the run's OWN `ARGS:` line, the one prefix every run prints, read with argparse semantics (the LAST occurrence
# of a repeated flag wins), exactly as analysis/argsline_guard.py reads it under STANDING RULE 20.
#
# argsline_guard is a REGISTERED file and is NOT imported (this module stays stdlib-only and standalone, as it
# re-types 227.6's sigmas and the scorers' MULTI_KIND lines instead of importing them): its tokenizer and last-wins
# rule are re-typed below, and tests/test_corpus_exclusions_check.py C24 pins them to it on real and synthetic
# ARGS lines (equal effective dicts, repeated flags and `--flag=value` included).
#
# One entry per flag: (kind, flag, standard value).  A run DEVIATES on a kind when its own ARGS line carries that
# flag with a value that is not NUMERICALLY the standard; an absent flag is the standard (argparse's default), so
# older logs written before the flag existed do not deviate.  A future factor flag is one line here.
# load / keys / is_excluded / filter_rows ignore all of this, as they ignore KINDS.
ARGS_KINDS = [
    ("ARGS_MOMENTUM_BASE", "momentum-param-base", "0.99"),  # cmo1's M9 arms (255.2); SGDm base momentum
    ("ARGS_WD_BASE", "weight-decay-base", "0.1"),           # cmo1's W0 arms (255.2); base weight decay
]
ARGS_LINE_PREFIX = "ARGS:"   # jobs/run_cifar.sh echoes it on EVERY run, before train.py is called
# PREFIX GUARANTEE (CORRECTIONS 245's, extended).  A witness is read as ONE kind because no name of KINDS +
# ARGS_KINDS is a prefix of another (V / B / G / C / R / W / ARGS_M / ARGS_W).  The two readers cannot cross
# either: no KINDS prefix is a prefix of `ARGS:` and none starts with `ARGS`, so witness_lines never collects an
# `ARGS:` line; and an ARGS-kind name is followed by `_`, not `:`, so no `<KIND>: ...` line is read as an ARGS
# line.  check() computes both and FAILs if an entry ever breaks them.
#
# ---- CORRECTIONS 294: TWO-ARGS RUNS -- two ARGS kinds deviating on the SAME run ---------------------------------
# `caw2` (CORRECTIONS 290) swaps the base optimiser to AdamW, whose harness flags carry `--momentum-param-base 0.9`
# (ARGS_MOMENTUM_BASE's non-standard value: the CSV `base` column says AdamW but no column carries the flag), and its
# X cell runs that recipe at `--weight-decay-base 1.0`, a DOSE arm.  So XS / XL deviate on BOTH ARGS kinds, a TSV row
# carries ONE witness, and the ARGS-value reader below FAILs a run that deviates on a kind its witness does not name
# ("deviates on <KIND> but is listed with a <KIND> witness") -- 284.8's declared limit, owed by 290.9.
#
# THE RULE (the ARGS axis's counterpart of 245's MULTI_KIND): such a run is listed ONCE, by the ARGS witness of ANY one
# of its deviating kinds, and MULTI_ARGS registers, for its (batch, arm), the witness of EVERY ARGS kind it deviates
# on.  The ARGS-value reader then accepts the kinds the row does not carry ONLY for a registered (batch, arm) and ONLY
# when the row is listed by an ARGS witness; the multi-ARGS block holds every registered witness to the run's OWN
# `ARGS:` line and the registered kind SET to the run's own deviating set.  Nothing is weakened: an arm with no entry
# still FAILs exactly as before (C26d / C38g), and the escape buys a stricter check on the kinds it covers.
# An entry registers TWO OR MORE kinds (check() FAILs a smaller one) in `args_witness` form, verbatim value.
#
# Re-typed (NOT imported) from the registered design `analysis/caw2_design.py` (`args_pairs`: AdamW base
# momentum-param-base "0.9"; ARM_TABLE: XS / XL wd token "1.0"; `args_deviating_kinds` names exactly XS / XL as the two-
# kind arms).  tests/test_corpus_exclusions_check.py C39 pins these literals to that design; C37 / C38 run the rule.
_CAW2_X_ARGS = {"ARGS_MOMENTUM_BASE": "ARGS_MOMENTUM_BASE: momentum-param-base=0.9",
                "ARGS_WD_BASE": "ARGS_WD_BASE: weight-decay-base=1.0"}
MULTI_ARGS = {
    ("caw2", "XS"): dict(_CAW2_X_ARGS),
    ("caw2", "XL"): dict(_CAW2_X_ARGS),
}


def multi_args_of(fn):
    """-> the registered {ARGS kind: witness} of a `<batch>-<arm>-...` run name or .out file name, or None."""
    return MULTI_ARGS.get(tuple(fn.split("-")[:2]))


# SCOPE of the completeness check.  The standard values above are the MECHANISM LINE's cell (255.2: ResNet18_c100 /
# PlainNet18_c100, CIFAR-100, 100 epochs, aug 1, clip -15:-2.3026, ms 1e-3, alpha0 1e-6, batch 100 -- `_pooled`'s
# filter, re-typed as STD_CELL / STD_NETWORKS).  Older corpus batches outside that cell ran momentum 0.9 as THEIR
# standard (128 ingested rows in 12 batches on 2026-09-18), so "deviates" is required-to-be-listed only INSIDE the
# standard cell; outside it the check counts the rows (DESCRIPTIVE) and the cell-mixing rule below still FAILs if
# any one 15-key cell would pool unlisted rows carrying different values.
STD_CELL = [("dataset", "CIFAR100"), ("epochs_requested", "100"), ("augment", "1"), ("beta_clip", "-15:-2.3026"),
            ("meta_stepsize", "1e-3"), ("alpha0", "1e-6"), ("batch_size", "100")]
STD_NETWORKS = ("PlainNet18_c100", "ResNet18_c100")
_ARGS_RE = re.compile(r"^\s*ARGS:\s*(.*)$")


def args_kind_names():
    return [k for k, _f, _s in ARGS_KINDS]


def args_witness(kind, value):
    """-> the TSV `witness` string for an ARGS kind: `<KIND>: <flag>=<value>`, the value verbatim from the run."""
    flag = dict((k, f) for k, f, _s in ARGS_KINDS)[kind]
    return "%s: %s=%s" % (kind, flag, value)


def _args_tokens(payload):
    """argsline_guard.tokenize, re-typed: shlex so quoted values survive, plain split if the line is not lexable."""
    try:
        return shlex.split(payload)
    except ValueError:
        return payload.split()


def _args_effective(payload_or_tokens):
    """argsline_guard.parse_flags + effective, re-typed: flag -> value, the LAST occurrence winning (argparse)."""
    tokens = _args_tokens(payload_or_tokens) if not isinstance(payload_or_tokens, list) else payload_or_tokens
    eff, i, n = {}, 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.startswith("--") and len(tok) > 2:
            if "=" in tok:
                flag, val = tok.split("=", 1)
                eff[flag.lstrip("-")] = val
                i += 1
                continue
            vals, j = [], i + 1
            while j < n and not (tokens[j].startswith("--") and len(tokens[j]) > 2):
                vals.append(tokens[j])
                j += 1
            eff[tok.lstrip("-")] = " ".join(vals)
            i = j
            continue
        i += 1
    return eff


def args_line_of(path):
    """-> the first `ARGS:` payload of a raw .out (head only: it is line 2 of every run_cifar.sh job), or None."""
    try:
        with open(path, errors="replace") as fh:
            for k, ln in enumerate(fh):
                if k > 200:
                    break
                m = _ARGS_RE.match(ln)
                if m:
                    return m.group(1).strip()
    except IOError:
        return None
    return None


def args_factors(path):
    """-> {flag: effective value} for the registered ARGS flags of one raw .out, or None if it prints no ARGS line."""
    payload = args_line_of(path)
    if payload is None:
        return None
    eff = _args_effective(payload)
    return dict((f, eff[f]) for _k, f, _s in ARGS_KINDS if f in eff)


def _num_equal(a, b):
    """argsline_guard.values_equal, re-typed: string equality with a numeric fallback (0 == 0.0, 1e-4 == 0.0001)."""
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def args_deviations(factors):
    """-> {kind: witness} for every registered flag the run carries at a non-standard value (absent = standard)."""
    out = {}
    for k, f, std in ARGS_KINDS:
        if f in factors and not _num_equal(factors[f], std):
            out[k] = args_witness(k, factors[f])
    return out


def in_standard_cell(row):
    """-> True if this CSV row sits in the mechanism line's standard cell, where ARGS_KINDS' standards are defined."""
    return (row.get("network") in STD_NETWORKS
            and all(row.get(c) == v for c, v in STD_CELL))


def multi_kind_of(fn):
    """-> the registered {kind: line} of a `<batch>-<arm>-...` run name or .out file name, or None."""
    return MULTI_KIND.get(tuple(fn.split("-")[:2]))


def kind_of(witness):
    for k, _off in KINDS:
        if witness.startswith(k + ":"):
            return k
    for k in args_kind_names():  # CORRECTIONS 263: the ARGS-value kinds, read the same way (`<KIND>: ...`)
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
    # CORRECTIONS 263: the same proof over KINDS + ARGS_KINDS, and across the two readers (see ARGS_KINDS)
    allnames = [k for k, _o in KINDS] + args_kind_names()
    acollide = [(ka, kb) for ka in allnames for kb in allnames if ka != kb and ka.startswith(kb)]
    for ka, kb in acollide:
        bad.append("kind name %r starts with %r: a %s witness would also be read as %s" % (ka, kb, ka, kb))
    across = ([k for k, _o in KINDS if ARGS_LINE_PREFIX.startswith(k) or k.startswith("ARGS")]
              + [k for k in args_kind_names() if _ARGS_RE.match(args_witness(k, "0"))])
    for k in across:
        bad.append("kind %r collides with the `%s` line prefix: one line would be read by both readers"
                   % (k, ARGS_LINE_PREFIX))
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
                # CORRECTIONS 263: an ARGS-value witness names no printed line; it is verified against the run's
                # own ARGS line in the block below, not in witness_lines.
                if kd is not None and kd not in args_kind_names() and wl[fn][kd] != [listed[fn]["witness"]]:
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
        # CORRECTIONS 251: 245's line names 245's four kinds, byte for byte; the kinds added since are named on the next line
        print("  kinds scanned (CORRECTIONS 245): %s; no prefix is a prefix of another, so no line is read as two kinds: %s"
              % (" / ".join(kd for kd, _off in KINDS[:4]), not collide))
        # CORRECTIONS 269: 251's line is frozen over the SIX kinds it registered (KINDS_AT_251), so it stays byte-identical
        # as kinds are added; the kinds added since are named on the next line.  `collide` is computed over ALL of KINDS,
        # so both verdicts are the stronger statement.
        print("  kinds scanned (CORRECTIONS 251): also %s, %d in all; no prefix of the %d is a prefix of another: %s"
              % (" / ".join(kd for kd, _off in KINDS[4:KINDS_AT_251]), KINDS_AT_251, KINDS_AT_251, not collide))
        print("  kinds scanned (CORRECTIONS 269): also %s, %d in all; no prefix of the %d is a prefix of another: %s"
              % (" / ".join(kd for kd, _off in KINDS[KINDS_AT_251:]), len(KINDS), len(KINDS), not collide))
        # TWO-KIND RUNS (CORRECTIONS 245): a listed run of a MULTI_KIND (batch, arm) prints exactly its registered line of
        # every kind registered there -- the kind its witness names and the one the TSV row cannot carry.
        nb = len(bad)
        n_two = 0
        n_by = {}  # CORRECTIONS 251: the same runs, by the number of kinds registered for them
        for fn in sorted(listed):
            des = multi_kind_of(fn)
            if des is None or fn not in wl:
                continue
            # CORRECTIONS 284: a one-kind (two-axis) entry is VERIFIED here like any other, but it is not counted on
            # 245's / 251's lines -- those two stay about the 2+-kind runs, and it is counted on the 284 line below.
            if len(des) >= MULTI_KINDS_AT_251:
                n_two += 1
                n_by[len(des)] = n_by.get(len(des), 0) + 1
            for kd, _off in KINDS:
                if kd in des and wl[fn][kd] != [des[kd]]:
                    bad.append("%s is registered with %s ON but prints %r, not the registered line (MULTI_KIND)"
                               % (fn, kd, wl[fn][kd]))
        print("  two-kind runs (CORRECTIONS 245): %d listed runs of a registered two-kind (batch, arm), every one printing"
              " exactly its registered ON line of each kind: %s" % (n_two, len(bad) == nb))
        # CORRECTIONS 284: 251's line is frozen over the entries registering MULTI_KINDS_AT_251 or more kinds, so it
        # stays byte-identical as one-kind two-axis entries are added; those are counted on their own line below.
        print("  multi-kind runs (CORRECTIONS 251): those runs by the number of kinds MULTI_KIND registers for them: %s"
              % ", ".join("%d kinds %d" % (n, n_by.get(n, 0)) for n in sorted(set(len(d) for d in MULTI_KIND.values()
                                                                                  if len(d) >= MULTI_KINDS_AT_251))))
        # ---- TWO-AXIS RUNS (CORRECTIONS 284) -----------------------------------------------------------------------
        # A listed run that deviates on an ARGS value AND turns a registered kind ON (cwd5's CARW2).  It is listed with
        # its ARGS witness -- the only listing both readers accept -- and its ON kinds are registered in MULTI_KIND,
        # where the loop above holds each to its registered line.  Both halves are re-checked here, independently of
        # the two readers, so this line's verdict is a statement and not a label.
        nb = len(bad)
        n_axis = 0
        for fn in sorted(listed):
            des = multi_kind_of(fn)
            ekd = kind_of(listed[fn]["witness"])
            if des is None or fn not in wl or ekd not in args_kind_names():
                continue
            n_axis += 1
            dev = args_deviations(args_factors(outs[fn]) or {})
            if dev.get(ekd) != listed[fn]["witness"]:
                bad.append("%s is a two-axis run listed with %r but its own ARGS line gives %r"
                           % (fn, listed[fn]["witness"], dev.get(ekd)))
            for kd in sorted(des):
                if wl[fn][kd] != [des[kd]]:
                    bad.append("%s is a two-axis run registered with %s ON but prints %r (MULTI_KIND)"
                               % (fn, kd, wl[fn][kd]))
        print("  two-axis runs (CORRECTIONS 284): %d listed runs deviate on an ARGS value AND print an ON line of a"
              " registered kind; each carries its ARGS witness (the only listing both readers accept) and its ON kinds"
              " are registered in MULTI_KIND, both verified here: %s" % (n_axis, len(bad) == nb))
        # ---- ARGS-VALUE WITNESSES (CORRECTIONS 263) ----------------------------------------------------------------
        # A listed ARGS row carries its run's own value; every INGESTED standard-cell run whose ARGS deviates is
        # listed; a batch with ARGS listings holds its other runs to the standard values; and no 15-key cell pools
        # unlisted rows with different values.  Nothing here reads a `<KIND>:` line, and nothing above reads an
        # `ARGS:` line (the prefix proof).
        nb = len(bad)
        rowby = dict(((r["run"], r["job_id"]), r) for r in rows)
        abatches = set(e["batch"] for e in ents if kind_of(e["witness"]) in args_kind_names())
        n_args = n_noargs = n_dev = n_dev_std = n_dev_out = n_dev_new = n_listed_args = 0
        cellvals = {}
        for fn, p in sorted(outs.items()):
            fac = args_factors(p)
            if fac is None:
                n_noargs += 1
                continue
            n_args += 1
            dev = args_deviations(fac)
            run, _sep, jid = fn[:-len(".out")].rpartition("-")
            ent = listed.get(fn)
            ekd = kind_of(ent["witness"]) if ent else None
            if ekd in args_kind_names():
                n_listed_args += 1
                if ekd not in dev:
                    bad.append("%s is listed with an %s witness but its own ARGS line does not deviate (%s)"
                               % (fn, ekd, ", ".join("%s=%s" % kv for kv in sorted(fac.items())) or "flag absent"))
                elif dev[ekd] != ent["witness"]:
                    bad.append("%s ARGS witness %r != listed %r" % (fn, dev[ekd], ent["witness"]))
            if dev:
                n_dev += 1
                row = rowby.get((run, jid))
                for kd in sorted(dev):
                    if kd == ekd:
                        continue
                    if ent is not None:
                        # CORRECTIONS 294: a registered two-ARGS (batch, arm) listed by an ARGS witness -- the other
                        # kind is held to MULTI_ARGS in the multi-ARGS block below, not skipped
                        if ekd in args_kind_names() and kd in (multi_args_of(fn) or {}):
                            continue
                        bad.append("%s deviates on %s but is listed with a %s witness" % (fn, kd, ekd))
                    elif row is not None and in_standard_cell(row):
                        bad.append("%s is a CSV row in the standard cell whose ARGS deviates (%s) but is NOT listed"
                                   % (fn, dev[kd]))
                    elif fn.split("-")[0] in abatches:
                        bad.append("%s is an unlisted run of a listed batch whose ARGS deviates (%s)" % (fn, dev[kd]))
                if ent is None:
                    if row is None:
                        n_dev_new += 1
                    elif in_standard_cell(row):
                        n_dev_std += 1
                    else:
                        n_dev_out += 1
                elif ekd in args_kind_names() and row is not None and in_standard_cell(row):
                    n_dev_std += 1
            if ent is None and (run, jid) in rowby:  # the cell-mixing rule reads UNLISTED ingested rows only
                ck = tuple(rowby[(run, jid)].get(c, "") for c in CELLKEYS)
                cellvals.setdefault(ck, {}).setdefault(
                    tuple((f, fac.get(f, "(absent)")) for _k, f, _s in ARGS_KINDS), []).append(fn)
        args_complete = len(bad) == nb  # before the cell-mixing rule, which prints its own verdict
        mixed = [(ck, v) for ck, v in cellvals.items() if len(v) > 1]
        for ck, v in sorted(mixed):
            bad.append("cell %s pools unlisted rows with different base-optimiser values: %s"
                       % ("/".join(x for x in ck if x),
                          "; ".join("%s -> %s" % (", ".join("%s=%s" % kv for kv in vals), ", ".join(sorted(f)[:3]))
                                    for vals, f in sorted(v.items()))))
        print("  ARGS-value kinds (CORRECTIONS 263): %s; the witness is the run's own `%s` line (a distinct prefix"
              " present on every run, read with argparse last-wins semantics); no kind name is a prefix of another"
              " and neither reader can take the other's line: %s"
              % (" / ".join("%s (--%s, standard %s)" % (k, f, s) for k, f, s in ARGS_KINDS),
                 ARGS_LINE_PREFIX, not (acollide or across)))
        print("  ARGS witness (CORRECTIONS 263): %d listed runs carry their listed ARGS value; %d .out files carry an"
              " ARGS line (%d without one); %d deviate from the standard, %d are CSV rows in the standard cell, every"
              " one listed: %s; %d deviating CSV rows outside the standard cell (DESCRIPTIVE, not listed); %d not in"
              " the CSV (not ingested, not required)"
              % (n_listed_args, n_args, n_noargs, n_dev, n_dev_std, args_complete, n_dev_out, n_dev_new))
        print("  ARGS cell mixing (CORRECTIONS 263): %d cells hold an unlisted ingested run; none pools two different"
              " (%s) value sets: %s"
              % (len(cellvals), ", ".join(f for _k, f, _s in ARGS_KINDS), not mixed))
        # ---- TWO-ARGS RUNS (CORRECTIONS 294) -----------------------------------------------------------------------
        # A listed run of a MULTI_ARGS (batch, arm): its witness names one of the registered kinds, it deviates on
        # EXACTLY the registered kinds, and each registered witness IS its own ARGS line's value.  Re-checked here
        # independently of the reader above, so the line's verdict is a statement and not a label.
        nb = len(bad)
        n_ma = 0
        for fn in sorted(listed):
            reg = multi_args_of(fn)
            if reg is None or fn not in outs:
                continue
            n_ma += 1
            ekd = kind_of(listed[fn]["witness"])
            dev = args_deviations(args_factors(outs[fn]) or {})
            if ekd not in reg:
                bad.append("%s is a two-ARGS run listed with a %s witness, not one of its registered ARGS kinds (%s)"
                           % (fn, ekd, " / ".join(sorted(reg))))
            if sorted(dev) != sorted(reg):
                bad.append("%s is registered in MULTI_ARGS with %s but its own ARGS line deviates on %s (MULTI_ARGS)"
                           % (fn, " / ".join(sorted(reg)), " / ".join(sorted(dev)) or "nothing"))
            for kd in sorted(reg):
                if dev.get(kd) != reg[kd]:
                    bad.append("%s is registered with %r but its own ARGS line gives %r (MULTI_ARGS)"
                               % (fn, reg[kd], dev.get(kd)))
        print("  multi-ARGS runs (CORRECTIONS 294): %d listed runs deviate on two or more ARGS kinds at once; each is"
              " listed ONCE by one of them and MULTI_ARGS registers ALL of them, every registered value equal to the"
              " run's own ARGS line: %s" % (n_ma, len(bad) == nb))
    # CORRECTIONS 294: MULTI_ARGS is well-formed -- two or more kinds per entry, each an `args_witness` of its own kind.
    # Checked with or without --runs; it prints nothing and only FAILs.
    aflag = dict((k, f) for k, f, _s in ARGS_KINDS)
    for (mb, ma), reg in sorted(MULTI_ARGS.items()):
        if len(reg) < 2:
            bad.append("MULTI_ARGS entry %s/%s registers %d ARGS kind(s); an entry is for runs deviating on TWO or more"
                       % (mb, ma, len(reg)))
        for kd, w in sorted(reg.items()):
            if kd not in aflag or kind_of(w) != kd or not w.startswith("%s: %s=" % (kd, aflag[kd])) \
                    or w == "%s: %s=" % (kd, aflag[kd]):
                bad.append("MULTI_ARGS entry %s/%s: %r is not a well-formed %s witness" % (mb, ma, w, kd))
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
