#!/usr/bin/env python3
"""csv1 DESCRIPTIVE readouts, all from the RAW .out files and the batch's own probe records.
DESCRIPTIVE / UNSURE -- nothing here is registered, nothing moves a gate, bar, stamp or FINAL."""
import json, math, os, sys

RUNSDIR = sys.argv[1]
ARMS = ("k01", "INERT", "SHADOWLOW", "NAIVELOW", "MUTE", "HEAD")
SEEDS_OF = {"k01": (136, 137, 138), "INERT": (136,), "SHADOWLOW": (136, 137, 138, 139),
            "NAIVELOW": (136, 137, 138, 139), "MUTE": (136, 137, 138), "HEAD": (136, 137, 138)}
SPE = 500   # steps per epoch
PROBE = 100


def mean(v):
    v = list(v)
    return math.fsum(v) / len(v)


def read_out(arm, seed):
    for fn in sorted(os.listdir(RUNSDIR)):
        if fn.startswith("csv1-%s-s%d-" % (arm, seed)) and fn.endswith(".out"):
            eps = {}
            for ln in open(os.path.join(RUNSDIR, fn), errors="replace"):
                if not ln.startswith("Epoch "):
                    continue
                head, rest = ln.split(",", 1)
                n = int(head[6:])
                tr = float(rest.split("Train Accuracy:")[1].split(",")[0].strip().rstrip("%").strip())
                te = float(rest.split("Test Accuracy:")[1].split(",")[0].strip().rstrip("%").strip())
                eps[n] = (tr, te)
            return eps
    return None


def read_probe(arm, seed):
    p = os.path.join(RUNSDIR, "csv1", "probe_csv1-%s-s%d" % (arm, seed), "probe.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()]


EPS = dict(((a, s), read_out(a, s)) for a in ARMS for s in SEEDS_OF[a])

print("=" * 100)
print("A. per-arm TEST (TRAIN) AFTER N EPOCHS.  'after N epochs' = the `Epoch N-1` line of the run's own .out")
print("   (the harness prints Epoch 0..99 for a 100-epoch run), arm mean over the arm's seeds, 2 dp.")
NS = (15, 17, 19, 30, 50, 100)
print("   %-10s %s" % ("arm", "  ".join("after %3d ep" % n for n in NS)))
for a in ARMS:
    cells = []
    for n in NS:
        e = n - 1
        te = mean(EPS[(a, s)][e][1] for s in SEEDS_OF[a])
        tr = mean(EPS[(a, s)][e][0] for s in SEEDS_OF[a])
        cells.append("%5.2f (%5.2f)" % (te, tr))
    print("   %-10s %s" % (a, "  ".join(cells)))
print()
print("   per seed, TEST only:")
for a in ARMS:
    for s in SEEDS_OF[a]:
        print("     %-10s s%d  %s" % (a, s, "  ".join("%6.2f" % EPS[(a, s)][n - 1][1] for n in NS)))

print()
print("=" * 100)
print("B. idx 50's APPLIED beta and the SHARED beta, at epochs 10 / 17 / 20 / 30 / 40.")
print("   record index = epoch * %d / %d.  For the scalar arms the shared beta is beta[0]; idx 50's APPLIED beta is" % (SPE, PROBE))
print("   ln(sv_a_applied) on the three SHADOW_VOTE arms and beta[0] itself on k01 / MUTE (no separation exists there).")
print("   For HEAD the grouping is blockwise [complement, idx50], so shared = beta[0] and idx 50's applied = beta[1].")
EPOCHS_B = (10, 17, 20, 30, 40)
print("   %-10s %-5s %s" % ("arm", "seed", "  ".join("ep%-3d shared / applied" % e for e in EPOCHS_B)))
for a in ARMS:
    for s in SEEDS_OF[a]:
        recs = read_probe(a, s)
        cells = []
        for e in EPOCHS_B:
            k = e * SPE // PROBE
            r = recs[k]
            assert r["step"] == e * SPE, (a, s, e, r["step"])
            shared = float(r["beta"][0])
            if a == "HEAD":
                applied = float(r["beta"][1])
            elif "sv_a_applied" in r:
                applied = math.log(float(r["sv_a_applied"][0]))
            else:
                applied = shared
            cells.append("%8.4f / %8.4f" % (shared, applied))
        print("   %-10s %-5d %s" % (a, s, "  ".join(cells)))

print()
print("=" * 100)
print("C. the first epoch line from which TEST stays > 2 pp above k01, for SHADOWLOW and NAIVELOW.")
print("   'stays' = true for that line and every later line to 99.  Two references are reported:")
print("     (i)  per seed vs the k01 ARM MEAN curve; (ii) per seed vs the SAME-SEED k01 run (s139 has no k01 twin);")
print("     (iii) arm mean vs arm mean (this is the scorer's own descriptive line).")
k01_mean = dict((e, mean(EPS[("k01", s)][e][1] for s in (136, 137, 138))) for e in range(100))


def onset(cx, cy):
    for e in range(100):
        if all(cx[f] - cy[f] > 2.0 for f in range(e, 100)):
            return e
    return None


for a in ("SHADOWLOW", "NAIVELOW"):
    for s in SEEDS_OF[a]:
        cx = dict((e, EPS[(a, s)][e][1]) for e in range(100))
        o1 = onset(cx, k01_mean)
        if s in (136, 137, 138):
            cy = dict((e, EPS[("k01", s)][e][1]) for e in range(100))
            o2 = onset(cx, cy)
        else:
            o2 = "n/a (no k01 at s139)"
        print("   %-10s s%d   vs k01 arm mean: %s   vs same-seed k01: %s" % (a, s, o1, o2))
    cm = dict((e, mean(EPS[(a, s)][e][1] for s in SEEDS_OF[a])) for e in range(100))
    print("   %-10s ARM MEAN vs k01 ARM MEAN: %s" % (a, onset(cm, k01_mean)))

print()
print("=" * 100)
print("D. does SHADOWLOW's SHARED beta trajectory resemble k01's, or HEAD's complement (group-0) trajectory?")
print("   All three are the group that carries the sum idx 50 votes into (HEAD's group 0 is the 52-tensor complement,")
print("   from which 50 is excluded).  Compared over all 500 probe records: RMS |diff|, max |diff|, Pearson r,")
print("   peak value / peak epoch, and the pin epoch (first record from which beta stays <= -14.99).")


def traj(a, s):
    return [float(r["beta"][0]) for r in read_probe(a, s)]


def stats(x, y):
    n = len(x)
    dx = [p - q for p, q in zip(x, y)]
    rms = math.sqrt(math.fsum(d * d for d in dx) / n)
    mx = max(abs(d) for d in dx)
    mux, muy = mean(x), mean(y)
    num = math.fsum((p - mux) * (q - muy) for p, q in zip(x, y))
    den = math.sqrt(math.fsum((p - mux) ** 2 for p in x) * math.fsum((q - muy) ** 2 for q in y))
    return rms, mx, (num / den if den else float("nan"))


def summarize(x):
    pk = max(range(len(x)), key=lambda i: x[i])
    pin = None
    for i in range(len(x)):
        if all(v <= -14.99 for v in x[i:]):
            pin = i * PROBE / float(SPE)
            break
    return x[pk], pk * PROBE / float(SPE), pin


ref_k01 = [mean(t) for t in zip(*[traj("k01", s) for s in (136, 137, 138)])]
ref_head = [mean(t) for t in zip(*[traj("HEAD", s) for s in (136, 137, 138)])]
ref_naive = [mean(t) for t in zip(*[traj("NAIVELOW", s) for s in SEEDS_OF["NAIVELOW"]])]
sl = [mean(t) for t in zip(*[traj("SHADOWLOW", s) for s in SEEDS_OF["SHADOWLOW"]])]
for nm, x in (("k01 (arm mean)", ref_k01), ("HEAD complement g0 (arm mean)", ref_head),
              ("NAIVELOW (arm mean)", ref_naive), ("SHADOWLOW (arm mean)", sl)):
    p, pe, pin = summarize(x)
    print("   %-32s peak %8.4f @ epoch %5.1f   pin %s" % (nm, p, pe, "never" if pin is None else "%.1f" % pin))
print()
for nm, y in (("k01", ref_k01), ("HEAD complement g0", ref_head), ("NAIVELOW", ref_naive)):
    rms, mx, r = stats(sl, y)
    print("   SHADOWLOW vs %-20s  RMS %7.4f   max |diff| %7.4f   Pearson r %+0.5f" % (nm, rms, mx, r))
print()
print("   late-window comparison (records 200..499, i.e. epochs 40..99):")
for nm, y in (("k01", ref_k01), ("HEAD complement g0", ref_head), ("NAIVELOW", ref_naive)):
    rms, mx, r = stats(sl[200:], y[200:])
    print("   SHADOWLOW vs %-20s  RMS %7.4f   max |diff| %7.4f   Pearson r %+0.5f" % (nm, rms, mx, r))
print()
print("   mean shared beta over epochs 40..99:  SHADOWLOW %8.4f   k01 %8.4f   HEAD-g0 %8.4f   NAIVELOW %8.4f"
      % (mean(sl[200:]), mean(ref_k01[200:]), mean(ref_head[200:]), mean(ref_naive[200:])))
