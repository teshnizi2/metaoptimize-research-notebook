# REGISTRATION — cycle 82b: **`sf1`'s LADDER IS WITHDRAWN BEFORE ANY DATA EXISTS**

Supersedes **§5, §6 and §7** of `docs/REGISTER-c82-singleton-law.md` (frozen at commit
`7c2479f`). That file is left **byte-identical**: renumbering or rewriting a committed
record mid-campaign is worse than the supersession (CORRECTIONS 110.4). This document is
the amendment; it carries the reason, the corrected design, and what may never be claimed.

**No `sf1` run exists. No `sf1` script exists. Nothing was submitted.** A pre-data
withdrawal costs nothing, and STANDING RULE 19 exists precisely so a registration can be
found wrong *before* it buys GPU-hours rather than after.

---

## 0. WHAT SURVIVES, UNCHANGED — §0 through §4 AND THE DO-NOT-RUN VERDICT

Re-verified this tick by re-running `analysis/c82_singleton_census.py` and by rebuilding
ResNet18's `named_parameters()` one-dimensional length list independently:

* the instrument reproduces ResNet18 **7/7 exactly** (params 11,173,962 · nodewise m 14,420
  · singletons 9,610 · 66.64% · 0.0860% of weights · nodewise1d m 4,851 · 41 one-D tensors);
* the 41 one-D lengths are `[64 ×10, 128 ×10, 256 ×10, 512 ×10, 10]`, summing to 9,610, and
  the ≥2-D tensors contribute exactly 4,810 nodewise groups;
* the structural identity `f = (2C+nc)/(3C+2nc) → b/(1+b) = 2/3` holds for every row;
* the cross-architecture spread is 66.290%–66.662%, max/min 1.0056, predicted gap spread
  **0.0039 pp against a 0.12 pp noise floor**.

**§3's verdict — the cross-architecture test is UNTESTABLE, DO NOT RUN — STANDS.** One
addition to the record, which strengthens it: the predicted ordering (R50 > R34 > R18 > R10
> R34_c100 > R18_c100 > R10_c100) is exactly "larger Σ conv-out-channels first, CIFAR-10
before CIFAR-100", because ∂f/∂C > 0 and ∂f/∂nc < 0. Even at infinite n that ordering would
be perfectly confounded with network size and dataset. It may not be resurrected as a cheap
side test.

---

## 1. WHY §5's LADDER IS WITHDRAWN — three defects, each re-derived here, each fatal

### 1.1 IT IS FROZEN IN A CELL ITS OWN R4 VOIDS
§5 freezes `BETA_CLIP = -15 : -2.3026` at `ms = 3e-4`, 100 ep. R4 voids any arm at a guard on
> 5% of records and voids the whole test if ≥ 3 of 7 rungs void. Re-derived from the twelve
`probes_ar1/probe.jsonl` files (10,000 records each): **`rec_lo` = 0.4521–0.4597 on 12/12**,
i.e. **9× over the threshold**, and that includes `chunk2325` and `chunk777`, which contain
*zero* singleton groups — so the bind is a property of the (ms, α₀, T) cell, not of the
partition, and **every sf1 rung and every sf1 comparator would inherit it**. Mechanism:
`HF.py`'s Lion meta update is `β ← (1 − ms·wd)·β − ms·sign(·)` with `wd_meta = 0`, so the
deepest coordinate reaches −15 at step (15 − 6.907755)/3e-4 = **26,974** of 50,000; the
measured first pin is 27,020. **36 jobs, ~30 GPU-hours, guaranteed VOID.**

### 1.2 R5 IS ARITHMETICALLY SELF-REFUTING
R5 says "L3 is run twice — shallow-first and deep-first selection of the k singleton tensors
**at the same f**". The 41 one-D lengths are wildly unequal, so f is a function of *which*
tensors, not of k. Re-computed:

| selection | k | singletons | m | f |
|---|---|---|---|---|
| shallow-first | 21 | 2,176 | 7,006 | **0.31059** (matches §5's frozen table exactly) |
| deep-first | 21 | 7,690 | 12,520 | **0.61422** |

That is 2× the f, not "the same f", and §5's own proportional form then predicts the two L3
arms differ by 1.2544 × (0.61422 − 0.31059) = **+0.381 pp**, against R5's own 0.24 pp
refutation trigger. **R5 fires and declares the law REFUTED precisely when the law is exactly
true.** The comparator is wrong too: K = 1604 is count-matched to m = 7,006, not to 12,520.
No deep-first k reproduces 2,176 singletons (k=5 → 2,058; k=6 → 2,570).

### 1.3 f IS COLLINEAR WITH m AND WITH THE COMPARATOR'S RESOLUTION — the same
### confound-by-construction §3 used to DISQUALIFY the count form
Along §5's ladder, f = 1 − (4,851 − k)/m with k ≤ 41 ≪ m, so f is a deterministic function of
m across the whole 0–0.666 range. Worse, the comparator moves in lockstep: I recomputed
Σ⌈numel/K⌉ over the true shape list and the comparator's ≥2-D group count runs
**4,810 → 5,251 → 5,949 → 6,966 → 8,750 → 11,302 → 14,380** across K = 2325 … 777, while the
`nodewisep` arm's treatment of the 11.16 M real weights is **constant at 4,810 groups at every
rung**. So "the gap tracks f" and "the gap tracks the comparator's resolution advantage on the
weights" make predictions that differ by at most ~0.06 pp — against a measured per-rung se of
0.118–0.200 pp. §3 rejected the count form for exactly this reason and then applied a weaker
standard to its own substitute.

### 1.4 TWO SMALLER DEFECTS, RECORDED SO THE REPLACEMENT DOES NOT INHERIT THEM
* **The proportional form is already falsified at its own f = 1 endpoint.** `HF.py:341-348`
  sets `chunk<K>` β = `ones(⌈numel/K⌉)`; at K = 1 that is one β per weight — the *same*
  partition as `weightwise`, so gap(f=1) ≡ 0 by identity. Measured, at byte-identical config
  (ResNet18/CIFAR-10/100 ep/ms=1e-4/α₀=1e-3/clip −15:−2.3026, granularity the only differing
  field): `ck1-k1` 90.790/91.230/90.918 vs `tw0-w` 91.200/91.142/91.360 →
  **−0.255 pp (se 0.146, t −1.74)**, against the registered form's prediction of **+1.115**.
  §4's "both harms are proportional to f and to nothing else" is therefore wrong: the second
  harm is a property of the COMPARATOR and vanishes at f = 1.
* **§5's prose and §5's table specify different partitions.** "ONE group on the remaining
  41−k" read literally is one group *total*; the table requires one group *per* remaining
  tensor (k=0 → m = 4,810 + 41 = 4,851 = nodewise1d). `HF.py` has no cross-tensor group
  construct in any granularity, so the literal reading is not implementable. **The table is
  right and the sentence is wrong.**

---

## 2. WHAT A FUTURE `sf1` MUST LOOK LIKE — registered now, submittable later

**IT IS NOT SUBMITTABLE TODAY AND NOT BECAUSE OF THIS WITHDRAWAL.** `nodewisep<k>` does not
exist: `Optimizers/HF.py` on the cluster carries `PATCH_GRANULARITY`, `PATCH_NODEBN` and
`PATCH_CHUNKWISE` and **no occurrence of `nodewisep` anywhere**, and no `tests/test_nodewisep.py`
exists. Writing the batch script now could only produce an instrument that refuses.

### 2.1 THE FIXED-m LADDER (replaces §5's ladder)
`nodewisep<k>` on the 41 one-D tensors (nodewise on the first k, one group on **each** of the
remaining 41−k) **and `chunk<K2>` on all ≥2-D tensors**, with K2 chosen per rung so that total
m is held at ≈14,420. Recomputed from the true shape list:

| rung | k | K2 | m | (off target) | singletons | f |
|---|---|---|---|---|---|---|
| L0 |  0 |  777 | 14,421 | +1  |     0 | 0.00000 |
| L1 |  7 |  802 | 14,409 | −11 |   448 | 0.03109 |
| L2 | 14 |  844 | 14,416 | −4  | 1,152 | 0.07991 |
| L3 | 21 |  914 | 14,422 | +2  | 2,176 | 0.15088 |
| L4 | 28 | 1070 | 14,423 | +3  | 3,968 | 0.27512 |
| L5 | 34 | 1417 | 14,428 | +8  | 6,528 | 0.45245 |
| L6 | 41 | 2325 | 14,420 | +0  | 9,610 | 0.66644 |

Total m varies by **0.13%** across the whole ladder while f sweeps 0 → 0.666, and **L0 IS
`chunk777` exactly**, so `ar1`'s own `chunk777` arm is a free endpoint. m is fixed, so the
ladder is its own contrast and **no separate comparator arm is needed** — which also removes
the comparator-collinearity of §1.3 entirely.

### 2.2 THE ORDERING CONTROL (replaces R5)
Match the **SUM of one-D lengths**, not the tensor COUNT. An exact fixed-f control exists:
**4 × length-512 tensors + 2 × length-64** → singletons = 2,176 **exactly** (identical to
shallow-first k=21), k = 6, f = 0.30993 vs 0.31059 in the old parameterisation. Under the
fixed-m ladder the analogous control is built the same way and re-derived at submission time.

### 2.3 THE BOX
`BETA_CLIP = -25 : -2.3026` — the floor released and provably unreachable at ms=3e-4/100 ep
(reachable interval [−21.907755, +8.092245]), the ceiling held at ar1's. **NOT −25:+9.0**:
see CORRECTIONS 111 and `bin/c82_field_wideclip.sh`'s revision note — the corpus's only
released-ceiling runs are 2/2 collapsed. Carry `fa1`'s pre-submit assertions verbatim
(wd_meta == 0, ms, α₀, epochs, `len(trainloader)` == 500, `HIER=none`).

### 2.4 THE REFUTATION CONDITIONS, CORRECTED
* **R1** — OLS `gap = a + b·f` on the interior rungs. `b̂ ≤ 0` or `t(b̂) < 2` → REFUTED,
  **except**: if the OLS residual sd exceeds 0.18 pp the branch is **UNDERPOWERED**, a verdict
  distinct from REFUTED and registered in advance. Simulation at the campaign's own per-rung
  se puts the false-refutation rate at 3.9% (se 0.12) / 12.8% (0.15) / 30.6% (0.20), so the
  branch is not optional.
* **R2** — **ONE threshold, not two.** The old "gap(L3) ≥ +0.45 **and** exceeds the point
  prediction with t ≥ 2" floats with the realised noise (it is really ≈ +0.53). Keep the
  bright line only, or keep the t-test with its se FIXED IN ADVANCE. Not both.
* **R3** — evaluate non-monotonicity on the **fitted residuals**, not on raw adjacent pairs.
  Adjacent predicted gaps are 0.13–0.18 pp apart against a 0.24 pp trigger, so the raw form
  spuriously NULLs at 4.1%/12.4%/30.6%.
* **R4** — unchanged in spirit, but with §2.3's box it should no longer fire.
* **R5** — replaced by §2.2.
* **n = 6 seeds per arm**, not 3. At the measured per-run sd of 0.179 pp, n=3 resolves 0.29 pp
  and n=6 resolves 0.21 pp.
* **DECOMPOSITION IS MANDATORY**: report `plateau5(nodewisep_k)` and the comparator term
  separately at every rung, never only the gap.
* **BOUNDARY CONDITION, REGISTERED**: gap(f=1) ≡ 0 by identity (chunk1 == weightwise). The
  fitted form may not be quoted outside [0, 0.666] and §6/§7's cross-family f extrapolations
  (b=3 → f=75%, the MNIST MLPs at f=50%) are **withdrawn**, because they assume f governs
  across families and the f=1 identity says it does not.

### 2.5 THE ORDER OF WORK
1. land `PATCH_NODEWISEP` in `Optimizers/HF.py`;
2. author `tests/test_nodewisep.py` whose FIRST assertion is that a config equals itself
   (STANDING RULE 20) and whose endpoints assert `nodewisep0 == nodewise1d` at m = 4,851 and
   `nodewisep41 == nodewise` at m = 14,420;
3. only then write `bin/c82b_sf1_fladder.sh` with a guard measuring m per rung from the
   ALLOCATED beta against §2.1.

Steps 1 and 2 modify cluster code and are outside the read-only remit under which this
document was written.

---

## 3. WHAT MAY AND MAY NOT BE CLAIMED, TODAY

**MAY**: the census (§1 of the frozen doc), the structural identity f = b/(1+b), the
DO-NOT-RUN verdict on the cross-architecture test, and the observation that "one step size per
output channel" gives every BatchNorm scale and shift its own step size on ResNet18 (9,610
size-1 groups, 66.64% of the partition, 0.086% of the weights).

**MAY NOT**: `gap(f) = −0.139 + 1.2544·f` in any form; any predicted rung value; any statement
that the partition gap "scales with" the singleton fraction; §6's ResNet34-vs-ResNet50
weight-coverage test; §7's b=0/1/3 extrapolations. Those are withdrawn until §2 is run.
Nothing here withdraws `ar1`'s A1 (+0.697) or A2 (−0.139) as **measurements** — only their use
as a two-point anchor for a law.
