> **SUPERSEDED IN PART — DO NOT SUBMIT §5.**  Sections **§5, §6 and §7** of this document
> are **WITHDRAWN**, before any `sf1` run existed, by
> [`docs/REGISTER-c82b-singleton-law-WITHDRAWAL.md`](REGISTER-c82b-singleton-law-WITHDRAWAL.md)
> and CORRECTIONS 111.  Three verified defects: the ladder is frozen in the (ms=3e-4,
> BETA_CLIP=-15:-2.3026) cell that its own R4 voids (rec_lo 0.4521-0.4597 on 12/12 ar1 arms);
> R5's "deep-first at the same f" is not at the same f (k=21 deep-first gives f=0.614, not
> 0.311) and self-fires; and f is collinear with m and with the comparator's resolution, the
> same confound-by-construction §3 uses to disqualify the count form.  **§0-§4 and §3's
> DO-NOT-RUN verdict on the cross-architecture test survive unchanged and were re-verified.**
> The body below is left BYTE-IDENTICAL on purpose (CORRECTIONS 110.4).

# REGISTRATION — cycle 82: the singleton-fraction law (`sf1`)

Registered **before any run of this batch exists** (STANDING RULE 19). Scorer, bands and
refutation conditions below are frozen at this commit. Instrument:
`analysis/c82_singleton_census.py` (validated) + `analysis/c82_law_registration.py`.

## 0. Instrument, verified against HF.py (read-only, cluster)

`HF.py:72` — `net_param_names_and_size = [(name, p.data.size()) for name,p in net.named_parameters()]`
`HF.py:286-290` — `nodewise`: `beta_i = ones(p_size[0])`  →  m = Σ shape[0]
`HF.py:299-311` — `nodewise1d`: `beta_i = ones(1 if ndim==1 else p_size[0])`
`HF.py:341-348` — `chunk<K>`: `beta_i = ones(ceil(numel/K))`  →  m = Σ ceil(numel_i / K)
A nodewise group is a SINGLETON iff `numel//shape[0] == 1` iff the tensor is 1-D.

Cross-check ResNet18 vs FINDINGS 77.6 / cycle 81 — **7/7 exact**:
params 11,173,962 · m 14,420 · singletons 9,610 · 66.64% · 0.09% of weights ·
nodewise1d m 4,851 · 41 one-dimensional tensors. Also independently reproduces
ResNet34 params 21,282,122, ResNet10 m 8,660, ResNet18_c100 m 14,600 (FINDINGS §…).

## 1. Census (7 architectures build_network.py can actually build)

| arch | params | nodewise m | singletons | f | weight cov | nodewise1d m |
|---|---|---|---|---|---|---|
| ResNet10      |  4,903,242 |  8,660 |  5,770 | 66.628% | 0.1177% |  2,915 |
| ResNet18      | 11,173,962 | 14,420 |  9,610 | 66.644% | 0.0860% |  4,851 |
| ResNet34      | 21,282,122 | 25,556 | 17,034 | 66.654% | 0.0800% |  8,595 |
| ResNet50      | 23,520,842 | 79,700 | 53,130 | 66.662% | 0.2259% | 26,677 |
| ResNet10_c100 |  4,949,412 |  8,840 |  5,860 | 66.290% | 0.1184% |  3,005 |
| ResNet18_c100 | 11,220,132 | 14,600 |  9,700 | 66.438% | 0.0865% |  4,941 |
| ResNet34_c100 | 21,328,292 | 25,736 | 17,124 | 66.537% | 0.0803% |  8,685 |

`ResNet50_c100` is NOT in build_network.py.

## 2. WHY f IS PINNED — structural identity, not coincidence

Every conv in these ResNets is `bias=False` and is followed by exactly one
`BatchNorm2d(affine=True)`, i.e. **b = 2 one-dimensional tensors per 2-D+ tensor**, each of
length = that conv's `shape[0]`. Therefore with C = Σ conv out-channels and nc = num_classes:

    singletons = 2C + nc      non-singletons = C + nc      m = 3C + 2nc
    f = (2C + nc) / (3C + 2nc)  ->  b/(1+b) = 2/3   as C/nc -> inf

Verified exactly for all 8 rows. The only deviation from 2/3 is the classifier head, a b=1
site embedded in a b=2 network. **f is a property of the conv->norm IDIOM, not of depth,
width, block type, or dataset.**

## 3. REGISTERED VERDICT ON THE CROSS-ARCHITECTURE TEST: **UNTESTABLE — DO NOT RUN**

f spread across all 7: 66.290% .. 66.662%, max/min = **1.0056**.
Gap predicted by the fraction form, anchored at ResNet18 = +0.697 pp:
0.6933 .. 0.6972 pp — **total spread 0.0039 pp**, against a seed-noise floor of 0.12 pp
(median) / 0.24 pp (p90). Ratio 0.033. Seeds per arm needed to resolve at t=2: **~11,000**.

Alternative carriers, and why they do not rescue it:
* **singleton COUNT** spreads 9.21x — but `r(count, m) = 0.9999996` by the identity above.
  A count-form law is **confounded-by-construction** with network size; the batch could not
  distinguish "gap tracks singleton count" from "gap tracks m" or "gap tracks depth".
* **weight coverage** spreads 2.82x (R34 0.0800% vs R50 0.2259% at 1.105x params). This is
  the ONE non-degenerate cross-architecture axis — registered as SECONDARY in §6.
* the **resolution penalty** on the real weights (nodewise mean group size on >=2-D tensors
  vs the matched chunk K) is 2.99x for EVERY architecture — also pinned by f = 2/3.

## 4. THE MECHANISM (what a per-scalar step size on a BN scale does)

`HF.py:180` — the nodewise meta-gradient for group g is `(u*v).reshape(shape[0],-1).sum(dim=1)`,
i.e. Σ_{i in g} u_i v_i. For |g| = 1 that is a **single unaveraged product**: the worst-SNR
estimator in the partition. nodewise therefore spends **2/3 of its beta budget** on
one-term estimators governing BN scales and shifts, while leaving the remaining 1/3 of the
budget to cover 99.91% of the weights at **3x coarser** resolution than the matched chunk
comparator. A BN scale gamma_c whose log-step-size random-walks multiplies an entire feature
map — which is why the harm is out of all proportion to 0.09% of the weights.

Both harms are proportional to **f**. That is precisely why the law is a *within-network*
law and cannot be tested by swapping ResNets.

## 5. PRIMARY REGISTERED TEST — the within-ResNet18 f-ladder (`sf1`)

New HF.py granularity required (NOT yet written; cluster file untouched): `nodewisep<k>` =
nodewise on the first k of the 41 one-dimensional tensors, ONE group on the remaining 41-k.
k=0 is exactly `nodewise1d`; k=41 is exactly `nodewise`. Model, dataset, seeds, ms, alpha0,
BETA_CLIP unchanged from `ar1`; **only the beta partition moves**.

Comparator at each rung = `chunk<K>` with K count-matched (all within 0.02%):

| rung | k | m | singletons | f | matched K | count@K | status |
|---|---|---|---|---|---|---|---|
| L0 |  0 |  4,851 |     0 | 0.00000 | 2325 |  4,851 | **already run (ar1 A2)** |
| L1 |  7 |  5,292 |   448 | 0.08466 | 2130 |  5,292 | to run |
| L2 | 14 |  5,989 | 1,152 | 0.19235 | 1879 |  5,990 | to run |
| L3 | 21 |  7,006 | 2,176 | 0.31059 | 1604 |  7,007 | to run |
| L4 | 28 |  8,791 | 3,968 | 0.45137 | 1277 |  8,791 | to run |
| L5 | 34 | 11,345 | 6,528 | 0.57541 |  989 | 11,343 | to run |
| L6 | 41 | 14,420 | 9,610 | 0.66644 |  777 | 14,421 | **already run (ar1 A1)** |

Both endpoints reproduce `ar1`'s own comparators (`chunk2325`, `chunk777`) exactly, so the
ladder is anchored on existing data and only L1..L5 need submitting:
5 rungs x 2 arms x 3 seeds = **30 jobs**, plus a 6-job deep-first-ordering control at L3.

Config frozen: ResNet18 / CIFAR-10, SGDm + Lion meta, ms = 3e-4, alpha0 = 1e-3,
BETA_CLIP = -15 : -2.3026, 100 epochs, 3 seeds/arm. **plateau5 is PRIMARY** (mean of last 5
test epochs). Statistic per rung: `gap(f) = plateau5(chunk_matched) - plateau5(nodewisep_k)`,
Welch se on n=3 v 3.

### Registered prediction — PROPORTIONAL FORM
Two-point anchor from `ar1`: gap(0.66644) = +0.697, gap(0) = -0.139.

    gap(f) = -0.139 + 1.2544 * f      [pp; slope = 0.836 / 0.66644]

| rung | f | predicted gap (pp) |
|---|---|---|
| L1 | 0.08466 | **-0.033** |
| L2 | 0.19235 | **+0.102** |
| L3 | 0.31059 | **+0.251** |
| L4 | 0.45137 | **+0.427** |
| L5 | 0.57541 | **+0.583** |

Predicted sign flip between L1 and L2 (zero crossing at f = 0.1108).

### REFUTATION CONDITIONS — frozen, not reinterpretable
* **R1 (the law dies).** Fit `gap = a + b*f` on the **five interior rungs only** (L1..L5), so
  both anchors are out-of-sample. If `b_hat <= 0`, OR `t(b_hat) < 2`, the singleton-fraction
  law is **REFUTED**: the gap does not track how much degenerate tail the partition has.
* **R2 (ramp vs step).** The live competitor is a THRESHOLD: nearly all of +0.697 appears as
  soon as any substantial singleton mass exists. Discriminant at L3 (f = 0.311): proportional
  predicts +0.251, step predicts >= +0.50. If `gap(L3) >= +0.45 pp` and it exceeds the
  proportional prediction with t >= 2, the **PROPORTIONAL form is REFUTED** and the surviving
  statement is a step law (still mechanistic, but a different claim, and it must be relabelled).
* **R3 (interior non-monotone).** If the interior rungs are non-monotone in f by more than
  0.24 pp (seed p90) at any adjacent pair, the ladder does not define a law and the result is
  **NULL**, not a refutation of either form.
* **R4 (void, not refuted).** Any arm at a beta guard on > 5% of records is UNINTERPRETABLE
  (rule 5) and that rung is dropped. If >= 3 of the 7 rungs void, the whole test is **VOID**.
* **R5 (ordering confound).** L3 is run twice — shallow-first and deep-first selection of the
  k singleton tensors at the same f. If the two L3 gaps differ by > 0.24 pp, f is NOT the
  carrier; WHICH tensors are singleton is, and the law as stated is **REFUTED**.

## 6. SECONDARY registered test — ResNet34 vs ResNet50 (weight-coverage form) — WEAK
Near-matched params (21.28M vs 23.52M, 1.105x) but singleton weight coverage 0.0800% vs
0.2259% (2.82x) at identical f. Fraction form predicts 0.697 for both; weight-coverage form
predicts **0.649 (R34) vs 1.831 (R50)**, a 1.18 pp separation ~10x the noise floor.
**Registered confound, stated in advance:** R50 is Bottleneck and R34 is BasicBlock, and they
differ in m and in size-multiset, so a positive result here is NOT clean evidence for
w-coverage — it is only evidence that *something* other than f moves the gap. `pp1`'s P2
(alignment null, -0.009 pp at fixed count AND size-multiset) defuses the block-type confound
partially, not fully. Run only if `sf1` returns R1-refuted.

## 7. What WOULD move f, if the ladder is judged too invasive
f = b/(1+b) where b = one-dimensional tensors per 2-D+ tensor. The available levers all
change the NORMALISATION IDIOM, not the depth/width/dataset:
* b=0 — BN `affine=False` (or norm-free): f -> 0.207% (only `linear.bias` survives)
* b=1 — conv `bias=True` + BN `affine=False`: f = 50%. **Already in build_network.py** as the
  MNIST MLPs M1/M2 (one bias per weight matrix): M1 m=120, sing=60, f=50.000%;
  M2 m=1,300, sing=650, f=50.000% — but that is a different task, so the pp gaps are not
  commensurable with the ResNet ones.
* b=2 — current ResNets: f = 66.6%
* b=3 — conv `bias=True` AND BN `affine=True`: f = 75%
A 4-rung {~0%, 50%, 66.6%, 75%} series on ResNet18/CIFAR-10 at fixed depth and width is the
fallback design. It is dirtier than §5 (the model itself changes at each rung) but the gap is
a within-model contrast at every rung, so the model change is absorbed.
