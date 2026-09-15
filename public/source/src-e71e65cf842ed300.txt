# The parent paper's exact configuration — extracted from the PDF

Source: MetaOptimize (Sharifnassab, Salehkaleybar, Sutton), arXiv:2402.02342v6, §7.1 and
Appendix Table 2. Extracted directly from the paper, so nothing here depends on recollection.

## CIFAR-10 (§7.1)

| item | value |
|---|---|
| model / data | ResNet-18, CIFAR-10, **batch size 100** |
| (base, meta) combinations tested | **(AdamW, Adam), (Lion, Lion), (RMSProp, Adam), (SGDm, Adam)** |
| meta step size η | **1e-3** for every MetaOptimize row |
| initial step size α₀ | **1e-6** for MetaOptimize rows; **1e-5** for the fixed-step-size AdamW baseline |
| discount γ | **1** |
| AdamW base | ρ = 0.9, λ = 0.999, κ = 0.1 |
| meta momentum c̄ | 0.9 |
| **blockwise partition** | **six blocks — "one for each linear layer and four blocks for the ResNet modules"** |
| **data augmentation** | **NOT MENTIONED ANYWHERE IN THE PAPER.** `augment`, `random crop`, `flip` return zero hits across the full text. The released code has none either. |
| seeds | **NOT STATED FOR §7.1.** See the correction below — the "averaged over 5 random seeds" phrase belongs to §7.2, not here. |
| error bars | **NONE.** `error bar`, `shaded`, `standard deviation`, `confidence interval` return **0** hits across the full text. |

## CORRECTION (cycle 88, CORRECTIONS 119.1) — the seed row was misattributed

The row above previously read *seeds | curves "averaged over 5 random seeds"* under the
**CIFAR-10 (§7.1)** table. **That is wrong, and it was wrong in the direction that flatters the
result we wanted to build on.**

`grep -niE "random seeds|averaged over"` over the full text (arXiv:2402.02342v6,
`~/.arxiv-mcp-server/papers/2402.02342.md`, 22,872 lines) returns exactly **two** hits, line
4655 and line 22813, and **both belong to §7.2, the non-stationary CIFAR-100 experiment**:

> line 4655: "We evaluated MetaOptimize in a non-stationary setting with 10 sequential tasks …
> Each curve is averaged over 5 random seeds."

**§7.1 states no seed count, no error bars, and no standard deviations.** So the CIFAR-10
learning curves — the sole basis for the blockwise > scalar inference — are of **unstated
replication, possibly single-seed**. This is a legitimate and generous explanation for why the
parent's granularity result was inconsistent, and it must be said in the paper rather than
quietly relied upon.

Also note **§7.2's blockwise is m = 2**, not 6 — line 4677, *"two blocks: one for the first
three layers and one for the last layer"*. The paper therefore runs **exactly two granularities
anywhere**: scalar, and a "blockwise" whose group count differs between experiments.

## CORRECTION (cycle 88, CORRECTIONS 119.2) — the §9 sentence, quoted IN FULL

Every previous internal quotation of the §9 Limitations sentence **elided a load-bearing
clause**. Verbatim, lines 5308-5311:

> **Blockwise step-sizes:** While step sizes can vary much in granularity, our experiments
> focused on scalar and blockwise step-sizes. While increasing the number of step sizes is
> anticipated to enhance performance, our experimental findings in Section 7 reveal that this
> improvement is not consistent **across the MetaOptimize approximations evaluated**. Further
> investigation is needed in future research.

The clause **"across the MetaOptimize approximations evaluated"** indexes the inconsistency by
**approximation / (base, meta) instantiation**, *not* by an accuracy-vs-m curve. The parent is
**not** reporting that the granularity curve turns down in the middle; it never measured a curve.

**CONSEQUENCE, BINDING.** Any sentence of the form *"we explain the parent's reported
non-monotonicity of the granularity curve"* misquotes a paper that does not exist and is
refutable by any referee who opens §7.1. And any answer to the §9 sentence **as written** must
vary the approximation axis — which our corpus does not: all **112** `chunk*`/`nodewise1d` rows
are `base=SGDm, meta=Lion`, at `meta_stepsize` ∈ {1e-4, 3e-4} only.

Finally, **"variance" appears 0 times in the parent paper** (`grep -c -i variance` = 0). That is
confirmed, but it is a statement about their vocabulary, not a licence for ours.

## The claim we are testing

§7.3, ImageNet: *"**Unlike CIFAR10**, here the blockwise versions of MetaOptimize showed no
improvement over the scalar versions."*

So the paper asserts blockwise **does** improve over scalar on CIFAR-10 and **does not** on
ImageNet. §7.1 itself only claims the weaker *"In every tested combination, MetaOptimize
outperforms its corresponding fixed-step-size baseline"* — i.e. MetaOptimize > fixed LR, not
blockwise > scalar. The blockwise>scalar claim on CIFAR-10 rests on the §7.3 aside and Fig. 1.

## Sensitivity (§7.5)

* η: *"there is generally no need for tuning, and the default value η = 1e-3 works universally
  well… All experiments in this section used this default value with no sweeping required."*
* γ: *"γ for values γ ≥ 0.999 … performance begins to degrade with smaller values of γ."*
  So **γ = 0.999 is inside the paper's own sanctioned range.**
* No clipping or bounding of β or the step sizes is mentioned anywhere.

## Deviations in our runs, now identified

1. **Meta-optimizer.** Our Gate 1 used meta = **Lion** for the SGDm arm; the paper's SGDm arm
   is **(SGDm, Adam)**. Gate 3 re-runs the SGDm arm with Adam meta to match.
2. **Augmentation.** We enable RandomCrop+Flip; the paper appears not to. Ours is the
   scientifically necessary choice (without it the task is memorised in epoch 1 and there is no
   optimisation headroom), but it means our absolute numbers are not directly comparable to
   the paper's, and any reproduction claim must say so.
3. **γ = 1 has a consequence the paper does not discuss.** With γ = 1 and α tiny, the trace
   update `h ← γ(1 − wd·α)h − Δw` has effectively **no decay term**, so `h` is an unbounded
   running sum. See `docs/FINDINGS.md` — the weightwise failure is a float32 overflow of this
   trace, not a step-size collapse. γ = 0.999 gives the trace a ~693-step half-life and should
   bound it; that experiment is running.
