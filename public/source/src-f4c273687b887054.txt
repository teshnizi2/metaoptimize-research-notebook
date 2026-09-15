#!/usr/bin/env python3
"""How much per-coordinate correlation can the kt2 coordinate probe actually EXCLUDE?

Cycle 43 measured, at true per-weight granularity, a pairwise sign-agreement excess of
+0.0001 pp across tensors and +0.0007 pp within a tensor, both against a
marginal-preserving circular-shift null (FINDINGS 43.3).  It is tempting to read that as
"the per-weight meta-gradients are independent, so pooling is not variance-limited".

That inference needs a power calculation, because a GROUP meta-gradient is a SUM over its
members (HF.py:149).  Averaging n coordinates amplifies any shared component by ~n relative
to the idiosyncratic part, so a per-weight correlation far too small to detect can still
dominate the aggregate.  This script asks the quantitative question directly:

    given a one-factor model with per-weight correlation rho, what is
      (a) the per-weight pairwise sign-agreement excess   (what kt2 measures)
      (b) the tensor-MEAN pairwise sign-agreement excess  (what KILLTEST sec.3 measures)
    and what rho does kt2's measured standard error actually exclude?

Ground truth is simulated, so the answer does not depend on any estimator in this repo.
"""
import numpy as np

# For jointly Gaussian zero-mean (X, Y) with correlation r:
#     P(sign X == sign Y) = 1/2 + arcsin(r)/pi
def sign_agree(r):
    return 0.5 + np.arcsin(np.clip(r, -1, 1)) / np.pi


def one_factor(rho_w, rho_x, n, R, rng):
    """Two tensors of n coords each.  Within-tensor correlation rho_w, cross-tensor rho_x.

    z_A,i = sqrt(rho_x) G + sqrt(rho_w - rho_x) F_A + sqrt(1 - rho_w) e_Ai
    so corr(z_Ai, z_Aj) = rho_w and corr(z_Ai, z_Bj) = rho_x  exactly.
    """
    G = rng.standard_normal((R, 1))
    FA = rng.standard_normal((R, 1))
    FB = rng.standard_normal((R, 1))
    eA = rng.standard_normal((R, n))
    eB = rng.standard_normal((R, n))
    a, b = np.sqrt(rho_x), np.sqrt(max(rho_w - rho_x, 0.0)),
    c = np.sqrt(max(1.0 - rho_w, 0.0))
    zA = a * G + b * FA + c * eA
    zB = a * G + b * FB + c * eB
    return zA, zB


def pairwise_excess(sA, sB):
    """Mean same-sign rate between every coord of A and every coord of B, exactly."""
    R = sA.shape[0]
    # E[(1 + s_i s_j)/2] averaged over all i in A, j in B
    prod = (sA.sum(1) * sB.sum(1)).mean() / (sA.shape[1] * sB.shape[1])
    return (0.5 + prod / 2.0 - 0.5) * 100.0  # in pp, excess over 50%


rng = np.random.default_rng(0)
R = 1000          # records, as in kt2's steady window
N_TRACK = 322     # ~19956 tracked coords / 62 tensors

print("=" * 78)
print("PART 1 -- analytic: what a per-weight rho does to the TENSOR MEAN")
print("=" * 78)
print("corr of two tensor means, one-factor model, rho_within = rho_cross = rho:")
print("    rho_means = rho*n / (1 + (n-1)*rho)")
print()
print(f"{'rho (per weight)':>18} {'n':>9} {'rho_means':>10} {'per-weight excess':>19} {'tensor-mean excess':>20}")
for rho in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
    for n in [1e4, 1e5, 1e6]:
        rm = rho * n / (1 + (n - 1) * rho)
        print(f"{rho:>18.1e} {n:>9.0e} {rm:>10.4f} "
              f"{(sign_agree(rho)-0.5)*100:>18.5f}pp {(sign_agree(rm)-0.5)*100:>19.3f}pp")
    print()

print("=" * 78)
print("PART 2 -- simulation check of the per-weight estimator at kt2's sample size")
print("=" * 78)
print(f"R={R} records, {N_TRACK} tracked coords per tensor (kt2's actual density)")
print(f"{'true rho_cross':>16} {'measured per-weight excess':>28} {'sd over 20 reps':>18}")
for rho in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]:
    vals = []
    for rep in range(20):
        zA, zB = one_factor(max(rho, 1e-12) * 2, rho, N_TRACK, R, rng)
        vals.append(pairwise_excess(np.sign(zA), np.sign(zB)))
    vals = np.array(vals)
    print(f"{rho:>16.1e} {vals.mean():>25.5f}pp {vals.std(ddof=1):>17.5f}pp")

print()
print("=" * 78)
print("PART 3 -- what kt2's measured standard error EXCLUDES")
print("=" * 78)
# kt2 a0=1e-3: across_tensor_within_block observed 50.0002%, "+1.17sd" vs the 50% null
obs_pp, nsd = 0.0002, 1.17
sd_pp = obs_pp / nsd
ub_pp = obs_pp + 1.96 * sd_pp
rho_ub = np.sin(np.pi * ub_pp / 100.0)
print(f"observed across-tensor excess = {obs_pp:.5f}pp = {nsd:.2f} sd  ->  1 sd = {sd_pp:.5f}pp")
print(f"95% upper bound on the excess  = {ub_pp:.5f}pp  ->  rho_cross <= {rho_ub:.3e}")
print()
for n in [1e4, 1e5, 1e6]:
    rm = rho_ub * n / (1 + (n - 1) * rho_ub)
    print(f"  a per-weight rho of {rho_ub:.2e} in tensors of n={n:.0e} gives tensor-mean "
          f"corr {rm:.3f} -> mean-sign excess {(sign_agree(rm)-0.5)*100:.2f}pp")
print()
print("KILLTEST sec.3 measured a tensor-level WITHIN-BLOCK excess of +1.0pp over the same")
print("shift null.  Inverting: rho_means = sin(pi*0.010) = %.4f" % np.sin(np.pi * 0.010))
rm_obs = np.sin(np.pi * 0.010)
for n in [1e4, 1e5, 1e6]:
    rho_needed = rm_obs / (n - (n - 1) * rm_obs)
    print(f"  reproducing it in tensors of n={n:.0e} needs per-weight rho = {rho_needed:.2e}"
          f"  -> per-weight excess {(sign_agree(rho_needed)-0.5)*100:.6f}pp")


print()
print("=" * 78)
print("PART 4 is slow (6 reps x 1000 x 19956) and is run separately; its output is stored at")
print("results/power_part4.txt and tabulated in FINDINGS 43.3c.  It measures the power of the")
print("tracked-subsample MAJORITY test against a GLOBAL COMMON MODE, which is the test that")
print("carries the marginal-bias conclusion.  Reproduce with the snippet in FINDINGS 43.3c.")
