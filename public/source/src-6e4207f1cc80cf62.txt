"""cX1_reduction_probe.py -- INSTRUMENT `HF.block_product`'s per-tensor terms.

WHAT THIS IS.  A zero-claim MEASUREMENT tool, not a scorer.  It runs the REAL
optimizer at the campaign's standard CIFAR-100 cell for a short prefix of
training and records, at EVERY meta-step, the 62 per-tensor inner products

    t_i = <h_condenced[i], g[i]> = (u[i]*v[i]).sum()

that `HF.block_product` sums inside each group.  It changes NOTHING: the wrapper
computes t_i and then calls the UNMODIFIED bound method, so the trajectory it
observes is the trajectory the unprobed run would have taken.  Nothing is
written to $WS/runs, so no aggregate.py ingest can see it.

WHY.  Every group's meta-signal is the UNNORMALISED SUM z_G = sum_{i in G} t_i,
and `Lion_meta_update` applies torch.sign() to it.  Two questions decide whether
any "normalise the reduction" experiment has power, and both are answerable from
the t_i alone, for free:

  Q1  DOMINATION.  How often does sign(z_G) equal sign(t_j) for the single
      j in G with the largest |t_j|?  (The briefing's claimed mechanism.)
  Q2  POWER OF THE INTERVENTION.  How often does sign(z_G) DIFFER from
      sign(sum_{i in G} t_i / n_i), the per-tensor-normalised reduction?
      If the two signs agree at ~100% of steps, then a per-tensor-normalised
      reduction produces the SAME meta-update as the standard one and any batch
      built on it is a null BY CONSTRUCTION and must not be run.

RUN (on the cluster, inside envs/mo):
  python3 analysis/cX1_reduction_probe.py --cifar-dir <cifar10 dir> \
      --stepsize-groups '[49,13]' --seed 0 --epochs 20 --out /path/prefix

Writes <prefix>.terms.npy (float64, steps x 62), <prefix>.meta.json.
"""
import argparse, json, os, sys, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar-dir", required=True)
    ap.add_argument("--stepsize-groups", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--meta-stepsize", type=float, default=1e-3)
    ap.add_argument("--alpha0", type=float, default=1e-6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn

    sys.path.insert(0, a.cifar_dir)
    os.chdir(a.cifar_dir)
    from build_network import build_network
    from Optimizers.build_optimizer import build_optimizer
    from load_data import load_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    class A:
        pass
    ar = A()
    ar.optimizer = "HF"
    ar.alg_base, ar.normalizer_param_base = "SGDm", None
    ar.momentum_param_base, ar.weight_decay_base = 0.99, 0.1
    ar.Lion_beta2_base = None
    ar.alg_meta, ar.normalizer_param_meta = "Lion", None
    ar.momentum_param_meta, ar.weight_decay_meta = 0.99, 0.0
    ar.Lion_beta2_meta = 0.9
    ar.meta_stepsize, ar.alpha0, ar.gamma = a.meta_stepsize, a.alpha0, 1.0
    ar.stepsize_groups = a.stepsize_groups
    ar.device = device

    criterion = nn.CrossEntropyLoss().to(device)
    net = build_network("ResNet18_c100", device)
    names = [n for n, _ in net.named_parameters()]
    numel = [int(p.numel()) for _, p in net.named_parameters()]
    opt = build_optimizer(net, ar, writer=_NullWriter())
    trainloader, _ = load_data("CIFAR100", a.batch_size, a.seed)

    terms = []
    orig_bp = opt.block_product

    def wrapped(u, v):
        with torch.no_grad():
            t = torch.stack([(u[i] * v[i]).sum() for i in range(len(u))])
            terms.append(t.detach().to(torch.float64).cpu().numpy())
        return orig_bp(u, v)

    opt.block_product = wrapped

    t0 = time.time()
    for ep in range(a.epochs):
        for data in trainloader:
            x, y = data[0].to(device), data[1].to(device)
            loss = criterion(net(x), y)
            opt.step(net, loss)
        print("epoch %d  steps %d  %.1f s" % (ep, len(terms), time.time() - t0),
              flush=True)

    arr = np.asarray(terms, dtype=np.float64)
    np.save(a.out + ".terms.npy", arr)
    json.dump({"spec": a.stepsize_groups, "seed": a.seed, "epochs": a.epochs,
               "steps": int(arr.shape[0]), "names": names, "numel": numel,
               "stepsize_type": getattr(opt, "stepsize_type", "?"),
               "len_beta_list": int(getattr(opt, "len_beta_list", -1)),
               "param_groups_indices":
                   [[int(i) for i in gg] for gg in
                    getattr(opt, "param_groups_indices", [])]},
              open(a.out + ".meta.json", "w"), indent=1)
    print("WROTE %s.terms.npy shape=%s" % (a.out, arr.shape))


class _NullWriter:
    def add_scalar(self, *x, **k):
        pass

    def add_scalars(self, *x, **k):
        pass

    def add_text(self, *x, **k):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    main()
