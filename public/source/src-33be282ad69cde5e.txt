"""PATCH_VALSPLIT -- an OPT-IN, ENVIRONMENT-CONTROLLED held-out VALIDATION split of the CIFAR training set, with a per-epoch
VAL line in the run's .out.  Written for 1.14 of docs/ICML-PLAN.md (CORRECTIONS 302; the batch that uses it is `cvl1`,
CORRECTIONS 303).  Applied ONLY to the isolated tree $METAOPT_WS/harness_cvl1/cifar10 (bin/cVL1_stage_harness.sh), ON TOP
of the cgw1 tree's bytes (load_data.py b52b58a3..., train.py 3fea309e...; HF.py 4732b74a... is NOT touched).  The live
shared harness and every registered tree are never patched.

-------------------------------------------------------------------------------
WHY
-------------------------------------------------------------------------------
Every number the count-matched partition audit reports is plateau5 = mean TEST accuracy over epochs 95-99, and every
hyperparameter of its core cell that was CHOSEN was chosen on TEST accuracy (CORRECTIONS 302.1): the harness has no
validation split (CORRECTIONS 135.1, PLAN.md B3).  This patch carves one out of the 50,000 training images so a batch can
read the same contrasts on data the model never trained on AND nobody ever selected on.

-------------------------------------------------------------------------------
THE SWITCH
-------------------------------------------------------------------------------
    VAL_SPLIT=<n_val>:<split_seed>          e.g. VAL_SPLIT=5000:302   (the registered value)

  * UNSET or EMPTY -> OFF.  OFF is the state every prior run was in: the train loader's dataset is the full CIFAR object,
    no val loader exists, no VAL line is printed, and training is bitwise the unpatched tree's (tests/test_valsplit.py V0,
    tests/test_valsplit_realrun.py RR1).  One witness line is printed on EVERY run of the tree:
        VAL_SPLIT: off
        VAL_SPLIT: on dataset=<D> n_val=<n> n_train=<N-n> classes=<K> per_class=<n/K> split_seed=<s> val_sha=<64 hex> train_sha=<64 hex>
    (the runner's ENV line cannot carry VAL_SPLIT; this line is the run's own record of it).  Its prefix `VAL_SPLIT`
    is not a prefix of any registered kind (VOTE_W, BETA_HOLD, ...) and none of them is a prefix of it.
  * THE SPLIT is CLASS-STRATIFIED and a function of (the training labels, n_val, split_seed) ONLY: for each class c in
    increasing order, the ascending indices of class c are permuted by ONE numpy RandomState(split_seed) (the legacy
    MT19937 stream, whose `permutation` is fixed across numpy versions) and the first n_val/K are held out.  It does not
    read or advance the global numpy or torch RNG, so it is IDENTICAL for every run seed.  val_sha / train_sha are the
    sha256 of the sorted index lists as little-endian int64.
  * The held-out images are read through a SECOND dataset object with the TEST transform (ToTensor + Normalize, no
    augmentation); the model trains on the remaining N - n_val through the original (AUGMENT) transform, with the same
    shuffled DataLoader as before.  The test loader is untouched.
  * VAL LINE, after the unchanged `Epoch %d, Train Accuracy ... Test Accuracy ...` line of every epoch:
        VAL: epoch <e> val_acc <acc to 2 d.p.> % n_val <n>
    computed by the harness's own compute_test_accuracy (net.eval(), no_grad, net.train()).  With n_val 5,000 every
    accuracy is a multiple of 0.02, so two decimals are EXACT.  It contains neither `Test Accuracy: ` nor
    `Epoch <n>, Train Accuracy`, so no corpus or scorer reader of test accuracy can read it.
  * PREMISES, LOUD (ValueError at load_data): a malformed value (anything but <digits>:<digits>); n_val 0, not a multiple
    of the class count, or >= N; a class with fewer than n_val/K images; a dataset other than CIFAR10 / CIFAR100.
  * Every character of a value is in [0-9:], so VAL_SPLIT=... is ONE token of sbatch's --export list.

-------------------------------------------------------------------------------
INERTNESS
-------------------------------------------------------------------------------
FOUR insertions, every one ADDITIVE; NO existing line is edited:
  load_data.py  1. ONE call `trainset = _vs_apply(...)` between the dataset branches' `else: 0/0` and the train DataLoader;
                2. ONE module block (VAL_LOADER = None and the helpers), appended after compute_test_accuracy;
  train.py      3. `import load_data as _vs_ld  # PATCH_VALSPLIT` right after `from load_data import ...`;
                4. ONE guarded VAL evaluation right after the per-epoch print.
With VAL_SPLIT unset or empty, _vs_apply prints `VAL_SPLIT: off` and returns trainset ITSELF (the same object), VAL_LOADER
stays None, (4) does nothing.  Deleting the four regions reproduces the pre files byte for byte (tests/test_valsplit.py V0).

    VS_TREE=$WS/harness_cvl1/cifar10 python3 patches/patch_valsplit.py
"""
import os
import sys

# ---- the registered regions (tests/test_valsplit.py reads these; the test proves deleting them restores the pre bytes) ----
L_ANCHOR_CALL = "    else: 0/0 # return error\n"
L_ANCHOR_LOADER = "    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)\n"
L_CALL = ("    # --- PATCH_VALSPLIT ---\n"
          "    trainset = _vs_apply(dataset_name, trainset, test_transform if dataset_name in ('CIFAR10', 'CIFAR100') else None, batch_size)\n"
          "    # --- end PATCH_VALSPLIT ---\n")
L_ANCHOR_END = "    return test_accuracy"
L_BLOCK_HEAD = "\n\n\n# ------------------------------------------------------------ PATCH_VALSPLIT\n"
L_BLOCK_TAIL = "# -------------------------------------------------------- end PATCH_VALSPLIT\n"
L_BLOCK = (
    L_BLOCK_HEAD +
    "# OPT-IN held-out validation split (VAL_SPLIT=<n_val>:<split_seed>); patches/patch_valsplit.py, CORRECTIONS 302.\n"
    "VAL_LOADER = None\n"
    "\n"
    "\n"
    "def _vs_parse(raw):\n"
    "    \"\"\"'' -> None (OFF); '<n_val>:<split_seed>' -> (n_val, split_seed); anything else raises.\"\"\"\n"
    "    import re as _re\n"
    "    if raw == '':\n"
    "        return None\n"
    "    m = _re.fullmatch(r'([0-9]+):([0-9]+)', raw)\n"
    "    if not m:\n"
    "        raise ValueError('PATCH_VALSPLIT: VAL_SPLIT %r is not <n_val>:<split_seed> (digits only)' % (raw,))\n"
    "    n_val, seed = int(m.group(1)), int(m.group(2))\n"
    "    if n_val <= 0:\n"
    "        raise ValueError('PATCH_VALSPLIT: n_val must be positive (VAL_SPLIT %r)' % (raw,))\n"
    "    if seed >= 2 ** 32:\n"
    "        raise ValueError('PATCH_VALSPLIT: split_seed must be < 2**32 (VAL_SPLIT %r)' % (raw,))\n"
    "    return n_val, seed\n"
    "\n"
    "\n"
    "def _vs_split(targets, n_val, split_seed):\n"
    "    \"\"\"Class-stratified split, a function of (targets, n_val, split_seed) ONLY.  -> (train_idx, val_idx), sorted int64.\n"
    "    ONE RandomState(split_seed); for each class in increasing order its ascending indices are permuted and the first\n"
    "    n_val/K are held out.  Never reads or advances the global numpy / torch RNG.\"\"\"\n"
    "    import numpy as _np\n"
    "    T = _np.asarray(targets, dtype=_np.int64)\n"
    "    N = int(T.shape[0])\n"
    "    classes = _np.unique(T)\n"
    "    K = int(classes.shape[0])\n"
    "    if n_val >= N:\n"
    "        raise ValueError('PATCH_VALSPLIT: n_val %d >= the %d training images' % (n_val, N))\n"
    "    if n_val % K:\n"
    "        raise ValueError('PATCH_VALSPLIT: n_val %d is not a multiple of the %d classes' % (n_val, K))\n"
    "    k = n_val // K\n"
    "    rs = _np.random.RandomState(split_seed)\n"
    "    val = []\n"
    "    for c in classes:\n"
    "        idx = _np.flatnonzero(T == c)\n"
    "        if idx.shape[0] <= k:\n"
    "            raise ValueError('PATCH_VALSPLIT: class %d has %d images, cannot hold out %d' % (int(c), idx.shape[0], k))\n"
    "        val.append(rs.permutation(idx)[:k])\n"
    "    val = _np.sort(_np.concatenate(val)).astype(_np.int64)\n"
    "    keep = _np.ones(N, dtype=bool)\n"
    "    keep[val] = False\n"
    "    train = _np.flatnonzero(keep).astype(_np.int64)\n"
    "    return train, val\n"
    "\n"
    "\n"
    "def _vs_sha(idx):\n"
    "    import hashlib as _h, numpy as _np\n"
    "    return _h.sha256(_np.asarray(idx, dtype='<i8').tobytes()).hexdigest()\n"
    "\n"
    "\n"
    "def _vs_witness(dataset_name, n_val, split_seed, n_train, n_val_got, K, per_class, train_idx, val_idx):\n"
    "    return ('VAL_SPLIT: on dataset=%s n_val=%d n_train=%d classes=%d per_class=%d split_seed=%d val_sha=%s train_sha=%s'\n"
    "            % (dataset_name, n_val_got, n_train, K, per_class, split_seed, _vs_sha(val_idx), _vs_sha(train_idx)))\n"
    "\n"
    "\n"
    "def _vs_apply(dataset_name, trainset, val_transform, batch_size):\n"
    "    \"\"\"OFF: print `VAL_SPLIT: off`, return trainset ITSELF.  ON: set VAL_LOADER, return the train Subset.\"\"\"\n"
    "    import os as _os\n"
    "    global VAL_LOADER\n"
    "    VAL_LOADER = None\n"
    "    p = _vs_parse(_os.environ.get('VAL_SPLIT', ''))\n"
    "    if p is None:\n"
    "        print('VAL_SPLIT: off', flush=True)\n"
    "        return trainset\n"
    "    if dataset_name not in ('CIFAR10', 'CIFAR100') or val_transform is None:\n"
    "        raise ValueError('PATCH_VALSPLIT: the split is defined for CIFAR10 / CIFAR100 only, not %r' % (dataset_name,))\n"
    "    n_val, split_seed = p\n"
    "    train_idx, val_idx = _vs_split(trainset.targets, n_val, split_seed)\n"
    "    K = len(set(int(t) for t in trainset.targets))\n"
    "    import torch as _torch\n"
    "    from torchvision import datasets as _ds\n"
    "    cls = _ds.CIFAR10 if dataset_name == 'CIFAR10' else _ds.CIFAR100\n"
    "    valfull = cls(root='./data', train=True, download=True, transform=val_transform)\n"
    "    if list(valfull.targets) != list(trainset.targets):\n"
    "        raise ValueError('PATCH_VALSPLIT: the val-transform copy of the train set has different labels')\n"
    "    tr = _torch.utils.data.Subset(trainset, [int(i) for i in train_idx])\n"
    "    va = _torch.utils.data.Subset(valfull, [int(i) for i in val_idx])\n"
    "    VAL_LOADER = _torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False)\n"
    "    print(_vs_witness(dataset_name, n_val, split_seed, len(tr), len(va), K, n_val // K, train_idx, val_idx), flush=True)\n"
    "    return tr\n" +
    L_BLOCK_TAIL)

T_ANCHOR_IMPORT = "from load_data import load_data, compute_test_accuracy\n"
T_IMPORT = "import load_data as _vs_ld  # PATCH_VALSPLIT\n"
T_ANCHOR_EPOCH = ("        if args.verbos: print('Epoch %d, Train Accuracy: %.2f %%, Test Accuracy: %.2f %%' "
                  "% (epoch, train_accuracy, test_accuracy))\n")
T_VAL = ("        # --- PATCH_VALSPLIT ---\n"
         "        if _vs_ld.VAL_LOADER is not None:\n"
         "            val_accuracy = compute_test_accuracy(net, _vs_ld.VAL_LOADER, args.device)\n"
         "            print('VAL: epoch %d val_acc %.2f %% n_val %d' % (epoch, val_accuracy, len(_vs_ld.VAL_LOADER.dataset)), flush=True)\n"
         "        # --- end PATCH_VALSPLIT ---\n")


def main():
    T = os.environ.get("VS_TREE", "")
    if not T:
        print("!!! VS_TREE is required: this patch is applied ONLY to an isolated tree (its cifar10 directory)")
        return 2
    pl, pt = os.path.join(T, "load_data.py"), os.path.join(T, "train.py")
    sl, st = open(pl).read(), open(pt).read()
    if "PATCH_VALSPLIT" in sl and "PATCH_VALSPLIT" in st:
        print("ALREADY_PATCHED")
        return 0
    try:
        assert "PATCH_VALSPLIT" not in sl and "PATCH_VALSPLIT" not in st, "one file is patched and the other is not"
        assert "VAL_SPLIT" not in sl + st and "_vs_" not in sl + st and "VAL_LOADER" not in sl + st, "unexpected _vs_ names"
        a1 = L_ANCHOR_CALL + L_ANCHOR_LOADER
        assert sl.count(a1) == 1, "load_data anchor 1 count=%d" % sl.count(a1)
        assert sl.endswith(L_ANCHOR_END) and sl.count(L_ANCHOR_END) == 1, "load_data does not end with compute_test_accuracy's return"
        assert "test_transform = transforms.Compose([transforms.ToTensor(), norm])" in sl, "no test_transform in the CIFAR branches"
        assert st.count(T_ANCHOR_IMPORT) == 1, "train anchor 1 count=%d" % st.count(T_ANCHOR_IMPORT)
        assert st.count(T_ANCHOR_EPOCH) == 1, "train anchor 2 count=%d" % st.count(T_ANCHOR_EPOCH)
    except AssertionError as e:
        print("!!! REFUSING TO PATCH: %s" % e)
        return 2
    sl2 = sl.replace(a1, L_ANCHOR_CALL + L_CALL + L_ANCHOR_LOADER, 1) + L_BLOCK
    st2 = st.replace(T_ANCHOR_IMPORT, T_ANCHOR_IMPORT + T_IMPORT, 1).replace(T_ANCHOR_EPOCH, T_ANCHOR_EPOCH + T_VAL, 1)
    for p, src in ((pl, sl), (pt, st)):
        bak = p + ".pre_valsplit"
        if os.path.exists(bak) and open(bak).read() != src:
            print("!!! %s exists and differs from the file being patched -- refusing" % bak)
            return 2
    for p, src in ((pl, sl), (pt, st)):
        if not os.path.exists(p + ".pre_valsplit"):
            open(p + ".pre_valsplit", "w").write(src)
    open(pl, "w").write(sl2)
    open(pt, "w").write(st2)
    print("PATCHED PATCH_VALSPLIT -> %s, %s (backups *.pre_valsplit)" % (pl, pt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
