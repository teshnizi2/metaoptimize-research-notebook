"""PATCH_VOTEWEIGHT -- an OPT-IN, ENVIRONMENT-CONTROLLED per-tensor weight on a
tensor's term INSIDE the shared meta-gradient sum, BEFORE the sign.  Written for
`cvt1` (CORRECTIONS 227).  Applied ONLY to the isolated tree
$METAOPT_WS/harness_cvt1/cifar10 (bin/cVT1_stage_harness.sh); the live shared
harness is never patched.

-------------------------------------------------------------------------------
WHY
-------------------------------------------------------------------------------
On the live source the scalar and blockwise meta-signals are

    z_G = sum_{i in G} <h_condenced[i], g[i]>          (block_product; UNNORMALISED)

and Lion_meta_update applies  beta <- beta - ms*sign(b2*momentum_meta + (1-b2)*z),
momentum_meta <- mom*momentum_meta + (1-mom)*z.  Only the SIGN of the sum moves
beta.  On PlainNet18_c100 (cpl1/cpl2) one tensor, layer4.1.bn2.weight (1-based
index 50 of 53), holds ~0.61 of k01's pinned-record mass and removing its term
flips that sign on 1.0000 of records; isolating it (its own step size) rescues.
Set-matching cannot say whether the rescue comes from REMOVING ITS VOTE from the
shared sum or from GIVING IT ITS OWN STEP SIZE, because no carrier-free tensor
comes within x691 of its term.  This patch intervenes on the vote directly:

    z_G = sum_{i in G} w_i * <h_condenced[i], g[i]>,   w_i = 1 unless named

`momentum_meta` is a LINEAR recursion in z, so weighting z also weights the
tensor's contribution to the momentum by exactly w_i: the intervention is a
reweighting of the tensor's whole vote, not of one step's.

-------------------------------------------------------------------------------
THE SWITCH
-------------------------------------------------------------------------------
    VOTE_W=<name>:<w>[/<name>:<w>...]         (environment variable)

  * UNSET or EMPTY  ->  OFF.  OFF is the state every prior run was in.
  * <name> is an exact parameter name of the LIVE model (resolved by NAME at
    init_meta, never by index); <w> is a plain non-negative decimal
    (`^[0-9]+(\\.[0-9]+)?$`: no sign, no exponent, no `inf`/`nan`).
  * Every character of a value is in [A-Za-z0-9_.:/], so `VOTE_W=...` is ONE
    token inside sbatch's comma-separated `--export` list.
  * LOUD, never silently partial: an unknown name, a repeated name, a malformed
    weight, a granularity other than `scalar` / `blockwise`, a named tensor that
    sits ALONE in its group (w on a singleton group's own sum is a positive
    per-group constant -- inert under sign() for w > 0, and for w = 0 it deletes
    the group's whole signal: neither is a reweighting INSIDE a shared sum), or a
    `tn:` (PATCH_REDNORM) spec -> ValueError at construction.
  * A witness line is printed at construction, on EVERY run:
        VOTE_W: off
        VOTE_W: on type=<scalar|blockwise> items=<idx1>:<name>:w=<repr>:group=<k>:groupsize=<n>[,...]
    (the runner's ENV line cannot carry VOTE_W -- the runner may differ from
    run_cifar.sh in its `cd` line only -- so THIS line is what RULE 20's separate
    ENV audit and the scorer's G-VOTEW gate read.)

-------------------------------------------------------------------------------
INERTNESS
-------------------------------------------------------------------------------
FOUR insertions, every one ADDITIVE; NO existing line is edited:
  1. a guarded early return at the top of block_product's `scalar` branch;
  2. a guarded early return at the top of block_product's `blockwise` branch;
  3. ONE call, `self._vw_init(net_param_names_and_size)`, after the last line of
     init_meta (`self.len_beta_list = len(self.beta)`), which sets `_vw_on`;
  4. the three new methods (`_vw_parse`, `_vw_init`, `_vw_block_product`),
     inserted as one block before `check_required_attributes`.
With `_vw_on` False, (1) and (2) fall through to the ORIGINAL code, byte-
unchanged (including PATCH_REDNORM's guard).  With `_vw_on` True and every named
weight 1.0, `_vw_block_product` computes the SAME per-tensor terms in the SAME
order and sums them the SAME way (python `sum` over a list, left to right), and
`t * 1.0 == t` bitwise in IEEE-754, so the identity setting is bitwise identical
too.  tests/test_voteweight.py proves the structure (deleting the four regions
reproduces the pre-patch file byte for byte) and the arithmetic on CPU;
tests/test_voteweight_realrun.py proves beta trajectory, probe records and final
weights BITWISE on a short real CIFAR-100 PlainNet run, off and at identity.

The guards read `getattr(self, '_vw_on', False)` so an HF object built by a path
that bypasses init_meta cannot raise.

WHAT THIS PATCH DOES NOT DO.  It adds no granularity, no spec prefix, no
hyperparameter on the command line; it does not touch the meta-update, the
clamp, the probe (PROBE_TENSOR's `z_tensor` stays the RAW per-tensor terms and
its `z_agg` is the harness's WEIGHTED z -- which is what makes the intervention
auditable from the records), the schedule, `beta_to_alpha` or `base_update`.
It does not change what the base optimizer does to any weight.
"""
import os, re, sys

P = os.environ.get("HF_PATH", "")
if not P:
    print("!!! HF_PATH is required: this patch is applied ONLY to an isolated tree's HF.py")
    sys.exit(2)
src = open(P).read()

if "PATCH_VOTEWEIGHT" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# --- 0. the assumptions this patch rests on, checked BEFORE anything is written ----
assert "def block_product(self, u, v):" in src, "block_product is not where expected"
assert "def Lion_meta_update(self,HtT_gradft):" in src, "Lion_meta_update is missing"
assert "_vw_" not in src, "unexpected pre-existing _vw_ names"
assert "VOTE_W" not in src, "unexpected pre-existing VOTE_W handling"

# --- 1. the scalar reduction --------------------------------------------------------
a1 = ("    def block_product(self, u, v):\n"
      "        if self.stepsize_type == 'scalar':\n")
assert src.count(a1) == 1, "anchor 1 count=%d -- refusing to patch" % src.count(a1)
n1 = ("    def block_product(self, u, v):\n"
      "        if self.stepsize_type == 'scalar':\n"
      "            # --- PATCH_VOTEWEIGHT ---\n"
      "            if getattr(self, '_vw_on', False):\n"
      "                return self._vw_block_product(u, v)\n"
      "            # --- end PATCH_VOTEWEIGHT ---\n")

# --- 2. the blockwise reduction -----------------------------------------------------
a2 = ("            return [sum([(u_*v_).sum() for u_,v_ in zip(u,v)])]\n"
      "        if self.stepsize_type == 'blockwise':\n")
assert src.count(a2) == 1, "anchor 2 count=%d -- refusing to patch" % src.count(a2)
n2 = ("            return [sum([(u_*v_).sum() for u_,v_ in zip(u,v)])]\n"
      "        if self.stepsize_type == 'blockwise':\n"
      "            # --- PATCH_VOTEWEIGHT ---\n"
      "            if getattr(self, '_vw_on', False):\n"
      "                return self._vw_block_product(u, v)\n"
      "            # --- end PATCH_VOTEWEIGHT ---\n")

# --- 3. the call at the end of init_meta --------------------------------------------
a3 = ("        self.param_numels = [int(np.prod(list(p_size))) for (_n, p_size) in net_param_names_and_size]\n"
      "        self.len_beta_list = len(self.beta)\n")
assert src.count(a3) == 1, "anchor 3 count=%d -- refusing to patch" % src.count(a3)
n3 = a3 + ("        self._vw_init(net_param_names_and_size)  # PATCH_VOTEWEIGHT\n")

# --- 4. the methods, before check_required_attributes -------------------------------
a4 = "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n"
assert src.count(a4) == 1, "anchor 4 count=%d -- refusing to patch" % src.count(a4)
n4 = (
    "    # ------------------------------------------------------ PATCH_VOTEWEIGHT\n"
    "    def _vw_parse(self, raw, names):\n"
    "        \"\"\"`<name>:<w>[/<name>:<w>...]` -> [(index0, name, w)].  Total and loud.\"\"\"\n"
    "        import re as _re, math as _math\n"
    "        out, seen = [], set()\n"
    "        for item in raw.split('/'):\n"
    "            m = _re.match(r'^([A-Za-z0-9_.]+):([0-9]+(?:\\.[0-9]+)?)$', item)\n"
    "            if not m:\n"
    "                raise ValueError('PATCH_VOTEWEIGHT: VOTE_W item %r is not <name>:<decimal>' % item)\n"
    "            nm, w = m.group(1), float(m.group(2))\n"
    "            if nm not in names:\n"
    "                raise ValueError('PATCH_VOTEWEIGHT: %r is not a parameter name of this model' % nm)\n"
    "            if nm in seen:\n"
    "                raise ValueError('PATCH_VOTEWEIGHT: %r is named twice in VOTE_W' % nm)\n"
    "            if not _math.isfinite(w) or w < 0.0:\n"
    "                raise ValueError('PATCH_VOTEWEIGHT: weight %r for %r is not finite and >= 0' % (w, nm))\n"
    "            seen.add(nm)\n"
    "            out.append((names.index(nm), nm, w))\n"
    "        return out\n"
    "\n"
    "    def _vw_init(self, net_param_names_and_size):\n"
    "        import os as _os\n"
    "        self._vw_on = False\n"
    "        self._vw_items = []\n"
    "        raw = _os.environ.get('VOTE_W', '')\n"
    "        if raw == '':\n"
    "            print('VOTE_W: off', flush=True)\n"
    "            return\n"
    "        if getattr(self, '_rednorm', False):\n"
    "            raise ValueError('PATCH_VOTEWEIGHT: VOTE_W cannot be combined with a tn: (PATCH_REDNORM) spec')\n"
    "        if self.stepsize_type not in ('scalar', 'blockwise'):\n"
    "            raise ValueError('PATCH_VOTEWEIGHT: VOTE_W needs a shared sum; stepsize_type is %r'\n"
    "                             % (self.stepsize_type,))\n"
    "        names = [n for (n, _s) in net_param_names_and_size]\n"
    "        items = self._vw_parse(raw, names)\n"
    "        if self.stepsize_type == 'scalar':\n"
    "            groups = [list(range(len(names)))]\n"
    "        else:\n"
    "            groups = self.param_groups_indices\n"
    "        desc = []\n"
    "        for i, nm, w in items:\n"
    "            k = [j for j, g in enumerate(groups) if i in g]\n"
    "            if len(k) != 1:\n"
    "                raise ValueError('PATCH_VOTEWEIGHT: %r is in %d groups' % (nm, len(k)))\n"
    "            if len(groups[k[0]]) < 2:\n"
    "                raise ValueError('PATCH_VOTEWEIGHT: %r is ALONE in group %d; a weight there is not a '\n"
    "                                 'reweighting inside a shared sum' % (nm, k[0]))\n"
    "            desc.append('%d:%s:w=%r:group=%d:groupsize=%d' % (i + 1, nm, w, k[0], len(groups[k[0]])))\n"
    "        self._vw_items = [(i, w) for i, _nm, w in items]\n"
    "        self._vw_on = True\n"
    "        print('VOTE_W: on type=%s items=%s' % (self.stepsize_type, ','.join(desc)), flush=True)\n"
    "\n"
    "    def _vw_block_product(self, u, v):\n"
    "        \"\"\"block_product with w_i on the named tensors' terms.  The per-tensor terms,\n"
    "        their order and the left-to-right python sum are the original branch's.\"\"\"\n"
    "        terms = [(u_*v_).sum() for u_,v_ in zip(u,v)]\n"
    "        for i, w in self._vw_items:\n"
    "            terms[i] = terms[i] * w\n"
    "        if self.stepsize_type == 'scalar':\n"
    "            return [sum(terms)]\n"
    "        return [torch.tensor([sum([terms[i] for i in group_indices]) for group_indices in self.param_groups_indices])]\n"
    "    # -------------------------------------------------- end PATCH_VOTEWEIGHT\n"
    "\n"
    + a4)

src2 = src.replace(a1, n1, 1).replace(a2, n2, 1).replace(a3, n3, 1).replace(a4, n4, 1)
bak = P + ".pre_voteweight"
if os.path.exists(bak):
    if open(bak).read() != src:
        print("!!! %s exists and differs from the file being patched -- refusing" % bak)
        sys.exit(2)
else:
    open(bak, "w").write(src)
open(P, "w").write(src2)
print("PATCHED PATCH_VOTEWEIGHT ->", P, "(backup", bak + ")")
