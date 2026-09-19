"""PATCH_DECAYMASK -- an OPT-IN, ENVIRONMENT-CONTROLLED mask that sets the BASE optimiser's coupled weight decay to 0
on a NAMED set of parameter tensors (all normalisation scales, or listed tensors), while every other tensor, the meta
update, and every hold patch run unchanged.  Written for `cwd1` / `cwd2` (CORRECTIONS 260 / 261).  Applied ONLY to the
isolated trees $METAOPT_WS/harness_cwd1/cifar10 (on top of cvt8's HF.py, ResNet18_c100) and
$METAOPT_WS/harness_cwd2/cifar10 (on top of cvt9's HF.py, PlainNet18_c100), built by bin/cWD_stage_harness.sh; the live
shared harness and every registered tree (cvt1 ... cvt9) are never patched.

-------------------------------------------------------------------------------
WHY
-------------------------------------------------------------------------------
Every campaign run uses SGDm with coupled weight decay 0.1, and HF.py's SGDm_base_update applies it to EVERY tensor,
normalisation scales included (LIMITS-PREP 3.2):

    delta = a * (m + wd*w);   w <- w - delta;   h <- gamma*(1 - wd*a)*h - delta

so a large step size on a BatchNorm scale is also a strong shrinkage of that scale (and WD enters the meta trace h).
Common practice ("no bias decay", He et al. arXiv:1812.01187) excludes norm scales from weight decay.  Two questions
need the mask: does the scalar collapse need WD on norm scales (cwd1, ResNet18_c100), and does the held large step on
PlainNet's layer4.1.bn2.weight do its damage through WD (cwd2)?  With DECAY_MASK on, for a masked tensor i:

    delta = a * m;            w <- w - delta;   h <- gamma*h - delta              (wd_i = 0 in BOTH places)

and every unmasked tensor runs the original two lines VERBATIM.  The momentum update is unchanged for every tensor.

-------------------------------------------------------------------------------
THE SWITCH
-------------------------------------------------------------------------------
    DECAY_MASK=normscale | <name>(+<name>)*          (environment variable)

  * UNSET or EMPTY -> OFF.  OFF is the state every prior run was in (the parent tree, bitwise).
  * normscale  = every 1-D parameter whose name ends in `.weight` (on the campaign's CIFAR nets these are exactly the
                 BatchNorm2d / GroupNorm scales; the launcher proves it against module types on the LIVE model).
  * <name>     = an exact parameter name of the LIVE model; `+` joins several; no duplicates.
  * Requires --alg-base SGDm and a base weight decay > 0 (a mask on WD 0 is vacuous).  Independent of the grouping and
    of every hold patch (VOTE_W / BETA_HOLD / GROUP_HOLD / REST_HOLD / COMP_HOLD / WINDOW_HOLD may be on or off).
  * LOUD, never silently partial: a malformed value, an unknown name, a duplicate, an empty `+` token, a base algorithm
    other than SGDm, or base weight decay <= 0 -> ValueError at construction.  Every character of a value is in
    [A-Za-z0-9_.+], so DECAY_MASK=... is ONE token of sbatch's comma-separated --export list.
  * A witness line is printed at construction on EVERY run (the runner's ENV line cannot carry DECAY_MASK):
        DECAY_MASK: off
        DECAY_MASK: on base=SGDm wd=<wd> spec=<spec> masked=<k> of=<T> numel=<N> idx=<i1,i2,...> names=<n1,n2,...>
    (idx 1-based, in parameter order).  The prefix starts with D, which none of VOTE_W / BETA_HOLD / GROUP_HOLD /
    COMP_HOLD / REST_HOLD / WINDOW_HOLD does, so no `startswith` reader of those kinds selects a DECAY_MASK line.

-------------------------------------------------------------------------------
HOW, AND WHAT IS NOT TOUCHED
-------------------------------------------------------------------------------
HF.__init__ binds `self.base_update = self.SGDm_base_update` BEFORE it calls init_meta.  When on, `_dm_init` (called in
init_meta) REBINDS `self.base_update` to `self._dm_SGDm_base_update`, a copy of SGDm_base_update whose unmasked branch is
the original two lines verbatim.  With an all-False mask the copy is BITWISE SGDm_base_update (tests/test_decaymask.py
D3).  The meta update, the clamp, every hold patch, PROBE and PROBE_TENSOR are untouched: on a forced (held) grouping the
applied step sizes are exactly the unmasked arm's.

-------------------------------------------------------------------------------
INERTNESS
-------------------------------------------------------------------------------
THREE insertions, every one ADDITIVE; NO existing line is edited:
  1. ONE call, `self._dm_init(net_param_names_and_size)`, in init_meta right before
     `self.trace_meta = [0.0 for _ in  range(self.len_beta_list)]` (i.e. after every hold patch's init call);
  2. a guarded `self._dm_attach(rec)` in _probe, right before `self._pt_attach(rec)  # PATCH_PROBE_TENSOR`;
  3. the four new methods (`_dm_init`, `_dm_resolve`, `_dm_SGDm_base_update`, `_dm_attach`) inserted as one block right
     before `def check_required_attributes`.
With DECAY_MASK unset or empty, `_dm_on` is False, (2) does nothing, (1) prints `DECAY_MASK: off` and returns: no
attribute but `_dm_on` is set, base_update is not rebound, the probe record gains no key.

The probe record of a DECAY_MASK run gains keys (appended before PROBE_TENSOR's; no existing key touched):
  dm_n        base updates completed so far by the masked update
  dm_masked   k, the number of masked tensors
  dm_skipped  cumulative number of (update, masked tensor) pairs that ran with wd 0 (must equal dm_n * k)
  dm_wdterm   || wd * a_i * w_i || over the masked tensors at the probe (the size of the WD term NOT applied; > 0)
  dm_norm     [||w_i||] per masked tensor, in idx order (descriptive: the Zhou et al. / WD-shrink readout)
  dm_absmin   min |w| over the masked tensors (descriptive)
  dm_small    number of masked entries with |w| < 1e-3 (descriptive)
"""
import os, sys

P = os.environ.get("HF_PATH", "")
if not P:
    print("!!! HF_PATH is required: this patch is applied ONLY to an isolated tree's HF.py")
    sys.exit(2)
src = open(P).read()

if "PATCH_DECAYMASK" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# --- 0. the assumptions this patch rests on, checked BEFORE anything is written ----
assert "_dm_" not in src, "unexpected pre-existing _dm_ names"
assert "DECAY_MASK" not in src, "unexpected pre-existing DECAY_MASK handling"
assert "PATCH_BETAHOLD" in src and "PATCH_VOTEWEIGHT" in src and "PATCH_PROBE_TENSOR" in src, \
    "the base must carry PATCH_VOTEWEIGHT, PATCH_BETAHOLD and PATCH_PROBE_TENSOR (cvt8's or cvt9's HF.py)"
assert ("PATCH_RESTHOLD" in src) != ("PATCH_WINDOWHOLD" in src), "the base must be cvt8's (RESTHOLD) or cvt9's (WINDOWHOLD) HF.py"
orig = ("    def SGDm_base_update(self,net,g):\n"
        "        #Base update\n"
        "        for w, grad, a ,i in zip(net.parameters(), g, self.alpha, range(self.num_layers)):\n"
        "            delta = a * (self.momentum_base[i] + self.args_base['weight_decay']*w.data)\n"
        "            w.data = w.data - delta\n"
        "            self.momentum_base[i] = self.args_base['momentum_param']*self.momentum_base[i] + (1 - self.args_base['momentum_param'])*grad\n"
        "            self.h_condenced[i] = self.gamma*(1-self.args_base['weight_decay']*a)*self.h_condenced[i] - delta\n")
assert src.count(orig) == 1, "SGDm_base_update is not the registered text -- refusing to patch"
assert src.count("            self.base_update = self.SGDm_base_update \n") == 1, "the SGDm binding in __init__ moved"

# --- 1. the call in init_meta, right before trace_meta ------------------------------
a1 = "        self.trace_meta = [0.0 for _ in  range(self.len_beta_list)]\n"
assert src.count(a1) == 1, "anchor 1 count=%d -- refusing to patch" % src.count(a1)
n1 = "        self._dm_init(net_param_names_and_size)  # PATCH_DECAYMASK\n" + a1

# --- 2. the attach in _probe, right before PATCH_PROBE_TENSOR's ----------------------
a2 = "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n"
assert src.count(a2) == 1, "anchor 2 count=%d -- refusing to patch" % src.count(a2)
n2 = ("        # --- PATCH_DECAYMASK ---\n"
      "        if getattr(self, '_dm_on', False):\n"
      "            self._dm_attach(rec)\n"
      "        # --- end PATCH_DECAYMASK ---\n") + a2

# --- 3. the methods, right before check_required_attributes --------------------------
a3 = "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n"
assert src.count(a3) == 1, "anchor 3 count=%d -- refusing to patch" % src.count(a3)
n3 = (
    "    # ------------------------------------------------------ PATCH_DECAYMASK\n"
    "    def _dm_init(self, net_param_names_and_size):\n"
    "        \"\"\"Parse DECAY_MASK, check every premise loudly, rebind base_update, print the witness.\"\"\"\n"
    "        import os as _os\n"
    "        self._dm_on = False\n"
    "        raw = _os.environ.get('DECAY_MASK', '')\n"
    "        if raw == '':\n"
    "            print('DECAY_MASK: off', flush=True)\n"
    "            return\n"
    "        names = [n for (n, _s) in net_param_names_and_size]\n"
    "        idx = self._dm_resolve(raw, net_param_names_and_size)\n"
    "        _ab = getattr(self, 'args_base', None) or {}\n"
    "        if _ab.get('alg') != 'SGDm':\n"
    "            raise ValueError('PATCH_DECAYMASK: needs --alg-base SGDm, got %r' % (_ab.get('alg'),))\n"
    "        _wd = _ab.get('weight_decay')\n"
    "        if _wd is None or not (float(_wd) > 0.0):\n"
    "            raise ValueError('PATCH_DECAYMASK: base weight decay %r is not > 0; a mask would be vacuous' % (_wd,))\n"
    "        self._dm_mask = [i in idx for i in range(len(names))]\n"
    "        self._dm_idx = sorted(idx)\n"
    "        self._dm_n = 0\n"
    "        self._dm_skipped = 0\n"
    "        self._dm_params = None\n"
    "        _numel = 0\n"
    "        for i in self._dm_idx:\n"
    "            _q = 1\n"
    "            for _d in net_param_names_and_size[i][1]:\n"
    "                _q *= int(_d)\n"
    "            _numel += _q\n"
    "        desc = ('DECAY_MASK: on base=SGDm wd=%r spec=%s masked=%d of=%d numel=%d idx=%s names=%s'\n"
    "                % (_wd, raw, len(self._dm_idx), len(names), _numel, ','.join('%d' % (i + 1) for i in self._dm_idx),\n"
    "                   ','.join(names[i] for i in self._dm_idx)))\n"
    "        self.base_update = self._dm_SGDm_base_update\n"
    "        self._dm_on = True\n"
    "        print(desc, flush=True)\n"
    "\n"
    "    def _dm_resolve(self, raw, net_param_names_and_size):\n"
    "        \"\"\"-> set of 0-based tensor indices.  normscale = every 1-D `*.weight`; else exact names joined by +.\"\"\"\n"
    "        import re as _re\n"
    "        names = [n for (n, _s) in net_param_names_and_size]\n"
    "        if not _re.match(r'^[A-Za-z0-9_.+]+$', raw):\n"
    "            raise ValueError('PATCH_DECAYMASK: DECAY_MASK %r has a character outside [A-Za-z0-9_.+]' % raw)\n"
    "        if raw == 'normscale':\n"
    "            idx = set(i for i, (n, s) in enumerate(net_param_names_and_size) if n.endswith('.weight') and len(s) == 1)\n"
    "            if not idx:\n"
    "                raise ValueError('PATCH_DECAYMASK: normscale matched no tensor')\n"
    "            return idx\n"
    "        toks = raw.split('+')\n"
    "        if any(t == '' for t in toks):\n"
    "            raise ValueError('PATCH_DECAYMASK: DECAY_MASK %r has an empty + token' % raw)\n"
    "        if len(set(toks)) != len(toks):\n"
    "            raise ValueError('PATCH_DECAYMASK: DECAY_MASK %r names a tensor twice' % raw)\n"
    "        idx = set()\n"
    "        for t in toks:\n"
    "            if t not in names:\n"
    "                raise ValueError('PATCH_DECAYMASK: %r is not a parameter name of this model' % t)\n"
    "            idx.add(names.index(t))\n"
    "        return idx\n"
    "\n"
    "    def _dm_SGDm_base_update(self,net,g):\n"
    "        \"\"\"SGDm_base_update with the coupled weight decay set to 0 on the masked tensors (update AND trace).\"\"\"\n"
    "        if self._dm_params is None:\n"
    "            self._dm_params = [p for p, _m in zip(net.parameters(), self._dm_mask) if _m]\n"
    "        for w, grad, a ,i in zip(net.parameters(), g, self.alpha, range(self.num_layers)):\n"
    "            if self._dm_mask[i]:\n"
    "                delta = a * self.momentum_base[i]\n"
    "                w.data = w.data - delta\n"
    "                self.momentum_base[i] = self.args_base['momentum_param']*self.momentum_base[i] + (1 - self.args_base['momentum_param'])*grad\n"
    "                self.h_condenced[i] = self.gamma*self.h_condenced[i] - delta\n"
    "                self._dm_skipped += 1\n"
    "                continue\n"
    "            delta = a * (self.momentum_base[i] + self.args_base['weight_decay']*w.data)\n"
    "            w.data = w.data - delta\n"
    "            self.momentum_base[i] = self.args_base['momentum_param']*self.momentum_base[i] + (1 - self.args_base['momentum_param'])*grad\n"
    "            self.h_condenced[i] = self.gamma*(1-self.args_base['weight_decay']*a)*self.h_condenced[i] - delta\n"
    "        self._dm_n += 1\n"
    "\n"
    "    def _dm_attach(self, rec):\n"
    "        \"\"\"APPENDS keys to the probe record; never touches an existing key.\"\"\"\n"
    "        rec['dm_n'] = int(self._dm_n)\n"
    "        rec['dm_masked'] = len(self._dm_idx)\n"
    "        rec['dm_skipped'] = int(self._dm_skipped)\n"
    "        if self._dm_params is None:\n"
    "            rec['dm_wdterm'] = None\n"
    "            rec['dm_norm'] = None\n"
    "            rec['dm_absmin'] = None\n"
    "            rec['dm_small'] = None\n"
    "            return\n"
    "        _wd = float(self.args_base['weight_decay'])\n"
    "        _ss = 0.0\n"
    "        _norm, _mins, _small = [], [], 0\n"
    "        for p, i in zip(self._dm_params, self._dm_idx):\n"
    "            _w = p.data\n"
    "            _ss += float(((_wd * self.alpha[i]) * _w).float().pow(2).sum().item())\n"
    "            _norm.append(float(_w.float().norm().item()))\n"
    "            _mins.append(float(_w.abs().min().item()))\n"
    "            _small += int((_w.abs() < 1e-3).sum().item())\n"
    "        rec['dm_wdterm'] = _ss ** 0.5\n"
    "        rec['dm_norm'] = _norm\n"
    "        rec['dm_absmin'] = min(_mins)\n"
    "        rec['dm_small'] = _small\n"
    "    # -------------------------------------------------- end PATCH_DECAYMASK\n"
    "\n") + a3

src2 = src.replace(a1, n1, 1).replace(a2, n2, 1).replace(a3, n3, 1)
bak = P + ".pre_decaymask"
if os.path.exists(bak):
    if open(bak).read() != src:
        print("!!! %s exists and differs from the file being patched -- refusing" % bak)
        sys.exit(2)
else:
    open(bak, "w").write(src)
open(P, "w").write(src2)
print("PATCHED PATCH_DECAYMASK ->", P, "(backup", bak + ")")
