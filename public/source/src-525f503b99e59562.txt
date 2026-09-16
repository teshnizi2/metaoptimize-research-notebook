"""PATCH_BETAHOLD -- an OPT-IN, ENVIRONMENT-CONTROLLED override of ONE step-size group's beta (its log step
size), applied after every meta update, while every other part of the meta-update runs unchanged.  Written for
`cvt4` (CORRECTIONS 237).  Applied ONLY to the isolated tree $METAOPT_WS/harness_cvt4/cifar10
(bin/cVT4_stage_harness.sh), ON TOP OF cvt1's PATCH_VOTEWEIGHT HF.py (sha 3f2b98e1...); the live shared
harness and cvt1's tree are never patched.

-------------------------------------------------------------------------------
WHY
-------------------------------------------------------------------------------
On PlainNet18_c100, isolating layer4.1.bn2.weight (idx 50) on its own step size rescues the scalar collapse
(HEAD ~64.5 vs k01 ~11.5); silencing its vote while it SHARES the step size does not (cvt1 MUTE, cvt3 MUTE50).
Isolation does two things at once: it takes 50's vote out of the others' sum AND it takes 50 off the others'
(large) step size.  236.7 read the records (descriptive, UNSURE) as pointing at 50's OWN LARGE STEP SIZE.  No
arm has varied 50's step size with everything else held.  This patch holds it directly, in HEAD's grouping
(so 50's vote stays out of the complement's sum):

    after Lion_meta_update and PATCH_CLIP on update n:   beta[0][g] := hold(n)     (every other group untouched)

-------------------------------------------------------------------------------
THE SWITCH
-------------------------------------------------------------------------------
    BETA_HOLD=<name>:floor | <name>:shared | <name>:tri:<P>        (environment variable)

  * UNSET or EMPTY -> OFF.  OFF is the state every prior run was in.
  * <name> is an exact parameter name of the LIVE model, resolved by NAME; it must sit ALONE in its group of a
    `blockwise` (sets:) grouping -- the hold is on that group's beta, and a group with other members would hold
    them too.
  * floor       hold(n) = the clamp's lower end (BETA_CLIP lo, -15) for every n, INCLUDING n = 0 (init).
  * shared      hold(n) = the OTHER group's beta as it stands after update n and its clamp (requires exactly 2
                groups), and at init.  Bitwise copy of a float32.
  * tri:<P>     a max-rate triangle, the shape of every landed k01 / MUTE shared beta and HEAD's isolated beta
                (analysis/cvt4_registration_derivations.py S1):
                    u(0) = 0;  u(n) = min(n-1, P) - max(0, n-1-P)
                    hold(n) = min(hi, max(lo, b0 + ms*u(n)))   (python double, stored into float32)
                b0 = the group's own initial beta (float32 log alpha0), ms = --meta-stepsize.  n-1 because the
                first meta update of every run is a no-op (h = 0 => z = 0 => sign 0).
  * LOUD, never silently partial: a malformed value, an unknown name, a name not alone in its group, a
    stepsize_type other than blockwise, `shared` without exactly 2 groups, a meta algorithm other than Lion,
    BETA_CLIP unset, HIER other than ''/'none', a tn: (PATCH_REDNORM) spec, or VOTE_W on at the same time
    -> ValueError at construction.  Every character of a value is in [A-Za-z0-9_.:], so BETA_HOLD=... is ONE
    token of sbatch's comma-separated --export list.
  * A witness line is printed at construction on EVERY run (the runner's ENV line cannot carry BETA_HOLD):
        BETA_HOLD: off
        BETA_HOLD: on type=blockwise group=<g> groupsize=1 name=<name> mode=floor value=<lo>
        BETA_HOLD: on type=blockwise group=<g> groupsize=1 name=<name> mode=shared source=<s> sourcesize=<n>
        BETA_HOLD: on type=blockwise group=<g> groupsize=1 name=<name> mode=tri P=<P> b0=<b0> ms=<ms> lo=<lo> hi=<hi> peak=<v>

-------------------------------------------------------------------------------
WHAT THE OVERRIDE DOES TO THE META-OPTIMISER'S STATE (Lion momentum) -- stated exactly
-------------------------------------------------------------------------------
Lion_meta_update runs UNCHANGED on the whole beta vector: beta[i] <- beta[i] - ms*sign(b2*mom[i] + (1-b2)*z[i])
elementwise, then mom[i] <- mom_param*mom[i] + (1-mom_param)*z[i], then PATCH_CLIP clamps elementwise.  Only
AFTER that is beta[0][g] overwritten.  So:
  * the other groups' beta and momentum are bitwise what they would have been given the same z (elementwise
    ops; nothing reads group g's entries);
  * group g's momentum_meta entry keeps being updated from group g's own z (50's raw term) exactly as the
    unpatched harness would -- it is NOT reset, frozen or copied from the source group;
  * that momentum is DEAD STATE: the only thing that ever reads it is group g's own Lion step, whose output is
    overwritten on the same update, and the hold is applied from init (n = 0) through the last update with no
    release.  It therefore never reaches an applied beta.  (A hold that were released mid-run would inherit a
    stale momentum; this patch has no release.)
  * PROBE_TENSOR's `beta_pre` for group g is the previous update's HELD value, its `mom_pre` / `z_agg` are the
    harness's own (dead) momentum and the raw z -- which is what lets the scorer recompute, per record, the
    value Lion WROTE before the override and check it against `bh_nat`.

-------------------------------------------------------------------------------
INERTNESS
-------------------------------------------------------------------------------
FOUR insertions, every one ADDITIVE; NO existing line is edited:
  1. a guarded `self._bh_apply()` in step(), after PATCH_CLIP's clamp and before `self._probe(...)`;
  2. ONE call, `self._bh_init(net_param_names_and_size)`, right after PATCH_VOTEWEIGHT's `_vw_init` call in
     init_meta;
  3. a guarded `self._bh_attach(rec)` in _probe, right before `self._pt_attach(rec)`;
  4. the six new methods (`_bh_init`, `_bh_value`, `_bh_set`, `_bh_apply`, `_bh_attach`, `_bh_witness_peak`)
     inserted as one block before `check_required_attributes`.
With BETA_HOLD unset or empty, `_bh_on` is False, (1) and (3) do nothing, (2) prints `BETA_HOLD: off` and returns:
no tensor is touched, the probe record gains no key.  tests/test_betahold.py proves the structure (deleting the
four regions reproduces the pre-patch file byte for byte); tests/test_betahold_realrun.py proves beta at every
step, the loss, probe.jsonl bytes and every final parameter and buffer BITWISE against cvt1's unpatched-by-this-
patch tree over a short real CIFAR-100 PlainNet run, and that each registered hold string bites.

The probe record of a HELD run gains four keys (appended; no existing key touched):
  bh_n       meta updates completed so far (must equal step + 2)
  bh_active  cumulative number of updates on which the held value differed from the value Lion + clamp wrote
  bh_nat     the value Lion + clamp wrote for group g on THIS update, before the override
  bh_held    the value group g holds after the override (must equal the record's beta[g])
"""
import os, sys

P = os.environ.get("HF_PATH", "")
if not P:
    print("!!! HF_PATH is required: this patch is applied ONLY to an isolated tree's HF.py")
    sys.exit(2)
src = open(P).read()

if "PATCH_BETAHOLD" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# --- 0. the assumptions this patch rests on, checked BEFORE anything is written ----
assert "PATCH_VOTEWEIGHT" in src, "PATCH_BETAHOLD is applied on top of PATCH_VOTEWEIGHT (cvt1's HF.py)"
assert "def Lion_meta_update(self,HtT_gradft):" in src, "Lion_meta_update is missing"
assert "_bh_" not in src, "unexpected pre-existing _bh_ names"
assert "BETA_HOLD" not in src, "unexpected pre-existing BETA_HOLD handling"

# --- 1. the apply, after the clamp and before the probe -----------------------------
a1 = ("            if self._beta_lo is not None:  # PATCH_CLIP\n"
      "                for _i in range(self.len_beta_list):\n"
      "                    self.beta[_i] = self.beta[_i].clamp(self._beta_lo, self._beta_hi)\n"
      "            self._probe(HtT_gradft)  # PATCH_PROBE\n")
assert src.count(a1) == 1, "anchor 1 count=%d -- refusing to patch" % src.count(a1)
n1 = ("            if self._beta_lo is not None:  # PATCH_CLIP\n"
      "                for _i in range(self.len_beta_list):\n"
      "                    self.beta[_i] = self.beta[_i].clamp(self._beta_lo, self._beta_hi)\n"
      "            # --- PATCH_BETAHOLD ---\n"
      "            if getattr(self, '_bh_on', False):\n"
      "                self._bh_apply()\n"
      "            # --- end PATCH_BETAHOLD ---\n"
      "            self._probe(HtT_gradft)  # PATCH_PROBE\n")

# --- 2. the call in init_meta, after PATCH_VOTEWEIGHT's ----------------------------
a2 = "        self._vw_init(net_param_names_and_size)  # PATCH_VOTEWEIGHT\n"
assert src.count(a2) == 1, "anchor 2 count=%d -- refusing to patch" % src.count(a2)
n2 = a2 + "        self._bh_init(net_param_names_and_size)  # PATCH_BETAHOLD\n"

# --- 3. the attach in _probe, before PATCH_PROBE_TENSOR's ---------------------------
a3 = "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n"
assert src.count(a3) == 1, "anchor 3 count=%d -- refusing to patch" % src.count(a3)
n3 = ("        # --- PATCH_BETAHOLD ---\n"
      "        if getattr(self, '_bh_on', False):\n"
      "            self._bh_attach(rec)\n"
      "        # --- end PATCH_BETAHOLD ---\n"
      + a3)

# --- 4. the methods, before check_required_attributes -------------------------------
a4 = "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n"
assert src.count(a4) == 1, "anchor 4 count=%d -- refusing to patch" % src.count(a4)
n4 = (
    "    # ------------------------------------------------------ PATCH_BETAHOLD\n"
    "    def _bh_init(self, net_param_names_and_size):\n"
    "        \"\"\"Parse BETA_HOLD, check every premise loudly, apply hold(0), print the witness.\"\"\"\n"
    "        import os as _os, re as _re\n"
    "        self._bh_on = False\n"
    "        raw = _os.environ.get('BETA_HOLD', '')\n"
    "        if raw == '':\n"
    "            print('BETA_HOLD: off', flush=True)\n"
    "            return\n"
    "        m = _re.match(r'^([A-Za-z0-9_.]+):(floor|shared|tri:([0-9]{1,6}))$', raw)\n"
    "        if not m:\n"
    "            raise ValueError('PATCH_BETAHOLD: BETA_HOLD %r is not <name>:floor | <name>:shared | <name>:tri:<P>' % raw)\n"
    "        nm, mode = m.group(1), m.group(2)\n"
    "        names = [n for (n, _s) in net_param_names_and_size]\n"
    "        if nm not in names:\n"
    "            raise ValueError('PATCH_BETAHOLD: %r is not a parameter name of this model' % nm)\n"
    "        if self.stepsize_type != 'blockwise':\n"
    "            raise ValueError('PATCH_BETAHOLD: needs a blockwise (sets:) grouping; stepsize_type is %r'\n"
    "                             % (self.stepsize_type,))\n"
    "        if getattr(self, '_rednorm', False):\n"
    "            raise ValueError('PATCH_BETAHOLD: cannot be combined with a tn: (PATCH_REDNORM) spec')\n"
    "        if getattr(self, '_vw_on', False):\n"
    "            raise ValueError('PATCH_BETAHOLD: cannot be combined with VOTE_W')\n"
    "        if getattr(self, '_hier', '') not in ('', 'none'):\n"
    "            raise ValueError('PATCH_BETAHOLD: cannot be combined with HIER=%r' % (self._hier,))\n"
    "        _am = getattr(self, 'args_meta', None) or {}\n"
    "        if _am.get('alg') != 'Lion':\n"
    "            raise ValueError('PATCH_BETAHOLD: the meta algorithm must be Lion; got %r' % (_am.get('alg'),))\n"
    "        if getattr(self, '_beta_lo', None) is None:\n"
    "            raise ValueError('PATCH_BETAHOLD: needs BETA_CLIP (the floor and the triangle live inside the clamp)')\n"
    "        i = names.index(nm)\n"
    "        k = [j for j, grp in enumerate(self.param_groups_indices) if i in grp]\n"
    "        if len(k) != 1 or len(self.param_groups_indices[k[0]]) != 1:\n"
    "            raise ValueError('PATCH_BETAHOLD: %r must sit ALONE in exactly one group' % nm)\n"
    "        self._bh_g = k[0]\n"
    "        self._bh_n = 0\n"
    "        self._bh_active = 0\n"
    "        self._bh_nat = None\n"
    "        self._bh_ms = float(_am['meta_stepsize'])\n"
    "        self._bh_lo, self._bh_hi = float(self._beta_lo), float(self._beta_hi)\n"
    "        self._bh_b0 = float(self.beta[0][self._bh_g].item())\n"
    "        self._bh_src = None\n"
    "        self._bh_P = None\n"
    "        desc = 'BETA_HOLD: on type=blockwise group=%d groupsize=1 name=%s' % (self._bh_g, nm)\n"
    "        if mode == 'floor':\n"
    "            self._bh_mode = 'floor'\n"
    "            desc += ' mode=floor value=%r' % (self._bh_lo,)\n"
    "        elif mode == 'shared':\n"
    "            if self.num_blocks != 2:\n"
    "                raise ValueError('PATCH_BETAHOLD: shared needs exactly 2 groups; there are %d' % self.num_blocks)\n"
    "            self._bh_mode = 'shared'\n"
    "            self._bh_src = 1 - self._bh_g\n"
    "            desc += ' mode=shared source=%d sourcesize=%d' % (self._bh_src, len(self.param_groups_indices[self._bh_src]))\n"
    "        else:\n"
    "            self._bh_mode = 'tri'\n"
    "            self._bh_P = int(m.group(3))\n"
    "            if self._bh_P < 1:\n"
    "                raise ValueError('PATCH_BETAHOLD: tri needs P >= 1')\n"
    "            desc += (' mode=tri P=%d b0=%r ms=%r lo=%r hi=%r peak=%r'\n"
    "                     % (self._bh_P, self._bh_b0, self._bh_ms, self._bh_lo, self._bh_hi, self._bh_witness_peak()))\n"
    "        self._bh_on = True\n"
    "        self._bh_set(self._bh_value(0))\n"
    "        print(desc, flush=True)\n"
    "\n"
    "    def _bh_value(self, n):\n"
    "        \"\"\"hold(n) as a python float (None for shared: the source group's float32 is copied).\"\"\"\n"
    "        if self._bh_mode == 'floor':\n"
    "            return self._bh_lo\n"
    "        if self._bh_mode == 'shared':\n"
    "            return None\n"
    "        u = 0 if n <= 0 else min(n - 1, self._bh_P) - max(0, n - 1 - self._bh_P)\n"
    "        return min(self._bh_hi, max(self._bh_lo, self._bh_b0 + self._bh_ms * u))\n"
    "\n"
    "    def _bh_witness_peak(self):\n"
    "        return min(self._bh_hi, max(self._bh_lo, self._bh_b0 + self._bh_ms * self._bh_P))\n"
    "\n"
    "    def _bh_set(self, value):\n"
    "        _b = self.beta[0].clone()\n"
    "        if self._bh_mode == 'shared':\n"
    "            _b[self._bh_g] = _b[self._bh_src]\n"
    "        else:\n"
    "            _b[self._bh_g] = value\n"
    "        self.beta[0] = _b\n"
    "\n"
    "    def _bh_apply(self):\n"
    "        \"\"\"After update n's Lion step and clamp: record what Lion wrote, then hold.\"\"\"\n"
    "        self._bh_n += 1\n"
    "        _nat = float(self.beta[0][self._bh_g].item())\n"
    "        self._bh_set(self._bh_value(self._bh_n))\n"
    "        _held = float(self.beta[0][self._bh_g].item())\n"
    "        self._bh_nat = _nat\n"
    "        if _held != _nat:\n"
    "            self._bh_active += 1\n"
    "\n"
    "    def _bh_attach(self, rec):\n"
    "        \"\"\"APPENDS keys to the probe record; never touches an existing key.\"\"\"\n"
    "        rec['bh_n'] = int(self._bh_n)\n"
    "        rec['bh_active'] = int(self._bh_active)\n"
    "        rec['bh_nat'] = self._bh_nat\n"
    "        rec['bh_held'] = float(self.beta[0][self._bh_g].item())\n"
    "    # -------------------------------------------------- end PATCH_BETAHOLD\n"
    "\n"
    + a4)

src2 = src.replace(a1, n1, 1).replace(a2, n2, 1).replace(a3, n3, 1).replace(a4, n4, 1)
bak = P + ".pre_betahold"
if os.path.exists(bak):
    if open(bak).read() != src:
        print("!!! %s exists and differs from the file being patched -- refusing" % bak)
        sys.exit(2)
else:
    open(bak, "w").write(src)
open(P, "w").write(src2)
print("PATCHED PATCH_BETAHOLD ->", P, "(backup", bak + ")")
