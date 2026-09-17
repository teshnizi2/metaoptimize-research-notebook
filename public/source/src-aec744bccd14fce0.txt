"""PATCH_WINDOWHOLD -- an OPT-IN, ENVIRONMENT-CONTROLLED TIME WINDOW on PATCH_BETAHOLD's `tri:<P>` hold: inside the
window [n0, n1) the singleton group keeps the triangle value BETA_HOLD wrote; outside it the group is held at the clamp
floor.  Written for `cvt9` (CORRECTIONS 249).  Applied ONLY to the isolated tree $METAOPT_WS/harness_cvt9/cifar10
(bin/cVT9_stage_harness.sh), ON TOP OF cvt6's PATCH_COMPHOLD HF.py (sha f1b9c8aa...); the live shared harness and
cvt1's, cvt4's, cvt6's and cvt7's trees are never patched.

-------------------------------------------------------------------------------
WHY
-------------------------------------------------------------------------------
cvt4 (240) and cvt6 (246) put layer4.1.bn2.weight (idx 50) on MUTE's replayed large step size (tri:9428) from init to
the end and found the arm at the k01 location, with the complement free (HOLDHIGH) and with the complement forced
open-loop onto HEAD's measured path (HIGHHEADPATH).  Neither batch had a time gate (240.4(2), 246.4(4)): WHICH PART of
50's large trajectory does the damage is untested.  This patch cuts the same triangle into windows:

    after Lion, PATCH_CLIP, PATCH_BETAHOLD and PATCH_COMPHOLD on update n:
        beta[0][g] := tri_P(n)   if n0 <= n < n1          (the value PATCH_BETAHOLD just wrote; nothing is changed)
        beta[0][g] := lo (-15)   otherwise                 (the clamp floor, as PATCH_BETAHOLD's `floor` mode)

-------------------------------------------------------------------------------
THE SWITCH
-------------------------------------------------------------------------------
    WINDOW_HOLD=<n0>:<n1> | <n0>:end        (environment variable; n0, n1 decimal integers, 1-8 digits)

  * UNSET or EMPTY -> OFF.  OFF is the state every prior run was in (cvt6's tree, bitwise).
  * REQUIRES BETA_HOLD ON in mode `tri` (PATCH_BETAHOLD's premises -- a singleton group of a blockwise grouping, Lion,
    BETA_CLIP, no VOTE_W, no HIER, no tn: -- are therefore already enforced).  COMP_HOLD is independent: it may be on
    or off.  g = BETA_HOLD's group.
  * n0 < n1 (`end` = no upper bound).  Update n = the number of completed meta updates; n = 0 is init.
  * LOUD, never silently partial: a malformed value, BETA_HOLD off, BETA_HOLD in mode floor or shared, n0 >= n1 ->
    ValueError at construction.  Every character of a value is in [A-Za-z0-9_.:], so WINDOW_HOLD=... is ONE token of
    sbatch's comma-separated --export list.
  * A witness line is printed at construction on EVERY run (the runner's ENV line cannot carry WINDOW_HOLD):
        WINDOW_HOLD: off
        WINDOW_HOLD: on type=blockwise group=<g> name=<name> base=tri P=<P> n0=<n0> n1=<n1|end> outside=floor value=<lo>
    Its prefix starts with W, which none of VOTE_W / BETA_HOLD / GROUP_HOLD / COMP_HOLD does, so no `startswith`
    reader of those kinds (analysis/corpus_exclusions.py's KINDS, the registered scorers) selects a WINDOW_HOLD line.

-------------------------------------------------------------------------------
WHAT THE OVERRIDE DOES TO THE META-OPTIMISER'S STATE (Lion momentum) -- stated exactly
-------------------------------------------------------------------------------
Lion_meta_update runs UNCHANGED on the whole beta vector; PATCH_CLIP clamps; PATCH_BETAHOLD overwrites beta[0][g] with
tri_P(n); PATCH_COMPHOLD (if on) overwrites the other group; only AFTER that does this patch overwrite beta[0][g] again,
and only outside the window.  So group g's momentum_meta entry keeps the unpatched recursion on 50's own raw z and is
DEAD STATE exactly as under PATCH_BETAHOLD (its only reader is group g's own Lion step, overwritten on the same update,
from init to the last update, no release).  The window changes WHICH exogenous value group g holds, never whether an
applied beta reads z or momentum.  PROBE_TENSOR's `beta_pre` for group g is the previous update's WINDOWED value.

-------------------------------------------------------------------------------
INERTNESS
-------------------------------------------------------------------------------
FOUR insertions, every one ADDITIVE; NO existing line is edited:
  1. a guarded `self._wh_apply()` in step(), right after PATCH_COMPHOLD's guarded `_ch_apply` region and before
     `self._probe(...)`;
  2. ONE call, `self._wh_init(net_param_names_and_size)`, right after PATCH_COMPHOLD's `_ch_init` call in init_meta;
  3. a guarded `self._wh_attach(rec)` in _probe, right after PATCH_COMPHOLD's guarded `_ch_attach` region and before
     `self._pt_attach(rec)`;
  4. the five new methods (`_wh_init`, `_wh_value`, `_wh_set`, `_wh_apply`, `_wh_attach`) inserted as one block after
     PATCH_COMPHOLD's method block, before `check_required_attributes`.
With WINDOW_HOLD unset or empty, `_wh_on` is False, (1) and (3) do nothing, (2) prints `WINDOW_HOLD: off` and returns:
no tensor is touched, the probe record gains no key.  tests/test_windowhold.py proves the structure (deleting the four
regions reproduces cvt6's HF.py byte for byte); tests/test_windowhold_realrun.py proves beta at every step, the loss,
probe.jsonl bytes and every final parameter and buffer BITWISE against cvt6's tree over a short real CIFAR-100 PlainNet
run, and that each registered window string bites.

The probe record of a WINDOW_HOLD run gains four keys (appended after PATCH_COMPHOLD's; no existing key touched):
  wh_n       meta updates completed so far (must equal step + 2)
  wh_active  cumulative number of updates on which the windowed value differed from the value PATCH_BETAHOLD wrote
  wh_nat     the value PATCH_BETAHOLD wrote for group g on THIS update (float32 tri_P(n)), before the window
  wh_held    the value group g holds after the window (must equal the record's beta[g])
PATCH_BETAHOLD's own bh_* keys are unchanged in meaning: bh_nat is still what Lion + clamp wrote, and bh_held (read at
probe time) is the windowed value.
"""
import os, sys

P = os.environ.get("HF_PATH", "")
if not P:
    print("!!! HF_PATH is required: this patch is applied ONLY to an isolated tree's HF.py")
    sys.exit(2)
src = open(P).read()

if "PATCH_WINDOWHOLD" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# --- 0. the assumptions this patch rests on, checked BEFORE anything is written ----
assert "PATCH_COMPHOLD" in src, "PATCH_WINDOWHOLD is applied on top of PATCH_COMPHOLD (cvt6's HF.py)"
assert "PATCH_BETAHOLD" in src and "PATCH_VOTEWEIGHT" in src, "the base must carry PATCH_BETAHOLD and PATCH_VOTEWEIGHT"
assert "def Lion_meta_update(self,HtT_gradft):" in src, "Lion_meta_update is missing"
assert "_wh_" not in src, "unexpected pre-existing _wh_ names"
assert "WINDOW_HOLD" not in src, "unexpected pre-existing WINDOW_HOLD handling"

# --- 1. the apply, after PATCH_COMPHOLD's and before the probe ----------------------
a1 = ("            # --- PATCH_COMPHOLD ---\n"
      "            if getattr(self, '_ch_on', False):\n"
      "                self._ch_apply()\n"
      "            # --- end PATCH_COMPHOLD ---\n"
      "            self._probe(HtT_gradft)  # PATCH_PROBE\n")
assert src.count(a1) == 1, "anchor 1 count=%d -- refusing to patch" % src.count(a1)
n1 = ("            # --- PATCH_COMPHOLD ---\n"
      "            if getattr(self, '_ch_on', False):\n"
      "                self._ch_apply()\n"
      "            # --- end PATCH_COMPHOLD ---\n"
      "            # --- PATCH_WINDOWHOLD ---\n"
      "            if getattr(self, '_wh_on', False):\n"
      "                self._wh_apply()\n"
      "            # --- end PATCH_WINDOWHOLD ---\n"
      "            self._probe(HtT_gradft)  # PATCH_PROBE\n")

# --- 2. the call in init_meta, after PATCH_COMPHOLD's -------------------------------
a2 = "        self._ch_init(net_param_names_and_size)  # PATCH_COMPHOLD\n"
assert src.count(a2) == 1, "anchor 2 count=%d -- refusing to patch" % src.count(a2)
n2 = a2 + "        self._wh_init(net_param_names_and_size)  # PATCH_WINDOWHOLD\n"

# --- 3. the attach in _probe, after PATCH_COMPHOLD's and before PATCH_PROBE_TENSOR's --
a3 = ("        # --- PATCH_COMPHOLD ---\n"
      "        if getattr(self, '_ch_on', False):\n"
      "            self._ch_attach(rec)\n"
      "        # --- end PATCH_COMPHOLD ---\n"
      "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")
assert src.count(a3) == 1, "anchor 3 count=%d -- refusing to patch" % src.count(a3)
n3 = ("        # --- PATCH_COMPHOLD ---\n"
      "        if getattr(self, '_ch_on', False):\n"
      "            self._ch_attach(rec)\n"
      "        # --- end PATCH_COMPHOLD ---\n"
      "        # --- PATCH_WINDOWHOLD ---\n"
      "        if getattr(self, '_wh_on', False):\n"
      "            self._wh_attach(rec)\n"
      "        # --- end PATCH_WINDOWHOLD ---\n"
      "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")

# --- 4. the methods, after PATCH_COMPHOLD's block, before check_required_attributes --
a4 = ("    # -------------------------------------------------- end PATCH_COMPHOLD\n"
      "\n"
      "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n")
assert src.count(a4) == 1, "anchor 4 count=%d -- refusing to patch" % src.count(a4)
n4 = (
    "    # -------------------------------------------------- end PATCH_COMPHOLD\n"
    "\n"
    "    # ------------------------------------------------------ PATCH_WINDOWHOLD\n"
    "    def _wh_init(self, net_param_names_and_size):\n"
    "        \"\"\"Parse WINDOW_HOLD, check every premise loudly, apply the window at n = 0, print the witness.\"\"\"\n"
    "        import os as _os, re as _re\n"
    "        self._wh_on = False\n"
    "        raw = _os.environ.get('WINDOW_HOLD', '')\n"
    "        if raw == '':\n"
    "            print('WINDOW_HOLD: off', flush=True)\n"
    "            return\n"
    "        m = _re.match(r'^([0-9]{1,8}):([0-9]{1,8}|end)$', raw)\n"
    "        if not m:\n"
    "            raise ValueError('PATCH_WINDOWHOLD: WINDOW_HOLD %r is not <n0>:<n1> | <n0>:end' % raw)\n"
    "        if not getattr(self, '_bh_on', False):\n"
    "            raise ValueError('PATCH_WINDOWHOLD: needs BETA_HOLD on (the window cuts its tri schedule)')\n"
    "        if self._bh_mode != 'tri':\n"
    "            raise ValueError('PATCH_WINDOWHOLD: BETA_HOLD mode %r is not tri' % (self._bh_mode,))\n"
    "        self._wh_n0 = int(m.group(1))\n"
    "        self._wh_n1 = None if m.group(2) == 'end' else int(m.group(2))\n"
    "        if self._wh_n1 is not None and self._wh_n0 >= self._wh_n1:\n"
    "            raise ValueError('PATCH_WINDOWHOLD: n0 %d must be < n1 %d' % (self._wh_n0, self._wh_n1))\n"
    "        self._wh_g = self._bh_g\n"
    "        self._wh_lo = self._bh_lo\n"
    "        self._wh_n = 0\n"
    "        self._wh_active = 0\n"
    "        self._wh_nat = None\n"
    "        names = [n for (n, _s) in net_param_names_and_size]\n"
    "        gname = names[self.param_groups_indices[self._wh_g][0]]\n"
    "        desc = ('WINDOW_HOLD: on type=blockwise group=%d name=%s base=tri P=%d n0=%d n1=%s outside=floor value=%r'\n"
    "                % (self._wh_g, gname, self._bh_P, self._wh_n0, 'end' if self._wh_n1 is None else '%d' % self._wh_n1,\n"
    "                   self._wh_lo))\n"
    "        self._wh_on = True\n"
    "        self._wh_set(0)\n"
    "        print(desc, flush=True)\n"
    "\n"
    "    def _wh_value(self, n):\n"
    "        \"\"\"the windowed schedule as a python float: PATCH_BETAHOLD's tri value inside [n0, n1), the floor outside.\"\"\"\n"
    "        if n >= self._wh_n0 and (self._wh_n1 is None or n < self._wh_n1):\n"
    "            return self._bh_value(n)\n"
    "        return self._wh_lo\n"
    "\n"
    "    def _wh_set(self, n):\n"
    "        _b = self.beta[0].clone()\n"
    "        _b[self._wh_g] = self._wh_value(n)\n"
    "        self.beta[0] = _b\n"
    "\n"
    "    def _wh_apply(self):\n"
    "        \"\"\"After update n's Lion step, clamp, PATCH_BETAHOLD and PATCH_COMPHOLD: record what PATCH_BETAHOLD wrote, then window.\"\"\"\n"
    "        self._wh_n += 1\n"
    "        _nat = float(self.beta[0][self._wh_g].item())\n"
    "        self._wh_set(self._wh_n)\n"
    "        _held = float(self.beta[0][self._wh_g].item())\n"
    "        self._wh_nat = _nat\n"
    "        if _held != _nat:\n"
    "            self._wh_active += 1\n"
    "\n"
    "    def _wh_attach(self, rec):\n"
    "        \"\"\"APPENDS keys to the probe record; never touches an existing key.\"\"\"\n"
    "        rec['wh_n'] = int(self._wh_n)\n"
    "        rec['wh_active'] = int(self._wh_active)\n"
    "        rec['wh_nat'] = self._wh_nat\n"
    "        rec['wh_held'] = float(self.beta[0][self._wh_g].item())\n"
    "    # -------------------------------------------------- end PATCH_WINDOWHOLD\n"
    "\n"
    "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n")

src2 = src.replace(a1, n1, 1).replace(a2, n2, 1).replace(a3, n3, 1).replace(a4, n4, 1)
bak = P + ".pre_windowhold"
if os.path.exists(bak):
    if open(bak).read() != src:
        print("!!! %s exists and differs from the file being patched -- refusing" % bak)
        sys.exit(2)
else:
    open(bak, "w").write(src)
open(P, "w").write(src2)
print("PATCHED PATCH_WINDOWHOLD ->", P, "(backup", bak + ")")
