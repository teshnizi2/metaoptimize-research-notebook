"""PATCH_GROUPHOLD -- an OPT-IN, ENVIRONMENT-CONTROLLED override of ONE step-size group's beta, where the group is
named by the COMPLETE SET of its member tensors (one or more), applied after every meta update, while every other
part of the meta-update runs unchanged.  Written for `cvt7` (CORRECTIONS 243).  Applied ONLY to the isolated tree
$METAOPT_WS/harness_cvt7/cifar10 (bin/cVT7_stage_harness.sh), ON TOP OF cvt4's PATCH_BETAHOLD HF.py (sha
84b345ad...), which is itself the LIVE ResNet harness's HF.py (4732b74a..., the tree `ciso1` / `cdep1` / `ciso2`
ran from) + PATCH_VOTEWEIGHT + PATCH_BETAHOLD.  The live shared harness, cvt1's and cvt4's trees are never patched.

-------------------------------------------------------------------------------
WHY A SECOND PATCH, AND NOT AN EDIT OF PATCH_BETAHOLD
-------------------------------------------------------------------------------
PATCH_BETAHOLD (237) requires the held tensor to sit ALONE in its group.  On ResNet18_c100 the isolation arm that
rescues (`ISO`, ciso1 / cdep1 / ciso2) isolates THREE carriers together in ONE group:
    sets:1-49,51-52,54-58,60-62/layer4.0.bn2.weight,layer4.0.shortcut.1.weight,layer4.1.bn2.weight   ([59,3])
so the hold has to act on that group.  Editing PATCH_BETAHOLD's inserted lines would make cvt4's registered HF.py
bytes a moving target; this patch is ADDITIVE on top of them (deleting its regions gives cvt4's HF.py back byte for
byte), and it uses its OWN switch, witness and probe keys, so a BETA_HOLD run and a GROUP_HOLD run can never be
confused by a reader.  The name deliberately does NOT start with `BETA_HOLD` or `VOTE_W`: the registered readers
(analysis/corpus_exclusions.py's witness_lines, cVT4's parse_run) select witness lines with startswith(<prefix>), so a
`BETA_HOLD_SET:` line would be misread as a BETA_HOLD witness.

-------------------------------------------------------------------------------
THE SWITCH
-------------------------------------------------------------------------------
    GROUP_HOLD=<name>[+<name>...]:floor | <name>[+<name>...]:tri:<P>        (environment variable)

  * UNSET or EMPTY -> OFF.  OFF is the state every prior run was in.
  * The names are exact parameter names of the LIVE model, resolved by NAME, written in MODEL ORDER (ascending
    index), each once; together they must be EXACTLY the membership of ONE group of a `blockwise` (sets:) grouping
    -- no member missing, no extra member -- so the hold is on that whole group's beta and on nothing else.
  * floor       hold(n) = the clamp's lower end (BETA_CLIP lo, -15) for every n, INCLUDING n = 0 (init).
  * tri:<P>     the max-rate triangle of PATCH_BETAHOLD, byte-for-byte the same formula:
                    u(0) = 0;  u(n) = min(n-1, P) - max(0, n-1-P)
                    hold(n) = min(hi, max(lo, b0 + ms*u(n)))   (python double, stored into float32)
                b0 = the group's own initial beta (float32 log alpha0), ms = --meta-stepsize.
  * LOUD, never silently partial: a malformed value, an unknown or repeated name, names out of model order, a set
    that is not exactly one group, a stepsize_type other than blockwise, a meta algorithm other than Lion, BETA_CLIP
    unset, HIER other than ''/'none', a tn: (PATCH_REDNORM) spec, VOTE_W on, or BETA_HOLD on at the same time
    -> ValueError at construction.  Every character of a value is in [A-Za-z0-9_.:+], no comma, so
    GROUP_HOLD=... is ONE token of sbatch's comma-separated --export list.
  * A witness line is printed at construction on EVERY run (the runner's ENV line cannot carry it):
        GROUP_HOLD: off
        GROUP_HOLD: on type=blockwise group=<g> groupsize=<k> names=<n1>+<n2>+... mode=floor value=<lo>
        GROUP_HOLD: on type=blockwise group=<g> groupsize=<k> names=<n1>+<n2>+... mode=tri P=<P> b0=<b0> ms=<ms> lo=<lo> hi=<hi> peak=<v>
    (The tree also prints `VOTE_W: off` and `BETA_HOLD: off` on every run.)

-------------------------------------------------------------------------------
THE META-OPTIMISER's STATE FOR THE HELD GROUP -- exactly PATCH_BETAHOLD's statement
-------------------------------------------------------------------------------
Lion_meta_update runs UNCHANGED on the whole beta vector, then PATCH_CLIP clamps it, then (PATCH_BETAHOLD is off)
this patch overwrites beta[0][g].  So the other group's beta and momentum are bitwise what they would have been given
the same z; group g's momentum keeps its own unpatched recursion on group g's own z (the three carriers' summed raw
term) and is DEAD STATE (only group g's own Lion step reads it, and that output is overwritten on the same update,
from init to the last update, with no release).  PROBE_TENSOR's `beta_pre` for group g is the previous update's HELD
value, its `mom_pre` / `z_agg` the harness's own -- so the scorer recomputes, per record, the value Lion WROTE and
checks it against `gh_nat`.

-------------------------------------------------------------------------------
INERTNESS
-------------------------------------------------------------------------------
FOUR insertions, every one ADDITIVE; NO existing line is edited:
  1. a guarded `self._gh_apply()` in step(), right after PATCH_BETAHOLD's guarded apply block and before
     `self._probe(...)`;
  2. ONE call, `self._gh_init(net_param_names_and_size)`, right after PATCH_BETAHOLD's `_bh_init` call in init_meta;
  3. a guarded `self._gh_attach(rec)` in _probe, right after PATCH_BETAHOLD's guarded attach block and before
     `self._pt_attach(rec)`;
  4. five new methods (`_gh_init`, `_gh_value`, `_gh_set`, `_gh_apply`, `_gh_attach`) inserted as one block
     before `check_required_attributes`.
With GROUP_HOLD unset or empty, `_gh_on` is False, (1) and (3) do nothing, (2) prints `GROUP_HOLD: off` and
returns.  tests/test_grouphold.py proves the structure (deleting the four regions reproduces cvt4's HF.py byte for
byte) and the semantics on synthetic z; tests/test_grouphold_realrun.py proves OFF BITWISE against the LIVE
ResNet harness (the ISO batches' tree) over a short real ResNet18_c100 run, and that each registered string bites.

The probe record of a HELD run gains four keys (appended; no existing key touched):
  gh_n       meta updates completed so far (must equal step + 2)
  gh_active  cumulative number of updates on which the held value differed from the value Lion + clamp wrote
  gh_nat     the value Lion + clamp wrote for group g on THIS update, before the override
  gh_held    the value group g holds after the override (must equal the record's beta[g])
"""
import os, sys

P = os.environ.get("HF_PATH", "")
if not P:
    print("!!! HF_PATH is required: this patch is applied ONLY to an isolated tree's HF.py")
    sys.exit(2)
src = open(P).read()

if "PATCH_GROUPHOLD" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# --- 0. the assumptions this patch rests on, checked BEFORE anything is written ----
assert "PATCH_BETAHOLD" in src and "PATCH_VOTEWEIGHT" in src, \
    "PATCH_GROUPHOLD is applied on top of PATCH_BETAHOLD (cvt4's HF.py)"
assert "def Lion_meta_update(self,HtT_gradft):" in src, "Lion_meta_update is missing"
assert "_gh_" not in src, "unexpected pre-existing _gh_ names"
assert "GROUP_HOLD" not in src, "unexpected pre-existing GROUP_HOLD handling"

# --- 1. the apply, after PATCH_BETAHOLD's apply and before the probe --------------
a1 = ("            # --- PATCH_BETAHOLD ---\n"
      "            if getattr(self, '_bh_on', False):\n"
      "                self._bh_apply()\n"
      "            # --- end PATCH_BETAHOLD ---\n"
      "            self._probe(HtT_gradft)  # PATCH_PROBE\n")
assert src.count(a1) == 1, "anchor 1 count=%d -- refusing to patch" % src.count(a1)
n1 = ("            # --- PATCH_BETAHOLD ---\n"
      "            if getattr(self, '_bh_on', False):\n"
      "                self._bh_apply()\n"
      "            # --- end PATCH_BETAHOLD ---\n"
      "            # --- PATCH_GROUPHOLD ---\n"
      "            if getattr(self, '_gh_on', False):\n"
      "                self._gh_apply()\n"
      "            # --- end PATCH_GROUPHOLD ---\n"
      "            self._probe(HtT_gradft)  # PATCH_PROBE\n")

# --- 2. the call in init_meta, after PATCH_BETAHOLD's ------------------------------
a2 = "        self._bh_init(net_param_names_and_size)  # PATCH_BETAHOLD\n"
assert src.count(a2) == 1, "anchor 2 count=%d -- refusing to patch" % src.count(a2)
n2 = a2 + "        self._gh_init(net_param_names_and_size)  # PATCH_GROUPHOLD\n"

# --- 3. the attach in _probe, after PATCH_BETAHOLD's, before PATCH_PROBE_TENSOR's ---
a3 = ("        # --- PATCH_BETAHOLD ---\n"
      "        if getattr(self, '_bh_on', False):\n"
      "            self._bh_attach(rec)\n"
      "        # --- end PATCH_BETAHOLD ---\n"
      "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")
assert src.count(a3) == 1, "anchor 3 count=%d -- refusing to patch" % src.count(a3)
n3 = ("        # --- PATCH_BETAHOLD ---\n"
      "        if getattr(self, '_bh_on', False):\n"
      "            self._bh_attach(rec)\n"
      "        # --- end PATCH_BETAHOLD ---\n"
      "        # --- PATCH_GROUPHOLD ---\n"
      "        if getattr(self, '_gh_on', False):\n"
      "            self._gh_attach(rec)\n"
      "        # --- end PATCH_GROUPHOLD ---\n"
      "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")

# --- 4. the methods, before check_required_attributes -------------------------------
a4 = "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n"
assert src.count(a4) == 1, "anchor 4 count=%d -- refusing to patch" % src.count(a4)
n4 = (
    "    # --------------------------------------------------- PATCH_GROUPHOLD\n"
    "    def _gh_init(self, net_param_names_and_size):\n"
    "        \"\"\"Parse GROUP_HOLD, check every premise loudly, apply hold(0), print the witness.\"\"\"\n"
    "        import os as _os, re as _re\n"
    "        self._gh_on = False\n"
    "        raw = _os.environ.get('GROUP_HOLD', '')\n"
    "        if raw == '':\n"
    "            print('GROUP_HOLD: off', flush=True)\n"
    "            return\n"
    "        m = _re.match(r'^([A-Za-z0-9_.]+(?:\\+[A-Za-z0-9_.]+)*):(floor|tri:([0-9]{1,6}))$', raw)\n"
    "        if not m:\n"
    "            raise ValueError('PATCH_GROUPHOLD: GROUP_HOLD %r is not <name>[+<name>...]:floor | '\n"
    "                             '<name>[+<name>...]:tri:<P>' % raw)\n"
    "        want = m.group(1).split('+')\n"
    "        mode = m.group(2)\n"
    "        names = [n for (n, _s) in net_param_names_and_size]\n"
    "        for nm in want:\n"
    "            if nm not in names:\n"
    "                raise ValueError('PATCH_GROUPHOLD: %r is not a parameter name of this model' % nm)\n"
    "        if len(set(want)) != len(want):\n"
    "            raise ValueError('PATCH_GROUPHOLD: a name is repeated in %r' % raw)\n"
    "        idx = [names.index(nm) for nm in want]\n"
    "        if idx != sorted(idx):\n"
    "            raise ValueError('PATCH_GROUPHOLD: names must be written in model order; got %r' % raw)\n"
    "        if self.stepsize_type != 'blockwise':\n"
    "            raise ValueError('PATCH_GROUPHOLD: needs a blockwise (sets:) grouping; stepsize_type is %r'\n"
    "                             % (self.stepsize_type,))\n"
    "        if getattr(self, '_rednorm', False):\n"
    "            raise ValueError('PATCH_GROUPHOLD: cannot be combined with a tn: (PATCH_REDNORM) spec')\n"
    "        if getattr(self, '_vw_on', False):\n"
    "            raise ValueError('PATCH_GROUPHOLD: cannot be combined with VOTE_W')\n"
    "        if getattr(self, '_bh_on', False):\n"
    "            raise ValueError('PATCH_GROUPHOLD: cannot be combined with BETA_HOLD')\n"
    "        if getattr(self, '_hier', '') not in ('', 'none'):\n"
    "            raise ValueError('PATCH_GROUPHOLD: cannot be combined with HIER=%r' % (self._hier,))\n"
    "        _am = getattr(self, 'args_meta', None) or {}\n"
    "        if _am.get('alg') != 'Lion':\n"
    "            raise ValueError('PATCH_GROUPHOLD: the meta algorithm must be Lion; got %r' % (_am.get('alg'),))\n"
    "        if getattr(self, '_beta_lo', None) is None:\n"
    "            raise ValueError('PATCH_GROUPHOLD: needs BETA_CLIP (the floor and the triangle live inside the clamp)')\n"
    "        k = [j for j, grp in enumerate(self.param_groups_indices) if sorted(grp) == idx]\n"
    "        if len(k) != 1:\n"
    "            raise ValueError('PATCH_GROUPHOLD: %r is not EXACTLY the membership of one group' % raw)\n"
    "        self._gh_g = k[0]\n"
    "        self._gh_n = 0\n"
    "        self._gh_active = 0\n"
    "        self._gh_nat = None\n"
    "        self._gh_ms = float(_am['meta_stepsize'])\n"
    "        self._gh_lo, self._gh_hi = float(self._beta_lo), float(self._beta_hi)\n"
    "        self._gh_b0 = float(self.beta[0][self._gh_g].item())\n"
    "        self._gh_P = None\n"
    "        desc = ('GROUP_HOLD: on type=blockwise group=%d groupsize=%d names=%s'\n"
    "                % (self._gh_g, len(idx), '+'.join(want)))\n"
    "        if mode == 'floor':\n"
    "            self._gh_mode = 'floor'\n"
    "            desc += ' mode=floor value=%r' % (self._gh_lo,)\n"
    "        else:\n"
    "            self._gh_mode = 'tri'\n"
    "            self._gh_P = int(m.group(3))\n"
    "            if self._gh_P < 1:\n"
    "                raise ValueError('PATCH_GROUPHOLD: tri needs P >= 1')\n"
    "            desc += (' mode=tri P=%d b0=%r ms=%r lo=%r hi=%r peak=%r'\n"
    "                     % (self._gh_P, self._gh_b0, self._gh_ms, self._gh_lo, self._gh_hi,\n"
    "                        min(self._gh_hi, max(self._gh_lo, self._gh_b0 + self._gh_ms * self._gh_P))))\n"
    "        self._gh_on = True\n"
    "        self._gh_set(self._gh_value(0))\n"
    "        print(desc, flush=True)\n"
    "\n"
    "    def _gh_value(self, n):\n"
    "        \"\"\"hold(n) as a python float.\"\"\"\n"
    "        if self._gh_mode == 'floor':\n"
    "            return self._gh_lo\n"
    "        u = 0 if n <= 0 else min(n - 1, self._gh_P) - max(0, n - 1 - self._gh_P)\n"
    "        return min(self._gh_hi, max(self._gh_lo, self._gh_b0 + self._gh_ms * u))\n"
    "\n"
    "    def _gh_set(self, value):\n"
    "        _b = self.beta[0].clone()\n"
    "        _b[self._gh_g] = value\n"
    "        self.beta[0] = _b\n"
    "\n"
    "    def _gh_apply(self):\n"
    "        \"\"\"After update n's Lion step and clamp: record what Lion wrote, then hold.\"\"\"\n"
    "        self._gh_n += 1\n"
    "        _nat = float(self.beta[0][self._gh_g].item())\n"
    "        self._gh_set(self._gh_value(self._gh_n))\n"
    "        _held = float(self.beta[0][self._gh_g].item())\n"
    "        self._gh_nat = _nat\n"
    "        if _held != _nat:\n"
    "            self._gh_active += 1\n"
    "\n"
    "    def _gh_attach(self, rec):\n"
    "        \"\"\"APPENDS keys to the probe record; never touches an existing key.\"\"\"\n"
    "        rec['gh_n'] = int(self._gh_n)\n"
    "        rec['gh_active'] = int(self._gh_active)\n"
    "        rec['gh_nat'] = self._gh_nat\n"
    "        rec['gh_held'] = float(self.beta[0][self._gh_g].item())\n"
    "    # ----------------------------------------------- end PATCH_GROUPHOLD\n"
    "\n"
    + a4)

src2 = src.replace(a1, n1, 1).replace(a2, n2, 1).replace(a3, n3, 1).replace(a4, n4, 1)
bak = P + ".pre_grouphold"
if os.path.exists(bak):
    if open(bak).read() != src:
        print("!!! %s exists and differs from the file being patched -- refusing" % bak)
        sys.exit(2)
else:
    open(bak, "w").write(src)
open(P, "w").write(src2)
print("PATCHED PATCH_GROUPHOLD ->", P, "(backup", bak + ")")
