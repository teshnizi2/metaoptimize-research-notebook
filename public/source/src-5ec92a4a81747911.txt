"""PATCH_SHADOWVOTE -- an OPT-IN, ENVIRONMENT-CONTROLLED dissociation of a tensor's APPLIED step size from the VOTE it
casts into the shared (scalar) meta-gradient sum.  Written for `csv1` (CORRECTIONS 262).  Applied ONLY to the isolated
tree $METAOPT_WS/harness_csv1/cifar10 (bin/cSV1_stage_harness.sh), ON TOP OF cvt9's HF.py (sha 816e3357..., PATCH_WINDOWHOLD
included); the live shared harness and every registered tree (cvt1, cvt4, cvt6, cvt7, cvt8, cvt9) are never patched.

-------------------------------------------------------------------------------
WHY
-------------------------------------------------------------------------------
At cvt1's cell (PlainNet18_c100, SGDm 0.99 / wd 0.1 + Lion meta, ms 1e-3, alpha0 1e-6) the scalar step size collapses.
cvt4 / cvt6 / cvt9 showed that holding layer4.1.bn2.weight (idx 50) on a LARGE step-size trajectory, with its vote OUT of
the complement's sum, is SUFFICIENT to stall.  Nothing tested NECESSITY: with the tensor's vote left IN the shared sum,
does keeping its APPLIED step size small prevent the stall?  In MetaOptimize the two cannot be separated: the tensor's
term <h_i, g_i> is built from its own trace h_i, and h_i is built from the step size the tensor actually took.  This
patch separates them for the named tensors, in the scalar grouping only:

    applied step size   a_ap = a_sh (the shared step size)          applied=shared
                        a_ap = exp(float32(BETA_CLIP lo))           applied=floor   (the value HOLDLOW's group applied)
    vote trace          h_i  = the SHADOW trace, built at a_sh      vote=shadow
                        h_i  = the NATURAL trace, built at a_ap     vote=natural   (what the harness does anyway)

    SHADOW trace:   h^s <- gamma*(1 - wd*a_sh)*h^s - a_sh*(m_i + wd*w_i)
    NATURAL trace:  h^n <- gamma*(1 - wd*a_ap)*h^n - a_ap*(m_i + wd*w_i)

m_i (SGDm's momentum) and w_i (the weights) are the REAL ones -- the tensor's state on the trajectory it actually
follows at a_ap.  The shared sum z = sum_i <h_i, g_i> reads h_condenced, so under vote=shadow the named tensor's slot in
h_condenced holds h^s and the natural trace is kept beside it (and vice versa), only for the probe records.
The shadow vote is therefore COUNTERFACTUAL: "what this tensor's term would be had it moved at the shared step size,
evaluated on the state it really has".  It is NOT the term the tensor casts in unmodified MetaOptimize (its weights and
gradient there follow the large step).

-------------------------------------------------------------------------------
THE SWITCH
-------------------------------------------------------------------------------
    SHADOW_VOTE=<vote>:<applied>:<name>[/<name>...]      vote in {shadow, natural}; applied in {floor, shared}

  * UNSET or EMPTY -> OFF.  OFF is the state every prior run was in (cvt9's tree, bitwise).
  * shadow:shared and natural:shared are INERT ON-PATHS: the two traces coincide, every alpha is the shared one, so the
    run is bitwise the OFF run (tests/test_shadowvote.py S3, tests/test_shadowvote_realrun.py RR2).
  * <name> is an exact parameter name of the LIVE model (resolved by NAME); list-capable ('/'-separated, no repeats).
  * PREMISES, LOUD (ValueError at construction): a malformed value; an unknown or repeated name; stepsize_type other than
    `scalar`; base algorithm other than SGDm (the trace recursion above is SGDm's); a tn: (PATCH_REDNORM) spec; VOTE_W,
    BETA_HOLD, COMP_HOLD or WINDOW_HOLD on; HIER other than '' / 'none'; SCHED other than '' / 'none'; applied=floor
    without BETA_CLIP.
  * Every character of a value is in [A-Za-z0-9_.:/], so SHADOW_VOTE=... is ONE token of sbatch's --export list.
  * A witness line is printed at construction on EVERY run (the runner's ENV line cannot carry SHADOW_VOTE):
        SHADOW_VOTE: off
        SHADOW_VOTE: on type=scalar base=SGDm vote=<vote> applied=<applied> floor=<lo|na> items=<idx1>:<name>:numel=<n>[,...]
    Its prefix starts with S, which none of VOTE_W / BETA_HOLD / GROUP_HOLD / COMP_HOLD / REST_HOLD / WINDOW_HOLD does
    (and SHADOW_VOTE is not a prefix of any of them), so no `startswith` reader of those kinds selects it.

-------------------------------------------------------------------------------
WHAT IT TOUCHES
-------------------------------------------------------------------------------
Only, for each named tensor i, on every step: (a) self.alpha[i] BEFORE base_update (a fresh list built by beta_to_alpha
that step; nothing else reads it); (b) under vote=shadow, h_condenced[i] AFTER base_update, recomputed from the SAME
pre-update references (h, m, w) with the SAME expression SGDm_base_update uses, at a_sh.  beta, momentum_meta, Lion,
the clamp, momentum_base, every other tensor and every other trace are untouched.  Under applied=floor PATCH_SCHED's
factor (1.0 for SCHED '' / 'none') is applied to the floor value exactly as step() applies it to the shared alpha.

-------------------------------------------------------------------------------
INERTNESS
-------------------------------------------------------------------------------
FIVE insertions, every one ADDITIVE; NO existing line is edited:
  1. a guarded `self._sv_pre(net, g)` in step(), right after the PROBE_TENSOR capture and before base_update;
  2. a guarded `self._sv_post(net)` in step(), right after `self.base_update(net,g)`;
  3. ONE call, `self._sv_init(net_param_names_and_size)`, right after PATCH_WINDOWHOLD's `_wh_init` call in init_meta;
  4. a guarded `self._sv_attach(rec)` in _probe, right after PATCH_WINDOWHOLD's `_wh_attach` region and before
     `self._pt_attach(rec)`;
  5. the five new methods (`_sv_init`, `_sv_pre`, `_sv_post`, `_sv_norms`, `_sv_attach`) inserted as one block after
     PATCH_WINDOWHOLD's method block, before `check_required_attributes`.
With SHADOW_VOTE unset or empty, `_sv_on` is False, (1), (2) and (4) do nothing, (3) prints `SHADOW_VOTE: off` and
returns: no tensor is touched, the probe record gains no key.  tests/test_shadowvote.py proves the structure (deleting
the five regions reproduces cvt9's HF.py byte for byte) and the arithmetic on CPU; tests/test_shadowvote_realrun.py proves
beta at every step, the loss, probe.jsonl bytes and every final parameter and buffer BITWISE against cvt9's tree over a
short real CIFAR-100 PlainNet run (OFF, and the two inert on-paths), and that the registered strings bite.

The probe record of a SHADOW_VOTE run gains these keys (one list entry per named tensor, registered order), appended
after PATCH_WINDOWHOLD's and before PROBE_TENSOR's; no existing key touched:
  sv_n         base updates completed so far (must equal step + 2: the record's step is the counter, which starts at
               -1 and is incremented at the END of step(), after _probe -- as PATCH_BETAHOLD's bh_n)
  sv_vote      'shadow' | 'natural';   sv_applied  'floor' | 'shared';   sv_idx  1-based indices
  sv_a_shared  the shared step size this update (== exp(float32 beta_pre))
  sv_a_applied the step size base_update applied to the tensor this update
  sv_z_vote    <vote trace, g_i> this update (== PROBE_TENSOR's z_tensor[i]: the term that entered the sum)
  sv_z_other   <other trace, g_i> this update (never entered the sum)
  sv_dw        ||w_i(after) - w_i(before)||                 (what really moved)
  sv_dref      ||a_sh * (m_i + wd*w_i)||                     (what the shared step size would have moved)
  sv_dapp      ||a_ap * (m_i + wd*w_i)||                     (what the applied step size moves, before float32 rounding)
  sv_hv_norm   ||vote trace|| after this update;   sv_ho_norm  ||other trace|| after this update
"""
import os, sys

P = os.environ.get("HF_PATH", "")
if not P:
    print("!!! HF_PATH is required: this patch is applied ONLY to an isolated tree's HF.py")
    sys.exit(2)
src = open(P).read()

if "PATCH_SHADOWVOTE" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# --- 0. the assumptions this patch rests on, checked BEFORE anything is written ----
assert "PATCH_WINDOWHOLD" in src and "PATCH_COMPHOLD" in src and "PATCH_BETAHOLD" in src and "PATCH_VOTEWEIGHT" in src, \
    "this patch sits on cvt9's HF.py (PATCH_WINDOWHOLD)"
assert "_sv_" not in src, "unexpected pre-existing _sv_ names"
assert "SHADOW_VOTE" not in src, "unexpected pre-existing SHADOW_VOTE handling"
assert "    def SGDm_base_update(self,net,g):\n" in src
assert ("            delta = a * (self.momentum_base[i] + self.args_base['weight_decay']*w.data)\n"
        "            w.data = w.data - delta\n"
        "            self.momentum_base[i] = self.args_base['momentum_param']*self.momentum_base[i] + (1 - self.args_base['momentum_param'])*grad\n"
        "            self.h_condenced[i] = self.gamma*(1-self.args_base['weight_decay']*a)*self.h_condenced[i] - delta\n") in src, \
    "SGDm_base_update is not the expression this patch replicates"

# --- 1. _sv_pre, after the PROBE_TENSOR capture, before base_update ------------------
a1 = ("            self._pt_capture(self.h_condenced, g, HtT_gradft)  # PATCH_PROBE_TENSOR\n"
      "            \n"
      "            self.base_update(net,g)\n")
assert src.count(a1) == 1, "anchor 1 count=%d -- refusing to patch" % src.count(a1)
n1 = ("            self._pt_capture(self.h_condenced, g, HtT_gradft)  # PATCH_PROBE_TENSOR\n"
      "            # --- PATCH_SHADOWVOTE ---\n"
      "            if getattr(self, '_sv_on', False):\n"
      "                self._sv_pre(net, g)\n"
      "            # --- end PATCH_SHADOWVOTE ---\n"
      "            \n"
      "            self.base_update(net,g)\n"
      "            # --- PATCH_SHADOWVOTE ---\n"
      "            if getattr(self, '_sv_on', False):\n"
      "                self._sv_post(net)\n"
      "            # --- end PATCH_SHADOWVOTE ---\n")

# --- 3. the call at the end of init_meta's patch calls --------------------------------
a3 = "        self._wh_init(net_param_names_and_size)  # PATCH_WINDOWHOLD\n"
assert src.count(a3) == 1, "anchor 3 count=%d -- refusing to patch" % src.count(a3)
n3 = a3 + "        self._sv_init(net_param_names_and_size)  # PATCH_SHADOWVOTE\n"

# --- 4. the probe attach ------------------------------------------------------------
a4 = ("        # --- end PATCH_WINDOWHOLD ---\n"
      "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")
assert src.count(a4) == 1, "anchor 4 count=%d -- refusing to patch" % src.count(a4)
n4 = ("        # --- end PATCH_WINDOWHOLD ---\n"
      "        # --- PATCH_SHADOWVOTE ---\n"
      "        if getattr(self, '_sv_on', False):\n"
      "            self._sv_attach(rec)\n"
      "        # --- end PATCH_SHADOWVOTE ---\n"
      "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")

# --- 5. the methods, after PATCH_WINDOWHOLD's block, before check_required_attributes --
a5 = ("    # -------------------------------------------------- end PATCH_WINDOWHOLD\n"
      "\n"
      "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n")
assert src.count(a5) == 1, "anchor 5 count=%d -- refusing to patch" % src.count(a5)
n5 = (
    "    # -------------------------------------------------- end PATCH_WINDOWHOLD\n"
    "\n"
    "    # ------------------------------------------------------ PATCH_SHADOWVOTE\n"
    "    def _sv_init(self, net_param_names_and_size):\n"
    "        \"\"\"Parse SHADOW_VOTE, check every premise loudly, print the witness.  Touches no tensor.\"\"\"\n"
    "        import os as _os, re as _re\n"
    "        self._sv_on = False\n"
    "        raw = _os.environ.get('SHADOW_VOTE', '')\n"
    "        if raw == '':\n"
    "            print('SHADOW_VOTE: off', flush=True)\n"
    "            return\n"
    "        m = _re.match(r'^(shadow|natural):(floor|shared):([A-Za-z0-9_.]+(?:/[A-Za-z0-9_.]+)*)$', raw)\n"
    "        if not m:\n"
    "            raise ValueError('PATCH_SHADOWVOTE: SHADOW_VOTE %r is not <shadow|natural>:<floor|shared>:<name>[/<name>...]' % raw)\n"
    "        names = [n for (n, _s) in net_param_names_and_size]\n"
    "        items = m.group(3).split('/')\n"
    "        for nm in items:\n"
    "            if nm not in names:\n"
    "                raise ValueError('PATCH_SHADOWVOTE: %r is not a parameter name of this model' % nm)\n"
    "        if len(set(items)) != len(items):\n"
    "            raise ValueError('PATCH_SHADOWVOTE: a name is repeated in %r' % raw)\n"
    "        if self.stepsize_type != 'scalar':\n"
    "            raise ValueError('PATCH_SHADOWVOTE: needs the scalar grouping (one shared sum); stepsize_type is %r'\n"
    "                             % (self.stepsize_type,))\n"
    "        _ab = getattr(self, 'args_base', None) or {}\n"
    "        if _ab.get('alg') != 'SGDm':\n"
    "            raise ValueError('PATCH_SHADOWVOTE: the trace recursion is SGDm\\'s; base alg is %r' % (_ab.get('alg'),))\n"
    "        if getattr(self, '_rednorm', False):\n"
    "            raise ValueError('PATCH_SHADOWVOTE: cannot be combined with a tn: (PATCH_REDNORM) spec')\n"
    "        for _flag, _kind in (('_vw_on', 'VOTE_W'), ('_bh_on', 'BETA_HOLD'), ('_ch_on', 'COMP_HOLD'), ('_wh_on', 'WINDOW_HOLD')):\n"
    "            if getattr(self, _flag, False):\n"
    "                raise ValueError('PATCH_SHADOWVOTE: cannot be combined with %s' % _kind)\n"
    "        if getattr(self, '_hier', '') not in ('', 'none'):\n"
    "            raise ValueError('PATCH_SHADOWVOTE: cannot be combined with HIER=%r' % (self._hier,))\n"
    "        if getattr(self, '_sched', '') not in ('', 'none'):\n"
    "            raise ValueError('PATCH_SHADOWVOTE: cannot be combined with SCHED=%r' % (self._sched,))\n"
    "        self._sv_vote, self._sv_applied = m.group(1), m.group(2)\n"
    "        if self._sv_applied == 'floor':\n"
    "            if getattr(self, '_beta_lo', None) is None:\n"
    "                raise ValueError('PATCH_SHADOWVOTE: applied=floor needs BETA_CLIP (the floor is its lower bound)')\n"
    "            self._sv_floor_alpha = np.exp(np.asarray(self._beta_lo, dtype=np.float32))\n"
    "            _fl = '%r' % (float(self._beta_lo),)\n"
    "        else:\n"
    "            self._sv_floor_alpha = None\n"
    "            _fl = 'na'\n"
    "        self._sv_idx = [names.index(nm) for nm in items]\n"
    "        self._sv_n = 0\n"
    "        self._sv_other = None\n"
    "        self._sv_save = []\n"
    "        self._sv_rec = None\n"
    "        desc = ('SHADOW_VOTE: on type=scalar base=SGDm vote=%s applied=%s floor=%s items=%s'\n"
    "                % (self._sv_vote, self._sv_applied, _fl,\n"
    "                   ','.join('%d:%s:numel=%d' % (i + 1, names[i], int(np.prod(list(net_param_names_and_size[i][1]))))\n"
    "                            for i in self._sv_idx)))\n"
    "        self._sv_on = True\n"
    "        print(desc, flush=True)\n"
    "\n"
    "    def _sv_pre(self, net, g):\n"
    "        \"\"\"After block_product and the PROBE_TENSOR capture, before base_update: set each named tensor's applied\n"
    "        step size and keep references to the pre-update h, other trace, m and w.  Reads the two terms for the record.\"\"\"\n"
    "        params = list(net.parameters())\n"
    "        if self._sv_other is None:\n"
    "            self._sv_other = dict((i, torch.zeros_like(self.h_condenced[i])) for i in self._sv_idx)\n"
    "        save, zv, zo = [], [], []\n"
    "        for i in self._sv_idx:\n"
    "            a_sh = self.alpha[i]\n"
    "            if self._sv_applied == 'shared':\n"
    "                a_ap = a_sh\n"
    "            else:\n"
    "                a_ap = self._sv_floor_alpha\n"
    "                if self._sched:  # the factor step() applied to the shared alpha (1.0 for SCHED '' / 'none')\n"
    "                    a_ap = a_ap * self._sched_factor()\n"
    "            self.alpha[i] = a_ap\n"
    "            save.append((i, a_sh, a_ap, self.h_condenced[i], self._sv_other[i], self.momentum_base[i], params[i].data))\n"
    "            zv.append((self.h_condenced[i] * g[i]).sum().detach())\n"
    "            zo.append((self._sv_other[i] * g[i]).sum().detach())\n"
    "        self._sv_save = save\n"
    "        self._sv_rec = {'zv': zv, 'zo': zo}\n"
    "\n"
    "    def _sv_post(self, net):\n"
    "        \"\"\"After base_update: under vote=shadow, rebuild the vote slot at the SHARED step size from the pre-update\n"
    "        references, with SGDm_base_update's own expression; keep the other trace beside it.\"\"\"\n"
    "        params = list(net.parameters())\n"
    "        wd = self.args_base['weight_decay']\n"
    "        dw, dref, dapp = [], [], []\n"
    "        for (i, a_sh, a_ap, hp, op, mp, wp) in self._sv_save:\n"
    "            if self._sv_vote == 'shadow':\n"
    "                delta = a_sh * (mp + wd*wp)\n"
    "                self.h_condenced[i] = self.gamma*(1-wd*a_sh)*hp - delta\n"
    "                d_o = a_ap * (mp + wd*wp)\n"
    "                self._sv_other[i] = self.gamma*(1-wd*a_ap)*op - d_o\n"
    "            else:\n"
    "                d_o = a_sh * (mp + wd*wp)\n"
    "                self._sv_other[i] = self.gamma*(1-wd*a_sh)*op - d_o\n"
    "            dw.append((params[i].data - wp).norm().detach())\n"
    "            dref.append(self._sv_norms(a_sh * (mp + wd*wp)))\n"
    "            dapp.append(self._sv_norms(a_ap * (mp + wd*wp)))\n"
    "        self._sv_n += 1\n"
    "        self._sv_rec.update({'dw': dw, 'dref': dref, 'dapp': dapp,\n"
    "                             'hv': [self._sv_norms(self.h_condenced[i]) for i in self._sv_idx],\n"
    "                             'ho': [self._sv_norms(self._sv_other[i]) for i in self._sv_idx],\n"
    "                             'a_sh': [s[1] for s in self._sv_save], 'a_ap': [s[2] for s in self._sv_save]})\n"
    "\n"
    "    def _sv_norms(self, t):\n"
    "        return t.norm().detach() if torch.is_tensor(t) else torch.tensor(abs(float(t)))\n"
    "\n"
    "    def _sv_attach(self, rec):\n"
    "        \"\"\"APPENDS keys to the probe record; never touches an existing key.\"\"\"\n"
    "        R = self._sv_rec or {}\n"
    "        fl = lambda xs: [float(x) for x in xs]\n"
    "        rec['sv_n'] = int(self._sv_n)\n"
    "        rec['sv_vote'] = self._sv_vote\n"
    "        rec['sv_applied'] = self._sv_applied\n"
    "        rec['sv_idx'] = [int(i + 1) for i in self._sv_idx]\n"
    "        rec['sv_a_shared'] = fl(R.get('a_sh', []))\n"
    "        rec['sv_a_applied'] = fl(R.get('a_ap', []))\n"
    "        rec['sv_z_vote'] = fl(R.get('zv', []))\n"
    "        rec['sv_z_other'] = fl(R.get('zo', []))\n"
    "        rec['sv_dw'] = fl(R.get('dw', []))\n"
    "        rec['sv_dref'] = fl(R.get('dref', []))\n"
    "        rec['sv_dapp'] = fl(R.get('dapp', []))\n"
    "        rec['sv_hv_norm'] = fl(R.get('hv', []))\n"
    "        rec['sv_ho_norm'] = fl(R.get('ho', []))\n"
    "    # -------------------------------------------------- end PATCH_SHADOWVOTE\n"
    "\n"
    "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n")

src2 = src.replace(a1, n1, 1).replace(a3, n3, 1).replace(a4, n4, 1).replace(a5, n5, 1)
bak = P + ".pre_shadowvote"
if os.path.exists(bak):
    if open(bak).read() != src:
        print("!!! %s exists and differs from the file being patched -- refusing" % bak)
        sys.exit(2)
else:
    open(bak, "w").write(src)
open(P, "w").write(src2)
print("PATCHED PATCH_SHADOWVOTE ->", P, "(backup", bak + ")")
