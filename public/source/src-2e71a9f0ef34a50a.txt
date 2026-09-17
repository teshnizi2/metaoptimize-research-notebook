"""PATCH_COMPHOLD -- an OPT-IN, ENVIRONMENT-CONTROLLED override of the COMPLEMENT group's beta (its log step size),
applied after every meta update, AFTER PATCH_BETAHOLD's hold on the singleton group.  Written for `cvt6`
(CORRECTIONS 242).  Applied ONLY to the isolated tree $METAOPT_WS/harness_cvt6/cifar10 (bin/cVT6_stage_harness.sh),
ON TOP OF cvt4's PATCH_BETAHOLD HF.py (sha 84b345ad...); the live shared harness, cvt1's tree and cvt4's tree are
never patched.

-------------------------------------------------------------------------------
WHY
-------------------------------------------------------------------------------
cvt4 (240) held layer4.1.bn2.weight's (idx 50) beta in HEAD's grouping and found OWN-STEP-MAGNITUDE: 50 on MUTE's
large step size stalls at k01, 50 at the floor keeps HEAD's level.  But in HOLDHIGH the complement's own FREE Lion
beta also collapsed early (on MUTE's path), so the batch could not say whether 50's large step size stalls the
network DIRECTLY or THROUGH the complement's early collapse (240.4(1)).  This patch holds the complement's beta
too, so the two can be crossed:

    after Lion_meta_update, PATCH_CLIP and PATCH_BETAHOLD on update n:   beta[0][c] := hold_c(n)

where c is the OTHER group of the 2-group grouping whose singleton BETA_HOLD holds.

-------------------------------------------------------------------------------
THE SWITCH
-------------------------------------------------------------------------------
    COMP_HOLD=tri:<P> | rec:<id>        (environment variable)

  * UNSET or EMPTY -> OFF.  OFF is the state every prior run was in (cvt4's tree, bitwise).
  * REQUIRES BETA_HOLD ON in mode floor or tri (PATCH_BETAHOLD's premises -- blockwise, Lion, BETA_CLIP, no VOTE_W,
    no HIER, no tn: -- are therefore already enforced), and EXACTLY 2 groups.  c = 1 - (BETA_HOLD's group).
  * tri:<P>   the max-rate triangle of PATCH_BETAHOLD (u(0) = 0; u(n) = min(n-1, P) - max(0, n-1-P);
              hold_c(n) = min(hi, max(lo, b0 + ms*u(n))), b0 = group c's own initial beta).
  * rec:<id>  an EXACT REPLAY of a recorded trajectory: the JSON file <dir of HF.py>/<id>.json with lists `n`
              (strictly increasing ints >= 1) and `v` (finite floats inside [lo, hi]), len >= 2:
                  n <  n[0]              hold_c(n) = b0
                  n[k] <= n < n[k+1]     hold_c(n) = v[k] + (v[k+1] - v[k]) * (n - n[k]) / float(n[k+1] - n[k])
                  n >= n[-1]             hold_c(n) = v[-1]
                  then min(hi, max(lo, .)), python double, stored into float32.
              The witness prints the file's sha256, so the replayed numbers are pinned by the witness line.
  * LOUD, never silently partial: a malformed value, BETA_HOLD off, BETA_HOLD in `shared` mode, a grouping with other
    than 2 groups, a missing / unreadable / malformed replay file, a knot outside the clamp -> ValueError at
    construction.  Every character of a value is in [A-Za-z0-9_.:], so COMP_HOLD=... is ONE token of sbatch's
    comma-separated --export list.
  * A witness line is printed at construction on EVERY run (the runner's ENV line cannot carry COMP_HOLD):
        COMP_HOLD: off
        COMP_HOLD: on type=blockwise group=<c> groupsize=<m> mode=tri P=<P> b0=<b0> ms=<ms> lo=<lo> hi=<hi> peak=<v>
        COMP_HOLD: on type=blockwise group=<c> groupsize=<m> mode=rec id=<id> sha256=<64 hex> knots=<K> n0=<n> n1=<n> b0=<b0> lo=<lo> hi=<hi> vmax=<v> vlast=<v>
    Its prefix deliberately does NOT start with `BETA_HOLD`, so readers that collect `BETA_HOLD` lines by prefix
    (analysis/corpus_exclusions.py's KINDS) see exactly the lines they saw on cvt4.

-------------------------------------------------------------------------------
WHAT THE OVERRIDE DOES TO THE META-OPTIMISER'S STATE (Lion momentum) -- stated exactly
-------------------------------------------------------------------------------
Lion_meta_update runs UNCHANGED on the whole beta vector; PATCH_CLIP clamps; PATCH_BETAHOLD overwrites beta[0][g];
only AFTER that is beta[0][c] overwritten.  With both groups held, NO applied beta depends on z or on momentum_meta:
  * both momentum_meta entries keep the unpatched recursion on their own raw z; they are NOT reset, frozen or copied;
  * they are DEAD STATE: the only reader of momentum_meta[i] is entry i's own Lion step, whose output is overwritten
    on the same update, from init (n = 0) to the last update, with no release;
  * PROBE_TENSOR's `beta_pre` for group c is the previous update's HELD value, its `mom_pre` / `z_agg` are the
    harness's own (dead) momentum and raw z -- which lets the scorer recompute, per record, the value Lion WROTE
    before the override and check it against `ch_nat`.

-------------------------------------------------------------------------------
INERTNESS
-------------------------------------------------------------------------------
FOUR insertions, every one ADDITIVE; NO existing line is edited:
  1. a guarded `self._ch_apply()` in step(), right after PATCH_BETAHOLD's guarded `_bh_apply` region and before
     `self._probe(...)`;
  2. ONE call, `self._ch_init(net_param_names_and_size)`, right after PATCH_BETAHOLD's `_bh_init` call in init_meta;
  3. a guarded `self._ch_attach(rec)` in _probe, right after PATCH_BETAHOLD's guarded `_bh_attach` region and before
     `self._pt_attach(rec)`;
  4. the five new methods (`_ch_init`, `_ch_value`, `_ch_set`, `_ch_apply`, `_ch_attach`) inserted as one block
     after PATCH_BETAHOLD's method block, before `check_required_attributes`.
With COMP_HOLD unset or empty, `_ch_on` is False, (1) and (3) do nothing, (2) prints `COMP_HOLD: off` and returns: no
tensor is touched, the probe record gains no key.  tests/test_comphold.py proves the structure (deleting the four
regions reproduces cvt4's HF.py byte for byte); tests/test_comphold_realrun.py proves beta at every step, the loss,
probe.jsonl bytes and every final parameter and buffer BITWISE against cvt4's tree over a short real CIFAR-100
PlainNet run (k01, HEAD, HOLDLOW, HOLDHIGH), and that each registered COMP_HOLD string bites.

The probe record of a COMP_HOLD run gains four keys (appended after PATCH_BETAHOLD's; no existing key touched):
  ch_n       meta updates completed so far (must equal step + 2)
  ch_active  cumulative number of updates on which the held value differed from the value Lion + clamp wrote
  ch_nat     the value Lion + clamp wrote for group c on THIS update, before the override
  ch_held    the value group c holds after the override (must equal the record's beta[c])
"""
import os, sys

P = os.environ.get("HF_PATH", "")
if not P:
    print("!!! HF_PATH is required: this patch is applied ONLY to an isolated tree's HF.py")
    sys.exit(2)
src = open(P).read()

if "PATCH_COMPHOLD" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# --- 0. the assumptions this patch rests on, checked BEFORE anything is written ----
assert "PATCH_BETAHOLD" in src, "PATCH_COMPHOLD is applied on top of PATCH_BETAHOLD (cvt4's HF.py)"
assert "PATCH_VOTEWEIGHT" in src, "the base must carry PATCH_VOTEWEIGHT (cvt1's HF.py under cvt4's)"
assert "def Lion_meta_update(self,HtT_gradft):" in src, "Lion_meta_update is missing"
assert "_ch_" not in src, "unexpected pre-existing _ch_ names"
assert "COMP_HOLD" not in src, "unexpected pre-existing COMP_HOLD handling"

# --- 1. the apply, after PATCH_BETAHOLD's and before the probe ----------------------
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
      "            # --- PATCH_COMPHOLD ---\n"
      "            if getattr(self, '_ch_on', False):\n"
      "                self._ch_apply()\n"
      "            # --- end PATCH_COMPHOLD ---\n"
      "            self._probe(HtT_gradft)  # PATCH_PROBE\n")

# --- 2. the call in init_meta, after PATCH_BETAHOLD's -------------------------------
a2 = "        self._bh_init(net_param_names_and_size)  # PATCH_BETAHOLD\n"
assert src.count(a2) == 1, "anchor 2 count=%d -- refusing to patch" % src.count(a2)
n2 = a2 + "        self._ch_init(net_param_names_and_size)  # PATCH_COMPHOLD\n"

# --- 3. the attach in _probe, after PATCH_BETAHOLD's and before PATCH_PROBE_TENSOR's --
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
      "        # --- PATCH_COMPHOLD ---\n"
      "        if getattr(self, '_ch_on', False):\n"
      "            self._ch_attach(rec)\n"
      "        # --- end PATCH_COMPHOLD ---\n"
      "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")

# --- 4. the methods, after PATCH_BETAHOLD's block, before check_required_attributes --
a4 = ("    # -------------------------------------------------- end PATCH_BETAHOLD\n"
      "\n"
      "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n")
assert src.count(a4) == 1, "anchor 4 count=%d -- refusing to patch" % src.count(a4)
n4 = (
    "    # -------------------------------------------------- end PATCH_BETAHOLD\n"
    "\n"
    "    # ------------------------------------------------------ PATCH_COMPHOLD\n"
    "    def _ch_init(self, net_param_names_and_size):\n"
    "        \"\"\"Parse COMP_HOLD, check every premise loudly, apply hold_c(0), print the witness.\"\"\"\n"
    "        import os as _os, re as _re, json as _json, hashlib as _hashlib, math as _math\n"
    "        self._ch_on = False\n"
    "        raw = _os.environ.get('COMP_HOLD', '')\n"
    "        if raw == '':\n"
    "            print('COMP_HOLD: off', flush=True)\n"
    "            return\n"
    "        m = _re.match(r'^(?:tri:([0-9]{1,6})|rec:([A-Za-z0-9_]{1,64}))$', raw)\n"
    "        if not m:\n"
    "            raise ValueError('PATCH_COMPHOLD: COMP_HOLD %r is not tri:<P> | rec:<id>' % raw)\n"
    "        if not getattr(self, '_bh_on', False):\n"
    "            raise ValueError('PATCH_COMPHOLD: needs BETA_HOLD on (COMP_HOLD holds the OTHER group of its grouping)')\n"
    "        if self._bh_mode not in ('floor', 'tri'):\n"
    "            raise ValueError('PATCH_COMPHOLD: BETA_HOLD mode %r is not floor or tri' % (self._bh_mode,))\n"
    "        if self.num_blocks != 2 or len(self.param_groups_indices) != 2:\n"
    "            raise ValueError('PATCH_COMPHOLD: needs exactly 2 groups; there are %d' % self.num_blocks)\n"
    "        self._ch_g = 1 - self._bh_g\n"
    "        self._ch_n = 0\n"
    "        self._ch_active = 0\n"
    "        self._ch_nat = None\n"
    "        self._ch_ms, self._ch_lo, self._ch_hi = self._bh_ms, self._bh_lo, self._bh_hi\n"
    "        self._ch_b0 = float(self.beta[0][self._ch_g].item())\n"
    "        self._ch_P = None\n"
    "        self._ch_kn = None\n"
    "        self._ch_kv = None\n"
    "        desc = 'COMP_HOLD: on type=blockwise group=%d groupsize=%d' % (self._ch_g, len(self.param_groups_indices[self._ch_g]))\n"
    "        if m.group(1) is not None:\n"
    "            self._ch_mode = 'tri'\n"
    "            self._ch_P = int(m.group(1))\n"
    "            if self._ch_P < 1:\n"
    "                raise ValueError('PATCH_COMPHOLD: tri needs P >= 1')\n"
    "            desc += (' mode=tri P=%d b0=%r ms=%r lo=%r hi=%r peak=%r'\n"
    "                     % (self._ch_P, self._ch_b0, self._ch_ms, self._ch_lo, self._ch_hi,\n"
    "                        min(self._ch_hi, max(self._ch_lo, self._ch_b0 + self._ch_ms * self._ch_P))))\n"
    "        else:\n"
    "            self._ch_mode = 'rec'\n"
    "            ident = m.group(2)\n"
    "            path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ident + '.json')\n"
    "            try:\n"
    "                body = open(path, 'rb').read()\n"
    "                d = _json.loads(body.decode('utf-8'))\n"
    "            except (OSError, ValueError) as ex:\n"
    "                raise ValueError('PATCH_COMPHOLD: replay file %r unreadable (%s)' % (path, type(ex).__name__))\n"
    "            kn, kv = d.get('n') if isinstance(d, dict) else None, d.get('v') if isinstance(d, dict) else None\n"
    "            if not (isinstance(kn, list) and isinstance(kv, list) and len(kn) == len(kv) and len(kn) >= 2\n"
    "                    and all(type(x) is int for x in kn) and kn[0] >= 1\n"
    "                    and all(kn[i] < kn[i + 1] for i in range(len(kn) - 1))\n"
    "                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) and _math.isfinite(x)\n"
    "                            and self._ch_lo <= x <= self._ch_hi for x in kv)\n"
    "                    and d.get('id') == ident):\n"
    "                raise ValueError('PATCH_COMPHOLD: replay file %r is not {id, n: increasing ints >= 1, v: finite '\n"
    "                                 'floats inside the clamp}' % (path,))\n"
    "            self._ch_kn = [int(x) for x in kn]\n"
    "            self._ch_kv = [float(x) for x in kv]\n"
    "            desc += (' mode=rec id=%s sha256=%s knots=%d n0=%d n1=%d b0=%r lo=%r hi=%r vmax=%r vlast=%r'\n"
    "                     % (ident, _hashlib.sha256(body).hexdigest(), len(kn), kn[0], kn[-1], self._ch_b0, self._ch_lo,\n"
    "                        self._ch_hi, max(self._ch_kv), self._ch_kv[-1]))\n"
    "        self._ch_on = True\n"
    "        self._ch_set(self._ch_value(0))\n"
    "        print(desc, flush=True)\n"
    "\n"
    "    def _ch_value(self, n):\n"
    "        \"\"\"hold_c(n) as a python float.\"\"\"\n"
    "        if self._ch_mode == 'tri':\n"
    "            u = 0 if n <= 0 else min(n - 1, self._ch_P) - max(0, n - 1 - self._ch_P)\n"
    "            return min(self._ch_hi, max(self._ch_lo, self._ch_b0 + self._ch_ms * u))\n"
    "        kn, kv = self._ch_kn, self._ch_kv\n"
    "        if n < kn[0]:\n"
    "            x = self._ch_b0\n"
    "        elif n >= kn[-1]:\n"
    "            x = kv[-1]\n"
    "        else:\n"
    "            lo_, hi_ = 0, len(kn) - 1\n"
    "            while hi_ - lo_ > 1:\n"
    "                mid = (lo_ + hi_) // 2\n"
    "                if kn[mid] <= n:\n"
    "                    lo_ = mid\n"
    "                else:\n"
    "                    hi_ = mid\n"
    "            k = lo_\n"
    "            x = kv[k] + (kv[k + 1] - kv[k]) * (n - kn[k]) / float(kn[k + 1] - kn[k])\n"
    "        return min(self._ch_hi, max(self._ch_lo, x))\n"
    "\n"
    "    def _ch_set(self, value):\n"
    "        _b = self.beta[0].clone()\n"
    "        _b[self._ch_g] = value\n"
    "        self.beta[0] = _b\n"
    "\n"
    "    def _ch_apply(self):\n"
    "        \"\"\"After update n's Lion step, clamp and PATCH_BETAHOLD: record what Lion wrote for group c, then hold.\"\"\"\n"
    "        self._ch_n += 1\n"
    "        _nat = float(self.beta[0][self._ch_g].item())\n"
    "        self._ch_set(self._ch_value(self._ch_n))\n"
    "        _held = float(self.beta[0][self._ch_g].item())\n"
    "        self._ch_nat = _nat\n"
    "        if _held != _nat:\n"
    "            self._ch_active += 1\n"
    "\n"
    "    def _ch_attach(self, rec):\n"
    "        \"\"\"APPENDS keys to the probe record; never touches an existing key.\"\"\"\n"
    "        rec['ch_n'] = int(self._ch_n)\n"
    "        rec['ch_active'] = int(self._ch_active)\n"
    "        rec['ch_nat'] = self._ch_nat\n"
    "        rec['ch_held'] = float(self.beta[0][self._ch_g].item())\n"
    "    # -------------------------------------------------- end PATCH_COMPHOLD\n"
    "\n"
    "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n")

src2 = src.replace(a1, n1, 1).replace(a2, n2, 1).replace(a3, n3, 1).replace(a4, n4, 1)
bak = P + ".pre_comphold"
if os.path.exists(bak):
    if open(bak).read() != src:
        print("!!! %s exists and differs from the file being patched -- refusing" % bak)
        sys.exit(2)
else:
    open(bak, "w").write(src)
open(P, "w").write(src2)
print("PATCHED PATCH_COMPHOLD ->", P, "(backup", bak + ")")
