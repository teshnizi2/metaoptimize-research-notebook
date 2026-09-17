"""PATCH_RESTHOLD -- an OPT-IN, ENVIRONMENT-CONTROLLED override of the REST of the network's step-size group (the group
PATCH_GROUPHOLD does NOT hold), applied after every meta update, AFTER PATCH_GROUPHOLD's hold.  Written for `cvt8`
(CORRECTIONS 248).  Applied ONLY to the isolated tree $METAOPT_WS/harness_cvt8/cifar10 (bin/cVT8_stage_harness.sh), ON
TOP OF cvt7's PATCH_GROUPHOLD HF.py (sha 2396f2be...); the live shared harness and every earlier tree are never patched.

-------------------------------------------------------------------------------
WHY A NEW PATCH, AND NOT PATCH_COMPHOLD
-------------------------------------------------------------------------------
cvt6's PATCH_COMPHOLD holds the complement of PATCH_BETAHOLD's SINGLETON: it REQUIRES BETA_HOLD on (`_bh_on`), and its
anchors expect PATCH_BETAHOLD's apply / attach blocks to be followed directly by `_probe` / `_pt_attach`.  On ResNet18_c100
the rescuing arm (ISO) holds three carriers in ONE group, so cvt7 holds them with PATCH_GROUPHOLD, which in turn REFUSES
BETA_HOLD.  PATCH_COMPHOLD therefore cannot be applied to cvt7's tree without editing it (its guard and its anchors).  This
patch is PATCH_COMPHOLD's schedule logic, byte-for-byte the same arithmetic, keyed on PATCH_GROUPHOLD instead, under its
OWN switch, witness and probe keys -- so a COMP_HOLD run (cvt6) and a REST_HOLD run (cvt8) can never be confused by a
reader.  The name begins with R: no registered witness prefix (VOTE_W, BETA_HOLD, GROUP_HOLD, COMP_HOLD) is a prefix of it
and it is a prefix of none of them (the CORRECTIONS 245 rule), and no line of any landed .out begins with it.

    after Lion_meta_update, PATCH_CLIP, PATCH_BETAHOLD (off) and PATCH_GROUPHOLD on update n:   beta[0][r] := hold_r(n)

where r is the OTHER group of the 2-group grouping whose group PATCH_GROUPHOLD holds.

-------------------------------------------------------------------------------
THE SWITCH
-------------------------------------------------------------------------------
    REST_HOLD=tri:<P> | rec:<id>        (environment variable)

  * UNSET or EMPTY -> OFF.  OFF is the state every prior run was in (cvt7's tree, bitwise).
  * REQUIRES GROUP_HOLD ON in mode floor or tri (PATCH_GROUPHOLD's premises -- blockwise, Lion, BETA_CLIP, no VOTE_W, no
    BETA_HOLD, no HIER, no tn: -- are therefore already enforced), and EXACTLY 2 groups.  r = 1 - (GROUP_HOLD's group).
  * tri:<P>   the max-rate triangle of PATCH_BETAHOLD / PATCH_GROUPHOLD (u(0) = 0; u(n) = min(n-1, P) - max(0, n-1-P);
              hold_r(n) = min(hi, max(lo, b0 + ms*u(n))), b0 = group r's own initial beta).
  * rec:<id>  an EXACT REPLAY of a recorded trajectory: the JSON file <dir of HF.py>/<id>.json with lists `n`
              (strictly increasing ints >= 1) and `v` (finite floats inside [lo, hi]), len >= 2, and `id` == <id>:
                  n <  n[0]              hold_r(n) = b0
                  n[k] <= n < n[k+1]     hold_r(n) = v[k] + (v[k+1] - v[k]) * (n - n[k]) / float(n[k+1] - n[k])
                  n >= n[-1]             hold_r(n) = v[-1]
                  then min(hi, max(lo, .)), python double, stored into float32.
              The witness prints the file's sha256, so the replayed numbers are pinned by the witness line.
  * LOUD, never silently partial: a malformed value, GROUP_HOLD off, a grouping with other than 2 groups, a missing /
    unreadable / malformed replay file, a knot outside the clamp, an id mismatch -> ValueError at construction.  Every
    character of a value is in [A-Za-z0-9_.:], so REST_HOLD=... is ONE token of sbatch's comma-separated --export list.
  * A witness line is printed at construction on EVERY run (the runner's ENV line cannot carry REST_HOLD):
        REST_HOLD: off
        REST_HOLD: on type=blockwise group=<r> groupsize=<m> mode=tri P=<P> b0=<b0> ms=<ms> lo=<lo> hi=<hi> peak=<v>
        REST_HOLD: on type=blockwise group=<r> groupsize=<m> mode=rec id=<id> sha256=<64 hex> knots=<K> n0=<n> n1=<n> b0=<b0> lo=<lo> hi=<hi> vmax=<v> vlast=<v>
    (The tree also prints `VOTE_W: off`, `BETA_HOLD: off` and its GROUP_HOLD line on every run.)

-------------------------------------------------------------------------------
WHAT THE OVERRIDE DOES TO THE META-OPTIMISER'S STATE (Lion momentum) -- stated exactly (as 242.3(5))
-------------------------------------------------------------------------------
Lion_meta_update runs UNCHANGED on the whole beta vector; PATCH_CLIP clamps; PATCH_GROUPHOLD overwrites beta[0][g]; only
AFTER that is beta[0][r] overwritten.  With both groups held, NO applied beta depends on z or on momentum_meta:
  * both momentum_meta entries keep the unpatched recursion on their own raw z; they are NOT reset, frozen or copied;
  * they are DEAD STATE: the only reader of momentum_meta[i] is entry i's own Lion step, whose output is overwritten on the
    same update, from init (n = 0) to the last update, with no release;
  * PROBE_TENSOR's `beta_pre` for group r is the previous update's HELD value, its `mom_pre` / `z_agg` are the harness's
    own (dead) momentum and raw z -- which lets the scorer recompute, per record, the value Lion WROTE before the
    override and check it against `rh_nat`.

-------------------------------------------------------------------------------
INERTNESS
-------------------------------------------------------------------------------
FOUR insertions, every one ADDITIVE; NO existing line is edited:
  1. a guarded `self._rh_apply()` in step(), right after PATCH_GROUPHOLD's guarded `_gh_apply` region and before
     `self._probe(...)`;
  2. ONE call, `self._rh_init(net_param_names_and_size)`, right after PATCH_GROUPHOLD's `_gh_init` call in init_meta;
  3. a guarded `self._rh_attach(rec)` in _probe, right after PATCH_GROUPHOLD's guarded `_gh_attach` region and before
     `self._pt_attach(rec)`;
  4. the five new methods (`_rh_init`, `_rh_value`, `_rh_set`, `_rh_apply`, `_rh_attach`) inserted as one block right
     after PATCH_GROUPHOLD's method block, before `check_required_attributes`.
With REST_HOLD unset or empty, `_rh_on` is False, (1) and (3) do nothing, (2) prints `REST_HOLD: off` and returns: no
tensor is touched, the probe record gains no key.  tests/test_resthold.py proves the structure (deleting the four regions
reproduces cvt7's HF.py byte for byte) and the semantics on synthetic z; tests/test_resthold_realrun.py proves beta at
every step, the loss, probe.jsonl bytes and every final parameter and buffer BITWISE against cvt7's tree over a short
real CIFAR-100 ResNet18_c100 run (k01, ISO, and GROUP_HOLD floor / tri:8609 / tri:9428), and that each registered
REST_HOLD string bites.

The probe record of a REST_HOLD run gains four keys (appended after PATCH_GROUPHOLD's; no existing key touched):
  rh_n       meta updates completed so far (must equal step + 2)
  rh_active  cumulative number of updates on which the held value differed from the value Lion + clamp wrote
  rh_nat     the value Lion + clamp wrote for group r on THIS update, before the override
  rh_held    the value group r holds after the override (must equal the record's beta[r])
"""
import os, sys

P = os.environ.get("HF_PATH", "")
if not P:
    print("!!! HF_PATH is required: this patch is applied ONLY to an isolated tree's HF.py")
    sys.exit(2)
src = open(P).read()

if "PATCH_RESTHOLD" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# --- 0. the assumptions this patch rests on, checked BEFORE anything is written ----
assert "PATCH_GROUPHOLD" in src and "PATCH_BETAHOLD" in src and "PATCH_VOTEWEIGHT" in src, \
    "PATCH_RESTHOLD is applied on top of PATCH_GROUPHOLD (cvt7's HF.py)"
assert "PATCH_COMPHOLD" not in src and "COMP_HOLD" not in src, "cvt7's tree carries no PATCH_COMPHOLD"
assert "def Lion_meta_update(self,HtT_gradft):" in src, "Lion_meta_update is missing"
assert "_rh_" not in src, "unexpected pre-existing _rh_ names"
assert "REST_HOLD" not in src, "unexpected pre-existing REST_HOLD handling"

# --- 1. the apply, after PATCH_GROUPHOLD's and before the probe ----------------------
a1 = ("            # --- PATCH_GROUPHOLD ---\n"
      "            if getattr(self, '_gh_on', False):\n"
      "                self._gh_apply()\n"
      "            # --- end PATCH_GROUPHOLD ---\n"
      "            self._probe(HtT_gradft)  # PATCH_PROBE\n")
assert src.count(a1) == 1, "anchor 1 count=%d -- refusing to patch" % src.count(a1)
n1 = ("            # --- PATCH_GROUPHOLD ---\n"
      "            if getattr(self, '_gh_on', False):\n"
      "                self._gh_apply()\n"
      "            # --- end PATCH_GROUPHOLD ---\n"
      "            # --- PATCH_RESTHOLD ---\n"
      "            if getattr(self, '_rh_on', False):\n"
      "                self._rh_apply()\n"
      "            # --- end PATCH_RESTHOLD ---\n"
      "            self._probe(HtT_gradft)  # PATCH_PROBE\n")

# --- 2. the call in init_meta, after PATCH_GROUPHOLD's -------------------------------
a2 = "        self._gh_init(net_param_names_and_size)  # PATCH_GROUPHOLD\n"
assert src.count(a2) == 1, "anchor 2 count=%d -- refusing to patch" % src.count(a2)
n2 = a2 + "        self._rh_init(net_param_names_and_size)  # PATCH_RESTHOLD\n"

# --- 3. the attach in _probe, after PATCH_GROUPHOLD's and before PATCH_PROBE_TENSOR's --
a3 = ("        # --- PATCH_GROUPHOLD ---\n"
      "        if getattr(self, '_gh_on', False):\n"
      "            self._gh_attach(rec)\n"
      "        # --- end PATCH_GROUPHOLD ---\n"
      "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")
assert src.count(a3) == 1, "anchor 3 count=%d -- refusing to patch" % src.count(a3)
n3 = ("        # --- PATCH_GROUPHOLD ---\n"
      "        if getattr(self, '_gh_on', False):\n"
      "            self._gh_attach(rec)\n"
      "        # --- end PATCH_GROUPHOLD ---\n"
      "        # --- PATCH_RESTHOLD ---\n"
      "        if getattr(self, '_rh_on', False):\n"
      "            self._rh_attach(rec)\n"
      "        # --- end PATCH_RESTHOLD ---\n"
      "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n")

# --- 4. the methods, after PATCH_GROUPHOLD's block, before check_required_attributes --
a4 = ("    # ----------------------------------------------- end PATCH_GROUPHOLD\n"
      "\n"
      "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n")
assert src.count(a4) == 1, "anchor 4 count=%d -- refusing to patch" % src.count(a4)
n4 = (
    "    # ----------------------------------------------- end PATCH_GROUPHOLD\n"
    "\n"
    "    # ---------------------------------------------------- PATCH_RESTHOLD\n"
    "    def _rh_init(self, net_param_names_and_size):\n"
    "        \"\"\"Parse REST_HOLD, check every premise loudly, apply hold_r(0), print the witness.\"\"\"\n"
    "        import os as _os, re as _re, json as _json, hashlib as _hashlib, math as _math\n"
    "        self._rh_on = False\n"
    "        raw = _os.environ.get('REST_HOLD', '')\n"
    "        if raw == '':\n"
    "            print('REST_HOLD: off', flush=True)\n"
    "            return\n"
    "        m = _re.match(r'^(?:tri:([0-9]{1,6})|rec:([A-Za-z0-9_]{1,64}))$', raw)\n"
    "        if not m:\n"
    "            raise ValueError('PATCH_RESTHOLD: REST_HOLD %r is not tri:<P> | rec:<id>' % raw)\n"
    "        if not getattr(self, '_gh_on', False):\n"
    "            raise ValueError('PATCH_RESTHOLD: needs GROUP_HOLD on (REST_HOLD holds the OTHER group of its grouping)')\n"
    "        if self._gh_mode not in ('floor', 'tri'):\n"
    "            raise ValueError('PATCH_RESTHOLD: GROUP_HOLD mode %r is not floor or tri' % (self._gh_mode,))\n"
    "        if self.num_blocks != 2 or len(self.param_groups_indices) != 2:\n"
    "            raise ValueError('PATCH_RESTHOLD: needs exactly 2 groups; there are %d' % self.num_blocks)\n"
    "        self._rh_g = 1 - self._gh_g\n"
    "        self._rh_n = 0\n"
    "        self._rh_active = 0\n"
    "        self._rh_nat = None\n"
    "        self._rh_ms, self._rh_lo, self._rh_hi = self._gh_ms, self._gh_lo, self._gh_hi\n"
    "        self._rh_b0 = float(self.beta[0][self._rh_g].item())\n"
    "        self._rh_P = None\n"
    "        self._rh_kn = None\n"
    "        self._rh_kv = None\n"
    "        desc = 'REST_HOLD: on type=blockwise group=%d groupsize=%d' % (self._rh_g, len(self.param_groups_indices[self._rh_g]))\n"
    "        if m.group(1) is not None:\n"
    "            self._rh_mode = 'tri'\n"
    "            self._rh_P = int(m.group(1))\n"
    "            if self._rh_P < 1:\n"
    "                raise ValueError('PATCH_RESTHOLD: tri needs P >= 1')\n"
    "            desc += (' mode=tri P=%d b0=%r ms=%r lo=%r hi=%r peak=%r'\n"
    "                     % (self._rh_P, self._rh_b0, self._rh_ms, self._rh_lo, self._rh_hi,\n"
    "                        min(self._rh_hi, max(self._rh_lo, self._rh_b0 + self._rh_ms * self._rh_P))))\n"
    "        else:\n"
    "            self._rh_mode = 'rec'\n"
    "            ident = m.group(2)\n"
    "            path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ident + '.json')\n"
    "            try:\n"
    "                body = open(path, 'rb').read()\n"
    "                d = _json.loads(body.decode('utf-8'))\n"
    "            except (OSError, ValueError) as ex:\n"
    "                raise ValueError('PATCH_RESTHOLD: replay file %r unreadable (%s)' % (path, type(ex).__name__))\n"
    "            kn, kv = d.get('n') if isinstance(d, dict) else None, d.get('v') if isinstance(d, dict) else None\n"
    "            if not (isinstance(kn, list) and isinstance(kv, list) and len(kn) == len(kv) and len(kn) >= 2\n"
    "                    and all(type(x) is int for x in kn) and kn[0] >= 1\n"
    "                    and all(kn[i] < kn[i + 1] for i in range(len(kn) - 1))\n"
    "                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) and _math.isfinite(x)\n"
    "                            and self._rh_lo <= x <= self._rh_hi for x in kv)\n"
    "                    and d.get('id') == ident):\n"
    "                raise ValueError('PATCH_RESTHOLD: replay file %r is not {id, n: increasing ints >= 1, v: finite '\n"
    "                                 'floats inside the clamp}' % (path,))\n"
    "            self._rh_kn = [int(x) for x in kn]\n"
    "            self._rh_kv = [float(x) for x in kv]\n"
    "            desc += (' mode=rec id=%s sha256=%s knots=%d n0=%d n1=%d b0=%r lo=%r hi=%r vmax=%r vlast=%r'\n"
    "                     % (ident, _hashlib.sha256(body).hexdigest(), len(kn), kn[0], kn[-1], self._rh_b0, self._rh_lo,\n"
    "                        self._rh_hi, max(self._rh_kv), self._rh_kv[-1]))\n"
    "        self._rh_on = True\n"
    "        self._rh_set(self._rh_value(0))\n"
    "        print(desc, flush=True)\n"
    "\n"
    "    def _rh_value(self, n):\n"
    "        \"\"\"hold_r(n) as a python float.\"\"\"\n"
    "        if self._rh_mode == 'tri':\n"
    "            u = 0 if n <= 0 else min(n - 1, self._rh_P) - max(0, n - 1 - self._rh_P)\n"
    "            return min(self._rh_hi, max(self._rh_lo, self._rh_b0 + self._rh_ms * u))\n"
    "        kn, kv = self._rh_kn, self._rh_kv\n"
    "        if n < kn[0]:\n"
    "            x = self._rh_b0\n"
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
    "        return min(self._rh_hi, max(self._rh_lo, x))\n"
    "\n"
    "    def _rh_set(self, value):\n"
    "        _b = self.beta[0].clone()\n"
    "        _b[self._rh_g] = value\n"
    "        self.beta[0] = _b\n"
    "\n"
    "    def _rh_apply(self):\n"
    "        \"\"\"After update n's Lion step, clamp and PATCH_GROUPHOLD: record what Lion wrote for group r, then hold.\"\"\"\n"
    "        self._rh_n += 1\n"
    "        _nat = float(self.beta[0][self._rh_g].item())\n"
    "        self._rh_set(self._rh_value(self._rh_n))\n"
    "        _held = float(self.beta[0][self._rh_g].item())\n"
    "        self._rh_nat = _nat\n"
    "        if _held != _nat:\n"
    "            self._rh_active += 1\n"
    "\n"
    "    def _rh_attach(self, rec):\n"
    "        \"\"\"APPENDS keys to the probe record; never touches an existing key.\"\"\"\n"
    "        rec['rh_n'] = int(self._rh_n)\n"
    "        rec['rh_active'] = int(self._rh_active)\n"
    "        rec['rh_nat'] = self._rh_nat\n"
    "        rec['rh_held'] = float(self.beta[0][self._rh_g].item())\n"
    "    # ------------------------------------------------ end PATCH_RESTHOLD\n"
    "\n"
    "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n")

src2 = src.replace(a1, n1, 1).replace(a2, n2, 1).replace(a3, n3, 1).replace(a4, n4, 1)
bak = P + ".pre_resthold"
if os.path.exists(bak):
    if open(bak).read() != src:
        print("!!! %s exists and differs from the file being patched -- refusing" % bak)
        sys.exit(2)
else:
    open(bak, "w").write(src)
open(P, "w").write(src2)
print("PATCHED PATCH_RESTHOLD ->", P, "(backup", bak + ")")
