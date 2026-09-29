"""PATCH_DECAYROUTE -- an OPT-IN, ENVIRONMENT-CONTROLLED switch that routes the BASE optimiser's weight decay through the
weight update only, through the meta trace only, or replaces it by an alpha-INDEPENDENT decay.  CORRECTIONS 305
(ICML-PLAN row 1.6, stamp DECOUPLED-NOT-TESTED).  Applied ONLY to the isolated tree $METAOPT_WS/harness_cdr1/cifar10 (a
byte copy of the live tree at cgw1's pinned shas, HF.py 4732b74a...), built by bin/cDR1_stage_harness.sh; the live
shared harness and every registered tree are never patched.

-------------------------------------------------------------------------------
WHY
-------------------------------------------------------------------------------
Every campaign run uses the harness's ALPHA-SCALED decay (CORRECTIONS 292 NAMING).  For each base optimiser, with
a = exp(beta) the learned step size, u the base direction (SGD: g; SGDm: the momentum buffer; RMSProp / AdamW: the
preconditioned gradient / momentum; Lion: the sign) and wd = --weight-decay-base:

    delta = a*(u + wd*w);   w <- w - delta;   h <- gamma*(1 - wd*a)*h - delta                        (OFF, unchanged)

so ONE flag moves two routes at once: the weights shrink by a*wd per step, AND the meta trace's horizon is cut to
~1/(wd*a) (the factor) with the decay's own derivative a*wd*w in delta.  Its realised shrink moves with the learned a.
Three modes separate the routes and supply the untested alpha-independent control:

  shrink_only   delta = a*(u + wd*w)   w <- w - delta                   h <- gamma*h - delta
                (the weight update is the unpatched one, BITWISE; the trace FACTOR drops the decay; delta is the applied
                 change, so the trace keeps the direct term a*wd*w)
  trace_only    delta = a*u            w <- w - delta                   h <- gamma*(1 - wd*a)*h - delta
                (the weights are not decayed; the trace keeps the factor (1 - wd*a); delta is the applied change)
  alpha_indep:LAMBDA
                delta = a*u            w <- w - LAMBDA*w - delta        h <- gamma*(1 - LAMBDA)*h - delta
                (SGDW / AdamW as Loshchilov & Hutter arXiv:1711.05101 wrote them, with a CONSTANT schedule multiplier:
                 the shrink LAMBDA*w is outside the step size, the momentum and any preconditioner, and does not move
                 with the learned a.  REQUIRES --weight-decay-base 0: the alpha-scaled decay is REPLACED, not stacked.)

THE TRACE OF alpha_indep IS THE HYPERGRADIENT (derivation, CORRECTIONS 305.3).  h_t stands for dw_t/dbeta under the
harness's own approximation (it drops d u_t / d beta: the Hessian and the beta-dependence of momentum and
preconditioner -- exact when u does not depend on w, e.g. a linear loss).  From w_{t+1} = (1 - LAMBDA) w_t - a u_t,
a = e^beta, d a/d beta = a:
    dw_{t+1}/dbeta = (1 - LAMBDA) dw_t/dbeta - a u_t     =>     h_{t+1} = gamma (1 - LAMBDA) h_t - delta_t,  delta_t = a u_t
The decay adds NO term to delta: LAMBDA*w_t has no direct beta-dependence (unlike a*wd*w_t in the alpha-scaled form,
whose derivative a*wd*w_t is why OFF's delta carries it).  The horizon is 1/LAMBDA, fixed, not 1/(wd*a).
shrink_only and trace_only are INTERVENTIONS, deliberately NOT the hypergradient of their own weight update (whose
correct traces are, respectively, OFF's and h <- gamma*h - a*u); tests/test_decayroute.py R4 checks all four against a
finite difference on a linear-loss model.

-------------------------------------------------------------------------------
THE SWITCH
-------------------------------------------------------------------------------
    DECAY_ROUTE=shrink_only | trace_only | alpha_indep:<LAMBDA>          (environment variable)

  * UNSET or EMPTY -> OFF.  OFF is the state every prior run was in (the parent tree, bitwise).
  * <LAMBDA> matches [0-9]+(\\.[0-9]+)?(e-?[0-9]+)? and 0 < float(LAMBDA) < 1; it reaches the update as the Python float
    float(<LAMBDA>) (on a float32 tensor the product is formed in float32; the witness prints both).
  * Every base algorithm of HF.py: SGD, SGDm, RMSProp, AdamW, Lion.  The momentum / normaliser updates, the meta update,
    the clamp, PROBE and PROBE_TENSOR are untouched.
  * LOUD, never silently partial (ValueError at construction): any other value (full match), an unknown base algorithm,
    base weight decay None; shrink_only / trace_only with base weight decay <= 0 (vacuous); alpha_indep with base weight
    decay != 0, or with a NON-CONSTANT PATCH_SCHED multiplier (SCHED other than unset / `none`, or SCHED set with
    SCHED_WARMUP > 0: the registered form is the constant-schedule SGDW / AdamW; the campaign's runs pass SCHED=none,
    whose factor is exactly 1.0).  Every character of a
    legal value is in [a-z0-9_.:-], so DECAY_ROUTE=... is ONE token of sbatch's comma-separated --export list.
  * A witness line is printed at construction on EVERY run (the runner's ENV line cannot carry DECAY_ROUTE):
        DECAY_ROUTE: off
        DECAY_ROUTE: on mode=<mode> base=<alg> wd=<repr> lambda=<repr|na> lambda_f32=<repr|na> gamma=<repr>
    The prefix `DECAY_R` is distinct from DECAY_MASK's `DECAY_M`; no `startswith("DECAY_MASK")` reader selects it.

-------------------------------------------------------------------------------
HOW, AND WHAT IS NOT TOUCHED
-------------------------------------------------------------------------------
HF.__init__ binds `self.base_update = self.<alg>_base_update` BEFORE it calls init_meta.  When on, `_dr_init` (called in
init_meta) REBINDS it to `_dr_<alg>_base_update`, which keeps the original preamble, loop and momentum / normaliser lines
verbatim and routes each tensor's weight and trace update through `_dr_core`.  In shrink_only, `_dr_core`'s delta and
weight lines are the original expressions, so the weights are BITWISE the unpatched update's (R3).

THREE insertions, every one ADDITIVE; NO existing line is edited:
  1. ONE call, `self._dr_init()`, in init_meta right before `self.trace_meta = [0.0 for _ in  range(self.len_beta_list)]`;
  2. a guarded `self._dr_attach(rec)` in _probe, right before `self._pt_attach(rec)  # PATCH_PROBE_TENSOR`;
  3. the twelve new methods, one block right before `def check_required_attributes`.
With DECAY_ROUTE unset or empty, `_dr_on` is False, (2) does nothing, (1) prints `DECAY_ROUTE: off` and returns: no
attribute but `_dr_on` is set, base_update is not rebound, the probe record gains no key.

The probe record of a DECAY_ROUTE run gains keys (appended before PROBE_TENSOR's; no existing key touched):
  dr_mode            the mode
  dr_lam             LAMBDA (alpha_indep) or null
  dr_n               base updates completed so far (== step + 2 at a record, the harness's counter convention)
  dr_shrink_meas     MEASURED per-step shrink at this step: sum_i <w0_i - a_i*u_i - w1_i, w0_i> / sum_i ||w0_i||^2
                     (float64, from the weights before / after the update; alpha_indep -> LAMBDA, trace_only -> 0,
                     shrink_only -> the ||w||^2-weighted a*wd)
  dr_shrink_applied  numel-weighted mean of the registered shrink coefficient (a*wd | 0 | LAMBDA)
  dr_trace_decay     numel-weighted mean of the trace's decay factor excluding gamma (1 | 1 - wd*a | 1 - LAMBDA)
The measurement runs only at probe steps and only READS the weights (the old tensor object survives `w.data = ...`).
"""
import os
import sys

P = os.environ.get("HF_PATH", "")
if not P:
    print("!!! HF_PATH is required: this patch is applied ONLY to an isolated tree's HF.py")
    sys.exit(2)
src = open(P).read()

if "PATCH_DECAYROUTE" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

# --- 0. the assumptions this patch rests on, checked BEFORE anything is written ----
if "_dr_" in src or "DECAY_ROUTE" in src:
    print("!!! unexpected pre-existing _dr_ / DECAY_ROUTE names -- refusing to patch"); sys.exit(2)
if "PATCH_DECAYMASK" in src:
    print("!!! the base carries PATCH_DECAYMASK, which also rebinds base_update -- refusing to patch"); sys.exit(2)
if "PATCH_PROBE_TENSOR" not in src:
    print("!!! the base must carry PATCH_PROBE_TENSOR (the live HF.py 4732b74a...) -- refusing to patch"); sys.exit(2)
ORIG = [
    "            delta_w = a * (grad + self.args_base['weight_decay']*w.data)\n",
    "            delta = a * (self.momentum_base[i] + self.args_base['weight_decay']*w.data)\n",
    "            delta_w = a * (torch.div(grad, (mu_base*tr_+self.epsilon)**.5) + self.args_base['weight_decay']*w.data)\n",
    "            delta_w = a * (torch.sign(self.args_base['Lion_beta2'] * moment + (1-self.args_base['Lion_beta2'])*grad) + self.args_base['weight_decay'] * w.data)\n",
    "            delta_w = a * (torch.div(m, (mu_base*tr_+self.epsilon)**.5) + self.args_base['weight_decay']*w.data)\n",
]
for o in ORIG:
    if src.count(o) != 1:
        print("!!! a base update is not the registered text (%r) -- refusing to patch" % o.strip()[:60]); sys.exit(2)
for b in ("            self.base_update = self.SGD_base_update\n", "            self.base_update = self.SGDm_base_update \n",
          "            self.base_update = self.RMSProp_base_update\n", "            self.base_update = self.AdamW_base_update\n",
          "            self.base_update =self.Lion_base_update\n"):
    if src.count(b) != 1:
        print("!!! a base_update binding in __init__ moved (%r) -- refusing to patch" % b.strip()); sys.exit(2)

a1 = "        self.trace_meta = [0.0 for _ in  range(self.len_beta_list)]\n"
a2 = "        self._pt_attach(rec)  # PATCH_PROBE_TENSOR\n"
a3 = "    def check_required_attributes(self, args_base, args_meta, required_attributes_base, required_attributes_meta):\n"
for k, a in ((1, a1), (2, a2), (3, a3)):
    if src.count(a) != 1:
        print("!!! anchor %d count=%d -- refusing to patch" % (k, src.count(a))); sys.exit(2)

n1 = "        self._dr_init()  # PATCH_DECAYROUTE\n" + a1
n2 = ("        # --- PATCH_DECAYROUTE ---\n"
      "        if getattr(self, '_dr_on', False):\n"
      "            self._dr_attach(rec)\n"
      "        # --- end PATCH_DECAYROUTE ---\n") + a2
BLOCK = r'''    # ------------------------------------------------------ PATCH_DECAYROUTE
    def _dr_parse(self, raw, args_base, sched):
        """DECAY_ROUTE value -> (mode, LAMBDA or None).  Pure: reads only its arguments.  LOUD on anything unregistered."""
        import re as _re
        import math as _math
        _m = _re.fullmatch(r'(shrink_only|trace_only)|alpha_indep:([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)', raw)
        if _m is None:
            raise ValueError('PATCH_DECAYROUTE: DECAY_ROUTE %r is not shrink_only | trace_only | alpha_indep:<LAMBDA>' % (raw,))
        _alg = args_base.get('alg')
        if _alg not in ('SGD', 'SGDm', 'RMSProp', 'AdamW', 'Lion'):
            raise ValueError('PATCH_DECAYROUTE: base algorithm %r has no registered route' % (_alg,))
        _wd = args_base.get('weight_decay')
        if _wd is None:
            raise ValueError('PATCH_DECAYROUTE: base weight decay is None; the route is undefined')
        _wd = float(_wd)
        if _m.group(1) is not None:
            if not (_wd > 0.0):
                raise ValueError('PATCH_DECAYROUTE: %s needs base weight decay > 0, got %r (vacuous)' % (_m.group(1), _wd))
            return _m.group(1), None
        _lam = float(_m.group(2))
        if not (_math.isfinite(_lam) and 0.0 < _lam < 1.0):
            raise ValueError('PATCH_DECAYROUTE: LAMBDA %r is not in (0, 1)' % (_m.group(2),))
        if _wd != 0.0:
            raise ValueError('PATCH_DECAYROUTE: alpha_indep REPLACES the alpha-scaled decay; it needs --weight-decay-base 0, got %r' % (_wd,))
        if sched:
            raise ValueError('PATCH_DECAYROUTE: alpha_indep is registered for a constant schedule; SCHED=%r is set' % (sched,))
        return 'alpha_indep', _lam

    def _dr_init(self):
        """Read DECAY_ROUTE, check every premise loudly, rebind base_update, print the witness on EVERY run."""
        import os as _os
        import struct as _struct
        self._dr_on = False
        raw = _os.environ.get('DECAY_ROUTE', '')
        if raw == '':
            print('DECAY_ROUTE: off', flush=True)
            return
        _sc = getattr(self, '_sched', '')
        _nonconst = (_sc not in ('', 'none')) or (bool(_sc) and getattr(self, '_sched_warm', 0) > 0)
        mode, lam = self._dr_parse(raw, self.args_base,
                                   ('SCHED=%s SCHED_WARMUP=%s' % (_sc, getattr(self, '_sched_warm', 0))) if _nonconst else '')
        self._dr_mode = mode
        self._dr_lam = lam
        self._dr_n = 0
        self._dr_every = int(_os.environ.get('PROBE', '0') or 0)
        self._dr_acc = None
        self._dr_meas = None
        _alg = self.args_base['alg']
        self.base_update = {'SGD': self._dr_SGD_base_update, 'SGDm': self._dr_SGDm_base_update,
                            'RMSProp': self._dr_RMSProp_base_update, 'AdamW': self._dr_AdamW_base_update,
                            'Lion': self._dr_Lion_base_update}[_alg]
        self._dr_on = True
        print('DECAY_ROUTE: on mode=%s base=%s wd=%r lambda=%s lambda_f32=%s gamma=%r'
              % (mode, _alg, float(self.args_base['weight_decay']), 'na' if lam is None else repr(lam),
                 'na' if lam is None else repr(_struct.unpack('f', _struct.pack('f', lam))[0]), self.gamma), flush=True)

    def _dr_meas_begin(self):
        """At a probe step, open the realised-shrink accumulators (read-only on the optimiser)."""
        if self._dr_every > 0 and self.counter % self._dr_every == 0:
            self._dr_acc = [0.0, 0.0, 0.0, 0.0, 0]
        else:
            self._dr_acc = None

    def _dr_meas_add(self, w0, dstep, w1, shrink_coef, trace_coef):
        _w0 = w0.double()
        _d = torch.as_tensor(dstep, dtype=torch.float64, device=_w0.device)
        _r = _w0 - _d - w1.double()
        _acc = self._dr_acc
        _acc[0] += float((_r * _w0).sum())
        _acc[1] += float((_w0 * _w0).sum())
        _acc[2] += float(torch.broadcast_to(torch.as_tensor(shrink_coef, dtype=torch.float64, device=_w0.device), w0.shape).sum())
        _acc[3] += float(torch.broadcast_to(torch.as_tensor(trace_coef, dtype=torch.float64, device=_w0.device), w0.shape).sum())
        _acc[4] += int(w0.numel())

    def _dr_meas_end(self):
        self._dr_n += 1
        _acc = self._dr_acc
        if _acc is None:
            return
        self._dr_meas = {'step': int(self.counter), 'shrink_meas': (_acc[0] / _acc[1]) if _acc[1] > 0 else None,
                         'shrink_applied': _acc[2] / _acc[4], 'trace_decay': _acc[3] / _acc[4]}
        self._dr_acc = None

    def _dr_core(self, w, a, u, i):
        """ONE tensor's weight and trace update under the registered route; u = the base direction, a*u = the step."""
        _w0 = w.data
        if self._dr_mode == 'shrink_only':
            delta = a * (u + self.args_base['weight_decay']*w.data)
            w.data = w.data - delta
            self.h_condenced[i] = self.gamma*self.h_condenced[i] - delta
            if self._dr_acc is not None:
                self._dr_meas_add(_w0, a * u, w.data, self.args_base['weight_decay']*a, 1.0)
        elif self._dr_mode == 'trace_only':
            delta = a * u
            w.data = w.data - delta
            self.h_condenced[i] = self.gamma*(1-self.args_base['weight_decay']*a)*self.h_condenced[i] - delta
            if self._dr_acc is not None:
                self._dr_meas_add(_w0, delta, w.data, 0.0, 1-self.args_base['weight_decay']*a)
        else:
            delta = a * u
            w.data = w.data - self._dr_lam*w.data - delta
            self.h_condenced[i] = self.gamma*(1-self._dr_lam)*self.h_condenced[i] - delta
            if self._dr_acc is not None:
                self._dr_meas_add(_w0, delta, w.data, self._dr_lam, 1-self._dr_lam)

    def _dr_SGD_base_update(self,net,g):
        self._dr_meas_begin()
        for w, grad, a ,i in zip(net.parameters(), g, self.alpha, range(self.num_layers)):
            self._dr_core(w, a, grad, i)
        self._dr_meas_end()

    def _dr_SGDm_base_update(self,net,g):
        self._dr_meas_begin()
        for w, grad, a ,i in zip(net.parameters(), g, self.alpha, range(self.num_layers)):
            self._dr_core(w, a, self.momentum_base[i], i)
            self.momentum_base[i] = self.args_base['momentum_param']*self.momentum_base[i] + (1 - self.args_base['momentum_param'])*grad
        self._dr_meas_end()

    def _dr_RMSProp_base_update(self,net,g):
        self._dr_meas_begin()
        self.lambda_base_t *= self.args_base['normalizer_param']
        mu_base = (1-self.args_base['normalizer_param'])/(1-self.lambda_base_t)
        self.trace_base = [self.args_base['normalizer_param']*self.trace_base[i] + g[i]**2 for i in range(self.num_layers)]
        for w, grad, a, tr_ ,i in zip(net.parameters(), g, self.alpha, self.trace_base, range(self.num_layers)):
            self._dr_core(w, a, torch.div(grad, (mu_base*tr_+self.epsilon)**.5), i)
        self._dr_meas_end()

    def _dr_Lion_base_update(self,net,g):
        self._dr_meas_begin()
        for w, grad, a, moment, i in zip(net.parameters(), g, self.alpha, self.momentum_base, range(self.num_layers)):
            self._dr_core(w, a, torch.sign(self.args_base['Lion_beta2'] * moment + (1-self.args_base['Lion_beta2'])*grad), i)
            self.momentum_base[i] = self.args_base['momentum_param'] * moment + (1-self.args_base['momentum_param'])*grad
        self._dr_meas_end()

    def _dr_AdamW_base_update(self,net,g):
        self._dr_meas_begin()
        self.lambda_base_t *= self.args_base['normalizer_param']
        mu_base = (1-self.args_base['normalizer_param'])/(1-self.lambda_base_t)
        self.momentum_base = [self.args_base['momentum_param']*self.momentum_base[i] + g[i] for i in range(self.num_layers)]
        self.trace_base = [self.args_base['normalizer_param']*self.trace_base[i] + g[i]**2 for i in range(self.num_layers)]
        for w, m, a, tr_ ,i in zip(net.parameters(), self.momentum_base, self.alpha, self.trace_base, range(self.num_layers)):
            self._dr_core(w, a, torch.div(m, (mu_base*tr_+self.epsilon)**.5), i)
        self._dr_meas_end()

    def _dr_attach(self, rec):
        """APPENDS keys to the probe record; never touches an existing key."""
        rec['dr_mode'] = self._dr_mode
        rec['dr_lam'] = self._dr_lam
        rec['dr_n'] = int(self._dr_n)
        _M = self._dr_meas
        if _M is not None and _M['step'] == rec.get('step'):
            rec['dr_shrink_meas'] = _M['shrink_meas']
            rec['dr_shrink_applied'] = _M['shrink_applied']
            rec['dr_trace_decay'] = _M['trace_decay']
        else:
            rec['dr_shrink_meas'] = None
            rec['dr_shrink_applied'] = None
            rec['dr_trace_decay'] = None
    # -------------------------------------------------- end PATCH_DECAYROUTE

'''
n3 = BLOCK + a3

src2 = src.replace(a1, n1, 1).replace(a2, n2, 1).replace(a3, n3, 1)
bak = P + ".pre_decayroute"
if os.path.exists(bak):
    if open(bak).read() != src:
        print("!!! %s exists and differs from the file being patched -- refusing" % bak)
        sys.exit(2)
else:
    open(bak, "w").write(src)
open(P, "w").write(src2)
print("PATCHED PATCH_DECAYROUTE ->", P, "(backup", bak + ")")
