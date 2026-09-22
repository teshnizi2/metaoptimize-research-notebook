"""THE HARNESS-NULL PROOF FOR `caw1` (CORRECTIONS 290), ON THE REAL GPU PATH.

`caw1` adds NO harness code: AdamW / Adam are existing algs of HF.py and every flag is an existing CLI flag of train.py.
But no landed run of this campaign has ever put an AdamW base on ResNet18_c100 / CIFAR-100, none has run it with
PROBE_TENSOR=1, and the batch's readings rest on premises no registered proof covers:

  THE HARNESS-NULL: that the runs are NOT what the ARGS say -- a flag that does not reach the update (the base or meta
  alg silently falling back to a default, the weight decay not entering the step, the two grains being the same
  experiment).  A NO-COLLAPSE reading produced by such a run would be a harness artefact, and this job is what
  separates the two.

On cwd1's tree, UNCHANGED, with the launcher's own environment, each arm's OWN ARGS (read through train.py's own
parse_args and its own -1 -> None conversion), 300 real CIFAR-100 steps per variant:

  RA1  THE AdamW BASE, BITWISE, AT BOTH REGISTERED DECAYS AND BOTH GRAINS.  At every step and every tensor:
           lambda' = lambda*b2 ; mu = (1-b2)/(1-lambda') ; m' = b1*m + g ; v' = b2*v + g**2
           w' = w - a*(m'/sqrt(mu*v' + eps) + wd*w) ; h' = gamma*(1 - wd*a)*h - a*(m'/sqrt(mu*v' + eps) + wd*w)
       with THE PASSED wd.  NON-VACUITY, three ways: the same run checked against the formula at the OTHER decay
       FAILS; checked against the campaign's SGDm base formula FAILS (the base did not fall back to SGDm); and the
       optimiser's own args_base carries alg AdamW and the passed wd.
  RA2  THE Adam META, BITWISE: beta' = (1 - ms*wdm)*beta - ms*M'/sqrt(mu_m*T' + eps), M' = b1m*M + z, T' = b2m*T + z**2,
       lambda_m' = lambda_m*b2m, at every step; the same run checked against the Lion meta formula FAILS.
  RA3  THE POSITIVE CONTROL K01 (SGDm 0.99 + Lion): SGDm base and Lion meta BITWISE -- the control is the mechanism cell.
  RA4  GRAIN: scalar arms carry ONE step size and layerwise arms 62 (HF.stepsize_type and the beta vector's length).
  RA5  WITNESSES: each run prints exactly the five `off` lines and ONE `PROBE_TENSOR: on ...` line equal to the design's
       witness for its arm (grain, 62 tensors, the META alg's own constants) followed by ` dir=`.
  RA6  THE VOTE DECOMPOSITION THE GATE READS (DOM_C, CORRECTIONS 290.4) HOLDS ON REAL RECORDS: on every PROBE_TENSOR
       record of the Adam-meta scalar arm, sum_i L_i with L_i = b1m/(1-b1m)*m_tensor_i + z_tensor_i equals
       b1m*mom_pre + z_agg within 1e-3 of sum_i |L_i| and has its sign; the same for the Lion control with
       L_i = b2*m_tensor_i + (1-b2)*z_tensor_i.  (No Adam-meta PROBE_TENSOR record exists anywhere in the corpus.)
  RA7  DETERMINISM: AS1 run twice is bitwise identical.
  RA8  THE ARMS ARE DISTINCT EXPERIMENTS: the five arms' final weights are PAIRWISE DISTINCT.
  RA9  WHICH ROUTE THE DECAY BITES AT alpha0 1e-6, MEASURED: the number of tensor-steps at which the decay term changes
       the float32 weight update (it is a*wd*|w| ~ 1e-7 |w| at wd 0.1) is COUNTED, not assumed; and at a TEST-ONLY
       alpha0 1e-2 the AdamW formula still holds bitwise and the wd-0.1 and wd-1e-2 runs' final weights differ.

Nothing registered is imported except analysis/caw1_design.py (the arm table) and tests/test_grouphold_realrun.py
(chk / FAILED), both UNEDITED.

  python3 tests/test_caw1_realrun.py --tree $WS/harness_cwd1/cifar10 --steps 300 --every 10 --seed 149 --work /tmp/caw1_rr
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
KEYS_OFF = ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "COMP_HOLD", "REST_HOLD", "WINDOW_HOLD", "DECAY_MASK")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# =====================================================================================================================
# THE CHILD: one arm's own ARGS through train.py's own parse_args, with every update verified against its formula
# =====================================================================================================================
def child(a):
    tree = a.tree
    os.chdir(tree)
    sys.path.insert(0, tree)
    import numpy as np
    import torch
    import torch.nn as nn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from torch.utils.tensorboard import SummaryWriter
    from build_network import build_network
    from Optimizers.build_optimizer import build_optimizer
    from load_data import load_data
    import ast as _ast, argparse as _argparse, types as _types
    _tsrc = open(os.path.join(tree, "train.py")).read()
    _fn = [n for n in _ast.parse(_tsrc).body if isinstance(n, _ast.FunctionDef) and n.name == "parse_args"]
    assert len(_fn) == 1, "train.py has no single parse_args"
    TR = _types.ModuleType("train_parse_args")
    TR.argparse = _argparse
    exec(compile(_ast.Module(body=_fn, type_ignores=[]), os.path.join(tree, "train.py"), "exec"), TR.__dict__)
    sys.argv = ["train.py"] + json.loads(a.argv)
    args = TR.parse_args()
    for k in ("normalizer_param_base", "momentum_param_base", "weight_decay_base", "Lion_beta2_base",
              "normalizer_param_meta", "momentum_param_meta", "weight_decay_meta", "Lion_beta2_meta",
              "meta_stepsize"):
        if getattr(args, k) == -1:
            setattr(args, k, None)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    writer = SummaryWriter(os.path.join(args.save_directory, "Tensorboard_outputs", args.run_name))
    criterion = nn.CrossEntropyLoss().to(args.device)
    net = build_network(args.NN_name, args.device)
    opt = build_optimizer(net, args, writer)
    trainloader, _t = load_data(args.dataset, args.batch_size, args.seed)
    params = list(net.parameters())
    AB, AM = opt.args_base, opt.args_meta
    wd = AB["weight_decay"]
    wd_other = float(a.wrong_wd)
    eps = opt.epsilon
    # stats: [w, h, m, v] mismatches against: the expected formula / the formula at the other wd / the SGDm formula
    st = {"expect": [0, 0, 0, 0], "wrongwd": [0, 0, 0, 0], "sgdm": [0, 0, 0, 0],
          "meta_expect": 0, "meta_wrongalg": 0, "decay_bites": 0, "tensor_steps": 0}
    ob, om = opt.base_update, opt.meta_update

    def base_wrapped(netx, g):
        wb = [p.data.clone() for p in params]
        mb = [m.clone() if torch.is_tensor(m) else m for m in opt.momentum_base]
        vb = [v.clone() if torch.is_tensor(v) else v for v in opt.trace_base]
        hb = [h.clone() for h in opt.h_condenced]
        lam_pre = opt.lambda_base_t
        ob(netx, g)
        gm = opt.gamma
        if AB["alg"] == "AdamW":
            b1, b2 = AB["momentum_param"], AB["normalizer_param"]
            lam = lam_pre * b2
            mu = (1 - b2) / (1 - lam)
            for i, p in enumerate(params):
                al = opt.alpha[i]
                m_e = b1 * mb[i] + g[i]
                v_e = b2 * vb[i] + g[i] ** 2
                step = torch.div(m_e, (mu * v_e + eps) ** .5)
                for lab, wdx in (("expect", wd), ("wrongwd", wd_other)):
                    d = al * (step + wdx * wb[i])
                    w_e, h_e = wb[i] - d, gm * (1 - wdx * al) * hb[i] - d
                    st[lab][0] += not torch.equal(p.data, w_e)
                    st[lab][1] += not torch.equal(opt.h_condenced[i], h_e)
                    st[lab][2] += not torch.equal(opt.momentum_base[i], m_e)
                    st[lab][3] += not torch.equal(opt.trace_base[i], v_e)
                st["tensor_steps"] += 1
                st["decay_bites"] += not torch.equal(wb[i] - al * (step + wd * wb[i]), wb[i] - al * step)
                # the harness-null: an SGDm base with the same momentum constant
                ms_e = b1 * mb[i] + (1 - b1) * g[i]
                d2 = al * (mb[i] + wd * wb[i])
                st["sgdm"][0] += not torch.equal(p.data, wb[i] - d2)
                st["sgdm"][2] += not torch.equal(opt.momentum_base[i], ms_e)
            st["lam_ok"] = st.get("lam_ok", 0) + (opt.lambda_base_t == lam)
        elif AB["alg"] == "SGDm":
            mp = AB["momentum_param"]
            for i, p in enumerate(params):
                al = opt.alpha[i]
                for lab, wdx in (("expect", wd), ("wrongwd", wd_other)):
                    d = al * (mb[i] + wdx * wb[i])
                    st[lab][0] += not torch.equal(p.data, wb[i] - d)
                    st[lab][1] += not torch.equal(opt.h_condenced[i], gm * (1 - wdx * al) * hb[i] - d)
                    st[lab][2] += not torch.equal(opt.momentum_base[i], mp * mb[i] + (1 - mp) * g[i])
                st["tensor_steps"] += 1
                st["decay_bites"] += not torch.equal(wb[i] - al * (mb[i] + wd * wb[i]), wb[i] - al * mb[i])
        else:
            st["unexpected_base"] = AB["alg"]

    def meta_wrapped(z):
        bb = [b.clone() for b in opt.beta]
        Mb = [m.clone() if torch.is_tensor(m) else m for m in opt.momentum_meta]
        Tb = [t.clone() if torch.is_tensor(t) else t for t in opt.trace_meta] if hasattr(opt, "trace_meta") else None
        lam_pre = getattr(opt, "lambda_meta_t", None)
        om(z)
        ms, wdm = AM["meta_stepsize"], AM["weight_decay"]
        if AM["alg"] == "Adam":
            b1, b2 = AM["momentum_param"], AM["normalizer_param"]
            lam = lam_pre * b2
            mu = (1 - b2) / (1 - lam)
            for i in range(opt.len_beta_list):
                M = b1 * Mb[i] + z[i]
                T = b2 * Tb[i] + z[i] ** 2
                be = (1 - ms * wdm) * bb[i] - torch.div(ms * M, (mu * T + eps) ** .5)
                st["meta_expect"] += not (torch.equal(opt.beta[i], be) and torch.equal(opt.momentum_meta[i], M)
                                          and torch.equal(opt.trace_meta[i], T))
                bl = (1 - ms * wdm) * bb[i] - ms * torch.sign(0.9 * Mb[i] + 0.1 * z[i])
                st["meta_wrongalg"] += not torch.equal(opt.beta[i], bl)
        elif AM["alg"] == "Lion":
            mp, b2 = AM["momentum_param"], AM["Lion_beta2"]
            for i in range(opt.len_beta_list):
                be = (1 - ms * wdm) * bb[i] - ms * torch.sign(b2 * Mb[i] + (1 - b2) * z[i])
                Mn = mp * Mb[i] + (1 - mp) * z[i]
                st["meta_expect"] += not (torch.equal(opt.beta[i], be) and torch.equal(opt.momentum_meta[i], Mn))
                ba = bb[i] - torch.div(ms * (0.9 * Mb[i] + z[i]), (0.001 * (z[i] ** 2) + eps) ** .5)
                st["meta_wrongalg"] += not torch.equal(opt.beta[i], ba)
        else:
            st["unexpected_meta"] = AM["alg"]

    opt.base_update = base_wrapped
    opt.meta_update = meta_wrapped
    t = 0
    while t < a.steps:
        for data in trainloader:
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            loss = criterion(net(inputs), labels)
            opt.step(net, loss)
            t += 1
            if t >= a.steps:
                break
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    writer.close()
    json.dump({"stats": st, "steps": t, "weights_sha256": h.hexdigest(), "alg_base": AB["alg"], "alg_meta": AM["alg"],
               "wd_seen": float(wd), "stepsize_type": opt.stepsize_type,
               "n_beta": int(opt.beta[0].reshape(-1).numel()), "len_beta_list": int(opt.len_beta_list),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}, open(a.out, "w"))


def run_arm(a, D, tag, arm, alpha0=None, wrong_wd=""):
    work = os.path.join(a.work, tag)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    pairs = D.args_pairs(arm, a.seed, work)
    if alpha0 is not None:
        pairs = [(f, alpha0 if f == "alpha0" else v) for f, v in pairs]
    argv = []
    for f, v in pairs:
        argv += ["--" + f, v]
    e = dict(os.environ)
    for k in KEYS_OFF:
        e.pop(k, None)
    # THE LAUNCHER'S OWN ENVIRONMENT (bin/cAW1_stdrecipe.sh's --export list), byte for byte, with a short probe period
    e.update({"AUGMENT": "1", "BETA_CLIP": "-15:-2.3026", "HIER": "none", "SCHED": "none", "PROBE": str(a.every),
              "PROBE_TENSOR": "1", "PROBE_DIR": os.path.join(work, "probe"), "PYTHONDONTWRITEBYTECODE": "1",
              "PYTHONUNBUFFERED": "1"})
    out = os.path.join(work, "result.json")
    ww = wrong_wd or ("1e-2" if D.WD[arm] == "0.1" else "0.1")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", "--tree", a.tree, "--argv", json.dumps(argv),
                        "--wrong-wd", ww, "--steps", str(a.steps), "--out", out],
                       env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode != 0 or not os.path.exists(out):
        print(p.stdout[-3000:])
        return None
    r = json.load(open(out))
    r["stdout"] = p.stdout.splitlines()
    pj = os.path.join(work, "probe", "probe.jsonl")
    r["records"] = [json.loads(x) for x in open(pj)] if os.path.exists(pj) else []
    print("  ran %-14s %-4s alpha0=%-5s base=%s meta=%s wd=%s type=%s n_beta=%d  %d steps on %s  %s  weights %s..."
          % (tag, arm, alpha0 or D.A0, r["alg_base"], r["alg_meta"], r["wd_seen"], r["stepsize_type"], r["n_beta"],
             r["steps"], r["gpu"], json.dumps(r["stats"]), r["weights_sha256"][:12]))
    return r


def decomposition(recs, pair):
    """RA6 on real records: -> (n, bad_sum, bad_sign)."""
    n = bad = bads = 0
    for r in recs:
        if "z_tensor" not in r:
            continue
        n += 1
        mp, b2 = r["pt_mp"], r["pt_b2"]
        z, m = r["z_tensor"], r["m_tensor"]
        if pair == "AW":
            L = [mp / (1 - mp) * mi + zi for mi, zi in zip(m, z)]
            app = mp * r["mom_pre"][0][0] + r["z_agg"][0][0]
        else:
            L = [b2 * mi + (1 - b2) * zi for mi, zi in zip(m, z)]
            app = b2 * r["mom_pre"][0][0] + (1 - b2) * r["z_agg"][0][0]
        s = sum(L)
        scale = sum(abs(x) for x in L)
        # the scorer's own SCALE-relative tolerance (1e-3 of sum_i |L_i|), not a relative one on the net term
        bad += not (abs(s - app) <= 1e-3 * scale or abs(s - app) <= 1e-30)
        bads += ((s > 0) != (app > 0)) and abs(app) > 1e-3 * scale
    return n, bad, bads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--tree", default="")
    ap.add_argument("--argv", default="[]")
    ap.add_argument("--wrong-wd", default="1e-2")
    ap.add_argument("--out", default="")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=149)
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    if a.child:
        child(a)
        return
    D = _load("caw1_design", os.path.join(REPO, "analysis", "caw1_design.py"))
    RR = _load("test_grouphold_realrun", os.path.join(HERE, "test_grouphold_realrun.py"))
    chk = RR.chk
    a.tree = os.path.abspath(a.tree)
    os.makedirs(a.work, exist_ok=True)
    for k in KEYS_OFF:
        os.environ.pop(k, None)

    def sh(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    print("BATCH caw1")
    print("DRIVER_SHA256 %s" % sh(os.path.abspath(__file__)))
    print("DESIGN_SHA256 %s" % sh(D.__file__))
    print("CWD_DESIGN_SHA256 %s" % sh(D.W.__file__))
    print("REGISTERED_TEST_SHA256 %s" % sh(RR.__file__))
    print("TREE %s HF %s build_network %s train %s" % (a.tree, sh(os.path.join(a.tree, "Optimizers", "HF.py")),
                                                        sh(os.path.join(a.tree, "build_network.py")),
                                                        sh(os.path.join(a.tree, "train.py"))))
    print("ARMS %s" % ",".join("%s=%s/%s/%s" % (x, D.PAIR_OF[x], D.SPEC[x], D.WD[x]) for x in D.ARMS))
    chk(sh(os.path.join(a.tree, "Optimizers", "HF.py")) == D.HF_SHA, "the tree's HF.py is cwd1's, UNCHANGED (94aedc33...)")
    chk(sh(os.path.join(a.tree, "build_network.py")) == D.BN_SHA, "the tree's build_network.py is cvt8's (c7998883...)")
    chk(sh(D.W.__file__) == D.CWD_DESIGN_SHA, "analysis/cwd_design.py is the registered file, UNEDITED (RULE 16)")
    print("\nsteps %d, PROBE every %d, PROBE_TENSOR=1, seed %d, AUGMENT=1, BETA_CLIP -15:-2.3026, HIER=none, net %s"
          % (a.steps, a.every, a.seed, D.NET))

    runs = {}
    print("\nRA1-RA5 EACH ARM'S OWN ARGS, THROUGH train.py's parse_args, ON THE REAL GPU PATH")
    for arm in D.ARMS:
        V = run_arm(a, D, "arm_" + arm, arm)
        runs[arm] = V
        if V is None:
            chk(False, "RA %s ran" % arm)
            continue
        pair = D.PAIR_OF[arm]
        s = V["stats"]
        base = "AdamW" if pair == "AW" else "SGDm"
        meta = D.META_ALG[pair]
        chk(V["steps"] == a.steps and V["alg_base"] == base and V["alg_meta"] == meta and V["wd_seen"] == D.WDF[arm],
            "RA1 %s the optimiser's own args carry base %s, meta %s and wd %r" % (arm, base, meta, D.WDF[arm]))
        nv = 4 if pair == "AW" else 3
        chk(s["expect"][:nv] == [0] * nv and "unexpected_base" not in s,
            "RA%s %s %s base: w', h', m'%s BITWISE the formula with THE PASSED wd %s at every one of %d steps, every"
            " tensor" % ("1" if pair == "AW" else "3", arm, base, ", v'" if pair == "AW" else "", D.WD[arm], a.steps),
            str(s["expect"]))
        chk(sum(s["wrongwd"]) > 0,
            "RA1 %s NON-VACUITY: the same run checked against the formula at the OTHER decay FAILS" % arm,
            "wrongwd %s" % s["wrongwd"])
        if pair == "AW":
            chk(s.get("lam_ok") == a.steps, "RA1 %s the second-moment bias-correction clock lambda advanced as b2^t" % arm)
            chk(s["sgdm"][0] > 0 and s["sgdm"][2] > 0,
                "RA1 %s THE HARNESS-NULL IS EXCLUDED: the run FAILS the SGDm base formula (w and m)" % arm, str(s["sgdm"]))
        chk(s["meta_expect"] == 0 and "unexpected_meta" not in s,
            "RA%s %s %s meta: beta (and its moments) BITWISE the formula at every step" % ("2" if pair == "AW" else "3",
                                                                                          arm, meta))
        chk(s["meta_wrongalg"] > 0, "RA2 %s NON-VACUITY: the same run FAILS the other meta alg's formula" % arm,
            "wrongalg %d" % s["meta_wrongalg"])
        want_n = 1 if D.SPEC[arm] == "scalar" else D.NTENS
        chk(V["stepsize_type"] == D.SPEC[arm] and V["n_beta"] == want_n and V["len_beta_list"] == 1,
            "RA4 %s grain: stepsize_type=%s with %d step size(s)" % (arm, D.SPEC[arm], want_n),
            "%s / %d" % (V["stepsize_type"], V["n_beta"]))
        offs = dict((k, [ln for ln in V["stdout"] if ln.startswith(k)]) for k in KEYS_OFF)
        chk(all(offs[k.split(":")[0]] == [k] for k in D.OFF_LINES) and not offs["COMP_HOLD"] and not offs["WINDOW_HOLD"],
            "RA5 %s printed exactly the five `off` witnesses and no COMP_HOLD / WINDOW_HOLD line" % arm)
        pt = [ln for ln in V["stdout"] if ln.startswith("PROBE_TENSOR:")]
        # this job probes every a.every steps; the batch every D.PROBE -- the witness is compared with that ONE token
        # substituted, and nothing else (a test-only defect of the first proof job 5079156, CORRECTIONS 290.6)
        wit = D.pt_witness_prefix(arm).replace("every=%d " % D.PROBE, "every=%d " % a.every, 1)
        chk(len(pt) == 1 and pt[0].startswith(wit + " dir="),
            "RA5 %s printed ONE PROBE_TENSOR line == the design's witness (every=%d here) `%s dir=...`" % (arm, a.every, wit),
            (pt or ["(none)"])[0][:160])
        n_rec = len(V["records"])
        chk(n_rec == a.steps // a.every and all("z_tensor" in r and len(r["z_tensor"]) == D.NTENS for r in V["records"]),
            "RA5 %s wrote %d probe records, each with a 62-tensor z_tensor" % (arm, a.steps // a.every), "%d" % n_rec)
        print("  RA9 %s decay term changed the float32 weight update at %d of %d tensor-steps (alpha0 %s, wd %s)"
              % (arm, s["decay_bites"], s["tensor_steps"], D.A0, D.WD[arm]))

    print("\nRA6 THE VOTE DECOMPOSITION DOM_C READS, ON REAL RECORDS")
    for arm in D.SCALAR_ARMS:
        V = runs.get(arm)
        if V is None:
            continue
        n, bad, bads = decomposition(V["records"], D.PAIR_OF[arm])
        chk(n == a.steps // a.every and bad == 0 and bads == 0,
            "RA6 %s sum_i L_i == the harness's applied momentum term (within 1e-3 of sum|L_i|) and has its sign on every record" % arm,
            "%d records, %d off, %d sign" % (n, bad, bads))

    print("\nRA7 DETERMINISM (AS1 twice)")
    V2 = run_arm(a, D, "arm_AS1_again", "AS1")
    chk(V2 is not None and runs.get("AS1") is not None and V2["weights_sha256"] == runs["AS1"]["weights_sha256"],
        "RA7 AS1 run twice: identical final weights")

    print("\nRA8 THE FIVE ARMS ARE DISTINCT EXPERIMENTS")
    shas = dict((k, v["weights_sha256"]) for k, v in runs.items() if v is not None)
    chk(len(shas) == len(D.ARMS) and len(set(shas.values())) == len(D.ARMS),
        "RA8 the five arms' final weights are PAIRWISE DISTINCT", " ".join("%s %s" % (k, v[:10]) for k, v in sorted(shas.items())))

    print("\nRA9 THE DECAY ROUTE AT A TEST-ONLY alpha0 1e-2 (NOT a batch value)")
    T1 = run_arm(a, D, "big_AS1", "AS1", alpha0="1e-2")
    T2 = run_arm(a, D, "big_AS2", "AS2", alpha0="1e-2")
    for tag, T in (("AS1", T1), ("AS2", T2)):
        chk(T is not None and T["stats"]["expect"] == [0, 0, 0, 0] and T["stats"]["meta_expect"] == 0
            and T["stats"]["decay_bites"] > 0,
            "RA9 %s at alpha0 1e-2 the AdamW formula holds bitwise and the decay term reaches the weights" % tag,
            str(T["stats"] if T else None))
    chk(T1 is not None and T2 is not None and T1["weights_sha256"] != T2["weights_sha256"],
        "RA9 at alpha0 1e-2 the wd-0.1 and wd-1e-2 runs' final weights DIFFER")

    fails = list(RR.FAILED)
    print("\n%s" % ("ALL PASS" if not fails else "FAILURES: %r" % fails))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
