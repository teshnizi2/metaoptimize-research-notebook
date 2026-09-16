#!/usr/bin/env python3
"""
c224_safety_sweep.py -- the CORRECTIONS 224 safety sweep of analysis/, made re-runnable.

    pyright --outputjson analysis > /tmp/pyright.json        # pyright 1.1.409 was used
    python3 analysis/c224_safety_sweep.py --pyright /tmp/pyright.json
    python3 analysis/c224_safety_sweep.py --scorepath        # the AST score()-reachability pass only

TWO PASSES.
  (1) TRIAGE of every pyright finding of the six "unbound / None" rules
      (reportPossiblyUnboundVariable, reportOptional{Operand,Subscript,MemberAccess,Call,Iterable}).
      An AST pass locates each finding's enclosing function chain; a finding inside a
      selftest runs on a fixed synthetic fixture.  Every non-selftest finding group was
      then READ BY HAND (CORRECTIONS 224.3) and the verdicts are recorded below as data:
        REACHABLE  a crash or wrong answer on plausible input (a dropped run, an empty arm);
        LATENT     reachable only if a fixed on-disk artefact or sibling module is absent
                   or malformed -- not plausible on the data the script scores;
        FIXED      reachable, in a NON-registered helper, and fixed with a test;
        ARTEFACT   the type checker cannot see the guard (a dict literal's value union, a
                   None-check / INCOMPLETE gate upstream, a short-circuit, a loop invariant).
  (2) SCORE-PATH: for each registered c*_score.py, does the selftest's call closure reach
      the function(s) the real (non-selftest) entry branch calls to score?  The cdn1 failure
      mode (CORRECTIONS 171) is a selftest that never does.

This file only READS analysis/.  It edits nothing.
"""
import argparse, ast, collections, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = {"reportPossiblyUnboundVariable", "reportOptionalOperand", "reportOptionalSubscript",
         "reportOptionalMemberAccess", "reportOptionalCall", "reportOptionalIterable"}

# (file, first line, last line) -> (verdict, reason).  Hand-read, CORRECTIONS 224.3.
OVERRIDES = [
    # PRE-fix line numbers; after the fix pyright no longer reports these two sites.
    ("probe5_window.py", 143, 144, "FIXED",
     "recompute_rho() is None when n_tot < 2 (rung token 'scal') or n_records < 2; "
     "reduce_dir crashed the reducer and its callers probe5_time_ladder / neff_instrument"),
    ("c53_score.py", 200, 230, "REACHABLE",
     "agg() returns (None, 0) when a rung has no probe dir; B0 subscripts it -> TypeError, not a B0 FAIL"),
    ("c53_score.py", 399, 430, "REACHABLE",
     "same agg() None for an absent rung in score_ns5's N0 gate"),
    ("c72_hz9_score.py", 145, 151, "REACHABLE",
     "pl() is None for a run whose .out is missing; H0 prints INCOMPLETE and continues, H0.5 then "
     "does None - float (and h1()'s mean() sums the None too)"),
    ("c72_hz9_score.py", 171, 171, "REACHABLE",
     "sem() is None for an arm with < 2 readable runs; h1b squares it"),
    ("cLV1_live_term_census.py", 561, 561, "REACHABLE",
     "indep_seed_levels() returns None when one .out is missing; statistics.mean(None)"),
    ("c53_budget_window.py", 190, 190, "LATENT",
     "neff_over() v is None when reduce_dir() rejects a torn probe dir"),
    ("c60_exception_mechanism.py", 785, 802, "LATENT", "rt is None only for an empty seed list"),
    ("c60_exception_mechanism.py", 954, 956, "LATENT", "nan_band() None only if EVERY conv tensor is constant"),
    ("c60_exception_mechanism.py", 1091, 1096, "LATENT",
     "analyse_weightwise() None for an unrecognised family in the fixed EXC_W gate list"),
    ("c63_span_reconcile.py", 433, 434, "LATENT", "span helpers None only on an empty/torn debt-cell dir"),
    ("c69_c100_armset.py", 112, 113, "LATENT", "stat() None for an arm with no readable plateau5 in a fixed post-hoc census"),
    ("c69_c100_armset.py", 184, 188, "LATENT", "stat() None for an arm with no readable plateau5 in a fixed post-hoc census"),
    ("c70_composition_audit.py", 604, 607, "LATENT", "stat() None for an empty arm in the fixed post-hoc block"),
    ("c74_bf9_score.py", 382, 382, "LATENT",
     "flagged None is guarded; the adjacent ['steady .5-1'] is a KeyError when that window is skipped (T < 16 records)"),
    ("c75_tw0_score.py", 461, 461, "LATENT", "as c74_bf9:382"),
    ("c75_frozen_free.py", 75, 75, "LATENT", "as c74_bf9:382"),
    ("cG16_eb_score.py", 712, 712, "LATENT", "noise_floor_from_csv() sigma None only if the stratum is empty in the CSV"),
    ("cLV1_live_term_census.py", 812, 819, "LATENT", "worst is None only if no archive cell exists"),
    ("cVH1_vgghorizon_score.py", 950, 986, "LATENT",
     "group_stats() None for an empty beta series on a run that passed the INCOMPLETE gate"),
    ("cvg1_attack_rederive.py", 101, 102, "LATENT", "re.search None if the manifest lacks the line"),
    ("probe5_floor.py", 189, 189, "LATENT", "decompose is None only if twochannel.py is absent"),
    # probe5_window.py line numbers are POST-fix (+5 from the 224 guard).
    ("probe5_window.py", 135, 135, "LATENT", "import fallback: twochannel.py absent from analysis/"),
    ("probe5_window.py", 187, 187, "LATENT", "import fallback: corr_range.py absent from analysis/"),
    ("probe5_window.py", 220, 224, "LATENT", "import fallback: corr_range.py absent from analysis/"),
    ("probe5_window.py", 395, 395, "LATENT", "import fallback inside the selftest"),
    ("probe5_window.py", 422, 422, "LATENT", "import fallback inside the selftest"),
]

REGISTERED_HELPERS = {"probe5_floor.py", "test_fence_mask.py", "argsline_guard.py"}


def is_registered(base):
    """c*-prefixed files are registered instruments or landed-record evidence (RULE 16);
    two non-c helpers are named as registered in CORRECTIONS (probe5_floor, test_fence_mask)."""
    return base.startswith("c") or base in REGISTERED_HELPERS


_SPANS = {}


def fn_chain(path, line):
    if path not in _SPANS:
        t = ast.parse(open(path).read())
        _SPANS[path] = sorted((n.lineno, n.end_lineno, n.name) for n in ast.walk(t)
                              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return [nm for a, b, nm in _SPANS[path] if a <= line <= b]


def verdict_of(base, line, rule, ctx, src):
    for f, a, b, v, why in OVERRIDES:
        if f == base and a <= line <= b:
            return v, why
    if ctx == "selftest":
        return "ARTEFACT", "inside a selftest: a fixed synthetic fixture"
    if rule == "reportPossiblyUnboundVariable":
        return "ARTEFACT", "bound on every path that reaches the use (loop / branch invariant)"
    if re.search(r'\b(o|occ|a|syn)\["(rec|q4|coord|first|min|max)_|\["win"\]', src):
        return "ARTEFACT", "dict-literal value union; that key always holds a number/dict"
    return "ARTEFACT", "guarded upstream (None-check, INCOMPLETE/VOID gate, short-circuit)"


def triage(pyright_json):
    d = json.load(open(pyright_json))
    out = []
    for x in d["generalDiagnostics"]:
        if x.get("rule") not in RULES:
            continue
        path = x["file"]
        base = os.path.basename(path)
        line = x["range"]["start"]["line"] + 1
        chain = fn_chain(path, line)
        ctx = ("selftest" if any("selftest" in n or n.startswith(("_test", "test_")) for n in chain)
               else "module" if not chain else "real")
        src = open(path).read().splitlines()[line - 1]
        v, why = verdict_of(base, line, x["rule"], ctx, src)
        out.append(dict(file=base, line=line, rule=x["rule"], ctx=ctx, verdict=v, why=why,
                        registered=is_registered(base)))
    return out


# ------------------------------------------------------------------ pass 2
def _called(node, funcs):
    s = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            s.add(n.func.id)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            s.add(n.id)
    return s & set(funcs)


def _subprocess_self(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in ("run", "check_output", "call", "Popen", "check_call") and \
                    re.search(r"__file__|argv|executable", ast.dump(n)):
                return True
    return False


def scorepath(path):
    t = ast.parse(open(path).read())
    funcs = {n.name: n for n in t.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    calls = {k: _called(v, funcs) for k, v in funcs.items()}

    def closure(roots):
        seen, st = set(), list(roots)
        while st:
            x = st.pop()
            if x in seen or x not in funcs:
                continue
            seen.add(x)
            st.extend(calls[x])
        return seen

    mainblk = [n for n in t.body if isinstance(n, ast.If) and "__main__" in ast.dump(n.test)]
    entry = [funcs["main"]] if "main" in funcs else mainblk
    roots = set()

    def walk(stmts):
        for s in stmts:
            if isinstance(s, ast.If):
                if "selftest" in ast.dump(s.test).lower():
                    walk(s.orelse)
                else:
                    walk(s.body)
                    walk(s.orelse)
                continue
            roots.update(x for x in _called(s, funcs) if "selftest" not in x)

    for e in entry:
        walk(e.body)
    roots.discard("main")
    st_clo = closure([k for k in funcs if "selftest" in k])
    sub = any(_subprocess_self(funcs[k]) for k in st_clo)
    if "score" in funcs:
        drivers = ["score"]
    else:
        big = max((len(closure([r])) for r in roots), default=0)
        drivers = sorted(r for r in roots if len(closure([r])) == big)
    reached = [r for r in drivers if r in st_clo]
    if sub or ("score" in funcs and "score" in st_clo):
        cls = "DRIVES"
    elif "score" not in funcs and reached:
        cls = "PARTIAL"        # sub-scorers reached; main()'s inline scoring is not
    else:
        cls = "NEVER"
    return dict(file=os.path.basename(path), cls=cls, drivers=drivers, reached=reached,
                subprocess_self=sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyright", help="pyright --outputjson analysis output")
    ap.add_argument("--scorepath", action="store_true")
    a = ap.parse_args()
    if a.pyright:
        rows = triage(a.pyright)
        print("TRIAGE  %d findings of %s" % (len(rows), ", ".join(sorted(rule[6:] for rule in RULES))))
        c = collections.Counter((r["registered"], r["verdict"]) for r in rows)
        for (reg, v), n in sorted(c.items()):
            print("  %-14s %-10s %4d" % ("registered" if reg else "non-registered", v, n))
        per = collections.defaultdict(collections.Counter)
        for r in rows:
            per[r["file"]][r["verdict"]] += 1
        print("\nPER FILE (non-ARTEFACT verdicts listed)")
        for f in sorted(per):
            print("  %-36s %-3s %s" % (f, "reg" if is_registered(f) else "-", dict(per[f])))
        print("\nNON-ARTEFACT FINDINGS")
        seen = set()
        for r in sorted(rows, key=lambda r: (r["file"], r["line"])):
            if r["verdict"] == "ARTEFACT":
                continue
            key = (r["file"], r["why"])
            lines = sorted({x["line"] for x in rows if (x["file"], x["why"]) == key})
            if key in seen:
                continue
            seen.add(key)
            print("  %-9s %s:%s  %s" % (r["verdict"], r["file"], ",".join(map(str, lines)), r["why"]))
    if a.scorepath or not a.pyright:
        print("\nSCORE-PATH  (does the selftest drive the real scoring entry?)")
        res = [scorepath(p) for p in sorted(glob.glob(os.path.join(HERE, "c*_score.py")))]
        for cls in ("NEVER", "PARTIAL", "DRIVES"):
            names = [r["file"] for r in res if r["cls"] == cls]
            print("  %-8s %2d  %s" % (cls, len(names), " ".join(names)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
