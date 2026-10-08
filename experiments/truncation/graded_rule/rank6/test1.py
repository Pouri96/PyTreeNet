"""Test 1 (audit, no tilt): is there a repairable cross-cut residual in LPDO-SVD once the noise is switched on?

PRE-REGISTERED DEFINITIONS (fixed before the grid was run; see reports/first_moves/rank6_noisy_results.md)
  model ising, Neel start, N = 8, T = 4, dt = 0.1, dephasing L_j = sqrt(gamma) Z_j, gamma in {0, .003, .01, .03, .1}, chi in {6, 8, 12}, kappa in {2, 4}.
  (a) both truncations (chi, kappa), centre-gauged (environment-aware) Kraus truncation, spcf gate quantities logged at every bond cut;
  (b) bond truncation only: kappa = KB (16, "infinity"); (c) Kraus truncation only: LPDO-TEBD with chi = CINF = 64 (bond truncation negligible: total bond discarded weight is logged and must be < 1e-8), kappa as in (a);
  for kappa = 2 the dense global-Kraus purification ('cd', exactly chi = infinity) is run as a cross-check of the CINF runs.
  e_a, e_b, e_c = nn rms of (a), (b), (c).  Bond share s_b = e_b / e_a, Kraus share s_c = e_c / e_a.
  gate on-fraction f = (number of logged bond cuts with B_res >= c0 (tail/1e-4)^0.65) / (all (N-1)*2*nsteps = 560 bond cuts of the run); window a = 2 is primary (the
  config in which the gate constants were calibrated, check_gate.py), a = 1 is reported as a robustness check.  rho_f(gamma, chi) = f(gamma, chi) / f(0, chi).
  SUCCESS (both):  at gamma = 0.01, for some kappa, at >= 2 values of chi with e_a >= 1e-3:  s_b >= 0.5  AND  rho_f >= 0.5.
  KILL (either):   rho_f(0.01, chi) < 0.25 at every chi  OR  [ s_c > 0.8 in every usable (e_a >= 1e-3) cell with gamma >= 0.01  AND  a Kraus-tilt probe gains < 1.3x ].
  AMBIGUOUS:       the residual survives only at gamma = 0.003.
    python test1.py [nproc]       -> results/test1_raw.jsonl  (resumable)
"""
import _p6  # noqa: F401
import json
import os
import sys
import time
import multiprocessing as mp
import lpdo_bench as B

MODEL, N, T, DT = 'ising', 8, 4.0, 0.1
GAMMAS = [0.0, 0.003, 0.01, 0.03, 0.1]
CHIS = [6, 8, 12]
KAPS = [2, 4]
KB = 16
CINF = 64
OUT = 'results/test1_raw.jsonl'


def jobs():
    J = []
    for g in GAMMAS:
        for chi in CHIS:
            for kap in (KAPS if g > 0 else [4]):
                J.append(('a', g, chi, kap))
    for g in GAMMAS[1:]:
        for chi in CHIS:
            J.append(('b', g, chi, KB))
    for g in GAMMAS[1:]:
        for kap in KAPS:
            J.append(('c', g, CINF, kap))
    for g in GAMMAS[1:]:
        J.append(('cd', g, 0, 2))
    return J


def key(j):
    return f'{j[0]}_g{j[1]:g}_chi{j[2]}_k{j[3]}'


def run(j):
    kind, g, chi, kap = j
    t = time.time()
    if kind == 'a':
        r = B.cell(MODEL, N, T, DT, g, chi, kap, probe_a=(2, 1), mode='mps', keep_log=True)
    elif kind == 'b':
        r = B.cell(MODEL, N, T, DT, g, chi, kap, probe_a=None, mode='mps')
    elif kind == 'c':
        r = B.cell(MODEL, N, T, DT, g, chi, kap, probe_a=None, mode='mps')
    else:
        r = B.cell(MODEL, N, T, DT, g, 0, kap, mode='kraus')
    r['job'] = key(j)
    r['kind'] = kind
    r['cpu'] = time.time() - t
    return r


if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    only = sys.argv[2] if len(sys.argv) > 2 else ''
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            done.add(json.loads(line)['job'])
    for g in GAMMAS:
        B.reference(MODEL, N, T, DT, g)
    todo = [j for j in jobs() if key(j) not in done and (not only or j[0] in only)]
    # heavy jobs first
    todo.sort(key=lambda j: {'c': 0, 'b': 1, 'a': 2, 'cd': 3}[j[0]])
    print(f'{len(todo)} jobs to run ({len(done)} done)', flush=True)
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(run, todo):
            with open(OUT, 'a') as f:
                f.write(json.dumps(r, default=B._tojson) + '\n')
            print(f"{r['job']:24s} nn={r['nn_rms']:.2e} E={r['E_abs']:.2e} tn={r['trace_norm']:.2e} ({r['cpu']:.0f}s)", flush=True)
