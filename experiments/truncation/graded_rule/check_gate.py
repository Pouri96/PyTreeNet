"""Validate the gate's benefit measure against the independent dense-state audit (same cuts, gating off).
    python check_gate.py model N T chi audit.json"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast
model, N, T, chi, aud = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cut = spcfast.SPCFast(model, N, a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2, gate_log=True)
M.run_tebd(model, N, chi, int(round(T / 0.1)), 0.1, cut, gates=M.make_gates(model, N, 0.1))
R = json.load(open(aud))
L = cut.gate_log
print(f'{len(L)} gate-log entries, {len(R)} audit records (audit skips cuts whose region leaves the chain)')
ab = {(r['step'], r['b']): r['svd']['s2'] for r in R}
# gate log has no step index: match by order, keeping entries whose region lies inside the chain
inside = [e for e in L if e[3] - 2 >= 0 and e[3] + 4 <= N]
print(f'{len(inside)} interior entries')
if len(inside) == len(R):
    B = np.array([e[1] for e in inside]); A = np.array([r['svd']['s2'] for r in R])
    print('gate B vs audit span-2 residual: max rel diff %.2e, median %.2e' % (np.max(np.abs(B - A) / A), np.median(np.abs(B - A) / A)))
