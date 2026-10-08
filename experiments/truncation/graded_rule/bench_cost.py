"""CPU cost of the numpy spc cut against plain SVD on one cell, min over repeats in one process (robust to a loaded machine).

    python bench_cost.py model N T chi reps [spcf arm string without the leading spcf]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, time
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import spcfast
model, N, T, chi, reps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
a, fw, iters = (int(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 8 else (2, 0.0, 4)
taus = [float(x) for x in sys.argv[9].split('-')] if len(sys.argv) > 9 else [1.0]
ks = tuple(int(x) for x in sys.argv[10].split('-')) if len(sys.argv) > 10 else (1, 2, 3)
eps = float(sys.argv[11]) if len(sys.argv) > 11 else 1e-5
G = M.make_gates(model, N, 0.1); n = int(round(T / 0.1))
def cpu(f):
    t = time.process_time(); w = time.time(); f(); return time.process_time() - t, time.time() - w
ts = [cpu(lambda: M.run_tebd(model, N, chi, n, 0.1, M.svd_cut, gates=G)) for _ in range(reps + 2)]
print('svd  cpu min %.3f  wall min %.3f' % (min(t[0] for t in ts), min(t[1] for t in ts)))
cut = spcfast.SPCFast(model, N, a=a, taus=taus, ks=ks, fw=fw, iters=iters, eps_min=eps)
t0 = time.process_time()
for L in range(2 + a, 2 + 2 * a + 1):
    cut._region(L, False); cut._region(L, True)
print('one-off region precompute %.2f s' % (time.process_time() - t0))
ts = []
for _ in range(reps):
    cut.fired = cut.calls = 0; cut.tm = {k: 0.0 for k in cut.tm}; cut.nmv = 0
    ts.append(cpu(lambda: M.run_tebd(model, N, chi, n, 0.1, cut, gates=G)))
i = int(np.argmin([t[0] for t in ts]))
print('spcf cpu min %.3f  wall min %.3f   (fired %d/%d, matvecs %d)  last-rep split %s' % (min(t[0] for t in ts), min(t[1] for t in ts), cut.fired, cut.calls, cut.nmv, {k: round(v, 2) for k, v in cut.tm.items()}))
