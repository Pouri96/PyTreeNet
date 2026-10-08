"""Cut-local cost split: run the same cell twice in one process so the second pass has every jit cache warm.

    python time_split.py N chi T a taus skip ftol
"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import _paths  # noqa
import mpsenh as M
import lookcut

N, chi, T, a = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
taus = [float(x) for x in sys.argv[5].split('-')]
skip, ftol = float(sys.argv[6]), float(sys.argv[7])
model, dt = 'ising', 0.1
OPT = sys.argv[8] if len(sys.argv) > 8 else 'scipy'
cut = lookcut.LookCut(model, N, a=a, taus=taus, ks=(1, 2, 3), fw=0.1, maxiter=40, skip_tol=skip, ftol=ftol,
                      gtol=(1e-10 if ftol < 1e-12 else 1e-6) if OPT == 'scipy' else 1e-4, opt=OPT)
ev = [0]
_min = lookcut.minimize
def counted(fun, x0, **kw):
    def f(x):
        ev[0] += 1
        return fun(x)
    return _min(f, x0, **kw)
lookcut.minimize = counted
for rep in (1, 2):
    c0, e0, f0, s0 = cut.calls, ev[0], cut.fired, cut.skipped
    t0, p0 = time.time(), time.process_time()
    M.run_tebd(model, N, chi, int(round(T / dt)), dt, cut, gates=M.make_gates(model, N, dt))
    print(f"pass {rep}: wall {time.time()-t0:.0f}s cpu {time.process_time()-p0:.0f}s cuts {cut.calls-c0} skipped {cut.skipped-s0} fired {cut.fired-f0} evals {ev[0]-e0}", flush=True)
