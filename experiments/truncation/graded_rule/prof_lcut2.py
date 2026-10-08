import os, sys, time, cProfile, pstats
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import _paths  # noqa
import mpsenh as M
import lookcut
model, N, chi, dt, T = 'ising', 12, 8, 0.1, 4.0
cut = lookcut.LookCut(model, N, a=2, taus=[0.5, 1.0, 1.5, 2.0], ks=(1, 2, 3), fw=0.1, maxiter=40)
pr = cProfile.Profile(); t0 = time.time(); pr.enable()
Tm, _ = M.run_tebd(model, N, chi, int(round(T / dt)), dt, cut, gates=M.make_gates(model, N, dt))
pr.disable(); print(f"total {time.time()-t0:.0f}s calls {cut.calls} fired {cut.fired} jit keys {len([k for k in cut._fun if isinstance(k, tuple) and k and k[0]=='vg'])}")
st = pstats.Stats(pr); st.sort_stats('cumulative')
import io; s = io.StringIO(); st.stream = s; st.print_stats(40); out = s.getvalue().splitlines()
for line in out:
    if any(w in line for w in ('lookcut.py', 'compile', 'backend_compile', 'minimize', '_maps', 'marginals', 'region_rho', 'pjit', 'apply_primitive', 'lbfgs', 'ncalls', 'svd', 'run_tebd')):
        print(line[:170])
