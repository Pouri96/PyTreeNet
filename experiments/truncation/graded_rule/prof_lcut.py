"""Where does cut-local spend its time? Ising N=12 chi=8, first 2 time units."""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import _paths  # noqa
import mpsenh as M
import lookcut

model, N, chi, dt, T = 'ising', 12, 8, 0.1, float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
cut = lookcut.LookCut(model, N, a=2, taus=[0.5, 1.0, 1.5, 2.0], ks=(1, 2, 3), fw=0.1, maxiter=40)
stat = dict(calls=0, trivial=0, opt=0, t_opt=0.0, evals=0, t_eval=0.0, discarded=[])
orig = cut.__call__.__func__

import scipy.optimize as so
_min = so.minimize
def timed_min(fun, x0, **kw):
    def f(x):
        t = time.perf_counter(); r = fun(x); stat['t_eval'] += time.perf_counter() - t; stat['evals'] += 1; return r
    return _min(f, x0, **kw)
lookcut.minimize = timed_min

def wrapped(theta, chi_, dirn, A, B, b):
    stat['calls'] += 1
    l, r = theta.shape[0], theta.shape[3]
    s = np.linalg.svd(theta.reshape(l * 2, 2 * r), compute_uv=False)
    k = M._rank(s, chi_)
    if k >= len(s) or np.sum(s[k:] ** 2) <= 0:
        stat['trivial'] += 1
    else:
        stat['opt'] += 1; stat['discarded'].append(np.sum(s[k:] ** 2) / np.sum(s ** 2))
    t = time.perf_counter(); out = orig(cut, theta, chi_, dirn, A, B, b)
    if not (k >= len(s) or np.sum(s[k:] ** 2) <= 0): stat['t_opt'] += time.perf_counter() - t
    return out

class W:
    def start(self, T_): cut.start(T_)
    def __call__(self, *a): return wrapped(*a)
t0 = time.time()
Tm, _ = M.run_tebd(model, N, chi, int(round(T / dt)), dt, W(), gates=M.make_gates(model, N, dt))
tot = time.time() - t0
d = np.array(stat['discarded'])
print(f"T={T} total {tot:.0f}s, cut calls {stat['calls']}, no truncation needed {stat['trivial']}, optimised {stat['opt']}, fired (accepted) {cut.fired}")
print(f"time in optimised cuts {stat['t_opt']:.0f}s ({stat['t_opt']/max(stat['opt'],1):.2f} s per optimised cut), objective evals {stat['evals']} ({stat['t_eval']/max(stat['evals'],1)*1e3:.1f} ms each, total {stat['t_eval']:.0f}s)")
if len(d): print("discarded weight at optimised cuts: median %.1e, 90th pct %.1e, share below 1e-4: %.2f, below 1e-6: %.2f" % (np.median(d), np.percentile(d, 90), (d < 1e-4).mean(), (d < 1e-6).mean()))
