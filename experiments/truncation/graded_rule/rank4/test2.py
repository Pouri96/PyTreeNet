"""Test 2: where does the error come from, and does a moving reference fix it?
python test2.py MODEL N T CHI OUT.json
Step 1: per-cut defect attribution for SVD-raw, SVD-renorm, DMT-I, DMT-sigma0 (identity check: sum_k c_k = run - exact).
Step 2: DMT-sigma(s) with ORACLE product marginals of the exact dense state rho(T - n dt) at step n."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import heis_mpo as H
import heis_attrib as A
import refs

model, N, T, chi, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
dt = 0.1
nsteps = int(round(T / dt))
i0 = N // 2 - 1
marg = refs.dense_traj(model, N, nsteps, dt)
ex = marg[:, i0, 2]
per = 2 * (N - 1)


def sigma_s_provider(shift=0):
    """Oracle covectors at step n: (1, <X>, <Y>, <Z>) of the exact dense state at s = T - n dt (one step 'late' if shift)."""
    def f(n):
        m = marg[max(nsteps - n + shift, 0)]
        v = np.zeros((N, 4))
        v[:, 0] = 1.0
        v[:, 1:] = m
        return v
    return f


arms = {
    'svd_raw': (H.make_cut('svd_raw'), None),
    'svd_renorm': (H.make_cut('svd_renorm'), None),
    'dmt_I': (H.make_cut('dmt'), H.ref_provider(model, N, 'I')),
    'dmt_neel': (H.make_cut('dmt'), H.ref_provider(model, N, 'neel')),
    'dmt_sigma_s': (H.make_cut('dmt'), sigma_s_provider(0)),
}
res = dict(model=model, N=N, T=T, chi=chi, i0=i0, dt=dt)
t0 = time.time()
for name, (cut, ref) in arms.items():
    a = A.run_attribution(model, N, i0, chi, T, dt, cut, ref)
    r = a['res']
    c = a['c']
    err = r['val'] - ex
    cn = c.reshape(nsteps, per).sum(axis=1)
    s = T - dt * np.arange(1, nsteps + 1)
    res[name] = dict(c=c.tolist(), cn=cn.tolist(), s=s.tolist(), val=r['val'].tolist(), err=err.tolist(),
                     dmax=float(np.max(np.abs(err))), dT=float(abs(err[-1])), sum_c=float(c.sum()),
                     identity_residual=float(abs(c.sum() - err[-1])), norm_T=float(r['norm'][-1]))
    print(f'{name:12s} dmax={res[name]["dmax"]:.3e} dT={res[name]["dT"]:.3e}  sum_c={c.sum():+.3e} run-exact={err[-1]:+.3e} '
          f'residual={res[name]["identity_residual"]:.1e}  ({time.time() - t0:.0f}s)', flush=True)
json.dump(res, open(out, 'w'))
