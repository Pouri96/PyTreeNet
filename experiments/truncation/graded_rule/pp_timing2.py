import numpy as np, hp_bench, mpsenh as M
import pauli_prop as pp
N, dt = 16, 0.1
for T in (1.0, 2.0, 3.0):
    ns = int(round(T / dt))
    ex = hp_bench.reference('ising', N, T, dt)
    obs = pp.sample_observables(N)
    exv = pp.exact_values(ex, N, obs)
    # SVD MPS reference error on the same 21 observables
    res = {}
    for chi in (4, 6, 8, 12):
        Tm, _ = M.run_tebd('ising', N, chi, ns, dt, M.svd_cut)
        ap = M.mps_to_dense(Tm)
        res[chi] = np.sqrt(np.mean((pp.exact_values(ap, N, obs) - exv) ** 2))
    print(f"T={T}: SVD-MPS rms error on the 21 central observables " + ", ".join(f"chi{c}:{e:.1e}" for c, e in res.items()), flush=True)
    for idx in (0, 5):
        st, tag = obs[idx]
        for eps in (1e-3, 1e-4, 1e-5):
            val, K, sec, tr = pp.propagate('ising', N, ns, dt, st, eps, cap=3_000_000, tmax=150)
            print(f"   {tag}: eps={eps:g} err {abs(val-exv[idx]):.1e} strings {K} time {sec:.1f}s {'(STOPPED)' if tr else ''}", flush=True)
            if tr or sec > 60: break
