import numpy as np, hp_bench, mpsenh as M
import pauli_prop as pp
N, T, dt = 16, 4.0, 0.1
ns = int(round(T / dt))
ex = hp_bench.reference('ising', N, T, dt)
obs = pp.sample_observables(N)
exv = pp.exact_values(ex, N, obs)
sel = [0, 5, 14]            # centre <X>, <Z Z>-type nn, nnn
for idx in sel:
    st, tag = obs[idx]
    print('observable', tag, 'exact', exv[idx])
    for eps in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        val, K, sec, tr = pp.propagate('ising', N, ns, dt, st, eps, cap=3_000_000, tmax=240)
        print(f"  eps={eps:g}: value {val:+.5f} err {abs(val-exv[idx]):.2e} max strings {K} time {sec:.0f}s {'(STOPPED)' if tr else ''}", flush=True)
        if tr or sec > 100: break
