"""Static ceiling: how well can ANY rank-chi MPS reproduce all local marginals of an exact dense state?

For an exact (Trotterised) dense state psi at time T, compare
  svd   the sequentially SVD-compressed MPS at bond dimension chi (the fidelity-oriented truncation)
  opt   the MPS found by minimising  sum_k mean_i || rho_{i,k}(MPS) - rho_{i,k}(psi) ||_F^2  (k = 1, 2, 3 site windows)
        started from the svd point, with plain L-BFGS and exact gradients (JAX)
No dynamics are involved. If opt is only a few times better than svd, no fixed-rank pure-state method can win
decisively at that chi, however it is built.

    python probe_ceiling.py ising 12 4.0 4,6,8
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
SCRATCH = os.environ.get('PYLIBS')
if SCRATCH:
    sys.path.insert(0, SCRATCH)
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64', True)
import _paths  # noqa: F401
import mpsenh as M

model, N, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
chis = [int(c) for c in sys.argv[4].split(',')]
nseeds = int(sys.argv[5]) if len(sys.argv) > 5 else 0
dt = 0.1
psi = M.dense_reference(model, N, int(round(T / dt)), dt, gates=M.make_gates(model, N, dt))
psi = psi / np.linalg.norm(psi)
KS = (1, 2, 3)


def bonds(chi):
    return [1] + [min(chi, 2 ** min(i, N - i)) for i in range(1, N)] + [1]


def mps_from_dense(v, chi):
    T_, rest, left = [], v.reshape(1, -1), 1
    for i in range(N - 1):
        m = rest.reshape(left * 2, -1)
        U, s, Vh = np.linalg.svd(m, full_matrices=False)
        k = min(chi, len(s))
        T_.append(U[:, :k].reshape(left, 2, k))
        rest = (s[:k, None] * Vh[:k])
        left = k
    T_.append(rest.reshape(left, 2, 1))
    return T_


def to_dense(Ts):
    v = Ts[0].reshape(2, -1)
    for t in Ts[1:]:
        v = (v.reshape(-1, t.shape[0]) @ t.reshape(t.shape[0], -1))
    return v.reshape(-1)


def rdms(v, k):
    out = []
    for i in range(N - k + 1):
        t = v.reshape(2 ** i, 2 ** k, -1)
        out.append(jnp.einsum('asb,atb->st', t, jnp.conj(t)))
    return out


tau = {k: [jnp.asarray(r) for r in rdms(jnp.asarray(psi), k)] for k in KS}


def errs(v):
    v = v / jnp.linalg.norm(v)
    return {k: jnp.sqrt(jnp.mean(jnp.array([jnp.sum(jnp.abs(r - t) ** 2) for r, t in zip(rdms(v, k), tau[k])]))) for k in KS}


def unpack(x, shapes):
    out, p = [], 0
    for sh in shapes:
        n = int(np.prod(sh))
        out.append((x[p:p + n] + 1j * x[p + n:p + 2 * n]).reshape(sh))
        p += 2 * n
    return out


def todense_j(Ts):
    v = Ts[0].reshape(2, -1)
    for t in Ts[1:]:
        v = v.reshape(-1, t.shape[0]) @ t.reshape(t.shape[0], -1)
    return v.reshape(-1)


def objective_factory(shapes):
    def f(x):
        v = todense_j(unpack(x, shapes))
        e = errs(v)
        return sum(e[k] ** 2 for k in KS)
    return jax.jit(jax.value_and_grad(f))


def pack(Ts):
    return np.concatenate([np.concatenate([t.real.reshape(-1) for t in Ts], 0), np.concatenate([t.imag.reshape(-1) for t in Ts], 0)])


def pack_interleaved(Ts):
    xs = []
    for t in Ts:
        xs += [t.real.reshape(-1), t.imag.reshape(-1)]
    return np.concatenate(xs)


def unpack_interleaved(x, shapes):
    out, p = [], 0
    for sh in shapes:
        n = int(np.prod(sh))
        out.append((x[p:p + n] + 1j * x[p + n:p + 2 * n]).reshape(sh))
        p += 2 * n
    return out


def run(chi, init_noise=0.0, seed=0):
    Ts = mps_from_dense(psi, chi)
    shapes = [t.shape for t in Ts]
    rng = np.random.default_rng(seed)
    if init_noise > 0:
        Ts = [t + init_noise * np.linalg.norm(t) / np.sqrt(t.size) * (rng.normal(size=t.shape) + 1j * rng.normal(size=t.shape)) for t in Ts]
    x0 = pack_interleaved(Ts)

    def f(x):
        v = todense_j(unpack_interleaved_j(x, shapes))
        e = errs(v)
        return sum(e[k] ** 2 for k in KS)

    def unpack_interleaved_j(x, shapes):
        out, p = [], 0
        for sh in shapes:
            n = int(np.prod(sh))
            out.append((x[p:p + n] + 1j * x[p + n:p + 2 * n]).reshape(sh))
            p += 2 * n
        return out

    vg = jax.jit(jax.value_and_grad(f))

    def fun(x):
        val, g = vg(jnp.asarray(x))
        return float(val), np.asarray(g)

    res = minimize(fun, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=1500, maxfun=3000, ftol=1e-16, gtol=1e-12))
    Tn = unpack_interleaved(res.x, shapes)
    return Ts_to_v(Tn), res


def Ts_to_v(Tn):
    return to_dense(Tn)


print(f"{model} N={N} T={T}: best rank-chi MPS for ALL 1-,2-,3-site marginals (rms Frobenius over windows)")
print(f"{'chi':>4}{'params':>8} | {'svd 1':>9}{'opt 1':>9} | {'svd 2':>9}{'opt 2':>9} | {'svd 3':>9}{'opt 3':>9} | {'F svd':>8}{'F opt':>9}   gain 2-site")
for chi in chis:
    Ts = mps_from_dense(psi, chi)
    v0 = to_dense(Ts)
    e0 = {k: float(x) for k, x in errs(jnp.asarray(v0)).items()}
    F0 = abs(np.vdot(psi, v0 / np.linalg.norm(v0))) ** 2
    best = None
    for sd in range(nseeds + 1):
        v1, res = run(chi, init_noise=0.0 if sd == 0 else 0.3, seed=sd)
        e1 = {k: float(x) for k, x in errs(jnp.asarray(v1)).items()}
        if best is None or sum(e1[k] ** 2 for k in KS) < sum(best[0][k] ** 2 for k in KS):
            best = (e1, abs(np.vdot(psi, v1 / np.linalg.norm(v1))) ** 2, res.nit)
    e1, F1, nit = best
    params = sum(t.size for t in Ts)
    print(f"{chi:>4}{params:>8} | {e0[1]:>9.2e}{e1[1]:>9.2e} | {e0[2]:>9.2e}{e1[2]:>9.2e} | {e0[3]:>9.2e}{e1[3]:>9.2e} | {F0:>8.4f}{F1:>9.4f}   {e0[2] / e1[2]:.1f}x  ({nit} it)", flush=True)
