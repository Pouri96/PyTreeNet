"""How fast does the marginal error of a compressed state grow under FREE exact evolution?

Start from the exact dense state at time T0, compress it to rank chi by (svd) sequential SVD, or (opt) the static
marginal fit of probe_ceiling.py (all 1-,2-,3-site marginals, no fidelity term), then evolve each EXACTLY (no further
compression) and track the error of the 1/2/3-site marginals against the exact trajectory.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
if os.environ.get('PYLIBS'):
    sys.path.insert(0, os.environ['PYLIBS'])
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64', True)
import _paths  # noqa: F401
import mpsenh as M

model, N, T0, chi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
nsteps = int(sys.argv[5]) if len(sys.argv) > 5 else 12
fw = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
dt = 0.1
Gs = M.make_gates(model, N, dt)
KS = (1, 2, 3)


def step(v):
    for b in list(range(N - 1)) + list(range(N - 2, -1, -1)):
        v = M.dense_gate(v, Gs[b], b, N)
    return v


def rdms(v, k):
    return [np.einsum('asb,atb->st', v.reshape(2 ** i, 2 ** k, -1), v.reshape(2 ** i, 2 ** k, -1).conj()) for i in range(N - k + 1)]


def err(v, ex):
    v = v / np.linalg.norm(v)
    return [float(np.sqrt(np.mean([np.sum(np.abs(a - b) ** 2) for a, b in zip(rdms(ex, k), rdms(v, k))]))) for k in KS]


def mps_from_dense(v):
    T_, rest, left = [], v.reshape(1, -1), 1
    for i in range(N - 1):
        U, s, Vh = np.linalg.svd(rest.reshape(left * 2, -1), full_matrices=False)
        k = min(chi, len(s)); T_.append(U[:, :k].reshape(left, 2, k)); rest = s[:k, None] * Vh[:k]; left = k
    T_.append(rest.reshape(left, 2, 1)); return T_


def dense_np(Ts):
    v = Ts[0].reshape(2, -1)
    for t in Ts[1:]:
        v = v.reshape(-1, t.shape[0]) @ t.reshape(t.shape[0], -1)
    return v.reshape(-1)


def unpack(x, shapes):
    out, p = [], 0
    for sh in shapes:
        n = int(np.prod(sh)); out.append((x[p:p + n] + 1j * x[p + n:p + 2 * n]).reshape(sh)); p += 2 * n
    return out


def dense_j(Ts):
    v = Ts[0].reshape(2, -1)
    for t in Ts[1:]:
        v = v.reshape(-1, t.shape[0]) @ t.reshape(t.shape[0], -1)
    return v.reshape(-1)


psi = M.mps_to_dense(M.initial_mps(model, N))
for _ in range(int(round(T0 / dt))):
    psi = step(psi)
psi = psi / np.linalg.norm(psi)
Ts = mps_from_dense(psi)
shapes = [t.shape for t in Ts]
v_svd = dense_np(Ts)
tau = {k: [jnp.asarray(r) for r in rdms(psi, k)] for k in KS}
vref = jnp.asarray(psi)


def f(x):
    v = dense_j(unpack(x, shapes)); v = v / jnp.linalg.norm(v)
    tot = 0.0
    for k in KS:
        out = []
        for i in range(N - k + 1):
            t = v.reshape(2 ** i, 2 ** k, -1); out.append(jnp.einsum('asb,atb->st', t, jnp.conj(t)))
        tot = tot + jnp.mean(jnp.array([jnp.sum(jnp.abs(r - q) ** 2) for r, q in zip(out, tau[k])]))
    if fw > 0:
        tot = tot + fw * (1 - jnp.abs(jnp.vdot(vref, v)) ** 2)
    return tot


vg = jax.jit(jax.value_and_grad(f))
x0 = np.concatenate([np.concatenate([t.real.reshape(-1), t.imag.reshape(-1)]) for t in Ts])
res = minimize(lambda x: (lambda a, b: (float(a), np.asarray(b)))(*vg(jnp.asarray(x))), x0, jac=True, method='L-BFGS-B',
               options=dict(maxiter=1500, maxfun=3000, ftol=1e-16, gtol=1e-12))
v_opt = dense_np(unpack(res.x, shapes))
print(f"{model} N={N} chi={chi}: compress the exact state at t={T0}, then evolve EXACTLY. error of 1/2/3-site marginals (rms Frobenius). fw={fw}")
print(f"fidelity at t={T0}:  svd {abs(np.vdot(psi, v_svd / np.linalg.norm(v_svd))) ** 2:.4f}   opt {abs(np.vdot(psi, v_opt / np.linalg.norm(v_opt))) ** 2:.4f}")
print(f"{'step':>5}{'t':>6} | {'svd 1':>9}{'svd 2':>9}{'svd 3':>9} | {'opt 1':>9}{'opt 2':>9}{'opt 3':>9} | {'opt/svd 2-site':>15}")
a, b, c = psi.copy(), v_svd.copy(), v_opt.copy()
for s in range(nsteps + 1):
    if s > 0:
        a, b, c = step(a), step(b), step(c)
    es, eo = err(b, a), err(c, a)
    print(f"{s:>5}{T0 + s * dt:>6.1f} | " + "".join(f"{x:>9.1e}" for x in es) + " | " + "".join(f"{x:>9.1e}" for x in eo) + f" | {eo[1] / es[1]:>15.3f}")
