"""Does the static marginal-fit advantage survive the dynamics?

A dense state evolves exactly through one Strang step at a time, and after every step it is compressed to a
rank-chi MPS by either
  svd   sequential SVD compression (fidelity oriented)
  fit   L-BFGS fit of all 1-, 2-, 3-site window marginals to those of the uncompressed step result, started at svd
The compressed state is carried to the next step. The final state is compared with the Trotterised dense
reference. This isolates the compression policy from the gate-by-gate sweep.

    PYLIBS=<dir with jax> python probe_stepfit.py ising 10 3.0 4,6 [maxiter] [fidelity_weight]
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys, time, math
if os.environ.get('PYLIBS'):
    sys.path.insert(0, os.environ['PYLIBS'])
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64', True)
import _paths  # noqa: F401
import mpsenh as M

model, N, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
chis = [int(c) for c in sys.argv[4].split(',')]
maxiter = int(sys.argv[5]) if len(sys.argv) > 5 else 300
fw = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
kmax = int(sys.argv[7]) if len(sys.argv) > 7 else 3
pw = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0   # weight of the all-pairs term
LA = int(sys.argv[9]) if len(sys.argv) > 9 else 0      # lookahead: also fit marginals of U^m v, m = 1..LA
LASET = [int(x) for x in os.environ['LASET'].split(',')] if os.environ.get('LASET') else list(range(LA + 1))
if os.environ.get('LASET'):
    LA = max(LASET)
KEVAL = (1, 2, 3, 4)
KS = tuple(range(1, kmax + 1))
KSLA = tuple(int(x) for x in os.environ['KSLA'].split(',')) if os.environ.get('KSLA') else KS   # window sizes fitted at m > 0
WLA = float(os.environ.get('WLA', 1.0))                                                          # weight of the m > 0 terms
WINA = int(os.environ.get('WIN', 0))       # local-region lookahead: evolve the reduced state of the region [i-WIN, i+k+WIN) under its own open-chain H
WTAUS = [float(x) for x in os.environ['WTAUS'].split(',')] if os.environ.get('WTAUS') else []
JET = int(os.environ.get('JET', 0))                       # fit the time derivatives d^j/dt^j of the window marginals, j = 1..JET
JETK = tuple(int(x) for x in os.environ.get('JETK', '2').split(','))
JTAU = float(os.environ.get('JTAU', 0.4))                 # weight of order j is (JTAU^j / j!)^2
WJ = float(os.environ.get('WJ', 1.0))
dt = 0.1
nsteps = int(round(T / dt))
Gs = M.make_gates(model, N, dt)


def step_dense(v):
    for b in range(N - 1):
        v = M.dense_gate(v, Gs[b], b, N)
    for b in range(N - 2, -1, -1):
        v = M.dense_gate(v, Gs[b], b, N)
    return v


PROF = [int(x) for x in os.environ['PROF'].split(',')] if os.environ.get('PROF') else None   # per-bond caps (N-1 values)


def mps_from_dense(v, chi):
    T_, rest, left = [], v.reshape(1, -1), 1
    for i in range(N - 1):
        m = rest.reshape(left * 2, -1)
        U, s, Vh = np.linalg.svd(m, full_matrices=False)
        k = min(PROF[i] if PROF else chi, len(s))
        T_.append(U[:, :k].reshape(left, 2, k))
        rest = s[:k, None] * Vh[:k]
        left = k
    T_.append(rest.reshape(left, 2, 1))
    return T_


def to_dense_np(Ts):
    v = Ts[0].reshape(2, -1)
    for t in Ts[1:]:
        v = v.reshape(-1, t.shape[0]) @ t.reshape(t.shape[0], -1)
    return v.reshape(-1)


def rdms_j(v, k):
    out = []
    for i in range(N - k + 1):
        t = v.reshape(2 ** i, 2 ** k, -1)
        out.append(jnp.einsum('asb,atb->st', t, jnp.conj(t)))
    return out


def rdms_np(v, k):
    return [np.einsum('asb,atb->st', v.reshape(2 ** i, 2 ** k, -1), v.reshape(2 ** i, 2 ** k, -1).conj()) for i in range(N - k + 1)]



PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]


def pair_rdms_j(v):
    t = v.reshape((2,) * N)
    out = []
    for i, j in PAIRS:
        m = jnp.moveaxis(t, [i, j], [0, 1]).reshape(4, -1)
        out.append(m @ jnp.conj(m).T)
    return jnp.stack(out)


def pair_rdms_np(v):
    t = v.reshape((2,) * N)
    out = []
    for i, j in PAIRS:
        m = np.moveaxis(t, [i, j], [0, 1]).reshape(4, -1)
        out.append(m @ m.conj().T)
    return np.array(out)



COARSE = float(os.environ.get('COARSE', 0))     # > 0: the lookahead propagator is one coarse brickwork Strang step of this dt
if COARSE > 0:
    GE, GO = M.make_gates(model, N, COARSE), M.make_gates(model, N, 2 * COARSE)
    LAYERS = [(b, GE[b]) for b in range(0, N - 1, 2)] + [(b, GO[b]) for b in range(1, N - 1, 2)] + [(b, GE[b]) for b in range(0, N - 1, 2)]


PROPS = os.environ.get('PROPS')     # e.g. "0.5e,0.5o,0.3e": lookahead term m applies propagator PROPS[m-1] to the state itself (not cumulative)
PLAYERS = []
if PROPS:
    for tok in PROPS.split(','):
        dtc, par = float(tok[:-1]), tok[-1]
        ge, go = M.make_gates(model, N, dtc), M.make_gates(model, N, 2 * dtc)
        first = range(0, N - 1, 2) if par == 'e' else range(1, N - 1, 2)
        second = range(1, N - 1, 2) if par == 'e' else range(0, N - 1, 2)
        PLAYERS.append([(b, ge[b]) for b in first] + [(b, go[b]) for b in second] + [(b, ge[b]) for b in first])


def prop_dense(v, idx):
    for b, G in PLAYERS[idx]:
        v = M.dense_gate(v, G, b, N)
    return v


def prop_j(v, idx):
    for b, G in PLAYERS[idx]:
        t = v.reshape(2 ** b, 4, 2 ** (N - b - 2))
        v = jnp.einsum('ij,ajb->aib', jnp.asarray(G).reshape(4, 4), t).reshape(-1)
    return v


def la_step_dense(v):
    if COARSE <= 0:
        return step_dense(v)
    for b, G in LAYERS:
        v = M.dense_gate(v, G, b, N)
    return v


def la_step_j(v):
    if COARSE <= 0:
        return step_j(v)
    for b, G in LAYERS:
        t = v.reshape(2 ** b, 4, 2 ** (N - b - 2))
        v = jnp.einsum('ij,ajb->aib', jnp.asarray(G).reshape(4, 4), t).reshape(-1)
    return v


def step_j(v):
    for b in list(range(N - 1)) + list(range(N - 2, -1, -1)):
        t = v.reshape(2 ** b, 4, 2 ** (N - b - 2))
        v = jnp.einsum('ij,ajb->aib', jnp.asarray(Gs[b]).reshape(4, 4), t).reshape(-1)
    return v


HT = [jnp.asarray(M.h_bond(model, b, N)).reshape(4, 4) for b in range(N - 1)]


def hmul_j(v):
    out = 0.0
    for b in range(N - 1):
        t = v.reshape(2 ** b, 4, 2 ** (N - b - 2))
        out = out + jnp.einsum('ij,ajb->aib', HT[b], t).reshape(-1)
    return out


def jets_j(v):
    """dict (j, k) -> (windows, 2^k, 2^k): window marginal of (ad_H)^j |v><v| = sum_{a+b=j} C(j,a) (-1)^b |H^b v><H^a v|."""
    us = [v]
    for a in range(JET):
        us.append(hmul_j(us[-1]))
    out = {}
    for k in JETK:
        for j in range(1, JET + 1):
            acc = 0.0
            for a in range(j + 1):
                b = j - a
                ta = [u.reshape(2 ** i, 2 ** k, -1) for i in range(N - k + 1) for u in (us[a],)]
                tb = [us[b].reshape(2 ** i, 2 ** k, -1) for i in range(N - k + 1)]
                cr = jnp.stack([jnp.einsum('asb,atb->st', y, jnp.conj(x)) for x, y in zip(ta, tb)])
                acc = acc + (math.comb(j, a) * (-1) ** b) * cr
            out[(j, k)] = acc
    return out


_WREG = {}


def win_unitaries(lo, hi):
    if (lo, hi) not in _WREG:
        L = hi - lo
        Hm = np.zeros((2 ** L, 2 ** L), dtype=complex)
        for b in range(lo, hi - 1):
            Hm += np.kron(np.kron(np.eye(2 ** (b - lo)), M.h_bond(model, b, N)), np.eye(2 ** (hi - b - 2)))
        ev, Vv = np.linalg.eigh(Hm)
        _WREG[(lo, hi)] = [jnp.asarray((Vv * np.exp(-1j * t * ev)) @ Vv.conj().T) for t in WTAUS]
    return _WREG[(lo, hi)]


def win_marginals(v, k):
    """list over taus of (windows, 2^k, 2^k): window marginal of the region state evolved for tau under the region's own H."""
    out = [[] for _ in WTAUS]
    for i in range(N - k + 1):
        lo, hi = max(0, i - WINA), min(N, i + k + WINA)
        L = hi - lo
        t = v.reshape(2 ** lo, 2 ** L, -1)
        rho = jnp.einsum('asb,atb->st', t, jnp.conj(t))
        off, r = i - lo, hi - i - k
        for j, U in enumerate(win_unitaries(lo, hi)):
            X = (U @ rho @ U.conj().T).reshape(2 ** off, 2 ** k, 2 ** r, 2 ** off, 2 ** k, 2 ** r)
            out[j].append(jnp.einsum('asbatb->st', X))
    return [jnp.stack(o) for o in out]


def unpack(x, shapes):
    out, p = [], 0
    for sh in shapes:
        n = int(np.prod(sh))
        out.append((x[p:p + n] + 1j * x[p + n:p + 2 * n]).reshape(sh))
        p += 2 * n
    return out


def pack(Ts):
    xs = []
    for t in Ts:
        xs += [t.real.reshape(-1), t.imag.reshape(-1)]
    return np.concatenate(xs)


def todense_j(Ts):
    v = Ts[0].reshape(2, -1)
    for t in Ts[1:]:
        v = v.reshape(-1, t.shape[0]) @ t.reshape(t.shape[0], -1)
    return v.reshape(-1)


def make_vg(shapes):
    def f(x, taus, vref):
        v = todense_j(unpack(x, shapes))
        nv = jnp.real(jnp.vdot(v, v))
        vn = v / jnp.sqrt(nv)
        tot = 0.0
        w = vn
        for m in range(LA + 1):
            if m > 0:
                w = prop_j(vn, m - 1) if PLAYERS else la_step_j(w)
            if m not in LASET:
                continue
            for k in (KS if m == 0 else KSLA):
                rs = rdms_j(w, k)
                tot = tot + (1.0 if m == 0 else WLA) * jnp.mean(jnp.array([jnp.sum(jnp.abs(r - t) ** 2) for r, t in zip(rs, taus[(m, k)])]))
            if pw > 0:
                tot = tot + pw * jnp.mean(jnp.sum(jnp.abs(pair_rdms_j(w) - taus[(m, 99)]) ** 2, axis=(1, 2)))
        if WINA > 0 and WTAUS:
            for k in KSLA:
                for j, val in enumerate(win_marginals(vn, k)):
                    tot = tot + WLA * jnp.mean(jnp.sum(jnp.abs(val - taus[(100 + j, k)]) ** 2, axis=(1, 2)))
        if JET > 0:
            jw = jets_j(vn)
            for (j, k), val in jw.items():
                wj = WJ * (JTAU ** j / math.factorial(j)) ** 2
                tot = tot + wj * jnp.mean(jnp.sum(jnp.abs(val - taus[(-j, k)]) ** 2, axis=(1, 2)))
        if fw > 0:
            tot = tot + fw * (1.0 - jnp.abs(jnp.vdot(vref, vn)) ** 2)
        return tot
    return jax.jit(jax.value_and_grad(f))


def compress_fit(v, chi, vg, shapes, x0):
    taus = {}
    w = v
    for m in range(LA + 1):
        if m > 0:
            w = prop_dense(v, m - 1) if PLAYERS else la_step_dense(w)
        if m not in LASET:
            continue
        for k in (KS if m == 0 else KSLA):
            taus[(m, k)] = [jnp.asarray(r) for r in rdms_np(w, k)]
        if pw > 0:
            taus[(m, 99)] = jnp.asarray(pair_rdms_np(w))
    vref = jnp.asarray(v)
    if WINA > 0 and WTAUS:
        for k in KSLA:
            for j, val in enumerate(win_marginals(vref / jnp.linalg.norm(vref), k)):
                taus[(100 + j, k)] = val
    if JET > 0:
        for (j, k), val in jets_j(vref / jnp.linalg.norm(vref)).items():
            taus[(-j, k)] = val

    def fun(x):
        val, g = vg(jnp.asarray(x), taus, vref)
        return float(val), np.asarray(g)
    res = minimize(fun, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, maxfun=2 * maxiter, ftol=1e-15, gtol=1e-11))
    return to_dense_np(unpack(res.x, shapes))


def errs(v, ex):
    v = v / np.linalg.norm(v)
    o = {}
    for k in KEVAL:
        a, b = rdms_np(ex, k), rdms_np(v, k)
        o[k] = float(np.sqrt(np.mean([np.sum(np.abs(x - y) ** 2) for x, y in zip(a, b)])))
    pe, pa = pair_rdms_np(ex), pair_rdms_np(v)
    d = np.array([j - i for i, j in PAIRS])
    err = np.sum(np.abs(pe - pa) ** 2, axis=(1, 2))
    o['p_near'] = float(np.sqrt(np.mean(err[d <= 2])))
    o['p_far'] = float(np.sqrt(np.mean(err[d >= 4])))
    o['infid'] = float(1 - abs(np.vdot(ex, v)) ** 2)
    o['E'] = float(abs(M.local_obs(ex, N, model)['E'] - M.local_obs(v, N, model)['E']))
    return o


psi0 = M.mps_to_dense(M.initial_mps(model, N))
ex = psi0.copy()
for _ in range(nsteps):
    ex = step_dense(ex)
ex = ex / np.linalg.norm(ex)

print(f"{model} N={N} T={T} dt={dt}: compress every step, fidelity_weight={fw}, maxiter={maxiter}, fit windows up to k={kmax}, pair weight={pw}, lookahead={LA} set={LASET}")
print(f"{'chi':>4}{'params':>8}  {'arm':>5}" + "".join(f"{h:>10}" for h in ('infid', '1-site', '2-site', '3-site', '4-site', 'pair d<=2', 'pair d>=4', 'E')) + "     secs")
for chi in chis:
    out = {}
    for arm in ('svd', 'fit'):
        v = psi0.copy()
        t0 = time.time()
        vg = shapes = None
        for s in range(nsteps):
            v = step_dense(v)
            v = v / np.linalg.norm(v)
            Ts = mps_from_dense(v, chi)
            if arm == 'fit':
                if vg is None:
                    shapes = [t.shape for t in Ts]
                    vg = make_vg(shapes)
                v = compress_fit(v, chi, vg, shapes, pack(Ts))
                v = v / np.linalg.norm(v)
            else:
                v = to_dense_np(Ts)
                v = v / np.linalg.norm(v)
        out[arm] = (errs(v, ex), sum(t.size for t in Ts), time.time() - t0)
    for arm in ('svd', 'fit'):
        e, p, secs = out[arm]
        print(f"{chi:>4}{p:>8}  {arm:>5}" + "".join(f"{e[k]:>10.2e}" for k in ('infid', 1, 2, 3, 4, 'p_near', 'p_far', 'E')) + f"{secs:>9.0f}s", flush=True)
    es, ef = out['svd'][0], out['fit'][0]
    print(f"{'':>17}{'fit/svd':>5}" + "".join(f"{ef[k] / es[k]:>10.2f}" for k in ('infid', 1, 2, 3, 4, 'p_near', 'p_far', 'E')), flush=True)
