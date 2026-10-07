"""Marginal-fit compression (MFC): compress an MPS to rank chi by fitting its window marginals.

One Trotter step is a gate sweep at an enlarged working rank chi_w followed by a compression to chi:
  dsvd   sequential SVD compression of the step result (the fidelity-oriented baseline with the same transient)
  mfc    L-BFGS fit of all 1-, 2-, ..., kmax-site window marginals of the step result, plus fw * (1 - F),
         started at the dsvd point; F is the fidelity with the step result
Marginals of an MPS are computed through transfer matrices, so nothing dense is formed. The objective and its
gradient come from JAX; the fit is gauge invariant and runs on bond-dimension templates so that it compiles once.
"""
import os
import sys
if os.environ.get('PYLIBS'):
    sys.path.insert(0, os.environ['PYLIBS'])
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64', True)
import mpsenh as M


# --------------------------------------------------------------------------- numpy MPS helpers
def right_canonicalize(T):
    """Right-canonical form with the norm and the centre on site 0."""
    T = [t.copy() for t in T]
    for i in range(len(T) - 1, 0, -1):
        l, s, r = T[i].shape
        Q, R = np.linalg.qr(T[i].reshape(l, s * r).conj().T)
        k = Q.shape[1]
        T[i] = Q.conj().T.reshape(k, s, r)
        T[i - 1] = np.tensordot(T[i - 1], R.conj().T, axes=([2], [0]))
    return T


def normalise(T):
    n = np.sqrt(abs(np.vdot(T[0], T[0])))
    T[0] = T[0] / n
    return T


def compress_svd(T, chi):
    """Sequential SVD truncation of a right-canonical MPS (centre on site 0), left to right."""
    T = [t.copy() for t in T]
    for i in range(len(T) - 1):
        l, s, r = T[i].shape
        U, sv, Vh = np.linalg.svd(T[i].reshape(l * s, r), full_matrices=False)
        k = min(chi, int(np.sum(sv > 1e-14 * sv[0])))
        T[i] = U[:, :k].reshape(l, s, k)
        T[i + 1] = np.tensordot((sv[:k, None] * Vh[:k]), T[i + 1], axes=([1], [0]))
    return T


def sweep_step(T, Gs, chi_w):
    """One Strang step (right sweep then left sweep) with SVD cuts at chi_w. T must be right-canonical."""
    N = len(T)
    for b in range(N - 1):
        th = np.einsum('abst,lstr->labr', Gs[b], np.tensordot(T[b], T[b + 1], axes=([2], [0])))
        T[b], T[b + 1], _, _ = M.svd_cut(th, chi_w, 'R')
    for b in range(N - 2, -1, -1):
        th = np.einsum('abst,lstr->labr', Gs[b], np.tensordot(T[b], T[b + 1], axes=([2], [0])))
        T[b], T[b + 1], _, _ = M.svd_cut(th, chi_w, 'L')
    return T


def pad_to(T, chi):
    """Zero-pad bond dimensions up to min(chi, 2^i, 2^(N-i)) (exact), so shapes are a fixed template."""
    N = len(T)
    bonds = [1] + [min(chi, 2 ** min(i, N - i)) for i in range(1, N)] + [1]
    out = []
    for i, t in enumerate(T):
        l, s, r = t.shape
        z = np.zeros((bonds[i], s, bonds[i + 1]), dtype=complex)
        z[:l, :, :r] = t
        out.append(z)
    return out


def max_bond(T):
    return max(t.shape[2] for t in T)


# --------------------------------------------------------------------------- JAX objective
def _env_left(Ts):
    L = [jnp.ones((1, 1), dtype=complex)]
    for A in Ts:
        L.append(jnp.einsum('ab,asc,bsd->cd', L[-1], jnp.conj(A), A))
    return L


def _env_right(Ts):
    R = [None] * (len(Ts) + 1)
    R[len(Ts)] = jnp.ones((1, 1), dtype=complex)
    for i in range(len(Ts) - 1, -1, -1):
        R[i] = jnp.einsum('asc,bsd,cd->ab', jnp.conj(Ts[i]), Ts[i], R[i + 1])
    return R


def _window(L, As, R):
    X = L[:, :, None, None]
    for A in As:
        X = jnp.einsum('abST,bsd,atc->cdSsTt', X, A, jnp.conj(A))
        c, d, S, s, T_, t = X.shape
        X = X.reshape(c, d, S * s, T_ * t)
    return jnp.einsum('cdST,cd->ST', X, R)


def marginals(Ts, ks):
    """rho_{i,k}(Ts) for every window of every size k in ks, normalised. dict k -> (windows, D, D).

    One progressive contraction per start site serves every window size that starts there.
    """
    N = len(Ts)
    L, R = _env_left(Ts), _env_right(Ts)
    nrm = jnp.real(L[N][0, 0])
    kmax = max(ks)
    out = {k: [] for k in ks}
    for i in range(N):
        X = L[i][:, :, None, None]
        for j in range(1, kmax + 1):
            if i + j > N:
                break
            A = Ts[i + j - 1]
            X = jnp.einsum('abST,bsd,atc->cdSsTt', X, A, jnp.conj(A))
            c, d, S, s_, T_, t_ = X.shape
            X = X.reshape(c, d, S * s_, T_ * t_)
            if j in out:
                out[j].append(jnp.einsum('cdST,cd->ST', X, R[i + j]))
    return {k: jnp.stack(v) / nrm for k, v in out.items()}


def overlap(Ta, Tb):
    """<Ta|Tb>."""
    E = jnp.ones((1, 1), dtype=complex)
    for A, B in zip(Ta, Tb):
        E = jnp.einsum('ab,asc,bsd->cd', E, jnp.conj(A), B)
    return E[0, 0]


def make_objective(shapes, ks, fw, weights):
    def unpack(x):
        out, p = [], 0
        for sh in shapes:
            n = int(np.prod(sh))
            out.append((x[p:p + n] + 1j * x[p + n:p + 2 * n]).reshape(sh))
            p += 2 * n
        return out

    def f(x, taus, Tw):
        Ts = unpack(x)
        marg = marginals(Ts, ks)
        tot = 0.0
        for k in ks:
            tot = tot + weights[k] * jnp.mean(jnp.sum(jnp.abs(marg[k] - taus[k]) ** 2, axis=(1, 2)))
        if fw > 0:
            nT = jnp.real(overlap(Ts, Ts))
            nW = jnp.real(overlap(Tw, Tw))
            tot = tot + fw * (1.0 - jnp.abs(overlap(Tw, Ts)) ** 2 / (nT * nW))
        return tot
    return jax.jit(jax.value_and_grad(f)), unpack


def pack(Ts):
    xs = []
    for t in Ts:
        xs += [t.real.reshape(-1), t.imag.reshape(-1)]
    return np.concatenate(xs)


class FitCompressor:
    """Compiles once per (template shapes, working shapes) and reuses the objective across steps."""
    def __init__(self, ks=(1, 2, 3), fw=0.01, maxiter=200, weights=None):
        self.ks, self.fw, self.maxiter = tuple(ks), fw, maxiter
        self.weights = weights or {k: 1.0 for k in ks}
        self._cache = {}
        self.evals = 0
        self.last = None

    def __call__(self, Tw, chi, init):
        shapes = tuple(t.shape for t in init)
        wshapes = tuple(t.shape for t in Tw)
        key = (shapes, wshapes)
        if key not in self._cache:
            self._cache[key] = make_objective(list(shapes), self.ks, self.fw, self.weights)
        vg, unpack = self._cache[key]
        Twj = [jnp.asarray(t) for t in Tw]
        taus = {k: v for k, v in marginals(Twj, self.ks).items()}

        def fun(x):
            val, g = vg(jnp.asarray(x), taus, Twj)
            self.evals += 1
            return float(val), np.asarray(g)
        x0 = pack(init)
        f0 = fun(x0)[0]
        res = minimize(fun, x0, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=self.maxiter, maxfun=2 * self.maxiter, ftol=1e-15, gtol=1e-11))
        self.last = (f0, float(res.fun), int(res.nit))
        return [np.asarray(t) for t in unpack(res.x)]


def run_mfc(model, N, chi, nsteps, dt, arm, chi_w_factor=2, ks=(1, 2, 3), fw=0.01, maxiter=200, progress=None):
    """Evolve nsteps Trotter steps with per-step compression 'dsvd' or 'mfc'. Returns (MPS, info)."""
    Gs = M.make_gates(model, N, dt)
    T = right_canonicalize(M.initial_mps(model, N))
    chi_w = int(round(chi_w_factor * chi))
    fc = FitCompressor(ks=ks, fw=fw, maxiter=maxiter) if arm == 'mfc' else None
    peak, nfit, hist = 1, 0, []
    for step in range(nsteps):
        T = sweep_step(T, Gs, chi_w)
        T = normalise(right_canonicalize(T))
        peak = max(peak, max_bond(T))
        if max_bond(T) > chi:
            T0 = normalise(compress_svd(T, chi))
            if arm == 'mfc':
                init = pad_to(T0, chi)
                Tw = pad_to(T, chi_w)
                T = fc(Tw, chi, init)
                hist.append(fc.last)
                nfit += 1
            else:
                T = T0
            T = normalise(right_canonicalize(T))
        if progress:
            progress(step, T)
    return T, dict(peak=peak, nfit=nfit, hist=hist, chi_w=chi_w)
