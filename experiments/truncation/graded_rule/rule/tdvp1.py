"""Fixed-bond-dimension 1-site TDVP on an open chain, second order (symmetric right-left sweep), Lanczos exponentials.

The state is first evolved by SVD-TEBD with no truncation until its largest bond reaches chi, truncated once to
chi, and then carried by TDVP, which never changes a bond dimension. This is the standard fixed-chi baseline for
local observables, with the same stored parameters as every other arm at that chi.
"""
import numpy as np
import mpsenh as M
import mpsenv
import mfc


def lanczos_expm(apply, v, tau, m=20, tol=1e-13):
    """exp(tau * A) v for a Hermitian A given by apply(), tau complex, by a Krylov projection."""
    shape = v.shape
    nrm = np.linalg.norm(v)
    if nrm == 0:
        return v
    V = [v.reshape(-1) / nrm]
    alpha, beta = [], []
    for j in range(m):
        w = apply(V[j].reshape(shape)).reshape(-1)
        a = np.vdot(V[j], w).real
        alpha.append(a)
        w = w - a * V[j] - (beta[-1] * V[j - 1] if j > 0 else 0)
        for u in V:                                  # full reorthogonalisation, m is small
            w = w - np.vdot(u, w) * u
        b = np.linalg.norm(w)
        if b < tol or j == m - 1:
            break
        beta.append(b)
        V.append(w / b)
    k = len(alpha)
    Tm = np.diag(alpha) + np.diag(beta[:k - 1], 1) + np.diag(beta[:k - 1], -1)
    ev, U = np.linalg.eigh(Tm)
    coef = U @ (np.exp(tau * ev) * U[0, :])
    out = sum(c * x for c, x in zip(coef, V[:k]))
    return nrm * out.reshape(shape)


class TDVP1:
    def __init__(self, model, N, T):
        self.N = N
        self.env = mpsenv.HEnv(model, N)
        self.T = [t.copy() for t in T]
        self.env.init(self.T)                       # right environments of a right-canonical chain

    def _h1(self, i):
        env = self.env
        return lambda x: env.apply1(x, env.L[i], i, env.R[i + 1])

    def _h0(self, i):
        """Bond matrix between site i-1 and site i."""
        env = self.env
        return lambda x: np.einsum('aij,ik,akl->jl', env.L[i], x, env.R[i], optimize=True)

    def step(self, dt):
        env, T, N = self.env, self.T, self.N
        h = dt / 2
        for i in range(N):                           # right sweep
            T[i] = lanczos_expm(self._h1(i), T[i], -1j * h)
            if i < N - 1:
                l, s, r = T[i].shape
                Q, R = np.linalg.qr(T[i].reshape(l * s, r))
                T[i] = Q.reshape(l, s, -1)
                env.L[i + 1] = env.up_left(env.L[i], T[i], i)
                R = lanczos_expm(self._h0(i + 1), R, +1j * h)
                T[i + 1] = np.tensordot(R, T[i + 1], axes=([1], [0]))
        for i in range(N - 1, -1, -1):               # left sweep
            T[i] = lanczos_expm(self._h1(i), T[i], -1j * h)
            if i > 0:
                l, s, r = T[i].shape
                Q, R = np.linalg.qr(T[i].reshape(l, s * r).conj().T)
                T[i] = Q.conj().T.reshape(-1, s, r)
                env.R[i] = env.up_right(env.R[i + 1], T[i], i)
                Rm = R.conj().T                      # T[i] = Rm @ Q^dag
                Rm = lanczos_expm(self._h0(i), Rm, +1j * h)
                T[i - 1] = np.tensordot(T[i - 1], Rm, axes=([2], [0]))


def run_tdvp(model, N, chi, T_final, dt_tebd, dt_tdvp):
    """SVD-TEBD (untruncated) until the largest bond exceeds chi, one truncation to chi, then 1-site TDVP."""
    Gs = M.make_gates(model, N, dt_tebd)
    T = mfc.right_canonicalize(M.initial_mps(model, N))
    t = 0.0
    while t < T_final - 1e-12:
        Tn = mfc.sweep_step([x.copy() for x in T], Gs, 10 ** 6)
        Tn = mfc.normalise(mfc.right_canonicalize(Tn))
        T, t = Tn, t + dt_tebd
        if mfc.max_bond(Tn) > chi:
            break
    T = mfc.pad_to(mfc.normalise(mfc.compress_svd(T, chi)), chi)
    T = mfc.normalise(mfc.right_canonicalize(T))
    # right_canonicalize can shrink padded bonds through QR; TDVP keeps whatever bond dimensions it is given
    solver = TDVP1(model, N, T)
    n = int(round((T_final - t) / dt_tdvp))
    for _ in range(n):
        solver.step(dt_tdvp)
    return solver.T, dict(t_start=t, steps=n)
