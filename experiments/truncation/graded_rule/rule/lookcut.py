"""Cut-local lookahead truncation (a drop-in cut for mpsenh.run_tebd).

At the cut of a gate on bonds (b, b+1) the two-site tensor theta has up to 2 chi singular directions. The kept subspace
is not the top-chi singular subspace but the one that minimises, inside the region of sites [b-a, b+2+a), the mismatch of
the 1-, 2-, 3-site window marginals after the region state has been evolved exactly for several short times tau under
the region's own Hamiltonian, plus fw times the discarded weight. The target is the same quantity computed from the
untruncated theta. Nothing larger than the single theta is ever stored, so the peak bond dimension is that of plain TEBD.

Right sweep:  theta_k = Q Q^dag theta with Q (2l x k) in span of the left singular vectors, left tensor Q, centre Q^dag theta.
Left sweep:   theta_k = theta Q Q^dag with Q (2r x k) in span of the right singular vectors, centre theta Q, right tensor Q^dag.
Q = qr(top-k + perp @ C), C optimised by L-BFGS from C = 0 (the SVD point).
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


class LookCut:
    def __init__(self, model, N, a=2, taus=(0.5, 1.0, 1.5, 2.0), ks=(1, 2, 3), fw=0.3, maxiter=40, include_static=True):
        self.model, self.N, self.a = model, N, a
        self.taus = ([0.0] if include_static else []) + list(taus)
        self.ks, self.fw, self.maxiter = tuple(ks), fw, maxiter
        self.T = None
        self._U = {}
        self._fun = {}
        self.fired = 0
        self.calls = 0

    def start(self, T):
        self.T = T

    # ------------------------------------------------------------------ region propagators
    def _unitaries(self, lo, hi):
        key = (lo, hi) if (lo == 0 or hi == self.N) else (self.a, 'bulk', hi - lo)
        if key not in self._U:
            L = hi - lo
            Hm = np.zeros((2 ** L, 2 ** L), dtype=complex)
            for bb in range(lo, hi - 1):
                Hm += np.kron(np.kron(np.eye(2 ** (bb - lo)), M.h_bond(self.model, bb, self.N)), np.eye(2 ** (hi - bb - 2)))
            ev, V = np.linalg.eigh(Hm)
            self._U[key] = [jnp.asarray((V * np.exp(-1j * t * ev)) @ V.conj().T) for t in self.taus]
        return self._U[key]

    # ------------------------------------------------------------------ objective
    def _objective(self, key, lo, hi, b):
        if key in self._fun:
            return self._fun[key]
        L = hi - lo
        off0 = b - lo                                   # site offset of the left site of theta inside the region
        Us = self._unitaries(lo, hi)

        wins = {k: list(range(0, L - k + 1)) for k in self.ks}

        def marginals(rho):
            out = []
            for U in Us:
                rt = U @ rho @ U.conj().T
                for k in self.ks:
                    vals = []
                    for off in wins[k]:
                        r = L - off - k
                        X = rt.reshape(2 ** off, 2 ** k, 2 ** r, 2 ** off, 2 ** k, 2 ** r)
                        vals.append(jnp.einsum('asbatb->st', X))
                    out.append(jnp.stack(vals))
            return out

        def region_rho(th, Lm, Rm):
            # Psi[o, xL, s1, s2, xR, o'] ; rho = W W^dag summed over the outer bonds
            Y = jnp.einsum('oxl,labr->oxabr', Lm, th)
            Y = jnp.einsum('oxabr,rzp->oxabzp', Y, Rm)
            D = Lm.shape[1] * 4 * Rm.shape[1]
            W = jnp.transpose(Y, (1, 2, 3, 4, 0, 5)).reshape(D, -1)
            return W @ W.conj().T

        def build(Mk, l, r):
            return Mk.reshape(l, 2, 2, r)

        self._fun[key] = (marginals, region_rho, build)
        return self._fun[key]

    def _maps(self, lo, hi, b, l, r):
        T = self.T
        if lo == b:
            Lm = np.eye(l, dtype=complex).reshape(l, 1, l)
        else:
            X = T[lo]                                                     # (o, 2, c)
            for j in range(lo + 1, b):
                X = np.tensordot(X, T[j], axes=([X.ndim - 1], [0]))
                X = X.reshape(X.shape[0], -1, X.shape[-1])
            Lm = X.reshape(X.shape[0], -1, X.shape[-1])                   # (o, 2^(b-lo), l)
        if hi == b + 2:
            Rm = np.eye(r, dtype=complex).reshape(r, 1, r)
        else:
            X = T[b + 2]                                                  # (c, 2, o)
            for j in range(b + 3, hi):
                X = np.tensordot(X, T[j], axes=([X.ndim - 1], [0]))
                X = X.reshape(X.shape[0], -1, X.shape[-1])
            Rm = X.reshape(X.shape[0], -1, X.shape[-1])                   # (r, 2^(hi-b-2), o')
        return Lm, Rm

    # ------------------------------------------------------------------ the cut
    def __call__(self, theta, chi, dirn, A, B, b):
        self.calls += 1
        l, r = theta.shape[0], theta.shape[3]
        Mm = theta.reshape(l * 2, 2 * r)
        U, s, Vh = np.linalg.svd(Mm, full_matrices=False)
        k = M._rank(s, chi)
        if k >= len(s) or np.sum(s[k:] ** 2) <= 0:
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        lo, hi = max(0, b - self.a), min(self.N, b + 2 + self.a)
        Lm, Rm = self._maps(lo, hi, b, l, r)
        L = hi - lo
        key = (l, r, k, len(s), hi - lo, lo == 0, hi == self.N, b - lo, Lm.shape, Rm.shape, dirn)
        marginals, region_rho, _ = self._objective(key, lo, hi, b)
        Lmj, Rmj = jnp.asarray(Lm), jnp.asarray(Rm)
        thj = jnp.asarray(theta)
        rho_t = region_rho(thj, Lmj, Rmj)
        rho_t = rho_t / jnp.real(jnp.trace(rho_t))
        tgt = marginals(rho_t)
        nrm2 = float(np.sum(s ** 2))
        if dirn == 'R':
            Bk, Bp = U[:, :k], U[:, k:]
        else:
            Bk, Bp = Vh[:k].conj().T, Vh[k:].conj().T
        Bkj, Bpj = jnp.asarray(Bk), jnp.asarray(Bp)
        Mj = jnp.asarray(Mm)
        nb = Bp.shape[1]
        fw = self.fw

        vgkey = ('vg',) + key
        if vgkey not in self._fun:
            def f(x, Bk_, Bp_, Mj_, tg, Lm_, Rm_):
                C = (x[:nb * k] + 1j * x[nb * k:]).reshape(nb, k)
                Q, _ = jnp.linalg.qr(Bk_ + Bp_ @ C)
                if dirn == 'R':
                    Mk = Q @ (Q.conj().T @ Mj_)
                else:
                    Mk = (Mj_ @ Q) @ Q.conj().T
                kept = jnp.real(jnp.vdot(Mk, Mk))
                rho = region_rho(Mk.reshape(l, 2, 2, r), Lm_, Rm_)
                rho = rho / jnp.real(jnp.trace(rho))
                mg = marginals(rho)
                tot = 0.0
                for u, v in zip(mg, tg):
                    tot = tot + jnp.mean(jnp.sum(jnp.abs(u - v) ** 2, axis=(1, 2)))
                return tot + fw * (1.0 - kept / jnp.real(jnp.vdot(Mj_, Mj_)))
            self._fun[vgkey] = jax.jit(jax.value_and_grad(f))
        vg = self._fun[vgkey]

        def fun(x):
            v, g = vg(jnp.asarray(x), Bkj, Bpj, Mj, tgt, Lmj, Rmj)
            return float(v), np.asarray(g)
        x0 = np.zeros(2 * nb * k)
        f0 = fun(x0)[0]
        res = minimize(fun, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=self.maxiter, ftol=1e-14, gtol=1e-10))
        if res.fun >= f0:
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        self.fired += 1
        x = res.x
        C = (x[:nb * k] + 1j * x[nb * k:]).reshape(nb, k)
        Q, _ = np.linalg.qr(Bk + Bp @ C)
        if dirn == 'R':
            cen = Q.conj().T @ Mm
            n = np.linalg.norm(cen)
            return Q.reshape(l, 2, k), (cen / n).reshape(k, 2, r), 0.0, True
        cen = Mm @ Q
        n = np.linalg.norm(cen)
        return (cen / n).reshape(l, 2, k), Q.conj().T.reshape(k, 2, r), 0.0, True
