"""SPC cut for TEBD: SVD compression followed by a first-order restoration of the cut-region marginals.

Compression at a gate cut is the orthogonal projection of the two-site tensor M onto its top-k singular subspace. The
restored functionals are the 1-, 2-, 3-site window marginals of the region around the cut, after the region state has
been evolved exactly under its own Hamiltonian for the times tau (same target and same region propagators as
lookcut.LookCut, taken from the untruncated two-site tensor already present in TEBD, so the peak bond dimension is chi).

The correction lives in the tangent space of the retained subspace, Q = qr(U_k + U_perp C). To first order in C the
functionals change linearly, r(C) = r0 + J C, while the fidelity loss is second order in C. The restoration is the
minimal-norm solution of J C = -r0 with the singular values of J below rcond * sigma_max discarded (mode 'hard'), or its
Tikhonov-smoothed form (mode 'soft'). Nothing is iterated beyond `passes` re-linearisations, and a step is accepted
only if it lowers the residual norm.
"""
import os
import sys
if os.environ.get('PYLIBS'):
    sys.path.insert(0, os.environ['PYLIBS'])
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
if os.environ.get('JAXCACHE'):
    jax.config.update('jax_compilation_cache_dir', os.environ['JAXCACHE'])
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
import mpsenh as M
import lookcut


class SPCCut(lookcut.LookCut):
    def __init__(self, model, N, a=2, taus=(0.5, 1.0, 1.5, 2.0), ks=(1, 2, 3), mode='hard', rcond=1e-2, passes=1,
                 include_static=True, eps_min=1e-10):
        super().__init__(model, N, a=a, taus=taus, ks=ks, fw=0.0, maxiter=0, include_static=include_static)
        self.mode, self.rcond, self.passes, self.eps_min = mode, rcond, passes, eps_min
        self.log = []

    def __call__(self, theta, chi, dirn, A, B, b):
        self.calls += 1
        l, r = theta.shape[0], theta.shape[3]
        Mm = theta.reshape(l * 2, 2 * r)
        U, s, Vh = np.linalg.svd(Mm, full_matrices=False)
        k = M._rank(s, chi)
        if k >= len(s) or np.sum(s[k:] ** 2) <= self.eps_min * np.sum(s ** 2):
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        lo, hi = max(0, b - self.a), min(self.N, b + 2 + self.a)
        Lm, Rm = self._maps(lo, hi, b, l, r)
        key = (l, r, k, len(s), hi - lo, lo == 0, hi == self.N, b - lo, Lm.shape, Rm.shape, dirn)
        marginals, region_rho, _ = self._objective(key, lo, hi, b)
        Lmj, Rmj = jnp.asarray(Lm), jnp.asarray(Rm)
        Mj = jnp.asarray(Mm)
        if ('tgt',) + key not in self._fun:
            def target(th, Lm_, Rm_):
                rho_t = region_rho(th, Lm_, Rm_)
                return marginals(rho_t / jnp.real(jnp.trace(rho_t)))
            self._fun[('tgt',) + key] = jax.jit(target)
        tgt = self._fun[('tgt',) + key](jnp.asarray(theta), Lmj, Rmj)
        if dirn == 'R':
            Bk, Bp = U[:, :k], U[:, k:]
        else:
            Bk, Bp = Vh[:k].conj().T, Vh[k:].conj().T
        Bkj, Bpj = jnp.asarray(Bk), jnp.asarray(Bp)
        nb = Bp.shape[1]
        nrm2 = float(np.sum(s ** 2))

        rkey = ('res',) + key
        if rkey not in self._fun:
            def res(x, Bk_, Bp_, Mj_, tg, Lm_, Rm_):
                C = (x[:nb * k] + 1j * x[nb * k:]).reshape(nb, k)
                Q, _ = jnp.linalg.qr(Bk_ + Bp_ @ C)
                Mk = Q @ (Q.conj().T @ Mj_) if dirn == 'R' else (Mj_ @ Q) @ Q.conj().T
                rho = region_rho(Mk.reshape(l, 2, 2, r), Lm_, Rm_)
                rho = rho / jnp.real(jnp.trace(rho))
                parts = []
                for u, v in zip(marginals(rho), tg):
                    d = (u - v) / jnp.sqrt(u.shape[0])
                    parts += [jnp.real(d).ravel(), jnp.imag(d).ravel()]
                kept = jnp.real(jnp.vdot(Mk, Mk))
                return jnp.concatenate(parts), kept
            def fk(x, *a):
                rv, kept = res(x, *a)
                return jnp.vdot(rv, rv), kept
            evalb = jax.jit(jax.vmap(fk, in_axes=(0,) + (None,) * 6))
            hard, rc2 = self.mode == 'hard', self.rcond ** 2
            alphas = jnp.array([1.0, 0.5, 0.25, 0.125])

            def stepfn(x, *a):
                J, rv = jax.jacfwd(lambda y: (lambda q: (q, q))(res(y, *a)[0]), has_aux=True)(x)
                w, V = jnp.linalg.eigh(J.T @ J)
                if hard:
                    inv = jnp.where(w > rc2 * w[-1], 1.0 / jnp.maximum(w, 1e-300), 0.0)
                else:
                    inv = 1.0 / (w + rc2 * w[-1])
                step = -V @ (inv * (V.T @ (J.T @ rv)))
                cand = x[None] + alphas[:, None] * step[None]
                f, kept = jax.vmap(fk, in_axes=(0,) + (None,) * 6)(cand, *a)
                return cand, f, kept
            self._fun[rkey] = (evalb, jax.jit(stepfn))
        evalb, stepfn = self._fun[rkey]
        args = (Bkj, Bpj, Mj, tgt, Lmj, Rmj)

        x = np.zeros(2 * nb * k)
        f0, k0 = evalb(jnp.asarray(x[None]), *args)
        f_svd, kept_svd = float(f0[0]), float(k0[0])
        f_cur, kept_cur = f_svd, kept_svd
        for _ in range(self.passes):
            cand, fn, kn = stepfn(jnp.asarray(x), *args)
            fn = np.asarray(fn)
            j = int(np.argmin(fn))
            if fn[j] >= f_cur * (1 - 1e-3):
                break
            f_cur, kept_cur, x = float(fn[j]), float(kn[j]), np.asarray(cand[j])
        if f_cur >= f_svd * (1 - 1e-3):
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        self.fired += 1
        self.log.append((f_svd, f_cur, (nrm2 - kept_svd) / nrm2, (nrm2 - kept_cur) / nrm2))
        C = (x[:nb * k] + 1j * x[nb * k:]).reshape(nb, k)
        Q, _ = np.linalg.qr(Bk + Bp @ C)
        if dirn == 'R':
            cen = Q.conj().T @ Mm
            n = np.linalg.norm(cen)
            return Q.reshape(l, 2, k), (cen / n).reshape(k, 2, r), 0.0, True
        cen = Mm @ Q
        n = np.linalg.norm(cen)
        return (cen / n).reshape(l, 2, k), Q.conj().T.reshape(k, 2, r), 0.0, True
