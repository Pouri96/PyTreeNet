"""SPC cut without JIT: one Gauss-Newton step of the lookahead objective, solved matrix-free, in plain numpy.

Same target and objective as lookcut.LookCut (1-,2-,3-site window marginals of the region around the cut, after exact
evolution of the region for the times tau, scaled by 1/sqrt(number of windows)), with two changes that make it cheap.

1. The objective is a fixed quadratic form in the region density matrix. For the Hermitian-basis vector h(rho)
   (diagonal, real and imaginary parts of the strict upper triangle) the residual is r = F (h(rho) - h(rho_target)), where a
   row of F is the Heisenberg-evolved Pauli string U^dag P U written in that basis, weighted by sqrt(omega_P) with omega_P the
   total window weight of P. F depends only on the region shape (number of sites, whether it touches the right chain end),
   so nothing is evolved in time during the run.
2. The correction lives in the tangent space of the retained subspace, Q = qr(U_k + U_perp C), and C is the minimiser of
   ||J C + r0||^2 + fw * sum_ji w_ji |C_ji|^2 / ||M||^2 with w_ji = s_i^2 - s_pj^2 (the second-order expansion of the
   discarded weight), by diagonally preconditioned CG with a fixed iteration count. J and J^T are applied by the
   closed-form chain C -> dM -> dW -> drho -> dh -> dr and its transpose, J itself is never formed.

A step is accepted only if the exact objective (residual plus fw times the discarded-weight fraction) drops.
"""
import itertools
import os
import time
import numpy as np
import scipy.sparse as sps
import mpsenh as M

PA = [np.eye(2, dtype=complex), M.X, M.Y, M.Z]


def _pstring(a):
    m = np.eye(1, dtype=complex)
    for x in a:
        m = np.kron(m, PA[x])
    return m


class SPCFast:
    def __init__(self, model, N, a=2, taus=(1.0, 2.0), ks=(1, 2, 3), include_static=True, fw=0.1, iters=6, cg_tol=1e-3,
                 eps_min=1e-7, f_min=0.0, every=1, alphas=(1.0, 0.5, 0.25), pattern='all', rel_skip=1e-2, precision='f32'):
        self.model, self.N, self.a = model, N, a
        self.taus = ([0.0] if include_static else []) + list(taus)
        self.ks, self.fw, self.iters, self.cg_tol = tuple(ks), fw, iters, cg_tol
        self.eps_min, self.f_min, self.every, self.alphas, self.pattern = eps_min, f_min, every, tuple(alphas), pattern
        self.T = None
        self._shape = {}
        self.fired = self.calls = self.skipped = 0
        self.log = []
        self.tm = dict(svd=0.0, setup=0.0, solve=0.0, line=0.0, other=0.0)
        self.nmv = 0
        self.debug = None
        self.rel_skip, self.tmax = rel_skip, 0.0
        self.cdt, self.fdt = (np.complex64, np.float32) if precision == 'f32' else (np.complex128, np.float64)
        self.precision = precision
        self.cand = []
        self._f32 = {}
        self.achosen = {}

    def start(self, T):
        self.T = T

    # ------------------------------------------------------------------ region shape data (state independent)
    def _region(self, L, at_end):
        key = (L, at_end)
        if key in self._shape:
            return self._shape[key]
        D = 2 ** L
        fn = None
        if os.environ.get('SPCF_CACHE'):
            tag = '_'.join(str(x) for x in ('v2', self.model, L, int(at_end), '-'.join(f'{t:g}' for t in self.taus), ''.join(map(str, self.ks))))
            fn = os.path.join(os.environ['SPCF_CACHE'], f'spcf_{tag}.npz')
            if os.path.exists(fn):
                z = np.load(fn)
                Fs = sps.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
                self._shape[key] = (D, (Fs, z['Fd']), np.triu_indices(D, 1), np.arange(D))
                return self._shape[key]
        Hm = np.zeros((D, D), dtype=complex)
        for bb in range(L - 1):
            nn = self.N - 2 if (at_end and bb == L - 2) else 3
            Hm += np.kron(np.kron(np.eye(2 ** bb), M.h_bond(self.model, nn, self.N)), np.eye(2 ** (L - bb - 2)))
        ev, V = np.linalg.eigh(Hm)
        omega = {}
        for k in self.ks:
            if k > L:
                continue
            nwin = L - k + 1
            for off in range(nwin):
                for s in itertools.product(range(4), repeat=k):
                    if any(s):
                        full = (0,) * off + s + (0,) * (L - off - k)
                        omega[full] = omega.get(full, 0.0) + 1.0 / (nwin * 2 ** k)
        strs = list(omega)
        Pm = np.array([_pstring(s) for s in strs])
        w = np.sqrt(np.array([omega[s] for s in strs]))[:, None]
        iu = np.triu_indices(D, 1)
        rows = []
        for tau in self.taus:
            U = (V * np.exp(-1j * tau * ev)) @ V.conj().T
            O = U.conj().T[None] @ Pm @ U[None]
            rows.append(w * np.concatenate([np.einsum('sii->si', O).real, 2 * O[:, iu[0], iu[1]].real,
                                            2 * O[:, iu[0], iu[1]].imag], axis=1))
        dg = np.arange(D)
        ns = len(strs) if self.taus and self.taus[0] == 0.0 else 0
        Fall = np.concatenate(rows, axis=0)
        Fs = sps.csr_matrix(np.where(np.abs(Fall[:ns]) > 1e-10, Fall[:ns], 0.0))
        Fd = np.ascontiguousarray(Fall[ns:])
        self._shape[key] = (D, (Fs, np.ascontiguousarray(Fd)), iu, dg)
        if fn:
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            np.savez(fn, data=Fs.data, indices=Fs.indices, indptr=Fs.indptr, shape=np.array(Fs.shape), Fd=self._shape[key][1][1])
        return self._shape[key]

    # ------------------------------------------------------------------ environment maps and the theta -> W map
    def _maps(self, lo, hi, b, l, r):
        T = self.T
        if lo == b:
            Lm = np.eye(l, dtype=complex).reshape(l, 1, l)
        else:
            X = T[lo]
            for j in range(lo + 1, b):
                X = np.tensordot(X, T[j], axes=([X.ndim - 1], [0]))
                X = X.reshape(X.shape[0], -1, X.shape[-1])
            Lm = X.reshape(X.shape[0], -1, X.shape[-1])
        if hi == b + 2:
            Rm = np.eye(r, dtype=complex).reshape(r, 1, r)
        else:
            X = T[b + 2]
            for j in range(b + 3, hi):
                X = np.tensordot(X, T[j], axes=([X.ndim - 1], [0]))
                X = X.reshape(X.shape[0], -1, X.shape[-1])
            Rm = X.reshape(X.shape[0], -1, X.shape[-1])
        return Lm, Rm

    @staticmethod
    def _W(th, Lm, Rm):
        Y = np.tensordot(Lm, th, axes=([2], [0]))                    # o x a b r
        Y = np.tensordot(Y, Rm, axes=([4], [0]))                     # o x a b z p
        sh = Y.shape
        return Y.transpose(1, 2, 3, 4, 0, 5).reshape(sh[1] * sh[2] * sh[3] * sh[4], sh[0] * sh[5]), sh

    @staticmethod
    def _Wadj(Gd, sh, Lm, Rm):
        G = Gd.reshape(sh[1], sh[2], sh[3], sh[4], sh[0], sh[5])      # x a b z o p
        G = np.tensordot(G, Rm.conj(), axes=([3, 5], [1, 2]))        # x a b o r
        G = np.tensordot(G, Lm.conj(), axes=([0, 3], [1, 0]))        # a b r l
        return G.transpose(3, 0, 1, 2)

    # ------------------------------------------------------------------ the cut
    def __call__(self, theta, chi, dirn, A, B, b):
        self.calls += 1
        pc = time.perf_counter
        t0 = pc()
        l, r = theta.shape[0], theta.shape[3]
        Mm = theta.reshape(l * 2, 2 * r)
        U, s, Vh = np.linalg.svd(Mm, full_matrices=False)
        k = M._rank(s, chi)
        self.tm['svd'] += pc() - t0
        nrm2 = float(np.sum(s ** 2))
        if k >= len(s) or np.sum(s[k:] ** 2) <= self.eps_min * nrm2:
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        tail = float(np.sum(s[k:] ** 2) / nrm2)
        self.tmax = max(self.tmax, tail)
        if tail < self.rel_skip * self.tmax:
            self.skipped += 1
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        if self.every > 1 and ((self.calls - 1) // (2 * (self.N - 1))) % self.every:
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        if (self.pattern in ('R', 'L') and dirn != self.pattern) or (self.pattern == 'alt' and (b % 2 == 0) != (dirn == 'R')):
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        t1 = pc()
        lo, hi = max(0, b - self.a), min(self.N, b + 2 + self.a)
        D, (Fs, Fd64), iu, dg = self._region(hi - lo, hi == self.N)
        cdt, fdt = self.cdt, self.fdt
        Fd32 = self._f32.get(id(Fd64))
        if Fd32 is None:
            Fd32 = self._f32[id(Fd64)] = np.ascontiguousarray(Fd64.astype(fdt))
        Lm, Rm = self._maps(lo, hi, b, l, r)
        nb = len(s) - k
        sk, sq = s[:k], s[k:]
        if dirn == 'R':
            Bk, Bp = U[:, :k], U[:, k:]
            P1, Q1, P2, Q2 = Bp, sk[:, None] * Vh[:k], Bk, sq[:, None] * Vh[k:]
        else:
            Bk, Bp = Vh[:k].conj().T, Vh[k:].conj().T
            P1, Q1, P2, Q2 = U[:, k:] * sq, Vh[:k], U[:, :k] * sk, Vh[k:]
        fw, nd, nh = self.fw, len(dg), D * D
        nu = len(iu[0])
        ns = Fs.shape[0]
        no, nX, nZ, npp = Lm.shape[0], Lm.shape[1], Rm.shape[1], Rm.shape[2]
        LmM, RmM = Lm.reshape(no * nX, l), Rm.reshape(r, nZ * npp)

        def Wof(th, LmM=LmM, RmM=RmM):                    # theta (l,2,2,r) -> W (D, o p), the region state with the outer bonds open
            Y = LmM @ th.reshape(l, 4 * r)
            Y = Y.reshape(no * nX * 4, r) @ RmM
            return Y.reshape(no, nX, 4, nZ, npp).transpose(1, 2, 3, 0, 4).reshape(D, no * npp)

        def hvec(rho):
            return np.concatenate([rho[dg, dg].real, rho[iu].real, rho[iu].imag])

        def hnorm(th):
            W = Wof(th)
            rho = W @ W.conj().T
            return hvec(rho) / rho[dg, dg].real.sum()

        def Fapply(H):                                   # residual rows for the columns of H (nh x m), F applied in float32
            Rd = (Fd32 @ H.astype(fdt)).astype(float)
            return np.concatenate([Fs @ H, Rd], axis=0) if ns else Rd

        Mk0 = (U[:, :k] * sk) @ Vh[:k]
        W0 = Wof(Mk0.reshape(l, 2, 2, r))
        rho0 = W0 @ W0.conj().T
        tr0 = rho0[dg, dg].real.sum()
        hn0 = hvec(rho0) / tr0
        ht = hnorm(theta)
        r0 = Fapply((hn0 - ht)[:, None])[:, 0]
        f_svd = float(r0 @ r0)
        kept_svd = float(np.sum(sk ** 2))
        self.cand.append((float(np.sum(sq ** 2) / nrm2), f_svd))
        if f_svd < self.f_min:
            self.skipped += 1
            self.tm['setup'] += pc() - t1
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        phi_svd = f_svd + fw * (1.0 - kept_svd / nrm2)
        ones_d = np.zeros(nh, dtype=fdt)
        ones_d[:nd] = 1.0
        # complex64 copies for the CG chain
        c64 = cdt
        LmMc, RmMc = LmM.astype(c64), RmM.astype(c64)
        W0c = W0.astype(c64)
        W0cH = W0c.conj().T
        hn0_32 = hn0.astype(fdt)
        tr0_32 = fdt(tr0)
        P1c, Q1c, P2c, Q2c = P1.astype(c64), Q1.astype(c64), P2.astype(c64), Q2.astype(c64)
        P1H, Q1H = P1c.conj().T, Q1c.conj().T
        LmMcc, RmMcc = LmMc.conj(), RmMc.conj()

        def jvp(x):
            C = (x[:nb * k] + 1j * x[nb * k:]).reshape(nb, k).astype(c64)
            dM = P1c @ C @ Q1c + P2c @ C.conj().T @ Q2c
            dW = Wof(dM.reshape(l, 2, 2, r), LmMc, RmMc)
            X1 = dW @ W0cH
            drho = X1 + X1.conj().T
            dh = hvec(drho)
            dh = (dh - hn0_32 * dh[:nd].sum()) / tr0_32
            out = Fd32 @ dh
            return np.concatenate([Fs @ dh.astype(float), out.astype(float)]) if ns else out.astype(float)

        def vjp(g):
            eta = Fd32.T @ g[ns:].astype(fdt)
            if ns:
                eta = eta + (Fs.T @ g[:ns]).astype(fdt)
            eta = (eta - (eta @ hn0_32) * ones_d) / tr0_32
            H = np.zeros((D, D), dtype=c64)
            H[iu] = (eta[nd:nd + nu] + 1j * eta[nd + nu:]) / 2
            H = H + H.conj().T
            H[dg, dg] = eta[:nd]
            Gd = (2 * (H @ W0c)).reshape(nX, 4, nZ, no, npp)
            G = Gd.transpose(0, 1, 3, 2, 4).reshape(nX * 4 * no, nZ * npp) @ RmMcc.T          # (x ab o, r)
            G = G.reshape(nX, 4, no, r).transpose(1, 3, 2, 0).reshape(4 * r, no * nX) @ LmMcc  # (ab r, l)
            Gth = G.reshape(2, 2, r, l).transpose(3, 0, 1, 2).reshape(l * 2, 2 * r)
            Gc = P1H @ Gth @ Q1H + Q2c @ Gth.conj().T @ P2c
            return np.concatenate([Gc.real.ravel(), Gc.imag.ravel()]).astype(float)

        wji = (sk[None, :] ** 2 - sq[:, None] ** 2).clip(min=0.0)
        ridge = np.tile(wji.ravel(), 2) * (fw / nrm2)
        scale = np.tile(np.broadcast_to(sk[None, :], (nb, k)).ravel(), 2) ** 2 / nrm2
        Minv = 1.0 / (3.0 * scale + ridge + 1e-9 * float(scale.max()))
        b_ = -vjp(r0)
        if self.debug is not None:
            def exact(x):
                C = (x[:nb * k] + 1j * x[nb * k:]).reshape(nb, k)
                Q, _ = np.linalg.qr(Bk + Bp @ C)
                Mk = Q @ (Q.conj().T @ Mm) if dirn == 'R' else (Mm @ Q) @ Q.conj().T
                return None, None, Fapply((hnorm(Mk.reshape(l, 2, 2, r)) - ht)[:, None])[:, 0]
            self.debug.append(dict(jvp=jvp, vjp=vjp, exact=exact, nx=2 * nb * k, nres=len(r0), r0=r0, f_svd=f_svd, Wof=Wof,
                                   Wold=lambda th: self._W(th, Lm, Rm)[0], l=l, r=r))
        t2 = pc()
        self.tm['setup'] += t2 - t1

        def Aop(p):
            self.nmv += 1
            return vjp(jvp(p)) + ridge * p

        x = np.zeros_like(b_)
        res = b_.copy()
        z = Minv * res
        p = z.copy()
        rz = res @ z
        bn = np.linalg.norm(b_)
        for _ in range(self.iters):
            Ap = Aop(p)
            al = rz / (p @ Ap)
            x += al * p
            res -= al * Ap
            if np.linalg.norm(res) < self.cg_tol * bn:
                break
            z = Minv * res
            rzn = res @ z
            p = z + (rzn / rz) * p
            rz = rzn
        t3 = pc()
        self.tm['solve'] += t3 - t2
        j, fa_j, kept_j, Qacc = -1, 0.0, 0.0, None
        for jj, al in enumerate(self.alphas):
            C = (al * x[:nb * k] + 1j * al * x[nb * k:]).reshape(nb, k)
            Q, _ = np.linalg.qr(Bk + Bp @ C)
            Mk = Q @ (Q.conj().T @ Mm) if dirn == 'R' else (Mm @ Q) @ Q.conj().T
            kj = float(np.vdot(Mk, Mk).real)
            rj = Fapply((hnorm(Mk.reshape(l, 2, 2, r)) - ht)[:, None])[:, 0]
            fj = float(rj @ rj)
            if fj + fw * (1.0 - kj / nrm2) < phi_svd * (1 - 1e-3):
                j, fa_j, kept_j, Qacc = jj, fj, kj, Q
                break
        self.tm['line'] += pc() - t3
        if j < 0:
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        self.fired += 1
        self.achosen[j] = self.achosen.get(j, 0) + 1
        self.log.append((f_svd, fa_j, (nrm2 - kept_svd) / nrm2, (nrm2 - kept_j) / nrm2))
        Q = Qacc
        if dirn == 'R':
            cen = Q.conj().T @ Mm
            n = np.linalg.norm(cen)
            return Q.reshape(l, 2, k), (cen / n).reshape(k, 2, r), 0.0, True
        cen = Mm @ Q
        n = np.linalg.norm(cen)
        return (cen / n).reshape(l, 2, k), Q.conj().T.reshape(k, 2, r), 0.0, True
