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
                 eps_min=1e-7, f_min=0.0, every=1, alphas=(1.0, 0.5, 0.25), pattern='all', rel_skip=1e-2, precision='f32', passes=1, beta=0.0, free=0.0, two=False, bmin=0, bmax=10 ** 6, aL=None, aR=None, wk=None, wst=1.0, wsta=1.0, wtau=None):
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
        self.passes = passes
        self.beta = beta
        self.free = free
        self.two = two
        self.bmin, self.bmax = bmin, bmax
        self.wk = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0} if wk is None else dict(zip(self.ks, wk))
        self.wst, self.wsta = wst, wsta
        self.wtau = wtau
        self.aL = a if aL is None else aL
        self.aR = a if aR is None else aR
        self.cand = []
        self._f32 = {}
        self.achosen = {}

    def start(self, T):
        self.T = T

    # ------------------------------------------------------------------ region shape data (state independent)
    def _region(self, L, at_end, pos=0):
        pos = pos if self.wst != 1.0 else 0
        key = (L, at_end, pos)
        if key in self._shape:
            return self._shape[key]
        D = 2 ** L
        fn = None
        if os.environ.get('SPCF_CACHE'):
            tag = '_'.join(str(x) for x in ('v3', self.model, L, int(at_end), pos, '-'.join(f'{t:g}' for t in self.taus), ''.join(map(str, self.ks)), '-'.join(f'{self.wk[kk]:g}' for kk in self.ks), f'{self.wst:g}', f'{self.wsta:g}', '' if self.wtau is None else '-'.join(f'{x:g}' for x in self.wtau)))
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
                wgt = self.wk.get(k, 1.0) * (self.wst if (off <= pos and off + k >= pos + 2) else 1.0)
                for s in itertools.product(range(4), repeat=k):
                    if any(s):
                        full = (0,) * off + s + (0,) * (L - off - k)
                        omega[full] = omega.get(full, 0.0) + wgt / (nwin * 2 ** k)
        strs = list(omega)
        Pm = np.array([_pstring(s) for s in strs])
        w = np.sqrt(np.array([omega[s] for s in strs]))[:, None]
        iu = np.triu_indices(D, 1)
        rows = []
        for ti, tau in enumerate(self.taus):
            U = (V * np.exp(-1j * tau * ev)) @ V.conj().T
            O = U.conj().T[None] @ Pm @ U[None]
            wt = np.sqrt(self.wsta) if tau == 0.0 else (np.sqrt(self.wtau[ti - (1 if self.taus[0] == 0.0 else 0)]) if self.wtau is not None else 1.0)
            rows.append(wt * w * np.concatenate([np.einsum('sii->si', O).real, 2 * O[:, iu[0], iu[1]].real,
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
        if b < self.bmin or b > self.bmax:
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        if (self.pattern in ('R', 'L') and dirn != self.pattern) or (self.pattern == 'alt' and (b % 2 == 0) != (dirn == 'R')):
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        t1 = pc()
        lo, hi = max(0, b - self.aL), min(self.N, b + 2 + self.aR)
        D, (Fs, Fd64), iu, dg = self._region(hi - lo, hi == self.N, b - lo)
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
        c64 = cdt
        LmMc, RmMc = LmM.astype(c64), RmM.astype(c64)
        LmMcc, RmMcc = LmMc.conj(), RmMc.conj()
        nsr = nrm2

        def make_chart(Bk_, Bp_, Mk_, svd_chart):
            """closures of the linearisation at the rank-k matrix Mk_ = projection of M on the subspace Bk_ (orthonormal), tangent
            directions Bp_ (orthonormal complement)"""
            nb_ = Bp_.shape[1]
            if dirn == 'R':
                Ak, Ap = Bk_.conj().T @ Mm, Bp_.conj().T @ Mm
                P1, Q1, P2, Q2 = Bp_, Ak, Bk_, Ap
                cs = np.sum(np.abs(Ak) ** 2, axis=1)
            else:
                MB = Mm @ Bk_
                P1, Q1, P2, Q2 = Mm @ Bp_, Bk_.conj().T, MB, Bp_.conj().T
                cs = np.sum(np.abs(MB) ** 2, axis=0)
            W0_ = Wof(Mk_.reshape(l, 2, 2, r))
            rho_ = W0_ @ W0_.conj().T
            tr_ = rho_[dg, dg].real.sum()
            hn_ = hvec(rho_) / tr_
            W0c = W0_.astype(c64)
            W0cH = W0c.conj().T
            hn_32, tr_32 = hn_.astype(fdt), fdt(tr_)
            P1c, Q1c, P2c, Q2c = P1.astype(c64), Q1.astype(c64), P2.astype(c64), Q2.astype(c64)
            P1H, Q1H = P1c.conj().T, Q1c.conj().T
            nc = nb_ * k
            two_ = self.two and svd_chart
            if two_:                                  # independent rotations of both bond subspaces, dM = U_p C_L S_k Vh_k + U_k S_k C_R^dag Vh_p
                P1, Q1, P2, Q2 = U[:, k:], sk[:, None] * Vh[:k], U[:, :k] * sk, Vh[k:]
                P1c, Q1c, P2c, Q2c = P1.astype(c64), Q1.astype(c64), P2.astype(c64), Q2.astype(c64)
                P1H, Q1H = P1c.conj().T, Q1c.conj().T
                cs = np.tile(sk ** 2, 1)
            nd_ = ((k * 2 * r) if dirn == 'R' else (k * 2 * l)) if (use_free and svd_chart) else 0
            Bkc = Bk_.astype(c64)
            BkH = Bkc.conj().T

            def jvp(x):
                nt = nc * (2 if two_ else 1) + nd_
                z = (x[:nt] + 1j * x[nt:]).astype(c64)
                C = z[:nc].reshape(nb_, k)
                if two_:
                    CR = z[nc:2 * nc].reshape(nb_, k)
                    dM = P1c @ C @ Q1c + P2c @ CR.conj().T @ Q2c
                else:
                    dM = P1c @ C @ Q1c + P2c @ C.conj().T @ Q2c
                if nd_:
                    Dm = z[nc:]
                    dM = dM + (Bkc @ Dm.reshape(k, 2 * r) if dirn == 'R' else Dm.reshape(2 * l, k) @ BkH)
                dW = Wof(dM.reshape(l, 2, 2, r), LmMc, RmMc)
                X1 = dW @ W0cH
                drho = X1 + X1.conj().T
                dh = hvec(drho)
                dh = (dh - hn_32 * dh[:nd].sum()) / tr_32
                out = Fd32 @ dh
                return np.concatenate([Fs @ dh.astype(float), out.astype(float)]) if ns else out.astype(float)

            def vjp(g):
                eta = Fd32.T @ g[ns:].astype(fdt)
                if ns:
                    eta = eta + (Fs.T @ g[:ns]).astype(fdt)
                eta = (eta - (eta @ hn_32) * ones_d) / tr_32
                H = np.zeros((D, D), dtype=c64)
                H[iu] = (eta[nd:nd + nu] + 1j * eta[nd + nu:]) / 2
                H = H + H.conj().T
                H[dg, dg] = eta[:nd]
                Gd = (2 * (H @ W0c)).reshape(nX, 4, nZ, no, npp)
                G = Gd.transpose(0, 1, 3, 2, 4).reshape(nX * 4 * no, nZ * npp) @ RmMcc.T
                G = G.reshape(nX, 4, no, r).transpose(1, 3, 2, 0).reshape(4 * r, no * nX) @ LmMcc
                Gth = G.reshape(2, 2, r, l).transpose(3, 0, 1, 2).reshape(l * 2, 2 * r)
                if two_:
                    Gc = P1H @ Gth @ Q1H
                    gz = np.concatenate([Gc.ravel(), (Q2c @ Gth.conj().T @ P2c).ravel()])
                else:
                    Gc = P1H @ Gth @ Q1H + Q2c @ Gth.conj().T @ P2c
                    gz = Gc.ravel()
                if nd_:
                    Gd_ = BkH @ Gth if dirn == 'R' else Gth @ Bkc
                    gz = np.concatenate([gz, Gd_.ravel()])
                return np.concatenate([gz.real, gz.imag]).astype(float)

            sc = np.broadcast_to(cs[None, :], (nb_, k)).ravel()
            if two_:
                sc = np.concatenate([sc, sc])
            if nd_:
                sc = np.concatenate([sc, np.full(nd_, self.free * float(np.mean(cs)))])
            return jvp, vjp, np.tile(sc, 2) / nsr

        def retract(Bk_, Bp_, x_, al):
            nb_ = Bp_.shape[1]
            nc = nb_ * k
            nt = len(x_) // 2
            z = al * (x_[:nt] + 1j * x_[nt:])
            if self.two and nt == 2 * nc and self.passes == 1:
                CL, CR = z[:nc].reshape(nb_, k), z[nc:].reshape(nb_, k)
                QL, _ = np.linalg.qr(U[:, :k] + U[:, k:] @ CL)
                QR, _ = np.linalg.qr(Vh[:k].conj().T + Vh[k:].conj().T @ CR)
                X = QL.conj().T @ Mm @ QR                      # k x k
                Mk = QL @ X @ QR.conj().T
                return (QL, QR), Mk, X
            C = z[:nc].reshape(nb_, k)
            Q, _ = np.linalg.qr(Bk_ + Bp_ @ C)
            if nt > nc:
                Dm = z[nc:]
                if dirn == 'R':
                    X = Q.conj().T @ Mm + Dm.reshape(k, 2 * r)
                    return Q, Q @ X, X
                X = Mm @ Q + Dm.reshape(2 * l, k)
                return Q, X @ Q.conj().T, X
            Mk = Q @ (Q.conj().T @ Mm) if dirn == 'R' else (Mm @ Q) @ Q.conj().T
            return Q, Mk, None

        use_free = self.free > 0 and self.passes == 1
        Kmat = None
        far = (lo > 0) if dirn == 'R' else (hi < self.N)
        if self.beta > 0 and far:
            # first-order change of the reduced state of the sites OUTSIDE the fit region (the one-sided block that the rotation
            # of the kept subspace acts on): sum_ji C_ji s_i^2 |p_j><k_i| + h.c., traced over the region sites, must vanish
            if dirn == 'R':
                Bk3, Bp3 = Bk.reshape(l, 2, k), Bp.reshape(l, 2, nb)
                psi = np.einsum('oxl,laj->oxaj', Lm, Bp3)
                phi = np.einsum('oxl,lai->oxai', Lm, Bk3)
                T4 = np.einsum('oxaj,pxai->opji', psi, phi.conj())
            else:
                Bk3, Bp3 = Bk.reshape(2, r, k), Bp.reshape(2, r, nb)
                psi = np.einsum('rzp,brj->pzbj', Rm, Bp3)
                phi = np.einsum('rzp,bri->pzbi', Rm, Bk3)
                T4 = np.einsum('pzbj,qzbi->pqji', psi, phi.conj())
            no_ = T4.shape[0]
            t_ = T4 * (sk ** 2)[None, None, None, :]
            tH = t_.conj().swapaxes(0, 1)
            iu_o = np.triu_indices(no_, 1)

            def herm_vec(Hm_):                      # Hermitian (o,o,m) -> real (o*o, m)
                return np.concatenate([np.einsum('iim->im', Hm_).real, Hm_[iu_o[0], iu_o[1]].real, Hm_[iu_o[0], iu_o[1]].imag], axis=0)
            Ka = herm_vec((t_ + tH).reshape(no_, no_, nb * k))
            Kb = herm_vec((1j * (t_ - tH)).reshape(no_, no_, nb * k))
            Kmat = np.concatenate([Ka, Kb], axis=1) / nrm2
            Kdiag = self.beta * np.sum(Kmat ** 2, axis=0)

        Bk_c, Bp_c, Mk_c, f_c, kept_c, Q_c, X_c = Bk, Bp, Mk0, f_svd, kept_svd, None, None
        phi_c = phi_svd
        t2 = pc()
        self.tm['setup'] += t2 - t1
        for ps in range(self.passes):
            t2 = pc()
            nb_ = Bp_c.shape[1]
            jvp, vjp, scale = make_chart(Bk_c, Bp_c, Mk_c, ps == 0)
            r_c = Fapply((hnorm(Mk_c.reshape(l, 2, 2, r)) - ht)[:, None])[:, 0] if ps else r0
            if ps == 0 and fw > 0:
                wji = (sk[None, :] ** 2 - sq[:, None] ** 2).clip(min=0.0)
                ridge = np.tile(wji.ravel(), 2) * (fw / nrm2)
            else:
                ridge = np.zeros(len(scale))
            Minv = 1.0 / (3.0 * scale + ridge + (Kdiag if (Kmat is not None and ps == 0) else 0.0) + 1e-9 * float(scale.max()))
            b_ = -vjp(r_c)
            if self.debug is not None and ps == 0:
                def exact(x):
                    Q, Mk, _X = retract(Bk, Bp, x, 1.0)
                    return None, None, Fapply((hnorm(Mk.reshape(l, 2, 2, r)) - ht)[:, None])[:, 0]
                self.debug.append(dict(jvp=jvp, vjp=vjp, exact=exact, nx=len(scale), nres=len(r0), r0=r0, f_svd=f_svd, Wof=Wof,
                                       Wold=lambda th: self._W(th, Lm, Rm)[0], l=l, r=r))

            useK = Kmat is not None and ps == 0

            def Aop(p_, jvp=jvp, vjp=vjp, ridge=ridge, useK=useK):
                self.nmv += 1
                out = vjp(jvp(p_)) + ridge * p_
                if useK:
                    out = out + self.beta * (Kmat.T @ (Kmat @ p_))
                return out

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
            ok = False
            for jj, al in enumerate(self.alphas):
                Q, Mk, Xc = retract(Bk_c, Bp_c, x, al)
                kj = float(np.vdot(Mk, Mk).real)
                rj = Fapply((hnorm(Mk.reshape(l, 2, 2, r)) - ht)[:, None])[:, 0]
                fj = float(rj @ rj)
                if fj + fw * (1.0 - kj / nrm2) < phi_c * (1 - 1e-3):
                    ok = True
                    break
            self.tm['line'] += pc() - t3
            if not ok:
                break
            if ps == 0:
                self.achosen[jj] = self.achosen.get(jj, 0) + 1
            f_c, kept_c, Q_c, Mk_c, X_c = fj, kj, Q, Mk, Xc
            phi_c = fj + fw * (1.0 - kj / nrm2)
            Bk_c = Q if not isinstance(Q, tuple) else Bk_c
            if ps + 1 < self.passes:
                Pb = Bp_c - Q @ (Q.conj().T @ Bp_c)
                Bp_c, _ = np.linalg.qr(Pb)
        if Q_c is None:
            self.rej = getattr(self, 'rej', {})
            self.rej[b] = self.rej.get(b, 0) + 1
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        self.fired += 1
        self.log.append((f_svd, f_c, (nrm2 - kept_svd) / nrm2, (nrm2 - kept_c) / nrm2, b, 1.0 if dirn == 'R' else 0.0, tail))
        Q = Q_c
        if isinstance(Q, tuple):
            QL, QR = Q
            if dirn == 'R':
                cen = X_c @ QR.conj().T
                n = np.linalg.norm(cen)
                return QL.reshape(l, 2, k), (cen / n).reshape(k, 2, r), 0.0, True
            cen = QL @ X_c
            n = np.linalg.norm(cen)
            return (cen / n).reshape(l, 2, k), QR.conj().T.reshape(k, 2, r), 0.0, True
        if dirn == 'R':
            cen = X_c if X_c is not None else Q.conj().T @ Mm
            n = np.linalg.norm(cen)
            return Q.reshape(l, 2, k), (cen / n).reshape(k, 2, r), 0.0, True
        cen = X_c if X_c is not None else Mm @ Q
        n = np.linalg.norm(cen)
        return (cen / n).reshape(l, 2, k), Q.conj().T.reshape(k, 2, r), 0.0, True
