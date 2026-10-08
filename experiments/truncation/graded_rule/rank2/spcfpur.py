"""d-generic SPC cut for a purification MPS: site = (physical qubit, ancilla), fused index (p, a) = dp * a_index... order (p, a).

Subclass of rule/spcfast.SPCFast. What changes with respect to the d = 2 cut:

* The two-site tensor theta is (l, d, d, r) with d = dp * da (dp = da = 2), the SVD is of the (l d) x (d r) matrix.
* The objective is unchanged. Its rows (the matrix F, the region Hamiltonian, the lookahead evolution) act on the PHYSICAL
  qubits of the region only, so ``_region`` / ``_strmeta`` / ``_low_span_resid`` are inherited. The ancillas of the region and
  the outer bonds are traced out: the region state W has the 2^L physical indices on the rows and (outer bond, every ancilla
  of the region, outer bond) on the columns, so rho_phys = W W^dag.
* The environment maps Lm (o, d^nl, l) and Rm (r, d^nr, p) of the base class are split into (physical, ancilla) factors and the
  ancilla factor is moved into the outer (column) index.
* ``_plain`` and the returned tensors are d-generic. beta (outside-region constraint) is not supported.

Everything else (Gauss-Newton step in the tangent space of the rank-k manifold, matrix-free CG, line search on the exact
objective) is the d = 2 algorithm line for line.
"""
import time
import numpy as np
import mpsenh as M
from spcfast import SPCFast


class SPCFPur(SPCFast):
    def __init__(self, *args, dp=2, da=2, **kw):
        super().__init__(*args, **kw)
        assert dp == 2, 'the F matrices are built for qubits'
        assert self.beta == 0.0, 'beta is not supported in the purification cut'
        self.dp, self.da = dp, da
        self.d = dp * da

    # ------------------------------------------------------------------ d-generic plain SVD cut (d = 2 in M.EnhCut._plain)
    def _plain(self, U, s, Vh, k, l, r, dirn):
        d = self.d
        s = s[:k]
        nrm = np.linalg.norm(s)
        U, Vh = U[:, :k], Vh[:k]
        if dirn == 'R':
            return U.reshape(l, d, k), ((s[:, None] / nrm) * Vh).reshape(k, d, r), 0.0, False
        return (U * (s / nrm)).reshape(l, d, k), Vh.reshape(k, d, r), 0.0, False

    # ------------------------------------------------------------------ environment maps with the ancillas moved to the column side
    def _maps(self, lo, hi, b, l, r):
        Lm, Rm = super()._maps(lo, hi, b, l, r)
        dp, da = self.dp, self.da
        nl, nr = b - lo, hi - (b + 2)
        o, p = Lm.shape[0], Rm.shape[2]
        Lx = Lm.reshape((o,) + (dp, da) * nl + (l,))
        perm = [0] + [2 + 2 * j for j in range(nl)] + [1 + 2 * j for j in range(nl)] + [1 + 2 * nl]
        Lx = Lx.transpose(perm).reshape(o * da ** nl, dp ** nl, l)
        Rx = Rm.reshape((r,) + (dp, da) * nr + (p,))
        perm = [0] + [1 + 2 * j for j in range(nr)] + [2 + 2 * j for j in range(nr)] + [1 + 2 * nr]
        Rx = Rx.transpose(perm).reshape(r, dp ** nr, da ** nr * p)
        return Lx, Rx

    # ------------------------------------------------------------------ the cut
    def __call__(self, theta, chi, dirn, A, B, b):
        self.calls += 1
        pc = time.perf_counter
        t0 = pc()
        d, dp, da = self.d, self.dp, self.da
        l, r = theta.shape[0], theta.shape[3]
        Mm = theta.reshape(l * d, d * r)
        U, s, Vh = np.linalg.svd(Mm, full_matrices=False)
        k = M._rank(s, chi)
        self.tm['svd'] += pc() - t0
        nrm2 = float(np.sum(s ** 2))
        if k >= len(s) or np.sum(s[k:] ** 2) <= self.eps_min * nrm2:
            return self._plain(U, s, Vh, k, l, r, dirn)
        tail = float(np.sum(s[k:] ** 2) / nrm2)
        self.tmax = max(self.tmax, tail)
        if tail < self.rel_skip * self.tmax:
            self.skipped += 1
            return self._plain(U, s, Vh, k, l, r, dirn)
        if self.every > 1 and ((self.calls - 1) // (2 * (self.N - 1))) % self.every:
            return self._plain(U, s, Vh, k, l, r, dirn)
        if b < self.bmin or b > self.bmax:
            return self._plain(U, s, Vh, k, l, r, dirn)
        if (self.pattern in ('R', 'L') and dirn != self.pattern) or (self.pattern == 'alt' and (b % 2 == 0) != (dirn == 'R')):
            return self._plain(U, s, Vh, k, l, r, dirn)
        t1 = pc()
        lo, hi = max(0, b - self.aL), min(self.N, b + 2 + self.aR)
        D, (Fs, Fd64), iu, dg = self._region(hi - lo, hi == self.N, b - lo)
        cdt, fdt = self.cdt, self.fdt
        Fd32 = self._f32.get(id(Fd64))
        if Fd32 is None:
            Fd32 = self._f32[id(Fd64)] = np.ascontiguousarray(Fd64.astype(fdt))
        Lm, Rm = self._maps(lo, hi, b, l, r)                # (o', X, l) and (r, Z, p')
        nb = len(s) - k
        sk, sq = s[:k], s[k:]
        fw, nd, nh = self.fw, len(dg), D * D
        nu = len(iu[0])
        ns = Fs.shape[0]
        no, nX, nZ, npp = Lm.shape[0], Lm.shape[1], Rm.shape[1], Rm.shape[2]
        LmM, RmM = Lm.reshape(no * nX, l), Rm.reshape(r, nZ * npp)

        def Wof(th, LmM=LmM, RmM=RmM):                    # theta (l,d,d,r) -> W (D, o' a1 a2 p'), rows = physical qubits of the region
            Y = LmM @ th.reshape(l, d * d * r)
            Y = Y.reshape(no * nX * d * d, r) @ RmM
            Y = Y.reshape(no, nX, dp, da, dp, da, nZ, npp).transpose(1, 2, 4, 6, 0, 3, 5, 7)
            return Y.reshape(D, no * da * da * npp)

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
        W0 = Wof(Mk0.reshape(l, d, d, r))
        rho0 = W0 @ W0.conj().T
        tr0 = rho0[dg, dg].real.sum()
        hn0 = hvec(rho0) / tr0
        ht = hnorm(theta)
        r0 = Fapply((hn0 - ht)[:, None])[:, 0]
        f_svd = float(r0 @ r0)
        kept_svd = float(np.sum(sk ** 2))
        self.cand.append((float(np.sum(sq ** 2) / nrm2), f_svd))
        if self.gate > 0 or self.gate_log is not None:
            Bres = self._low_span_resid(r0, hi - lo)
            cost = self.gate_c0 * (tail / 1e-4) ** self.gate_exp
            if self.gate_log is not None:
                self.gate_log.append((tail, Bres, cost, b))
            if self.gate > 0 and Bres < self.gate * cost:
                self.skipped += 1
                self.tm['setup'] += pc() - t1
                return self._plain(U, s, Vh, k, l, r, dirn)
        if f_svd < self.f_min:
            self.skipped += 1
            self.tm['setup'] += pc() - t1
            return self._plain(U, s, Vh, k, l, r, dirn)
        phi_svd = f_svd + fw * (1.0 - kept_svd / nrm2)
        ones_d = np.zeros(nh, dtype=fdt)
        ones_d[:nd] = 1.0
        c64 = cdt
        LmMc, RmMc = LmM.astype(c64), RmM.astype(c64)
        LmMcc, RmMcc = LmMc.conj(), RmMc.conj()
        nsr = nrm2
        if dirn == 'R':
            Bk, Bp = U[:, :k], U[:, k:]
        else:
            Bk, Bp = Vh[:k].conj().T, Vh[k:].conj().T

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
            W0_ = Wof(Mk_.reshape(l, d, d, r))
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
            nd_ = ((k * d * r) if dirn == 'R' else (k * d * l)) if (use_free and svd_chart) else 0
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
                    dM = dM + (Bkc @ Dm.reshape(k, d * r) if dirn == 'R' else Dm.reshape(d * l, k) @ BkH)
                dW = Wof(dM.reshape(l, d, d, r), LmMc, RmMc)
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
                # W columns are (o', a1, a2, p'); put the rows back to (X, p1, a1, p2, a2, o') x (Z, p')
                Gd = (2 * (H @ W0c)).reshape(nX, dp, dp, nZ, no, da, da, npp)
                G = Gd.transpose(0, 1, 5, 2, 6, 4, 3, 7).reshape(nX * d * d * no, nZ * npp) @ RmMcc.T
                G = G.reshape(nX, d * d, no, r).transpose(1, 3, 2, 0).reshape(d * d * r, no * nX) @ LmMcc
                Gth = G.reshape(d, d, r, l).transpose(3, 0, 1, 2).reshape(l * d, d * r)
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
                    X = Q.conj().T @ Mm + Dm.reshape(k, d * r)
                    return Q, Q @ X, X
                X = Mm @ Q + Dm.reshape(d * l, k)
                return Q, X @ Q.conj().T, X
            Mk = Q @ (Q.conj().T @ Mm) if dirn == 'R' else (Mm @ Q) @ Q.conj().T
            return Q, Mk, None

        use_free = self.free > 0 and self.passes == 1
        Bk_c, Bp_c, Mk_c, f_c, kept_c, Q_c, X_c = Bk, Bp, Mk0, f_svd, kept_svd, None, None
        phi_c = phi_svd
        t2 = pc()
        self.tm['setup'] += t2 - t1
        for ps in range(self.passes):
            t2 = pc()
            nb_ = Bp_c.shape[1]
            jvp, vjp, scale = make_chart(Bk_c, Bp_c, Mk_c, ps == 0)
            r_c = Fapply((hnorm(Mk_c.reshape(l, d, d, r)) - ht)[:, None])[:, 0] if ps else r0
            if ps == 0 and fw > 0:
                wji = (sk[None, :] ** 2 - sq[:, None] ** 2).clip(min=0.0)
                ridge = np.tile(wji.ravel(), 2) * (fw / nrm2)
            else:
                ridge = np.zeros(len(scale))
            Minv = 1.0 / (3.0 * scale + ridge + 1e-9 * float(scale.max()))
            b_ = -vjp(r_c)
            if self.debug is not None and ps == 0:
                def exact(x):
                    Q, Mk, _X = retract(Bk, Bp, x, 1.0)
                    return None, None, Fapply((hnorm(Mk.reshape(l, d, d, r)) - ht)[:, None])[:, 0]
                self.debug.append(dict(jvp=jvp, vjp=vjp, exact=exact, nx=len(scale), nres=len(r0), r0=r0, f_svd=f_svd, Wof=Wof,
                                       l=l, r=r, lo=lo, hi=hi, b=b, theta=theta.copy(), dirn=dirn, Tsnap=[t.copy() for t in self.T],
                                       Lm=Lm, Rm=Rm, D=D))

            def Aop(p_, jvp=jvp, vjp=vjp, ridge=ridge):
                self.nmv += 1
                return vjp(jvp(p_)) + ridge * p_

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
                rj = Fapply((hnorm(Mk.reshape(l, d, d, r)) - ht)[:, None])[:, 0]
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
            return self._plain(U, s, Vh, k, l, r, dirn)
        self.fired += 1
        self.log.append((f_svd, f_c, (nrm2 - kept_svd) / nrm2, (nrm2 - kept_c) / nrm2, b, 1.0 if dirn == 'R' else 0.0, tail))
        Q = Q_c
        if isinstance(Q, tuple):
            QL, QR = Q
            if dirn == 'R':
                cen = X_c @ QR.conj().T
                n = np.linalg.norm(cen)
                return QL.reshape(l, d, k), (cen / n).reshape(k, d, r), 0.0, True
            cen = QL @ X_c
            n = np.linalg.norm(cen)
            return (cen / n).reshape(l, d, k), QR.conj().T.reshape(k, d, r), 0.0, True
        if dirn == 'R':
            cen = X_c if X_c is not None else Q.conj().T @ Mm
            n = np.linalg.norm(cen)
            return Q.reshape(l, d, k), (cen / n).reshape(k, d, r), 0.0, True
        cen = X_c if X_c is not None else Mm @ Q
        n = np.linalg.norm(cen)
        return (cen / n).reshape(l, d, k), Q.conj().T.reshape(k, d, r), 0.0, True
