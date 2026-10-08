"""Chunked purification spcf cut: the region state W is never materialised.

``SPCFPur`` (spcfpur.py) builds W = Wof(theta), a (D, no*da*da*npp) matrix (D = 2^L rows, the physical qubits of the region; the columns are
(outer left bond with the left-window ancillas) x (ancillas of the two cut sites) x (outer right bond with the right-window ancillas)), and
works with rho = W W^dag, dW W0^dag (jvp) and H W0 pushed back through the environments (vjp).  At a = 2 W has about 4096 chi^2 entries.

Everything that touches W is a sum over the column index, and the first factor of the column index is the outer index o' of the left
environment map Lm (no, nX, l).  A block of o' values is a contiguous block of rows of Lm.reshape(no*nX, l), so

    W[:, block]            = Wblk(theta, o0, o1)                 (D, nb*da*da*npp), built from the rows o0*nX:o1*nX of Lm only,
    rho                    = sum_blocks W_b W_b^dag
    X1 = dW W0^dag         = sum_blocks dW_b W0_b^dag             (jvp; W0_b is recomputed from the rank-k matrix, not stored)
    G_theta (vjp)          = sum_blocks (rows o0:o1 of the Lm contraction of the (2 H W0_b) pushed through Rm)

so the peak transient is O(D * nb * da^2 * npp) with nb = ``blk`` instead of O(D * no * da^2 * npp), and the result is the same sum in a
different order.  The state independent residual table Fd is kept in the working precision only (``lean_tables``), a third of what SPCFast holds.
The price is that W0_b is recomputed in every jvp and vjp (about 9 extra Wof-sized contractions per cut at iters = 4); the
option ``w0_fast`` recomputes it directly in the working precision (complex64) instead of in complex128 followed by the cast that the
unchunked cut uses.

Everything else (SVD, tangent charts, retraction, line search, CG, the logic of when a cut fires) is a line-for-line copy of
SPCFPur.__call__, so with precision='f64' the output agrees with the unchunked cut to rounding.
"""
import time
import numpy as np
import mpsenh as M
from spcfpur import SPCFPur


class SPCFPurChunk(SPCFPur):
    def __init__(self, *args, blk=8, w0_fast=False, lean_tables=True, **kw):
        super().__init__(*args, **kw)
        assert blk >= 1
        self.blk = int(blk)
        self.w0_fast = w0_fast
        self.lean_tables = lean_tables

    def _region(self, L, at_end, pos=0):
        """SPCFPur._region, then keep only the working-precision copy of the dense residual table Fd (the cut only ever applies the float32 copy;
        SPCFast keeps the float64 original as well, about 3x the memory).  No numerical change: the cut used Fd.astype(fdt) already."""
        key = (L, at_end, pos if self.wst != 1.0 else 0)
        first = key not in self._shape
        out = super()._region(L, at_end, pos)
        if self.lean_tables and first:
            Dm, (Fs, Fd), iu, dg = out
            if Fd.dtype != np.dtype(self.fdt):
                Fl = np.ascontiguousarray(Fd.astype(self.fdt))
                out = (Dm, (Fs, Fl), iu, dg)
                self._shape[key] = out
                self._f32[id(Fl)] = Fl
        return out

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
        blk = self.blk
        blocks = [(o0, min(no, o0 + blk)) for o0 in range(0, no, blk)]
        ddr = d * d * r

        def Wblk(thm, LmM_, RmM_, o0, o1):               # thm (l, d*d*r) -> W[:, o0:o1] (D, (o1-o0) da da npp), rows = physical qubits of the region
            nbk = o1 - o0
            Y = LmM_[o0 * nX:o1 * nX] @ thm
            Y = Y.reshape(nbk * nX * d * d, r) @ RmM_
            Y = Y.reshape(nbk, nX, dp, da, dp, da, nZ, npp).transpose(1, 2, 4, 6, 0, 3, 5, 7)
            return Y.reshape(D, nbk * da * da * npp)

        def Wof(th, LmM_=LmM, RmM_=RmM):                  # the full W, only for the debug hook (tests); the cut itself never calls it
            thm = th.reshape(l, ddr)
            return np.concatenate([Wblk(thm, LmM_, RmM_, o0, o1) for o0, o1 in blocks], axis=1)

        def rho_of(th):                                   # rho = W W^dag accumulated over blocks, same dtype as th @ Lm
            thm = th.reshape(l, ddr)
            rho = None
            for o0, o1 in blocks:
                W = Wblk(thm, LmM, RmM, o0, o1)
                rb = W @ W.conj().T
                rho = rb if rho is None else rho + rb
            return rho

        def hvec(rho):
            return np.concatenate([rho[dg, dg].real, rho[iu].real, rho[iu].imag])

        def hnorm(th):
            rho = rho_of(th)
            return hvec(rho) / rho[dg, dg].real.sum()

        def Fapply(H):                                   # residual rows for the columns of H (nh x m), F applied in float32
            Rd = (Fd32 @ H.astype(fdt)).astype(float)
            return np.concatenate([Fs @ H, Rd], axis=0) if ns else Rd

        Mk0 = (U[:, :k] * sk) @ Vh[:k]
        rho0 = rho_of(Mk0.reshape(l, d, d, r))
        tr0 = rho0[dg, dg].real.sum()
        hn0 = hvec(rho0) / tr0
        del rho0
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
            rho_ = rho_of(Mk_.reshape(l, d, d, r))
            tr_ = rho_[dg, dg].real.sum()
            hn_ = hvec(rho_) / tr_
            del rho_
            Mk_m = Mk_.reshape(l, ddr)                    # W0 is rebuilt block by block from the rank-k matrix, never stored
            Mk_mc = Mk_m.astype(c64) if self.w0_fast else None

            def W0c_blk(o0, o1):
                if Mk_mc is not None:
                    return Wblk(Mk_mc, LmMc, RmMc, o0, o1)
                return Wblk(Mk_m, LmM, RmM, o0, o1).astype(c64)

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
                dMm = dM.reshape(l, ddr)
                X1 = None
                for o0, o1 in blocks:
                    dW = Wblk(dMm, LmMc, RmMc, o0, o1)
                    xb = dW @ W0c_blk(o0, o1).conj().T
                    X1 = xb if X1 is None else X1 + xb
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
                # per block of o': (2 H W0_b) with rows put back to (X, p1, a1, p2, a2, o') x (Z, p'), pushed through Rm, then through the o' rows of Lm
                Gacc = None
                for o0, o1 in blocks:
                    nbk = o1 - o0
                    Gd = (2 * (H @ W0c_blk(o0, o1))).reshape(nX, dp, dp, nZ, nbk, da, da, npp)
                    G = Gd.transpose(0, 1, 5, 2, 6, 4, 3, 7).reshape(nX * d * d * nbk, nZ * npp) @ RmMcc.T
                    del Gd
                    G = G.reshape(nX, d * d, nbk, r).transpose(1, 3, 2, 0).reshape(d * d * r, nbk * nX) @ LmMcc[o0 * nX:o1 * nX]
                    Gacc = G if Gacc is None else Gacc + G
                Gth = Gacc.reshape(d, d, r, l).transpose(3, 0, 1, 2).reshape(l * d, d * r)
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
                                       Lm=Lm, Rm=Rm, D=D, rho_of=rho_of, hnorm=hnorm, ht=ht, hn0=hn0, nblocks=len(blocks)))

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
