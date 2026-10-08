"""spcf bond-tilt cut for an LPDO (rank 6, Test 2 arm).  Read-only derivative of rank2/spcfpur.py (same Gauss-Newton step, same CG, same line search),
generalised to per-site Kraus dimensions K1, K2 of the two cut sites and to the Gram-compressed environments of gatelog.py.

theta is (l, 2, K1, 2, K2, r); the SVD / tangent space live on Mm = theta.reshape(l*2*K1, 2*K2*r).  The objective is the physical-RDM objective of
SPCFast (rows of F act on the 2^L physical indices of the region).  The region matrix W has the physical indices on its rows and
(left-environment column index c_L, Kraus legs of the two cut sites, right-environment column index c_R) on its columns, rho_phys = W W^dag.
The environments enter through c_L, c_R of size <= (2^nL * l), (2^nR * r) (exact compression of the Gram matrices, see gatelog.py).

Supported options (all others of SPCFast are rejected): a, taus, ks, fw, iters, cg_tol, alphas, eps_min, rel_skip, gate (+ gate_c0/exp), gate_log.
Call signature for lpdo.run_lpdo(cut=...):  cut(theta, chi, dirn, T, b) -> (A, B) or None (None = plain SVD cut).
"""
import _p6  # noqa: F401
import time
import numpy as np
import mpsenh as M
import gatelog as G
from spcfast import SPCFast


def _factor(Gm, tol=1e-13):
    """Gm[(x),(y)] Hermitian PSD (as 4-index [X,l,Y,m]) -> Lm'[c, X, l] with sum_c Lm'[c,x] conj(Lm'[c,y]) = Gm[x,y]."""
    X, l, Y, m = Gm.shape
    Gf = Gm.reshape(X * l, Y * m)
    Gf = 0.5 * (Gf + Gf.conj().T)
    lam, V = np.linalg.eigh(Gf)
    keep = lam > tol * max(lam[-1], 1e-300)
    Lp = (V[:, keep] * np.sqrt(lam[keep])).T                  # [c, (X,l)]
    return Lp.reshape(-1, X, l)


class SPCFLpdo(SPCFast):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        assert self.beta == 0.0 and not self.two and self.free == 0.0 and self.passes == 1
        self.last = None

    def start(self, T, N=None):
        self.T = T

    @staticmethod
    def _plain(U, s, Vh, k, l, r, dirn, K1, K2):
        s = s[:k]
        nrm = np.linalg.norm(s)
        U, Vh = U[:, :k], Vh[:k]
        if dirn == 'R':
            return U.reshape(l, 2, K1, k), ((s[:, None] / nrm) * Vh).reshape(k, 2, K2, r)
        return (U * (s / nrm)).reshape(l, 2, K1, k), Vh.reshape(k, 2, K2, r)

    def _maps_l(self, T, lo, b, l, r, hi):
        GL = G.left_gram(T, lo, b, l)
        GR = G.right_gram(T, b, hi, r)                         # [Z, r, V, s]
        Lm = _factor(GL)                                        # [c, X, l]
        Rf = _factor(GR.transpose(0, 1, 2, 3))                  # [c, Z, r]
        Rm = Rf.transpose(2, 1, 0)                              # [r, Z, c]
        return Lm, Rm

    def __call__(self, theta, chi, dirn, T, b):
        self.calls += 1
        pc = time.perf_counter
        N = self.N
        l, K1, K2, r = theta.shape[0], theta.shape[2], theta.shape[4], theta.shape[5]
        d1, d2 = 2 * K1, 2 * K2
        Mm = theta.reshape(l * d1, d2 * r)
        U, s, Vh = np.linalg.svd(Mm, full_matrices=False)
        k = M._rank(s, chi)
        nrm2 = float(np.sum(s ** 2))
        if k >= len(s) or np.sum(s[k:] ** 2) <= self.eps_min * nrm2:
            return self._plain(U, s, Vh, k, l, r, dirn, K1, K2)
        tail = float(np.sum(s[k:] ** 2) / nrm2)
        self.tmax = max(self.tmax, tail)
        if tail < self.rel_skip * self.tmax:
            self.skipped += 1
            return self._plain(U, s, Vh, k, l, r, dirn, K1, K2)
        if b < self.bmin or b > self.bmax:
            return self._plain(U, s, Vh, k, l, r, dirn, K1, K2)
        t1 = pc()
        lo, hi = max(0, b - self.aL), min(N, b + 2 + self.aR)
        D, (Fs, Fd64), iu, dg = self._region(hi - lo, hi == N, b - lo)
        cdt, fdt = self.cdt, self.fdt
        Fd32 = self._f32.get(id(Fd64))
        if Fd32 is None:
            Fd32 = self._f32[id(Fd64)] = np.ascontiguousarray(Fd64.astype(fdt))
        Lm, Rm = self._maps_l(T, lo, b, l, r, hi)           # (cL, X, l) and (r, Z, cR)
        nb = len(s) - k
        sk, sq = s[:k], s[k:]
        fw, nd, nh = self.fw, len(dg), D * D
        nu = len(iu[0])
        ns = Fs.shape[0]
        no, nX, nZ, npp = Lm.shape[0], Lm.shape[1], Rm.shape[1], Rm.shape[2]
        LmM, RmM = Lm.reshape(no * nX, l), Rm.reshape(r, nZ * npp)
        ncol = no * K1 * K2 * npp

        def Wof(th, LmM=LmM, RmM=RmM):
            Y = LmM @ th.reshape(l, 2 * K1 * 2 * K2 * r)
            Y = Y.reshape(no * nX * 2 * K1 * 2 * K2, r) @ RmM
            Y = Y.reshape(no, nX, 2, K1, 2, K2, nZ, npp).transpose(1, 2, 4, 6, 0, 3, 5, 7)
            return Y.reshape(D, ncol)

        def hvec(rho):
            return np.concatenate([rho[dg, dg].real, rho[iu].real, rho[iu].imag])

        def hnorm(th):
            W = Wof(th)
            rho = W @ W.conj().T
            return hvec(rho) / rho[dg, dg].real.sum()

        def Fapply(H):
            Rd = (Fd32 @ H.astype(fdt)).astype(float)
            return np.concatenate([Fs @ H, Rd], axis=0) if ns else Rd

        Mk0 = (U[:, :k] * sk) @ Vh[:k]
        W0 = Wof(Mk0.reshape(l, 2, K1, 2, K2, r))
        rho0 = W0 @ W0.conj().T
        hn0 = hvec(rho0) / rho0[dg, dg].real.sum()
        ht = hnorm(theta)
        r0 = Fapply((hn0 - ht)[:, None])[:, 0]
        f_svd = float(r0 @ r0)
        kept_svd = float(np.sum(sk ** 2))
        if self.gate > 0 or self.gate_log is not None:
            Bres = self._low_span_resid(r0, hi - lo)
            cost = self.gate_c0 * (tail / 1e-4) ** self.gate_exp
            if self.gate_log is not None:
                self.gate_log.append((tail, Bres, cost, b))
            if self.gate > 0 and Bres < self.gate * cost:
                self.skipped += 1
                return self._plain(U, s, Vh, k, l, r, dirn, K1, K2)
        if f_svd < self.f_min:
            self.skipped += 1
            return self._plain(U, s, Vh, k, l, r, dirn, K1, K2)
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

        def make_chart(Bk_, Bp_, Mk_):
            nb_ = Bp_.shape[1]
            if dirn == 'R':
                Ak, Ap = Bk_.conj().T @ Mm, Bp_.conj().T @ Mm
                P1, Q1, P2, Q2 = Bp_, Ak, Bk_, Ap
                cs = np.sum(np.abs(Ak) ** 2, axis=1)
            else:
                MB = Mm @ Bk_
                P1, Q1, P2, Q2 = Mm @ Bp_, Bk_.conj().T, MB, Bp_.conj().T
                cs = np.sum(np.abs(MB) ** 2, axis=0)
            W0_ = Wof(Mk_.reshape(l, 2, K1, 2, K2, r))
            rho_ = W0_ @ W0_.conj().T
            tr_ = rho_[dg, dg].real.sum()
            hn_ = hvec(rho_) / tr_
            W0c = W0_.astype(c64)
            W0cH = W0c.conj().T
            hn_32, tr_32 = hn_.astype(fdt), fdt(tr_)
            P1c, Q1c, P2c, Q2c = P1.astype(c64), Q1.astype(c64), P2.astype(c64), Q2.astype(c64)
            P1H = P1c.conj().T
            Q1H = Q1c.conj().T
            nc = nb_ * k

            def jvp(x):
                z = (x[:nc] + 1j * x[nc:]).astype(c64)
                C = z.reshape(nb_, k)
                dM = P1c @ C @ Q1c + P2c @ C.conj().T @ Q2c
                dW = Wof(dM.reshape(l, 2, K1, 2, K2, r), LmMc, RmMc)
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
                Gd = (2 * (H @ W0c)).reshape(nX, 2, 2, nZ, no, K1, K2, npp)        # rows (X,a,b,Z), cols (o,i,j,p)
                Gx = np.tensordot(Gd, RmMcc.reshape(r, nZ, npp), axes=([3, 7], [1, 2]))   # X a b o i j r
                Gx = np.tensordot(Gx, LmMcc.reshape(no, nX, l), axes=([0, 3], [1, 0]))    # a b i j r l
                Gth = Gx.transpose(5, 0, 2, 1, 3, 4).reshape(l * d1, d2 * r)              # l a i b j r
                Gc = P1H @ Gth @ Q1H + Q2c @ Gth.conj().T @ P2c
                gz = Gc.ravel()
                return np.concatenate([gz.real, gz.imag]).astype(float)

            sc = np.broadcast_to(cs[None, :], (nb_, k)).ravel()
            return jvp, vjp, np.tile(sc, 2) / nsr

        def retract(Bk_, Bp_, x_, al):
            nb_ = Bp_.shape[1]
            nc = nb_ * k
            nt = len(x_) // 2
            z = al * (x_[:nt] + 1j * x_[nt:])
            C = z[:nc].reshape(nb_, k)
            Q, _ = np.linalg.qr(Bk_ + Bp_ @ C)
            Mk = Q @ (Q.conj().T @ Mm) if dirn == 'R' else (Mm @ Q) @ Q.conj().T
            return Q, Mk

        jvp, vjp, scale = make_chart(Bk, Bp, Mk0)
        if fw > 0:
            wji = (sk[None, :] ** 2 - sq[:, None] ** 2).clip(min=0.0)
            ridge = np.tile(wji.ravel(), 2) * (fw / nrm2)
        else:
            ridge = np.zeros(len(scale))
        Minv = 1.0 / (3.0 * scale + ridge + 1e-9 * float(scale.max()))
        b_ = -vjp(r0)
        if self.debug is not None:
            def exact(x):
                Q, Mk = retract(Bk, Bp, x, 1.0)
                return Fapply((hnorm(Mk.reshape(l, 2, K1, 2, K2, r)) - ht)[:, None])[:, 0]
            self.debug.append(dict(jvp=jvp, vjp=vjp, exact=exact, nx=len(scale), nres=len(r0), r0=r0, f_svd=f_svd, Wof=Wof, l=l, r=r, K1=K1, K2=K2,
                                   b=b, lo=lo, hi=hi, theta=theta.copy(), D=D, Lm=Lm, Rm=Rm))

        def Aop(p_):
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
        ok = False
        for jj, al in enumerate(self.alphas):
            Q, Mk = retract(Bk, Bp, x, al)
            kj = float(np.vdot(Mk, Mk).real)
            rj = Fapply((hnorm(Mk.reshape(l, 2, K1, 2, K2, r)) - ht)[:, None])[:, 0]
            fj = float(rj @ rj)
            if fj + fw * (1.0 - kj / nrm2) < phi_svd * (1 - 1e-3):
                ok = True
                break
        if not ok:
            self.rej = getattr(self, 'rej', {})
            self.rej[b] = self.rej.get(b, 0) + 1
            return self._plain(U, s, Vh, k, l, r, dirn, K1, K2)
        self.fired += 1
        self.achosen[jj] = self.achosen.get(jj, 0) + 1
        self.log.append((f_svd, fj, (nrm2 - kept_svd) / nrm2, (nrm2 - kj) / nrm2, b, 1.0 if dirn == 'R' else 0.0, tail))
        if dirn == 'R':
            cen = Q.conj().T @ Mm
            n = np.linalg.norm(cen)
            return Q.reshape(l, 2, K1, k), (cen / n).reshape(k, 2, K2, r)
        cen = Mm @ Q
        n = np.linalg.norm(cen)
        return (cen / n).reshape(l, 2, K1, k), Q.conj().T.reshape(k, 2, K2, r)
