"""Cut-local tilt for a Pauli-fused operator MPS (d = 4) with DIAGONAL targets.

Port of ``rule/spcfast.py`` (one Gauss-Newton step in the Grassmann chart Q = qr(U_k + U_perp C), solved matrix-free by diagonally
preconditioned CG on J^T J, accepted by a line search only if the exact residual drops) with the Hermitian region density matrix
replaced by the label marginal of the region.

For the region of L sites around the cut (a sites each side of the two-site tensor theta) let W = Lm . theta . Rm be the (4^L, o*p)
matrix with the outer bonds open.  The environments are isometric, so the Pauli-label distribution of the region is

        h(x) = sum_c W[x, c]^2 / sum W^2                (x = labels of the L sites, 4^L entries)

and every target is a fixed 0/1 summation of h: for each window of k = 1..3 consecutive sites and each of its 4^k label patterns alpha,
p_win(alpha) = sum of h over the x that carry alpha on the window.  Residual rows r = Msp (h(truncated) - h(theta)), scaled by
sqrt(w_k / n_windows(k)); 360 rows for L = 6.  No 4^L x 4^L matrix is ever formed:  dh = 2 rowsum(W0 * dW),  vjp:  dW^T = 2 eta[:,None] * W0.

Classes
  OpCut    SVD cut (the 'svd' arm) with optional diagnostics: logs, for every truncating cut, the residual r0 of the SVD point
           (rms over all rows, per k, and over the span-2 rows) in probability units.
  SPCOp    the tilt.  tilt=False reproduces OpCut bit for bit through the same code path.
All real float64.
"""
import time
import numpy as np
import scipy.sparse as sps
import op_tebd as O
import mpsenh as M


def _plain(U, s, Vh, k, l, r, dirn):
    s = s[:k]
    nrm = np.linalg.norm(s)
    U, Vh = U[:, :k], Vh[:k]
    if dirn == 'R':
        return U.reshape(l, 4, k), ((s[:, None] / nrm) * Vh).reshape(k, 4, r), 0.0, False
    return (U * (s / nrm)).reshape(l, 4, k), Vh.reshape(k, 4, r), 0.0, False


class OpCut:
    def __init__(self, N, aL=2, aR=2, ks=(1, 2, 3), wk=None, diag=False, eps_min=1e-7, rel_skip=0.0, dtype=np.float64):
        self.N, self.aL, self.aR = N, aL, aR
        self.ks = tuple(ks)
        self.wk = {k: 1.0 for k in range(1, 7)} if wk is None else dict(zip(self.ks, wk))
        self.diag, self.eps_min, self.rel_skip = diag, eps_min, rel_skip
        self.T = None
        self.calls = 0
        self.trunc = 0
        self.tmax = 0.0
        self.logkept = 0.0                    # sum of log(kept weight / total weight): the norm^2 an UNrenormalised cut would carry
        self.lognorm_series = []              # (call index, logkept)
        self.log = []
        self.tm = dict(svd=0.0, setup=0.0, solve=0.0, line=0.0)
        self._mm = {}
        self.debug = None

    def start(self, T):
        self.T = T

    # ---------------------------------------------------------------- region tables
    def _Mmap(self, L):
        if L in self._mm:
            return self._mm[L]
        D = 4 ** L
        x = np.arange(D)
        rows, cols, vals = [], [], []
        span, kk, wts = [], [], []
        ridx = 0
        for k in self.ks:
            if k > L:
                continue
            nwin = L - k + 1
            for off in range(nwin):
                wl = (x // 4 ** (L - off - k)) % 4 ** k
                w = np.sqrt(self.wk.get(k, 1.0) / nwin)
                rows.append(ridx + wl)
                cols.append(x)
                vals.append(np.full(D, w))
                for a in range(4 ** k):
                    digs = [(a // 4 ** (k - 1 - j)) % 4 for j in range(k)]
                    nz = [j for j, d in enumerate(digs) if d]
                    span.append((nz[-1] - nz[0] + 1) if nz else 0)
                    kk.append(k)
                    wts.append(w)
                ridx += 4 ** k
        Msp = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ridx, D))
        out = (Msp, np.array(span), np.array(kk), np.array(wts))
        self._mm[L] = out
        return out

    def _env(self, lo, hi, b, l, r):
        T = self.T
        if lo == b:
            Lm = np.eye(l).reshape(l, 1, l)
        else:
            X = T[lo]
            for j in range(lo + 1, b):
                X = np.tensordot(X, T[j], axes=([X.ndim - 1], [0]))
                X = X.reshape(X.shape[0], -1, X.shape[-1])
            Lm = X.reshape(X.shape[0], -1, X.shape[-1])
        if hi == b + 2:
            Rm = np.eye(r).reshape(r, 1, r)
        else:
            X = T[b + 2]
            for j in range(b + 3, hi):
                X = np.tensordot(X, T[j], axes=([X.ndim - 1], [0]))
                X = X.reshape(X.shape[0], -1, X.shape[-1])
            Rm = X.reshape(X.shape[0], -1, X.shape[-1])
        return Lm, Rm

    def _region(self, b, l, r):
        """Closures of the theta -> W map and its adjoint for the region around bond b."""
        lo, hi = max(0, b - self.aL), min(self.N, b + 2 + self.aR)
        L = hi - lo
        Lm, Rm = self._env(lo, hi, b, l, r)
        no, nX, nZ, npp = Lm.shape[0], Lm.shape[1], Rm.shape[1], Rm.shape[2]
        D = nX * 16 * nZ
        assert D == 4 ** L
        LmM, RmM = Lm.reshape(no * nX, l), Rm.reshape(r, nZ * npp)

        def Wof(th):
            Y = LmM @ th.reshape(l, 16 * r)
            Y = Y.reshape(no * nX * 16, r) @ RmM
            return Y.reshape(no, nX, 16, nZ, npp).transpose(1, 2, 3, 0, 4).reshape(D, no * npp)

        def Wadj(G):
            G5 = G.reshape(nX, 16, nZ, no, npp)
            Gt = G5.transpose(0, 1, 3, 2, 4).reshape(nX * 16 * no, nZ * npp) @ RmM.T
            Gt = Gt.reshape(nX, 16, no, r).transpose(1, 3, 2, 0).reshape(16 * r, no * nX) @ LmM
            return Gt.reshape(4, 4, r, l).transpose(3, 0, 1, 2)

        return dict(lo=lo, hi=hi, L=L, D=D, Wof=Wof, Wadj=Wadj, Lm=Lm, Rm=Rm)

    @staticmethod
    def hn_of(Wof, th):
        W = Wof(th)
        h = np.einsum('ij,ij->i', W, W)
        return h / h.sum()

    def _rowstats(self, r0, L):
        Msp, span, kk, wts = self._Mmap(L)
        u = r0 / wts
        out = [float(np.sqrt(np.mean(u ** 2)))]
        for k in (1, 2, 3):
            m = kk == k
            out.append(float(np.sqrt(np.mean(u[m] ** 2))) if m.any() else 0.0)
        m = span == 2
        out.append(float(np.sqrt(np.mean(u[m] ** 2))) if m.any() else 0.0)
        return out

    # ---------------------------------------------------------------- the cut
    def __call__(self, theta, chi, dirn, A, B, b):
        self.calls += 1
        pc = time.perf_counter
        t0 = pc()
        l, r = theta.shape[0], theta.shape[3]
        Mm = theta.reshape(l * 4, 4 * r)
        U, s, Vh = np.linalg.svd(Mm, full_matrices=False)
        k = M._rank(s, chi)
        self.tm['svd'] += pc() - t0
        nrm2 = float(np.sum(s ** 2))
        tail = float(np.sum(s[k:] ** 2) / nrm2)
        self.logkept += float(np.log1p(-tail)) if tail < 1.0 else -np.inf
        if k >= len(s) or np.sum(s[k:] ** 2) <= self.eps_min * nrm2:
            return _plain(U, s, Vh, k, l, r, dirn)
        self.trunc += 1
        self.tmax = max(self.tmax, tail)
        return self._truncate(theta, Mm, U, s, Vh, k, l, r, dirn, tail, nrm2, b)

    def _truncate(self, theta, Mm, U, s, Vh, k, l, r, dirn, tail, nrm2, b):
        if self.diag:
            t1 = time.perf_counter()
            reg = self._region(b, l, r)
            Wof = reg['Wof']
            Mk0 = (U[:, :k] * s[:k]) @ Vh[:k]
            r0 = self._Mmap(reg['L'])[0] @ (self.hn_of(Wof, Mk0.reshape(l, 4, 4, r)) - self.hn_of(Wof, theta))
            st = self._rowstats(r0, reg['L'])
            self.log.append((self.calls, b, 1.0 if dirn == 'R' else 0.0, tail, *st, 0.0, float(r0 @ r0), float(r0 @ r0), tail))
            self.tm['setup'] += time.perf_counter() - t1
        return _plain(U, s, Vh, k, l, r, dirn)


class SPCOp(OpCut):
    def __init__(self, N, aL=2, aR=2, ks=(1, 2, 3), wk=None, fw=0.0, iters=4, cg_tol=1e-3, eps_min=1e-7, rel_skip=1e-2,
                 alphas=(1.0, 0.5, 0.25), passes=1, tilt=True, gate=0.0, gate_c0=1.2e-4, gate_exp=0.65, log_gate=False):
        super().__init__(N, aL=aL, aR=aR, ks=ks, wk=wk, diag=False, eps_min=eps_min, rel_skip=rel_skip)
        self.fw, self.iters, self.cg_tol, self.alphas, self.passes = fw, iters, cg_tol, tuple(alphas), passes
        self.tilt = tilt
        self.gate, self.gate_c0, self.gate_exp = gate, gate_c0, gate_exp
        self.fired = self.skipped = self.rejected = 0
        self.nmv = 0
        self.achosen = {}

    def _truncate(self, theta, Mm, U, s, Vh, k, l, r, dirn, tail, nrm2, b):
        pc = time.perf_counter
        if self.tail_skip(tail):
            self.skipped += 1
            return _plain(U, s, Vh, k, l, r, dirn)
        t1 = pc()
        reg = self._region(b, l, r)
        Wof, Wadj, L, D = reg['Wof'], reg['Wadj'], reg['L'], reg['D']
        Msp, span, kk, wts = self._Mmap(L)
        nres = Msp.shape[0]
        sk, sq = s[:k], s[k:]
        nb = len(s) - k
        if dirn == 'R':
            Bk, Bp = U[:, :k], U[:, k:]
        else:
            Bk, Bp = Vh[:k].T, Vh[k:].T
        fw = self.fw
        Mk0 = (U[:, :k] * sk) @ Vh[:k]
        ht = self.hn_of(Wof, theta)
        h0n = self.hn_of(Wof, Mk0.reshape(l, 4, 4, r))
        r0 = Msp @ (h0n - ht)
        f_svd = float(r0 @ r0)
        kept_svd = float(np.sum(sk ** 2))
        st = self._rowstats(r0, L)
        if self.gate > 0:
            cost = self.gate_c0 * (tail / 1e-4) ** self.gate_exp
            if st[4] < self.gate * cost:                      # rms span-2 residual (probability units) below the benefit/cost threshold
                self.skipped += 1
                self.tm['setup'] += pc() - t1
                self.log.append((self.calls, b, 1.0 if dirn == 'R' else 0.0, tail, *st, 0.0, f_svd, f_svd, tail))
                return _plain(U, s, Vh, k, l, r, dirn)
        if not self.tilt:
            self.tm['setup'] += pc() - t1
            self.log.append((self.calls, b, 1.0 if dirn == 'R' else 0.0, tail, *st, 0.0, f_svd, f_svd, tail))
            return _plain(U, s, Vh, k, l, r, dirn)
        phi_svd = f_svd + fw * (1.0 - kept_svd / nrm2)

        def make_chart(Bk_, Bp_, Mk_):
            nb_ = Bp_.shape[1]
            if dirn == 'R':
                Ak, Ap = Bk_.T @ Mm, Bp_.T @ Mm
                P1, Q1, P2, Q2 = Bp_, Ak, Bk_, Ap
                cs = np.sum(Ak ** 2, axis=1)
            else:
                MB = Mm @ Bk_
                P1, Q1, P2, Q2 = Mm @ Bp_, Bk_.T, MB, Bp_.T
                cs = np.sum(MB ** 2, axis=0)
            W0 = Wof(Mk_.reshape(l, 4, 4, r))
            h = np.einsum('ij,ij->i', W0, W0)
            tr = h.sum()
            hn = h / tr
            nc = nb_ * k

            def jvp(x):
                C = x[:nc].reshape(nb_, k)
                dM = P1 @ C @ Q1 + P2 @ C.T @ Q2
                dW = Wof(dM.reshape(l, 4, 4, r))
                dh = 2.0 * np.einsum('ij,ij->i', W0, dW)
                dh = (dh - hn * dh.sum()) / tr
                return Msp @ dh

            def vjp(g):
                eta = Msp.T @ g
                eta = (eta - (eta @ hn)) / tr
                Gth = Wadj(2.0 * eta[:, None] * W0).reshape(l * 4, 4 * r)
                return (P1.T @ Gth @ Q1.T + Q2 @ Gth.T @ P2).ravel()

            sc = np.broadcast_to(cs[None, :], (nb_, k)).ravel()
            return jvp, vjp, sc / nrm2

        def retract(Bk_, Bp_, x_, al):
            nb_ = Bp_.shape[1]
            C = (al * x_[:nb_ * k]).reshape(nb_, k)
            Q, _ = np.linalg.qr(Bk_ + Bp_ @ C)
            Mk = Q @ (Q.T @ Mm) if dirn == 'R' else (Mm @ Q) @ Q.T
            return Q, Mk

        Bk_c, Bp_c, Mk_c, f_c, kept_c, Q_c = Bk, Bp, Mk0, f_svd, kept_svd, None
        phi_c = phi_svd
        t2 = pc()
        self.tm['setup'] += t2 - t1
        for ps in range(self.passes):
            t2 = pc()
            jvp, vjp, scale = make_chart(Bk_c, Bp_c, Mk_c)
            if ps:
                r_c = Msp @ (self.hn_of(Wof, Mk_c.reshape(l, 4, 4, r)) - ht)
            else:
                r_c = r0
            if ps == 0 and fw > 0:
                wji = (sk[None, :] ** 2 - sq[:, None] ** 2).clip(min=0.0)
                ridge = wji.ravel() * (fw / nrm2)
            else:
                ridge = np.zeros(len(scale))
            Minv = 1.0 / (3.0 * scale + ridge + 1e-9 * float(scale.max()))
            b_ = -vjp(r_c)
            if self.debug is not None and ps == 0:
                def exact(x, Bk=Bk, Bp=Bp):
                    Q, Mk = retract(Bk, Bp, x, 1.0)
                    return Msp @ (self.hn_of(Wof, Mk.reshape(l, 4, 4, r)) - ht)
                self.debug.append(dict(jvp=jvp, vjp=vjp, exact=exact, nx=len(scale), nres=nres, r0=r0, f_svd=f_svd, Wof=Wof, Wadj=Wadj,
                                       l=l, r=r, b=b, k=k, nb=nb, L=L, Msp=Msp, theta=theta, region=reg, dirn=dirn, ht=ht, hn0=h0n,
                                       Tcopy=[t.copy() for t in self.T], wts=wts, span=span, kk=kk))

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
                Q, Mk = retract(Bk_c, Bp_c, x, al)
                kj = float(np.sum(Mk * Mk))
                rj = Msp @ (self.hn_of(Wof, Mk.reshape(l, 4, 4, r)) - ht)
                fj = float(rj @ rj)
                if fj + fw * (1.0 - kj / nrm2) < phi_c * (1 - 1e-3):
                    ok = True
                    break
            self.tm['line'] += pc() - t3
            if not ok:
                break
            if ps == 0:
                self.achosen[jj] = self.achosen.get(jj, 0) + 1
            f_c, kept_c, Q_c, Mk_c = fj, kj, Q, Mk
            phi_c = fj + fw * (1.0 - kj / nrm2)
            Bk_c = Q
            if ps + 1 < self.passes:
                Pb = Bp_c - Q @ (Q.T @ Bp_c)
                Bp_c, _ = np.linalg.qr(Pb)
        if Q_c is None:
            self.rejected += 1
            self.log.append((self.calls, b, 1.0 if dirn == 'R' else 0.0, tail, *st, 0.0, f_svd, f_svd, tail))
            return _plain(U, s, Vh, k, l, r, dirn)
        self.fired += 1
        self.log.append((self.calls, b, 1.0 if dirn == 'R' else 0.0, tail, *st, 1.0, f_svd, f_c, (nrm2 - kept_c) / nrm2))
        Q = Q_c
        if dirn == 'R':
            cen = Q.T @ Mm
            n = np.linalg.norm(cen)
            return Q.reshape(l, 4, k), (cen / n).reshape(k, 4, r), 0.0, True
        cen = Mm @ Q
        n = np.linalg.norm(cen)
        return (cen / n).reshape(l, 4, k), Q.T.reshape(k, 4, r), 0.0, True

    def tail_skip(self, tail):
        return tail < self.rel_skip * self.tmax


LOG_COLS = ['call', 'b', 'dirR', 'tail', 'rms_all', 'rms_k1', 'rms_k2', 'rms_k3', 'rms_span2', 'fired', 'f_svd', 'f_final', 'tail_final']
