"""Observable pin at the cut: SVD compression, then a minimal displacement that restores observables read before the cut.

At a gate cut of a TEBD sweep the two-site tensor M (2l x 2r) has orthonormal environments, so the expectation value of
any operator supported on a region around the cut is a quadratic form in M. The targets t_j are those expectation values
for the untruncated M (nothing larger than M is ever stored, the bond dimension never exceeds chi). The rank-k state M_k
left by the SVD is displaced by dM, the minimal-norm solution of the linearised pin J dM = -(o(M_k) - t), with J the
gradients of the quadratic forms projected on a tangent space of the rank-k matrices:

    tan   full tangent space at the current point, followed by a rank-k retraction (SVD)
    fix   the kept subspace on the side that the sweep leaves behind stays fixed, only the centre tensor moves (the
          spc_mps correction C -> C + mu g, generalised from one scalar to several constraints)
    coef  both kept subspaces stay fixed, only a k x k coefficient matrix moves

The SVD residual is orthogonal to the tangent space of the rank-k matrices at M_k, so a tangent displacement changes the
fidelity only at second order.

Observables (rows of the pin)
    static   all Pauli strings with support span <= k inside the region [b-a, b+2+a)
    ahead    the same strings evolved under the region's own Hamiltonian for the times taus, U^dag P U
    energy   the energy of the whole chain, <theta|H_eff|theta> from Hamiltonian environments
"""
import itertools
import numpy as np
import mpsenh as M

PA = [np.eye(2, dtype=complex), M.X, M.Y, M.Z]


def pauli_strings(L, k):
    """All non-identity Pauli strings on L sites whose support (first to last non-identity site) spans at most k sites."""
    out = []
    for span in range(1, k + 1):
        for start in range(0, L - span + 1):
            if span == 1:
                inner = [()]
                ends = [(x, x) for x in (1, 2, 3)]
            else:
                inner = list(itertools.product(range(4), repeat=span - 2))
                ends = [(x, y) for x in (1, 2, 3) for y in (1, 2, 3)]
            for e in ends:
                for mid in inner:
                    s = [0] * L
                    if span == 1:
                        s[start] = e[0]
                    else:
                        s[start], s[start + span - 1] = e
                        s[start + 1:start + span - 1] = mid
                    out.append(tuple(s))
    return out


def dense(s):
    m = np.eye(1, dtype=complex)
    for x in s:
        m = np.kron(m, PA[x])
    return m


class ObsPin:
    def __init__(self, model, N, a=1, k=2, taus=(), mech='tan', passes=3, rcond=1e-3, w_energy=0.0, w_ahead=1.0,
                 damp=1.0, eps_min=1e-10, static=True, alphas=(1.0, 0.5, 0.25), min_gain=1e-3, touch=False, gam=1.0, cover=1.0):
        self.model, self.N, self.a, self.k = model, N, a, k
        self.taus, self.mech, self.passes, self.rcond = tuple(taus), mech, passes, rcond
        self.w_energy, self.w_ahead, self.damp, self.eps_min = w_energy, w_ahead, damp, eps_min
        self.static, self.alphas, self.min_gain = static, tuple(alphas), min_gain
        self.touch, self.gam, self.cover = touch, gam, cover
        self.henv = None
        if w_energy > 0:
            import mpsenv
            self.henv = mpsenv.HEnv(model, N)
        self.T = None
        self._ops = {}
        self.calls = self.fired = self.skipped = 0
        self.log = []

    def start(self, T):
        self.T = T

    # ------------------------------------------------------------------ region operators
    def _region_ops(self, lo, hi, off):
        L = hi - lo
        key = (lo, hi) if (lo == 0 or hi == self.N) else ('bulk', L)
        if key in self._ops:
            return self._ops[key]
        strs = pauli_strings(L, self.k)
        if self.touch:
            strs = [s for s in strs if s[off] or s[off + 1]]
        S = np.stack([dense(s) for s in strs])                              # (n, D, D)
        g = np.array([self.gam ** (-(sum(1 for x in s if x) - 1)) for s in strs])
        blocks, w = [], []
        if self.static:
            blocks.append(S)
            w += list(g)
        if self.taus:
            D = 2 ** L
            Hm = np.zeros((D, D), dtype=complex)
            for bb in range(lo, hi - 1):
                Hm += np.kron(np.kron(np.eye(2 ** (bb - lo)), M.h_bond(self.model, bb, self.N)), np.eye(2 ** (hi - bb - 2)))
            ev, V = np.linalg.eigh(Hm)
            for t in self.taus:
                U = (V * np.exp(-1j * t * ev)) @ V.conj().T
                blocks.append(np.einsum('ij,njk,kl->nil', U.conj().T, S, U))
                w += list(self.w_ahead * g)
        self._ops[key] = (np.concatenate(blocks), np.array(w))
        return self._ops[key]

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

    # ------------------------------------------------------------------ observables and their gradients
    def _eval(self, Mn, ctx, grad):
        """Values (and gradients wrt conj(M), as matrices) of every pinned quadratic form at the normalised matrix Mn."""
        l, r, Lm, Rm, P = ctx['l'], ctx['r'], ctx['Lm'], ctx['Rm'], ctx['P']
        th = Mn.reshape(l, 2, 2, r)
        Y = np.einsum('oxl,labr->oxabr', Lm, th, optimize=True)
        Y = np.einsum('oxabr,rzp->xabzop', Y, Rm, optimize=True)
        sh = Y.shape
        W = Y.reshape(sh[0] * 4 * sh[3], -1)
        PW = np.matmul(P, W)
        nw = float(np.vdot(W, W).real)
        o = np.einsum('dc,jdc->j', W.conj(), PW).real / nw
        G = None
        if grad:
            Z = (PW - o[:, None, None] * W[None]).reshape((len(o),) + sh)
            A = np.einsum('jxabzop,oxl->jlabzp', Z, Lm.conj(), optimize=True)
            G = np.einsum('jlabzp,rzp->jlabr', A, Rm.conj(), optimize=True).reshape(len(o), 2 * l, 2 * r) / nw
        if self.henv is not None:
            Hth = self.henv.apply_many([th])[0]
            E = float(np.vdot(th, Hth).real)
            o = np.append(o, E)
            if grad:
                GE = ((Hth - E * th) / float(np.vdot(th, th).real)).reshape(1, 2 * l, 2 * r)
                G = np.concatenate([G, GE])
        return o, G

    @staticmethod
    def _project(G, mech, dirn, Uc, Vc):
        if mech == 'tan':
            GV = G @ Vc
            UG = np.einsum('ik,nij->nkj', Uc.conj(), G, optimize=True)
            UGV = np.einsum('ik,nij->nkj', Uc.conj(), GV, optimize=True)
            return np.einsum('ik,nkj->nij', Uc, UG, optimize=True) + GV @ Vc.conj().T - np.einsum(
                'ik,nkj->nij', Uc, UGV @ Vc.conj().T, optimize=True)
        if mech == 'fix':
            if dirn == 'R':
                return np.einsum('ik,nkj->nij', Uc, np.einsum('ik,nij->nkj', Uc.conj(), G, optimize=True), optimize=True)
            return (G @ Vc) @ Vc.conj().T
        UGV = np.einsum('ik,nij->nkj', Uc.conj(), G @ Vc, optimize=True)
        return np.einsum('ik,nkj->nij', Uc, UGV @ Vc.conj().T, optimize=True)

    # ------------------------------------------------------------------ the cut
    def __call__(self, theta, chi, dirn, A, B, b):
        self.calls += 1
        l, r = theta.shape[0], theta.shape[3]
        Mm = theta.reshape(2 * l, 2 * r)
        U, s, Vh = np.linalg.svd(Mm, full_matrices=False)
        k = M._rank(s, chi)
        if self.henv is not None:
            self.henv.prep(b)
        if k >= len(s) or np.sum(s[k:] ** 2) <= self.eps_min * np.sum(s ** 2):
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        lo, hi = max(0, b - self.a), min(self.N, b + 2 + self.a)
        Lm, Rm = self._maps(lo, hi, b, l, r)
        P, w = self._region_ops(lo, hi, b - lo)
        if self.henv is not None:
            w = np.append(w, self.w_energy)
        ctx = dict(l=l, r=r, Lm=Lm, Rm=Rm, P=P)
        M0 = Mm / np.linalg.norm(Mm)
        t, _ = self._eval(M0, ctx, False)
        Uk, Vk = U[:, :k], Vh[:k].conj().T
        Mc = (Uk * s[:k]) @ Vk.conj().T
        Mc = Mc / np.linalg.norm(Mc)
        Uc, Vc = Uk, Vk
        o, G = self._eval(Mc, ctx, True)
        rv = (o - t) * w
        if self.cover < 1.0:
            order = np.argsort(-np.abs(rv))
            cum = np.cumsum(rv[order] ** 2)
            nkeep = int(np.searchsorted(cum, self.cover * cum[-1])) + 1
            mask = np.zeros(len(w))
            mask[order[:nkeep]] = 1.0
            w = w * mask
            rv = rv * mask
        res0 = res = float(np.linalg.norm(rv))
        for _ in range(self.passes):
            if res < 1e-12:
                break
            Gt = self._project(G, self.mech, dirn, Uc, Vc)
            J = 2.0 * w[:, None] * np.concatenate([Gt.real.reshape(len(w), -1), Gt.imag.reshape(len(w), -1)], axis=1)
            Uj, sj, Vjt = np.linalg.svd(J, full_matrices=False)
            keep = sj > self.rcond * sj[0]
            dx = -Vjt[keep].T @ ((Uj[:, keep].T @ rv) / sj[keep])
            h = dx.size // 2
            dM = self.damp * (dx[:h] + 1j * dx[h:]).reshape(2 * l, 2 * r)
            best = None
            for al in self.alphas:
                C = Mc + al * dM
                if self.mech != 'fix':
                    Uq, sq, Vq = np.linalg.svd(C, full_matrices=False)
                    C = (Uq[:, :k] * sq[:k]) @ Vq[:k]
                C = C / np.linalg.norm(C)
                oc, _ = self._eval(C, ctx, False)
                rc = float(np.linalg.norm((oc - t) * w))
                if best is None or rc < best[0]:
                    best = (rc, C)
            if best[0] >= res * (1 - self.min_gain):
                break
            res, Mc = best
            if self.mech == 'tan':
                Uq, sq, Vq = np.linalg.svd(Mc, full_matrices=False)
                Uc, Vc = Uq[:, :k], Vq[:k].conj().T
            o, G = self._eval(Mc, ctx, True)
            rv = (o - t) * w
            res = float(np.linalg.norm(rv))
        if res >= res0 * (1 - self.min_gain):
            self.skipped += 1
            return M.EnhCut._plain(U, s, Vh, k, l, r, dirn)
        self.fired += 1
        f_svd = float(np.sum(s[k:] ** 2) / np.sum(s ** 2))
        f_pin = float(1.0 - abs(np.vdot(M0, Mc)) ** 2)
        self.log.append((res0, res, f_svd, f_pin))
        if self.mech == 'fix':
            if dirn == 'R':
                cen = Uk.conj().T @ Mc
                return Uk.reshape(l, 2, k), (cen / np.linalg.norm(cen)).reshape(k, 2, r), 0.0, True
            cen = Mc @ Vk
            return (cen / np.linalg.norm(cen)).reshape(l, 2, k), Vk.conj().T.reshape(k, 2, r), 0.0, True
        Uq, sq, Vq = np.linalg.svd(Mc, full_matrices=False)
        return M.EnhCut._plain(Uq, sq, Vq, k, l, r, dirn)
