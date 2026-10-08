"""spcf gate quantities (tail, B_res, cost) at every LPDO bond cut, audit only (no tilt).

Definitions are those of rule/spcfast.py (SPCFast.__call__ with gate_log=True) and check_gate.py:
  * the cut is examined only if it truncates (k < len(s)), tail = sum s[k:]^2 / sum s^2 > eps_min (1e-7), and tail >= rel_skip (1e-2) *
    the running maximum of tail (same early-return order as SPCFast; skipped cuts are not logged);
  * window: sites lo..hi-1 with lo = max(0, b-a), hi = min(N, b+2+a)  (a = 2 is the calibration config of check_gate.py / spcf_grid.py BEST);
  * the physical region reduced density matrix rho = W W^dag of the UNTRUNCATED theta and of the rank-k SVD truncation Mk0, each normalised by its
    trace; r0 = F (hvec(rho_trunc) - hvec(rho_theta)) with the SAME F matrices (SPCFast._region: static rows then the tau=1.0 lookahead rows, float32 for
    the dense rows, the same Fapply), B_res = SPCFast._low_span_resid(r0) = rms over the span-2 static Pauli strings of r0/sqrt(omega);
  * cost = c0 (tail/1e-4)^0.65, c0 = 1.2e-4;  the gate would fire iff B_res >= cost  (SPCFast: skip if B_res < gate*cost, gate=1).
What is new: the LPDO has Kraus legs.  They are traced, so they join the column side of W.  Instead of forming W (its column count is
c_L * K_b * K_{b+1} * c_R) the environments enter through their Gram matrices over (physical, bond) -- the identical rho = W W^dag:
    GL[X,l,Y,m] = sum_c Lm[c,X,l] conj(Lm[c,Y,m]),  c = (outer bond, Kraus legs of the left environment sites);  GR likewise.
    rho[(X,a,b,Z),(Y,c,d,V)] = sum GL[X,l,Y,m] theta[l,a,i,b,j,r] conj(theta[m,c,i,d,j,s]) GR[Z,r,V,s]   (i,j = Kraus legs of the two cut sites).
For K=1 this is exactly SPCFast's W W^dag, so the gate log must reproduce SPCFast.gate_log (test0.py checks it).
"""
import _p6  # noqa: F401
import numpy as np
import spcfast


def left_gram(T, lo, b, l):
    """GL[X,l,Y,m] for the left environment sites lo..b-1 (identity if there are none)."""
    if lo == b:
        return np.eye(l, dtype=complex).reshape(1, l, 1, l)
    A = T[lo]                                                   # o p K m
    o, p, K, m = A.shape
    Z = A.transpose(0, 2, 1, 3).reshape(o * K, p, m)            # c x l
    for j in range(lo + 1, b):
        Y = np.einsum('cxm,mpkn->ckxpn', Z, T[j])
        c, k, x, p, n = Y.shape
        Z = Y.reshape(c * k, x * p, n)
    c, X, l_ = Z.shape
    mat = Z.reshape(c, X * l_)
    G = mat.T @ mat.conj()                                      # [(X,l),(Y,m)] = sum_c Z[c,X,l] conj(Z[c,Y,m])
    return G.reshape(X, l_, X, l_)


def right_gram(T, b, hi, r):
    """GR[Z,r,V,s] for the right environment sites b+2..hi-1 (identity if there are none)."""
    if hi == b + 2:
        return np.eye(r, dtype=complex).reshape(1, r, 1, r)
    Z = T[b + 2]                                                # r p K m  ->  [r, x, c, m]
    r_, p, K, m = Z.shape
    Z = Z.reshape(r_, p, K, m)
    for j in range(b + 3, hi):
        Y = np.einsum('rxcm,mpkn->rxpckn', Z, T[j])
        r_, x, p, c, k, n = Y.shape
        Z = Y.reshape(r_, x * p, c * k, n)
    r_, X, c, m = Z.shape
    R = Z.transpose(1, 0, 2, 3).reshape(X * r_, c * m)          # [(X,r), (c,outer)]
    G = R @ R.conj().T
    return G.reshape(X, r_, X, r_)


def region_rho(GL, GR, th):
    """rho[(X,a,b,Z),(Y,c,d,V)] of the region, Kraus legs of the cut sites traced; th is (l,a,i,b,j,r)."""
    S1 = np.tensordot(GL, th, axes=([1], [0]))                  # X Y m a i b j r
    S2 = np.tensordot(S1, th.conj(), axes=([2, 4, 6], [0, 2, 4]))   # X Y a b r c d s
    S3 = np.tensordot(S2, GR, axes=([4, 7], [1, 3]))            # X Y a b c d Z V
    X, Z = S3.shape[0], S3.shape[6]
    D = X * 4 * Z
    return S3.transpose(0, 2, 3, 6, 1, 4, 5, 7).reshape(D, D)


class GateProbe:
    """Callable hook for lpdo.run_lpdo.  `a` may be an int or a tuple of window half-widths evaluated side by side (each gets its own log)."""
    GATE_C0, GATE_EXP = 1.2e-4, 0.65

    def __init__(self, model, N, a=(2,), eps_min=1e-7, rel_skip=1e-2):
        self.model, self.N = model, N
        self.avals = tuple(a) if not isinstance(a, int) else (a,)
        self.eps_min, self.rel_skip = eps_min, rel_skip
        self.sp = {av: spcfast.SPCFast(model, N, a=av, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=eps_min, rel_skip=rel_skip, gate_log=True)
                   for av in self.avals}
        self.tmax = 0.0
        self.log = {av: [] for av in self.avals}     # (tail, Bres, cost, b)
        self.ncalls = 0
        self.ntrunc = 0
        self.entries = []                            # per logged cut: (op index, b, dirn, tail)
        self._f32 = {}

    def start(self, T, N):
        pass

    def _Fapply(self, av, key, Fs, Fd64, H):
        Fd32 = self._f32.get((av, key))
        if Fd32 is None:
            Fd32 = self._f32[(av, key)] = np.ascontiguousarray(Fd64.astype(np.float32))
        Rd = (Fd32 @ H.astype(np.float32)).astype(float)
        return np.concatenate([Fs @ H, Rd], axis=0)

    @staticmethod
    def _hvec(rho, iu, dg):
        return np.concatenate([rho[dg, dg].real, rho[iu].real, rho[iu].imag])

    def __call__(self, T, th, U, s, Vh, k, chi, b, dirn):
        self.ncalls += 1
        nrm2 = float(np.sum(s ** 2))
        if k >= len(s) or np.sum(s[k:] ** 2) <= self.eps_min * nrm2:
            return
        self.ntrunc += 1
        tail = float(np.sum(s[k:] ** 2) / nrm2)
        self.tmax = max(self.tmax, tail)
        if tail < self.rel_skip * self.tmax:
            return
        l, K1, K2, r = th.shape[0], th.shape[2], th.shape[4], th.shape[5]
        Mk0 = (U[:, :k] * s[:k]) @ Vh[:k]
        thk = Mk0.reshape(l, 2, K1, 2, K2, r)
        cost = self.GATE_C0 * (tail / 1e-4) ** self.GATE_EXP
        self.entries.append((self.ncalls, b, dirn, tail))
        for av in self.avals:
            sp = self.sp[av]
            N = self.N
            lo, hi = max(0, b - av), min(N, b + 2 + av)
            D, (Fs, Fd64), iu, dg = sp._region(hi - lo, hi == N, b - lo)
            GL = left_gram(T, lo, b, l)
            GR = right_gram(T, b, hi, r)
            rt = region_rho(GL, GR, th)
            rk = region_rho(GL, GR, thk)
            ht = self._hvec(rt, iu, dg) / rt[dg, dg].real.sum()
            hk = self._hvec(rk, iu, dg) / rk[dg, dg].real.sum()
            r0 = self._Fapply(av, (hi - lo, hi == N, b - lo), Fs, Fd64, (hk - ht)[:, None])[:, 0]
            Bres = sp._low_span_resid(r0, hi - lo)
            self.log[av].append((tail, Bres, cost, b))
            # keep the full residual norm too (f_svd of SPCFast), cheap and useful for the audit
            if not hasattr(self, 'fsvd'):
                self.fsvd = {a_: [] for a_ in self.avals}
            self.fsvd[av].append(float(r0 @ r0))
