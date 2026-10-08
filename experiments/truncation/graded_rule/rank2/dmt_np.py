"""Standalone numpy DMT (White, Zaletel, Mong, Refael, arXiv:1707.01506, Sec. III and App. B) on a Pauli-basis MPDO, and the Frobenius MPDO baseline.

Object: rho = sum_sigma c_sigma sigma_1 x ... x sigma_N with sigma in (I, X, Y, Z); c is stored as an MPS (l, 4, r) with REAL entries.  The two-qubit
Trotter gates act as real 16 x 16 superoperators S_{ab,cd} = tr[(s_a x s_b) G (s_c x s_d) G^dag] / 4.  Same circuit (gates of mpsenh.make_gates,
Strang forward + backward sweeps) and the same physical initial state (product MPDO (1 + tanh(mu_i) Z_i)/2) as the purification arms.

DMT cut at the bond (b | b+1), with theta = U S Vh (left basis x_alpha = U, right basis y_beta = Vh, centre matrix M = diag(S)):
    T^L_{alpha mu} = tr[x_alpha sigma^mu_b] ~ sum_a v_a U[(a, mu), alpha],   v = A_0^{I} ... A_{b-1}^{I}
    T^R_{beta  mu} = tr[y_beta  sigma^mu_{b+1}] ~ sum_c Vh[beta, (mu, c)] w_c, w = B_{b+2}^{I} ... B_{N-1}^{I}
    Q_L, Q_R = orthonormal bases of the column spaces of T^L, T^R (Gram-Schmidt in the order mu = I, X, Y, Z; rank r_L, r_R <= 4); first column = identity direction
    connected matrix  Mt = M - (M e_R)(e_L^T M) / (e_L^T M e_R)
    D = Q_L,perp^T Mt Q_R,perp (lower-right block); keep its chi' = chi - r_L - r_R largest singular values  ->  D'
    M' = M - Q_L,perp (D - D') Q_R,perp^T            (first block-row and block-column of M untouched, so tr rho, rho_{0..b+1} and rho_{b..N-1} are exactly preserved)
    second SVD M' = U2 S2 V2^T (rank <= r_L + r_R + chi'), recombine: left = U U2, right = S2 V2^T Vh.
Total bond <= chi, so the stored parameters of the MPDO (chi, 4, chi) equal those of the purification (chi, 4, chi).
"""
import time
import numpy as np
import mpsenh as M
import purlib as P

PA = [np.eye(2, dtype=complex), M.X, M.Y, M.Z]
OPS2 = [np.kron(PA[a], PA[b]) for a in range(4) for b in range(4)]


def pauli_super(G):
    """G (4,4) unitary on two qubits -> real (4,4,4,4) superoperator in the Pauli basis, indices (a', b', a, b)."""
    S = np.zeros((16, 16))
    for c, Oc in enumerate(OPS2):
        ev = G @ Oc @ G.conj().T
        for a, Oa in enumerate(OPS2):
            S[a, c] = np.trace(Oa.conj().T @ ev).real / 4
    return S.reshape(4, 4, 4, 4)


def mpdo_gates(model, N, dt):
    return [pauli_super(G.reshape(4, 4)) for G in M.make_gates(model, N, dt)]


def initial_mpdo(N, family, mu):
    T = []
    for m in P.mu_list(N, family, mu):
        t = np.tanh(m) if not np.isinf(m) else float(np.sign(m))
        T.append(np.array([0.5, 0.0, 0.0, 0.5 * t]).reshape(1, 4, 1))
    return T


def mpdo_to_vec(T):
    v = T[0].reshape(4, -1)
    for t in T[1:]:
        v = (v @ t.reshape(t.shape[0], -1)).reshape(-1, t.shape[2])
    return v.reshape(-1)


def vec_to_rho(c, N):
    Pm = np.array(PA)                                   # (4, 2, 2)
    t = c.reshape([4] * N).astype(complex)
    for _ in range(N):
        t = np.tensordot(t, Pm, axes=([0], [0]))        # consumes the first Pauli axis, appends (i, j)
    t = t.transpose(list(range(0, 2 * N, 2)) + list(range(1, 2 * N, 2)))
    return t.reshape(2 ** N, 2 ** N)


def gs_basis(Tm, tol=1e-12):
    """Orthonormal basis of the column space of Tm (columns in order), Gram-Schmidt twice; columns whose residual is below tol * max column norm are dropped."""
    cols, scale = [], max(np.linalg.norm(Tm, axis=0).max(), 1e-300)
    for j in range(Tm.shape[1]):
        v = Tm[:, j].astype(float).copy()
        for _ in range(2):
            for q in cols:
                v = v - q * (q @ v)
        n = np.linalg.norm(v)
        if n > tol * scale:
            cols.append(v / n)
    return np.array(cols).T if cols else np.zeros((Tm.shape[0], 0))


def _split(U, s, Vh, k, l, r, dirn, U2=None, Vh2=None):
    if dirn == 'R':
        return U[:, :k].reshape(l, 4, k), (s[:k, None] * Vh[:k]).reshape(k, 4, r)
    return (U[:, :k] * s[:k]).reshape(l, 4, k), Vh[:k].reshape(k, 4, r)


class FrobCut:
    """Plain SVD truncation of the Pauli MPS (the Frobenius MPDO); the trace is not preserved."""
    def __init__(self):
        self.fired = self.calls = 0

    def __call__(self, theta, chi, dirn, T, b):
        l, r = theta.shape[0], theta.shape[3]
        U, s, Vh = np.linalg.svd(theta.reshape(l * 4, 4 * r), full_matrices=False)
        k = min(chi, int(np.sum(s > 1e-13 * s[0])))
        self.calls += 1
        self.fired += int(k < int(np.sum(s > 1e-13 * s[0])))
        return _split(U, s, Vh, k, l, r, dirn)


class DMTCut:
    def __init__(self, check=False, offset=0):
        self.fired = self.calls = 0
        self.check = check
        self.offset = offset          # extra bond dimension granted on top of chi (the 'DMT at chi + 8' arm uses offset = 8)
        self.log = []

    def __call__(self, theta, chi, dirn, T, b):
        chi = chi + self.offset
        l, r = theta.shape[0], theta.shape[3]
        self.calls += 1
        U, s, Vh = np.linalg.svd(theta.reshape(l * 4, 4 * r), full_matrices=False)
        k0 = int(np.sum(s > 1e-13 * s[0]))
        U, s, Vh = U[:, :k0], s[:k0], Vh[:k0]
        if k0 <= chi:
            return _split(U, s, Vh, k0, l, r, dirn)           # nothing to truncate
        v = np.ones(1)
        for t in T[:b]:
            v = v @ t[:, 0, :]
        w = np.ones(1)
        for t in T[b + 2:][::-1]:
            w = t[:, 0, :] @ w
        TL = np.einsum('a,amk->km', v, U.reshape(l, 4, k0))
        TR = np.einsum('kmc,c->km', Vh.reshape(k0, 4, r), w)
        QL, QR = gs_basis(TL), gs_basis(TR)
        rL, rR = QL.shape[1], QR.shape[1]
        chip = chi - rL - rR
        if chip < 0:
            raise ValueError(f'DMT needs chi >= {rL + rR}, got {chi}')
        Mc = np.diag(s)
        eL, eR = QL[:, 0], QR[:, 0]
        Mt = Mc - np.outer(Mc @ eR, eL @ Mc) / (eL @ Mc @ eR)
        QLp = np.linalg.qr(QL, mode='complete')[0][:, rL:]
        QRp = np.linalg.qr(QR, mode='complete')[0][:, rR:]
        D = QLp.T @ Mt @ QRp
        if min(D.shape) == 0:
            return _split(U, s, Vh, k0, l, r, dirn)
        Ud, sd, Vd = np.linalg.svd(D, full_matrices=False)
        nz = int(np.sum(sd > 1e-13 * max(s[0], 1e-300)))
        if nz <= chip:
            return _split(U, s, Vh, k0, l, r, dirn)           # the connected lower-right block already fits
        Dt = (Ud[:, :chip] * sd[:chip]) @ Vd[:chip]
        Mp = Mc - QLp @ (D - Dt) @ QRp.T
        U2, s2, V2h = np.linalg.svd(Mp, full_matrices=False)
        k = min(chi, int(np.sum(s2 > 1e-13 * s2[0])))
        self.fired += 1
        UU, VV = U @ U2, V2h @ Vh
        if self.check:
            self.log.append(dict(b=b, rL=rL, rR=rR, k=k, k0=k0, trunc=float(np.linalg.norm(D - Dt) / np.linalg.norm(Mc)),
                                 dropped=float(np.sqrt(np.sum(s2[k:] ** 2)))))
        return _split(UU, s2, VV, k, l, r, dirn)


def run_tebd_mpdo(T0, Ss, chi, nsteps, cut, hook=None):
    N = len(T0)
    T = [t.copy() for t in T0]
    t0 = time.time()
    for _ in range(nsteps):
        for b in range(N - 1):
            th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', Ss[b], th)
            if hook:
                hook(th, T, b, 'R', 'pre')
            T[b], T[b + 1] = cut(th, chi, 'R', T, b)
            if hook:
                hook(th, T, b, 'R', 'post')
        for b in range(N - 2, -1, -1):
            th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', Ss[b], th)
            if hook:
                hook(th, T, b, 'L', 'pre')
            T[b], T[b + 1] = cut(th, chi, 'L', T, b)
            if hook:
                hook(th, T, b, 'L', 'post')
    return T, time.time() - t0


def mpdo_metrics(ref, T):
    """Errors of the MPDO T against the Reference (physical rho of the dense purification), trace-normalised; plus trace and lambda_min."""
    N, model = ref.N, ref.model
    rho = vec_to_rho(mpdo_to_vec(T), N)
    herm = float(np.abs(rho - rho.conj().T).max())
    rho = 0.5 * (rho + rho.conj().T)
    trace = float(np.trace(rho).real)
    rho = rho / trace
    S = P.RDMSet(rho, N)
    ob = P.obs_from_rdms(S, model)
    e = {k + '_rms': P._rms(ref.obs[k], ob[k]) for k in ('single', 'nn', 'nnn')}
    e['E_abs'] = float(abs(ref.obs['E'] - ob['E']))
    e['rdm2'] = P.marg_err(ref.S, S, 2)
    e['rdm3'] = P.marg_err(ref.S, S, 3)
    e['zz3'], e['zz4'] = P._rms(ref.obs['zz3'], ob['zz3']), P._rms(ref.obs['zz4'], ob['zz4'])
    e['zzfar'] = float(np.sqrt(np.mean(np.concatenate([(ref.obs['zz3'] - ob['zz3']) ** 2, (ref.obs['zz4'] - ob['zz4']) ** 2]))))
    e['far3'], e['far4'] = P._rms(ref.obs['far3'], ob['far3']), P._rms(ref.obs['far4'], ob['far4'])
    w = np.linalg.eigvalsh(rho - ref.rho)
    e['tdist'] = float(0.5 * np.sum(np.abs(w)))
    e['lam_min'] = float(np.linalg.eigvalsh(rho)[0])
    e['trace'] = trace
    e['herm_err'] = herm
    return e
