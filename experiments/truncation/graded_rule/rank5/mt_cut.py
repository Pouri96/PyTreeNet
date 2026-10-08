"""Rank 5 first move: dense single-cut multi-target truncation harness.

Plan: reports/first_moves/rank5_dmrg.md.  One cut of a dense N-site state set; the shared LEFT basis (sites 0..b) is
chosen from the stacked targets and every target is projected as Q Q^dag M_f ('R' direction of spcfast).  Nothing is
truncated anywhere else, so the two-site tensor of the plan is the full state reshaped to (2^(b+1), 2^(N-b-1)).

Fit region (spcfast default a=2): sites b-2 .. b+3 (6 sites).  Objective of spcfast with taus=() and wk=1:
    f(M~) = sum_{k=1,2,3} 1/nwin_k  sum_{windows w of k sites in the region} || rho_w(M~) - rho_w(M) ||_F^2
(rho normalised to unit trace), i.e. the residual r = F (h(rho) - h(rho_exact)) of SPCFast written window by window.

Arms (plan, "Where to start", step 3):
  i    stacked SVD, literature weights
  ii   weight grid (plan grid) with an oracle pick on E_near;  ii+ = denser oracle (random simplex search + Nelder-Mead)
  iii  dressed rho + a sum_P P rho P over the 1-site Paulis on the window sites of the left block, a in {1e-3,1e-2,1e-1}
  iv   spcf-multi: dense Gauss-Newton / Levenberg-Marquardt on one shared C, Q = qr(B_k + B_perp C), finite-difference
       Jacobian, equal per-target residual weights, started from (i); kappa-constrained extra discarded weight
       (iv-u: unconstrained, an upper bound on what the tilt can do)
  v    per-target separate SVD (unshared basis), reference ceiling
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import json
import time
import itertools
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _paths  # noqa: F401,E402  (this repository's pytreenet and rule/ ahead of any other copy)

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
import mpsenh as M  # noqa: E402
import exact_ref  # noqa: E402

SX, SY, SZ = M.X, M.Y, M.Z
PAULIS = [SX, SY, SZ]
EXTRA_FIELDS = {'tfim_crit': (1.0, 0.1)}      # near-critical TFIM, kept out of mpsenh.FIELDS as the plan says
ETA = 0.2
TAU_STEP = 0.1


# ------------------------------------------------------------------------------------------ Hamiltonians and targets
def _site_op(P, i, N):
    return sp.kron(sp.kron(sp.identity(2 ** i), sp.csr_matrix(P), format='csr'), sp.identity(2 ** (N - i - 1)), format='csr')


def build_H(model, N):
    """real sparse H (all four models are real); site 0 is the most significant bit, as in exact_ref."""
    if model in EXTRA_FIELDS:
        hx, hz = EXTRA_FIELDS[model]
        H = sp.csr_matrix((2 ** N, 2 ** N), dtype=float)
        for i in range(N - 1):
            H = H + _site_op(SZ, i, N).real @ _site_op(SZ, i + 1, N).real
        for i in range(N):
            H = H + hx * _site_op(SX, i, N).real + hz * _site_op(SZ, i, N).real
        return H.tocsr()
    H = exact_ref.sparse_H(model, N)
    assert abs(H.imag).max() < 1e-14
    return H.real.tocsr()


def lowest(H, k, seed=0):
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(H.shape[0])
    ev, V = spla.eigsh(H, k=k, which='SA', tol=1e-13, v0=v0, ncv=min(H.shape[0] - 1, max(60, 4 * k)))
    o = np.argsort(ev)
    return ev[o], V[:, o]


def lanczos_weights(H, v, m=300):
    """Ritz energies and weights of v = A|0>/|A|0>| (full re-orthogonalisation) for the spectral function"""
    n = len(v)
    Q = np.zeros((m, n))
    al, be = np.zeros(m), np.zeros(m)
    q = v / np.linalg.norm(v)
    Q[0] = q
    mm = m
    for j in range(m):
        w = H @ Q[j]
        al[j] = Q[j] @ w
        w = w - al[j] * Q[j] - (be[j - 1] * Q[j - 1] if j else 0)
        w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
        w -= Q[:j + 1].T @ (Q[:j + 1] @ w)
        bn = np.linalg.norm(w)
        if j + 1 == m or bn < 1e-12:
            mm = j + 1
            break
        be[j] = bn
        Q[j + 1] = w / bn
    T = np.diag(al[:mm]) + np.diag(be[:mm - 1], 1) + np.diag(be[:mm - 1], -1)
    th, S = np.linalg.eigh(T)
    return th, S[0] ** 2


def spectral_peak(H, E0, a0, eta=ETA, m=300):
    """omega of the main peak and of the half-maximum on its high-energy side, of the eta-broadened S(omega) of A|0>"""
    th, w = lanczos_weights(H, a0, m)
    om = np.linspace(0.0, th[-1] - E0 + 1.0, 8001)
    Sw = (w[None, :] * (eta / np.pi) / ((om[:, None] - (th[None, :] - E0)) ** 2 + eta ** 2)).sum(1)
    ip = int(np.argmax(Sw))
    below = np.where(Sw[ip:] <= 0.5 * Sw[ip])[0]
    ih = ip + (int(below[0]) if len(below) else len(Sw) - 1 - ip)
    return float(om[ip]), float(om[ih]), float(Sw[ip]), float(Sw[ih])


def solve_cv(H, E0, a0, omega, eta=ETA):
    """x = (H - E0 - omega - i eta)^-1 a0 for real H and real a0; returns (Re x, Im x).  Kuehner-White Eq. (24):
    Im x = eta [(H-E0-omega)^2 + eta^2]^-1 a0 solved by CG, Re x = (H-E0-omega) Im x / eta."""
    n = len(a0)
    sh = E0 + omega

    def mv(y):
        z = H @ y - sh * y
        return H @ z - sh * z + eta ** 2 * y

    K = spla.LinearOperator((n, n), matvec=mv, dtype=float)
    xi, info = spla.cg(K, eta * a0, rtol=1e-14, atol=0.0, maxiter=20000)
    if info != 0:
        raise RuntimeError(f'CG did not converge, info={info}')
    xr = (H @ xi - sh * xi) / eta
    return xr, xi


def neel_dense(N):
    return M.mps_to_dense(M.neel_mps(N)).astype(complex)


class TargetSet:
    """name, list of dense vectors (unnormalised, as the physics defines them), literature weights, grid of weights, meta"""

    def __init__(self, name, vecs, w_lit, grid, meta=None):
        self.name = name
        self.vecs = [np.asarray(v) for v in vecs]
        self.nrm = np.array([np.linalg.norm(v) for v in self.vecs])
        self.vn = [v / n for v, n in zip(self.vecs, self.nrm)]
        self.w_lit = np.asarray(w_lit, float)
        self.grid = grid                        # list of (label, weights)
        self.meta = meta or {}
        self.F = len(self.vecs)
        self.cplx = any(np.iscomplexobj(v) for v in self.vecs)


def make_sets(model, N, which=('S1', 'S2', 'S3', 'S4'), c=None, t_list=(1.0, 2.0, 3.0), tau=TAU_STEP, eta=ETA, cache=None):
    """Return list of TargetSet and a dict of shared data (H, E, ...) for one model."""
    H = build_H(model, N)
    ev, V = lowest(H, 3)
    c = N // 2 - 1 if c is None else c
    sets = []
    sh = dict(H=H, E=ev, c=c)
    if 'S1' in which:
        sets.append(TargetSet('S1', [V[:, 0]], [1.0], [('lit', [1.0])], dict(E=[ev[0]])))
    if 'S2' in which:
        grid = [(f'g{g}', [g, (1 - g) / 2, (1 - g) / 2]) for g in (0.05, 0.1, 0.25, 0.5)]
        sets.append(TargetSet('S2', [V[:, 0], V[:, 1], V[:, 2]], [1 / 3] * 3, grid, dict(E=list(ev))))
    if 'S3' in which:
        Aop = _site_op(SZ, c, N).real
        a0 = Aop @ V[:, 0]
        wp, wh, Sp, Sh = spectral_peak(H, ev[0], a0, eta)
        sh['omega'] = (wp, wh)
        for tag, om in (('p', wp), ('h', wh)):
            xr, xi = solve_cv(H, ev[0], a0, om, eta)
            G = a0 @ (xr + 1j * xi)
            grid = [(f'g{g}', [g, (1 - g) / 3, (1 - g) / 3, (1 - g) / 3]) for g in (0.05, 0.1, 0.25, 0.5)]
            sets.append(TargetSet(f'S3{tag}', [V[:, 0], a0, xr, xi], [0.1, 0.3, 0.3, 0.3], grid,
                                  dict(E=[ev[0]], omega=om, eta=eta, G=complex(G), Aop=Aop)))
    if 'S4' in which:
        psi0 = neel_dense(N)
        Hc = -1j * H
        for t in t_list:
            # one call per time: expm_multiply(start=t>0, ...) returns wrongly scaled vectors in this scipy
            vs = [spla.expm_multiply((t + j * tau / 3.0) * Hc, psi0) for j in range(4)]
            grid = [('fw_a', [1 / 2, 1 / 6, 1 / 6, 1 / 6]), ('fw_b', [1 / 2, 1 / 8, 1 / 8, 1 / 4]),
                    ('equal', [1 / 4] * 4), ('fw_c', [1 / 3, 1 / 6, 1 / 6, 1 / 3])]
            sets.append(TargetSet(f'S4t{t:g}', [vs[i] for i in range(4)], [1 / 3, 1 / 6, 1 / 6, 1 / 3], grid,
                                  dict(t=t, tau=tau)))
    return sets, sh


# ------------------------------------------------------------------------------------------ geometry and window objective
class Geom:
    """cut after site b of an N-chain; left block sites 0..b, right block b+1..N-1; fit region sites b-a .. b+1+a"""

    def __init__(self, N, b, a=2, ks=(1, 2, 3)):
        self.N, self.b, self.a = N, b, a
        self.lo, self.hi = b - a, b + 2 + a
        assert self.lo >= 0 and self.hi <= N, 'fit region leaves the chain'
        self.dimL, self.dimR = 2 ** (b + 1), 2 ** (N - b - 1)
        self.oL, self.rL = 2 ** self.lo, 2 ** (b + 1 - self.lo)
        self.rR, self.oR = 2 ** (self.hi - b - 1), 2 ** (N - self.hi)
        self.L = self.hi - self.lo
        self.D = 2 ** self.L
        self.ks = tuple(ks)
        self.wins = [(s, k) for k in ks for s in range(self.L - k + 1)]
        self.cw = {k: 1.0 / (self.L - k + 1) for k in ks}          # 1/nwin_k
        # windows of 1..3 sites anywhere in the chain, classified against the fit region
        self.allwins = [(s, k) for k in (1, 2, 3) for s in range(N - k + 1)]
        self.cls = {}
        for (s, k) in self.allwins:
            if s >= self.lo and s + k <= self.hi:
                self.cls[(s, k)] = 'near'
            elif s + k <= self.lo or s >= self.hi:
                self.cls[(s, k)] = 'far'
            else:
                self.cls[(s, k)] = 'edge'
        self._iu = {k: np.triu_indices(2 ** k, 1) for k in (1, 2, 3)}
        self.nres = sum(4 ** k for (s, k) in self.wins)

    # --- region density matrix from a left-right matrix (batched on leading axes)
    def region_rho(self, Mt):
        bs = Mt.shape[:-2]
        W = Mt.reshape(bs + (self.oL, self.rL, self.rR, self.oR))
        W = np.moveaxis(W, -4, -2).reshape(bs + (self.rL * self.rR, self.oL * self.oR))
        rho = W @ np.swapaxes(W.conj(), -1, -2)
        tr = np.trace(rho, axis1=-2, axis2=-1).real
        return rho / tr[..., None, None]

    def win_rdm(self, rho, s, k):
        """partial trace of the region rho (…, D, D) to the window (s,k) in region coordinates"""
        bs = rho.shape[:-2]
        A, K, C = 2 ** s, 2 ** k, 2 ** (self.L - s - k)
        R = rho.reshape(bs + (A, K, C, A, K, C))
        return np.einsum('...aicajc->...ij', R)

    def hvec(self, rho):
        """weighted real vector of all window differences' basis: diag, sqrt2 Re upper, sqrt2 Im upper per window, times
        sqrt(1/nwin_k); batched on leading axes.  |hvec(rho) - hvec(tau)|^2 is the spcfast objective."""
        out = []
        for (s, k) in self.wins:
            R = self.win_rdm(rho, s, k)
            iu = self._iu[k]
            w = np.sqrt(self.cw[k])
            out.append(w * np.concatenate([np.einsum('...ii->...i', R).real, np.sqrt(2) * R[..., iu[0], iu[1]].real,
                                           np.sqrt(2) * R[..., iu[0], iu[1]].imag], axis=-1))
        return np.concatenate(out, axis=-1)

    # --- RDMs of arbitrary windows from a dense state matrix
    def state_rdm(self, Mt, s, k):
        t = np.asarray(Mt).reshape((2,) * self.N)
        t = np.moveaxis(t, list(range(s, s + k)), list(range(k))).reshape(2 ** k, -1)
        r = t @ t.conj().T
        return r / np.trace(r).real

    def all_rdms(self, Mt):
        return {w: self.state_rdm(Mt, *w) for w in self.allwins}


def trace_distance(A, B):
    return 0.5 * float(np.abs(np.linalg.eigvalsh((A - B + (A - B).conj().T) / 2)).sum())


# ------------------------------------------------------------------------------------------ the arms (i), (ii), (iii), (v)
def stack_basis(Ms, w):
    """left singular vectors (a full unitary) and singular values of the stacked matrix [sqrt(w_f) M_f]"""
    X = np.concatenate([np.sqrt(wi) * Mi for wi, Mi in zip(w, Ms)], axis=1)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    if U.shape[1] < U.shape[0]:
        U, _ = np.linalg.qr(np.concatenate([U, np.random.default_rng(0).standard_normal((U.shape[0], U.shape[0] - U.shape[1]))], axis=1))
    return U, s


def project(Ms, Q):
    return [Q @ (Q.conj().T @ Mi) for Mi in Ms]


def arm_stack(Ms, w, chi):
    U, s = stack_basis(Ms, w)
    return U[:, :chi], U


def arm_dressed(Ms, w, chi, a, geom, ops=None):
    """rho' = rho + a sum_{s in window sites of the left block, P in X,Y,Z} P_s rho P_s, rho = sum_f w_f M_f M_f^dag"""
    rho = sum(wi * Mi @ Mi.conj().T for wi, Mi in zip(w, Ms))
    if ops is None:
        ops = left_pauli_ops(geom)
    rd = rho.copy()
    for P in ops:
        rd = rd + a * (P @ rho @ P.conj().T)
    if np.isrealobj(rho):
        rd = rd.real
    ev, V = np.linalg.eigh((rd + rd.conj().T) / 2)
    return V[:, ::-1][:, :chi]


def left_pauli_ops(geom):
    ops = []
    for s in range(geom.lo, geom.b + 1):
        for P in PAULIS:
            ops.append(_site_op(P, s, geom.b + 1).toarray())
    return ops


def arm_separate(Ms, chi):
    out = []
    for Mi in Ms:
        U, _, _ = np.linalg.svd(Mi, full_matrices=False)
        out.append(U[:, :chi] @ (U[:, :chi].conj().T @ Mi))
    return out


def eps_of(Ms, Mt):
    """discarded weight fraction of every (normalised) target"""
    return np.array([1.0 - np.linalg.norm(m) ** 2 / np.linalg.norm(o) ** 2 for m, o in zip(Mt, Ms)])


# ------------------------------------------------------------------------------------------ metrics
class Scorer:
    """exact-side data for one (geometry, target set); score(projected matrices) -> metric dict"""

    def __init__(self, geom, tset, sh):
        self.g, self.ts, self.sh = geom, tset, sh
        self.Ms = [v.reshape(geom.dimL, geom.dimR) for v in tset.vn]
        self.rho_ex = [geom.region_rho(m) for m in self.Ms]
        self.hv_ex = [geom.hvec(r) for r in self.rho_ex]
        self.rdm_ex = [geom.all_rdms(m) for m in self.Ms]
        H = sh['H']
        self.Hv = H
        self.Eex = [float(np.real(np.vdot(v, H @ v))) for v in tset.vn]
        N = geom.N
        self.Zops = [_site_op(SZ, i, N).real for i in range(N)]
        self.sz_ex = [np.array([float(np.real(np.vdot(v, Z @ v))) for Z in self.Zops]) for v in tset.vn]

    def objective(self, Mts):
        """sum_f ||h(rho(M~_f)) - h(rho(M_f))||^2  (the spcf objective with equal target weights) and per-target values"""
        per = []
        for Mt, he in zip(Mts, self.hv_ex):
            h = self.g.hvec(self.g.region_rho(Mt))
            per.append(float(np.sum((h - he) ** 2)))
        return float(sum(per)), per

    def score(self, Mts, detail=False):
        g, ts = self.g, self.ts
        out = {}
        eps = eps_of(self.Ms, Mts)
        out['eps'] = eps.tolist()
        out['fid'] = (1.0 - eps).tolist()
        tdn, tde, tdf = [], [], []
        for f, Mt in enumerate(Mts):
            rd = g.all_rdms(Mt)
            tn, te, tf = [], [], []
            for w in g.allwins:
                d = trace_distance(rd[w], self.rdm_ex[f][w])
                {'near': tn, 'edge': te, 'far': tf}[g.cls[w]].append(d)
            tdn.append(max(tn)), tde.append(max(te)), tdf.append(max(tf))
        out['E_near_t'], out['E_edge_t'], out['E_far_t'] = tdn, tde, tdf
        out['E_near'] = float(max(tdn))
        out['E_edge'] = float(max(tde))
        out['E_far'] = float(max(tdf))
        out['E_out'] = float(max(max(tde), max(tdf)))                  # any window not contained in the fit region
        out['obj'], out['obj_t'] = self.objective(Mts)
        # eigen-energy errors (Rayleigh quotient of the projected target) vs the exact value of the same target
        en = []
        for f, Mt in enumerate(Mts):
            v = Mt.reshape(-1)
            en.append(float(np.real(np.vdot(v, self.Hv @ v)) / np.real(np.vdot(v, v))))
        out['dE_t'] = [abs(e - e0) for e, e0 in zip(en, self.Eex)]
        if ts.name.startswith('S3'):
            v = [m.reshape(-1) * n for m, n in zip(Mts, ts.nrm)]
            Gt = np.vdot(v[1], v[2] + 1j * v[3])
            out['G_rel'] = float(abs(Gt - ts.meta['G']) / abs(ts.meta['G']))
        if ts.name.startswith('S4'):
            rms = []
            for f, Mt in enumerate(Mts):
                v = Mt.reshape(-1)
                v = v / np.linalg.norm(v)
                sz = np.array([float(np.real(np.vdot(v, Z @ v))) for Z in self.Zops])
                rms.append(float(np.sqrt(np.mean((sz - self.sz_ex[f]) ** 2))))
            out['Sz_rms_t'] = rms
            out['Sz_rms'] = float(max(rms))
        return out

    # --- gate quantities of the SVD cut (spcfast: Bres vs c0 (tail/1e-4)^0.65)
    def bres(self, Mts):
        """per target: rms over the 45 span-2 Pauli strings of the region of Tr[P (rho~ - rho)], and the relative tail"""
        g = self.g
        res = []
        for f, Mt in enumerate(Mts):
            rho = g.region_rho(Mt)
            vals = []
            for s in range(g.L - 1):
                R = g.win_rdm(rho, s, 2)
                T = g.win_rdm(self.rho_ex[f], s, 2)
                for P in PAULIS:
                    for Q in PAULIS:
                        PQ = np.kron(P, Q)
                        vals.append(float(np.real(np.trace(PQ @ (R - T)))))
            res.append(float(np.sqrt(np.mean(np.square(vals)))))
        return res


GATE_C0, GATE_EXP = 1.2e-4, 0.65


def gate_cost(tail):
    return GATE_C0 * (max(tail, 1e-300) / 1e-4) ** GATE_EXP


# ------------------------------------------------------------------------------------------ spcf-multi: dense GN / SQP
def complete_basis(Q):
    """orthonormal (Bk, Bp): Bk spans Q, Bp its orthogonal complement"""
    Qf, _ = np.linalg.qr(Q, mode='complete')
    k = Q.shape[1]
    return Qf[:, :k], Qf[:, k:]


class SPCMulti:
    """Shared-C Gauss-Newton on the Grassmannian chart Q = qr(B_k + B_perp C) around the current subspace (the chart is
    re-centred after every accepted step).  Residual r = (h(rho(Q Q^dag M_f)) - h(rho(M_f)))_f over all targets with equal
    weights; Jacobian by forward finite differences of the true map (the plan's 'dense Gauss-Newton with finite-difference
    Jacobian').  Constraint (trust region): eps_f(Q) <= (1 + kappa) eps_f(Q_stack) for every target, i.e. the per-target
    fidelity loss against the stacked-SVD basis is at most kappa times that basis's discarded weight of the same target.
    Each step is the solution of  min |r_c + J x|^2 + lam |x|^2  s.t. the second-order model of the constraints, via its
    F-dimensional dual; the second-order model of eps_f is exact (closed form from the left density matrix of target f)."""

    def __init__(self, geom, Ms, hv_ex, Q0, cplx, kappa=0.1, h=1e-6, chunk=96):
        self.g, self.Ms, self.hv_ex = geom, Ms, hv_ex
        self.Q0 = Q0
        self.chi = Q0.shape[1]
        self.nb = Q0.shape[0] - self.chi
        self.cplx = cplx
        self.kappa, self.h, self.chunk = kappa, h, chunk
        self.npar = self.nb * self.chi * (2 if cplx else 1)
        self.F = len(Ms)
        self.nrm2 = np.array([np.linalg.norm(m) ** 2 for m in Ms])
        self.eps0 = np.maximum(eps_of(Ms, project(Ms, Q0)), 1e-14)
        self.rho = [m @ m.conj().T / n2 for m, n2 in zip(Ms, self.nrm2)]
        self.nev = 0

    def to_C(self, X):
        X = np.atleast_2d(X)
        n = self.nb * self.chi
        if self.cplx:
            return (X[:, :n] + 1j * X[:, n:]).reshape(len(X), self.nb, self.chi)
        return X.reshape(len(X), self.nb, self.chi)

    def Q_of(self, X, Bk, Bp):
        C = self.to_C(X)
        Q, _ = np.linalg.qr(Bk[None] + Bp[None] @ C)
        return Q

    def evaluate(self, X, Bk, Bp):
        """X (n, npar) -> r (n, F*nres), eps (n, F)"""
        X = np.atleast_2d(X)
        rs, es = [], []
        for i0 in range(0, len(X), self.chunk):
            Q = self.Q_of(X[i0:i0 + self.chunk], Bk, Bp)
            rr, ee = [], []
            for f, Mf in enumerate(self.Ms):
                R = np.swapaxes(Q.conj(), -1, -2) @ Mf                      # (n, chi, dimR)
                Mt = Q @ R
                ee.append(1.0 - np.sum(np.abs(R) ** 2, axis=(-2, -1)) / self.nrm2[f])
                rr.append(self.g.hvec(self.g.region_rho(Mt)) - self.hv_ex[f][None])
            rs.append(np.concatenate(rr, axis=-1))
            es.append(np.stack(ee, axis=-1))
        self.nev += len(X)
        return np.concatenate(rs), np.concatenate(es)

    def eval_Q(self, Q):
        """residual and eps of an explicit orthonormal Q (chi columns)"""
        rr, ee = [], []
        for f, Mf in enumerate(self.Ms):
            R = Q.conj().T @ Mf
            ee.append(1.0 - np.sum(np.abs(R) ** 2) / self.nrm2[f])
            rr.append(self.g.hvec(self.g.region_rho(Q @ R)) - self.hv_ex[f])
        self.nev += 1
        return np.concatenate(rr), np.array(ee)

    def eps_model(self, Bk, Bp):
        """per target: (eps_c, grad (npar), Mq (npar,npar)) with eps(x) = eps_c + grad.x + x^T Mq x to second order"""
        out = []
        n = self.nb * self.chi
        for f in range(self.F):
            rho = self.rho[f]
            Kk, Kkp, Kpp = Bk.conj().T @ rho @ Bk, Bk.conj().T @ rho @ Bp, Bp.conj().T @ rho @ Bp
            epsc = 1.0 - float(np.trace(Kk).real)
            ell = (-2.0 * Kkp.T).reshape(-1)                      # ell[p*chi+i] = -2 Kkp[i,p]
            Hc = np.kron(np.eye(self.nb), Kk.T) - np.kron(Kpp, np.eye(self.chi))
            if self.cplx:
                grad = np.concatenate([ell.real, -ell.imag])
                Hr, Hi = Hc.real, Hc.imag
                Mq = np.block([[Hr, -Hi], [Hi, Hr]])
            else:
                grad = ell.real
                Mq = Hc.real
            out.append((epsc, grad, 0.5 * (Mq + Mq.T)))
        return out

    def _dual_step(self, JtJ2, g0, emod, lam, nu0=None, kmod=None):
        """solve min x^T (JtJ2/2 ... ) as described; returns x, nu.  Objective  q(x) = g0.x + 1/2 x^T JtJ2 x  (+ lam/2 |x|^2)"""
        n = self.npar
        I = np.eye(n)
        kmod = self.kappa if kmod is None else kmod
        if self.kappa is None:
            x = -np.linalg.solve(JtJ2 + lam * I, g0)
            return x, np.zeros(self.F), True
        sc = np.array([1.0 / self.eps0[f] for f in range(self.F)])
        base = JtJ2 + lam * I

        def prim(nu):
            Hs = base.copy()
            gs = g0.copy()
            for f in range(self.F):
                if nu[f] > 0:
                    Hs += 2.0 * nu[f] * sc[f] * emod[f][2]
                    gs = gs + nu[f] * sc[f] * emod[f][1]
            return Hs, gs

        def cons(x):
            return np.array([(emod[f][0] + emod[f][1] @ x + x @ emod[f][2] @ x) * sc[f] - 1.0 - kmod
                             for f in range(self.F)])

        # projected Newton / active set on the F-dimensional dual
        nu = np.zeros(self.F) if nu0 is None else nu0.copy()
        nus = (g0 @ g0) ** 0.5 + 1e-30
        for _ in range(60):
            Hs, gs = prim(nu)
            try:
                L = np.linalg.cholesky(Hs)
            except np.linalg.LinAlgError:
                return None, nu, False
            x = -np.linalg.solve(Hs, gs)
            c = cons(x)
            act = (nu > 0) | (c > 0)
            if np.all(c[~act] <= 1e-12) and np.all(np.abs(c[nu > 0]) < 1e-4 * self.kappa) and np.all(c <= 1e-4 * self.kappa):
                return x, nu, True
            # Newton on the active constraints: dc_f/dnu_k = (grad_f + 2 Mq_f x)^T dx/dnu_k, dx/dnu_k = -Hs^-1 (sc_k (grad_k + 2 Mq_k x))
            A = np.array([sc[f] * (emod[f][1] + 2.0 * emod[f][2] @ x) for f in range(self.F)])        # (F, n)
            Dm = A @ np.linalg.solve(Hs, A.T)
            idx = np.where(act)[0]
            if len(idx) == 0:
                return x, nu, True
            rhs = c[idx]                      # dc/dnu = -Dm  =>  Dm d = c  (raise nu where violated)
            try:
                d = np.linalg.solve(Dm[np.ix_(idx, idx)] + 1e-12 * np.eye(len(idx)) * np.trace(Dm) / max(self.F, 1), rhs)
            except np.linalg.LinAlgError:
                return None, nu, False
            dn = np.zeros(self.F)
            dn[idx] = d
            # step with non-negativity, damped
            t = 1.0
            for _ in range(30):
                trial = np.maximum(nu + t * dn, 0.0)
                Ht, gt = prim(trial)
                try:
                    np.linalg.cholesky(Ht)
                    break
                except np.linalg.LinAlgError:
                    t *= 0.5
            nu = np.maximum(nu + t * dn, 0.0)
        Hs, gs = prim(nu)
        x = -np.linalg.solve(Hs, gs)
        return x, nu, bool(np.all(cons(x) <= 5e-2 * self.kappa))

    def run(self, maxit=40, tol=1e-4, lam0=1e-3, verbose=False):
        chi, nb = self.chi, self.nb
        Qc = self.Q0.copy()
        rc, ec = self.eval_Q(Qc)
        f0 = fc = float(rc @ rc)
        if self.npar == 0:                                    # chi = full dimension: nothing to tilt
            return Qc, dict(f0=f0, f=fc, nit=0, nev=self.nev, hist=[f0], g_f=(ec / self.eps0 - 1.0).tolist())
        lam = lam0 * 1.0
        hist = [f0]
        stall = 0
        nu = None
        nit = 0
        kap = self.kappa
        for it in range(maxit):
            nit += 1
            Bk, Bp = complete_basis(Qc)
            X = self.h * np.eye(self.npar)
            r1, _ = self.evaluate(X, Bk, Bp)
            J = ((r1 - rc[None]) / self.h).T                               # (m, npar)
            JtJ2 = 2.0 * J.T @ J
            g0 = 2.0 * J.T @ rc
            emod = self.eps_model(Bk, Bp) if kap is not None else None
            dsc = np.maximum(np.diag(JtJ2), 1e-12 * np.diag(JtJ2).max())
            accepted = False
            kmod = kap
            ncorr = 0
            for tr in range(30):
                x, nu_new, okd = self._dual_step(JtJ2, g0, emod, lam * float(np.mean(dsc)), nu, kmod)
                if x is None or not okd:
                    lam *= 4.0
                    continue
                Qn = self.Q_of(x[None], Bk, Bp)[0]
                rn, en = self.eval_Q(Qn)
                fn = float(rn @ rn)
                gmax = float(np.max(en / self.eps0 - 1.0))
                if kap is not None and gmax > kap * (1 + 1e-3) and ncorr < 6:
                    kmod -= 1.2 * (gmax - kap)               # second-order correction of the constraint level
                    ncorr += 1
                    continue
                if fn < fc and (kap is None or gmax <= kap * (1 + 1e-3)):
                    accepted = True
                    break
                lam *= 4.0
                ncorr = 0
                kmod = kap
            if not accepted:
                break
            nu = nu_new
            rel = (fc - fn) / fc
            Qc, rc, ec, fc = Qn, rn, en, fn
            lam = max(lam / 3.0, 1e-8)
            hist.append(fc)
            if verbose:
                print(f'   sqp it{it:3d} f={fc:.5e} lam={lam:.1e} gmax={gmax:+.3e} nu={np.array2string(nu, precision=2)} rel={rel:.1e}', flush=True)
            if rel < tol:
                stall += 1
                if stall >= 3:
                    break
            else:
                stall = 0
        # final exact feasibility (up to the 2% slack of the line search): scale back along the geodesic-free ray in the
        # last chart is impossible after re-centring, so interpolate the projector coefficients with the start instead
        g_f = ec / self.eps0 - 1.0
        if kap is not None and np.max(g_f) > kap:
            lo_, hi_ = 0.0, 1.0
            Bk0, Bp0 = complete_basis(self.Q0)
            # coordinates of span(Qc) in the chart of Q0 (valid while Bk0^dag Qc is invertible)
            Cc = (Bp0.conj().T @ Qc) @ np.linalg.inv(Bk0.conj().T @ Qc)
            xc = np.concatenate([Cc.real.ravel(), Cc.imag.ravel()]) if self.cplx else Cc.real.ravel()
            for _ in range(40):
                mid = 0.5 * (lo_ + hi_)
                Qm = self.Q_of((mid * xc)[None], Bk0, Bp0)[0]
                _, em = self.eval_Q(Qm)
                if float(np.max(em / self.eps0 - 1.0)) <= kap:
                    lo_ = mid
                else:
                    hi_ = mid
            Qc = self.Q_of((lo_ * xc)[None], Bk0, Bp0)[0]
            rc, ec = self.eval_Q(Qc)
            fc = float(rc @ rc)
        info = dict(f0=f0, f=fc, nit=nit, nev=self.nev, hist=hist, g_f=(ec / self.eps0 - 1.0).tolist())
        return Qc, info


# ------------------------------------------------------------------------------------------ arms driver
class Cell:
    """one (model, N, b, target set): exact data, scorer, and the arms at a given chi"""

    def __init__(self, model, N, b, tset, sh):
        self.model, self.N, self.b, self.ts, self.sh = model, N, b, tset, sh
        self.g = Geom(N, b)
        self.sc = Scorer(self.g, tset, sh)
        self.Ms = self.sc.Ms
        self._ops = None
        self._wnear = [w for w in self.g.wins]

    # --- fast E_near (near windows only), used inside the oracle searches
    def e_near_fast(self, Mts):
        g, sc = self.g, self.sc
        if not hasattr(self, '_tau_win'):
            self._tau_win = [{k: np.stack([g.win_rdm(sc.rho_ex[f], s, k) for (s, kk) in g.wins if kk == k]) for k in g.ks}
                             for f in range(len(sc.rho_ex))]
        worst = 0.0
        for f, Mt in enumerate(Mts):
            rho = g.region_rho(Mt)
            for k in g.ks:
                R = np.stack([g.win_rdm(rho, s, k) for (s, kk) in g.wins if kk == k]) - self._tau_win[f][k]
                R = 0.5 * (R + np.swapaxes(R.conj(), -1, -2))
                worst = max(worst, 0.5 * float(np.abs(np.linalg.eigvalsh(R)).sum(axis=-1).max()))
        return worst

    def stack_Q(self, w, chi):
        U, s = stack_basis(self.Ms, w)
        return U[:, :chi], U

    def oracle_grid(self, chi):
        """arm (ii): plan grid + the literature weights; oracle pick on E_near.  returns (label, w, Q, E, table)"""
        cand = [('lit', self.ts.w_lit)] + [(lab, np.asarray(w, float)) for lab, w in self.ts.grid]
        best, tab = None, []
        for lab, w in cand:
            w = np.asarray(w, float) / np.sum(w)
            Q, _ = self.stack_Q(w, chi)
            E = self.e_near_fast(project(self.Ms, Q))
            tab.append((lab, w.tolist(), E))
            if best is None or E < best[3]:
                best = (lab, w, Q, E)
        return best[0], best[1], best[2], best[3], tab

    def oracle_dense(self, chi, nsamp=200, seed=0):
        """arm (ii+): denser oracle over the whole weight simplex (random Dirichlet + vertices + Nelder-Mead polish)"""
        from scipy.optimize import minimize
        F = self.ts.F
        if F == 1:
            Q, _ = self.stack_Q(np.ones(1), chi)
            return np.ones(1), Q, self.e_near_fast(project(self.Ms, Q))
        rng = np.random.default_rng(seed)

        def ev(w):
            w = np.asarray(w, float)
            w = np.abs(w) / np.sum(np.abs(w))
            Q, _ = self.stack_Q(np.maximum(w, 1e-9), chi)
            return self.e_near_fast(project(self.Ms, Q)), w, Q

        cands = [np.eye(F)[i] * 0.97 + 0.03 / F for i in range(F)] + [np.ones(F) / F, self.ts.w_lit] + \
                list(rng.dirichlet(np.ones(F), nsamp)) + list(rng.dirichlet(0.4 * np.ones(F), nsamp // 2))
        best = None
        for w in cands:
            E, w2, Q = ev(w)
            if best is None or E < best[0]:
                best = (E, w2, Q)
        res = minimize(lambda z: ev(np.exp(z))[0], np.log(np.maximum(best[1], 1e-6)), method='Nelder-Mead',
                       options=dict(maxiter=120, xatol=1e-3, fatol=1e-9))
        E2, w2, Q2 = ev(np.exp(res.x))
        if E2 < best[0]:
            best = (E2, w2, Q2)
        return best[1], best[2], best[0]

    def dressed(self, chi, a):
        if self._ops is None:
            self._ops = left_pauli_ops(self.g)
        return arm_dressed(self.Ms, self.ts.w_lit, chi, a, self.g, self._ops)

    def all_arms(self, chi, kappas=(0.1,), dressed_a=(1e-3, 1e-2, 1e-1), dressed_ext=(0.3, 1.0), maxit=40, log=None):
        """returns dict arm -> dict(Q=..., metrics=..., extra=...); Q is a shared (dimL,chi) basis or a list for arm v"""
        sc, ts = self.sc, self.ts
        out = {}
        Qi, _ = self.stack_Q(ts.w_lit, chi)
        out['i'] = dict(Q=Qi, m=sc.score(project(self.Ms, Qi)))
        lab, w, Q, E, tab = self.oracle_grid(chi)
        out['ii'] = dict(Q=Q, m=sc.score(project(self.Ms, Q)), label=lab, w=w.tolist(), table=tab)
        w2, Q2, E2 = self.oracle_dense(chi)
        out['ii_plus'] = dict(Q=Q2, m=sc.score(project(self.Ms, Q2)), w=w2.tolist())
        tabd = {}
        bestd = None
        for a in tuple(dressed_a) + tuple(dressed_ext):
            Qd = self.dressed(chi, a)
            m = sc.score(project(self.Ms, Qd))
            tabd[a] = m['E_near']
            if a in dressed_a and (bestd is None or m['E_near'] < bestd[0]):
                bestd = (m['E_near'], a, Qd, m)
        out['iii'] = dict(Q=bestd[2], m=bestd[3], a=bestd[1], table=tabd)
        # supplementary: best over the extended grid
        a_ext = min(tabd, key=tabd.get)
        out['iii_ext'] = dict(a=a_ext, E_near=tabd[a_ext])
        # (v) separate SVDs
        Mv = arm_separate(self.Ms, chi)
        out['v'] = dict(Q=None, m=sc.score(Mv))
        # (iv) spcf-multi
        for kap in kappas:
            spc = SPCMulti(self.g, self.Ms, sc.hv_ex, Qi, cplx=ts.cplx, kappa=kap)
            t0 = time.time()
            Qn, info = spc.run(maxit=maxit)
            info['time'] = time.time() - t0
            info.pop('hist', None)
            name = 'iv' if kap == 0.1 else ('iv_u' if kap is None else f'iv_k{kap:g}')
            out[name] = dict(Q=Qn, m=sc.score(project(self.Ms, Qn)), info=info)
        return out

    # --- A0 gate quantities of the SVD (arm i) cut
    def gate(self, chi):
        sc, ts = self.sc, self.ts
        Qi, _ = self.stack_Q(ts.w_lit, chi)
        Mt = project(self.Ms, Qi)
        eps = eps_of(self.Ms, Mt)
        bres = np.array(sc.bres(Mt))
        cost = np.array([gate_cost(e) for e in eps])
        F = ts.F
        w = ts.w_lit
        eq = dict(tail=float(eps.mean()), bres=float(np.sqrt(np.mean(bres ** 2))))
        lw = dict(tail=float(np.sum(w * eps)), bres=float(np.sqrt(np.sum(w * bres ** 2))))
        for d in (eq, lw):
            d['cost'] = gate_cost(d['tail'])
            d['ratio'] = d['bres'] / d['cost']
        return dict(eps=eps.tolist(), bres=bres.tolist(), cost=cost.tolist(), ratio_t=(bres / cost).tolist(),
                    eq=eq, lw=lw, ratio_max=float(np.max(bres / cost)), ratio_min=float(np.min(bres / cost)))


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, complex) or isinstance(o, np.complexfloating):
        return [float(np.real(o)), float(np.imag(o))]
    return o


def run_a1(model, N, b, chis=(4, 6, 8), sets=None, kappas=(0.1,), out_dir=None, tag='', maxit=40, which=('S1', 'S2', 'S3', 'S4')):
    """A0 gate quantities and A1 arms for one (model, cut).  Writes JSON (metrics) and NPZ (shared bases)."""
    out_dir = Path(out_dir or HERE / 'results')
    out_dir.mkdir(exist_ok=True)
    tsets, sh = make_sets(model, N, which=which)
    if sets is not None:
        tsets = [t for t in tsets if t.name in sets]
    rec = dict(model=model, N=N, b=b, chis=list(chis), kappas=[k for k in kappas], cells=[], sh_E=sh['E'].tolist(),
               omega=sh.get('omega'))
    bases = {}
    fn = out_dir / f'a1_{model}_N{N}_b{b}{tag}'
    t00 = time.time()
    for ts in tsets:
        cell = Cell(model, N, b, ts, sh)
        for chi in chis:
            t0 = time.time()
            arms = cell.all_arms(chi, kappas=kappas, maxit=maxit)
            gate = cell.gate(chi)
            c = dict(set=ts.name, chi=chi, gate=gate, meta={k: v for k, v in ts.meta.items() if k not in ('Aop',)})
            for name, a in arms.items():
                d = {k: v for k, v in a.items() if k != 'Q'}
                c[name] = d
                if a.get('Q') is not None:
                    bases[f'{ts.name}|{chi}|{name}'] = a['Q']
            c['time'] = time.time() - t0
            rec['cells'].append(c)
            i_, iv_ = arms['i']['m'], arms['iv']['m']
            print(f'{model} b={b} {ts.name:5s} chi={chi}: E_near i {i_["E_near"]:.3e} ii {arms["ii"]["m"]["E_near"]:.3e} '
                  f'ii+ {arms["ii_plus"]["m"]["E_near"]:.3e} iii {arms["iii"]["m"]["E_near"]:.3e} iv {iv_["E_near"]:.3e} '
                  f'v {arms["v"]["m"]["E_near"]:.3e} | iv/i {iv_["E_near"] / i_["E_near"]:.2f} | {time.time() - t0:.0f}s', flush=True)
            json.dump(to_jsonable(rec), open(str(fn) + '.json', 'w'))
            np.savez_compressed(str(fn) + '_bases.npz', **bases)
    rec['wall'] = time.time() - t00
    json.dump(to_jsonable(rec), open(str(fn) + '.json', 'w'))
    return rec


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['a1'])
    ap.add_argument('--model', default='ising')
    ap.add_argument('--N', type=int, default=12)
    ap.add_argument('--b', type=int, default=5)
    ap.add_argument('--chis', default='4,6,8')
    ap.add_argument('--sets', default=None)
    ap.add_argument('--kappas', default='0.1')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ks = [None if k == 'inf' else float(k) for k in a.kappas.split(',')]
    run_a1(a.model, a.N, a.b, tuple(int(x) for x in a.chis.split(',')), sets=a.sets.split(',') if a.sets else None,
           kappas=ks, tag=a.tag, out_dir=a.out)
