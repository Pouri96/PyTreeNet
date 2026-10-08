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
    X = np.concatenate([np.sqrt(wi) * Mi for wi, Mi in zip(w, Ms)], axis=1)
    U, s, _ = np.linalg.svd(X, full_matrices=True)
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


# ------------------------------------------------------------------------------------------ spcf-multi: dense GN / LM
class SPCMulti:
    """Shared-C Gauss-Newton on the Grassmannian chart Q = qr(B_k + B_perp C).  Residual vector
    r = (h(rho(Q Q^dag M_f)) - h(rho(M_f)))_f over all targets (equal weights), Jacobian by forward finite differences.
    Optional constraint: eps_f(Q) <= (1 + kappa) eps_f(Q_stack) for every target (squared hinge, then a final scale-back)."""

    def __init__(self, geom, Ms, hv_ex, Bk, Bp, cplx, kappa=0.1, h=1e-6, chunk=96):
        self.g, self.Ms, self.hv_ex = geom, Ms, hv_ex
        self.Bk, self.Bp = Bk, Bp
        self.nb, self.chi = Bp.shape[1], Bk.shape[1]
        self.cplx = cplx
        self.kappa, self.h, self.chunk = kappa, h, chunk
        self.npar = self.nb * self.chi * (2 if cplx else 1)
        self.F = len(Ms)
        self.eps0 = np.maximum(eps_of(Ms, project(Ms, Bk)), 1e-14)
        self.nev = 0
        self.nrm2 = np.array([np.linalg.norm(m) ** 2 for m in Ms])

    def to_C(self, X):
        X = np.atleast_2d(X)
        n = self.nb * self.chi
        if self.cplx:
            return (X[:, :n] + 1j * X[:, n:]).reshape(len(X), self.nb, self.chi)
        return X.reshape(len(X), self.nb, self.chi)

    def Q_of(self, X):
        C = self.to_C(X)
        Y = self.Bk[None] + self.Bp[None] @ C
        Q, _ = np.linalg.qr(Y)
        return Q

    def evaluate(self, X):
        """X (n, npar) -> r (n, F*nres), eps (n, F)"""
        X = np.atleast_2d(X)
        rs, es = [], []
        for i0 in range(0, len(X), self.chunk):
            Q = self.Q_of(X[i0:i0 + self.chunk])
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

    def aug(self, X, mu):
        r, e = self.evaluate(X)
        if self.kappa is None:
            return r, e, np.zeros((len(r), 0))
        hinge = np.sqrt(mu) * np.maximum(0.0, e / self.eps0[None] - 1.0 - self.kappa)
        return r, e, hinge

    def merit(self, r, hinge):
        return float(np.sum(r ** 2) + np.sum(hinge ** 2))

    def jac(self, x, mu, r0, hinge0):
        X = np.tile(x, (self.npar, 1)) + self.h * np.eye(self.npar)
        r1, e1, h1 = self.aug(X, mu)
        J = np.concatenate([(r1 - r0[None]) / self.h, (h1 - hinge0[None]) / self.h], axis=1).T
        return J

    def run_ls(self, mu_factors=(1e3, 1e5), max_nfev=150, method='trf'):
        """same problem through scipy least_squares (trust region, scaled by the Jacobian columns); staged hinge weight,
        final scale-back for exact feasibility"""
        from scipy.optimize import least_squares
        x = np.zeros(self.npar)
        r, e, hg = self.aug(x[None], 0.0)
        f0 = float(np.sum(r ** 2))
        mus = mu_factors if self.kappa is not None else (0.0,)
        nfev = 0
        hist = [f0]
        cache = {}

        def fun(xx, mu):
            r_, e_, h_ = self.aug(xx[None], mu)
            cache['last'] = (xx.copy(), r_[0], h_[0])
            return np.concatenate([r_[0], h_[0]])

        def jac(xx, mu):
            lx, r_, h_ = cache['last']
            if not np.array_equal(lx, xx):
                fun(xx, mu)
                lx, r_, h_ = cache['last']
            return self.jac(xx, mu, r_, h_)

        for mf in mus:
            mu = mf * f0
            sol = least_squares(fun, x, jac=jac, args=(mu,), method=method, x_scale='jac', tr_solver='exact',
                                ftol=1e-10, xtol=1e-10, gtol=1e-12, max_nfev=max_nfev)
            x = sol.x
            nfev += sol.nfev
            rr, ee, _ = self.aug(x[None], 0.0)
            hist.append(float(np.sum(rr[0] ** 2)))
        shrink = 1.0
        if self.kappa is not None:
            r, e, _ = self.aug(x[None], 0.0)
            viol = float(np.max(e[0] / self.eps0 - 1.0))
            if viol > self.kappa:
                lo_, hi_ = 0.0, 1.0
                for _ in range(40):
                    mid = 0.5 * (lo_ + hi_)
                    _, em, _ = self.aug((mid * x)[None], 0.0)
                    if float(np.max(em[0] / self.eps0 - 1.0)) <= self.kappa:
                        lo_ = mid
                    else:
                        hi_ = mid
                shrink = lo_
                x = shrink * x
        r, e, _ = self.aug(x[None], 0.0)
        info = dict(f0=f0, f=float(np.sum(r[0] ** 2)), nit=nfev, nev=self.nev, shrink=shrink, hist=hist,
                    g_f=(e[0] / self.eps0 - 1.0).tolist(), status=int(sol.status))
        return x, info

    def run(self, maxit=30, mu_factors=(1e3, 1e5), tol=1e-4, lam0=1e-3, verbose=False, x0=None):
        x = np.zeros(self.npar) if x0 is None else x0.copy()
        r, e, hg = self.aug(x[None], 0.0)
        f0 = float(np.sum(r ** 2))
        hist = [f0]
        lam = lam0
        mus = mu_factors if self.kappa is not None else (0.0,)
        nit = 0
        for mf in mus:
            mu = mf * f0 / 1.0
            r, e, hg = self.aug(x[None], mu)
            r0, hg0 = r[0], hg[0]
            phi = self.merit(r0, hg0)
            stall = 0
            for it in range(maxit):
                nit += 1
                J = self.jac(x, mu, r0, hg0)
                rv = np.concatenate([r0, hg0])
                g = J.T @ rv
                A = J.T @ J
                dA = np.diag(A).copy()
                dA = np.maximum(dA, 1e-12 * dA.max() + 1e-300)
                ok = False
                for _ in range(14):
                    try:
                        step = -np.linalg.solve(A + lam * np.diag(dA), g)
                    except np.linalg.LinAlgError:
                        lam *= 10
                        continue
                    xn = x + step
                    rn, en, hn = self.aug(xn[None], mu)
                    phin = self.merit(rn[0], hn[0])
                    if phin < phi:
                        ok = True
                        break
                    lam *= 4.0
                if not ok:
                    break
                rel = (phi - phin) / phi
                x, r0, hg0, phi = xn, rn[0], hn[0], phin
                lam = max(lam / 3.0, 1e-9)
                hist.append(float(np.sum(r0 ** 2)))
                if verbose:
                    print(f'   it{nit:3d} f={hist[-1]:.4e} merit={phi:.4e} lam={lam:.1e} rel={rel:.2e}', flush=True)
                if rel < tol:
                    stall += 1
                    if stall >= 2:
                        break
                else:
                    stall = 0
        # exact feasibility: scale the chart coordinates back along the ray until every target is within kappa
        shrink = 1.0
        if self.kappa is not None:
            r, e, _ = self.aug(x[None], 0.0)
            viol = float(np.max(e[0] / self.eps0 - 1.0))
            if viol > self.kappa:
                lo_, hi_ = 0.0, 1.0
                for _ in range(40):
                    mid = 0.5 * (lo_ + hi_)
                    _, em, _ = self.aug((mid * x)[None], 0.0)
                    if float(np.max(em[0] / self.eps0 - 1.0)) <= self.kappa:
                        lo_ = mid
                    else:
                        hi_ = mid
                shrink = lo_
                x = shrink * x
        r, e, _ = self.aug(x[None], 0.0)
        info = dict(f0=f0, f=float(np.sum(r[0] ** 2)), nit=nit, nev=self.nev, shrink=shrink, hist=hist,
                    g_f=(e[0] / self.eps0 - 1.0).tolist())
        return x, info
