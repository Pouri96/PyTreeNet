"""MPS-native TEBD sweep (Strang, mixed canonical) with two cut rules:

  svd   plain Eckart-Young truncation of the two-site tensor
  enh   window-graded trust-region cut

The enhanced cut works on the two-site tensor theta (l,2,2,r) with a left-isometric neighbour A
and a right-isometric neighbour B, so the window reduced density matrix over sites
[b-1, b, b+1, b+2] is  rho = W W^dag  with  W = A theta B.  Nothing is dense and nothing is
enumerated over Pauli strings.

Objective (verified equal to the 111-string graded sum in verify_window.py)
    f(rho) = gamma 2^k || D_{gamma^-1/2}(rho - tau) ||_F^2  +  wE (Tr[H_W (rho - tau)])^2
with tau the window RDM of the UNTRUNCATED theta and D a single-site depolarising map.

Step: steepest-descent direction d = -A^dag(B^dag(F W)) with F = c D^2(rho - tau) + wE dE H_W,
projected on the tangent space of the rank-chi manifold at the SVD point, step length t chosen on a
grid inside the trust region ||M - M_svd||^2 <= kappa * eps (eps = discarded weight), retraction is
the FACTORED product (U + tZ)(S V^dag + t U^dag d), exactly rank chi, followed by one QR.
kappa = 0 or eps < eps_gate reproduces the SVD cut bit for bit.
"""
import time
import numpy as np
import scipy.linalg as sla

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
HX, HZ = 0.9045, 0.8090


FIELDS = {'ising': (HX, HZ), 'isingdw': (HX, HZ), 'ising2': (1.4, 0.4)}


def h_bond(model, b, N):
    if model in FIELDS:
        hx, hz = FIELDS[model]
        h = np.kron(Z, Z) + np.kron(hx * X + hz * Z, I2)
        if b == N - 2:
            h = h + np.kron(I2, hx * X + hz * Z)
        return h
    if model == 'heis':
        return sum(np.kron(P, P) for P in (X, Y, Z))
    raise ValueError(model)


def make_gates(model, N, dt):
    return [sla.expm(-1j * (dt / 2) * h_bond(model, b, N)).reshape(2, 2, 2, 2) for b in range(N - 1)]


def neel_mps(N):
    return [np.array([1, 0] if i % 2 == 0 else [0, 1], dtype=complex).reshape(1, 2, 1) for i in range(N)]


def domain_wall_mps(N):
    return [np.array([1, 0] if i < N // 2 else [0, 1], dtype=complex).reshape(1, 2, 1) for i in range(N)]


def initial_mps(model, N):
    return domain_wall_mps(N) if model == 'isingdw' else neel_mps(N)


# ------------------------------------------------------------------ dense reference helpers
def mps_to_dense(T):
    v = T[0].reshape(2, -1)
    for t in T[1:]:
        c = t.shape[0]
        v = (v @ t.reshape(c, -1)).reshape(-1, t.shape[2])
    return v.reshape(-1)


def dense_gate(v, G, b, N):
    t = v.reshape(2 ** b, 4, 2 ** (N - b - 2))
    return np.einsum('ij,ajb->aib', G.reshape(4, 4), t).reshape(-1)


def rdm2(v, N, i, j):
    t = np.moveaxis(v.reshape([2] * N), [i, j], [0, 1]).reshape(4, -1)
    return t @ t.conj().T


PAUL = [X, Y, Z]


def local_obs(v, N, model):
    v = v / np.linalg.norm(v)
    nn = [rdm2(v, N, i, i + 1) for i in range(N - 1)]
    nnn = [rdm2(v, N, i, i + 2) for i in range(N - 2)]
    single = []
    for i in range(N):
        r = nn[i] if i < N - 1 else nn[N - 2]
        r = r.reshape(2, 2, 2, 2)
        rho = np.einsum('abcb->ac', r) if i < N - 1 else np.einsum('abad->bd', r)
        single += [np.trace(rho @ P).real for P in PAUL]
    nnv = [np.trace(r @ np.kron(P, Q)).real for r in nn for P in PAUL for Q in PAUL]
    nnnv = [np.trace(r @ np.kron(P, Q)).real for r in nnn for P in PAUL for Q in PAUL]
    E = sum(np.trace(nn[b] @ h_bond(model, b, N)).real for b in range(N - 1))
    return dict(single=np.array(single), nn=np.array(nnv), nnn=np.array(nnnv), E=E)


def errors(ex, ap, N, model):
    a, b = local_obs(ex, N, model), local_obs(ap, N, model)
    o = {k + '_rms': float(np.sqrt(np.mean((a[k] - b[k]) ** 2))) for k in ('single', 'nn', 'nnn')}
    o['E_abs'] = float(abs(a['E'] - b['E']))
    pe, pa = ex / np.linalg.norm(ex), ap / np.linalg.norm(ap)
    o['infid'] = float(1 - abs(np.vdot(pe, pa)) ** 2)
    return o


# ------------------------------------------------------------------ window machinery
_DSUP = {}


def _depol_apply(Mat, dims, mu):
    n = len(dims)
    out = Mat.reshape(dims + dims)
    for k in range(n):
        ax = (k, n + k)
        T = np.moveaxis(out, ax, (0, 1))
        tr = np.trace(T, axis1=0, axis2=1)
        T = mu * T
        T[0, 0] += (1 - mu) * tr / 2
        T[1, 1] += (1 - mu) * tr / 2
        out = np.moveaxis(T, (0, 1), ax)
    D = int(np.prod(dims))
    return out.reshape(D, D)


def depol_super(dims, mu):
    """Superoperator matrix (D^2 x D^2) acting on row-major vec of a D x D matrix."""
    key = (dims, round(mu, 12))
    if key not in _DSUP:
        D = int(np.prod(dims))
        S = np.zeros((D * D, D * D), dtype=complex)
        for a in range(D * D):
            E = np.zeros(D * D, dtype=complex)
            E[a] = 1.0
            S[:, a] = _depol_apply(E.reshape(D, D), dims, mu).reshape(-1)
        _DSUP[key] = S
    return _DSUP[key]


def embed2(h4, i0, k):
    return np.kron(np.kron(np.eye(2 ** i0), h4), np.eye(2 ** (k - i0 - 2)))


class Window:
    """Everything about one bond's window that does not depend on the state."""
    def __init__(self, model, N, b, hasA, hasB, gamma):
        self.dsL = 2 if hasA else 1
        self.dsR = 2 if hasB else 1
        self.dims = tuple([2] * (int(hasA) + 2 + int(hasB)))
        self.D = int(np.prod(self.dims))
        self.k = len(self.dims)
        self.gamma = gamma
        self.cW = gamma * 2.0 ** self.k
        i0 = int(hasA)
        HW = embed2(h_bond(model, b, N), i0, self.k)
        if hasA:
            HW = HW + embed2(h_bond(model, b - 1, N), 0, self.k)
        if hasB:
            HW = HW + embed2(h_bond(model, b + 1, N), i0 + 1, self.k)
        self.HW = HW
        self.S_obj = depol_super(self.dims, 1.0 / gamma)   # D^dag D = D_{1/gamma}


def make_wide(A, B, win):
    D = win.D

    def wide(Th):
        Yt = np.tensordot(A, Th, axes=([2], [0]))
        W = np.tensordot(Yt, B, axes=([4], [0]))
        return W.transpose(1, 2, 3, 4, 0, 5).reshape(D, -1)

    def back(Ym):
        ll, rr = A.shape[0], B.shape[2]
        Yt = Ym.reshape(win.dsL, 2, 2, win.dsR, ll, rr).transpose(4, 0, 1, 2, 3, 5)
        Zt = np.tensordot(A.conj(), Yt, axes=([0, 1], [0, 1]))
        return np.tensordot(Zt, B.conj(), axes=([3, 4], [1, 2]))

    return wide, back


def objective(rho, tau, win, wE):
    d = rho - tau
    dd = (win.S_obj @ d.reshape(-1)).reshape(win.D, win.D)
    fW = win.cW * float(np.vdot(d, dd).real)
    dE = float(np.trace(win.HW @ d).real)
    return fW + wE * dE * dE, dd, dE


# ------------------------------------------------------------------ cuts
EPS_KEEP = 1e-13


TOL = 0.0


def _rank(s, chi):
    k = min(chi, int(np.sum(s > EPS_KEEP * s[0])))
    if TOL > 0:
        tail = np.cumsum((s ** 2)[::-1])[::-1]          # tail[j] = sum_{i>=j} s_i^2
        kt = int(np.argmax(tail <= TOL * tail[0])) if np.any(tail <= TOL * tail[0]) else len(s)
        k = min(k, max(kt, 1))
    return k


def svd_cut(theta, chi, dirn):
    l, r = theta.shape[0], theta.shape[3]
    M = theta.reshape(l * 2, 2 * r)
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    k = _rank(s, chi)
    s = s[:k]
    nrm = np.linalg.norm(s)
    U, Vh = U[:, :k], Vh[:k]
    if dirn == 'R':
        return U.reshape(l, 2, k), ((s[:, None] / nrm) * Vh).reshape(k, 2, r), 0.0, False
    return (U * (s / nrm)).reshape(l, 2, k), Vh.reshape(k, 2, r), 0.0, False


_GRIDS = {}


def _grids(m, ngrid, refine):
    """Unit-scale line-search points: the coarse log grid, and the refinement offsets (fractions of the best point)."""
    key = (m, ngrid, refine)
    if key not in _GRIDS:
        gridv = np.concatenate([[0.0], np.logspace(0, -2.5, ngrid)])
        if m == 1:
            base = gridv[:, None]
        else:
            t1, t2 = np.meshgrid(gridv, gridv, indexing='ij')
            base = np.stack([t1.ravel(), t2.ravel()], axis=1)
        offs, w = [], 0.5
        ax = np.linspace(-1.0, 1.0, 9)
        for _ in range(refine):
            mesh = np.meshgrid(*([1 + w * ax] * m), indexing='ij')
            offs.append(np.stack([x.ravel() for x in mesh], axis=1))
            w *= 0.45
        _GRIDS[key] = (base, offs)
    return _GRIDS[key]


class EnhCut:
    def __init__(self, model, N, kappa=0.1, gamma=4.0, wE=100.0, eps_gate=0.0, otol=0.0, ngrid=24,
                 check=False, ndir=1, gate_rel=0.0, window='r2', energy_only=False,
                 energy='window', refine=0):
        self.window, self.energy_only = window, energy_only
        self.refine = refine
        self.henv = None
        if energy == 'global':
            import mpsenv
            self.henv = mpsenv.HEnv(model, N)
        self.model, self.N = model, N
        self.kappa, self.gamma, self.wE = kappa, gamma, wE
        self.eps_gate, self.otol, self.ngrid, self.check = eps_gate, otol, ngrid, check
        self.ndir, self.gate_rel = ndir, gate_rel
        self.decay = 0.5 ** (1.0 / N)
        self.emax = 0.0
        self.wins = {}
        self.fired = 0
        self.truncating = 0
        self.log = []

    def win(self, b, hasA, hasB):
        key = (b, hasA, hasB)
        if key not in self.wins:
            w = Window(self.model, self.N, b, hasA, hasB, self.gamma)
            if self.energy_only:
                w.cW = 0.0
            self.wins[key] = w
        return self.wins[key]

    def __call__(self, theta, chi, dirn, A, B, b):
        l, r = theta.shape[0], theta.shape[3]
        M = theta.reshape(l * 2, 2 * r)
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        k = _rank(s, chi)
        if k >= len(s) or np.sum(s[k:] ** 2) <= 0:
            return self._plain(U, s, Vh, k, l, r, dirn)
        eps = float(np.sum(s[k:] ** 2))
        self.truncating += 1
        gate = self.eps_gate
        if self.gate_rel > 0:
            # discarded weight grows with time, so a running MEAN is always below the newest cut and
            # gates nothing.  The gate is relative to a decayed running MAXIMUM instead (half-life N cuts).
            gate = max(gate, self.gate_rel * self.emax)
            self.emax = max(eps, self.emax * self.decay)
        if self.kappa <= 0 or eps < gate:
            return self._plain(U, s, Vh, k, l, r, dirn)
        out = self._enhanced(theta, U, s, Vh, k, eps, l, r, dirn, A, B, b)
        if out is None:
            return self._plain(U, s, Vh, k, l, r, dirn)
        self.fired += 1
        return out

    @staticmethod
    def _plain(U, s, Vh, k, l, r, dirn):
        s = s[:k]
        nrm = np.linalg.norm(s)
        U, Vh = U[:, :k], Vh[:k]
        if dirn == 'R':
            return U.reshape(l, 2, k), ((s[:, None] / nrm) * Vh).reshape(k, 2, r), 0.0, False
        return (U * (s / nrm)).reshape(l, 2, k), Vh.reshape(k, 2, r), 0.0, False

    def _enhanced(self, theta, U, s, Vh, k, eps, l, r, dirn, A, B, b):
        if self.window == 'bond':
            # window = the two sites of theta: no neighbour contractions at all
            win = self.win(b, False, False)

            def wide(Th):
                return Th.transpose(1, 2, 0, 3).reshape(4, -1)

            def back(Ym):
                return Ym.reshape(2, 2, l, r).transpose(2, 0, 1, 3)
        else:
            hasA, hasB = A is not None, B is not None
            win = self.win(b, hasA, hasB)
            Ae = A if hasA else np.eye(l, dtype=complex).reshape(l, 1, l)
            Be = B if hasB else np.eye(r, dtype=complex).reshape(r, 1, r)
            wide, back = make_wide(Ae, Be, win)
        glob = self.henv is not None
        Uk, Vk = U[:, :k], Vh[:k].conj().T                  # (L,k), (R,k)
        nrm = np.sqrt(1.0 - eps)
        S = s[:k] / nrm
        Wt = wide(theta)
        tau = Wt @ Wt.conj().T
        Ms = (Uk * S) @ Vh[:k]
        W0 = wide(Ms.reshape(l, 2, 2, r))
        rho0 = W0 @ W0.conj().T
        f0, dd, dE = objective(rho0, tau, win, 0.0 if glob else self.wE)
        if glob:
            # global energy: <theta|H_eff|theta> is the energy of the whole chain
            self.henv.prep(b)
            Hth, HM = self.henv.apply_many([theta, Ms.reshape(l, 2, 2, r)])
            Eth = float(np.vdot(theta, Hth).real)
            HM = HM.reshape(l * 2, 2 * r)
            E0 = float(np.vdot(Ms, HM).real)
            dEg = E0 - Eth
            f0 = f0 + self.wE * dEg * dEg
        if f0 < self.otol ** 2:
            return None
        if glob:
            raw = []
            if win.cW > 0:
                raw.append(-back((win.cW * dd) @ W0))
            if self.wE > 0:
                raw.append(-(self.wE * dEg) * HM)
        elif self.ndir == 2 and self.wE > 0:
            raw = [-back((win.cW * dd) @ W0), -back((self.wE * dE * win.HW) @ W0)]
        else:
            raw = [-back((win.cW * dd + self.wE * dE * win.HW) @ W0)]
        gs, Uds, Zfs, dVs = [], [], [], []
        for d in raw:
            d = d.reshape(l * 2, 2 * r)
            d = d - np.vdot(Ms, d) * Ms
            Ud = Uk.conj().T @ d                              # (k,R)
            dV = d @ Vk                                       # (L,k)
            dP = dV - Uk @ (Uk.conj().T @ dV)                 # P_U^perp d V
            g = Uk @ Ud + dP @ Vk.conj().T                    # tangent projection
            if float(np.vdot(g, g).real) < 1e-300:
                continue
            gs.append(g)
            Uds.append(Ud)
            dVs.append(dV)
            Zfs.append(dP / S[None, :])
        m = len(gs)
        if m == 0:
            return None
        G = np.array([[np.vdot(gs[a], gs[c]).real for c in range(m)] for a in range(m)])
        Ws = [wide(g.reshape(l, 2, 2, r)) for g in gs]
        # rho(t) = (rho0 + sum_a t_a C_a + sum_{a<=c} t_a t_c C_ac) / (1 + t^T G t)
        Ca = []
        for a in range(m):
            C = W0 @ Ws[a].conj().T
            Ca.append(C + C.conj().T)
        Cac = {}
        for a in range(m):
            for c in range(a, m):
                if a == c:
                    Cac[(a, c)] = Ws[a] @ Ws[a].conj().T
                else:
                    C = Ws[a] @ Ws[c].conj().T
                    Cac[(a, c)] = C + C.conj().T
        if glob:
            Hg = [h.reshape(l * 2, 2 * r) for h in self.henv.apply_many([g.reshape(l, 2, 2, r) for g in gs])]
            Eg1 = np.array([np.vdot(Ms, h).real for h in Hg])
            Hgg = np.array([[np.vdot(gs[a], Hg[c]).real for c in range(m)] for a in range(m)])
        bud = self.kappa * eps
        # dlt (1+q) = V_0 + sum_a t_a V_a + sum_{a<=c} t_a t_c V_ac, so the window objective is
        # cW (phi^T Gam phi) / (1+q)^2 with phi the monomials and Gam a K x K Gram matrix, K = 1+m+m(m+1)/2.
        mons = [()]
        Vs = [(rho0 - tau).reshape(-1)]
        for a in range(m):
            mons.append((a,))
            Vs.append(Ca[a].reshape(-1))
        for a in range(m):
            for c in range(a, m):
                mons.append((a, c))
                Vs.append((Cac[(a, c)] - (G[a, a] if a == c else 2.0 * G[a, c]) * tau).reshape(-1))
        V = np.stack(Vs, axis=1)
        Gam = (V.conj().T @ (win.S_obj @ V)).real
        Gam = 0.5 * (Gam + Gam.T)
        eV = np.real(V.T @ win.HW.T.reshape(-1)) if not glob else None

        def fvals_at(pts):
            q = np.einsum('pa,ac,pc->p', pts, G, pts)
            cols = [np.ones(len(pts))]
            for mon in mons[1:]:
                col = pts[:, mon[0]]
                for a in mon[1:]:
                    col = col * pts[:, a]
                cols.append(col)
            phi = np.stack(cols, axis=1)
            fW = win.cW * np.einsum('pk,kl,pl->p', phi, Gam, phi) / (1.0 + q) ** 2
            if glob:
                En = (E0 + 2.0 * pts @ Eg1 + np.einsum('pa,ac,pc->p', pts, Hgg, pts)) / (1.0 + q)
                dEs = En - Eth
            else:
                dEs = (phi @ eV) / (1.0 + q)
            return fW + self.wE * dEs * dEs, q

        tmax = np.array([np.sqrt(bud / G[a, a]) for a in range(m)])
        base, offs = _grids(m, self.ngrid, self.refine)
        pts = base * tmax[None, :]
        fv, q = fvals_at(pts)
        fv = np.where(q <= bud, fv, np.inf)
        ib = int(np.argmin(fv))
        best_pt, best_f = pts[ib].copy(), float(fv[ib])
        for off in offs:
            if not np.any(best_pt > 0):
                break
            lp = off * best_pt[None, :]
            fl, ql = fvals_at(lp)
            fl = np.where(ql <= bud, fl, np.inf)
            j = int(np.argmin(fl))
            if fl[j] < best_f:
                best_pt, best_f = lp[j].copy(), float(fl[j])
        if best_f > f0 * (1 - 1e-9) or not np.any(best_pt > 0):
            return None
        tv = best_pt
        # Factored retraction, exactly rank k.  Right sweep: (U + tZ)(S V^dag + t U^dag d), whose left
        # factor satisfies P^dag P = I + t^2 Z^dag Z (U^dag Z = 0), so a Cholesky orthonormalisation is
        # exact and well conditioned.  Left sweep: the mirrored form (U S + t d V)(V + t Z_V)^dag.
        if dirn == 'L':
            ZVs = [(Uds[a].conj().T - Vk @ (Uds[a] @ Vk).conj().T) / S[None, :] for a in range(m)]
        for _ in range(3):
            if dirn == 'R':
                P = Uk + sum(tv[a] * Zfs[a] for a in range(m))
                Xm = S[:, None] * Vk.conj().T + sum(tv[a] * Uds[a] for a in range(m))
                n2 = float(np.trace((P.conj().T @ P) @ (Xm @ Xm.conj().T)).real)
                inner = float(np.trace(S[:, None] * (Xm @ Vk)).real)
            else:
                Pl = Uk * S[None, :] + sum(tv[a] * dVs[a] for a in range(m))
                Qr = Vk + sum(tv[a] * ZVs[a] for a in range(m))
                n2 = float(np.trace((Pl.conj().T @ Pl) @ (Qr.conj().T @ Qr)).real)
                inner = float(np.sum(S * S) + sum(tv[a] * np.trace(S[:, None] * (Uds[a] @ Vk)).real
                                                  for a in range(m)))
            D2 = n2 + 1.0 - 2.0 * inner
            if D2 <= bud * (1 + 1e-3):
                break
            tv = tv * 0.98 * np.sqrt(bud / D2)
        else:
            return None
        if self.check:
            self.log.append(dict(eps=eps, f0=f0, f_pred=best_f, t=tv.tolist(), D2=D2, n2=n2, m=m))
        if dirn == 'R':
            Lc = sla.cholesky(P.conj().T @ P, lower=True)
            Q = sla.solve_triangular(Lc, P.conj().T, lower=True).conj().T
            cen = (Lc.conj().T @ Xm) / np.sqrt(n2)
            return Q.reshape(l, 2, k), cen.reshape(k, 2, r), D2, True
        Lv = sla.cholesky(Qr.conj().T @ Qr, lower=True)
        right = sla.solve_triangular(Lv, Qr.conj().T, lower=True)
        cen = (Pl @ Lv) / np.sqrt(n2)
        return cen.reshape(l, 2, k), right.reshape(k, 2, r), D2, True


# ------------------------------------------------------------------ driver
def run_tebd(model, N, chi, nsteps, dt, cut, ref_every=0, ref=None, gates=None):
    """Strang TEBD sweeps.  Returns the final MPS and per-run timing.  `cut` is svd_cut or an EnhCut."""
    Gs = gates or make_gates(model, N, dt)
    T = initial_mps(model, N)
    henv = getattr(cut, 'henv', None)
    if henv is not None:
        henv.init(T)
    if hasattr(cut, 'start'):
        cut.start(T)
    t0 = time.time()
    for step in range(nsteps):
        for b in range(N - 1):
            th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', Gs[b], th)
            A = T[b - 1] if b >= 1 else None
            B = T[b + 2] if b + 2 <= N - 1 else None
            if cut is svd_cut:
                T[b], T[b + 1], _, _ = svd_cut(th, chi, 'R')
            else:
                T[b], T[b + 1], _, _ = cut(th, chi, 'R', A, B, b)
            if henv is not None:
                henv.after_right_cut(b, T[b])
        for b in range(N - 2, -1, -1):
            th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', Gs[b], th)
            A = T[b - 1] if b >= 1 else None
            B = T[b + 2] if b + 2 <= N - 1 else None
            if cut is svd_cut:
                T[b], T[b + 1], _, _ = svd_cut(th, chi, 'L')
            else:
                T[b], T[b + 1], _, _ = cut(th, chi, 'L', A, B, b)
            if henv is not None:
                henv.after_left_cut(b, T[b + 1])
        if hasattr(cut, 'after_step'):
            cut.after_step(T)
    return T, time.time() - t0


def dense_reference(model, N, nsteps, dt, gates=None):
    Gs = gates or make_gates(model, N, dt)
    v = mps_to_dense(initial_mps(model, N))
    for step in range(nsteps):
        for b in range(N - 1):
            v = dense_gate(v, Gs[b], b, N)
        for b in range(N - 2, -1, -1):
            v = dense_gate(v, Gs[b], b, N)
    return v


def ranks(T):
    return [t.shape[2] for t in T[:-1]]


def stored_params(T):
    return int(sum(t.size for t in T))


class PinCut:
    """An inner cut (svd_cut or an EnhCut) followed by exact preservation of the whole-chain energy.

    The centre tensor C of the truncated state is moved by C + mu g, g = (H_eff - E) C, with mu the
    smallest-modulus root of the scalar quadratic that restores the energy the two-site tensor had
    before the cut. C is a single-site tensor, so no bond changes. This is the spc_mps correction applied
    at every cut of a TEBD sweep.
    """
    def __init__(self, model, N, inner=None, tol=1e-13, mode='cut'):
        import mpsenv
        self.henv = mpsenv.HEnv(model, N)
        self.inner = inner
        self.tol = tol
        self.mode = mode          # 'cut' pins to the energy before each cut, 'step' pins to E_0 once per full step
        self.E0 = None
        self.fired = 0
        self.truncating = 0
        self.steps = 0

    def start(self, T):
        self.E0 = self.henv.full_energy(T)

    def after_step(self, T):
        if self.mode != 'step':
            return
        # the left sweep ends with the centre on site 0, R[1] up to date
        C, did = self._pin(T[0], lambda x: self.henv.apply1(x, self.henv.L[0], 0, self.henv.R[1]), self.E0)
        T[0] = C
        self.steps += 1
        self.fired += int(did)

    def _pin(self, C, h, Etarget):
        C = C / np.linalg.norm(C)
        HC = h(C)
        E = float(np.vdot(C, HC).real)
        if abs(E - Etarget) < self.tol:
            return C, False
        g = HC - E * C
        gg = float(np.vdot(g, g).real)
        if gg < 1e-300:
            return C, False
        q = float(np.vdot(g, h(g)).real) - Etarget * gg
        c0 = E - Etarget
        disc = gg * gg - q * c0
        if disc < 0:
            return C, False
        mu = -c0 / (gg + np.sqrt(disc))
        C = C + mu * g
        return C / np.linalg.norm(C), True

    def __call__(self, theta, chi, dirn, A, B, b):
        henv = self.henv
        henv.prep(b)
        th = theta / np.linalg.norm(theta)
        Eth = float(np.vdot(th, henv.apply_many([th])[0]).real)
        if self.inner is None:
            Tb, Tb1, d2, f = svd_cut(theta, chi, dirn)
        elif isinstance(self.inner, EnhCut):
            Tb, Tb1, d2, f = self.inner(theta, chi, dirn, A, B, b)
        else:
            Tb, Tb1, d2, f = self.inner(theta, chi, dirn)
        self.truncating += 1
        if self.mode != 'cut':
            return Tb, Tb1, d2, f
        if dirn == 'R':
            Lb1 = henv.up_left(henv.L[b], Tb, b)
            C, did = self._pin(Tb1, lambda x: henv.apply1(x, Lb1, b + 1, henv.R[b + 2]), Eth)
            Tb1 = C
        else:
            Rb1 = henv.up_right(henv.R[b + 2], Tb1, b + 1)
            C, did = self._pin(Tb, lambda x: henv.apply1(x, henv.L[b], b, Rb1), Eth)
            Tb = C
        self.fired += int(did)
        return Tb, Tb1, d2, f
