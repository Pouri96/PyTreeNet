"""Purification-MPS TEBD toolkit (fused site = (physical qubit, ancilla), d = 4, index order (p, a)).

* initial states: products of tilted Bell pairs  psi_i = sqrt(p_up)|up up> + sqrt(p_dn)|dn dn>, p_up = e^{mu_i}/(2 cosh mu_i),
  so rho_0 = prod_i e^{mu_i Z_i}/Z.  Families: ``stag`` mu_i = (-1)^i mu (mu = inf is the Neel state, the ``ising`` cell) and ``dw`` mu_i = +mu on
  the left half, -mu on the right half (mu = inf is ``isingdw``).
* gauges: ``plain`` gate G x 1_anc;  ``back`` (Karrasch-Bardarson-Moore) gate G x G^*.  Physical reduced density matrices do not depend on the gauge.
* d-generic copies of mpsenh.svd_cut / run_tebd / mps_to_dense, dense reference with snapshots, metrics on the PHYSICAL density matrix.
"""
import os
import time
import numpy as np
import scipy.linalg as sla
from pathlib import Path
import mpsenh as M

HERE = Path(__file__).resolve().parent
REF = HERE / '_refcache'
D = 4
PAUL = M.PAUL


def mu_list(N, family, mu):
    if family == 'stag':
        return [mu * (-1) ** i for i in range(N)]
    if family == 'dw':
        return [mu if i < N // 2 else -mu for i in range(N)]
    raise ValueError(family)


def site_vec(mui):
    """(p, a) amplitudes of the tilted Bell pair (index 0 = up = Z +1, 1 = down)."""
    if np.isinf(mui):
        pu = 1.0 if mui > 0 else 0.0
    else:
        pu = np.exp(mui) / (2 * np.cosh(mui))
    v = np.zeros(4, dtype=complex)
    v[0] = np.sqrt(pu)          # |up, up>
    v[3] = np.sqrt(1 - pu)      # |dn, dn>
    return v


def initial_pur_mps(N, family, mu):
    return [site_vec(m).reshape(1, 4, 1) for m in mu_list(N, family, mu)]


def fuse_gate(G, gauge):
    """G (p1', p2', p1, p2) -> (4,4,4,4) acting on fused (p, a) sites; ancilla identity (plain) or G^* (back)."""
    if gauge == 'plain':
        Ga = np.einsum('ac,bd->abcd', np.eye(2), np.eye(2)).astype(complex)
    elif gauge == 'back':
        Ga = G.conj()
    else:
        raise ValueError(gauge)
    return np.einsum('pqrs,abcd->paqbrcsd', G, Ga).reshape(4, 4, 4, 4)


def pur_gates(model, N, dt, gauge):
    return [fuse_gate(G, gauge) for G in M.make_gates(model, N, dt)]


# ------------------------------------------------------------------ d-generic MPS pieces
def svd_cut_d(theta, chi, dirn):
    l, d, r = theta.shape[0], theta.shape[1], theta.shape[3]
    Mm = theta.reshape(l * d, d * r)
    U, s, Vh = np.linalg.svd(Mm, full_matrices=False)
    k = M._rank(s, chi)
    s = s[:k]
    nrm = np.linalg.norm(s)
    U, Vh = U[:, :k], Vh[:k]
    if dirn == 'R':
        return U.reshape(l, d, k), ((s[:, None] / nrm) * Vh).reshape(k, d, r), 0.0, False
    return (U * (s / nrm)).reshape(l, d, k), Vh.reshape(k, d, r), 0.0, False


def run_tebd_d(T0, Gs, chi, nsteps, cut):
    """Strang sweeps exactly as mpsenh.run_tebd, any local dimension.  cut: svd_cut_d or an SPCFPur (called as cut(theta, chi, dirn, A, B, b))."""
    N = len(T0)
    T = [t.copy() for t in T0]
    if hasattr(cut, 'start'):
        cut.start(T)
    t0 = time.time()
    for _ in range(nsteps):
        for b in range(N - 1):
            th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', Gs[b], th)
            A = T[b - 1] if b >= 1 else None
            B = T[b + 2] if b + 2 <= N - 1 else None
            if cut is svd_cut_d:
                T[b], T[b + 1], _, _ = svd_cut_d(th, chi, 'R')
            else:
                T[b], T[b + 1], _, _ = cut(th, chi, 'R', A, B, b)
        for b in range(N - 2, -1, -1):
            th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
            th = np.einsum('abst,lstr->labr', Gs[b], th)
            A = T[b - 1] if b >= 1 else None
            B = T[b + 2] if b + 2 <= N - 1 else None
            if cut is svd_cut_d:
                T[b], T[b + 1], _, _ = svd_cut_d(th, chi, 'L')
            else:
                T[b], T[b + 1], _, _ = cut(th, chi, 'L', A, B, b)
    return T, time.time() - t0


def mps_to_dense_d(T):
    d = T[0].shape[1]
    v = T[0].reshape(d, -1)
    for t in T[1:]:
        c = t.shape[0]
        v = (v @ t.reshape(c, -1)).reshape(-1, t.shape[2])
    return v.reshape(-1)


def stored_params(T):
    return int(sum(t.size for t in T))


def ranks(T):
    return [t.shape[2] for t in T[:-1]]


# ------------------------------------------------------------------ dense reference
def dense_gate_d(v, G, b, N):
    t = v.reshape(4 ** b, 16, 4 ** (N - b - 2))
    return np.matmul(G.reshape(16, 16), t).reshape(-1)


def initial_dense(N, family, mu):
    v = np.ones(1, dtype=complex)
    for m in mu_list(N, family, mu):
        v = np.kron(v, site_vec(m))
    return v


def dense_snapshots(model, N, family, mu, gauge, dt, Ts):
    """Evolve the dense purification with the same fused Trotter gates; return {T: psi} at the steps T/dt."""
    Gs = pur_gates(model, N, dt, gauge)
    v = initial_dense(N, family, mu)
    steps = {int(round(T / dt)): T for T in Ts}
    out = {}
    for step in range(1, max(steps) + 1):
        for b in range(N - 1):
            v = dense_gate_d(v, Gs[b], b, N)
        for b in range(N - 2, -1, -1):
            v = dense_gate_d(v, Gs[b], b, N)
        if step in steps:
            out[steps[step]] = v.copy()
    return out


def mu_tag(mu):
    return 'inf' if np.isinf(mu) else f'{mu:g}'


def reference_pur(model, N, family, mu, gauge, T, dt=0.1, Ts=(2.0, 3.0, 4.0)):
    """Cached dense purification vector at time T (the cache holds all of Ts from one trajectory)."""
    REF.mkdir(exist_ok=True)
    path = REF / f'pur_{model}_N{N}_{family}_mu{mu_tag(mu)}_{gauge}_T{T:g}_dt{dt}.npy'
    if path.exists():
        return np.load(path)
    Tall = sorted(set(Ts) | {T})
    snaps = dense_snapshots(model, N, family, mu, gauge, dt, Tall)
    for Tk, v in snaps.items():
        np.save(REF / f'pur_{model}_N{N}_{family}_mu{mu_tag(mu)}_{gauge}_T{Tk:g}_dt{dt}.npy', v)
    return snaps[T]


# ------------------------------------------------------------------ physical density matrix and metrics
def psi_to_rho(psi, N):
    """Physical rho = Tr_anc |psi><psi| (2^N x 2^N) from the fused vector with site order (p0 a0 p1 a1 ...)."""
    t = psi.reshape([2, 2] * N)
    t = t.transpose(list(range(0, 2 * N, 2)) + list(range(1, 2 * N, 2))).reshape(2 ** N, 2 ** N)
    return t @ t.conj().T


def rdm_k(rho, N, i, k):
    R = rho.reshape(2 ** i, 2 ** k, 2 ** (N - i - k), 2 ** i, 2 ** k, 2 ** (N - i - k))
    return np.einsum('asbatb->st', R)


class RDMSet:
    """All k-site marginals (k = 1..5) of a normalised physical rho, computed once."""
    def __init__(self, rho, N, kmax=5):
        self.N = N
        tr = np.trace(rho).real
        self.tr = tr
        rho = rho / tr
        self.r = {k: [rdm_k(rho, N, i, k) for i in range(N - k + 1)] for k in range(1, kmax + 1)}


def _ev(r, op):
    return float(np.trace(r @ op).real)


def obs_from_rdms(S, model):
    N = S.N
    nn = S.r[2]
    nnn = []
    for i in range(N - 2):
        R3 = S.r[3][i].reshape(2, 2, 2, 2, 2, 2)                        # (s0 s1 s2; s0' s1' s2')
        nnn.append(np.einsum('abcdbf->acdf', R3).reshape(4, 4))          # trace the middle site
    single = []
    for i in range(N):
        rho1 = S.r[1][i]
        single += [_ev(rho1, P) for P in PAUL]
    nnv = [_ev(r, np.kron(P, Q)) for r in nn for P in PAUL for Q in PAUL]
    nnnv = [_ev(r, np.kron(P, Q)) for r in nnn for P in PAUL for Q in PAUL]
    E = sum(_ev(nn[b], M.h_bond(model, b, N)) for b in range(N - 1))
    out = dict(single=np.array(single), nn=np.array(nnv), nnn=np.array(nnnv), E=E)
    I2 = np.eye(2)
    for rr in (3, 4):
        zz, p9 = [], []
        for i in range(N - rr):
            Rk = S.r[rr + 1][i]
            for P in PAUL:
                for Q in PAUL:
                    ops = P
                    for _ in range(rr - 1):
                        ops = np.kron(ops, I2)
                    ops = np.kron(ops, Q)
                    v = _ev(Rk, ops)
                    p9.append(v)
                    if P is PAUL[2] and Q is PAUL[2]:
                        zz.append(v)
        out[f'zz{rr}'] = np.array(zz)
        out[f'far{rr}'] = np.array(p9)
    return out


def _rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def marg_err(Sa, Sb, k):
    e = [np.linalg.norm(Sa.r[k][i] - Sb.r[k][i]) ** 2 for i in range(Sa.N - k + 1)]
    return float(np.sqrt(np.mean(e)))


class Reference:
    """Dense reference for one cell: exact psi, rho, marginals, observables."""
    def __init__(self, model, N, family, mu, gauge, T, dt=0.1):
        self.model, self.N = model, N
        self.psi = reference_pur(model, N, family, mu, gauge, T, dt)
        self.psi = self.psi / np.linalg.norm(self.psi)
        self.rho = psi_to_rho(self.psi, N)
        self.S = RDMSet(self.rho, N)
        self.obs = obs_from_rdms(self.S, model)


def pur_metrics(ref, Tm, full=True):
    """Errors of the MPS Tm (purification) against the Reference, all on the physical density matrix."""
    N, model = ref.N, ref.model
    psi = mps_to_dense_d(Tm)
    nrm = np.linalg.norm(psi)
    psi = psi / nrm
    rho = psi_to_rho(psi, N)
    S = RDMSet(rho, N)
    ob = obs_from_rdms(S, model)
    e = {k + '_rms': _rms(ref.obs[k], ob[k]) for k in ('single', 'nn', 'nnn')}
    e['E_abs'] = float(abs(ref.obs['E'] - ob['E']))
    e['rdm2'] = marg_err(ref.S, S, 2)
    e['rdm3'] = marg_err(ref.S, S, 3)
    e['zz3'], e['zz4'] = _rms(ref.obs['zz3'], ob['zz3']), _rms(ref.obs['zz4'], ob['zz4'])
    e['zzfar'] = float(np.sqrt(np.mean(np.concatenate([(ref.obs['zz3'] - ob['zz3']) ** 2, (ref.obs['zz4'] - ob['zz4']) ** 2]))))
    e['far3'], e['far4'] = _rms(ref.obs['far3'], ob['far3']), _rms(ref.obs['far4'], ob['far4'])
    e['infid_pur'] = float(1 - abs(np.vdot(ref.psi, psi)) ** 2)
    if full:
        w = np.linalg.eigvalsh(rho - ref.rho)
        e['tdist'] = float(0.5 * np.sum(np.abs(w)))
        e['lam_min'] = float(np.linalg.eigvalsh(rho)[0])
    e['norm'] = float(nrm)
    return e
