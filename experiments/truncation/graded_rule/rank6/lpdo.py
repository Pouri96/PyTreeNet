"""LPDO-TEBD toolkit with dephasing Lindblad noise (rank 6 first move).

Object.  A locally purified density operator rho = X X^dag.  Site tensor A_j[l, p, k, r]: bond l, physical p (dim 2), Kraus k
(dim K_j, varies by site), bond r.  Trace rho = <X|X>.  The Kraus legs are traced, so the PHYSICAL region reduced density matrices
are quadratic in the tensors.

Dynamics (one operation list shared by the LPDO driver, the dense Lindblad reference and the dense Kraus-only purification, so that
only truncation error is measured, as in mps_bench.py).  Per full step of length dt:
    right sweep:  noise(0); for b = 0..N-2: gate(b, R), noise(b+1)
    left sweep :  noise(N-1); for b = N-2..0: gate(b, L), noise(b)
with noise(j) = dephasing channel of duration dt/2 on site j, consecutive noise ops on one site merged.  Every site is dephased for
a total time dt per step, and the noise always acts on the orthogonality centre (after a R cut the centre is b+1, after a L cut it
is b), so the Kraus truncation done right after it is the environment-gauged (Guo-Yang style) Frobenius-optimal one.
Gates are exactly the gates of mpsenh (expm(-i dt/2 h_b), applied R then L), acting on the physical leg only.
Dephasing generator L_j = sqrt(gamma) Z_j: channel of duration tau is rho -> (1-p) rho + p Z rho Z with p = (1 - exp(-2 gamma tau))/2.

Truncations.
    bond  : SVD of theta.reshape(l*2*K_b, 2*K_{b+1}*r) to rank mpsenh._rank(s, chi)   (with K=1 this is mpsenh.svd_cut)
    Kraus : after a noise op the Kraus leg is K -> 2K; SVD of the centre tensor reshaped (l*2*r, 2K), keep min(kappa, numerical rank).
Optional probe hook is called at every bond cut before the truncation is applied (gate-quantity logging, see gatelog.py).
"""
import _p6  # noqa: F401
import time
import numpy as np
import mpsenh as M

PAUL = M.PAUL
KRAUS_TOL = 1e-12          # relative singular-value cut for the numerical Kraus rank
Z2 = np.array([1.0, -1.0])


# ----------------------------------------------------------------------------------------------------- operation list
def schedule(N, nsteps):
    ops = []
    for _ in range(nsteps):
        ops.append(['noise', 0, 1])
        for b in range(N - 1):
            ops.append(['gate', b, 'R'])
            ops.append(['noise', b + 1, 1])
        ops.append(['noise', N - 1, 1])
        for b in range(N - 2, -1, -1):
            ops.append(['gate', b, 'L'])
            ops.append(['noise', b, 1])
    out = []
    for op in ops:
        if op[0] == 'noise' and out and out[-1][0] == 'noise' and out[-1][1] == op[1]:
            out[-1][2] += op[2]
        else:
            out.append(list(op))
    return out


def dephasing_p(gamma, dt, m):
    return 0.5 * (1.0 - np.exp(-2.0 * gamma * m * dt / 2.0))


# ----------------------------------------------------------------------------------------------------- LPDO driver
def initial_lpdo(model, N):
    return [t.reshape(t.shape[0], 2, 1, t.shape[2]).astype(complex) for t in M.initial_mps(model, N)]


def kraus_dephase_truncate(A, pn, kappa, tol=KRAUS_TOL):
    """Apply the dephasing channel to the centre tensor A (l,2,K,r): K -> 2K (index k*2+q), then truncate the Kraus leg to kappa.
    Returns the new tensor (normalised) and the discarded weight fraction."""
    l, p, K, r = A.shape
    A0 = np.sqrt(1.0 - pn) * A
    A1 = np.sqrt(pn) * A * Z2[None, :, None, None]
    Ap = np.stack([A0, A1], axis=3).reshape(l, p, 2 * K, r)
    Mk = Ap.transpose(0, 1, 3, 2).reshape(l * p * r, 2 * K)
    U, s, Vh = np.linalg.svd(Mk, full_matrices=False)
    kk = max(1, min(kappa, int(np.sum(s > tol * s[0]))))
    tot = float(np.sum(s ** 2))
    disc = float(np.sum(s[kk:] ** 2)) / tot
    sk = s[:kk]
    new = (U[:, :kk] * (sk / np.linalg.norm(sk))).reshape(l, p, r, kk).transpose(0, 1, 3, 2)
    return np.ascontiguousarray(new), disc


def run_lpdo(model, N, chi, kappa, nsteps, dt, gamma, probe=None, gates=None, kraus_tol=KRAUS_TOL):
    """LPDO-TEBD.  chi = bond cap, kappa = Kraus cap.  Returns (T, info)."""
    Gs = gates or M.make_gates(model, N, dt)
    T = initial_lpdo(model, N)
    info = dict(bond_disc=0.0, kraus_disc=0.0, kraus_disc_max=0.0, bond_disc_max=0.0, ncut=0, maxK=1, maxchi=1)
    t0 = time.time()
    if probe is not None:
        probe.start(T, N)
    for op in schedule(N, nsteps):
        if op[0] == 'gate':
            b, dirn = op[1], op[2]
            th = np.tensordot(T[b], T[b + 1], axes=([3], [0]))                   # l s k t m r
            th = np.einsum('abst,lsktmr->lakbmr', Gs[b], th)
            l, K1, K2, r = th.shape[0], th.shape[2], th.shape[4], th.shape[5]
            U, s, Vh = np.linalg.svd(th.reshape(l * 2 * K1, 2 * K2 * r), full_matrices=False)
            k = M._rank(s, chi)
            if probe is not None:
                probe(T, th, U, s, Vh, k, chi, b, dirn)
            nr2 = float(np.sum(s ** 2))
            d = float(np.sum(s[k:] ** 2)) / nr2
            info['bond_disc'] += d
            info['bond_disc_max'] = max(info['bond_disc_max'], d)
            info['ncut'] += 1
            s = s[:k]
            nrm = np.linalg.norm(s)
            U, Vh = U[:, :k], Vh[:k]
            if dirn == 'R':
                T[b], T[b + 1] = U.reshape(l, 2, K1, k), ((s[:, None] / nrm) * Vh).reshape(k, 2, K2, r)
            else:
                T[b], T[b + 1] = (U * (s / nrm)).reshape(l, 2, K1, k), Vh.reshape(k, 2, K2, r)
            info['maxchi'] = max(info['maxchi'], k)
        elif gamma > 0:
            j = op[1]
            T[j], d = kraus_dephase_truncate(T[j], dephasing_p(gamma, dt, op[2]), kappa, kraus_tol)
            info['kraus_disc'] += d
            info['kraus_disc_max'] = max(info['kraus_disc_max'], d)
            info['maxK'] = max(info['maxK'], T[j].shape[2])
    info['wall'] = time.time() - t0
    info['Ks'] = [t.shape[2] for t in T]
    info['bonds'] = [t.shape[3] for t in T[:-1]]
    info['params'] = int(sum(t.size for t in T))
    return T, info


# ----------------------------------------------------------------------------------------------------- dense conversions
def lpdo_to_rho(T):
    """Dense physical density matrix (2^N x 2^N) of an LPDO: split at the middle bond, trace the Kraus legs half by half."""
    N = len(T)
    h = N // 2
    Z = T[0][0]                                             # p k m
    Z = Z.reshape(2, Z.shape[1], Z.shape[2])                # x c m
    for j in range(1, h):
        Y = np.einsum('xcm,mpkn->xpckn', Z, T[j])
        a, b_, c, d, e = Y.shape
        Z = Y.reshape(a * b_, c * d, e)
    P, c, m = Z.shape
    mat = Z.transpose(0, 2, 1).reshape(P * m, c)
    GL = (mat @ mat.conj().T).reshape(P, m, P, m)
    Zr = T[N - 1][:, :, :, 0]                               # m p k
    Zr = Zr.reshape(Zr.shape[0], 2, Zr.shape[2])            # m x c
    for j in range(N - 2, h - 1, -1):
        Y = np.einsum('mpkn,nxc->mpxkc', T[j], Zr)
        a, b_, c, d, e = Y.shape
        Zr = Y.reshape(a, b_ * c, d * e)
    m2, Pr, cr = Zr.shape
    matr = Zr.reshape(m2 * Pr, cr)
    GR = (matr @ matr.conj().T).reshape(m2, Pr, m2, Pr)
    rho = np.tensordot(GL, GR, axes=([1, 3], [0, 2]))       # x y z w
    rho = rho.transpose(0, 2, 1, 3).reshape(P * Pr, P * Pr)
    return rho


def _apply_left(G4, rho, b, N):
    a = 2 ** b
    return np.matmul(G4, rho.reshape(a, 4, -1)).reshape(rho.shape)


def _gate_rho(rho, G, b, N):
    G4 = G.reshape(4, 4)
    X = _apply_left(G4, rho, b, N)
    return _apply_left(G4, X.conj().T, b, N).conj().T


def _zsign(N, j):
    return 1.0 - 2.0 * ((np.arange(2 ** N) >> (N - 1 - j)) & 1)


def dense_lindblad(model, N, nsteps, dt, gamma, gates=None):
    """Exact (no truncation) density matrix under the same operation list."""
    Gs = gates or M.make_gates(model, N, dt)
    psi = M.mps_to_dense(M.initial_mps(model, N))
    rho = np.outer(psi, psi.conj())
    zs = [_zsign(N, j) for j in range(N)]
    for op in schedule(N, nsteps):
        if op[0] == 'gate':
            rho = _gate_rho(rho, Gs[op[1]], op[1], N)
        elif gamma > 0:
            pn = dephasing_p(gamma, dt, op[2])
            rho = (1 - pn) * rho + pn * rho * np.outer(zs[op[1]], zs[op[1]])
    return rho


# ----------------------------------------------------------------------------------------------------- metrics on rho
_LET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def rdm(rho, N, sites):
    r = rho.reshape([2] * (2 * N))
    row = [_LET[i] for i in range(N)]
    col = [_LET[N + i] if i in sites else _LET[i] for i in range(N)]
    out = [row[i] for i in sites] + [col[i] for i in sites]
    d = 2 ** len(sites)
    return np.einsum(''.join(row + col) + '->' + ''.join(out), r).reshape(d, d)


def local_obs_rho(rho, N, model):
    nn = [rdm(rho, N, (i, i + 1)) for i in range(N - 1)]
    nnn = [rdm(rho, N, (i, i + 2)) for i in range(N - 2)]
    single = []
    for i in range(N):
        r = nn[i] if i < N - 1 else nn[N - 2]
        r = r.reshape(2, 2, 2, 2)
        r1 = np.einsum('abcb->ac', r) if i < N - 1 else np.einsum('abad->bd', r)
        single += [np.trace(r1 @ P).real for P in PAUL]
    nnv = [np.trace(r @ np.kron(P, Q)).real for r in nn for P in PAUL for Q in PAUL]
    nnnv = [np.trace(r @ np.kron(P, Q)).real for r in nnn for P in PAUL for Q in PAUL]
    E = sum(np.trace(nn[b] @ M.h_bond(model, b, N)).real for b in range(N - 1))
    dg = np.diag(rho).real
    far = {}
    for dist in (3, 4):
        far[dist] = np.array([float(np.sum(dg * _zsign(N, i) * _zsign(N, i + dist))) for i in range(N - dist)])
    return dict(single=np.array(single), nn=np.array(nnv), nnn=np.array(nnnv), E=E, far3=far[3], far4=far[4])


def errors_rho(rho_ex, rho_ap, N, model, obs_ex=None):
    a = obs_ex if obs_ex is not None else local_obs_rho(rho_ex, N, model)
    b = local_obs_rho(rho_ap, N, model)
    o = {k + '_rms': float(np.sqrt(np.mean((a[k] - b[k]) ** 2))) for k in ('single', 'nn', 'nnn')}
    o['E_abs'] = float(abs(a['E'] - b['E']))
    o['far3_rms'] = float(np.sqrt(np.mean((a['far3'] - b['far3']) ** 2)))
    o['far4_rms'] = float(np.sqrt(np.mean((a['far4'] - b['far4']) ** 2)))
    d = rho_ap - rho_ex
    d = 0.5 * (d + d.conj().T)
    ev = np.linalg.eigvalsh(d)
    o['trace_norm'] = float(np.sum(np.abs(ev)))
    o['hs_norm'] = float(np.linalg.norm(d))
    o['trace_rho'] = float(np.trace(rho_ap).real)
    o['purity'] = float(np.trace(rho_ap @ rho_ap).real)
    o['lam_min'] = float(np.linalg.eigvalsh(0.5 * (rho_ap + rho_ap.conj().T))[0])
    return o


# ----------------------------------------------------------------------------------------------------- dense Kraus-only purification
def dense_kraus_only(model, N, nsteps, dt, gamma, kappa, gates=None, tol=KRAUS_TOL):
    """chi = infinity: the full purification psi[P, K_0..K_{N-1}] is evolved with the same operation list; after each noise op the
    Kraus leg of that site is truncated to kappa by the global Frobenius-optimal projector.  Returns (rho, info)."""
    Gs = gates or M.make_gates(model, N, dt)
    psi0 = M.mps_to_dense(M.initial_mps(model, N))
    psi = psi0.reshape(2 ** N, 1).astype(complex)
    Ks = [1] * N
    zs = [_zsign(N, j) for j in range(N)]
    info = dict(kraus_disc=0.0, kraus_disc_max=0.0)
    t0 = time.time()
    for op in schedule(N, nsteps):
        if op[0] == 'gate':
            b = op[1]
            G4 = Gs[b].reshape(4, 4)
            psi = np.matmul(G4, psi.reshape(2 ** b, 4, -1)).reshape(psi.shape)
        elif gamma > 0:
            j = op[1]
            pn = dephasing_p(gamma, dt, op[2])
            KL, Kj, KR = int(np.prod(Ks[:j], dtype=np.int64)), Ks[j], int(np.prod(Ks[j + 1:], dtype=np.int64))
            psi4 = psi.reshape(2 ** N, KL, Kj, KR)
            # Gram of the enlarged Kraus leg without forming it:  G'[(k,q),(k',q')] = sum_P e_q(P) conj(e_q'(P)) Gp[P,k,k']
            Gp = np.empty((2 ** N, Kj, Kj), dtype=complex)
            for P in range(2 ** N):
                m = psi4[P].transpose(1, 0, 2).reshape(Kj, KL * KR)
                Gp[P] = m @ m.conj().T
            sgn = zs[j]
            e0 = np.sqrt(1 - pn)
            e1 = np.sqrt(pn) * sgn
            Gb = np.zeros((Kj, 2, Kj, 2), dtype=complex)
            Gb[:, 0, :, 0] = (e0 * e0) * Gp.sum(0)
            Gb[:, 0, :, 1] = (e0 * (e1[:, None, None] * Gp).sum(0))
            Gb[:, 1, :, 0] = (e0 * (e1[:, None, None] * Gp).sum(0))
            Gb[:, 1, :, 1] = (e1 * e1)[:, None, None].__mul__(Gp).sum(0)
            Gm = Gb.reshape(2 * Kj, 2 * Kj)
            Gm = 0.5 * (Gm + Gm.conj().T)
            lam, V = np.linalg.eigh(Gm)
            lam, V = lam[::-1], V[:, ::-1]
            lam = np.clip(lam, 0.0, None)
            kk = max(1, min(kappa, int(np.sum(np.sqrt(lam) > tol * np.sqrt(lam[0])))))
            disc = float(np.sum(lam[kk:]) / np.sum(lam))
            info['kraus_disc'] += disc
            info['kraus_disc_max'] = max(info['kraus_disc_max'], disc)
            Vk = V[:, :kk].reshape(Kj, 2, kk)                      # [(k,q), kk]
            nrm = np.sqrt(np.sum(lam[:kk]))
            # new[P, kk, ...] = sum_{k,q} conj(Vk[k,q,kk]) e_q(P) psi[P, k]; two sign sectors of the physical bit j
            Vc = Vk.conj()
            out = np.empty((2 ** N, KL, kk, KR), dtype=complex)
            bit = ((np.arange(2 ** N) >> (N - 1 - j)) & 1)
            for sb, sv in ((0, 1.0), (1, -1.0)):
                Wm = np.einsum('kqx,q->xk', Vc, np.array([e0, np.sqrt(pn) * sv])) / nrm       # (kk, Kj)
                idx = np.nonzero(bit == sb)[0]
                sub = psi4[idx]                                    # (n, KL, Kj, KR)
                out[idx] = np.einsum('xk,nakb->naxb', Wm, sub, optimize=True)
            psi = out.reshape(2 ** N, -1)
            Ks[j] = kk
    rho = psi @ psi.conj().T
    info['wall'] = time.time() - t0
    info['Ks'] = list(Ks)
    return rho, info
