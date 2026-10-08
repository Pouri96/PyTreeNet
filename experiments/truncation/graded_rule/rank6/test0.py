"""Test 0: correctness gates for the rank-6 first move (everything that does not involve the tilt).
    python test0.py            -> results/test0.json (+ printed PASS/FAIL lines)
"""
import _p6  # noqa: F401
import json
import sys
import time
import numpy as np
import scipy.linalg as sla
import mpsenh as M
import spcfast
import lpdo as L
import gatelog as G

out = {}
fails = []


def check(name, ok, **kw):
    out[name] = dict(ok=bool(ok), **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in kw.items()})
    print(('PASS ' if ok else 'FAIL ') + name, kw, flush=True)
    if not ok:
        fails.append(name)


model, dt = 'ising', 0.1

# ---- 0.1 dense Lindblad at gamma = 0 equals the pure reference |psi><psi| (the dense_reference of mpsenh)
for N, T in ((8, 2.0), (8, 4.0)):
    ns = int(round(T / dt))
    psi = M.dense_reference(model, N, ns, dt)
    rho0 = L.dense_lindblad(model, N, ns, dt, 0.0)
    d = np.abs(rho0 - np.outer(psi, psi.conj())).max()
    check(f'0.1 dense Lindblad(gamma=0) == |psi><psi|  N={N} T={T}', d < 1e-12, maxabs=d)

# ---- 0.2 the dissipator: channel convention against the exact Lindbladian exponential of one qubit, and Trotter convergence on a chain
Zm = M.Z
Lsup = np.zeros((4, 4), dtype=complex)
g = 0.37
for a in range(4):
    E = np.zeros(4, dtype=complex)
    E[a] = 1
    r = E.reshape(2, 2)
    Lsup[:, a] = (g * (Zm @ r @ Zm - r)).reshape(-1)
rin = np.array([[0.6, 0.2 - 0.3j], [0.2 + 0.3j, 0.4]])
tau = 0.8
rex = (sla.expm(Lsup * tau) @ rin.reshape(-1)).reshape(2, 2)
pn = L.dephasing_p(g, 2 * tau, 1)             # m=1 means duration dt/2 -> choose dt = 2 tau
rch = (1 - pn) * rin + pn * Zm @ rin @ Zm
check('0.2a dephasing channel == exp(L tau) for L(rho)=gamma(Z rho Z - rho)', np.abs(rex - rch).max() < 1e-13, maxabs=np.abs(rex - rch).max())

# full-model Lindblad generator on N=4, exact exponential vs the Strang operation list at dt = 0.2, 0.1, 0.05
N4, Tt, gam = 4, 1.2, 0.15
H = np.zeros((2 ** N4, 2 ** N4), dtype=complex)
for bnd in range(N4 - 1):
    H += np.kron(np.kron(np.eye(2 ** bnd), M.h_bond(model, bnd, N4)), np.eye(2 ** (N4 - bnd - 2)))
# h_bond already contains the single-site fields on the left site of every bond (and the right site of the last): check Hamiltonian = sum_b h_b
Zs = [np.kron(np.kron(np.eye(2 ** j), M.Z), np.eye(2 ** (N4 - j - 1))) for j in range(N4)]
dim = 2 ** N4
Lfull = -1j * (np.kron(H, np.eye(dim)) - np.kron(np.eye(dim), H.T))
for Zj in Zs:
    Lfull += gam * (np.kron(Zj, Zj.T) - np.eye(dim * dim))
psi = M.mps_to_dense(M.initial_mps(model, N4))
r0 = np.outer(psi, psi.conj())
rex = (sla.expm(Lfull * Tt) @ r0.reshape(-1)).reshape(dim, dim)
errs = []
for dtt in (0.2, 0.1, 0.05):
    ns = int(round(Tt / dtt))
    rr = L.dense_lindblad(model, N4, ns, dtt, gam)
    errs.append(np.abs(rr - rex).max())
check('0.2b Strang operation list converges to exact Lindblad (error ~ dt^p, p>=1.8 expected)', errs[1] / errs[2] > 3.0 and errs[0] / errs[1] > 3.0, errs=[float(e) for e in errs], order=float(np.log2(errs[1] / errs[2])))

# ---- 0.3 properties of the dense reference at gamma > 0
N, ns = 8, 40
for gam in (0.01, 0.1):
    rr = L.dense_lindblad(model, N, ns, dt, gam)
    ev = np.linalg.eigvalsh(0.5 * (rr + rr.conj().T))
    check(f'0.3 dense Lindblad gamma={gam}: trace 1, Hermitian, PSD', abs(np.trace(rr).real - 1) < 1e-12 and np.abs(rr - rr.conj().T).max() < 1e-13 and ev[0] > -1e-12,
          trace=np.trace(rr).real, herm=np.abs(rr - rr.conj().T).max(), lam_min=ev[0], purity=np.trace(rr @ rr).real)

# ---- 0.4 exact LPDO (large chi, kappa) == dense Lindblad;  LPDO with kappa=2, chi large == dense Kraus-only purification with kappa=2
N, ns = 6, 3
for gam in (0.0, 0.1):
    ex = L.dense_lindblad(model, N, ns, dt, gam)
    Tl, info = L.run_lpdo(model, N, 200, 64, ns, dt, gam)
    d1 = np.abs(L.lpdo_to_rho(Tl) - ex).max()
    check(f'0.4a exact LPDO == dense Lindblad  N={N} gamma={gam}', d1 < 1e-10, maxabs=d1, Ks=info['Ks'], bonds=info['bonds'])
    if gam > 0:
        for kap in (1, 2, 3):
            Tl, info = L.run_lpdo(model, N, 400, kap, 2 * ns, dt, gam)
            rd, infd = L.dense_kraus_only(model, N, 2 * ns, dt, gam, kap)
            d2 = np.abs(L.lpdo_to_rho(Tl) - rd).max()
            check(f'0.4b LPDO(chi=inf,kappa={kap}, centre-gauged) == dense global Kraus truncation  N={N} gamma={gam}', d2 < 1e-9, maxabs=d2,
                  kraus_disc_mps=info['kraus_disc'], kraus_disc_dense=infd['kraus_disc'], Ks=info['Ks'])

# ---- 0.5 gamma = 0: LPDO-SVD == MPS-SVD (mpsenh.run_tebd with svd_cut)
N, T = 8, 4.0
ns = int(round(T / dt))
Gs = M.make_gates(model, N, dt)
for chi in (6, 8, 12):
    Tm, _ = M.run_tebd(model, N, chi, ns, dt, M.svd_cut, gates=Gs)
    Tl, info = L.run_lpdo(model, N, chi, 2, ns, dt, 0.0, gates=Gs)
    bit = all(np.array_equal(Tm[j], Tl[j][:, :, 0, :]) for j in range(N))
    dmax = max(np.abs(Tm[j] - Tl[j][:, :, 0, :]).max() for j in range(N))
    vm, vl = M.mps_to_dense(Tm), M.mps_to_dense([t[:, :, 0, :] for t in Tl])
    check(f'0.5 gamma=0 LPDO-SVD == MPS-SVD  N={N} T={T} chi={chi}', dmax < 1e-12, bitwise=bit, max_tensor_diff=dmax, state_diff=np.abs(vm - vl).max())

# ---- 0.6 gate quantities of the LPDO audit == SPCFast.gate_log at gamma = 0 (K=1), a = 2 and a = 1
for chi in (6, 8):
    for av in (2, 1):
        sp = spcfast.SPCFast(model, N, a=av, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2, gate_log=True, gate=1e9)
        Tm, _ = M.run_tebd(model, N, chi, ns, dt, sp, gates=Gs)
        pr = G.GateProbe(model, N, a=(av,))
        Tl, info = L.run_lpdo(model, N, chi, 2, ns, dt, 0.0, probe=pr, gates=Gs)
        A = np.array(sp.gate_log)
        B = np.array(pr.log[av])
        same_n = A.shape == B.shape
        if same_n:
            rb = np.max(np.abs(A[:, 1] - B[:, 1]) / np.maximum(A[:, 1], 1e-30))
            rt = np.max(np.abs(A[:, 0] - B[:, 0]) / A[:, 0])
            rc = np.max(np.abs(A[:, 2] - B[:, 2]) / A[:, 2])
            same_b = bool(np.all(A[:, 3] == B[:, 3]))
        else:
            rb = rt = rc = np.inf
            same_b = False
        check(f'0.6 gate log (tail, B_res, cost, b) vs SPCFast.gate_log  N={N} chi={chi} a={av}', same_n and same_b and rb < 1e-4 and rt < 1e-9 and rc < 1e-9,
              n_spcf=len(A), n_mine=len(B), max_rel_B=rb, max_rel_tail=rt, max_rel_cost=rc)

# ---- 0.7 region RDM with Kraus legs traced == partial trace of the full dense rho (K > 1 states, canonical-form environments)
class RegionCheck:
    def __init__(self, N, a):
        self.N, self.a, self.worst, self.n = N, a, 0.0, 0

    def start(self, T, N):
        pass

    def __call__(self, T, th, U, s, Vh, k, chi, b, dirn):
        N, a = self.N, self.a
        if self.n > 400 or (self.n % 7):
            self.n += 1
            return
        self.n += 1
        l, K1, K2, r = th.shape[0], th.shape[2], th.shape[4], th.shape[5]
        Tp = list(T)
        Tp[b] = U.reshape(l, 2, K1, len(s))
        Tp[b + 1] = ((s[:, None]) * Vh).reshape(len(s), 2, K2, r)
        rho = L.lpdo_to_rho(Tp)
        lo, hi = max(0, b - a), min(N, b + 2 + a)
        ref = L.rdm(rho, N, tuple(range(lo, hi)))
        GL, GR = G.left_gram(T, lo, b, l), G.right_gram(T, b, hi, r)
        mine = G.region_rho(GL, GR, th)
        self.worst = max(self.worst, np.abs(mine - ref).max() / max(np.abs(ref).max(), 1e-300))


N = 8
for gam, kap, chi, av in ((0.1, 4, 8, 2), (0.1, 4, 8, 1), (0.03, 3, 6, 2)):
    rc = RegionCheck(N, av)
    L.run_lpdo(model, N, chi, kap, 12, dt, gam, probe=rc)
    check(f'0.7 region rho (Kraus traced, Gram form) == partial trace of dense rho  gamma={gam} kappa={kap} chi={chi} a={av}', rc.worst < 1e-12, worst_rel=rc.worst, n=rc.n)

# ---- 0.8 positivity by construction + trace normalisation of the LPDO
Tl, info = L.run_lpdo(model, 8, 8, 4, 40, dt, 0.1)
rl = L.lpdo_to_rho(Tl)
ev = np.linalg.eigvalsh(0.5 * (rl + rl.conj().T))
check('0.8 LPDO rho: trace 1, PSD (lam_min >= -1e-14), Hermitian', abs(np.trace(rl).real - 1) < 1e-12 and ev[0] > -1e-14 and np.abs(rl - rl.conj().T).max() < 1e-14,
      trace=np.trace(rl).real, lam_min=ev[0])

json.dump(out, open('results/test0.json', 'w'), indent=1)
print('\nTEST 0:', 'ALL PASS' if not fails else f'FAILED: {fails}')
sys.exit(1 if fails else 0)
