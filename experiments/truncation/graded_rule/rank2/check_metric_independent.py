"""Independent check of the physical-error metric used in the purification benches (the heis control asked for it):
 (1) reference: the dense physical rho evolved directly (G rho G^dag on the 2^N x 2^N matrix, no ancilla at all) against Reference.rho;
 (2) approximate: the 2-site physical marginals of a spcf purification MPS from transfer matrices (never forming a dense state) against the dense route of purlib.
    python check_metric_independent.py model family mu T chi
"""
import sys
import _p2  # noqa: F401
import numpy as np
import mpsenh as M
import purlib as P
from pur_bench import make_cut

model, fam, mu, T, chi = sys.argv[1], sys.argv[2], (np.inf if sys.argv[3] == 'inf' else float(sys.argv[3])), float(sys.argv[4]), int(sys.argv[5])
N, dt = 10, 0.1
n = int(round(T / dt))
# (1) direct dense rho
rho = np.ones((1, 1), dtype=complex)
for m in P.mu_list(N, fam, mu):
    pu = 1.0 if np.isinf(m) and m > 0 else (0.0 if np.isinf(m) else np.exp(m) / (2 * np.cosh(m)))
    rho = np.kron(rho, np.diag([pu, 1 - pu]))
Gs = M.make_gates(model, N, dt)


def apply(rho, G, b):
    t = rho.reshape(2 ** b, 4, 2 ** (N - b - 2), 2 ** b, 4, 2 ** (N - b - 2))
    G4 = G.reshape(4, 4)
    t = np.einsum('ij,ajbckd->aibckd', G4, t)
    t = np.einsum('aibckd,lk->aibcld', t, G4.conj())
    return t.reshape(2 ** N, 2 ** N)


for _ in range(n):
    for b in range(N - 1):
        rho = apply(rho, Gs[b], b)
    for b in range(N - 2, -1, -1):
        rho = apply(rho, Gs[b], b)
ref = P.Reference(model, N, fam, mu, 'plain', T)
print('(1) direct dense rho vs purification-derived reference rho: max abs diff', np.abs(rho - ref.rho).max())
# (2) transfer-matrix 2-site marginals of the spcf purification MPS
cut = make_cut('spcf', model, N)
Tm, _ = P.run_tebd_d(P.initial_pur_mps(N, fam, mu), P.pur_gates(model, N, dt, 'plain'), chi, n, cut)
psi = P.mps_to_dense_d(Tm)
psi = psi / np.linalg.norm(psi)
Sd = P.RDMSet(P.psi_to_rho(psi, N), N)


def marg2(Tm, i):
    L = np.ones((1, 1), dtype=complex)
    for t in Tm[:i]:
        L = np.einsum('xy,xpar,ypaq->rq', L, t.conj().reshape(t.shape[0], 2, 2, -1), t.reshape(t.shape[0], 2, 2, -1))
    Rr = np.ones((1, 1), dtype=complex)
    for t in Tm[i + 2:][::-1]:
        Rr = np.einsum('xpar,ypaq,rq->xy', t.conj().reshape(t.shape[0], 2, 2, -1), t.reshape(t.shape[0], 2, 2, -1), Rr)
    A, B = Tm[i].reshape(Tm[i].shape[0], 2, 2, -1), Tm[i + 1].reshape(Tm[i + 1].shape[0], 2, 2, -1)
    # rho[(p1 p2),(p1' p2')] = sum L[x,y] conj(A[x,p1',a1,m]) A[y,p1,a1,n] conj(B[m,p2',a2,r]) B[n,p2,a2,s] R[r,s]
    r = np.einsum('xy,xqam,ypan,mkbr,nlbs,rs->plqk', L, A.conj(), A, B.conj(), B, Rr)
    return r.reshape(4, 4)


worst = max(np.abs(marg2(Tm, i) / np.trace(marg2(Tm, i)).real - Sd.r[2][i]).max() for i in range(N - 1))
print('(2) transfer-matrix 2-site marginals vs dense-route marginals, max abs diff', worst)
e_tm = np.sqrt(np.mean([np.linalg.norm(marg2(Tm, i) / np.trace(marg2(Tm, i)).real - rho_ref) ** 2
                        for i, rho_ref in enumerate(P.RDMSet(rho / np.trace(rho).real, N).r[2])]))
e_pm = P.pur_metrics(ref, Tm)['rdm2']
print(f'    rdm2 from transfer matrices against the directly evolved dense rho: {e_tm:.6e};  pur_metrics rdm2: {e_pm:.6e}')
