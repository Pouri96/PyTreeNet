"""Small-case checks of the Kronecker / whitening math used in wlib.py (no TEBD).  Results are printed and written as JSON.

    python whiten/validate_math.py whiten/results/validate_math.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy.optimize as so
import wlib as W

rng = np.random.default_rng(1)
out = {}


def crand(*sh):
    return rng.standard_normal(sh) + 1j * rng.standard_normal(sh)


def rand_pd(n, cond=1e3):
    Q, _ = np.linalg.qr(crand(n, n))
    w = np.logspace(0, -np.log10(cond), n)
    return (Q * w) @ Q.conj().T


# 1. weighted low-rank approximation with a Kronecker metric: the whitened SVD is the global minimiser.
na, nb_, k = 6, 5, 2
M = crand(na, nb_)
L, R = rand_pd(na), rand_pd(nb_)
Mk = W.weighted_lra_full(M, k, L, R)
wnorm = lambda D: np.sqrt(np.trace(D.conj().T @ L @ D @ R).real)
e_closed = wnorm(M - Mk)
rank_ok = np.linalg.matrix_rank(Mk, tol=1e-10) == k
# (a) analytic value: sqrt(sum of the discarded squared singular values of L^{1/2} M R^{1/2})
s_w = np.linalg.svd(W.herm_pow(L, .5) @ M @ W.herm_pow(R, .5), compute_uv=False)
e_theory = np.sqrt(np.sum(s_w[k:] ** 2))
# (b) never beaten by plain SVD or by 400 local optimisations (L-BFGS, random starts) of the weighted error over rank-k factors
U, s, Vh = np.linalg.svd(M)
e_svd = wnorm(M - (U[:, :k] * s[:k]) @ Vh[:k])


def obj(v):
    A = (v[:na * k] + 1j * v[na * k:2 * na * k]).reshape(na, k)
    B = (v[2 * na * k:2 * na * k + nb_ * k] + 1j * v[2 * na * k + nb_ * k:]).reshape(k, nb_)
    return wnorm(M - A @ B) ** 2


best = 1e9
for t in range(60):
    v0 = rng.standard_normal(2 * k * (na + nb_))
    r_ = so.minimize(obj, v0, method='L-BFGS-B', options=dict(maxiter=2000, ftol=1e-15, gtol=1e-10))
    best = min(best, r_.fun)
out['whitened_svd_global_min'] = dict(closed_form=float(e_closed), theory=float(e_theory), plain_svd=float(e_svd), best_of_60_lbfgs=float(np.sqrt(best)), rank_ok=bool(rank_ok),
                                       closed_equals_theory=bool(abs(e_closed - e_theory) < 1e-10), not_beaten=bool(np.sqrt(best) >= e_closed - 1e-7), plain_svd_worse=bool(e_svd >= e_closed - 1e-12))

# 2. subspace form used by the probe: Q from whitened_subspace equals the column / row space of the full weighted M_k
Q, _ = W.whitened_subspace(M, k, L, R, 'R')
P = Q @ Q.conj().T
out['subspace_R_matches_Mk'] = float(np.linalg.norm(P @ Mk - Mk))
Q2, _ = W.whitened_subspace(M, k, L, R, 'L')
out['subspace_L_matches_Mk'] = float(np.linalg.norm(Mk @ (Q2 @ Q2.conj().T) - Mk))
# mu = 0 reduces to the plain SVD subspace
Lw, Rw = W.damped_pair(L, R, 0.0)
Q0, _ = W.whitened_subspace(M, k, Lw, Rw, 'R')
out['mu0_equals_svd_subspace'] = float(np.linalg.norm(Q0 @ Q0.conj().T - U[:, :k] @ U[:, :k].conj().T))

# 3. Shampoo factors of a rank-1 gradient family recover the Kronecker form exactly (u v^H, sums over a product set)
n = 40
us, vs = crand(n, na), crand(n, nb_)
Gam_prod = np.einsum('ia,jb->ijab', us[:7], vs[:6]).reshape(-1, na, nb_)           # product set: sum_{ij} (u_i)(v_j)^H-type
Gam_prod = np.einsum('ia,jb->ijab', us[:7], vs[:6].conj()).reshape(-1, na, nb_)
G1 = W.gram_G1(Gam_prod)
G1_ref = np.einsum('iab,icd->abcd', Gam_prod, Gam_prod.conj()).reshape(na * nb_, na * nb_)
out['gram_G1_vs_einsum'] = float(np.linalg.norm(G1 - G1_ref))
Lsh, Rsh = W.shampoo(Gam_prod)
K = np.kron(Lsh, Rsh.T)
c = np.vdot(K, G1).real / np.vdot(K, K).real
out['shampoo_product_set_relerr'] = float(np.linalg.norm(G1 - c * K) / np.linalg.norm(G1))
Lv, Rv, fid = W.vlp_rank1_gamma(Gam_prod)
Kv = np.kron(Lv, Rv.T)
c = np.vdot(Kv, G1).real / np.vdot(Kv, Kv).real
out['vlp_product_set_relerr'] = float(np.linalg.norm(G1 - c * Kv) / np.linalg.norm(G1))
out['vlp_product_set_fidelity'] = fid
# generic (non-product) set: Shampoo is only approximate, VLP is the best Kronecker fit (never worse in Frobenius norm)
Gam_gen = crand(30, na, nb_) * rng.uniform(0.2, 2, (30, 1, 1))
G1g = W.gram_G1(Gam_gen)
Ls_, Rs_ = W.shampoo(Gam_gen); Kx = np.kron(Ls_, Rs_.T); cx = np.vdot(Kx, G1g).real / np.vdot(Kx, Kx).real
Lv_, Rv_, _ = W.vlp_rank1_gamma(Gam_gen); Kv_ = np.kron(Lv_, Rv_.T); cv = np.vdot(Kv_, G1g).real / np.vdot(Kv_, Kv_).real
out['generic_set_relerr'] = dict(shampoo=float(np.linalg.norm(G1g - cx * Kx) / np.linalg.norm(G1g)), vlp=float(np.linalg.norm(G1g - cv * Kv_) / np.linalg.norm(G1g)))
# the objective actually used: quadratic form sum_i |<G_i, dM>|^2 versus Tr(dM^H L dM R)
dM = crand(na, nb_)
q_true = sum(abs(np.vdot(g, dM)) ** 2 for g in Gam_prod)
q_sh = c_ = None
Lp, Rp = Lsh, Rsh
q_kron = np.trace(dM.conj().T @ Lp @ dM @ Rp).real
cc = q_true / q_kron
out['quadform_product_set_scale_const_check'] = [float(cc)] + [float(sum(abs(np.vdot(g, d)) ** 2 for g in Gam_prod) / np.trace(d.conj().T @ Lp @ d @ Rp).real) for d in (crand(na, nb_), crand(na, nb_))]

# 4. real VLP on an exactly Kronecker real PSD matrix
n1, n2 = 8, 5
A0 = rng.standard_normal((n1, n1)); A0 = A0 @ A0.T; B0 = rng.standard_normal((n2, n2)); B0 = B0 @ B0.T
A_, B_, fid = W.vlp_rank1_real(np.kron(A0, B0), n1, n2)
out['vlp_real_exact'] = dict(relerr=float(np.linalg.norm(np.kron(A_, B_) - np.kron(A0, B0)) / np.linalg.norm(np.kron(A0, B0))), fidelity=fid)
Gn = np.kron(A0, B0) + 0.05 * np.linalg.norm(np.kron(A0, B0)) / (n1 * n2) * (lambda X: X + X.T)(rng.standard_normal((n1 * n2, n1 * n2)))
A_, B_, fid = W.vlp_rank1_real(Gn, n1, n2)
out['vlp_real_noisy'] = dict(relerr=float(np.linalg.norm(np.kron(A_, B_) - Gn) / np.linalg.norm(Gn)), fidelity=fid)

# 5. structured Kronecker solve equals the dense solve (real and complex Hermitian)
for cplx in (False, True):
    Ah = crand(6, 6) if cplx else rng.standard_normal((6, 6)); Ah = Ah @ Ah.conj().T
    Bh = crand(4, 4) if cplx else rng.standard_normal((4, 4)); Bh = Bh @ Bh.conj().T
    g = crand(24) if cplx else rng.standard_normal(24)
    dl = 1e-3
    la = np.linalg.eigvalsh(Ah).max(); lb = np.linalg.eigvalsh(Bh).max()
    dense = np.linalg.solve(np.kron(Ah, Bh) + dl * la * lb * np.eye(24), g)
    out[f'kron_solve_{"complex" if cplx else "real"}'] = float(np.linalg.norm(W.kron_solve(Ah, Bh, g, dl) - dense) / np.linalg.norm(dense))
# realification: z^H H z = x^T rho(H) x
H = crand(7, 7); H = H + H.conj().T; z = crand(7); x = np.concatenate([z.real, z.imag])
out['realify'] = float(abs((z.conj() @ H @ z).real - x @ W.realify(H) @ x))
print(json.dumps(out, indent=1))
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
