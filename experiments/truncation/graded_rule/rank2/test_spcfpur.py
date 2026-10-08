"""Test 0(a): checks of rank2/spcfpur.py (d = 4 fused site + ancilla), ported from test_spcfast.py.

    python test_spcfpur.py [gauge] [mu] [family]

1. W-form: the physical region RDM W W^dag built from the environment maps and the two-site tensor against the RDM of the dense
   purification (ancillas traced out), at every recorded cut.   [new: tests the ancilla routing]
2. F-form against the window-marginal objective on random density matrices (F is inherited from d = 2; same as test_spcfast.py).
3. adjoint pair <g, J x> = <J^T g, x> at every recorded cut.
4. linearisation error of the exact (retracted) residual against the matrix-free Jacobian, shrinking with the step.
5. fired cuts never raise the objective (log: f_c <= f_svd).
"""
import sys
import _p2  # noqa: F401
import json
import numpy as np
import mpsenh as M
import purlib as P
from spcfpur import SPCFPur

gauge = sys.argv[1] if len(sys.argv) > 1 else 'plain'
mu = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
family = sys.argv[3] if len(sys.argv) > 3 else 'stag'
rng = np.random.default_rng(1)
model, N, chi = 'ising', 8, 4
cut = SPCFPur(model, N, a=2, taus=[1.0, 2.0], fw=0.0)
cut.debug = []
T0 = P.initial_pur_mps(N, family, mu)
Gs = P.pur_gates(model, N, 0.1, gauge)
P.run_tebd_d(T0, Gs, chi, 25, cut)
print(f'gauge={gauge} mu={mu} family={family}: cuts {cut.calls} fired {cut.fired} debug {len(cut.debug)}')
res = {'gauge': gauge, 'mu': mu, 'family': family, 'cuts': cut.calls, 'fired': cut.fired, 'recorded': len(cut.debug)}

# 1. W form against the dense purification
worst_w = 0.0
for dd in cut.debug:
    Tl = dd['Tsnap']
    b, lo, hi, th = dd['b'], dd['lo'], dd['hi'], dd['theta']
    # left: (4^b, l)
    left = np.ones((1, 1), dtype=complex)
    for t in Tl[:b]:
        left = np.tensordot(left, t, axes=([1], [0])).reshape(-1, t.shape[2])
    right = np.ones((1, 1), dtype=complex)
    for t in reversed(Tl[b + 2:]):
        right = np.tensordot(t, right, axes=([2], [0])).reshape(t.shape[0], -1)
    v = (left @ th.reshape(th.shape[0], -1)).reshape(-1, th.shape[3]) @ right
    v = v.reshape(-1)
    rho = P.psi_to_rho(v, N)
    R = P.rdm_k(rho, N, lo, hi - lo)
    W = dd['Wof'](th)
    Rw = W @ W.conj().T
    worst_w = max(worst_w, np.abs(R - Rw).max() / np.abs(R).max())
print('1. W-form vs dense physical region RDM, max rel err (float32 maps):', worst_w)
res['W_vs_dense'] = worst_w

# 2. F-form against the direct objective
L, ks = 6, (1, 2, 3)
D, (Fs, Fd32), iu, dg = cut._region(L, False)
F = np.concatenate([Fs.toarray(), Fd32.astype(float)], axis=0)


def rand_rho(n):
    W = rng.normal(size=(D, n)) + 1j * rng.normal(size=(D, n))
    r = W @ W.conj().T
    return r / np.trace(r).real


def hvec(rho):
    return np.concatenate([rho[dg, dg].real, rho[iu].real, rho[iu].imag])


Hm = np.zeros((D, D), dtype=complex)
for bb in range(L - 1):
    Hm += np.kron(np.kron(np.eye(2 ** bb), M.h_bond(model, 3, N)), np.eye(2 ** (L - bb - 2)))
ev, V = np.linalg.eigh(Hm)


def direct(rho, tgt):
    tot = 0.0
    for tau in cut.taus:
        U = (V * np.exp(-1j * tau * ev)) @ V.conj().T
        d = U @ (rho - tgt) @ U.conj().T
        for k in ks:
            nwin = L - k + 1
            acc = 0.0
            for off in range(nwin):
                r = L - off - k
                X = d.reshape(2 ** off, 2 ** k, 2 ** r, 2 ** off, 2 ** k, 2 ** r)
                acc += np.sum(np.abs(np.einsum('asbatb->st', X)) ** 2)
            tot += acc / nwin
    return tot


a, b = rand_rho(20), rand_rho(30)
rv = F @ (hvec(a) - hvec(b))
print('2. objective F-form vs direct', rv @ rv, direct(a, b))
res['F_form'] = (float(rv @ rv), float(direct(a, b)))

# 3. adjoint pair
worst_adj = 0.0
for d in cut.debug:
    x = rng.normal(size=d['nx'])
    g = rng.normal(size=d['nres'])
    lhs, rhs = g @ d['jvp'](x), d['vjp'](g) @ x
    worst_adj = max(worst_adj, abs(lhs - rhs) / (abs(lhs) + 1e-30))
print('3. adjoint rel err (max over recorded cuts):', worst_adj)
res['adjoint_max'] = float(worst_adj)

# 4. linearisation
worst_lin = 0.0
for d in cut.debug:
    x = rng.normal(size=d['nx'])
    eps = 1e-6 / np.linalg.norm(x)
    num = (d['exact'](eps * x)[2] - d['r0']) / eps
    ana = d['jvp'](x)
    worst_lin = max(worst_lin, np.linalg.norm(num - ana) / np.linalg.norm(ana))
print('4. linearisation rel err at step 1e-6 (max):', worst_lin)
res['lin_1e-6_max'] = float(worst_lin)
res['lin_by_step'] = {}
for rel in (1e-1, 1e-2, 1e-3, 1e-4):
    errs = []
    for d in cut.debug[::3]:
        x = rng.normal(size=d['nx'])
        x *= rel / np.linalg.norm(x)
        num = d['exact'](x)[2] - d['r0']
        ana = d['jvp'](x)
        errs.append(np.linalg.norm(num - ana) / np.linalg.norm(ana))
    print('   step', rel, 'median linearisation rel err', np.median(errs), 'max', np.max(errs))
    res['lin_by_step'][str(rel)] = (float(np.median(errs)), float(np.max(errs)))

# 5. monotone
lg = np.array(cut.log)
print('5. fired cuts', len(lg), 'max f_c/f_svd', float(np.max(lg[:, 1] / lg[:, 0])), 'median', float(np.median(lg[:, 1] / lg[:, 0])))
res['ratio_max'] = float(np.max(lg[:, 1] / lg[:, 0]))
res['ratio_median'] = float(np.median(lg[:, 1] / lg[:, 0]))
out = f'results/test0a_{gauge}_{family}_mu{P.mu_tag(mu)}.json'
json.dump(res, open(out, 'w'), indent=1)
print('saved', out)
