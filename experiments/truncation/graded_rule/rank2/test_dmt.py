"""Checks of dmt_np.py.   python test_dmt.py
(a) the Pauli superoperators are orthogonal and reproduce G rho G^dag;
(b) at large chi the MPDO TEBD equals the exact physical rho of the dense purification (same circuit, same initial state) to 1e-12;
(c) at every DMT cut tr rho, rho_{0..b+1} and rho_{b..N-1} are unchanged (dense check, N = 6);
(d) the Frobenius cut does change them (sanity of the check).
"""
import _p2  # noqa: F401
import json
import numpy as np
import mpsenh as M
import purlib as P
import dmt_np as D

res = {}
rng = np.random.default_rng(0)
# (a)
G = M.make_gates('ising', 4, 0.1)[1].reshape(4, 4)
S = D.pauli_super(G).reshape(16, 16)
print('(a) S S^T - 1:', np.abs(S @ S.T - np.eye(16)).max())
rho = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
rho = rho + rho.conj().T
c = np.array([np.trace(O.conj().T @ rho).real / 4 for O in D.OPS2])
rho2 = G @ rho @ G.conj().T
c2 = np.array([np.trace(O.conj().T @ rho2).real / 4 for O in D.OPS2])
print('    S c - c(G rho G^dag):', np.abs(S @ c - c2).max())
res['a_orth'] = float(np.abs(S @ S.T - np.eye(16)).max())
res['a_apply'] = float(np.abs(S @ c - c2).max())

# (b) exact evolution
N, model, fam, mu, T_ = 6, 'ising', 'stag', 0.5, 1.0
ref = P.Reference(model, N, fam, mu, 'plain', T_)
Ss = D.mpdo_gates(model, N, 0.1)
T0 = D.initial_mpdo(N, fam, mu)
Tm, _ = D.run_tebd_mpdo(T0, Ss, 4096, int(round(T_ / 0.1)), D.FrobCut())
rho_m = D.vec_to_rho(D.mpdo_to_vec(Tm), N)
print('(b) MPDO (chi large) vs dense physical rho: max abs diff', np.abs(rho_m - ref.rho).max(), ' trace', np.trace(rho_m).real, ' max bond', max(t.shape[2] for t in Tm))
res['b_exact'] = float(np.abs(rho_m - ref.rho).max())
# initial state check
r0 = D.vec_to_rho(D.mpdo_to_vec(T0), N)
psi0 = P.initial_dense(N, fam, mu)
print('    initial rho vs purification:', np.abs(r0 - P.psi_to_rho(psi0, N)).max())
res['b_init'] = float(np.abs(r0 - P.psi_to_rho(psi0, N)).max())


def dense_with(T, b, th=None):
    if th is None:
        return D.vec_to_rho(D.mpdo_to_vec(T), N)
    left = np.ones((1, 1))
    for t in T[:b]:
        left = np.tensordot(left, t, axes=([1], [0])).reshape(-1, t.shape[2])
    right = np.ones((1, 1))
    for t in reversed(T[b + 2:]):
        right = np.tensordot(t, right, axes=([2], [0])).reshape(t.shape[0], -1)
    v = (left @ th.reshape(th.shape[0], -1)).reshape(-1, th.shape[3]) @ right
    return D.vec_to_rho(v.reshape(-1), N)


def preserved(rho, b):
    tr = np.trace(rho)
    left = P.rdm_k(rho, N, 0, b + 2)
    right = P.rdm_k(rho, N, b, N - b)
    return tr, left, right


# (c), (d)
for name, cut in (('DMT', D.DMTCut(check=True)), ('Frobenius', D.FrobCut())):
    state = {}
    worst = dict(tr=0.0, left=0.0, right=0.0)
    ncut = 0

    def hook(th, T, b, dirn, phase):
        global ncut
        if phase == 'pre':
            state['pre'] = preserved(dense_with(T, b, th), b)
        else:
            post = preserved(dense_with(T, b), b)
            pre = state['pre']
            worst['tr'] = max(worst['tr'], abs(post[0] - pre[0]))
            worst['left'] = max(worst['left'], np.abs(post[1] - pre[1]).max())
            worst['right'] = max(worst['right'], np.abs(post[2] - pre[2]).max())
            ncut += 1

    Tm, _ = D.run_tebd_mpdo(T0, Ss, 10, 10, cut, hook=hook)
    extra = f' (fired {cut.fired}/{cut.calls})'
    print(f'(c/d) {name}: chi=10, 10 steps, {ncut} cuts{extra}: max change of tr rho {worst["tr"]:.2e}, rho_[0..b+1] {worst["left"]:.2e}, rho_[b..N-1] {worst["right"]:.2e}')
    res['cd_' + name] = dict(worst, cuts=ncut, fired=cut.fired, calls=cut.calls)
json.dump(res, open('results/test_dmt.json', 'w'), indent=1)
