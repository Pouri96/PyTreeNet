"""Unit tests for rank 10: exactness, DMT-n preserved functionals, wsvd limits (lam = 0 -> svd, lam -> inf -> DMT constraint), static vs MPO
consistency, cross-check of dmt:1 against rank4's DMTCut, energy sum-rule at the cut level.  python rank10/test_wsvd_op.py"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
import json
import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pmpo as P
import wsvd_op as W
import heis_ref as H

out = {}
ok_all = True


def check(name, val, tol):
    global ok_all
    ok = bool(val <= tol)
    ok_all &= ok
    out[name] = dict(value=float(val), tol=tol, ok=ok)
    print(f'{"PASS" if ok else "FAIL"}  {name:<92s} {val:.3e}  (tol {tol:g})', flush=True)


def note(name, val):
    out[name] = val
    print(f'      {name}: {val}', flush=True)


# ---------------------------------------------------------------------------------------------------- 1. chi = infinity
N, nsteps, dt = 6, 6, 0.1
for model in ('isingT',):
    G = H.circuit16(model, N, dt)
    c = H.start_operator(model, N, 2, 'E')
    for _ in range(nsteps):
        for b in range(N - 1):
            c = P.dense_gate(c, G[b], b, N)
        for b in range(N - 2, -1, -1):
            c = P.dense_gate(c, G[b], b, N)
    for spec in ('svd', 'dmt:1', 'dmt:2', 'wsvd:3:10', 'wsvd:0:5'):
        r = P.run_heis(model, N, H.start_mpo(model, N, 2, 'E'), 10 ** 6, nsteps, dt, W.make_cut(spec))
        check(f'chi=inf {model} {spec} vs dense operator (N=6, 6 steps)', np.max(np.abs(P.dense_op(r['T']) - c)), 1e-12)
    g = 1.6
    r = P.run_heis(model, N, H.start_mpo(model, N, 2, 'E'), 10 ** 6, nsteps, dt, W.make_cut('svd'), gamma=g)
    check(f'chi=inf {model} rw:1.6 (unweighted) vs dense operator', np.max(np.abs(P.dense_op(P.unweight(r['T'], g)) - c)), 1e-11)

# ---------------------------------------------------------------------------------------------------- 2. preserved functionals (dense-operator hook)
def contract_sites(d, sites, v=None):
    for i in sorted(sites, reverse=True):
        d = np.take(d, 0, axis=i) if v is None else np.tensordot(d, v[i], axes=([i], [0]))
    return d


def hook_factory(N3, worst, nmax):
    def hook(b, dirn, step, snap, th, A, B):
        def full_op(mid):
            cur = None
            for i in range(b):
                cur = snap[i][0] if cur is None else np.tensordot(cur, snap[i], axes=([-1], [0]))
            cur = mid[0] if cur is None else np.tensordot(cur, mid, axes=([-1], [0]))
            for i in range(b + 2, N3):
                cur = np.tensordot(cur, snap[i], axes=([-1], [0]))
            return cur[..., 0]
        d = full_op(np.tensordot(A, B, axes=([2], [0]))) - full_op(th)               # (4,)*N3 error tensor
        for n in range(1, nmax + 1):
            # preserved: identity on sites < b-n+1 (functional supported on [b-n+1, N-1]) and on sites > b+n
            lo = max(b - n + 1, 0)
            fa = contract_sites(d, range(0, lo))
            hi = min(b + n, N3 - 1)
            fb = contract_sites(d, range(hi + 1, N3))
            worst[('pres', n)] = max(worst.get(('pres', n), 0.0), np.max(np.abs(fa)) if np.size(fa) else 0.0, np.max(np.abs(fb)) if np.size(fb) else 0.0)
            # negative control: window [b-n, b+n+1] (n+1 sites each side) with the rest traced
            lo2, hi2 = max(b - n, 0), min(b + n + 1, N3 - 1)
            fw = contract_sites(d, [i for i in range(N3) if i < lo2 or i > hi2])
            worst[('win', n)] = max(worst.get(('win', n), 0.0), np.max(np.abs(fw)))
        # energy sum rule: sum_x eps_x functional
        c1a, c2a = P.local_coeffs_dense(d.reshape(-1), N3)
        worst['E'] = max(worst.get('E', 0.0), abs(P.energy_density(model, c1a, c2a).sum()))
    return hook


model = 'isingT'
N3 = 8
for spec, chi, nmax in (('dmt:1', 12, 1), ('dmt:2', 32, 2), ('wsvd:1e8:0', 12, 1), ('wsvd:0:1e8', 32, 2), ('svd', 12, 2)):
    worst = {}
    cut = W.make_cut(spec)
    P.run_heis(model, N3, H.start_mpo(model, N3, 3, 'E'), chi, 10, 0.1, cut, hook=hook_factory(N3, worst, nmax))
    tag = f'{spec} chi={chi} N=8'
    if spec.startswith('dmt') or spec.startswith('wsvd'):
        n_p = 1 if spec in ('dmt:1', 'wsvd:1e8:0') else 2
        tol = 1e-12 if spec.startswith('dmt') else 1e-7
        check(f'{tag}: preserved functionals of radius {n_p}, worst abs violation over all cuts', worst[('pres', n_p)], tol)
        check(f'{tag}: energy sum-rule change per cut (sum_x eps_x)', worst['E'], tol)
        note(f'{tag}: window of radius {n_p} (2n+2 sites) violation (expected NONZERO for dmt)', worst[('win', n_p)])
        note(f'{tag}: ntruncating cuts', cut.ntrunc)
    else:
        note(f'{tag}: SVD violation radius1/radius2 (expected NONZERO)', [worst[('pres', 1)], worst[('pres', 2)]])
        note(f'{tag}: SVD energy sum-rule change per cut (expected NONZERO)', worst['E'])
        check('svd loses the energy sum-rule at the cut level (worst per-cut change must exceed 1e-6)', 0.0 if worst['E'] > 1e-6 else 1.0, 0.0)

# ---------------------------------------------------------------------------------------------------- 3. wsvd lam = 0
N4, chi4, ns4 = 8, 12, 8
T0 = H.start_mpo('isingT', N4, 3, 'E')
r_svd = P.run_heis('isingT', N4, T0, chi4, ns4, 0.1, W.make_cut('svd'))
r_w0 = P.run_heis('isingT', N4, T0, chi4, ns4, 0.1, W.make_cut('wsvd:0:0'))
check('wsvd lam=0 vs svd: max |T_svd - T_wsvd| over all tensors (bitwise path)', max(np.max(np.abs(a - b)) for a, b in zip(r_svd['T'], r_w0['T'])), 0.0)
r_w1 = P.run_heis('isingT', N4, T0, chi4, ns4, 0.1, W.OpCut('wsvd', lam1=1e-13, lam2=0.0))
check('wsvd lam=1e-13 (general path) vs svd, dense operator', np.max(np.abs(P.dense_op(r_w1['T']) - P.dense_op(r_svd['T']))), 1e-9)

# ---------------------------------------------------------------------------------------------------- 4. bond-gauge static route vs MPO cut
import static_cut as SC


def dense_to_mpo_full(c, N):
    x = c.reshape(1, -1)
    Ts, l = [], 1
    for i in range(N - 1):
        U, s, Vh = np.linalg.svd(x.reshape(l * 4, -1), full_matrices=False)
        k = int(np.sum(s > 1e-15 * s[0]))
        Ts.append(U[:, :k].reshape(l, 4, k))
        x = s[:k, None] * Vh[:k]
        l = k
    Ts.append(x.reshape(l, 4, 1))
    return Ts


def right_canon(Ts, upto):
    Ts = [t.copy() for t in Ts]
    for i in range(len(Ts) - 1, upto, -1):
        l, p, r = Ts[i].shape
        Q, R = np.linalg.qr(Ts[i].reshape(l, p * r).T)
        Ts[i] = Q.T.reshape(-1, p, r)
        Ts[i - 1] = np.tensordot(Ts[i - 1], R.T, axes=([2], [0]))
    return Ts


def force_iter(mv, rmv, n, K, dense=None):
    return W.topk_op(mv, rmv, n, K, p=K + 24)


Nst, bst = 8, 3
G = H.circuit16('isingT', Nst, 0.1)
c = H.start_operator('isingT', Nst, 3, 'E')
for _ in range(10):
    for b in range(Nst - 1):
        c = P.dense_gate(c, G[b], b, Nst)
    for b in range(Nst - 2, -1, -1):
        c = P.dense_gate(c, G[b], b, Nst)
P.EPS_S = 1e-12                                   # same bond cutoff as StaticCut(kcut=1e-12)
Ts = right_canon(dense_to_mpo_full(c, Nst), bst + 1)
theta = np.tensordot(Ts[bst], Ts[bst + 1], axes=([2], [0]))
Lr, Rr = P.envs(Ts)
ctx = dict(b=bst, N=Nst, T=Ts, Lr=Lr, Rr=Rr)
sc = SC.StaticCut(c, Nst, bst, 'isingT')
Mx = sc.M
note('N=8 b=3: bond dimension k of the exact operator at step 10', sc.k)


def mpo_cut_matrix(cut, chi):
    A, B, _ = cut(theta, chi, 'R', ctx)
    T2 = list(Ts)
    T2[bst], T2[bst + 1] = A, B
    return P.dense_op(T2).reshape(Mx.shape)


for chi in (10, 20):
    for spec, chi_u in (('svd', chi), ('wsvd:10:0', chi), ('wsvd:0:30', chi), ('wsvd:3:100', chi), ('dmt:1', chi), ('dmt:2', chi + 22)):
        Pm, Qm, info = sc.arm_factors(spec, chi_u)
        Ms = Pm[:, :chi_u] @ Qm[:, :chi_u].T
        Mm = mpo_cut_matrix(W.make_cut(spec), chi_u)
        # DMT's near/far split is defined through the projection of the coordinate near rows on the bond space, which includes directions at
        # the bond cutoff (sigma ~ 1e-12 s0); roundoff-level rotations of those directions move M' by up to ~1e-6, hence the looser tolerance
        tol_el = 1e-5 if spec.startswith('dmt') else 1e-9
        check(f'static bond-gauge {spec} vs MPO cut (N=8, b=3, chi={chi_u}), dense path', np.max(np.abs(Ms - Mm)), tol_el)
        check(f'   same, Frobenius error norms relative difference', abs(np.linalg.norm(Mx - Ms) - np.linalg.norm(Mx - Mm)) / np.linalg.norm(Mx - Ms), 1e-6)
        # same through the iterative solver (no dense shortcut)
        kind = spec.split(':')[0]
        kw = dict(n=int(spec.split(':')[1])) if kind == 'dmt' else (dict(lam1=float(spec.split(':')[1]), lam2=float(spec.split(':')[2])) if kind == 'wsvd' else {})
        Pb, Qb, info = W.bond_factors(kind, sc.s, sc.QL1, sc.QLx, sc.QR1, sc.QRx, chi_u, solver=force_iter, **kw)
        Mi = (sc.U @ Pb) @ (sc.V @ Qb).T
        check(f'static bond-gauge {spec} iterative solver vs dense solver (chi={chi_u})  [converged={info["converged"]}, {info["iters"]} it]',
              np.max(np.abs(Mi - Ms)), 1e-8)

P.EPS_S = 1e-14
# class tables from factors vs direct dense computation
lay = sc.lay
Pm, Qm, _ = sc.arm_factors('wsvd:3:10', 24)
tabs = sc.tables(Pm, Qm, [8, 16, 24])
for chi in (8, 16, 24):
    E = Mx - Pm[:, :chi] @ Qm[:, :chi].T
    Sd = np.array([[np.sum(E[np.ix_(ra, rc)] ** 2) for rc in sc.cols_by_c] for ra in sc.rows_by_a])
    check(f'class table from factors vs direct (chi={chi})', np.max(np.abs(np.array(tabs[chi]['S']) - Sd)), 1e-12)
    c1, c2 = P.local_coeffs_dense(E.reshape(-1), Nst)
    check(f'C(x) error from factors vs direct (chi={chi})', np.max(np.abs(np.array(tabs[chi]['dC']) - P.energy_density('isingT', c1, c2))), 1e-12)
check('class table of the exact matrix sums to ||M||_F^2', abs(sc.M2.sum() - sc.fro2) / sc.fro2, 1e-12)
c1, c2 = P.local_coeffs_dense(c, Nst)
check('C(x) positions: exact C(x) from bond-matrix entries vs dense vector',
      np.max(np.abs(sc.C_of_factors(np.zeros((Mx.shape[0], 1)), np.zeros((Mx.shape[1], 1))) - P.energy_density('isingT', c1, c2))), 1e-13)

# iterative top-K solver on a larger structured problem (k = 1500): singular values and the optimal truncation error
rng = np.random.default_rng(3)
k = 1500
s_big = np.sort(np.exp(-np.arange(k) / 18.0) * (1 + 0.3 * rng.random(k)))[::-1]
QLb = np.linalg.qr(rng.standard_normal((k, 6)))[0]
QRb = np.linalg.qr(rng.standard_normal((k, 6)))[0]
Pb, Qb, info = W.bond_factors('wsvd', s_big, QLb[:, :2], QLb[:, 2:], QRb[:, :2], QRb[:, 2:], 40, lam1=5.0, lam2=20.0)
w1, w2 = np.sqrt(26.0), np.sqrt(21.0)
Lh = np.eye(k) + (w1 - 1) * QLb[:, :2] @ QLb[:, :2].T + (w2 - 1) * QLb[:, 2:] @ QLb[:, 2:].T
Rh = np.eye(k) + (w1 - 1) * QRb[:, :2] @ QRb[:, :2].T + (w2 - 1) * QRb[:, 2:] @ QRb[:, 2:].T
Wd = Lh @ np.diag(s_big) @ Rh
sd = np.linalg.svd(Wd, compute_uv=False)
Mp = Pb @ Qb.T
Lih = np.linalg.inv(Lh)
Rih = np.linalg.inv(Rh)
werr = np.linalg.norm(Lh @ (np.diag(s_big) - Mp) @ Rh)
check(f'topk_op wsvd (k=1500, chi=40): weighted error vs Eckart-Young optimum, relative [{info["iters"]} it, converged={info["converged"]}]',
      abs(werr - np.sqrt(np.sum(sd[40:] ** 2))) / werr, 1e-9)

# ---------------------------------------------------------------------------------------------------- 5. wsvd limits
chi = 40
for n, (l1, l2) in ((1, (1e8, 0.0)), (2, (0.0, 1e8))):
    Pm, Qm, _ = sc.arm_factors(f'wsvd:{l1:g}:{l2:g}', chi)
    Mp = Pm @ Qm.T
    rows, cols = SC.near_idx(lay, n)
    pres = max(np.max(np.abs((Mp - Mx)[rows, :])), np.max(np.abs((Mp - Mx)[:, cols])))
    check(f'wsvd lam=1e8 (n={n}) reproduces the DMT-n preserved rows/cols of the exact operator', pres, 1e-8)
    Pd, Qd, _ = sc.arm_factors(f'dmt:{n}', chi)
    Md = Pd @ Qd.T
    check(f'dmt:{n} preserved rows/cols exact', max(np.max(np.abs((Md - Mx)[rows, :])), np.max(np.abs((Md - Mx)[:, cols]))), 1e-12)
    check(f'wsvd lam=1e8 (n={n}) matches dmt:{n} on the preserved coefficients', max(np.max(np.abs((Mp - Md)[rows, :])), np.max(np.abs((Mp - Md)[:, cols]))), 1e-8)
    # at lam = 1e4 the light block is not yet polluted by the 1e8 condition number: DMT is a feasible point, so wsvd must not be worse
    Pw, Qw, _ = sc.arm_factors(f'wsvd:{1e4 if n == 1 else 0:g}:{1e4 if n == 2 else 0:g}', chi)
    ew, ed = np.linalg.norm(Mx - Pw @ Qw.T), np.linalg.norm(Mx - Md)
    note(f'frobenius error at chi={chi}: wsvd(lam=1e4,n={n}), dmt:{n}', [float(ew), float(ed)])
    check(f'wsvd lam=1e4 (n={n}) Frobenius error <= 1.001 x dmt:{n} (DMT is a feasible point of the limit problem)', max(ew / ed - 1.001, 0.0), 0.0)
    check(f'wsvd rank <= chi (n={n}); numerical rank excess', max(0, int(np.sum(np.linalg.svd(Mp, compute_uv=False) > 1e-9)) - chi), 0)
# Eckart-Young certificate on the full problem
Pm, Qm, _ = sc.arm_factors('wsvd:3:10', 25)
w1, w2 = np.sqrt(14.0), np.sqrt(11.0)
Lh_f = np.eye(sc.k) + (w1 - 1) * sc.QL1 @ sc.QL1.T + (w2 - 1) * sc.QLx @ sc.QLx.T
Rh_f = np.eye(sc.k) + (w1 - 1) * sc.QR1 @ sc.QR1.T + (w2 - 1) * sc.QRx @ sc.QRx.T
Wf = Lh_f @ np.diag(sc.s) @ Rh_f
sfull = np.linalg.svd(Wf, compute_uv=False)
Mb = sc.U.T @ (Pm @ Qm.T) @ sc.V
werr = np.linalg.norm(Lh_f @ (np.diag(sc.s) - Mb) @ Rh_f)
check('weighted residual equals sqrt(sum of discarded weighted singular values^2) (Eckart-Young, N=8 exact operator)', abs(werr - np.sqrt(np.sum(sfull[25:] ** 2))) / werr, 1e-9)

# ---------------------------------------------------------------------------------------------------- 6. class bookkeeping
Pd, Qd, _ = sc.arm_factors('dmt:1', 14)
S_d1 = np.array(sc.tables(Pd, Qd, [14])[14]['S'])
sp = SC.span_classes(S_d1)
check('dmt:1 error is zero (to 1e-14 abs in squared norm) on span<=3 and one-sided strings', max(sp['le3'], sp['onesided']), 1e-14)
check('dmt:1 error is nonzero on (2,2) strings', 0.0 if S_d1[2, 2] > 1e-12 else 1.0, 0.0)
Pd, Qd, _ = sc.arm_factors('dmt:2', 40)
S_d2 = np.array(sc.tables(Pd, Qd, [40])[40]['S'])
check('dmt:2 error is zero on every a<=2 or c<=2 string', max(S_d2[:3, :].max(), S_d2[:, :3].max()), 1e-14)

# ---------------------------------------------------------------------------------------------------- 7. cross-check of dmt:1 with rank4 DMTCut (read-only import)
spec = importlib.util.spec_from_file_location('r4_heis_mpo', str(HERE.parent / 'rank4' / 'heis_mpo.py'))
r4 = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(HERE.parent / 'rank4'))
spec.loader.exec_module(r4)
N5, chi5, ns5 = 10, 12, 12
T0 = P.init_local(N5, 4, np.array([0, 0, 0, 1.0]))
c4 = r4.make_cut('dmt')
cut10 = W.make_cut('dmt:1')
worst4 = []


class Wrap:
    needs_env = True

    def __call__(self, theta, chi, dirn, ctx):
        A, B, d = cut10(theta, chi, dirn, ctx)
        A4, B4, d4 = c4(theta, chi, dirn, dict(kL=ctx['Lr'][ctx['b']], kR=ctx['Rr'][ctx['b'] + 2]))
        worst4.append(np.max(np.abs(np.tensordot(A, B, axes=([2], [0])) - np.tensordot(A4, B4, axes=([2], [0])))))
        return A, B, d


P.run_heis('ising', N5, T0, chi5, ns5, 0.1, Wrap())
check(f'dmt:1 (this module) vs rank4 DMT-I on identical inputs, worst over {len(worst4)} cuts (N=10, chi=12, 12 steps)', max(worst4), 1e-11)
res4 = r4.run_heis('ising', N5, 4, chi5, ns5, 0.1, r4.make_cut('dmt'), r4.ref_provider('ising', N5, 'I'), pauli=3)
res10 = P.run_heis('ising', N5, T0, chi5, ns5, 0.1, W.make_cut('dmt:1'))
note('free-running trajectories after 12 steps: max |O_rank4 - O_rank10| (roundoff-level differences are amplified by 12 steps of truncation)',
     float(np.max(np.abs(r4.dense_op(res4['T']).reshape(-1) - P.dense_op(res10['T'])))))

out['all_ok'] = bool(ok_all)
json.dump(out, open(HERE / 'results' / 'test_wsvd_op.json', 'w'), indent=1)
print('\nTEST wsvd_op:', 'ALL PASS' if ok_all else 'FAILURES PRESENT')
