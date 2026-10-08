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
        d = d.sum(axis=i) if v is None else np.tensordot(d, v[i], axes=([i], [0]))
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
        check('svd loses the energy sum-rule at the cut level (worst per-cut change > 1e-6)', 1e-6 - worst['E'] if worst['E'] > 1e-6 else 1.0, 0.0)

# ---------------------------------------------------------------------------------------------------- 3. wsvd lam = 0
N4, chi4, ns4 = 8, 12, 8
T0 = H.start_mpo('isingT', N4, 3, 'E')
r_svd = P.run_heis('isingT', N4, T0, chi4, ns4, 0.1, W.make_cut('svd'))
r_w0 = P.run_heis('isingT', N4, T0, chi4, ns4, 0.1, W.make_cut('wsvd:0:0'))
check('wsvd lam=0 vs svd: max |T_svd - T_wsvd| over all tensors (bitwise path)', max(np.max(np.abs(a - b)) for a, b in zip(r_svd['T'], r_w0['T'])), 0.0)
r_w1 = P.run_heis('isingT', N4, T0, chi4, ns4, 0.1, W.OpCut('wsvd', lam1=1e-13, lam2=0.0))
check('wsvd lam=1e-13 (general path) vs svd, dense operator', np.max(np.abs(P.dense_op(r_w1['T']) - P.dense_op(r_svd['T']))), 1e-9)

# ---------------------------------------------------------------------------------------------------- 4. static bond-matrix tools vs the MPO cut
def dense_to_mpo(c, N, b):
    """Exact mixed-canonical MPO of a flat 4^N vector with orthogonality centre on bond b: sites <= b left-isometric except we return
    the unsplit centre as theta (sites b, b+1)."""
    x = c.reshape(1, -1)
    Ts = []
    l = 1
    for i in range(b):                                     # left-isometric sites 0..b-1
        U, s, Vh = np.linalg.svd(x.reshape(l * 4, -1), full_matrices=False)
        k = int(np.sum(s > 1e-14 * s[0]))
        Ts.append(U[:, :k].reshape(l, 4, k))
        x = s[:k, None] * Vh[:k]
        l = k
    rest = x                                              # (l, 4^(N-b)) : sites b..N-1
    tail = []
    y = rest
    r_dims = []
    # right-isometric sites N-1 .. b+2
    for i in range(N - 1, b + 1, -1):
        yy = y.reshape(y.shape[0] * (4 ** (i - b)) // (4 ** (i - b)), -1) if False else y
        y = y.reshape(l * 4 ** (i - b), 4)
        pass
    return None


def exact_mpo_from_dense(c, N, b):
    """Left-isometric sites 0..b-1, centre theta on sites b, b+1, right-isometric sites b+2..N-1; returns (T list with theta already
    split as T[b] = U-ish, T[b+1] = s Vh-ish, but only used through their product), theta."""
    t = c.reshape([4] * N)
    # left part
    Ts = []
    x = t.reshape(1, 4, -1)
    for i in range(b):
        l = x.shape[0]
        M2 = x.reshape(l * 4, -1)
        U, s, Vh = np.linalg.svd(M2, full_matrices=False)
        k = int(np.sum(s > 1e-14 * s[0]))
        Ts.append(U[:, :k].reshape(l, 4, k))
        x = (s[:k, None] * Vh[:k]).reshape(k, 4, -1)
    # x: (l, 4 [site b], rest)
    l = x.shape[0]
    rest = x.reshape(l, 4, 4, -1) if False else None
    xr = x.reshape(l * 4, -1)                              # (l*4_b, 4^(N-b-1))
    # right part from the right end
    nR = N - b - 1
    Rs = []
    y = xr.reshape(l * 4, 4 ** nR)
    cur = y
    for j in range(nR - 1):                               # peel sites N-1 .. b+2
        left = cur.shape[0]
        M2 = cur.reshape(left * 4 ** (nR - 1 - j - 0) // 4 ** (nR - 1 - j), -1) if False else cur
        break
    return None


# simpler exact conversion: sequential SVD left to right, then right to left re-canonicalisation by QR
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
    """Make sites > upto right-isometric (pushing weight left)."""
    Ts = [t.copy() for t in Ts]
    for i in range(len(Ts) - 1, upto, -1):
        l, p, r = Ts[i].shape
        Q, R = np.linalg.qr(Ts[i].reshape(l, p * r).T)       # Ts[i] = R^T Q^T
        Ts[i] = Q.T.reshape(-1, p, r)
        Ts[i - 1] = np.tensordot(Ts[i - 1], R.T, axes=([2], [0]))
    return Ts


Nst, bst = 8, 3
G = H.circuit16('isingT', Nst, 0.1)
c = H.start_operator('isingT', Nst, 3, 'E')
for _ in range(10):
    for b in range(Nst - 1):
        c = P.dense_gate(c, G[b], b, Nst)
    for b in range(Nst - 2, -1, -1):
        c = P.dense_gate(c, G[b], b, Nst)
Ts = dense_to_mpo_full(c, Nst)
Ts = right_canon(Ts, bst + 1)                              # sites > b+1 right-isometric; sites < b left-isometric by construction
theta = np.tensordot(Ts[bst], Ts[bst + 1], axes=([2], [0]))
Lr, Rr = P.envs(Ts)
ctx = dict(b=bst, N=Nst, T=Ts, Lr=Lr, Rr=Rr)
Mx = c.reshape(4 ** (bst + 1), -1)
lay = W.static_layout(Nst, bst)


def mpo_cut_matrix(cut, chi):
    A, B, _ = cut(theta, chi, 'R', ctx)
    T2 = list(Ts)
    T2[bst], T2[bst + 1] = A, B
    return P.dense_op(T2).reshape(Mx.shape)


for chi in (10, 20):
    Msvd = mpo_cut_matrix(W.make_cut('svd'), chi)
    fac = W.StaticFactors(Mx, np.ones(lay['nrow']), np.ones(lay['ncol']), chi)
    check(f'static svd vs MPO svd cut (N=8, b=3, chi={chi})', np.max(np.abs(Mx - fac.error(Mx, chi) - Msvd)), 1e-10)
    for lam1, lam2 in ((10.0, 0.0), (0.0, 30.0), (3.0, 100.0)):
        wl, wr = W.weights_wsvd(lay, lam1, lam2)
        fac = W.StaticFactors(Mx, wl, wr, chi)
        Mm = mpo_cut_matrix(W.OpCut('wsvd', lam1=lam1, lam2=lam2), chi)
        check(f'static wsvd({lam1},{lam2}) vs MPO wsvd cut (N=8, b=3, chi={chi})', np.max(np.abs(Mx - fac.error(Mx, chi) - Mm)), 1e-9)
    for n in (1, 2):
        chi_n = chi + 22 if n == 2 else chi
        d = W.StaticDMT(Mx, lay, n, chi_n)
        Mm = mpo_cut_matrix(W.OpCut('dmt', n=n), chi_n)
        check(f'static dmt:{n} vs MPO dmt:{n} cut (N=8, b=3, chi={chi_n})', np.max(np.abs(Mx - d.error(Mx, chi_n) - Mm)), 1e-9)

# ---------------------------------------------------------------------------------------------------- 5. wsvd limits on the static matrix
chi = 40
lam = 1e8
for n, (l1, l2) in ((1, (lam, 0.0)), (2, (0.0, lam))):
    wl, wr = W.weights_wsvd(lay, l1, l2)
    fac = W.StaticFactors(Mx, wl, wr, chi)
    Mp = Mx - fac.error(Mx, chi)
    rows, cols = W.near_idx(lay, n)
    pres = max(np.max(np.abs((Mp - Mx)[rows, :])), np.max(np.abs((Mp - Mx)[:, cols])))
    check(f'wsvd lam=1e8 (n={n}) vs exact on the preserved rows/cols (static)', pres, 1e-8)
    d = W.StaticDMT(Mx, lay, n, chi)
    Ed = d.error(Mx, chi)
    check(f'dmt:{n} preserved rows/cols exact (static)', max(np.max(np.abs(Ed[rows, :])), np.max(np.abs(Ed[:, cols]))), 1e-14)
    ew, ed = np.linalg.norm(Mx - Mp), np.linalg.norm(Ed)
    note(f'frobenius error at chi={chi}: wsvd(lam=1e8,n={n}) vs dmt:{n}', [float(ew), float(ed)])
    check(f'wsvd lam=1e8 (n={n}) Frobenius error <= dmt:{n} (DMT is a feasible point)', ew / ed - 1 if ew > ed else 0.0, 1e-8)
    s = np.linalg.svd(Mp, compute_uv=False)
    check(f'wsvd rank <= chi (n={n}), numerical rank excess', max(0, int(np.sum(s > 1e-9 * s[0])) - chi), 0)
# Eckart-Young certificate for a finite-lam weighted problem
wl, wr = W.weights_wsvd(lay, 3.0, 10.0)
fac = W.StaticFactors(Mx, wl, wr, 25)
Wm = (wl[:, None] * Mx) * wr[None, :]
sfull = np.linalg.svd(Wm, compute_uv=False)
werr = np.linalg.norm(wl[:, None] * fac.error(Mx, 25) * wr[None, :])
check('weighted residual equals sqrt(sum of discarded weighted singular values^2) (Eckart-Young)', abs(werr - np.sqrt(np.sum(sfull[25:] ** 2))) / werr, 1e-9)

# ---------------------------------------------------------------------------------------------------- 6. class bookkeeping
sc = W.ClassScorer(Nst, bst, 'isingT', Mx, 3)
check('class table sums to ||M||_F^2', abs(sc.M2.sum() - sc.fro2) / sc.fro2, 1e-12)
c1, c2 = P.local_coeffs_dense(c, Nst)
check('C(x) from the bond matrix == C(x) from the dense vector', np.max(np.abs(sc.Mx_C - P.energy_density('isingT', c1, c2))), 1e-13)
S_d1 = sc.class_sq(W.StaticDMT(Mx, lay, 1, 14).error(Mx, 14))
sp = W.span_classes(S_d1)
check('dmt:1 error is exactly zero on span<=3, a<=1 and c<=1 strings', max(sp['le3'], sp['onesided']), 1e-28)
check('dmt:1 error is nonzero on (2,2) strings', 0.0 if S_d1[2, 2] > 0 else 1.0, 0.0)
S_d2 = sc.class_sq(W.StaticDMT(Mx, lay, 2, 40).error(Mx, 40))
check('dmt:2 error is exactly zero on every a<=2 or c<=2 string', max(S_d2[:3, :].max(), S_d2[:, :3].max()), 1e-28)

# ---------------------------------------------------------------------------------------------------- 7. cross-check of dmt:1 with rank4 DMTCut (read-only import)
spec = importlib.util.spec_from_file_location('r4_heis_mpo', str(HERE.parent / 'rank4' / 'heis_mpo.py'))
r4 = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(HERE.parent / 'rank4'))
spec.loader.exec_module(r4)
N5, chi5, ns5 = 10, 12, 12
res4 = r4.run_heis('ising', N5, 4, chi5, ns5, 0.1, r4.make_cut('dmt'), r4.ref_provider('ising', N5, 'I'), pauli=3)
T0 = P.init_local(N5, 4, np.array([0, 0, 0, 1.0]))
res10 = P.run_heis('ising', N5, T0, chi5, ns5, 0.1, W.make_cut('dmt:1'))
d4 = r4.dense_op(res4['T']).reshape(-1)
d10 = P.dense_op(res10['T'])
check('dmt:1 (this module) vs rank4 DMT-I, dense operator after 12 steps (N=10, chi=12)', np.max(np.abs(d4 - d10)), 1e-9)
note('||O|| after 12 steps (rank4 vs rank10)', [float(np.linalg.norm(d4)), float(np.linalg.norm(d10))])

out['all_ok'] = bool(ok_all)
json.dump(out, open(HERE / 'results' / 'test_wsvd_op.json', 'w'), indent=1)
print('\nTEST wsvd_op:', 'ALL PASS' if ok_all else 'FAILURES PRESENT')
