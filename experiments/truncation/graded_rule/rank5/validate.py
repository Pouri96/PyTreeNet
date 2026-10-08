"""Validation of the rank-5 dense single-cut harness.  Prints PASS/FAIL lines and writes results/validation.json.

    python rank5/validate.py [out.json]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mt_cut as T  # noqa: E402  (imports _paths first)
import numpy as np  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
import spcfast  # noqa: E402
import mpsenh as M  # noqa: E402

out = {}
ok_all = True


def check(name, val, tol, extra=None):
    global ok_all
    ok = bool(val <= tol)
    ok_all &= ok
    out[name] = dict(value=float(val), tol=tol, ok=ok, **(extra or {}))
    print(f'{"PASS" if ok else "FAIL"}  {name:<78s} {val:.3e}  (tol {tol:g})', flush=True)


def note(name, **kw):
    out[name] = kw
    print('NOTE  ' + name + '  ' + '  '.join(f'{k}={v}' for k, v in kw.items()), flush=True)


# ---------------------------------------------------------------------------------------------- 1. objective vs brute force
N = 12
sets, sh = T.make_sets('ising', N)
for b in (5, 3):
    g = T.Geom(N, b)
    for ts in sets:
        if ts.name not in ('S2', 'S4t2'):
            continue
        cell = T.Cell('ising', N, b, ts, sh)
        Q, _ = cell.stack_Q(ts.w_lit, 4)
        Mt = T.project(cell.Ms, Q)
        f_fast = cell.sc.objective(Mt)[0]
        # independent: full-state RDMs of every window of the region, sum_k 1/nwin_k sum_w ||drho||_F^2
        f_brute = 0.0
        for f in range(ts.F):
            for (s, k) in g.wins:
                s_abs = g.lo + s
                d = g.state_rdm(Mt[f], s_abs, k) - cell.sc.rdm_ex[f][(s_abs, k)]
                f_brute += g.cw[k] * float(np.sum(np.abs(d) ** 2))
        check(f'objective hvec vs brute force RDMs, b={b} {ts.name} (rel diff)', abs(f_fast - f_brute) / f_brute, 1e-9)

# ---------------------------------------------------------------------------------------------- 2. chi = full rank is exact
for b in (5, 3):
    g = T.Geom(N, b)
    for ts in sets:
        cell = T.Cell('ising', N, b, ts, sh)
        chi = g.dimL
        arms = cell.all_arms(chi, kappas=(0.1,))
        worst = max(arms[a]['m']['E_near'] for a in ('i', 'ii', 'ii_plus', 'iii', 'v', 'iv'))
        worst_out = max(arms[a]['m']['E_out'] for a in ('i', 'ii', 'ii_plus', 'iii', 'v', 'iv'))
        check(f'chi=full ({chi}) reproduces exact RDMs, b={b} {ts.name} (max E_near over arms)', worst, 1e-12)
        check(f'chi=full ({chi}) reproduces exact RDMs, b={b} {ts.name} (max E_out over arms)', worst_out, 1e-12)
        if b == 5 and ts.name == 'S1':
            check('chi=full, S1: discarded weight', max(arms['i']['m']['eps']), 1e-12)

# ---------------------------------------------------------------------------------------------- 3. Jacobian, quadratic model, GN
ts = [s for s in sets if s.name == 'S2'][0]
cell = T.Cell('ising', N, 5, ts, sh)
g = cell.g
chi = 6
Q0, _ = cell.stack_Q(ts.w_lit, chi)
spc = T.SPCMulti(g, cell.Ms, cell.sc.hv_ex, Q0, cplx=False, kappa=0.1)
Bk, Bp = T.complete_basis(Q0)
r0, _ = spc.eval_Q(Bk)
npar = spc.npar
Xf = 1e-6 * np.eye(npar)
Jf = ((spc.evaluate(Xf, Bk, Bp)[0] - r0[None]) / 1e-6).T
Jc = ((spc.evaluate(1e-5 * np.eye(npar), Bk, Bp)[0] - spc.evaluate(-1e-5 * np.eye(npar), Bk, Bp)[0]) / 2e-5).T
check('FD Jacobian: forward(1e-6) vs central(1e-5), rel Frobenius', np.linalg.norm(Jf - Jc) / np.linalg.norm(Jc), 1e-4)
# first-order prediction of the residual: the relative error of the predicted change must shrink linearly with the step
errs = []
x0 = rng = np.random.default_rng(3)
d = rng.standard_normal(npar)
for sc_ in (1e-3, 1e-4, 1e-5):
    x = sc_ * d
    rn = spc.evaluate(x[None], Bk, Bp)[0][0]
    errs.append(np.linalg.norm(rn - r0 - Jf @ x) / np.linalg.norm(rn - r0))
note('Jacobian first-order prediction error vs step (steps 1e-3,1e-4,1e-5 x randn)', errs=[float(e) for e in errs])
check('Jacobian predicts the residual change of a 1e-5 step (rel error of the change)', errs[-1], 1e-3)
check('Jacobian prediction error scales linearly with the step (err(1e-4)/err(1e-5), expect ~10)', abs(np.log10(errs[1] / errs[2]) - 1.0), 0.3)
# quadratic model of eps_f
em = spc.eps_model(Bk, Bp)
x = 1e-3 * rng.standard_normal(npar)
Qn = spc.Q_of(x[None], Bk, Bp)[0]
_, e = spc.eval_Q(Qn)
pred = np.array([em[f][0] + em[f][1] @ x + x @ em[f][2] @ x for f in range(spc.F)])
check('second-order eps_f model vs exact eps_f at |x|~1e-3*sqrt(n) (rel error of the change)',
      np.max(np.abs((e - pred) / (e - np.array([em[f][0] for f in range(spc.F)])))), 1e-2)
# GN / SQP run: monotone, reduces residual, respects the budget
Qn, info = spc.run(maxit=40)
h = info['hist']
check('spcf-multi: residual objective non-increasing along the run (max increase, rel)',
      max([0.0] + [(h[i + 1] - h[i]) / h[i] for i in range(len(h) - 1)]), 0.0)
check('spcf-multi: residual objective reduced (final/initial, must be < 1)', info['f'] / info['f0'], 0.999)
check('spcf-multi: per-target extra discarded weight within kappa=0.1 (max g_f - 0.1, rel)', max(0.0, max(info['g_f']) - 0.1) / 0.1, 2e-3)
# the stacked-SVD point is a stationary point of the weighted discarded weight: first-order eps_f changes cancel
gw = sum(ts.w_lit[f] * em[f][1] for f in range(spc.F))
check('stacked SVD: sum_f w_f grad eps_f = 0 (|.| / max_f |grad eps_f|)', np.linalg.norm(gw) / max(np.linalg.norm(em[f][1]) for f in range(spc.F)), 1e-8)
# kappa -> 0 reproduces arm (i)
spc0 = T.SPCMulti(g, cell.Ms, cell.sc.hv_ex, Q0, cplx=False, kappa=1e-8)
Qz, infoz = spc0.run(maxit=10)
m_i = cell.sc.score(T.project(cell.Ms, Q0))
m_z = cell.sc.score(T.project(cell.Ms, Qz))
check('kappa->0 reproduces plain stacked SVD (|E_near ratio - 1|)', abs(m_z['E_near'] / m_i['E_near'] - 1.0), 1e-3)

# real vs complex parametrisation on a real target set: complex must not do worse
spcr = T.SPCMulti(g, cell.Ms, cell.sc.hv_ex, Q0, cplx=False, kappa=0.1)
spcc = T.SPCMulti(g, cell.Ms, cell.sc.hv_ex, Q0, cplx=True, kappa=0.1)
Qr, ir = spcr.run(maxit=40)
Qc_, ic = spcc.run(maxit=40)
note('real vs complex chart, S2 chi=6 b=5', f_real=ir['f'], f_complex=ic['f'], f0=ir['f0'])
check('real-chart optimum within 2% of the complex-chart optimum on real targets (f_real/f_complex - 1)', max(0.0, ir['f'] / ic['f'] - 1.0), 0.02)

# ---------------------------------------------------------------------------------------------- 4. cross-check against spcfast.SPCFast
def exact_mps_tensors(N, b):
    """exact mixed-canonical MPS skeleton of a dense state: identity isometries left of the cut, identity right of it"""
    Tn = []
    for j in range(N):
        if j < b:
            l = 2 ** j
            A = np.zeros((l, 2, 2 * l), dtype=complex)
            for a in range(l):
                for s in range(2):
                    A[a, s, 2 * a + s] = 1.0
            Tn.append(A)
        elif j > b + 1:
            r = 2 ** (N - j - 1)
            B = np.zeros((2 * r, 2, r), dtype=complex)
            for c in range(r):
                for s in range(2):
                    B[s * r + c, s, c] = 1.0          # left bond = (site j, sites j+1..), site j most significant
            Tn.append(B)
        else:
            Tn.append(None)
    return Tn


for tag, vec_ts, chi_x in (('S4t3 psi(t)', 'S4t3', 4), ('S4t2 psi(t)', 'S4t2', 6)):
    ts1 = [s for s in sets if s.name == vec_ts][0]
    b = 5
    psi = ts1.vn[0]
    # single-target set
    ts_single = T.TargetSet(vec_ts + '_single', [psi], [1.0], [('lit', [1.0])], {})
    cell1 = T.Cell('ising', N, b, ts_single, sh)
    g1 = cell1.g
    Tn = exact_mps_tensors(N, b)
    theta = psi.reshape(2 ** b, 2, 2, 2 ** (N - b - 2))
    cut = spcfast.SPCFast('ising', N, a=2, taus=(), include_static=True, fw=0.01, iters=300, cg_tol=1e-9, eps_min=0.0, rel_skip=0.0,
                          precision='f64', alphas=(1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01), gate_log=True)
    cut.start(Tn)
    res = cut(theta, chi_x, 'R', None, None, b)
    f_svd, f_c, eps_svd, eps_c = cut.log[0][:4] if cut.log else (None,) * 4
    Q1, _ = cell1.stack_Q(np.ones(1), chi_x)
    f_mine_svd = cell1.sc.objective(T.project(cell1.Ms, Q1))[0]
    check(f'spcfast cross-check {tag}: f_svd (their residual^2) vs my objective of the SVD basis (rel)', abs(f_svd - f_mine_svd) / f_mine_svd, 1e-6)
    # evaluate my objective on spcfast's returned basis
    Atn = res[0].reshape(2 ** b * 2, chi_x)
    f_mine_theirs = cell1.sc.objective(T.project(cell1.Ms, Atn))[0]
    check(f'spcfast cross-check {tag}: f_c of their returned basis vs my objective on it (rel)', abs(f_c - f_mine_theirs) / f_c, 1e-6)
    # is their returned basis orthonormal with the reported discarded weight?
    e_theirs = float(T.eps_of(cell1.Ms, T.project(cell1.Ms, Atn))[0])
    check(f'spcfast cross-check {tag}: reported discarded weight vs recomputed (abs)', abs(e_theirs - eps_c), 1e-8)
    # my constrained optimum at their discarded weight can only be lower in f
    kap = e_theirs / eps_svd - 1.0
    spc1 = T.SPCMulti(g1, cell1.Ms, cell1.sc.hv_ex, Q1, cplx=True, kappa=kap)
    Qm, im = spc1.run(maxit=60)
    note(f'spcfast cross-check {tag}', f_svd=f_svd, f_theirs=f_c, f_mine=im['f'], kappa_theirs=kap, eps_svd=eps_svd, eps_theirs=eps_c,
         eps_mine=float(T.eps_of(cell1.Ms, T.project(cell1.Ms, Qm))[0]))
    check(f'spcfast cross-check {tag}: my SQP optimum at their discarded weight is at least as good (f_mine/f_theirs - 1, must be <= 0)',
          im['f'] / f_c - 1.0, 1e-3)
    # positive control: single target, kappa = 0.1, 1.0 -> E_near gain
    for kk in (0.1, 1.0):
        spck = T.SPCMulti(g1, cell1.Ms, cell1.sc.hv_ex, Q1, cplx=True, kappa=kk)
        Qk, ik = spck.run(maxit=60)
        m0 = cell1.sc.score(T.project(cell1.Ms, Q1))
        m1 = cell1.sc.score(T.project(cell1.Ms, Qk))
        note(f'positive control single target {tag} kappa={kk}', E_near_svd=m0['E_near'], E_near_spcf=m1['E_near'], ratio=m1['E_near'] / m0['E_near'],
             f_ratio=ik['f'] / ik['f0'], E_out_ratio=m1['E_out'] / m0['E_out'])

# ---------------------------------------------------------------------------------------------- 5. CV solve and spectral function
N8 = 8
H8 = T.build_H('ising', N8)
Hd = H8.toarray()
ev, V = np.linalg.eigh(Hd)
c = 3
a0 = T._site_op(T.SZ, c, N8).real @ V[:, 0]
wp, wh, Sp, Sh = T.spectral_peak(H8, ev[0], a0, T.ETA, m=250)
# exact spectral function from the full eigendecomposition
om = np.linspace(0, ev[-1] - ev[0] + 1, 8001)
amp = (V.T @ a0) ** 2
Sw = ((amp[None, :] * (T.ETA / np.pi) / ((om[:, None] - (ev[None, :] - ev[0])) ** 2 + T.ETA ** 2)).sum(1))
check('spectral peak omega: Lanczos vs full ED (abs)', abs(wp - om[np.argmax(Sw)]), 2e-3)
xr, xi = T.solve_cv(H8, ev[0], a0, wp, T.ETA)
res = (Hd - (ev[0] + wp) * np.eye(len(a0)) - 1j * T.ETA * np.eye(len(a0))) @ (xr + 1j * xi) - a0
check('correction vector: |(H-E0-w-i eta) x - A|0>| (abs)', np.linalg.norm(res), 1e-9)
Gcv = a0 @ (xr + 1j * xi)
Gex = np.sum(amp / (ev - ev[0] - wp - 1j * T.ETA))
check('G(omega+i eta) from the CG correction vector vs the spectral sum (rel)', abs(Gcv - Gex) / abs(Gex), 1e-9)
check('S(omega) = Im G / pi at the peak vs Lanczos-weights spectral function (rel)', abs(Gcv.imag / np.pi - Sw[np.argmax(Sw)]) / Sw.max(), 2e-3)

# lowest eigenpairs from eigsh vs dense
e3, V3 = T.lowest(H8, 3)
check('eigsh lowest-3 energies vs dense ED (abs)', np.max(np.abs(e3 - ev[:3])), 1e-9)

# time evolution of the Neel state vs dense expm
import scipy.linalg as sl
tvs = [spla.expm_multiply((1.0 + j * 0.1 / 3) * (-1j) * H8, T.neel_dense(N8)) for j in range(4)]
exact = [sl.expm(-1j * (1.0 + j * 0.1 / 3) * Hd) @ T.neel_dense(N8) for j in range(4)]
check('Neel evolution (expm_multiply) vs dense expm (max abs)', max(np.linalg.norm(a - b) for a, b in zip(tvs, exact)), 1e-10)

# ---------------------------------------------------------------------------------------------- 6. dressed rho at a=0 equals arm (i); tfim_crit builder
cell = T.Cell('ising', N, 5, [s for s in sets if s.name == 'S3p'][0], sh)
Qa = T.arm_dressed(cell.Ms, cell.ts.w_lit, 6, 0.0, cell.g)
Qi, _ = cell.stack_Q(cell.ts.w_lit, 6)
check('dressed rho with a=0 spans the same subspace as the stacked SVD (|P_a - P_i|_F)', np.linalg.norm(Qa @ Qa.conj().T - Qi @ Qi.conj().T), 1e-8)
T.EXTRA_FIELDS['_tmp'] = (M.HX, M.HZ)
check('tfim builder with the ising fields equals exact_ref.sparse_H(ising) (max abs)', abs(T.build_H('_tmp', 8) - T.build_H('ising', 8)).max(), 1e-13)

note('summary', all_pass=bool(ok_all))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else HERE / 'results' / 'validation.json', 'w'), indent=1)
print('ALL PASS' if ok_all else 'SOME CHECKS FAILED')
