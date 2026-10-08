"""Correctness checks, part B (tilt): window marginals from the region tensor against dense marginals, the adjoint pair and the finite-difference
linearisation of the tilt Jacobian at every recorded cut, kappa=0-style identity (tilt off == svd bit for bit), residual decrease, canonical form."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import op_tebd as O
import spcop

rng = np.random.default_rng(1)
ok_all = True


def check(name, val, tol):
    global ok_all
    ok = bool(val <= tol)
    ok_all &= ok
    print(f'{"PASS" if ok else "FAIL"}  {name}: {val:.3e} (tol {tol:.0e})', flush=True)


model, N, chi, nsteps, dt = 'ising', 8, 4, 25, 0.1
gops = O.pauli_gates(model, N, dt)

# --- run the tilt with debug recording
cut = spcop.SPCOp(N, aL=2, aR=2, iters=4, rel_skip=0.0)
cut.debug = []
T, _ = O.run_op_tebd(N, chi, nsteps, gops, cut)
print('cuts', cut.calls, 'truncating', cut.trunc, 'fired', cut.fired, 'rejected', cut.rejected, 'skipped', cut.skipped, 'debug', len(cut.debug), flush=True)

# 1. region marginal map against dense marginals of the state with theta spliced in
worst = 0.0
for d in cut.debug[::7]:
    Tc, b = d['Tcopy'], d['b']
    th = d['theta']
    Tm = [t for t in Tc[:b]] + [th.reshape(th.shape[0], 16, th.shape[3])] + [t for t in Tc[b + 2:]]
    c = O.mps_to_dense(Tm)
    c = c / np.linalg.norm(c)
    reg = d['region']
    lo, hi, L = reg['lo'], reg['hi'], reg['L']
    # region label distribution from W vs dense marginal over sites lo..hi-1
    W = reg['Wof'](th)
    h = np.einsum('ij,ij->i', W, W)
    hd = O.marginal_sites(c, N, list(range(lo, hi)))
    worst = max(worst, np.abs(h / h.sum() - hd).max())
    # window rows
    rows = d['Msp'] @ (h / h.sum())
    ridx = 0
    for k in cut.ks:
        nwin = L - k + 1
        for off in range(nwin):
            pm = O.marginal_sites(c, N, list(range(lo + off, lo + off + k)))
            w = np.sqrt(cut.wk[k] / nwin)
            worst = max(worst, np.abs(rows[ridx:ridx + 4 ** k] / w - pm).max())
            ridx += 4 ** k
check('region h and 360-row window marginals == dense marginals (spliced theta)', worst, 1e-12)

# 2. adjoint pair and linearisation at every recorded cut
worst_adj, worst_lin = 0.0, 0.0
for d in cut.debug:
    x = rng.normal(size=d['nx'])
    g = rng.normal(size=d['nres'])
    lhs, rhs = g @ d['jvp'](x), d['vjp'](g) @ x
    worst_adj = max(worst_adj, abs(lhs - rhs) / (abs(lhs) + 1e-30))
    eps = 1e-6 / np.linalg.norm(x)
    num = (d['exact'](eps * x) - d['r0']) / eps
    ana = d['jvp'](x)
    worst_lin = max(worst_lin, np.linalg.norm(num - ana) / np.linalg.norm(ana))
check(f'adjoint <g,Jx> = <J^T g,x> rel err (max over {len(cut.debug)} cuts)', worst_adj, 1e-10)
check('finite-difference linearisation rel err (step 1e-6, max)', worst_lin, 1e-4)
for rel in (1e-1, 1e-2, 1e-3, 1e-4):
    errs = []
    for d in cut.debug[::3]:
        x = rng.normal(size=d['nx'])
        x *= rel / np.linalg.norm(x)
        num = d['exact'](x) - d['r0']
        ana = d['jvp'](x)
        errs.append(np.linalg.norm(num - ana) / np.linalg.norm(ana))
    print(f'  step {rel:g}: median linearisation rel err {np.median(errs):.2e}  max {np.max(errs):.2e}')

# 3. Wadj is the adjoint of Wof
d = cut.debug[len(cut.debug) // 2]
th = rng.normal(size=(d['l'], 4, 4, d['r']))
G = rng.normal(size=d['region']['Wof'](th).shape)
check('<G, Wof(th)> = <Wadj(G), th>', abs(np.sum(G * d['Wof'](th)) - np.sum(d['Wadj'](G) * th)) / abs(np.sum(G * d['Wof'](th))), 1e-12)

# 4. residual decrease at fired cuts, accepted step is also not worse in kept weight than SVD would allow (kept_c <= kept_svd by construction)
lg = np.array(cut.log)
fired = lg[:, 9] == 1
print(f'  fired cuts {fired.sum()}; f_final/f_svd median {np.median(lg[fired, 11] / lg[fired, 10]):.3f}, max {np.max(lg[fired, 11] / lg[fired, 10]):.3f}')
check('residual decreases at every fired cut (max f_final/f_svd - 1)', max(0.0, np.max(lg[fired, 11] / lg[fired, 10]) - 1), 0.0)

# 5. tilt off == svd bit for bit; diag-only OpCut == svd bit for bit; large chi == dense
Ts, _ = O.run_op_tebd(N, chi, nsteps, gops, O.svd_cut_op)
cs = O.mps_to_dense(Ts)
c_off = O.mps_to_dense(O.run_op_tebd(N, chi, nsteps, gops, spcop.SPCOp(N, tilt=False, rel_skip=0.0))[0])
c_diag = O.mps_to_dense(O.run_op_tebd(N, chi, nsteps, gops, spcop.OpCut(N, diag=True))[0])
check('SPCOp(tilt=False) == svd (max abs diff, bit for bit)', np.abs(c_off - cs).max(), 0.0)
check('OpCut(diag=True) == svd (max abs diff, bit for bit)', np.abs(c_diag - cs).max(), 0.0)
cinf = O.mps_to_dense(O.run_op_tebd(N, 4 ** (N // 2), nsteps, gops, spcop.SPCOp(N))[0])
p1d, cz0d, snaps = O.dense_series(model, N, nsteps, dt, snap_steps=(nsteps,))
check('SPCOp at chi = inf == dense', np.abs(cinf - snaps[nsteps]).max(), 1e-11)
c_tilt = O.mps_to_dense(T)
print(f'  tilt vs svd state differ by {np.abs(c_tilt / np.linalg.norm(c_tilt) - cs / np.linalg.norm(cs)).max():.3e} (nonzero: the tilt acts)')

# 6. canonical form after the run (centre at site 0), unit norm
orth = 0.0
for j in range(1, N):
    t = T[j]
    m = t.reshape(t.shape[0], -1)
    orth = max(orth, np.abs(m @ m.T - np.eye(m.shape[0])).max())
check('right-isometry of T[1:] after a tilt run', orth, 1e-12)
check('||c|| = 1 after a tilt run', abs(np.linalg.norm(O.mps_to_dense(T)) - 1), 1e-12)

# 7. same checks with ks=(1,) and a=1 and the other direction conventions covered above (dirn R and L both occur)
cut1 = spcop.SPCOp(N, aL=1, aR=1, ks=(1, 2), iters=4, rel_skip=0.0, passes=2)
cut1.debug = []
T1, _ = O.run_op_tebd(N, chi, 20, gops, cut1)
wa, wl = 0.0, 0.0
for d in cut1.debug:
    x = rng.normal(size=d['nx'])
    g = rng.normal(size=d['nres'])
    lhs, rhs = g @ d['jvp'](x), d['vjp'](g) @ x
    wa = max(wa, abs(lhs - rhs) / (abs(lhs) + 1e-30))
    eps = 1e-6 / np.linalg.norm(x)
    wl = max(wl, np.linalg.norm((d['exact'](eps * x) - d['r0']) / eps - d['jvp'](x)) / np.linalg.norm(d['jvp'](x)))
check(f'a=1, ks=(1,2), passes=2: adjoint ({len(cut1.debug)} cuts)', wa, 1e-10)
check('a=1, ks=(1,2), passes=2: linearisation', wl, 1e-4)
print('ALL PASS' if ok_all else 'SOME FAILED')
