"""Checks of spcflpdo.py: adjoint pair and finite-difference linearisation at K>1 cuts; K = 1 equivalence with rule/spcfast.SPCFast (mps_bench-style spcf)."""
import _p6  # noqa: F401
import json
import numpy as np
import mpsenh as M
import spcfast
import lpdo as L
import spcflpdo as S

out = {}
rng = np.random.default_rng(1)
model, N, dt = 'ising', 8, 0.1
BEST = dict(a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)

# 1. adjoint + linearisation at K>1 cuts
cut = S.SPCFLpdo(model, N, **BEST)
cut.debug = []
L.run_lpdo(model, N, 6, 3, 25, dt, 0.05, cut=cut)
print('cuts', cut.calls, 'fired', cut.fired, 'debug', len(cut.debug), 'K sizes seen', sorted({(d['K1'], d['K2']) for d in cut.debug}))
wa, wl = 0.0, 0.0
for d in cut.debug:
    x = rng.normal(size=d['nx']); g = rng.normal(size=d['nres'])
    lhs, rhs = g @ d['jvp'](x), d['vjp'](g) @ x
    wa = max(wa, abs(lhs - rhs) / (abs(lhs) + 1e-30))
print('adjoint rel err (max)', wa)
out['adjoint_max_rel'] = wa
for rel in (1e-1, 1e-2, 1e-3, 1e-4):
    errs = []
    for d in cut.debug[::3]:
        x = rng.normal(size=d['nx']); x *= rel / np.linalg.norm(x)
        num = d['exact'](x) - d['r0']
        ana = d['jvp'](x)
        errs.append(np.linalg.norm(num - ana) / np.linalg.norm(ana))
    print('step', rel, 'median linearisation rel err', np.median(errs), 'max', np.max(errs))
    out[f'lin_{rel:g}'] = dict(median=float(np.median(errs)), max=float(np.max(errs)))
# 1b. same adjoint check in float64 (the f32 value above is limited by single precision, as in rule/spcfast.py itself: 5.9e-5 in test_spcfast.py)
cut64 = S.SPCFLpdo(model, N, precision='f64', **BEST)
cut64.debug = []
L.run_lpdo(model, N, 6, 3, 25, dt, 0.05, cut=cut64)
wa64 = 0.0
for d in cut64.debug:
    x = rng.normal(size=d['nx']); g = rng.normal(size=d['nres'])
    lhs, rhs = g @ d['jvp'](x), d['vjp'](g) @ x
    wa64 = max(wa64, abs(lhs - rhs) / (abs(lhs) + 1e-30))
print('adjoint rel err (max), float64', wa64)
out['adjoint_max_rel_f64'] = wa64
# 2. K = 1 equivalence: gamma = 0, LPDO + SPCFLpdo versus MPS + SPCFast, same config
for chi in (6, 8):
    c1 = spcfast.SPCFast(model, N, **BEST)
    Tm, _ = M.run_tebd(model, N, chi, 40, dt, c1)
    c2 = S.SPCFLpdo(model, N, **BEST)
    Tl, info = L.run_lpdo(model, N, chi, 2, 40, dt, 0.0, cut=c2)
    vm, vl = M.mps_to_dense(Tm), M.mps_to_dense([t[:, :, 0, :] for t in Tl])
    ov = abs(np.vdot(vm, vl))
    ex = M.dense_reference(model, N, 40, dt)
    em = M.errors(ex, vm, N, model)['nn_rms']; el = M.errors(ex, vl, N, model)['nn_rms']
    print(f'K=1 chi={chi}: fired mps {c1.fired}/{c1.calls} lpdo {c2.fired}/{c2.calls}  |<m|l>|={ov:.10f}  nn_rms mps {em:.4e} lpdo {el:.4e}')
    out[f'K1_chi{chi}'] = dict(fired_mps=c1.fired, fired_lpdo=c2.fired, overlap=float(ov), nn_mps=float(em), nn_lpdo=float(el))
json.dump(out, open('results/test_spcflpdo.json', 'w'), indent=1)
