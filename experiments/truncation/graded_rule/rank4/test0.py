"""Test 0: plumbing and exactness.  Prints PASS/FAIL lines and writes results/test0.json."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import heis_mpo as H
import refs
import mpsenh as M
import pauli_prop as PP

out = {}
ok_all = True


def check(name, val, tol):
    global ok_all
    ok = bool(val <= tol)
    ok_all &= ok
    out[name] = dict(value=float(val), tol=tol, ok=ok)
    print(f'{"PASS" if ok else "FAIL"}  {name:<64s} {val:.3e}  (tol {tol:g})', flush=True)


# 1. dense marginals vs the independent exact_values of pauli_prop
N, dt, ns, i0 = 6, 0.1, 5, 2
marg = refs.dense_traj('ising', N, ns, dt)
ex = M.dense_reference('ising', N, ns, dt)
obs = [({i: p}, None) for i in range(N) for p in (1, 2, 3)]
exv = PP.exact_values(ex, N, obs).reshape(N, 3)
check('dense marginals vs pauli_prop.exact_values (N=6, n=5)', np.max(np.abs(exv - marg[ns])), 1e-12)

# 2. chi = infinity: MPO readout vs dense, every step, all three models, three observables' Paulis (only Z needs the
#    dense marg; X,Y checked via pauli_prop eps=0 at the final time)
for model in ('ising', 'ising2', 'heis'):
    marg = refs.dense_traj(model, N, ns, dt)
    for cutname, kind in (('svd_raw', None), ('svd_renorm', None), ('dmt', 'neel'), ('dmt', 'I')):
        cut = H.make_cut(cutname)
        ref = H.ref_provider(model, N, kind) if kind else None
        r = H.run_heis(model, N, i0, 10 ** 6, ns, dt, cut, ref)
        check(f'chi=inf {model} {cutname}:{kind} vs dense, max over steps', np.max(np.abs(r['val'] - marg[:, i0, 2])), 1e-12)
    for pauli in (1, 2, 3):
        r = H.run_heis(model, N, i0, 10 ** 6, ns, dt, H.make_cut('svd_raw'), None, pauli=pauli)
        pp = PP.propagate(model, N, ns, dt, {i0: pauli}, 0.0)[0]
        check(f'chi=inf {model} pauli={pauli} MPO vs PP eps=0 (final)', abs(r['val'][-1] - pp), 1e-12)
        exv = marg[ns, i0, pauli - 1]
        check(f'chi=inf {model} pauli={pauli} MPO vs dense (final)', abs(r['val'][-1] - exv), 1e-12)
    r = H.run_heis(model, N, i0, 10 ** 6, ns, dt, H.make_cut('svd_raw'), None)
    check(f'chi=inf {model} norm conserved', np.max(np.abs(r['norm'] - 1)), 1e-12)

# 3. finite chi: DMT preserved functionals (theta-level, every cut, whole run)
for model in ('ising', 'ising2', 'heis'):
    for kind in ('neel', 'I'):
        for (N2, chi, nsteps) in ((8, 8, 12), (10, 12, 12)):
            cut = H.make_cut('dmt', check=True)
            r = H.run_heis(model, N2, N2 // 2 - 1, chi, nsteps, 0.1, cut, H.ref_provider(model, N2, kind))
            v = cut.viol
            check(f'DMT-{kind} {model} N={N2} chi={chi}: max |functional_a violation|', v['a'], 1e-12)
            check(f'DMT-{kind} {model} N={N2} chi={chi}: max |functional_b violation|', v['b'], 1e-12)
            check(f'DMT-{kind} {model} N={N2} chi={chi}: max |Tr[sigma_ref delta]|', v['full'], 1e-12)
            out[f'ntrunc DMT-{kind} {model} N={N2} chi={chi}'] = cut.ntrunc
            rs = np.array(cut.reserved)
            out[f'reserved DMT-{kind} {model} N={N2} chi={chi}'] = dict(mean_rL=float(rs[:, 0].mean()) if len(rs) else None,
                                                                          mean_rR=float(rs[:, 1].mean()) if len(rs) else None,
                                                                          n=len(rs))
            print(f'      truncating cuts: {cut.ntrunc}, mean reserved (rL, rR) = '
                  f'{(rs.mean(0).round(2).tolist() if len(rs) else None)}')

# 4. independent dense-operator check of the same functionals through the hook (N=7, chi=9), plus negative controls:
#    SVD violates them, and a 4-site window (b-1..b+2) is NOT preserved by DMT (window = 3 sites by construction)
N3, chi3, ns3 = 7, 9, 8


def contract_sites(d, sites, v):
    """Contract the axes `sites` of the dense tensor d with covectors v[site] (highest axis first)."""
    for i in sorted(sites, reverse=True):
        d = np.tensordot(d, v[i], axes=([i], [0]))
    return d


for kind in ('neel', 'I'):
    for cutname in ('dmt', 'svd_raw'):
        model = 'ising'
        v = H.initial_covectors(model, N3) if kind == 'neel' else H.identity_covectors(N3)
        worst = dict(a=0.0, b=0.0, full=0.0, w4=0.0)

        def hook(b, dirn, step, snap, th, A, B):
            def full_op(mid):
                cur = None
                for i in range(b):
                    cur = snap[i][0] if cur is None else np.tensordot(cur, snap[i], axes=([-1], [0]))
                cur = mid[0] if cur is None else np.tensordot(cur, mid, axes=([-1], [0]))
                for i in range(b + 2, N3):
                    cur = np.tensordot(cur, snap[i], axes=([-1], [0]))
                return cur[..., 0]
            d = full_op(np.tensordot(A, B, axes=([2], [0]))) - full_op(th)       # (4,)*N3
            fa = contract_sites(d, range(b + 2, N3), v)
            fb = contract_sites(d, range(0, b), v)
            ff = contract_sites(d, range(N3), v)
            w4 = contract_sites(d, [i for i in range(N3) if i < b - 1 or i > b + 2], v)
            worst['a'] = max(worst['a'], np.max(np.abs(fa)))
            worst['b'] = max(worst['b'], np.max(np.abs(fb)))
            worst['full'] = max(worst['full'], abs(float(ff)))
            worst['w4'] = max(worst['w4'], np.max(np.abs(w4)))
        cut = H.make_cut(cutname)
        H.run_heis(model, N3, 3, chi3, ns3, 0.1, cut, H.ref_provider(model, N3, kind) if cutname == 'dmt' else None, hook=hook)
        tag = f'dense-op hook N={N3} chi={chi3} {cutname}-{kind}'
        if cutname == 'dmt':
            check(tag + ': functional_a', worst['a'], 1e-12)
            check(tag + ': functional_b', worst['b'], 1e-12)
            check(tag + ': Tr[sigma_ref delta]', worst['full'], 1e-12)
            out[tag + ': 4-site window b-1..b+2 (expected NONZERO)'] = float(worst['w4'])
            print(f'      4-site window violation (expected nonzero, negative control): {worst["w4"]:.3e}')
        else:
            out[tag + ' (expected NONZERO)'] = dict(a=float(worst['a']), b=float(worst['b']), full=float(worst['full']))
            print(f'      SVD violates (negative control): a={worst["a"]:.3e} b={worst["b"]:.3e} full={worst["full"]:.3e}')

# 5. a first look at accuracy at finite chi, N=10 (not a pass/fail)
N4, ns4 = 10, 20
marg = refs.dense_traj('ising', N4, ns4, 0.1)
print('N=10 ising T=2 Z_4, chi=8: max error over steps')
for name, kind in (('svd_raw', None), ('svd_renorm', None), ('dmt', 'I'), ('dmt', 'neel')):
    r = H.run_heis('ising', N4, 4, 8, ns4, 0.1, H.make_cut(name), H.ref_provider('ising', N4, kind) if kind else None)
    e = np.max(np.abs(r['val'] - marg[:, 4, 2]))
    print(f'   {name}:{kind}  err_max={e:.3e}  norm_T={r["norm"][-1]:.4f}')
    out[f'preview N=10 chi=8 {name}:{kind}'] = float(e)

out['all_ok'] = bool(ok_all)
json.dump(out, open(_HERE / 'results' / 'test0.json', 'w'), indent=1)
print('\nTEST 0:', 'ALL PASS' if ok_all else 'FAILURES PRESENT')
