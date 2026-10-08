"""Supplementary diagnostics on the A1 grid (not part of the pre-registered criteria; reported separately).

    python rank5/supp.py dressed  model b      # dressed rho with smaller a (tail-scaled) for every stored cell
    python rank5/supp.py effw     model b      # is the spcf-multi basis an implicit reweighting of the stacked SVD?
    python rank5/supp.py kappa    model b sets # kappa sweep (0.1 relative, 0.3, 1.0, common-budget 0.1) of arm (iv)

Reads results/a1_{model}_N12_b{b}.json and the stored shared bases, writes results/supp_{kind}_{model}_N12_b{b}.json.
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
import mt_cut as T  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

RES = HERE / 'results'
SMALL_A = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)


def load_cells(model, b, N=12, tag=''):
    rec = json.load(open(RES / f'a1_{model}_N{N}_b{b}{tag}.json'))
    bases = np.load(RES / f'a1_{model}_N{N}_b{b}{tag}_bases.npz')
    tsets, sh = T.make_sets(model, N)
    return rec, bases, {t.name: t for t in tsets}, sh


def dressed_small(model, b, N=12):
    rec, bases, tsets, sh = load_cells(model, b, N)
    out = []
    for c in rec['cells']:
        ts = tsets[c['set']]
        cell = T.Cell(model, N, b, ts, sh)
        chi = c['chi']
        row = dict(set=c['set'], chi=chi, E_near_i=c['i']['m']['E_near'], E_near_iii_prereg=c['iii']['m']['E_near'], E_near_iv=c['iv']['m']['E_near'])
        tab = {}
        for a in SMALL_A:
            Q = cell.dressed(chi, a)
            tab[a] = cell.e_near_fast(T.project(cell.Ms, Q))
        row['table'] = tab
        ab = min(tab, key=tab.get)
        row['best_small_a'], row['E_near_iii_small'] = ab, tab[ab]
        # best over small and pre-registered: also the E_out and fidelity of the best-small basis
        Q = cell.dressed(chi, ab)
        m = cell.sc.score(T.project(cell.Ms, Q))
        row['m_small'] = {k: m[k] for k in ('E_near', 'E_out', 'eps')}
        out.append(row)
        print(f'{model} b={b} {c["set"]:5s} chi={chi}: i {row["E_near_i"]:.3e}  iii(prereg) {row["E_near_iii_prereg"]:.3e}  iii(small a={ab:g}) {row["E_near_iii_small"]:.3e}  iv {row["E_near_iv"]:.3e}', flush=True)
    json.dump(T.to_jsonable(out), open(RES / f'supp_dressed_{model}_N{N}_b{b}.json', 'w'))


def proj_dist(Qa, Qb):
    return float(np.linalg.norm(Qa @ Qa.conj().T - Qb @ Qb.conj().T))


def effw(model, b, N=12):
    """Find the weight vector w whose stacked SVD basis is closest (projector Frobenius distance) to the spcf-multi basis,
    and the weight vector whose per-target discarded weights match those of spcf-multi.  If the reweighted SVD reproduces
    E_near(iv), the tilt is an implicit reweighting."""
    rec, bases, tsets, sh = load_cells(model, b, N)
    out = []
    for c in rec['cells']:
        if c['set'] == 'S1':
            continue
        ts = tsets[c['set']]
        cell = T.Cell(model, N, b, ts, sh)
        chi = c['chi']
        Qiv = bases[f'{c["set"]}|{chi}|iv']
        Qi = bases[f'{c["set"]}|{chi}|i']
        F = ts.F

        def Qw(z):
            w = np.exp(z - z.max())
            w = w / w.sum()
            return cell.stack_Q(w, chi)[0], w

        best = None
        for z0 in [np.log(ts.w_lit)] + [np.log(np.eye(F)[i] * 0.9 + 0.1 / F) for i in range(F)]:
            r = minimize(lambda z: proj_dist(Qw(z)[0], Qiv), z0, method='Nelder-Mead', options=dict(maxiter=400, xatol=1e-4, fatol=1e-10))
            if best is None or r.fun < best.fun:
                best = r
        Qb, wb = Qw(best.x)
        d_i = proj_dist(Qi, Qiv)
        mb = cell.sc.score(T.project(cell.Ms, Qb))
        row = dict(set=c['set'], chi=chi, dist_stack_to_iv=d_i, dist_best_w_to_iv=float(best.fun), w=wb.tolist(),
                   explained=1.0 - float(best.fun) / d_i if d_i > 0 else None,
                   E_near_i=c['i']['m']['E_near'], E_near_iv=c['iv']['m']['E_near'], E_near_bestw=mb['E_near'],
                   eps_iv=c['iv']['m']['eps'], eps_i=c['i']['m']['eps'], eps_bestw=mb['eps'])
        out.append(row)
        print(f'{model} b={b} {c["set"]:5s} chi={chi}: proj-dist stack->iv {d_i:.3e}, best-w->iv {best.fun:.3e} (explained {row["explained"]:.2f}); '
              f'E_near i {row["E_near_i"]:.3e} iv {row["E_near_iv"]:.3e} bestw {mb["E_near"]:.3e}  w={np.round(wb, 3)}', flush=True)
    json.dump(T.to_jsonable(out), open(RES / f'supp_effw_{model}_N{N}_b{b}.json', 'w'))


def kappa_sweep(model, b, sets, N=12, kappas=(0.1, 0.3, 1.0), maxit=40):
    rec, bases, tsets, sh = load_cells(model, b, N)
    out = []
    for c in rec['cells']:
        if c['set'] not in sets:
            continue
        ts = tsets[c['set']]
        cell = T.Cell(model, N, b, ts, sh)
        chi = c['chi']
        Qi = bases[f'{c["set"]}|{chi}|i']
        mi = c['i']['m']
        row = dict(set=c['set'], chi=chi, E_near_i=mi['E_near'], E_out_i=mi['E_out'], E_near_ii_plus=c['ii_plus']['m']['E_near'])
        for kap in kappas:
            for bud in ('rel', 'common'):
                if bud == 'common' and kap != 0.1:
                    continue
                spc = T.SPCMulti(cell.g, cell.Ms, cell.sc.hv_ex, Qi, cplx=ts.cplx, kappa=kap, budget=bud)
                Q, info = spc.run(maxit=maxit)
                m = cell.sc.score(T.project(cell.Ms, Q))
                row[f'{bud}_{kap:g}'] = dict(E_near=m['E_near'], E_out=m['E_out'], f_ratio=info['f'] / info['f0'], g_f=info['g_f'], eps=m['eps'])
        out.append(row)
        print(f'{model} b={b} {c["set"]:5s} chi={chi}: ' + '  '.join(f'{k} {v["E_near"] / mi["E_near"]:.2f}' for k, v in row.items() if isinstance(v, dict)), flush=True)
        json.dump(T.to_jsonable(out), open(RES / f'supp_kappa_{model}_N{N}_b{b}.json', 'w'))


if __name__ == '__main__':
    kind, model, b = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if kind == 'dressed':
        dressed_small(model, b)
    elif kind == 'effw':
        effw(model, b)
    elif kind == 'kappa':
        kappa_sweep(model, b, sys.argv[4].split(','))
