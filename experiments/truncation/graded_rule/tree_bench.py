import sys, os, json, time
import _paths  # noqa: F401  (this repository's pytreenet and rule/ first)
HERE = os.path.dirname(os.path.abspath(__file__))
# The tree arm reuses rank_allocation's spin-boson cells, models and BUG integrator rather than copying them.
RA = os.path.join(os.path.dirname(HERE), "rank_allocation")
sys.path.insert(1, RA)
os.chdir(RA)
import numpy as np
import multiprocessing as mp
from _rage_vendor import BUG
import cells, run
import tree_rule as TR
from models import spin_boson as sb
from pytreenet.operators.common_operators import bosonic_operators, pauli_matrices

CELL = 'six_wide'


def op_ttnos(cell, rig, with_energy=True):
    from fractions import Fraction
    from pytreenet.operators.hamiltonian import Hamiltonian
    from pytreenet.operators.tensorproduct import TensorProduct
    from pytreenet.ttno.ttno_class import TTNO
    dims, delta = cell.spec['dims'], cell.spec['delta']
    w, g = sb.ohmic_modes(len(dims), cell.spec.get('alpha', 2.8))
    model = sb.SpinBoson(w, g, dims, delta=delta)
    edges, root = sb.mode_tree(sb.balanced_assignment(model))
    template = sb.build_state(edges, root, model.site_dims)
    mats = model.operators()
    nrm = {k: float(np.linalg.norm(v, 2)) for k, v in mats.items()}
    specs = [{sb.SPIN: 'sx'}, {sb.SPIN: 'sz'}]
    for k, d in enumerate(dims):
        lab = model.mode_labels[k]
        specs += [{lab: f'n{d}'}, {lab: f'x{d}'}, {sb.SPIN: 'sz', lab: f'x{d}'}]
    out = []
    for sp in specs:
        ham = Hamiltonian()
        ham.conversion_dictionary.update(mats)
        ham.add_term((Fraction(1), 'c', TensorProduct(dict(sp))))
        ham.coeffs_mapping['c'] = complex(1.0)
        ham.include_identities(template)
        sc = float(np.prod([nrm[v] for v in sp.values()]))
        out.append((TTNO.from_hamiltonian(ham, template), 1.0 / sc ** 2))
    if with_energy:
        hn = sum(abs(v) * float(np.prod([nrm[o] for o in ops.values()])) for v, _, ops in model.terms())
        out.append((rig.ttno, 1.0 / hn ** 2))
    return out


def observables(v, dims, w, g, delta):
    v = v / np.linalg.norm(v)
    T = v.reshape(dims)
    sx, _, sz = pauli_matrices()
    out = {}
    M = T.reshape(dims[0], -1)
    rs = M @ M.conj().T
    out['sz'] = float(np.trace(rs @ sz).real); out['sx'] = float(np.trace(rs @ sx).real)
    n = np.zeros(len(w)); x = np.zeros(len(w)); zx = np.zeros(len(w))
    for k in range(len(w)):
        d = dims[1 + k]
        cr, an, num = bosonic_operators(dimension=d)
        X = np.asarray(cr + an); Nn = np.asarray(num)
        Mk = np.moveaxis(T, [0, 1 + k], [0, 1]).reshape(dims[0] * d, -1)
        rho = (Mk @ Mk.conj().T).reshape(dims[0], d, dims[0], d)
        rm = np.einsum('sasb->ab', rho)
        n[k] = np.trace(rm @ Nn).real; x[k] = np.trace(rm @ X).real
        zx[k] = np.einsum('sast,st,ab->', rho.transpose(0, 1, 2, 3), np.zeros((2, 2)), np.zeros((d, d))) if False else \
            np.einsum('sata,ts->', np.einsum('sjtk,jk->stjk', rho, np.zeros((d, d))) if False else np.zeros((2, 2, d, d)), np.zeros((2, 2))) if False else \
            float(np.einsum('sjtk,st,kj->', rho, np.asarray(sz), X).real)
    E = delta / 2 * out['sx'] + float(np.dot(w, n)) + float(np.dot(g, zx))
    out.update(n=n, x=x, zx=zx, E=E)
    return out


def errors(o, e):
    return dict(infid=None, sz_err=abs(o['sz'] - e['sz']), sx_err=abs(o['sx'] - e['sx']),
                n_rms=float(np.sqrt(np.mean((o['n'] - e['n']) ** 2))),
                x_rms=float(np.sqrt(np.mean((o['x'] - e['x']) ** 2))),
                zx_rms=float(np.sqrt(np.mean((o['zx'] - e['zx']) ** 2))),
                E_abs=abs(o['E'] - e['E']))


def job(a):
    arm, chi = a
    cell = cells.CELLS[CELL]; rig = cell.rig()
    dims = [2] + cell.spec['dims']
    w, g = sb.ohmic_modes(len(cell.spec['dims']), cell.spec.get('alpha', 2.8))
    delta = cell.spec['delta']
    ex = observables(rig.exact, dims, w, g, delta)
    if arm == 'svd':
        TR.configure(rig.ttno, kappa=0.0)
    elif arm.startswith('rule'):
        kappa = float(arm.split(':')[1])
        TR.configure(rig.ttno, kappa=kappa)
    else:
        kappa = float(arm.split(':')[1])
        TR.configure(rig.ttno, kappa=kappa, ops=op_ttnos(cell, rig, with_energy=(arm.startswith('obs:'))))
    TR.install()
    t0 = time.time()
    solver = BUG(rig.start, rig.ttno, rig.dt, rig.dt * rig.nsteps, [], config=run._config(rig, chi=chi))
    for _ in range(rig.nsteps):
        solver.run_one_time_step()
    vec = rig.vector(solver.state)
    r = errors(observables(vec, dims, w, g, delta), ex)
    r['infid'] = rig.infidelity(solver.state)
    r.update(arm=arm, chi=chi, params=sb.parameters(solver.state), bond=sb.widest_bond(solver.state),
             fired=TR.STATE['fired'], cuts=TR.STATE['cuts'], wall=time.time() - t0)
    return r


if __name__ == '__main__':
    arms = sys.argv[1].split(',')
    chis = [int(c) for c in sys.argv[2].split(',')]
    out = sys.argv[3]; nproc = int(sys.argv[4])
    jobs = [(a, c) for c in chis for a in arms]
    res = []
    t0 = time.time()
    with mp.Pool(nproc) as p:
        for r in p.imap_unordered(job, jobs):
            res.append(r); json.dump(res, open(out, 'w'))
            print(f"{len(res)}/{len(jobs)} chi={r['chi']} {r['arm']} infid={r['infid']:.3e} nn? zx={r['zx_rms']:.2e} E={r['E_abs']:.2e} fired={r['fired']}/{r['cuts']} {int(time.time()-t0)}s", flush=True)
