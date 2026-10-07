import sys, os, json, time
import _paths  # noqa: F401  (this repository's pytreenet and rule/ first)
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import multiprocessing as mp
from fractions import Fraction
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import rage_rule as RR
import mpsenh as M
from _chain import product_mps
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.ttno.ttno_class import TTNO
from _mps_vendor import RAGE_MPS, RAGEMPSConfig, TimeEvoMode

OPS = {'X': M.X, 'Y': M.Y, 'Z': M.Z}


def init_indices(model, N):
    if model == 'isingdw':
        return [0 if i < N // 2 else 1 for i in range(N)]
    return [i % 2 for i in range(N)]


def build(model, N):
    ids = [f'qubit{i}' for i in range(N)]
    template = product_mps(init_indices(model, N), 2, 1, node_prefix='qubit', root_site=0)
    ham = Hamiltonian()
    ham.conversion_dictionary.update(OPS)
    terms = []                                               # (coeff value, symbol, {site: op})
    if model in M.FIELDS:
        hx, hz = M.FIELDS[model]
        for i in range(N - 1):
            terms.append((1.0, 'J', {ids[i]: 'Z', ids[i + 1]: 'Z'}))
        for i in range(N):
            terms.append((hx, 'hx', {ids[i]: 'X'}))
            terms.append((hz, 'hz', {ids[i]: 'Z'}))
    else:
        for i in range(N - 1):
            for P in 'XYZ':
                terms.append((1.0, 'J', {ids[i]: P, ids[i + 1]: P}))
    for v, c, ops in terms:
        ham.add_term((Fraction(1), c, TensorProduct(dict(ops))))
        ham.coeffs_mapping[c] = complex(v)
    ham.include_identities(template)
    ttno = TTNO.from_hamiltonian(ham, template)
    # dense sparse H in site order 0 = most significant
    H = sp.csr_matrix((2 ** N, 2 ** N), dtype=complex)
    for v, _, ops in terms:
        mat = None
        for i in range(N):
            loc = sp.csr_matrix(OPS[ops[ids[i]]]) if ids[i] in ops else sp.identity(2, dtype=complex, format='csr')
            mat = loc if mat is None else sp.kron(mat, loc, format='csr')
        H = H + v * mat
    return ids, template, ttno, H.tocsr()


def exact_vec(model, N, H, T):
    v0 = M.mps_to_dense(M.initial_mps(model, N))
    return expm_multiply(-1j * T * H, v0)


def state_vector(state, N):
    return M.mps_to_dense([RR.to_lpr(state, f'qubit{i}', N) for i in range(N)])


def stored(state):
    return int(sum(np.asarray(state.tensors[n]).size for n in state.nodes))


def job(a):
    model, N, T, dt, arm, chi = a
    ids, template, ttno, H = build(model, N)
    ex = exact_vec(model, N, H, T)
    cfg = RAGEMPSConfig(max_bond_dim=chi, rel_tol=1e-12, total_tol=1e-12, max_aug_bond_dim=(np.inf if not os.environ.get('AUG') else int(float(os.environ['AUG']) * chi)),
                        max_segment_bond=np.inf, max_segment_width=None, hermitian=True,
                        warmup_sweeps=2, warmup_every_step=None,
                        galerkin_evo_mode=TimeEvoMode.KRYLOV, solver_options={'krylov_tol': 1e-10})
    if arm == 'lib':
        RR.uninstall()
    elif arm == 'svd':
        RR.install(model, N, 0.0)
    else:
        p = arm.split(':')
        RR.install(model, N, float(p[1]), wE=float(p[3]), ndir=int(p[5]))
    nsteps = int(round(T / dt))
    solver = RAGE_MPS(template, ttno, dt, T, [], config=cfg)
    t0 = time.time()
    for _ in range(nsteps):
        solver.run_one_time_step()
    ap = state_vector(solver.state, N)
    e = M.errors(ex, ap, N, model)
    e.update(arm=arm, chi=chi, params=stored(solver.state), wall=round(time.time() - t0, 1),
             cuts=RR.STATE['cuts'], fired=getattr(RR.STATE['cut'], 'fired', 0) if arm not in ('lib',) else 0)
    return e


if __name__ == '__main__':
    model, N, T, dt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    arms = sys.argv[5].split(',')
    chis = [int(c) for c in sys.argv[6].split(',')]
    out = sys.argv[7]; nproc = int(sys.argv[8])
    jobs = [(model, N, T, dt, a, c) for c in chis for a in arms]
    res = []
    t0 = time.time()
    with mp.Pool(nproc) as p:
        for r in p.imap_unordered(job, jobs):
            res.append(r); json.dump(res, open(out, 'w'))
            print(f"{len(res)}/{len(jobs)} chi={r['chi']} {r['arm']} P={r['params']} infid={r['infid']:.3e} nn={r['nn_rms']:.2e} E={r['E_abs']:.2e} fired={r['fired']}/{r['cuts']} {int(time.time()-t0)}s", flush=True)
