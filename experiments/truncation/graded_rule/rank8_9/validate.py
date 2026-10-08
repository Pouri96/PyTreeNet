"""Validation of the pieces of the ranks 8/9 first move.  python rank8_9/validate.py > rank8_9/results/validation.txt"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
from functools import reduce
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import _paths  # noqa: E402,F401
os.environ.setdefault('SPCF_CACHE', str(HERE / '_cache'))
import mpsenh as M  # noqa: E402
import spcfast as S  # noqa: E402  (the stock module, unpatched)
import models as Mo  # noqa: E402
import mps_tools as MT  # noqa: E402
from spcfast_ext import SPCFastExt  # noqa: E402


def ok(msg, cond, val=''):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg} {val}")
    return cond


def main():
    # 1. fermionic bookkeeping: E = sum c R + const equals <H> for random states, L = 4 (8 qubits) and L = 8 ground state
    for basis in ('site', 'mo'):
        m = Mo.PPPModel(L=4, basis=basis)
        rng = np.random.default_rng(3)
        w = rng.normal(size=2 ** 8) + 1j * rng.normal(size=2 ** 8)
        w /= np.linalg.norm(w)
        e1 = (m.elem.coef * m.elem.values(w)).sum() + m.elem.const
        e2 = np.vdot(w, m.matvec(w))
        ok(f'PPP L=4 {basis}: E from elements vs <H> (random state, all sectors)', abs(e1 - e2) < 1e-10, f'{abs(e1 - e2):.1e}')
    # 2. independent JW check of the L = 2 PPP Hamiltonian spectrum, and basis invariance
    N = 4
    A = [reduce(np.kron, [np.diag([1., -1.])] * q + [np.array([[0, 1], [0, 0.]])] + [np.eye(2)] * (N - q - 1)) for q in range(N)]
    tt = np.array([[0, -2.6], [-2.6, 0.]])
    V12 = 14.397 / np.sqrt((14.397 / 11.26) ** 2 + 1.4 ** 2)
    n = [a.T @ a for a in A]
    ni = [n[0] + n[1], n[2] + n[3]]
    H = sum(tt[i, j] * A[2 * i + s].T @ A[2 * j + s] for i in range(2) for j in range(2) for s in range(2))
    H = H + 11.26 * (n[0] @ n[1] + n[2] @ n[3]) + V12 * (ni[0] - np.eye(16)) @ (ni[1] - np.eye(16))
    for basis in ('site', 'mo'):
        m = Mo.PPPModel(L=2, basis=basis)
        Hm = np.array([m.matvec(np.eye(16)[:, k]) for k in range(16)]).T
        ok(f'PPP L=2 {basis}: spectrum equals an independent Pauli-JW construction', np.allclose(np.sort(np.linalg.eigvalsh(H)), np.sort(np.linalg.eigvalsh((Hm + Hm.conj().T) / 2))))
    # 3. Pauli model equals mpsenh.h_bond
    pm = Mo.PauliModel(6, Mo.tfim_terms(6, 0.9045, 0.809), 't')
    Hs = sum(np.kron(np.kron(np.eye(2 ** b), M.h_bond('ising', b, 6)), np.eye(2 ** (6 - b - 2))) for b in range(5))
    ok('TFIM Pauli model equals mpsenh.h_bond sum', np.allclose(pm.H.toarray(), Hs))
    # 4. exact MPS, sweep with svd_cut equals the cut done independently on the dense state at the first bond
    v = pm.ground_state()
    T = MT.dense_to_right_mps(v, 6)
    ok('dense -> right-canonical MPS -> dense is exact', np.linalg.norm(MT.to_dense(T) - v) < 1e-12)
    # 5. patched SPCFast: with ext=None identical to the stock class (bitwise on the dense state), and with ext = own RDM equal to stock
    N = 12
    pm = Mo.PauliModel(N, Mo.tfim_terms(N, 1.0, 0.0), 'crit')
    v = pm.ground_state()
    outs = []
    for kind in ('stock', 'patched', 'patched_ext_self'):
        T = MT.dense_to_right_mps(v, N)
        if kind == 'stock':
            cut = S.SPCFast('ising', N, a=2, taus=(), fw=0.0)
        else:
            cut = SPCFastExt('ising', N, a=2, taus=(), fw=0.0)
        MT.sweep_compress(T, 3, cut)
        outs.append((MT.to_dense(T), cut.fired))
    ok('SPCFastExt(ext=None) is bit-for-bit the stock SPCFast', np.array_equal(outs[0][0], outs[1][0]), f'fired {outs[0][1]} vs {outs[1][1]}')
    # ext hook wiring: if the external target is the window RDM of the *current* MPS (= the pre-cut state at the moment of the call),
    # the sweep must reproduce the stock one (checks site order, normalisation and the (lo, hi) convention of the hook)
    T = MT.dense_to_right_mps(v, N)
    cut = SPCFastExt('ising', N, a=2, taus=(), fw=0.0)
    calls = []

    def ext_self(lo, hi):
        calls.append((lo, hi))
        return Mo.rdm_sites(MT.to_dense(T), N, list(range(lo, hi)))
    cut.ext = ext_self
    MT.sweep_compress(T, 3, cut)
    dd = np.abs(MT.to_dense(T) - outs[0][0]).max()
    ok('ext hook given the RDM of the current MPS reproduces the stock sweep', dd < 1e-6 and cut.fired == outs[0][1] and len(calls) > 0,
       f'max |diff| {dd:.1e}, fired {cut.fired} vs {outs[0][1]}, hook called {len(calls)}x')
    # and with the exact reference RDM the sweep differs (the hook does something)
    T = MT.dense_to_right_mps(v, N)
    cut = SPCFastExt('ising', N, a=2, taus=(), fw=0.0)
    cut.ext = lambda lo, hi: Mo.rdm_sites(v, N, list(range(lo, hi)))
    MT.sweep_compress(T, 3, cut)
    ok('ext hook given the exact reference RDM changes the result', np.abs(MT.to_dense(T) - outs[0][0]).max() > 1e-8,
       f'max |diff| {np.abs(MT.to_dense(T) - outs[0][0]).max():.1e}')
    # 6. cut equivalence to M.svd_cut when spcf never fires
    T1 = MT.dense_to_right_mps(v, N)
    MT.sweep_compress(T1, 3, M.svd_cut)
    T2 = MT.dense_to_right_mps(v, N)
    never = SPCFastExt('ising', N, a=2, taus=(), fw=0.0, f_min=1e9)
    MT.sweep_compress(T2, 3, never)
    ok('spcf with f_min = inf reproduces the SVD sweep bit for bit', np.array_equal(MT.to_dense(T1), MT.to_dense(T2)), f'fired {never.fired}')
    # 7. variational fit never lowers the fidelity; DMRG energy above ED and close
    T = MT.dense_to_right_mps(v, N)
    MT.sweep_compress(T, 3, M.svd_cut)
    f0 = abs(np.vdot(v, MT.to_dense(T))) ** 2
    Tf, hist = MT.var_fit(v, T, 3)
    ok('variational fit: fidelity non-decreasing from the SVD sweep', all(h >= f0 - 1e-12 for h in hist) and all(b >= a - 1e-12 for a, b in zip(hist, hist[1:])),
       f'infid {1 - f0:.3e} -> {[f"{1 - h:.3e}" for h in hist]}')
    W = MT.nn_mpo(N, pm.terms)
    Td, h = MT.dmrg2(W, N, 16, nsweeps=20)
    d = MT.to_dense(Td)
    ok('DMRG(chi=16) energy within 1e-6 of ED and not below it', -1e-9 < np.vdot(d, pm.matvec(d)).real - pm.e0 < 1e-6, f'{np.vdot(d, pm.matvec(d)).real - pm.e0:.2e}')
    ph = Mo.PauliModel(N, Mo.heis_terms(N), 'h')
    ph.ground_state()
    Td, h = MT.dmrg2(MT.nn_mpo(N, ph.terms), N, 24, nsweeps=25)
    d = MT.to_dense(Td)
    ok('Heisenberg DMRG(chi=24) reaches the singlet ground state', abs(np.vdot(d, ph.matvec(d)).real - ph.e0) < 1e-6, f'{np.vdot(d, ph.matvec(d)).real - ph.e0:.2e}')
    # 8. L = 8 ground states: basis invariance of E0, number sector, residual
    es = []
    for basis in ('site', 'mo'):
        m = Mo.PPPModel(L=8, basis=basis)
        v8 = m.ground_state()
        es.append(m.e0)
        nm, nv, sm, sv = m.number_stats(v8)
        e_el = (m.elem.coef * m.elem.values(v8)).sum().real + m.elem.const
        ok(f'PPP L=8 {basis}: residual, <N>, Var N, E(elements)', m.resid < 1e-10 and abs(nm - 8) < 1e-9 and nv < 1e-10 and abs(e_el - m.e0) < 1e-9,
           f'resid {m.resid:.1e}, <N> {nm:.12f}, VarN {nv:.1e}, E_el-E0 {e_el - m.e0:.1e}, gap {m.gap:.3f}')
    ok('PPP L=8: E0 identical in site and MO basis', abs(es[0] - es[1]) < 1e-9, f'{es[0]:.10f} {es[1]:.10f}')


if __name__ == '__main__':
    main()
