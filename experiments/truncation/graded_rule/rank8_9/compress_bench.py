"""Ranks 8/9 first move: post-hoc compression of an exact ground state, scored on all 1+2-body RDM elements.

    python rank8_9/compress_bench.py selftest
    python rank8_9/compress_bench.py t0 OUT.json [CELL ...]               # SVD sweep ladder, locality budget of the error
    python rank8_9/compress_bench.py t1 OUT.json CELL CHI[,CHI..] ARM[,ARM..]   # arms at matched chi
        ARM in svd, spcf (fw=0), spcf_fw (fw=0.01), spcf_ext, spcf_ext_fw, varfit, dmrg (Pauli local controls only)

Cells: tfim_ising, tfim_crit, heis, lr{0.5,1.5,3}_{nat,rand}, ppp_{site,mo}_{nat,fied,rand}.  N = 16 qubits throughout.
The compression step is one left-to-right sweep over the exact right-canonical MPS with no gates (the 'reverse schedule' step):
at bond b the two-site tensor is cut to chi by ``mpsenh.svd_cut`` or by ``SPCFast`` (a=2, taus=(), static targets only).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import json
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import _paths  # noqa: E402,F401
os.environ.setdefault('SPCF_CACHE', str(HERE / '_cache'))
import mpsenh as M  # noqa: E402
import models as Mo  # noqa: E402
import mps_tools as MT  # noqa: E402
from spcfast_ext import SPCFastExt  # noqa: E402

N_QUBITS = 16
SPANS = (2, 3, 4, 6, 8)
# transverse field of the long-range Ising cells: h = 1, the nearest-neighbour coupling |i-j|^-alpha at |i-j| = 1 (fixed before any
# T0 number was seen; an entropy-maximising choice would pick the near-degenerate cat-state regime h -> 0 and is not used)
H_LR = {0.5: 1.0, 1.5: 1.0, 3.0: 1.0}
LR_SEED = 7
PPP_SEED = 11


# ---------------------------------------------------------------------------------------------- cells
class Cell:
    def __init__(self, name, model, info):
        self.name, self.model, self.info = name, model, info
        self.N = model.N
        self.elem = model.elem
        t = time.time()
        self.v = model.ground_state()
        self.R_ex = self.elem.values(self.v)
        self.E_ex = float(np.vdot(self.v, model.matvec(self.v)).real)
        self.t_build = time.time() - t
        self._rho = {}

    def ext(self, lo, hi):
        if (lo, hi) not in self._rho:
            self._rho[lo, hi] = Mo.rdm_sites(self.v, self.N, list(range(lo, hi)))
        return self._rho[lo, hi]

    @property
    def is_fermion(self):
        return hasattr(self.model, 'number_stats')


def make_cell(name, N=N_QUBITS):
    if name == 'tfim_ising':
        return Cell(name, Mo.PauliModel(N, Mo.tfim_terms(N, 0.9045, 0.809), name), dict(kind='local control'))
    if name == 'tfim_crit':
        return Cell(name, Mo.PauliModel(N, Mo.tfim_terms(N, 1.0, 0.0), name), dict(kind='local control'))
    if name == 'heis':
        return Cell(name, Mo.PauliModel(N, Mo.heis_terms(N), name), dict(kind='local control'))
    if name.startswith('tfimh'):                       # extra local controls (not in the plan): transverse-field Ising, hz = 0
        return Cell(name, Mo.PauliModel(N, Mo.tfim_terms(N, float(name[5:]), 0.0), name), dict(kind='local control (extra)'))
    if name.startswith('xxz'):                         # extra local control: XXZ chain, Delta = float(name[3:])
        D = float(name[3:])
        t = [(1.0, ((i, a), (i + 1, a))) for i in range(N - 1) for a in 'XY'] + [(D, ((i, 'Z'), (i + 1, 'Z'))) for i in range(N - 1)]
        return Cell(name, Mo.PauliModel(N, t, name), dict(kind='local control (extra)'))
    if name.startswith('lr'):
        a, order = name[2:].split('_')
        alpha = float(a)
        perm = None if order == 'nat' else np.random.default_rng(LR_SEED).permutation(N)
        return Cell(name, Mo.PauliModel(N, Mo.lr_ising_terms(N, alpha, H_LR[alpha], perm), name),
                    dict(kind='non-local proxy', alpha=alpha, h=H_LR[alpha], order=order, perm=None if perm is None else perm.tolist()))
    if name.startswith('ppp'):
        _, basis, order = name.split('_')
        L = N // 2
        perm = None
        if order != 'nat':
            if order == 'rand':
                perm = np.random.default_rng(PPP_SEED).permutation(L)
            elif order == 'fied':
                m0 = Mo.PPPModel(L, basis)
                perm = Mo.fiedler_order(Mo.orbital_mutual_information(m0.ground_state(), L))
            else:
                raise ValueError(order)
        return Cell(name, Mo.PPPModel(L, basis, perm), dict(kind='non-local proxy', basis=basis, order=order,
                                                            perm=None if perm is None else [int(x) for x in perm]))
    raise ValueError(name)


# ---------------------------------------------------------------------------------------------- metrics
def metrics(cell, v):
    v = v / np.linalg.norm(v)
    e = cell.elem
    R = e.values(v)
    d = R - cell.R_ex
    w = e.coef * d
    a2, w2, wr = np.abs(d) ** 2, np.abs(w) ** 2, w.real
    o = {}
    o['infid'] = float(1.0 - abs(np.vdot(cell.v, v)) ** 2)
    o['dE'] = float(np.vdot(v, cell.model.matvec(v)).real - cell.E_ex)
    o['dE_elem'] = float(wr.sum())
    for s in SPANS:
        m = e.span <= s
        o[f'loc{s}'] = float(a2[m].sum() / a2.sum())
        o[f'locE{s}'] = float(w2[m].sum() / w2.sum())
        o[f'locEs{s}'] = float(wr[m].sum() / wr.sum()) if wr.sum() != 0 else float('nan')
        o[f'locEa{s}'] = float(np.abs(wr)[m].sum() / np.abs(wr).sum())
    near = e.span <= 4
    o['rms_loc4'] = float(np.sqrt(a2[near].sum()))
    o['rms_far4'] = float(np.sqrt(a2[~near].sum()))
    o['rms_all'] = float(np.sqrt(a2.sum()))
    o['hw_all'] = float(np.sqrt(w2.sum()))
    o['hw2'] = float(np.sqrt(w2[e.kind == 2].sum()))
    o['hw2_far4'] = float(np.sqrt(w2[(e.kind == 2) & ~near].sum()))
    o['hw2_loc4'] = float(np.sqrt(w2[(e.kind == 2) & near].sum()))
    if cell.is_fermion:
        nm, nv, sm, sv = cell.model.number_stats(v)
        o['N_mean'], o['N_var'], o['Sz_mean'], o['Sz_var'] = nm, nv, sm, sv
        o['leak'] = float(max(nv, sv))
    if hasattr(e, 'idx'):                      # Pauli models: the rank-9 observables
        Nq = cell.N
        for key, ids in (('obsX', [e.idx[((i, 'X'),)] for i in range(Nq)]),
                         ('obsZZ1', [e.idx[((i, 'Z'), (i + 1, 'Z'))] for i in range(Nq - 1)]),
                         ('obsZZ3', [e.idx[((i, 'Z'), (i + 3, 'Z'))] for i in range(Nq - 3)])):
            o[key] = float(np.sqrt(np.mean(np.abs(d[ids]) ** 2)))
    return o


# ---------------------------------------------------------------------------------------------- arms
def run_arm(cell, arm, chi, seed=0):
    N = cell.N
    T = MT.dense_to_right_mps(cell.v, N)
    info = {}
    t0 = time.perf_counter()
    if arm == 'svd':
        MT.sweep_compress(T, chi, M.svd_cut)
    elif arm.startswith('spcf'):
        fw = 0.01 if arm.endswith('_fw') else 0.0
        sp = SPCFastExt('ising', N, a=2, taus=(), fw=fw)
        if arm.startswith('spcf_ext'):
            sp.ext = cell.ext
        MT.sweep_compress(T, chi, sp)
        info.update(fired=sp.fired, calls=sp.calls, skipped=sp.skipped, rejected=int(sum(getattr(sp, 'rej', {}).values())))
    elif arm == 'varfit':
        MT.sweep_compress(T, chi, M.svd_cut)
        T, hist = MT.var_fit(cell.v, T, 3)
        info['fid_hist'] = [1 - h for h in hist]
    elif arm == 'dmrg':
        W = MT.nn_mpo(N, cell.model.terms)
        T, hist = MT.dmrg2(W, N, chi, nsweeps=20, seed=seed)
        info['sweeps'] = len(hist)
        info['E_dmrg'] = hist[-1]
    else:
        raise ValueError(arm)
    info['wall'] = time.perf_counter() - t0
    info['ranks'] = MT.ranks(T)
    return MT.to_dense(T), info


def main_selftest():
    for name in ('tfim_crit', 'ppp_mo_nat'):
        c = make_cell(name)
        print(name, 'E_ex', c.E_ex, 'build', round(c.t_build, 1), 's  E from elements',
              float((c.elem.coef * c.R_ex).sum().real + c.elem.const))
        for arm in ('svd', 'spcf', 'varfit'):
            v, info = run_arm(c, arm, 8)
            m = metrics(c, v)
            print(' ', arm, {k: (round(x, 8) if isinstance(x, float) else x) for k, x in m.items() if k in ('infid', 'dE', 'dE_elem', 'loc4', 'locE4', 'leak')}, info.get('fired'))


LADDER = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def main_t0(out, names):
    res = {}
    if Path(out).exists():
        res = json.load(open(out))
    for name in names:
        t = time.time()
        c = make_cell(name)
        rec = dict(info=c.info, E_ex=c.E_ex, gap=getattr(c.model, 'gap', None), resid=getattr(c.model, 'resid', None), rows=[])
        for chi in LADDER:
            v, info = run_arm(c, 'svd', chi)
            m = metrics(c, v)
            m.update(chi=chi, ranks=info['ranks'])
            rec['rows'].append(m)
            if m['infid'] < 1e-9:
                break
        res[name] = rec
        json.dump(res, open(out, 'w'), indent=1)
        print(f'{name}: done in {time.time() - t:.0f}s', flush=True)
        for m in rec['rows']:
            print(f"   chi={m['chi']:<3} infid={m['infid']:.2e} dE={m['dE']:.2e}  f_loc(4)={m['loc4']:.3f}  f_loc^E(4)={m['locE4']:.3f}  "
                  f"f_loc^E(3)={m['locE3']:.3f}  signed={m['locEs4']:.2f} abs={m['locEa4']:.3f}", flush=True)


def main_t1(out, name, chis, arms):
    res = {}
    if Path(out).exists():
        res = json.load(open(out))
    c = make_cell(name)
    res.setdefault(name, dict(info=c.info, E_ex=c.E_ex, runs=[]))
    for chi in chis:
        for arm in arms:
            t = time.time()
            v, info = run_arm(c, arm, chi)
            m = metrics(c, v)
            m.update(chi=chi, arm=arm, **{k: x for k, x in info.items()})
            res[name]['runs'].append(m)
            json.dump(res, open(out, 'w'), indent=1)
            print(f"{name} chi={chi} {arm:<12} infid={m['infid']:.3e} dE={m['dE']:.3e} hw2={m['hw2']:.3e} far4={m['rms_far4']:.3e} "
                  f"loc4={m['rms_loc4']:.3e} leak={m.get('leak', float('nan')):.1e} wall={info['wall']:.1f}s ({time.time() - t:.0f}s)", flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'selftest':
        main_selftest()
    elif cmd == 't0':
        main_t0(sys.argv[2], sys.argv[3:])
    elif cmd == 't1':
        main_t1(sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(',')], sys.argv[5].split(','))
