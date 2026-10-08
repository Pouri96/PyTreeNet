"""Validation of rank2/spcfpur_chunk.py against rank2/spcfpur.py (the unchunked cut).

    python test_chunk.py [OUT.json]

A. Random cells: random purification MPS (d = 4) on N = 8 sites, random two-site tensor with a decaying spectrum, windows a = 1 and 2, bonds at
   the chain edges and in the bulk, truncation to chi = 3 .. 6, directions R and L, block sizes 1, 2, 3, 5 and 'all' (= unchunked inside the chunk code).
   Compared, chunked vs unchunked: region RDM (W W^dag), initial residual r0, jvp(x), vjp(g) on random vectors (first-pass chart), one full cut (returned
   tensors, the log entry, the accepted line-search index).  Adjoint identity <g, J x> = <J^T g, x> for both classes.
B. Cells recorded along a TEBD run (N = 8, chi = 4, T = 2.5, mu = 1 staggered, plain gauge, 350 cuts): the same comparisons on every fired cut,
   plus the accumulated state of two whole TEBD runs (chunked vs unchunked): max difference of the final MPS tensors and of the final rdm2.
Both for precision f64 (tolerance 1e-10 / 1e-9) and f32 (1e-4 / 1e-3), the pre-registered numbers.
"""
import sys
import json
import _p2  # noqa: F401
import numpy as np
import purlib as P
from spcfpur import SPCFPur
from spcfpur_chunk import SPCFPurChunk
from cost_scaling import rand_mps, theta_with_spectrum

OPT = dict(fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), f_min=0.0, every=1, eps_min=0.0, pattern='all', rel_skip=0.0)
rng = np.random.default_rng(7)


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / (np.linalg.norm(b) + 1e-300))


def compare_cell(T, theta, chi, dirn, b, N, a, prec, blk, rng):
    ref = SPCFPur('ising', N, a=a, precision=prec, **OPT)
    chk = SPCFPurChunk('ising', N, a=a, precision=prec, blk=blk, **OPT)
    ref.debug, chk.debug = [], []
    ref.start(T)
    chk.start(T)
    o1 = ref(theta.copy(), chi, dirn, None, None, b)
    o2 = chk(theta.copy(), chi, dirn, None, None, b)
    res = dict(a=a, prec=prec, blk=blk, b=b, dirn=dirn, chi=chi, fired=(ref.fired, chk.fired))
    if not ref.debug:
        res['no_debug'] = True
        return res
    d1, d2 = ref.debug[0], chk.debug[0]
    th = d1['theta']
    W = d1['Wof'](th)
    rho_ref = W @ W.conj().T
    res['rho'] = rel(d2['rho_of'](th), rho_ref)
    Wc = d2['Wof'](th)
    res['W_assembled'] = rel(Wc, W)
    res['r0'] = rel(d2['r0'], d1['r0'])
    x = rng.normal(size=d1['nx'])
    g = rng.normal(size=d1['nres'])
    j1, j2 = d1['jvp'](x), d2['jvp'](x)
    v1, v2 = d1['vjp'](g), d2['vjp'](g)
    res['jvp'] = rel(j2, j1)
    res['vjp'] = rel(v2, v1)
    res['adjoint_chunk'] = float(abs(g @ j2 - v2 @ x) / (abs(g @ j2) + 1e-300))
    res['adjoint_ref'] = float(abs(g @ j1 - v1 @ x) / (abs(g @ j1) + 1e-300))
    e = rng.normal(size=d1['nx'])
    e *= 1e-3 / np.linalg.norm(e)
    res['exact_resid'] = rel(d2['exact'](e)[2], d1['exact'](e)[2])
    res['cut'] = max(rel(o2[0], o1[0]), rel(o2[1], o1[1]))
    res['cut_flag'] = (o1[3], o2[3])
    res['alpha'] = (dict(ref.achosen), dict(chk.achosen))
    if ref.log and chk.log:
        res['log'] = max(abs(u - v) / (abs(u) + 1e-300) for u, v in zip(ref.log[0], chk.log[0]))
    res['nblocks'] = d2['nblocks']
    return res


def part_A():
    rows = []
    N = 8
    for a in (1, 2):
        for prec in ('f64', 'f32'):
            for chi in (4, 6):
                rr = np.random.default_rng(100 * chi + a)
                T, c = rand_mps(N, 4, chi, rr)
                for b in (0, 2, 3, 4, N - 2):
                    l, r = T[b].shape[0], T[b + 1].shape[2]
                    if min(l, r) < 2:
                        continue
                    th = theta_with_spectrum(l, 4, r, rr, decay=0.35)
                    for dirn in ('R', 'L'):
                        for blk in (1, 2, 3, 5, 10 ** 6):
                            rows.append(compare_cell(T, th, min(chi, 2 * min(l, r)) if b else chi, dirn, b, N, a, prec, blk, rr))
    return rows


def part_B():
    rows = []
    N, chi = 8, 4
    for a in (1, 2):
        for prec in ('f64', 'f32'):
            cut = SPCFPur('ising', N, a=a, taus=[1.0, 2.0], fw=0.0, precision=prec)
            cut.debug = []
            T0 = P.initial_pur_mps(N, 'stag', 1.0)
            Gs = P.pur_gates('ising', N, 0.1, 'plain')
            P.run_tebd_d(T0, Gs, chi, 25, cut)
            rr = np.random.default_rng(3)
            for dd in cut.debug[::2]:
                for blk in (1, 7):
                    kw = dict(OPT)
                    ref = SPCFPur('ising', N, a=a, precision=prec, taus=[1.0, 2.0], fw=0.0, f_min=0.0, every=1, eps_min=0.0, pattern='all', rel_skip=0.0)
                    chk = SPCFPurChunk('ising', N, a=a, precision=prec, blk=blk, taus=[1.0, 2.0], fw=0.0, f_min=0.0, every=1, eps_min=0.0, pattern='all',
                                       rel_skip=0.0)
                    ref.debug, chk.debug = [], []
                    ref.start(dd['Tsnap'])
                    chk.start(dd['Tsnap'])
                    o1 = ref(dd['theta'].copy(), chi, dd['dirn'], None, None, dd['b'])
                    o2 = chk(dd['theta'].copy(), chi, dd['dirn'], None, None, dd['b'])
                    d1, d2 = ref.debug[0], chk.debug[0]
                    x = rr.normal(size=d1['nx'])
                    g = rr.normal(size=d1['nres'])
                    rows.append(dict(a=a, prec=prec, blk=blk, b=dd['b'], jvp=rel(d2['jvp'](x), d1['jvp'](x)), vjp=rel(d2['vjp'](g), d1['vjp'](g)),
                                     r0=rel(d2['r0'], d1['r0']), cut=max(rel(o2[0], o1[0]), rel(o2[1], o1[1])), fired=(ref.fired, chk.fired),
                                     adjoint_chunk=float(abs(g @ d2['jvp'](x) - d2['vjp'](g) @ x) / abs(g @ d2['jvp'](x)))))
    return rows


def part_C():
    """two whole TEBD runs, chunked against unchunked, final state"""
    out = []
    N, chi = 8, 6
    for a in (1, 2):
        for prec in ('f64', 'f32'):
            res = []
            for cls, kw in ((SPCFPur, {}), (SPCFPurChunk, dict(blk=5))):
                cut = cls('ising', N, a=a, precision=prec, **kw)
                T0 = P.initial_pur_mps(N, 'stag', 0.5)
                Gs = P.pur_gates('ising', N, 0.1, 'plain')
                Tm, _ = P.run_tebd_d(T0, Gs, chi, 30, cut)
                res.append((Tm, cut))
            (T1, c1), (T2, c2) = res
            out.append(dict(a=a, prec=prec, max_tensor_diff=max(float(np.abs(x - y).max()) for x, y in zip(T1, T2)), fired=(c1.fired, c2.fired),
                            calls=(c1.calls, c2.calls), achosen=(c1.achosen, c2.achosen),
                            psi_overlap_defect=float(1 - abs(np.vdot(P.mps_to_dense_d(T1) / np.linalg.norm(P.mps_to_dense_d(T1)),
                                                                     P.mps_to_dense_d(T2) / np.linalg.norm(P.mps_to_dense_d(T2)))))))
    return out


def summarise(rows, keys):
    out = {}
    for prec in ('f64', 'f32'):
        sel = [r for r in rows if r['prec'] == prec and 'jvp' in r]
        out[prec] = {k: max(r[k] for r in sel if k in r) for k in keys}
        out[prec]['n'] = len(sel)
        out[prec]['fired_all_equal'] = all(r['fired'][0] == r['fired'][1] for r in sel)
        out[prec]['n_fired'] = int(sum(r['fired'][0] for r in sel))
    return out


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'results/test_chunk.json'
    A = part_A()
    keys = ['rho', 'W_assembled', 'r0', 'jvp', 'vjp', 'adjoint_chunk', 'adjoint_ref', 'exact_resid', 'cut', 'log']
    sA = summarise(A, keys)
    print('A. random cells (max over cells and block sizes):')
    for prec in sA:
        print(' ', prec, {k: (f'{v:.2e}' if isinstance(v, float) else v) for k, v in sA[prec].items()})
    alpha_same = all(r['alpha'][0] == r['alpha'][1] for r in A if 'alpha' in r)
    print('   same accepted line-search steps in all cells:', alpha_same, ', cells with a fired cut:', sum(r['fired'][0] for r in A if 'jvp' in r), 'of',
          len([r for r in A if 'jvp' in r]))
    B = part_B()
    sB = summarise(B, ['r0', 'jvp', 'vjp', 'cut', 'adjoint_chunk'])
    print('B. cells from TEBD run (fired cuts):')
    for prec in sB:
        print(' ', prec, {k: (f'{v:.2e}' if isinstance(v, float) else v) for k, v in sB[prec].items()})
    C = part_C()
    print('C. whole TEBD runs:')
    for c in C:
        print('  ', c)
    json.dump(dict(A=A, B=B, C=C, sA=sA, sB=sB, alpha_same=alpha_same), open(out, 'w'), indent=1, default=str)
