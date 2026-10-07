"""SVD-only scan: does a cell sit at useful truncation pressure?

A cell can show an effect of the rule only if SVD's infidelity front spans several decades inside the
bond-dimension ladder without reaching its floor. This runs plain SVD alone, which is cheap, over a
grid of (N, T) and prints the front.

    python pressure_scan.py heis 16,20 1.5,2.5,3.5 16,32,64,96 out.json 6
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import sys, json, time
import multiprocessing as mp
import _paths  # noqa: F401  (this repository's pytreenet and rule/ first)
import mpsenh as M

DT = 0.1


def one(args):
    model, N, T, chis = args
    nsteps = int(round(T / DT))
    Gs = M.make_gates(model, N, DT)
    t0 = time.time()
    ex = M.dense_reference(model, N, nsteps, DT, gates=Gs)
    ref_s = time.time() - t0
    rows = []
    for chi in chis:
        Tm, w = M.run_tebd(model, N, chi, nsteps, DT, M.svd_cut, gates=Gs)
        e = M.errors(ex, M.mps_to_dense(Tm), N, model)
        rows.append(dict(model=model, N=N, T=T, chi=chi, params=M.stored_params(Tm), wall=round(w, 2),
                         infid=e['infid'], nn_rms=e['nn_rms'], E_abs=e['E_abs']))
    return dict(model=model, N=N, T=T, ref_seconds=round(ref_s, 1), rows=rows)


if __name__ == '__main__':
    model = sys.argv[1]
    Ns = [int(x) for x in sys.argv[2].split(',')]
    Ts = [float(x) for x in sys.argv[3].split(',')]
    chis = [int(x) for x in sys.argv[4].split(',')]
    out, nproc = sys.argv[5], int(sys.argv[6])
    jobs = [(model, N, T, chis) for N in Ns for T in Ts]
    res = []
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(one, jobs):
            res.append(r)
            json.dump(res, open(out, 'w'))
            front = '  '.join(f"chi{x['chi']}:{x['infid']:.1e}" for x in r['rows'])
            print(f"N={r['N']:>2} T={r['T']:<4} (ref {r['ref_seconds']}s)  {front}", flush=True)
