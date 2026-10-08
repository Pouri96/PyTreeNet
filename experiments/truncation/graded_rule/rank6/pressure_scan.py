"""Pressure scan: SVD-LPDO errors (no probes) over chi, kappa, gamma at N=8, to choose cells with nn rms in [1e-3, 1e-1]."""
import _p6  # noqa: F401
import sys, time, json
import multiprocessing as mp
import lpdo_bench as B

def one(a):
    model, N, T, dt, g, chi, kap = a
    t = time.time()
    r = B.cell(model, N, T, dt, g, chi, kap)
    r['cpu'] = time.time() - t
    return r

if __name__ == '__main__':
    model, N, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    gs = [float(x) for x in sys.argv[4].split(',')]
    chis = [int(x) for x in sys.argv[5].split(',')]
    kaps = [int(x) for x in sys.argv[6].split(',')]
    out = sys.argv[7]
    nproc = int(sys.argv[8]) if len(sys.argv) > 8 else 2
    for g in gs:
        B.reference(model, N, T, 0.1, g)
    jobs = [(model, N, T, 0.1, g, c, k) for g in gs for c in chis for k in (kaps if g > 0 else [kaps[0]])]
    rows = []
    with mp.Pool(nproc) as pool:
        for r in pool.imap(one, jobs):
            rows.append(r)
            print(f"g={r['gamma']:<6g} chi={r['chi']:<3d} kap={r['kappa']:<2d} nn={r['nn_rms']:.2e} nnn={r['nnn_rms']:.2e} sgl={r['single_rms']:.2e} E={r['E_abs']:.2e} tn={r['trace_norm']:.2e}  bdisc={r['bond_disc']:.1e} kdisc={r['kraus_disc']:.1e} maxK={r['maxK']} maxchi={r['maxchi']} ({r['cpu']:.0f}s)", flush=True)
    B.dump(rows, out)
