"""Fixed-chi comparison against the CONTINUOUS-time exact state: SVD-TEBD, 1-site TDVP, and MFC.

    PYLIBS=<jax dir> python tdvp_bench.py ising 14 6.0 8,12,16 [arms]    arms: svd,tdvp,dsvd,mfc (default svd,tdvp)
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, time, json
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import exact_ref
import hp_bench
import tdvp1
import mfc

model, N, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
chis = [int(c) for c in sys.argv[4].split(',')]
arms = sys.argv[5].split(',') if len(sys.argv) > 5 else ['svd', 'tdvp']
fw = float(os.environ.get('FW', '0.03'))
KMAX = int(os.environ.get('KMAX', '3'))
dt = 0.1
ex = exact_ref.continuous(model, N, T)


def score(Tm):
    ap = M.mps_to_dense(Tm)
    e = M.errors(ex, ap, N, model)
    e.update(hp_bench.marginal_errors(ex, ap, N))
    return e


print(f"{model} N={N} T={T}: errors against the CONTINUOUS-time exact state (Trotter floor at dt=0.1: single 6e-4, nn 9e-4 for Ising N=14 T=6)")
print(f"{'chi':>4}{'params':>8} {'arm':>6}{'infid':>10}{'single':>10}{'nn':>10}{'nnn':>10}{'rdm2':>10}{'rdm3':>10}{'secs':>8}", flush=True)
rows = []
for chi in chis:
    for arm in arms:
        t0 = time.time()
        if arm == 'svd':
            Tm, _ = M.run_tebd(model, N, chi, int(round(T / dt)), dt, M.svd_cut, gates=M.make_gates(model, N, dt))
        elif arm == 'tdvp':
            Tm, _ = tdvp1.run_tdvp(model, N, chi, T, dt, 0.05)
        elif arm == 'dsvd':
            Tm, _ = mfc.run_mfc(model, N, chi, int(round(T / dt)), dt, 'dsvd', chi_w_factor=2)
        elif arm == 'mfc':
            Tm, _ = mfc.run_mfc(model, N, chi, int(round(T / dt)), dt, 'mfc', chi_w_factor=2, ks=tuple(range(1, KMAX + 1)), fw=fw, maxiter=200)
        e = score(Tm)
        e.update(arm=arm, chi=chi, params=M.stored_params(Tm), wall=round(time.time() - t0, 1))
        rows.append(e)
        print(f"{chi:>4}{e['params']:>8} {arm:>6}{e['infid']:>10.2e}{e['single_rms']:>10.2e}{e['nn_rms']:>10.2e}{e['nnn_rms']:>10.2e}{e['rdm2']:>10.2e}{e['rdm3']:>10.2e}{e['wall']:>8.0f}", flush=True)
        json.dump(rows, open(f"tdvp_bench_{model}_N{N}_T{T}_chi{'-'.join(map(str, chis))}.json", 'w'))
