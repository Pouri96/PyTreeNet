"""End-to-end TEBD check of the whitened-SVD cut against SVD and production spcf (same dense Trotter reference, same metrics as pareto_spcf.py).

    python whiten/spcw_bench.py model N T chi_list out.json [arms=svd,spcf,spcw:all:10000:0,spcw:static:100:0]
arms: svd | spcf (best configuration 2:0:4:1.0:1-2-3:0:1:1e-7:all:1e-2) | spcw:<mode all|static>:<mu>:<accept 0|1>
Reuses hp_bench.reference / mpsenh.errors / hp_bench.marginal_errors by import (mfc_bench.py and pareto_spcf.py are not edited).  The
spcw arms are the oracle-grade (slow) implementation, so wall times are reported only for completeness and carry no cost claim.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import _paths  # noqa: F401
import mpsenh as M
import hp_bench
import spcfast
import spcwhite

BEST = dict(a=2, fw=0.0, iters=4, taus=[1.0], ks=(1, 2, 3), eps_min=1e-7, rel_skip=1e-2)


def make_cut(arm, model, N):
    if arm == 'svd':
        return M.svd_cut
    if arm == 'spcf':
        return spcfast.SPCFast(model, N, **BEST)
    p = arm.split(':')
    assert p[0] == 'spcw'
    return spcwhite.SPCWhite(model, N, mode=p[1], mu=float(p[2]), accept=bool(int(p[3])), **BEST)


def main():
    model, N, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    chis = [int(c) for c in sys.argv[4].split(',')]
    out = sys.argv[5]
    arms = sys.argv[6].split(',') if len(sys.argv) > 6 else ['svd', 'spcf', 'spcw:all:10000:0', 'spcw:static:100:0']
    dt = 0.1
    n = int(round(T / dt))
    G = M.make_gates(model, N, dt)
    ex = hp_bench.reference(model, N, T, dt)
    rows = []
    for chi in chis:
        for arm in arms:
            cut = make_cut(arm, model, N)
            w0, c0 = time.time(), time.process_time()
            Tm, _ = M.run_tebd(model, N, chi, n, dt, cut, gates=G)
            wall, cpu = time.time() - w0, time.process_time() - c0
            ap = M.mps_to_dense(Tm)
            e = M.errors(ex, ap, N, model)
            e.update(hp_bench.marginal_errors(ex, ap, N))
            row = dict(arm=arm, chi=chi, params=M.stored_params(Tm), wall=round(wall, 2), cpu=round(cpu, 2), **e)
            if arm != 'svd':
                row.update(fired=cut.fired, calls=cut.calls)
            rows.append(row)
            json.dump(rows, open(out, 'w'))
            print(f"{arm:20s} chi={chi:3d} infid={e['infid']:.3e} rdm2={e['rdm2']:.2e} 1site={e['single_rms']:.2e} nn={e['nn_rms']:.2e} nnn={e['nnn_rms']:.2e} "
                  f"E={e['E_abs']:.2e} | wall {wall:.1f}s" + (f" fired {cut.fired}/{cut.calls}" if arm != 'svd' else ''), flush=True)
    # ratios to SVD and retained log-gain relative to spcf
    print('\nratios to SVD (and retained log-gain g = log(ratio_arm)/log(ratio_spcf)):')
    for chi in chis:
        base = next(r for r in rows if r['chi'] == chi and r['arm'] == 'svd')
        sp = next((r for r in rows if r['chi'] == chi and r['arm'] == 'spcf'), None)
        for r in rows:
            if r['chi'] != chi or r['arm'] in ('svd',):
                continue
            rat = {k: r[k] / base[k] for k in ('single_rms', 'nn_rms', 'rdm2', 'infid')}
            g = ''
            if sp is not None and r['arm'] != 'spcf':
                g = ' g(rdm2) ' + f"{np.log(rat['rdm2']) / np.log(sp['rdm2'] / base['rdm2']):.2f}" + ' g(nn) ' + f"{np.log(rat['nn_rms']) / np.log(sp['nn_rms'] / base['nn_rms']):.2f}"
            print(f"  chi={chi:3d} {r['arm']:20s} " + ' '.join(f'{k}/svd={v:.3f}' for k, v in rat.items()) + g)


if __name__ == '__main__':
    main()
