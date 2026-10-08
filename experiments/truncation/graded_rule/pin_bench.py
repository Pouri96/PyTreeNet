"""Bench for the observable pin (rule/obspin.py) at matched final bond dimension.

    python pin_bench.py MODEL N T dt "arm arm ..." chi,chi out.json NPROC
Arms (space separated)
  svd                          classic per-gate SVD
  pinE:cut | pinE:step         existing whole-chain energy pin (mpsenh.PinCut, the spc_mps scalar correction)
  pin:key=val;key=val          rule/obspin.ObsPin, keys a k taus mech passes rcond wE wl damp eps static touch gam cover
                               taus like 0.5-1.0 (empty = no lookahead), static=0 drops the static rows
Each (arm, chi) is a separate process job. Reports errors against the exact solution and the pin diagnostics.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, json, time
import numpy as np
import multiprocessing as mp
import _paths  # noqa: F401
import mpsenh as M
import hp_bench
import obspin


def parse(spec):
    d = {}
    for kv in spec.split(';'):
        if kv:
            k, v = kv.split('=')
            d[k] = v
    return d


def make_cut(arm, model, N):
    if arm == 'svd':
        return None
    if arm.startswith('pinE'):
        return M.PinCut(model, N, mode=arm.split(':')[1])
    d = parse(arm.split(':', 1)[1])
    taus = tuple(float(x) for x in d.get('taus', '').split('-') if x)
    return obspin.ObsPin(model, N, a=int(d.get('a', 1)), k=int(d.get('k', 2)), taus=taus, mech=d.get('mech', 'tan'),
                         passes=int(d.get('passes', 3)), rcond=float(d.get('rcond', 1e-3)), w_energy=float(d.get('wE', 0)),
                         w_ahead=float(d.get('wl', 1.0)), damp=float(d.get('damp', 1.0)), eps_min=float(d.get('eps', 1e-10)),
                         static=d.get('static', '1') == '1', touch=d.get('touch', '0') == '1', gam=float(d.get('gam', 1.0)), cover=float(d.get('cover', 1.0)))


def one(args):
    model, N, T, dt, arm, chi = args
    nsteps = int(round(T / dt))
    ex = hp_bench.reference(model, N, T, dt)
    cut = make_cut(arm, model, N)
    t0, c0 = time.time(), time.process_time()
    Tm, _ = M.run_tebd(model, N, chi, nsteps, dt, M.svd_cut if cut is None else cut, gates=M.make_gates(model, N, dt))
    wall, cpu = time.time() - t0, time.process_time() - c0
    ap = M.mps_to_dense(Tm)
    e = M.errors(ex, ap, N, model)
    e.update(hp_bench.marginal_errors(ex, ap, N))
    e.update(arm=arm, chi=chi, params=M.stored_params(Tm), wall=round(wall, 1), cpu=round(cpu, 1))
    lg = getattr(cut, 'log', None)
    if lg:
        lg = np.array(lg)
        e.update(fired=cut.fired, calls=cut.calls, res_ratio=float(np.median(lg[:, 1] / np.maximum(lg[:, 0], 1e-300))),
                 cutinf_ratio=float(np.median(lg[:, 3] / np.maximum(lg[:, 2], 1e-300))))
    elif cut is not None:
        e.update(fired=getattr(cut, 'fired', 0))
    return e


if __name__ == '__main__':
    model, N, T, dt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    arms, chis = sys.argv[5].split(), [int(c) for c in sys.argv[6].split(',')]
    out, nproc = sys.argv[7], int(sys.argv[8]) if len(sys.argv) > 8 else 4
    hp_bench.reference(model, N, T, dt)
    jobs = [(model, N, T, dt, a, c) for c in chis for a in arms]
    res, t0 = [], time.time()
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(one, jobs):
            res.append(r)
            json.dump(res, open(out, 'w'))
            print(f"{len(res)}/{len(jobs)} chi={r['chi']:>2} {r['arm']:<46} infid={r['infid']:.3e} 1s={r['single_rms']:.2e} nn={r['nn_rms']:.2e} "
                  f"nnn={r['nnn_rms']:.2e} E={r['E_abs']:.2e} cpu={r['cpu']:.0f}s [{int(time.time() - t0)}s]", flush=True)
