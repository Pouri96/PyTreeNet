"""End to end: MPS-native TEBD, plain SVD cut vs the enhanced cut, against the dense Trotter reference.

Arms are  'svd'  or  'enh:kappa:gamma:wE:eps_gate'.  Prints same-chi error ratios (params identical by
construction, asserted) and wall time ratios.  Python overhead dominates wall time at small chi, so the
per-cut cost law is measured separately in mps_cost.py.
"""
import sys, json, time
import numpy as np
import multiprocessing as mp
import _paths  # noqa: F401  (rule/ onto sys.path)
import mpsenh as M


def one(args):
    model, N, chi, T, dt, arm = args
    nsteps = int(round(T / dt))
    Gs = M.make_gates(model, N, dt)
    if arm == 'svd':
        cut = M.svd_cut
    elif arm.startswith('enhG') or arm.startswith('enhg'):
        # global energy through Hamiltonian environments: enhG = energy only, enhg = window + energy
        p = arm.split(':')
        cut = M.EnhCut(model, N, kappa=float(p[1]), gamma=4.0, wE=float(p[2]), energy='global',
                       energy_only=arm.startswith('enhG'), refine=4)
    elif arm.startswith('enhb') or arm.startswith('enhe'):
        # energy-only: enhb = bond term inside theta (no neighbour contractions), enhe = 4-site window
        p = arm.split(':')
        cut = M.EnhCut(model, N, kappa=float(p[1]), gamma=4.0, wE=float(p[2]),
                       window='bond' if arm.startswith('enhb') else 'r2', energy_only=True, ndir=1)
    else:
        p = arm.split(':')
        cut = M.EnhCut(model, N, kappa=float(p[1]), gamma=float(p[2]), wE=float(p[3]),
                       eps_gate=float(p[4]) if len(p) > 4 else 0.0,
                       ndir=int(p[5]) if len(p) > 5 else 1,
                       gate_rel=float(p[6]) if len(p) > 6 else 0.0)
    Tm, wall = M.run_tebd(model, N, chi, nsteps, dt, cut, gates=Gs)
    ex = M.dense_reference(model, N, nsteps, dt, gates=Gs)
    e = M.errors(ex, M.mps_to_dense(Tm), N, model)
    e.update(chi=chi, arm=arm, params=M.stored_params(Tm), maxrank=max(M.ranks(Tm)), wall=round(wall, 3))
    if arm != 'svd':
        e.update(fired=cut.fired, truncating=cut.truncating)
    return e


if __name__ == '__main__':
    model, N, T, dt = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    arms = sys.argv[5].split(',')
    chis = [int(c) for c in sys.argv[6].split(',')]
    out = sys.argv[7]
    nproc = int(sys.argv[8]) if len(sys.argv) > 8 else 3
    jobs = [(model, N, c, T, dt, a) for c in chis for a in arms]
    res = []
    t0 = time.time()
    with mp.Pool(nproc) as pool:
        for r in pool.imap_unordered(one, jobs):
            res.append(r)
            json.dump(res, open(out, 'w'))
            print(f"done chi={r['chi']} {r['arm']} ({len(res)}/{len(jobs)}, {time.time() - t0:.0f}s)", flush=True)
