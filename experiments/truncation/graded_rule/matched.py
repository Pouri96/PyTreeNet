"""Matched-WALL-TIME comparison: the rule at chi, against SVD at every chi, same model/T/dt, serial."""
import os
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import sys, json
import numpy as np
import _paths  # noqa: F401  (rule/ onto sys.path)
import mpsenh as M
model,N,T,dt=sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4])
nsteps=int(round(T/dt))
Gs=M.make_gates(model,N,dt)
ex=M.dense_reference(model,N,nsteps,dt,gates=Gs)
out=[]
def run(chi,arm):
    if arm=='svd':
        cut=M.svd_cut
    else:
        cut=M.EnhCut(model,N,kappa=0.1,gamma=4.0,wE=1e4,energy='global',energy_only=False,refine=4)
    Tm,w=M.run_tebd(model,N,chi,nsteps,dt,cut,gates=Gs)
    e=M.errors(ex,M.mps_to_dense(Tm),N,model)
    e.update(chi=chi,arm=arm,params=M.stored_params(Tm),wall=w)
    out.append(e); json.dump(out,open(sys.argv[5],'w'))
    print(f"{arm:>5} chi={chi:>3} params={e['params']:>6} wall={w:7.2f}s infid={e['infid']:.3e} nn={e['nn_rms']:.3e} E={e['E_abs']:.3e}",flush=True)
for chi in [int(c) for c in sys.argv[6].split(',')]:
    run(chi,'svd')
for chi in [int(c) for c in sys.argv[7].split(',')]:
    run(chi,'rule')
