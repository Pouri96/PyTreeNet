#!/bin/bash
# Final-state Pareto runs, TIME mode (error series + CPU), N = 10, 6 cells, new variants only (SVD and DMT time/error are reused from eqerr_raw_*.json,
# spcf a = 2 unchunked at chi 12/16/24 from eqerr_raw too; a = 1 at 12/16/24 is results/a1_traj.json)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
NP=${1:-3}
OUT=results/pareto_time.json
timeout 1100 python3 traj_jobs.py --cells all --arms spcf-a1 --chis 32,48,64 --mode time --nproc $NP --out $OUT
timeout 1100 python3 traj_jobs.py --cells all --arms spcfc-a1-b8-wf --chis 12,16,24,32,48,64 --mode time --nproc $NP --out $OUT
timeout 1100 python3 traj_jobs.py --cells all --arms spcfc-a2-b1-wf --chis 12,16,24,32,48 --mode time --nproc $NP --out $OUT
