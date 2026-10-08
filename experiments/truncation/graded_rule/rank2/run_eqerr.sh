#!/bin/bash
# eqerr_time.py for the 6 cells, one invocation per cell (each < 20 min, resumable), 3 single-threaded workers
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
for cell in stag_mu0.5 stag_mu1 stag_mu0.25 dw_mu0.5 dw_mu1 dw_mu0.25; do
  extra=""; [ "$cell" = "stag_mu0.5" ] && extra="--noise"
  timeout 1100 python eqerr_time.py $cell --nproc 3 $extra > results/eqerr_run_$cell.log 2>&1
  echo "$cell exit $?" >> results/eqerr_run_status.log
done
