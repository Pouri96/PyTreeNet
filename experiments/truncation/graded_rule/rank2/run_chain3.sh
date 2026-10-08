#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NPROC=3
cd "$(dirname "$0")"
for spec in "stag 0.25 4 back" "dw 0.5 4 plain" "dw 0.25 4 back" "stag 0.5 3 plain" "dw 1 4 plain"; do
  set -- $spec
  timeout 1700 python test2_dmt.py $1 $2 $3 12,16,24,32 $4 results/test2_$1_mu$2_T$3.json > results/test2_$1_mu$2_T$3.log 2>&1
done
