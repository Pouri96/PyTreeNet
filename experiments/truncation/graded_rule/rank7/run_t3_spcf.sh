#!/bin/bash
# T3: spcf arms (taus=none, static targets only).  usage: run_t3_spcf.sh TAG REL_SKIP EPS_MIN LABEL...
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=rank7/results
tag=$1; rs=$2; eps=$3; shift 3
for lab in "$@"; do
  for chi in 8 12 16 24; do
    [ -f $R/t3_${lab}_spcf_${tag}_$chi.npz ] && continue
    timeout 1700 python -u rank7/floquet.py $lab "spcf:2:0:4:none:1-2-3:0:1:$eps:all:$rs" $chi $R/t3_${lab}_spcf_${tag}_$chi.npz > $R/t3_${lab}_spcf_${tag}_$chi.txt 2>&1
  done
done
echo done > $R/t3_spcf_${tag}_$1_done.flag
