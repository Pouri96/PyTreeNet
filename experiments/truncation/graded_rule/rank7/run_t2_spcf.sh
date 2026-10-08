#!/bin/bash
# T2: spcf arms.  usage: run_t2_spcf.sh TAG REL_SKIP EPS_MIN   (OPT = the pareto_spcf.py default shape of the plan, 2:0:4:1.0:1-2-3:0:1:EPS:all:REL)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=rank7/results
tag=$1; rs=$2; eps=$3
for chi in 16 24 32; do
  timeout 1700 python -u rank7/ladder_t.py ising 20 8.0 "spcf:2:0:4:1.0:1-2-3:0:1:$eps:all:$rs" $chi $R/t2_spcf_${tag}_$chi.npz > $R/t2_spcf_${tag}_$chi.txt 2>&1
done
echo done > $R/t2_spcf_${tag}_done.flag
