#!/bin/bash
# T2 control (extra): spcf with taus=none (static window targets only, as T3 must use) on the same continuous-time cell, ungated.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=rank7/results
for chi in 16 24 32; do
  timeout 1700 python -u rank7/ladder_t.py ising 20 8.0 "spcf:2:0:4:none:1-2-3:0:1:1e-7:all:1e-2" $chi $R/t2_spcf_s001_$chi.npz > $R/t2_spcf_s001_$chi.txt 2>&1
done
echo done > $R/t2_spcf_s001_done.flag
