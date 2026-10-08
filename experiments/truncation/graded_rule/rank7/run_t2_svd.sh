#!/bin/bash
# T2: SVD ladder (16, 24, 32, 36, 48, 64, 96; 36 is added so that t*_SVD(1.5 * 24) is measured, not interpolated).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=rank7/results
for chi in 16 24 32 36 48 64 96; do
  timeout 1700 python -u rank7/ladder_t.py ising 20 8.0 svd $chi $R/t2_svd_$chi.npz > $R/t2_svd_$chi.txt 2>&1
done
echo done > $R/t2_svd_done.flag
