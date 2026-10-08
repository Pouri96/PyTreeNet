#!/bin/bash
# T3: SVD ladders on the kicked Ising chain, settings given as arguments (A B C D, see floquet.py)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=rank7/results
for lab in "$@"; do
  for chi in 6 8 12 16 24 32 48 64; do
    [ -f $R/t3_${lab}_svd_$chi.npz ] && continue
    timeout 1700 python -u rank7/floquet.py $lab svd $chi $R/t3_${lab}_svd_$chi.npz > $R/t3_${lab}_svd_$chi.txt 2>&1
  done
done
echo done > $R/t3_svd_$1_done.flag
