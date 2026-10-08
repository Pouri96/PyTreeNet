#!/bin/bash
# Test 1: N=16, T=6, dt=0.1; three models.  Pre-registered chi set {12,16,24,32,48}; supplementary {20,28,40,64}.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
MPO=""
for chi in 12 16 24 32 48 20 28 40 64; do
  for arm in svd_raw svd_renorm dmt_I dmt_neel; do MPO="$MPO mpo:$arm:$chi"; done
done
MPS=""
for c in 6 8 10 12 14 17 20 23 28 34 40 45 56 68 90; do MPS="$MPS mps:$c"; done
PPL=""
for e in 1e-2 5e-3 3e-3 2e-3 1e-3 5e-4 3e-4 2e-4 1e-4 5e-5; do PPL="$PPL pp:$e"; done
for model in ising ising2 heis; do
  python bench.py $model 16 6.0 0.1 results/test1_${model}_N16_T6.json 2 $MPO $MPS $PPL > results/test1_${model}_N16_T6.log 2>&1
done
