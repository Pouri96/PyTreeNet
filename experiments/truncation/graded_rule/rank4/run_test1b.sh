#!/bin/bash
# Test 1, second launch (the first launch was stopped because the low-eps Pauli-propagation and chi_s=90 cells were too slow
# on a shared machine).  Ising: the missing cells only; ising2 and heis: full set with the trimmed ladders.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
MPO=""
for chi in 12 16 24 32 48 20 28 40 64; do
  for arm in svd_raw svd_renorm dmt_I dmt_neel; do MPO="$MPO mpo:$arm:$chi"; done
done
MPS=""
for c in 6 8 10 12 14 17 20 23 28 34 40 45 56 68; do MPS="$MPS mps:$c"; done
PPL=""
for e in 1e-2 5e-3 3e-3 2e-3 1e-3 5e-4 3e-4; do PPL="$PPL pp:$e"; done
python bench.py ising 16 6.0 0.1 results/test1_ising_N16_T6_b.json 2 pp:3e-4 > results/test1_ising_N16_T6_b.log 2>&1
for model in ising2 heis; do
  python bench.py $model 16 6.0 0.1 results/test1_${model}_N16_T6.json 2 $MPO $MPS $PPL > results/test1_${model}_N16_T6.log 2>&1
done
