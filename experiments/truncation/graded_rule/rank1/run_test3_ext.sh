#!/bin/bash
# extension of Test 3 beyond the registered single chi: Ising chi 4,16,32; Heisenberg (informational) chi 8,16,32; TFIM chi 8,16,32.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
python rank1/op_bench.py ising 10 6.0 0.1 svd,svdt,spcop,rw:1.6 4,16,32 rank1/results/test3_ising_ext.json 2 1 > rank1/results/test3_ising_ext.log 2>&1
python rank1/op_bench.py heis 10 6.0 0.1 svd,svdt,spcop,rw:1.6 8,16,32 rank1/results/test3_heis.json 2 1 > rank1/results/test3_heis.log 2>&1
python rank1/op_bench.py tfim 10 6.0 0.1 svd,spcop,rw:1.6 8,16,32 rank1/results/test3_tfim.json 2 1 > rank1/results/test3_tfim.log 2>&1
