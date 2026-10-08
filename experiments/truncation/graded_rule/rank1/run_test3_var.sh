#!/bin/bash
# registered 'spcop gated' arm (gate constants refit on Ising, rank1/results/gate_fit_ising.json) plus supplementary variants / controls
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")/.."
C0=1.290e-06; EX=0.916
python rank1/op_bench.py ising 10 6.0 0.1 "spcop:gate=1;gate_c0=$C0;gate_exp=$EX,spcop:gate=3;gate_c0=$C0;gate_exp=$EX,spcop:gate=10;gate_c0=$C0;gate_exp=$EX" 8,16,32 rank1/results/test3_ising_gate.json 2 1 > rank1/results/test3_ising_gate.log 2>&1
python rank1/op_bench.py ising 10 6.0 0.1 "rw:1.2,rw:2.0,rw:3.0" 8,16,32 rank1/results/test3_ising_rw.json 2 0 > rank1/results/test3_ising_rw.log 2>&1
python rank1/op_bench.py ising 10 6.0 0.1 "spcop:fw=0.03,spcop:fw=0.1,spcop:ks=12,spcop:passes=3,spcop:rel_skip=0" 8,16 rank1/results/test3_ising_var.json 2 0 > rank1/results/test3_ising_var.log 2>&1
