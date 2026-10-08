#!/bin/bash
# A0 + A1 grid: all models x both cuts, N=12, chi in {4,6,8}, kappa = 0.1.  Two lanes (4 cores are shared with other agents).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p rank5/results
lane() {
  for mb in "$@"; do
    m=${mb%:*}; b=${mb#*:}
    timeout 1700 python rank5/mt_cut.py a1 --model $m --N 12 --b $b > rank5/results/a1_${m}_N12_b${b}.txt 2>&1
  done
}
lane ising:5 tfim_crit:5 ising2:5 heis:5 &
lane ising:3 tfim_crit:3 ising2:3 heis:3 &
wait
echo done > rank5/results/a1_grid_done.flag
