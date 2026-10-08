#!/bin/bash
# does more linearisation / a more conservative threshold close the gap to lcut on the cells where spc is weakest
PY="C:/Users/edpou/Desktop/BUG/.venv/Scripts/python.exe"
T="0.5-1.0-1.5-2.0:1-2-3"
A="spc:2:hard:0.1:1:$T,spc:2:hard:0.03:1:$T,spc:2:hard:0.01:4:$T,spc:2:soft:0.03:3:$T,spc:2:hard:0.003:4:$T"
timeout 1700 $PY -u mfc_bench.py isingdw 12 4.0 0.1 $A 8  results/spc_grid_dw12.json 5 > out_spc_grid_dw12.txt 2>&1
timeout 1700 $PY -u mfc_bench.py ising2  12 4.0 0.1 $A 12 results/spc_grid_ising2_12.json 5 > out_spc_grid_ising2_12.txt 2>&1
