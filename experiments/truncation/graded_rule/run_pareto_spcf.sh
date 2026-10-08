#!/bin/bash
# final Pareto data of the numpy spc cut against plain SVD (accuracy + steady-state cost), two cells, each well inside the 30 minute cap
PY="C:/Users/edpou/Desktop/BUG/.venv/Scripts/python.exe"
timeout 1700 $PY -u pareto_spcf.py ising 12 4.0 4,6,8,10,12,16,20,24,32,48,64 4,6,8,10,12,16 results/pareto_spcf_n12.json 5 > out_pareto_spcf_n12.txt 2>&1
timeout 1700 $PY -u pareto_spcf.py ising 16 5.0 6,8,12,16,24,32,48,64 8,12,16,24,32 results/pareto_spcf_n16_T5.json 3 > out_pareto_spcf_n16_T5.txt 2>&1
