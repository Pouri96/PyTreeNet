#!/bin/bash
# full ladders to the SVD floor with the tail floor 1e-7 (default): spc and SVD at every chi, interleaved timing
PY="C:/Users/edpou/Desktop/BUG/.venv/Scripts/python.exe"
timeout 1700 $PY -u pareto_spcf.py ising 12 4.0 4,6,8,10,12,16,20,24,28,32,36,40,48,64 4,6,8,10,12,16,20,24,28,32,36,40,48,64 results/pareto_spcf_n12_floor.json 3 > out_pareto_spcf_n12_floor.txt 2>&1
timeout 1700 $PY -u pareto_spcf.py ising 16 5.0 8,12,16,24,32,48,56,64,72,80,96 8,12,16,24,32,48,56,64,72,80,96 results/pareto_spcf_n16_T5_floor.json 2 > out_pareto_spcf_n16_T5_floor.txt 2>&1
