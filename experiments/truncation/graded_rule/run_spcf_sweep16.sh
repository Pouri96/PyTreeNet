#!/bin/bash
# variants of the numpy spc cut on the four N=16 cells, each cell one invocation (well inside 30 minutes)
export PYLIBS="C:/Users/edpou/AppData/Local/Temp/claude/c--Users-edpou-Desktop-BUG/0e79467a-dca4-46a6-9ecf-ad2c2f175ea5/scratchpad/pylibs"
PY="C:/Users/edpou/Desktop/BUG/.venv/Scripts/python.exe"
R=":0:1:1e-10:all:1e-2"
A="spcf:2:0:4:1.0:1-2-3$R,spcf:2:0:3:1.0:1-2-3$R,spcf:2:0:6:1.0:1-2-3$R,spcf:2:0:4:1.0-2.0:1-2-3$R,spcf:2:0:4:0.5:1-2-3$R,spcf:2:0:4:1.5:1-2-3$R,spcf:2:0.03:4:1.0:1-2-3$R,spcf:2:0:4:1.0:1-2$R,spcf:3:0:4:1.0:1-2-3$R"
timeout 1700 $PY -u mfc_bench.py ising   16 5.0 0.1 svd,$A 16 results/spcf_sweep16_ising.json 3 > out_spcf_sweep16_ising.txt 2>&1
timeout 1700 $PY -u mfc_bench.py ising2  16 4.0 0.1 svd,$A 12 results/spcf_sweep16_ising2.json 3 > out_spcf_sweep16_ising2.txt 2>&1
timeout 1700 $PY -u mfc_bench.py isingdw 16 5.0 0.1 svd,$A 12 results/spcf_sweep16_dw.json 3 > out_spcf_sweep16_dw.txt 2>&1
timeout 1700 $PY -u mfc_bench.py heis    16 1.5 0.1 svd,$A 12 results/spcf_sweep16_heis.json 3 > out_spcf_sweep16_heis.txt 2>&1
