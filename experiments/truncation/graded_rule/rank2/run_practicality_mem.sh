#!/bin/bash
# Final-state Pareto runs, MEMORY modes (fresh process per run): rss = growth of the resident-set high-water mark over the whole run with the F tables
# resident but not counted (their size is recorded, headline = rss + tables); rssA = as built (table construction inside the run); tmem = tracemalloc.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
NP=${1:-3}
R=results/pareto_rss.json            # baselines (SVD plain / back, DMT)
RS=results/pareto_rss_spcf.json      # spcf variants
timeout 1100 python3 traj_jobs.py --cells all --arms spcfc-a2-b1-wf --chis 12,16,24,32,48 --mode rss --nproc $NP --out $RS
timeout 1100 python3 traj_jobs.py --cells all --arms spcf-a1,spcfc-a1-b8-wf --chis 12,16,24,32,48,64 --mode rss --nproc $NP --out $RS
timeout 1100 python3 traj_jobs.py --cells all --arms spcf-a2 --chis 12,16,24 --mode rss --nproc $NP --out $RS
# baselines (SVD plain/back chi 8..96, DMT chi 12..96, rss mode, output $R) were run separately before this script:
#   traj_jobs.py --cells all --arms svd_plain,svd_back --chis 8,12,16,20,24,32,40,48,64,80,96 --mode rss --nproc 1 --out $R
#   traj_jobs.py --cells all --arms dmt --chis 12,16,20,24,32,40,48,64,80,96 --mode rss --nproc 1 --out $R
# as built (table build transient included), one cell
RA=results/pareto_rssA.json
timeout 1100 python3 traj_jobs.py --cells stag_mu0.5 --arms spcf-a2,spcfc-a2-b1-wf --chis 12,16,24 --mode rssA --nproc $NP --out $RA
timeout 1100 python3 traj_jobs.py --cells stag_mu0.5 --arms spcf-a1,spcfc-a1-b8-wf --chis 12,24,48 --mode rssA --nproc $NP --out $RA
# tracemalloc decomposition, two cells
TM=results/pareto_tmem.json
timeout 1100 python3 traj_jobs.py --cells stag_mu0.5,dw_mu0.5 --arms spcf-a1,spcfc-a1-b8-wf,spcf-a2,spcfc-a2-b1-wf --chis 12,24 --mode tmem --nproc $NP --out $TM
timeout 1100 python3 traj_jobs.py --cells stag_mu0.5,dw_mu0.5 --arms svd_plain,dmt --chis 16,48 --mode tmem --nproc $NP --out $TM
