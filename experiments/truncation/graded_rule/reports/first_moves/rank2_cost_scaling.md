# Rank 2: how the cost of the purification spcf cut scales with χ

Script: `rank2/cost_scaling.py`. Raw results: `rank2/results/cost_scaling_*.json`. Figure: `rank2/figures/cost_scaling.png`.

## Setup

- One fired cut at a bulk bond of a random purification MPS (N = 16, d = 4).
- Every bond in the spcf window is at χ. An N = 10 chain caps the window's outer bonds at 16, which hides the real growth, so a longer chain is needed.
- The two-site tensor has a TEBD-like decaying spectrum, truncated 4χ → χ.
- spcf uses the options from Tests 1 and 2 (iters = 4, τ = 1.0, ks = 1-2-3). Every skip rule is off, so each cut fires.
- One BLAS thread. The table reports the median over 2-3 repeats.
- Peak memory is measured with tracemalloc.
- Caveat: one other job was running on the machine at the same time.

## Results

| χ | SVD cut | spcf a = 2, time | ratio | memory | spcf a = 1, time | ratio | memory |
|---|---|---|---|---|---|---|---|
| 8 | 0.21 ms | 32 ms | 154x | 16 MB | 2.9 ms | 13x | 1.2 MB |
| 16 | 0.9 ms | 138 ms | 151x | 65 MB | 9.2 ms | 11x | 4.6 MB |
| 32 | 3.7 ms | 0.82 s | 220x | 259 MB | 44 ms | 11x | 18 MB |
| 48 | 9.9 ms | 2.2 s | 224x | 583 MB | 120 ms | 13x | 41 MB |
| 64 | 19 ms | 5.7 s | 300x | 1.0 GB | 219 ms | 11x | 73 MB |
| 96 | 52 ms | n/a | n/a | n/a | 585 ms | 11x | 163 MB |

## Reading

**Window a = 2 (the configuration behind every Test 1 and Test 2 result) is not scalable.**
- Its per-cut overhead *grows* with χ: from 150x to 300x SVD between χ = 8 and 64.
- Its peak memory grows as about χ² with a large prefactor, reaching 1 GB per cut at χ = 64. Extrapolated, that is about 4 GB at χ = 128.
- The cause is the region state W. It has 2^6 physical rows and (4χ)·16·(4χ) columns, because the 2 ancillas on each side of the cut and the outer bonds all sit on the column side. So W holds about 4096χ² entries, and the two products that build it cost about 4096χ³.
- This is cubic, the same order as the SVD, but with a constant of order 10², multiplied again by about 15 objective evaluations per cut.

**Window a = 1 scales like SVD.**
- Its overhead is flat at 11-13x SVD from χ = 8 to 96.
- Its memory is about 7x SVD's.
- Both are cubic in χ with a fixed constant, so a = 1 stays practical at large χ.

## Practicality estimate (not measured)

In the Test 1/2 runs, the cut fired on about 35% of cuts. At equal error, SVD needs about 2x the χ, which is about 8x the SVD time per cut.

- **a = 2:** about 0.35 × 200 / 8 ≈ 9x slower than SVD at equal error at moderate χ, and worse as χ grows. Memory caps it at χ ≈ 100 on a 16 GB machine.
- **a = 1:** about 0.35 × 12 / 8 ≈ 0.5x, which is near or below break-even.

Whether a = 1 keeps the accuracy gain is **unknown**. In pure states, a = 1 overfit: the region residual fell 90% but the RDM error got worse. That is the single test that decides whether rank 2 can be practical at large χ.
