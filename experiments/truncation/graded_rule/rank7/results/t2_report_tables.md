### T2 criteria by gate
| gate | (a) cases holding (first-sample t*) | (a) failing cases | (b) win fraction vs best post-processed SVD, min / median over 6 cells | (c) median eps_c ratio spcf/SVD (chi 16 / 24 / 32) | peak r(nn) (chi 16 / 24 / 32) | r(nn) at t=8 (chi 16 / 24 / 32) |
|---|---|---|---|---|---|---|
| g001: rs=1e-2, eps=1e-7, taus=1 (ungated) | 12/12 | - | 0.88 / 1.00 | 0.32 / 0.30 / 0.30 | 2.12 / 1.99 / 1.95 | 1.39 / 1.71 / 1.68 |
| g06: rs=0.6, eps=1e-7, taus=1 | 11/12 | chi24 1site 0.001 (5 vs 5.5) | 0.88 / 1.00 | 0.31 / 0.30 / 0.34 | 2.00 / 1.91 / 1.78 | 1.32 / 1.58 / 1.56 |
| g03e5: rs=0.3, eps=1e-5, taus=1 | 6/12 | chi16 nn 0.001 (3.5 vs 4.5); chi16 1site 0.001 (4 vs 4.5); chi24 nn 0.001 (4.5 vs 5.5); chi24 1site 0.001 (4.5 vs 5.5); chi32 nn 0.001 (5 vs 6); chi32 1site 0.001 (5 vs 6) | 0.62 / 0.73 | 0.35 / 0.40 / 0.34 | 2.09 / 1.95 / 1.83 | 1.40 / 1.69 / 1.65 |
| s001: rs=1e-2, eps=1e-7, taus=none | 7/12 | chi16 nn 0.001 (4 vs 4.5); chi16 1site 0.001 (4 vs 4.5); chi24 1site 0.001 (5 vs 5.5); chi32 nn 0.001 (5.5 vs 6); chi32 1site 0.001 (5.5 vs 6) | 0.88 / 1.00 | 0.34 / 0.29 / 0.35 | 2.01 / 1.88 / 1.81 | 1.31 / 1.61 / 1.62 |

### t*(1e-3) and t*(1e-2) (first sample time with error above tol; inf = never up to T = 8), nn / 1-site
| arm | nn 1e-3 | 1-site 1e-3 | nn 1e-2 | 1-site 1e-2 |
|---|---|---|---|---|
| SVD chi=16 | 3.5 | 4 | 4.5 | 5 |
| SVD chi=24 | 4.5 | 4.5 | 6 | 6 |
| SVD chi=32 | 5 | 5 | 7 | 7 |
| SVD chi=36 | 5.5 | 5.5 | inf | 7.5 |
| SVD chi=48 | 6 | 6 | inf | inf |
| SVD chi=64 | 6.5 | 6.5 | inf | inf |
| SVD chi=96 | 8 | 8 | inf | inf |
| spcf g001 chi=16 | 4.5 | 4.5 | 7 | 7 |
| spcf g06 chi=16 | 4.5 | 4.5 | 7 | 6.5 |
| spcf g03e5 chi=16 | 3.5 | 4 | 7 | 6.5 |
| spcf s001 chi=16 | 4 | 4 | 7 | 6.5 |
| SVD x F_MPS chi=16 | 3.5 | 4 | 4.5 | 5 |
| SVD logF-extrap (<=3 largest chi<=16) | 3.5 | 4 | 4.5 | 5 |
| SVD 1/chi-extrap (<=3 largest chi<=16) | 3.5 | 4 | 4.5 | 5 |
| spcf g001 chi=24 | 5.5 | 5.5 | inf | inf |
| spcf g06 chi=24 | 5.5 | 5 | inf | inf |
| spcf g03e5 chi=24 | 4.5 | 4.5 | inf | inf |
| spcf s001 chi=24 | 5.5 | 5 | inf | inf |
| SVD x F_MPS chi=24 | 4.5 | 4.5 | 6 | 6.5 |
| SVD logF-extrap (<=3 largest chi<=24) | 4.5 | 5 | 7 | 7 |
| SVD 1/chi-extrap (<=3 largest chi<=24) | 3.5 | 3.5 | 4.5 | 4.5 |
| spcf g001 chi=32 | 6.5 | 6 | inf | inf |
| spcf g06 chi=32 | 6 | 6 | inf | inf |
| spcf g03e5 chi=32 | 5 | 5 | inf | inf |
| spcf s001 chi=32 | 5.5 | 5.5 | inf | inf |
| SVD x F_MPS chi=32 | 5 | 5 | 7 | inf |
| SVD logF-extrap (<=3 largest chi<=32) | 5 | 5 | 7.5 | 7.5 |
| SVD 1/chi-extrap (<=3 largest chi<=32) | 3.5 | 4 | 4.5 | 5 |
| SVD logF-extrap top3 (48,64,96) | 8 | inf | inf | inf |
| SVD 1/chi-extrap top3 (48,64,96) | 6 | 6 | inf | inf |

### nn rms error at t = 4.5 / 6.5 / 8.0 (growth phase)
| arm | chi=16 | chi=24 | chi=32 |
|---|---|---|---|
| SVD | 1.0e-02 / 2.9e-02 / 2.3e-02 | 2.1e-03 / 1.4e-02 / 1.5e-02 | 4.7e-04 / 8.6e-03 / 1.1e-02 |
| SVD x F_MPS | 1.0e-02 / 2.9e-02 / 2.6e-02 | 2.1e-03 / 1.4e-02 / 1.6e-02 | 4.7e-04 / 8.7e-03 / 1.2e-02 |
| SVD x F_true (oracle) | 1.2e-02 / 6.0e-02 / 9.9e-02 | 2.2e-03 / 2.5e-02 / 5.9e-02 | 4.7e-04 / 1.3e-02 / 3.6e-02 |
| SVD logF-extrap | 1.0e-02 / 2.9e-02 / 2.3e-02 | 1.2e-03 / 9.1e-03 / 1.7e-02 | 7.0e-04 / 6.9e-03 / 1.2e-02 |
| SVD 1/chi-extrap | 1.0e-02 / 2.9e-02 / 2.3e-02 | 1.5e-02 / 2.3e-02 / 2.9e-02 | 1.1e-02 / 1.6e-02 / 1.8e-02 |
| spcf g001 | 1.7e-03 / 7.6e-03 / 1.6e-02 | 4.3e-04 / 3.2e-03 / 7.6e-03 | 1.6e-04 / 1.5e-03 / 4.6e-03 |
| spcf g06 | 1.7e-03 / 8.6e-03 / 1.7e-02 | 4.7e-04 / 3.6e-03 / 8.7e-03 | 1.6e-04 / 2.0e-03 / 5.4e-03 |
| spcf g03e5 | 3.8e-03 / 7.9e-03 / 1.6e-02 | 2.1e-03 / 3.4e-03 / 7.8e-03 | 4.7e-04 / 1.9e-03 / 4.8e-03 |
| spcf s001 | 2.5e-03 / 8.8e-03 / 1.7e-02 | 4.8e-04 / 3.8e-03 / 8.4e-03 | 1.8e-04 / 1.9e-03 / 5.0e-03 |
| spcf g001 x F_MPS | 1.8e-03 / 8.4e-03 / 1.9e-02 | 4.3e-04 / 3.6e-03 / 9.0e-03 | 1.6e-04 / 1.7e-03 / 5.1e-03 |
