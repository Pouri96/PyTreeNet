# Independent operators before the state diagram

`TTNO.from_hamiltonian` drives down `rank(Gamma)`. That is the smallest bond a cut admits only
when the operator chains on each side are linearly independent as operators, which the symbolic
elimination cannot check because it holds them as symbols. `rewrite_over_independent_operators`
establishes the condition exactly, and `from_hamiltonian` keeps whichever of the two state
diagrams carries the smaller total bond.

- `independent_operators.tex`, `independent_operators.pdf` — the derivation, five pages, written
  for a reader who knows the construction. Build with
  `pdflatex -interaction=nonstopmode independent_operators`, three times.
- `independent_operators_probe.py` — every number in the note. Run from the repository root with
  `python experiments/independent_operators/independent_operators_probe.py`. It writes
  `probe_output.txt` next to itself and asserts, in every case, that the dense operator is
  unchanged and that the bond does not come out above the one the terms as given produce.
- `probe_output.txt` — a saved run.

Families A to E improve, F to J must not move. The largest effect measured here is an XY chain
whose hopping is written with ladder operators and whose anisotropy is written with Cartesian
components, `[6,6,6,6,6]` becoming `[4,4,4,4,4]`.
