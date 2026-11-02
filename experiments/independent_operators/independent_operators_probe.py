"""
Measures what the independent-operator rewrite changes in a TTNO bond dimension.

Run from the PyTreeNet checkout, so that ``pytreenet`` is importable::

    python experiments/independent_operators/independent_operators_probe.py

Every number of the companion note comes from the ``probe_output.txt`` this writes next to
itself. Families A to F are the cases the rewrite improves, families G to J the cases it must
leave exactly as they were.

For each case the bond on every cut is recorded three times, without the rewrite, with it, and
as the operator-Schmidt rank of a singular value decomposition of the dense operator, which is
the smallest bond any representation of that operator can have. The dense operator is compared
between the two builds, so a reduction that moved the operator would be caught.
"""
from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.lindbladian import generate_lindbladian
from pytreenet.operators.qcircuits.qgate import MultiControlledGate, QGate
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.random import random_tensor_node
from pytreenet.random.random_hamiltonian import random_hamiltonian
from pytreenet.random.random_special_ttns import random_mps
from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.special_ttn.pauli import (
    pauli_dense_lindbladian, pauli_lindbladian_ttno, pauli_product_mps, pauli_site_ids)
from pytreenet.core.ttn import TreeTensorNetwork
from pytreenet.ttno.independent_operators import (
    _Exact, _reduced_row_echelon, rewrite_over_independent_operators)
from pytreenet.ttno.state_diagram import TTNOFinder
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator as TTNO

IDENTITY = np.eye(2, dtype=complex)
PAULI_X, PAULI_Y, PAULI_Z = (np.asarray(m, dtype=complex) for m in pauli_matrices())
PROJ_0 = np.diag([1.0, 0.0]).astype(complex)
PROJ_1 = np.diag([0.0, 1.0]).astype(complex)
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)
S_PLUS = np.array([[0, 1], [0, 0]], dtype=complex)
S_MINUS = np.array([[0, 0], [1, 0]], dtype=complex)
NUMBER = (IDENTITY - PAULI_Z) / 2

_LINES: List[str] = []


def say(text: str = "") -> None:
    """Prints a line and keeps it for the saved output."""
    print(text)
    _LINES.append(text)


# ----------------------------------------------------------------------------------------
# helpers, taken from tests/ttno/test_independent_operators.py so the numbers match the tests
# ----------------------------------------------------------------------------------------

def chain_reference(num_sites: int) -> Tuple[MatrixProductState, List[str]]:
    """Returns an MPS to use as the reference tree, and its site identifiers in order."""
    mps = random_mps(num_sites, 2, 2, root_site=0)
    site_ids = [f"site{i}" for i in range(num_sites)]
    assert set(site_ids) == set(mps.nodes), "the MPS does not use the expected site ids"
    return mps, site_ids


def chain_with_ids(site_ids: List[str], phys_dim: int = 2) -> MatrixProductState:
    """Returns an MPS carrying exactly the given identifiers, left to right."""
    num = len(site_ids)
    shapes = [(1, phys_dim)] + [(1, 1, phys_dim)] * (num - 2) + [(1, phys_dim)]
    tensors = [np.ones(shape, dtype=complex) for shape in shapes]
    return MatrixProductState.from_tensor_list(tensors,
                                               node_prefix=list(site_ids),
                                               root_site=0)


def controlled_gate_hamiltonian(site_ids, controls, target, target_matrix) -> Hamiltonian:
    """The three strings of a controlled gate, as MultiControlledGate builds them."""
    controlled, subtracted, identity_string = (TensorProduct() for _ in range(3))
    for position, value in controls.items():
        controlled.add_operator(site_ids[position], f"Proj{value}")
        subtracted.add_operator(site_ids[position], f"Proj{value}")
        identity_string.add_operator(site_ids[position], "I2")
    controlled.add_operator(site_ids[target], "U")
    subtracted.add_operator(site_ids[target], "I2")
    identity_string.add_operator(site_ids[target], "I2")
    hamiltonian = Hamiltonian()
    hamiltonian.add_term((Fraction(1), "1", controlled))
    hamiltonian.add_term((Fraction(-1), "1", subtracted))
    hamiltonian.add_term((Fraction(1), "1", identity_string))
    hamiltonian.update_mappings({"I2": IDENTITY,
                                 "Proj0": PROJ_0,
                                 "Proj1": PROJ_1,
                                 "U": target_matrix},
                                {"1": 1.0 + 0.0j})
    return hamiltonian


def chain_bonds(ttno: TTNO, site_ids: List[str]) -> List[int]:
    """Returns the bond dimension on every cut of a chain TTNO, left to right."""
    return [ttno.tensors[site_ids[i + 1]].shape[0] for i in range(len(site_ids) - 1)]


def dense_matrix(ttno: TTNO, site_ids: List[str]) -> np.ndarray:
    """Returns the dense operator of a TTNO with ``site_ids[0]`` the most significant leg."""
    matrix, order = ttno.as_matrix(order=list(site_ids))
    assert order == list(site_ids), f"as_matrix returned the order {order}"
    return matrix


def operator_schmidt_ranks(matrix, num_sites: int, tol: float = 1e-10) -> List[int]:
    """Returns the operator-Schmidt rank of a dense operator on every cut of a chain."""
    tensor = np.asarray(matrix).reshape([2] * (2 * num_sites))
    ranks = []
    for cut in range(1, num_sites):
        permutation = [*range(cut),
                       *range(num_sites, num_sites + cut),
                       *range(cut, num_sites),
                       *range(num_sites + cut, 2 * num_sites)]
        reshuffled = tensor.transpose(permutation).reshape(4 ** cut, -1)
        singular = np.linalg.svd(reshuffled, compute_uv=False)
        ranks.append(int((singular > tol).sum()))
    return ranks


def all_bonds(ttno: TreeTensorNetwork) -> Dict[str, int]:
    """Returns the dimension of every parent bond of a tree, keyed by the child."""
    return {node_id: node.parent_leg_dim()
            for node_id, node in ttno.nodes.items() if not node.is_root()}


def symbols_per_node(hamiltonian: Hamiltonian) -> Dict[str, int]:
    """Returns how many distinct symbols each node carries."""
    counts: Dict[str, set] = {}
    for _, _, tensor_product in hamiltonian.terms:
        for node_id, symbol in tensor_product.items():
            counts.setdefault(node_id, set()).add(symbol)
    return {node_id: len(symbols) for node_id, symbols in counts.items()}


def build_both(hamiltonian: Hamiltonian,
               reference,
               method: TTNOFinder = TTNOFinder.SGE
               ) -> Tuple[TTNO, TTNO, float, float]:
    """Returns the TTNO without and with the rewrite, and the time each build took."""
    start = time.perf_counter()
    without = TTNO.from_hamiltonian(hamiltonian, reference, method=method,
                                    combine_dependent_operators=False)
    plain_time = time.perf_counter() - start
    start = time.perf_counter()
    with_pass = TTNO.from_hamiltonian(hamiltonian, reference, method=method)
    pass_time = time.perf_counter() - start
    return without, with_pass, plain_time, pass_time


def chain_case(name: str,
               hamiltonian: Hamiltonian,
               reference,
               site_ids: List[str],
               expect_drop: bool
               ) -> Tuple[List[int], List[int], List[int]]:
    """Reports one chain case and checks the operator did not move."""
    without, with_pass, plain_time, pass_time = build_both(hamiltonian, reference)
    reference_matrix = dense_matrix(without, site_ids)
    np.testing.assert_allclose(dense_matrix(with_pass, site_ids), reference_matrix,
                               atol=1e-12, err_msg=f"{name} moved the operator")
    before = chain_bonds(without, site_ids)
    after = chain_bonds(with_pass, site_ids)
    ranks = operator_schmidt_ranks(reference_matrix, len(site_ids))
    raised = [i for i, (a, b) in enumerate(zip(after, before)) if a > b]
    off_rank = after != ranks
    padded = hamiltonian.pad_with_identities(reference)
    terms_before = len(padded.terms)
    terms_after = len(rewrite_over_independent_operators(padded).terms)
    moved = "yes" if before != after else "no "
    flags = ""
    if raised:
        flags += f" RAISED on cuts {raised}"
    if off_rank:
        flags += " ABOVE THE RANK"
    say(f"  {name:<30} before {str(before):<21} after {str(after):<21} "
        f"rank {str(ranks):<21} drop {moved} terms {terms_before}->{terms_after} "
        f"t {plain_time:.3f}s/{pass_time:.3f}s{flags}")
    if expect_drop:
        assert before != after, f"{name} was expected to improve and did not"
    return before, after, ranks


# ----------------------------------------------------------------------------------------
# families that improve
# ----------------------------------------------------------------------------------------

def family_a() -> None:
    """A controlled phase, controls on the first sites, target on the last."""
    say("A. controlled phase, U = exp(0.7i) I, controls leftmost, target rightmost")
    phase = np.exp(0.7j) * IDENTITY
    for num_controls in (1, 2, 3):
        for num_sites in range(num_controls + 1, 9):
            mps, site_ids = chain_reference(num_sites)
            controls = {position: 1 for position in range(num_controls)}
            hamiltonian = controlled_gate_hamiltonian(site_ids, controls,
                                                      num_sites - 1, phase)
            chain_case(f"m={num_controls} N={num_sites}", hamiltonian, mps, site_ids,
                       expect_drop=num_sites > num_controls + 1)
    say()


def family_b() -> None:
    """A controlled phase whose controls do not sit next to the target."""
    say("B. controlled phase with a gap, controls {0, 3}, target rightmost")
    phase = np.exp(0.7j) * IDENTITY
    for num_sites in range(5, 9):
        mps, site_ids = chain_reference(num_sites)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1, 3: 1},
                                                  num_sites - 1, phase)
        chain_case(f"N={num_sites}", hamiltonian, mps, site_ids, expect_drop=True)
    say()


def family_c() -> None:
    """A controlled identity, where the whole gate is the identity."""
    say("C. controlled identity, U = I")
    for num_sites in (4, 6):
        mps, site_ids = chain_reference(num_sites)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1, 1: 1},
                                                  num_sites - 1, IDENTITY)
        chain_case(f"N={num_sites}", hamiltonian, mps, site_ids, expect_drop=True)
    say()


def family_d() -> None:
    """Hamiltonians whose symbols hide a dependence between projectors or number operators."""
    say("D. dependent symbol sets on a node")

    mps, site_ids = chain_reference(2)
    hamiltonian = Hamiltonian()
    for frac, left, right in ((Fraction(1), "I2", "X"),
                              (Fraction(-1), "Proj0", "X"),
                              (Fraction(-1), "Proj1", "X"),
                              (Fraction(1), "Proj0", "Z")):
        hamiltonian.add_term((frac, "1", TensorProduct({site_ids[0]: left,
                                                         site_ids[1]: right})))
    hamiltonian.update_mappings({"I2": IDENTITY, "X": PAULI_X, "Z": PAULI_Z,
                                 "Proj0": PROJ_0, "Proj1": PROJ_1},
                                {"1": 1.0 + 0.0j})
    chain_case("projectors sum to identity", hamiltonian, mps, site_ids, expect_drop=True)

    for num_sites in (4, 6):
        mps, site_ids = chain_reference(num_sites)
        hamiltonian = Hamiltonian()
        for i in range(num_sites - 1):
            hamiltonian.add_term((Fraction(1), "J",
                                  TensorProduct({site_ids[i]: "n",
                                                  site_ids[i + 1]: "n"})))
        for i in range(num_sites):
            hamiltonian.add_term((Fraction(1), "h",
                                  TensorProduct({site_ids[i]: "Z"})))
            hamiltonian.add_term((Fraction(1), "mu",
                                  TensorProduct({site_ids[i]: "n"})))
        hamiltonian.update_mappings({"n": NUMBER, "Z": PAULI_Z, "I2": IDENTITY},
                                    {"J": 1.0 + 0.0j, "h": 0.3 + 0.0j,
                                     "mu": 0.7 + 0.0j})
        chain_case(f"number and field, N={num_sites}", hamiltonian, mps, site_ids,
                   expect_drop=True)

    # an XY chain whose hopping is written with ladder operators and whose anisotropy is
    # written with Cartesian components, so a node carries {S+, S-, X, Y} of which only
    # two are independent
    for num_sites in (4, 6):
        mps, site_ids = chain_reference(num_sites)
        hamiltonian = Hamiltonian()
        for i in range(num_sites - 1):
            for left, right in (("Sp", "Sm"), ("Sm", "Sp")):
                hamiltonian.add_term((Fraction(1), "J",
                                      TensorProduct({site_ids[i]: left,
                                                     site_ids[i + 1]: right})))
            for symbol in ("X", "Y"):
                hamiltonian.add_term((Fraction(1), "g",
                                      TensorProduct({site_ids[i]: symbol,
                                                     site_ids[i + 1]: symbol})))
        for i in range(num_sites):
            hamiltonian.add_term((Fraction(1), "h",
                                  TensorProduct({site_ids[i]: "Z"})))
        hamiltonian.update_mappings({"Sp": S_PLUS, "Sm": S_MINUS, "X": PAULI_X,
                                     "Y": PAULI_Y, "Z": PAULI_Z, "I2": IDENTITY},
                                    {"J": 1.0 + 0.0j, "g": 0.4 + 0.0j,
                                     "h": 0.3 + 0.0j})
        chain_case(f"XY, ladder and Cartesian, N={num_sites}", hamiltonian, mps,
                   site_ids, expect_drop=True)
    say()


def symbolic_lindbladian(num_qubits: int, field_symbols, jump_symbols, jump_dict):
    """Returns the symbolic Lindbladian of a transverse-field chain and its site order."""
    qubits = [f"q{i}" for i in range(num_qubits)]
    system = Hamiltonian()
    for i in range(num_qubits - 1):
        system.add_term((Fraction(1), "J", TensorProduct({qubits[i]: "Z",
                                                          qubits[i + 1]: "Z"})))
    for qubit in qubits:
        for symbol in field_symbols:
            system.add_term((Fraction(1), "h", TensorProduct({qubit: symbol})))
    system.update_mappings({"X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z, "I2": IDENTITY},
                           {"J": 1.0 + 0.0j, "h": 0.5 + 0.0j})
    jumps = [(Fraction(1), "g", TensorProduct({qubit: symbol}))
             for qubit in qubits for symbol in jump_symbols]
    lindbladian = generate_lindbladian(system, jumps, dict(jump_dict),
                                       {"g": 0.1 + 0.0j})
    site_ids = [f"{qubit}{suffix}" for qubit in qubits
                for suffix in ("_ket", "_bra")]
    return lindbladian, site_ids


def family_e() -> None:
    """The symbolic Lindbladian of the library, on a chain of ket and bra nodes."""
    say("E. generate_lindbladian on a chain of ket and bra nodes")
    say("   every node carries more than d^2 = 4 distinct symbols, so it is dependent")
    cases = (("field X, jump sigma-minus", ["X"], ["Lm"], {"Lm": SIGMA_MINUS}),
             ("field X and Y, jump sigma-minus", ["X", "Y"], ["Lm"],
              {"Lm": SIGMA_MINUS}),
             ("field X and Y, jumps sigma-minus and Z", ["X", "Y"], ["Lm", "Zj"],
              {"Lm": SIGMA_MINUS, "Zj": PAULI_Z}))
    for label, fields, jump_symbols, jump_dict in cases:
        for num_qubits in (2, 3):
            lindbladian, site_ids = symbolic_lindbladian(num_qubits, fields,
                                                         jump_symbols, jump_dict)
            mps = chain_with_ids(site_ids)
            counts = symbols_per_node(lindbladian.pad_with_identities(mps))
            say(f"  {label}, N={num_qubits}, symbols per node "
                f"{sorted(counts.values(), reverse=True)}")
            chain_case(f"N={num_qubits}", lindbladian, mps, site_ids,
                       expect_drop=False)
    say()


def family_f() -> None:
    """Controlled gates whose target is independent of the identity, which must not move."""
    say("F. controlled gates with an independent target, the negative control")
    theta = 2 * np.pi / 8
    targets = (("diag(1, exp(i theta)), a controlled phase gate",
                np.diag([1.0, np.exp(1j * theta)]).astype(complex)),
               ("X, a controlled X", PAULI_X),
               ("H, a controlled Hadamard",
                np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)))
    for label, matrix in targets:
        mps, site_ids = chain_reference(4)
        hamiltonian = controlled_gate_hamiltonian(site_ids, {0: 1}, 3, matrix)
        padded = hamiltonian.pad_with_identities(mps)
        same_object = rewrite_over_independent_operators(padded) is padded
        before, after, ranks = chain_case(f"target {label}", hamiltonian, mps,
                                          site_ids, expect_drop=False)
        say(f"    the terms are returned unchanged {same_object}")
        assert before == after, f"{label} moved"
    say()


# ----------------------------------------------------------------------------------------
# families that must not move
# ----------------------------------------------------------------------------------------

def construct_tree_root_at_1():
    """The eight-site reference tree of the randomised experiment of the SGE paper."""
    from pytreenet.ttns.ttns import TreeTensorNetworkState
    ttns = TreeTensorNetworkState()
    node1, tensor1 = random_tensor_node((1, 1, 2), identifier="site1")
    node2, tensor2 = random_tensor_node((1, 1, 1, 2), identifier="site2")
    node3, tensor3 = random_tensor_node((1, 2), identifier="site3")
    node4, tensor4 = random_tensor_node((1, 2), identifier="site4")
    node5, tensor5 = random_tensor_node((1, 1, 1, 2), identifier="site5")
    node6, tensor6 = random_tensor_node((1, 2), identifier="site6")
    node7, tensor7 = random_tensor_node((1, 1, 2), identifier="site7")
    node8, tensor8 = random_tensor_node((1, 2), identifier="site8")
    ttns.add_root(node1, tensor1)
    ttns.add_child_to_parent(node2, tensor2, 0, "site1", 0)
    ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
    ttns.add_child_to_parent(node4, tensor4, 0, "site2", 2)
    ttns.add_child_to_parent(node5, tensor5, 0, "site1", 1)
    ttns.add_child_to_parent(node6, tensor6, 0, "site5", 1)
    ttns.add_child_to_parent(node7, tensor7, 0, "site5", 2)
    ttns.add_child_to_parent(node8, tensor8, 0, "site7", 1)
    return ttns


def family_g() -> None:
    """The randomised Hamiltonians of the SGE paper, whose pool is a basis already."""
    say("G. randomised Hamiltonians over {X, Y, Z, I2} on the paper's eight-site tree")
    conversion = {"X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z, "I2": IDENTITY}
    untouched = 0
    sizes = []
    start = time.perf_counter()
    for seed in range(20):
        tree = construct_tree_root_at_1()
        # an integer seed makes random_symbolic_term repeat one term for ever, so a
        # generator is passed instead
        hamiltonian = random_hamiltonian(30, list(conversion), tree,
                                         num_sites=(2, 4),
                                         conversion_dict=conversion,
                                         seed=np.random.default_rng(seed))
        padded = hamiltonian.pad_with_identities(tree)
        sizes.append(len(padded.terms))
        if rewrite_over_independent_operators(padded) is padded:
            untouched += 1
    say(f"  30 terms per seed, {min(sizes)} to {max(sizes)} strings after padding, "
        f"{untouched}/20 returned the same object in {time.perf_counter() - start:.2f}s")
    assert untouched == 20, "the paper's own experiment moved"
    identical = 0
    for seed in range(3):
        tree = construct_tree_root_at_1()
        hamiltonian = random_hamiltonian(8, list(conversion), tree,
                                         num_sites=(2, 4),
                                         conversion_dict=conversion,
                                         seed=np.random.default_rng(seed))
        without, with_pass, plain_time, pass_time = build_both(hamiltonian, tree)
        if all_bonds(without) == all_bonds(with_pass):
            identical += 1
        say(f"  seed {seed}, 8 terms built both ways, bonds identical "
            f"{all_bonds(without) == all_bonds(with_pass)}, "
            f"{plain_time:.2f}s against {pass_time:.2f}s")
    assert identical == 3, "a built TTNO moved"
    say()


def family_h() -> None:
    """The multi-controlled gate of the library, whose target is never a phase."""
    say("H. shipped MultiControlledGate on 4 qubits")
    for operation in (QGate.PAULI_X, QGate.PAULI_Y, QGate.PAULI_Z, QGate.HADAMARD):
        mps, site_ids = chain_reference(4)
        gate = MultiControlledGate([site_ids[0]], [site_ids[1]], site_ids[3],
                                   operation, "MCG")
        hamiltonian = gate.as_sum_of_products()
        padded = hamiltonian.pad_with_identities(mps)
        same_object = rewrite_over_independent_operators(padded) is padded
        without, with_pass, _, _ = build_both(hamiltonian, mps)
        before = chain_bonds(without, site_ids)
        after = chain_bonds(with_pass, site_ids)
        np.testing.assert_allclose(dense_matrix(with_pass, site_ids),
                                   dense_matrix(without, site_ids), atol=1e-12)
        say(f"  {operation.value:<10} before {str(before):<12} after {str(after):<12} "
            f"same object {same_object}")
        assert before == after and same_object, f"{operation} moved"
    say()


def family_i() -> None:
    """The fused Pauli Lindbladian, whose float operators are nearly dependent."""
    say("I. fused Pauli Lindbladian against its dense generator")

    def random_ket(rng):
        vec = rng.normal(size=2) + 1j * rng.normal(size=2)
        return vec / np.linalg.norm(vec)

    def random_hermitian(rng, dim):
        mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        return (mat + mat.conj().T) / 2

    for num_qubits in (3, 4):
        rng = np.random.default_rng(99)
        bonds = {(q, q + 1): random_hermitian(rng, 4) for q in range(num_qubits - 1)}
        bonds[(0, 2)] = random_hermitian(rng, 4)
        sites = {q: random_hermitian(rng, 2) for q in range(num_qubits)}
        jumps = {q: [(SIGMA_MINUS, 0.1 + 0.05 * q)] for q in range(num_qubits)}
        jumps[0].append((PAULI_Z, 0.07))
        state = pauli_product_mps([random_ket(rng) for _ in range(num_qubits)])
        ids = pauli_site_ids(state)
        ttno = pauli_lindbladian_ttno(state, bonds, sites, jumps)
        dense = pauli_dense_lindbladian(num_qubits, bonds, sites, jumps)
        matrix, order = ttno.as_matrix()
        dims = [int(round(np.sqrt(ttno.nodes[n].open_dimension()))) for n in order]
        tensor = matrix.reshape(dims + dims)
        total = len(order)
        keep = [order.index(s) for s in ids]
        drop = [i for i in range(total) if i not in keep]
        tensor = tensor.transpose(keep + drop
                                 + [total + i for i in keep]
                                 + [total + i for i in drop])
        built = tensor.reshape(4 ** len(keep), 4 ** len(keep))
        error = float(np.abs(built - dense).max())
        say(f"  N={num_qubits} largest deviation from the dense generator {error:.2e}")
        assert error < 1e-10, "the guard did not protect the Lindbladian"
    say()


def family_j() -> None:
    """Input the rewrite must hand back unchanged rather than empty."""
    say("J. degenerate input")
    mps, site_ids = chain_reference(3)
    single = Hamiltonian()
    single.add_term((Fraction(1), "1", TensorProduct({site_ids[0]: "X",
                                                       site_ids[2]: "Z"})))
    single.update_mappings({"X": PAULI_X, "Z": PAULI_Z}, {"1": 1.0 + 0.0j})
    say(f"  single term, same object {rewrite_over_independent_operators(single) is single}")

    mps, site_ids = chain_reference(2)
    cancelling = Hamiltonian()
    for frac, left in ((Fraction(1), "Proj0"), (Fraction(-1), "Proj0")):
        cancelling.add_term((frac, "1", TensorProduct({site_ids[0]: left,
                                                        site_ids[1]: "I2"})))
    for frac in (Fraction(1), Fraction(-1)):
        cancelling.add_term((frac, "1", TensorProduct({site_ids[0]: "I2",
                                                        site_ids[1]: "I2"})))
    cancelling.update_mappings({"I2": IDENTITY, "Proj0": PROJ_0}, {"1": 1.0 + 0.0j})
    rewritten = rewrite_over_independent_operators(cancelling)
    say(f"  every term cancels, result empty {rewritten.is_empty()}")
    assert not rewritten.is_empty(), "the rewrite emptied the Hamiltonian"
    say()


def family_k() -> None:
    """How far an exactly rational dependence can be from holding in double precision."""
    say("K. a dependence that holds over the rationals and not in double precision")
    rng = np.random.default_rng(11)

    def diagonal():
        return np.diag(rng.normal(size=4) + 1j * rng.normal(size=4))

    base = diagonal()
    drift = diagonal()
    nearly = [base + k * 1e-9 * drift for k in range(4)]
    nearly.append(diagonal())
    flat = np.array([op.ravel() for op in nearly])
    singular = np.linalg.svd(flat[:4], compute_uv=False)
    exact = [[_Exact.of(entry) for entry in op.ravel()] for op in nearly]
    system = [[*(held[i] for held in exact[:4]), exact[4][i]]
              for i in range(len(exact[0]))]
    reduced, pivots = _reduced_row_echelon(system)
    coordinates = {pivot: row[-1] for pivot, row in zip(pivots, reduced) if row[-1]}
    largest = max(abs(complex(value)) for value in coordinates.values())
    rebuilt = np.zeros_like(nearly[4])
    for position, value in coordinates.items():
        rebuilt = rebuilt + complex(value) * nearly[position]
    miss = float(np.abs(rebuilt - nearly[4]).max()) / float(np.abs(nearly[4]).max())
    say(f"  five operators on a node of dimension 4, exact rank {len(pivots)}, "
        f"numerical rank {np.linalg.matrix_rank(flat, tol=1e-10)}")
    say(f"  singular values of the first four {np.array2string(singular, precision=2)}")
    say(f"  largest exact coordinate {largest:.2e}, "
        f"rebuilding misses the operator by {miss:.1f} times its largest entry")

    site_ids = ["site0", "site1"]
    mps = chain_with_ids(site_ids, phys_dim=4)
    hamiltonian = Hamiltonian()
    names = [f"near{k}" for k in range(len(nearly))]
    for k, name in enumerate(names):
        hamiltonian.add_term((Fraction(k + 1), "1",
                              TensorProduct({site_ids[0]: name,
                                             site_ids[1]: "I4"})))
    hamiltonian.update_mappings({"I4": np.eye(4, dtype=complex),
                                 **{name: op.astype(complex)
                                    for name, op in zip(names, nearly)}},
                                {"1": 1.0 + 0.0j})
    padded = hamiltonian.pad_with_identities(mps)
    refused = rewrite_over_independent_operators(padded) is padded
    say(f"  the node is left as it is {refused}")
    assert refused, "the guard did not refuse the node"
    say()



def main() -> None:
    """Runs every family and saves the output."""
    say("Independent-operator rewrite, measured bonds")
    say("=" * 100)
    say()
    for family in (family_a, family_b, family_c, family_d, family_e, family_f,
                   family_g, family_h, family_i, family_j, family_k):
        family()
    out = Path(__file__).with_name("probe_output.txt")
    out.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
