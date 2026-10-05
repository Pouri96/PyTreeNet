import warnings
from copy import deepcopy
from unittest import TestCase, main

import numpy as np

from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.special_ttn.pauli import (
    PAULIS, U_PAULI, assert_fused_pauli, network_is_real, pauli_coeffs_to_rho,
    pauli_from_dense, pauli_hermiticity_defect, pauli_local_expectation, pauli_product_mps,
    pauli_product_tree_allphys, pauli_product_tree_virtnode, pauli_purity, pauli_rescale_trace,
    pauli_rho_to_coeffs, pauli_site_ids, pauli_site_vector, pauli_to_coeffs, pauli_to_dense,
    pauli_trace, pauli_trace_cap, realify_network)

SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)


def random_ket(rng):
    vec = rng.normal(size=2) + 1j * rng.normal(size=2)
    return vec / np.linalg.norm(vec)


def random_hermitian(rng, dim):
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (mat + mat.conj().T) / 2


def product_rho(kets):
    rho = np.ones((1, 1), dtype=complex)
    for ket in kets:
        rho = np.kron(rho, np.outer(ket, ket.conj()))
    return rho


def random_real_net(template, chi, rng):
    """A real network on the topology of ``template`` with every virtual bond set to ``chi``."""
    net = deepcopy(template)
    for node_id, node in net.nodes.items():
        open_dim = int(np.prod([net.tensors[node_id].shape[leg] for leg in node.open_legs]))
        shape = [chi] * node.nneighbours() + [open_dim]
        net.replace_tensor(node_id, rng.normal(size=shape), new_shape=True)
    return net


def three_topologies(kets):
    return {"chain": pauli_product_mps(kets),
            "allphys": pauli_product_tree_allphys(kets),
            "heap": pauli_product_tree_allphys(kets, order="heap"),
            "virtnode": pauli_product_tree_virtnode(kets)}


class PauliTestCase(TestCase):
    def setUp(self):
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        self.rng = np.random.default_rng(1234)


class TestBasis(PauliTestCase):
    def test_u_pauli_is_unitary(self):
        self.assertTrue(np.allclose(U_PAULI @ U_PAULI.conj().T, np.eye(4), atol=1e-14))

    def test_paulis_are_hermitian(self):
        for pauli in PAULIS:
            self.assertTrue(np.allclose(pauli, pauli.conj().T))

    def test_coefficients_round_trip(self):
        for n in (1, 2, 3):
            rho = random_hermitian(self.rng, 2 ** n)
            coeffs = pauli_rho_to_coeffs(rho, n)
            self.assertTrue(np.allclose(pauli_coeffs_to_rho(coeffs, n), rho, atol=1e-13))

    def test_coefficients_are_real_iff_hermitian(self):
        herm = random_hermitian(self.rng, 8)
        self.assertLess(np.abs(pauli_rho_to_coeffs(herm, 3).imag).max(), 1e-13)
        non_herm = herm + 1j * np.eye(8)
        self.assertGreater(np.abs(pauli_rho_to_coeffs(non_herm, 3).imag).max(), 1e-3)

    def test_coefficients_are_traces_against_normalised_paulis(self):
        rho = random_hermitian(self.rng, 4)
        coeffs = pauli_rho_to_coeffs(rho, 2).reshape(4, 4)
        for a in range(4):
            for b in range(4):
                sigma = np.kron(PAULIS[a], PAULIS[b]) / 2
                self.assertAlmostEqual(coeffs[a, b], np.trace(sigma @ rho).real, places=12)

    def test_site_vector_matches_pure_state_coefficients(self):
        ket = random_ket(self.rng)
        expected = pauli_rho_to_coeffs(np.outer(ket, ket.conj()), 1)
        self.assertTrue(np.allclose(pauli_site_vector(ket), expected.real, atol=1e-14))


class TestBuilders(PauliTestCase):
    def test_product_states_match_dense(self):
        for n in (2, 3, 4, 5):
            kets = [random_ket(self.rng) for _ in range(n)]
            rho = product_rho(kets)
            for name, state in three_topologies(kets).items():
                with self.subTest(n=n, topology=name):
                    self.assertTrue(np.allclose(pauli_to_dense(state), rho, atol=1e-13))
                    self.assertTrue(network_is_real(state, tol=0.0))

    def test_single_qubit_chain(self):
        ket = random_ket(self.rng)
        state = pauli_product_mps([ket])
        self.assertTrue(np.allclose(pauli_to_dense(state), product_rho([ket]), atol=1e-13))

    def test_all_bonds_are_one(self):
        kets = [random_ket(self.rng) for _ in range(5)]
        for name, state in three_topologies(kets).items():
            for node_id, node in state.nodes.items():
                for neighbour in node.neighbouring_nodes():
                    self.assertEqual(1, node.neighbour_dim(neighbour), msg=name)

    def test_virtual_nodes_have_trivial_open_legs(self):
        kets = [random_ket(self.rng) for _ in range(5)]
        state = pauli_product_tree_virtnode(kets)
        site_ids = set(pauli_site_ids(state))
        self.assertGreater(len(state.nodes), len(site_ids))
        for node_id in set(state.nodes) - site_ids:
            self.assertEqual(1, state.nodes[node_id].open_dimension())

    def test_all_phys_tree_has_one_node_per_qubit(self):
        kets = [random_ket(self.rng) for _ in range(6)]
        for order in ("inorder", "heap"):
            state = pauli_product_tree_allphys(kets, order=order)
            self.assertEqual(6, len(state.nodes))
        with self.assertRaises(ValueError):
            pauli_product_tree_allphys(kets, order="bogus")

    def test_from_dense_is_exact_and_real(self):
        for n in (2, 3, 4):
            rho = random_hermitian(self.rng, 2 ** n)
            rho = rho @ rho.conj().T
            rho /= np.trace(rho)
            state = pauli_from_dense(rho, tol=0.0)
            self.assertTrue(network_is_real(state, tol=0.0))
            self.assertTrue(np.allclose(pauli_to_dense(state), rho, atol=1e-12))

    def test_from_dense_of_a_product_state_has_bond_one(self):
        rho = product_rho([random_ket(self.rng) for _ in range(4)])
        state = pauli_from_dense(rho)
        for node in state.nodes.values():
            for neighbour in node.neighbouring_nodes():
                self.assertEqual(1, node.neighbour_dim(neighbour))


class TestSiteIds(PauliTestCase):
    def test_site_ids_in_qubit_order(self):
        kets = [random_ket(self.rng) for _ in range(5)]
        for name, state in three_topologies(kets).items():
            self.assertEqual([f"qubit{q}" for q in range(5)], pauli_site_ids(state), msg=name)

    def test_custom_prefix(self):
        kets = [random_ket(self.rng) for _ in range(3)]
        state = pauli_product_mps(kets, node_prefix="s")
        self.assertEqual(["s0", "s1", "s2"], pauli_site_ids(state))
        with self.assertRaises(ValueError):
            pauli_site_ids(state, prefix="qubit")

    def test_pure_state_is_rejected(self):
        tensors = [self.rng.normal(size=(2, 2)), self.rng.normal(size=(2, 2, 2)),
                   self.rng.normal(size=(2, 2))]
        pure = MatrixProductState.from_tensor_list(tensors, node_prefix="qubit")
        with self.assertRaisesRegex(ValueError, "pure state or a ket/bra"):
            assert_fused_pauli(pure)

    def test_gap_in_indices_is_rejected(self):
        tensors = [self.rng.normal(size=(1, 4)), self.rng.normal(size=(1, 1, 4)),
                   self.rng.normal(size=(1, 4))]
        state = MatrixProductState.from_tensor_list(tensors, node_prefix=["q0", "q1", "q3"])
        with self.assertRaisesRegex(ValueError, "must be q0..q2"):
            pauli_site_ids(state)

    def test_assert_fused_returns_site_ids(self):
        state = pauli_product_tree_virtnode([random_ket(self.rng) for _ in range(4)])
        self.assertEqual([f"qubit{q}" for q in range(4)], assert_fused_pauli(state))


class TestTraceCapAndReadout(PauliTestCase):
    def test_cap_has_bond_one_and_matches_topology(self):
        kets = [random_ket(self.rng) for _ in range(5)]
        for name, state in three_topologies(kets).items():
            state = random_real_net(state, 3, self.rng)
            cap = pauli_trace_cap(state)
            self.assertEqual(set(state.nodes), set(cap.nodes), msg=name)
            for node in cap.nodes.values():
                for neighbour in node.neighbouring_nodes():
                    self.assertEqual(1, node.neighbour_dim(neighbour), msg=name)

    def test_trace_purity_and_expectations_match_dense(self):
        kets = [random_ket(self.rng) for _ in range(4)]
        sigma_z = PAULIS[3]
        for name, template in three_topologies(kets).items():
            state = random_real_net(template, 3, self.rng)
            rho = pauli_to_dense(state)
            with self.subTest(topology=name):
                np.testing.assert_allclose(pauli_trace(state), np.trace(rho), rtol=1e-11)
                np.testing.assert_allclose(pauli_purity(state), np.trace(rho @ rho).real,
                                           rtol=1e-11)
                for q in range(4):
                    for op in (sigma_z, SIGMA_MINUS):
                        full = np.kron(np.kron(np.eye(2 ** q), op), np.eye(2 ** (3 - q)))
                        np.testing.assert_allclose(pauli_local_expectation(state, op, q),
                                                   np.trace(full @ rho), rtol=1e-10)

    def test_trace_of_a_complex_typed_real_network(self):
        state = random_real_net(pauli_product_mps([random_ket(self.rng) for _ in range(4)]),
                                3, self.rng)
        complex_state = deepcopy(state)
        complex_state.to_complex(128)
        self.assertAlmostEqual(pauli_trace(state), pauli_trace(complex_state), places=12)

    def test_cap_is_reusable(self):
        state = random_real_net(pauli_product_mps([random_ket(self.rng) for _ in range(4)]),
                                3, self.rng)
        cap = pauli_trace_cap(state)
        self.assertEqual(pauli_trace(state), pauli_trace(state, cap=cap))

    def test_to_coeffs_order_is_qubit_order_on_every_topology(self):
        kets = [random_ket(self.rng) for _ in range(5)]
        expected = pauli_rho_to_coeffs(product_rho(kets), 5).real
        for name, state in three_topologies(kets).items():
            self.assertTrue(np.allclose(pauli_to_coeffs(state), expected, atol=1e-13), msg=name)


class TestRealityAndHermiticity(PauliTestCase):
    def test_real_network_has_zero_defect(self):
        state = random_real_net(pauli_product_mps([random_ket(self.rng) for _ in range(4)]),
                                3, self.rng)
        self.assertEqual(0.0, pauli_hermiticity_defect(state))

    def test_complex_network_defect_matches_dense(self):
        template = pauli_product_mps([random_ket(self.rng) for _ in range(3)])
        state = random_real_net(template, 3, self.rng)
        state.to_complex(128)
        node_id = "qubit1"
        state.replace_tensor(node_id, state.tensors[node_id]
                             * np.exp(0.7j), new_shape=False)
        rho = pauli_to_dense(state)
        expected = np.linalg.norm(rho - rho.conj().T)
        self.assertGreater(expected, 1e-2)
        self.assertAlmostEqual(expected, pauli_hermiticity_defect(state), places=6)

    def test_network_is_real(self):
        state = pauli_product_mps([random_ket(self.rng) for _ in range(3)])
        self.assertTrue(network_is_real(state))
        state.to_complex(128)
        self.assertTrue(network_is_real(state))
        state.replace_tensor("qubit0", state.tensors["qubit0"] * 1j, new_shape=False)
        self.assertFalse(network_is_real(state))

    def test_realify_casts_complex_typed_real_values(self):
        state = pauli_product_mps([random_ket(self.rng) for _ in range(3)])
        state.to_complex(128)
        realify_network(state, "test")
        for tensor in state.tensors.values():
            self.assertTrue(np.isrealobj(tensor))

    def test_realify_refuses_an_imaginary_part(self):
        state = pauli_product_mps([random_ket(self.rng) for _ in range(3)])
        state.to_complex(128)
        state.replace_tensor("qubit2", state.tensors["qubit2"] * 1j, new_shape=False)
        with self.assertRaisesRegex(ValueError, "qubit2"):
            realify_network(state, "test")


class TestRescaleTrace(PauliTestCase):
    def test_rescale_makes_trace_one(self):
        for name, template in three_topologies([random_ket(self.rng) for _ in range(4)]).items():
            state = random_real_net(template, 2, self.rng)
            trace = pauli_trace(state)
            self.assertTrue(pauli_rescale_trace(state, trace), msg=name)
            self.assertAlmostEqual(1.0, pauli_trace(state).real, places=10, msg=name)

    def test_rescale_skips_an_undefined_trace(self):
        state = pauli_product_mps([random_ket(self.rng) for _ in range(3)])
        before = deepcopy(state.tensors)
        self.assertFalse(pauli_rescale_trace(state, 0.0))
        self.assertFalse(pauli_rescale_trace(state, complex("nan")))
        for node_id, tensor in before.items():
            self.assertTrue(np.array_equal(tensor, state.tensors[node_id]))


if __name__ == "__main__":
    main()
