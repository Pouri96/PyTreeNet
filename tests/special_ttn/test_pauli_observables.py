"""Expectation values of a fused Pauli chain against dense references that share no code with them."""
import warnings
from unittest import TestCase, main

import numpy as np

from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.special_ttn.pauli import (
    pauli_expectation,
    pauli_from_dense,
    pauli_local_expectation,
    pauli_local_expectation_sweep,
    pauli_product_mps,
    pauli_product_tree_allphys,
    pauli_reweight_network,
    pauli_site_ids,
)
from pytreenet.special_ttn.pauli.dmt import _read_chain

N = 5
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)


def embed(op, qubits, n=N):
    """``op`` on the listed qubits of an ``n``-qubit register, in the order listed."""
    rest = [q for q in range(n) if q not in qubits]
    order = list(qubits) + rest
    full = np.kron(op, np.eye(2 ** (n - len(qubits)))).reshape((2,) * (2 * n))
    inverse = np.argsort(order)
    return full.transpose(list(inverse) + [n + p for p in inverse]).reshape(2**n, 2**n)


def mixed_rho(rng, n=N, rank=6):
    g = rng.normal(size=(2**n, rank)) + 1j * rng.normal(size=(2**n, rank))
    rho = g @ g.conj().T
    return rho / np.trace(rho).real


def random_hermitian(rng, dim):
    h = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (h + h.conj().T) / 2


def rooted_at_the_end(net):
    """The same chain with its root on the last qubit, so every leg order is different."""
    chain = _read_chain(net, pauli_site_ids(net))
    core = [chain[0][0]] + chain[1:-1] + [chain[-1][:, 0]]
    return MatrixProductState.from_tensor_list(core, node_prefix="qubit", root_site=len(core) - 1)


def scaled(net, factor):
    net.replace_tensor(net.root_id, net.tensors[net.root_id] * factor, new_shape=False)
    return net


class ReadoutTestCase(TestCase):
    def setUp(self):
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        self.rng = np.random.default_rng(5)
        self.rho = mixed_rho(self.rng)
        self.net = pauli_from_dense(self.rho, tol=1e-14)

    def dense_local(self, op, rho=None):
        rho = self.rho if rho is None else rho
        return np.array([np.trace(embed(op, [q]) @ rho).real for q in range(N)])


class TestLocalSweep(ReadoutTestCase):
    def test_it_matches_the_dense_trace_for_every_qubit(self):
        for name, op in (("X", X), ("Y", Y), ("Z", Z), ("random", random_hermitian(self.rng, 2))):
            with self.subTest(op=name):
                got = pauli_local_expectation_sweep(self.net, op)
                self.assertEqual(got.shape, (N,))
                np.testing.assert_allclose(got, self.dense_local(op), atol=1e-13)

    def test_a_hermitian_operator_on_a_real_network_gives_a_real_array(self):
        self.assertEqual(pauli_local_expectation_sweep(self.net, Y).dtype, np.float64)

    def test_it_agrees_with_the_per_qubit_function(self):
        sweep = pauli_local_expectation_sweep(self.net, X)
        single = [pauli_local_expectation(self.net, X, q).real for q in range(N)]
        np.testing.assert_allclose(sweep, single, atol=1e-13)

    def test_a_chain_rooted_at_the_other_end_gives_the_same_values(self):
        flipped = rooted_at_the_end(self.net)
        self.assertEqual(flipped.root_id, f"qubit{N - 1}")
        np.testing.assert_allclose(pauli_local_expectation_sweep(flipped, Z),
                                   self.dense_local(Z), atol=1e-13)

    def test_normalising_divides_by_the_trace(self):
        net = scaled(pauli_from_dense(self.rho, tol=1e-14), 0.9)
        raw = pauli_local_expectation_sweep(net, Z)
        np.testing.assert_allclose(raw, 0.9 * self.dense_local(Z), atol=1e-13)
        np.testing.assert_allclose(pauli_local_expectation_sweep(net, Z, normalized=True),
                                   self.dense_local(Z), atol=1e-13)

    def test_a_vanishing_trace_cannot_be_normalised_by(self):
        net = scaled(pauli_from_dense(self.rho, tol=1e-14), 1e-15)
        with self.assertRaisesRegex(ValueError, "too small"):
            pauli_local_expectation_sweep(net, Z, normalized=True)

    def test_a_product_state_beyond_dense_reach_is_exact(self):
        n = 40
        thetas = 0.3 + 0.07 * np.arange(n)
        kets = [np.array([np.cos(t), np.sin(t)], dtype=complex) for t in thetas]
        net = pauli_product_mps(kets)
        np.testing.assert_allclose(pauli_local_expectation_sweep(net, Z), np.cos(2 * thetas),
                                   atol=1e-13)
        np.testing.assert_allclose(pauli_local_expectation_sweep(net, X), np.sin(2 * thetas),
                                   atol=1e-13)

    def test_the_input_is_not_modified(self):
        before = {k: v.copy() for k, v in self.net.tensors.items()}
        pauli_local_expectation_sweep(self.net, Z, normalized=True)
        for node_id, tensor in before.items():
            self.assertTrue(np.array_equal(tensor, self.net.tensors[node_id]))

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "2x2"):
            pauli_local_expectation_sweep(self.net, np.eye(4))
        gauged = pauli_reweight_network(pauli_from_dense(self.rho, tol=1e-14), 2.0)
        with self.assertRaisesRegex(ValueError, "physical_state"):
            pauli_local_expectation_sweep(gauged, Z)
        tree = pauli_product_tree_allphys([np.array([1.0, 0.0], dtype=complex)] * 5)
        with self.assertRaisesRegex(ValueError, "chain"):
            pauli_local_expectation_sweep(tree, Z)


class TestExpectation(ReadoutTestCase):
    def terms(self):
        bonds = {(0, 1): random_hermitian(self.rng, 4), (2, 3): random_hermitian(self.rng, 4),
                 (0, 3): random_hermitian(self.rng, 4), (1, 4): random_hermitian(self.rng, 4)}
        sites = {0: random_hermitian(self.rng, 2), 3: random_hermitian(self.rng, 2)}
        return bonds, sites

    def dense(self, bonds, sites, rho=None):
        rho = self.rho if rho is None else rho
        total = sum(np.trace(embed(h, list(pair)) @ rho) for pair, h in bonds.items())
        return (total + sum(np.trace(embed(h, [q]) @ rho) for q, h in sites.items())).real

    def test_random_site_and_bond_terms_including_long_range(self):
        bonds, sites = self.terms()
        got = pauli_expectation(self.net, bonds, sites)
        self.assertIsInstance(got, float)
        self.assertAlmostEqual(got, self.dense(bonds, sites), places=12)

    def test_a_two_point_correlator_is_a_single_bond_term(self):
        for pair in ((0, 1), (1, 4), (0, 4)):
            with self.subTest(pair=pair):
                got = pauli_expectation(self.net, bond_terms={pair: np.kron(Z, Z)})
                want = np.trace(embed(np.kron(Z, Z), list(pair)) @ self.rho).real
                self.assertAlmostEqual(got, want, places=12)

    def test_the_energy_of_a_model_is_read_with_the_terms_that_define_it(self):
        bonds = {(q, q + 1): np.kron(Z, Z) + 0.4 * np.kron(X, X) for q in range(N - 1)}
        sites = {q: 0.7 * X + 0.3 * Z for q in range(N)}
        self.assertAlmostEqual(pauli_expectation(self.net, bonds, sites),
                               self.dense(bonds, sites), places=12)

    def test_a_chain_rooted_at_the_other_end_gives_the_same_value(self):
        bonds, sites = self.terms()
        got = pauli_expectation(rooted_at_the_end(self.net), bonds, sites)
        self.assertAlmostEqual(got, self.dense(bonds, sites), places=12)

    def test_normalising_divides_by_the_trace(self):
        bonds, sites = self.terms()
        net = scaled(pauli_from_dense(self.rho, tol=1e-14), 0.9)
        self.assertAlmostEqual(pauli_expectation(net, bonds, sites),
                               0.9 * self.dense(bonds, sites), places=12)
        self.assertAlmostEqual(pauli_expectation(net, bonds, sites, normalized=True),
                               self.dense(bonds, sites), places=12)

    def test_no_terms_reads_the_trace(self):
        net = scaled(pauli_from_dense(self.rho, tol=1e-14), 0.9)
        self.assertAlmostEqual(pauli_expectation(net), 0.9, places=12)

    def test_a_product_state_beyond_dense_reach_is_exact(self):
        n = 40
        thetas = 0.3 + 0.07 * np.arange(n)
        kets = [np.array([np.cos(t), np.sin(t)], dtype=complex) for t in thetas]
        net = pauli_product_mps(kets)
        zz = pauli_expectation(net, bond_terms={(3, 30): np.kron(Z, Z)})
        self.assertAlmostEqual(zz, np.cos(2 * thetas[3]) * np.cos(2 * thetas[30]), places=12)
        bonds = {(q, q + 1): np.kron(Z, Z) for q in range(n - 1)}
        sites = {q: X for q in range(n)}
        want = (sum(np.cos(2 * thetas[q]) * np.cos(2 * thetas[q + 1]) for q in range(n - 1))
                + np.sin(2 * thetas).sum())
        self.assertAlmostEqual(pauli_expectation(net, bonds, sites), want, places=10)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            pauli_expectation(self.net, site_terms={N + 1: Z})
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            pauli_expectation(self.net, site_terms={0: np.array([[0, 1], [0, 0]])})
        gauged = pauli_reweight_network(pauli_from_dense(self.rho, tol=1e-14), 2.0)
        with self.assertRaisesRegex(ValueError, "physical_state"):
            pauli_expectation(gauged, site_terms={0: Z})
        tree = pauli_product_tree_allphys([np.array([1.0, 0.0], dtype=complex)] * 5)
        with self.assertRaisesRegex(ValueError, "chain"):
            pauli_expectation(tree, site_terms={0: Z})


if __name__ == "__main__":
    main()
