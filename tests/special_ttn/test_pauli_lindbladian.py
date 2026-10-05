import warnings
from unittest import TestCase, main

import numpy as np
from scipy.linalg import expm

from pytreenet.special_ttn.pauli import (
    pauli_coeffs_to_rho, pauli_dense_lindbladian, pauli_lindbladian_ttno, pauli_product_mps,
    pauli_product_tree_allphys, pauli_product_tree_virtnode, pauli_rho_to_coeffs, pauli_site_ids)

SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)
SIGMA_Z = np.diag([1.0, -1.0]).astype(complex)


def random_hermitian(rng, dim):
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (mat + mat.conj().T) / 2


def random_model(rng, n):
    bonds = {(q, q + 1): random_hermitian(rng, 4) for q in range(n - 1)}
    if n >= 3:
        bonds[(0, 2)] = random_hermitian(rng, 4)
    sites = {q: random_hermitian(rng, 2) for q in range(n)}
    jumps = {q: [(SIGMA_MINUS, 0.1 + 0.05 * q)] for q in range(n)}
    jumps[0].append((SIGMA_Z, 0.07))
    return bonds, sites, jumps


def topologies(n):
    kets = [np.array([1, 0], dtype=complex)] * n
    return {"chain": pauli_product_mps(kets), "allphys": pauli_product_tree_allphys(kets),
            "virtnode": pauli_product_tree_virtnode(kets)}


def ttno_dense(ttno, site_ids):
    """The TTNO as a ``4^N x 4^N`` matrix in qubit order."""
    mat, order = ttno.as_matrix()
    dims = [int(round(np.sqrt(ttno.nodes[n].open_dimension()))) for n in order]
    tensor = mat.reshape(dims + dims)
    n = len(order)
    keep = [order.index(s) for s in site_ids]
    drop = [i for i in range(n) if i not in keep]
    tensor = tensor.transpose(keep + drop + [n + i for i in keep] + [n + i for i in drop])
    k = len(keep)
    return tensor.reshape(4 ** k, 4 ** k)


class PauliTestCase(TestCase):
    def setUp(self):
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        self.rng = np.random.default_rng(2024)


class TestTTNOAgainstDense(PauliTestCase):
    def test_both_generators_on_every_topology(self):
        for n in (2, 3, 4, 5):
            bonds, sites, jumps = random_model(self.rng, n)
            dense = pauli_dense_lindbladian(n, bonds, sites, jumps)
            for name, state in topologies(n).items():
                site_ids = pauli_site_ids(state)
                for generator, target in (("L", dense), ("H_L", 1j * dense)):
                    with self.subTest(n=n, topology=name, generator=generator):
                        ttno = pauli_lindbladian_ttno(state, bonds, sites, jumps,
                                                      generator=generator)
                        np.testing.assert_allclose(ttno_dense(ttno, site_ids), target,
                                                   atol=1e-12)

    def test_long_range_term_is_included(self):
        n = 4
        bonds = {(0, 3): np.kron(SIGMA_Z, SIGMA_Z)}
        dense = pauli_dense_lindbladian(n, bonds, {}, {})
        self.assertGreater(np.abs(dense).max(), 0.5)
        for name, state in topologies(n).items():
            ttno = pauli_lindbladian_ttno(state, bonds, {}, {})
            np.testing.assert_allclose(ttno_dense(ttno, pauli_site_ids(state)), dense,
                                       atol=1e-12, err_msg=name)

    def test_noise_only_and_hamiltonian_only_models(self):
        n = 3
        state = pauli_product_mps([np.array([1, 0], dtype=complex)] * n)
        site_ids = pauli_site_ids(state)
        _, sites, jumps = random_model(self.rng, n)
        for bonds_, sites_, jumps_ in (({}, sites, {}), ({}, {}, jumps)):
            dense = pauli_dense_lindbladian(n, bonds_, sites_, jumps_)
            ttno = pauli_lindbladian_ttno(state, bonds_, sites_, jumps_)
            np.testing.assert_allclose(ttno_dense(ttno, site_ids), dense, atol=1e-12)

    def test_h_l_is_i_times_l_tensorwise_in_dtype(self):
        state = pauli_product_mps([np.array([1, 0], dtype=complex)] * 3)
        bonds, sites, jumps = random_model(self.rng, 3)
        real = pauli_lindbladian_ttno(state, bonds, sites, jumps, generator="L")
        cplx = pauli_lindbladian_ttno(state, bonds, sites, jumps, generator="H_L")
        for tensor in real.tensors.values():
            self.assertEqual(np.float64, tensor.dtype)
        for tensor in cplx.tensors.values():
            self.assertEqual(np.complex128, tensor.dtype)
        site_ids = pauli_site_ids(state)
        np.testing.assert_allclose(ttno_dense(cplx, site_ids), 1j * ttno_dense(real, site_ids),
                                   atol=1e-13)

    def test_explicit_site_ids_match_the_default(self):
        n = 3
        state = pauli_product_mps([np.array([1, 0], dtype=complex)] * n)
        bonds, sites, jumps = random_model(self.rng, n)
        site_ids = pauli_site_ids(state)
        default = pauli_lindbladian_ttno(state, bonds, sites, jumps)
        explicit = pauli_lindbladian_ttno(state, bonds, sites, jumps, site_ids=site_ids)
        np.testing.assert_allclose(ttno_dense(default, site_ids), ttno_dense(explicit, site_ids),
                                   atol=1e-14)

    def test_input_validation(self):
        state = pauli_product_mps([np.array([1, 0], dtype=complex)] * 3)
        zz = np.kron(SIGMA_Z, SIGMA_Z)
        with self.assertRaisesRegex(ValueError, "generator"):
            pauli_lindbladian_ttno(state, {}, {}, {}, generator="bogus")
        with self.assertRaisesRegex(ValueError, "i < j"):
            pauli_lindbladian_ttno(state, {(1, 0): zz}, {}, {})
        with self.assertRaisesRegex(ValueError, "not Hermitian"):
            pauli_lindbladian_ttno(state, {(0, 1): 1j * zz}, {}, {})
        with self.assertRaisesRegex(ValueError, "not real"):
            pauli_lindbladian_ttno(state, {}, {0: SIGMA_MINUS}, {})


class TestDenseReference(PauliTestCase):
    def test_generator_is_real_and_trace_preserving(self):
        bonds, sites, jumps = random_model(self.rng, 3)
        dense = pauli_dense_lindbladian(3, bonds, sites, jumps)
        self.assertEqual(np.float64, dense.dtype)
        self.assertTrue(np.allclose(dense[0], 0, atol=1e-13))

    def test_unitary_part_rotates_by_the_commutator(self):
        n = 2
        h = random_hermitian(self.rng, 4)
        t = 0.8
        dense = pauli_dense_lindbladian(n, {(0, 1): h}, {}, {})
        rho = random_hermitian(self.rng, 4)
        u = expm(-1j * t * h)
        expected = pauli_rho_to_coeffs(u @ rho @ u.conj().T, n).real
        got = expm(t * dense) @ pauli_rho_to_coeffs(rho, n).real
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_single_site_term_rotates_the_bloch_vector(self):
        t = 0.6
        dense = pauli_dense_lindbladian(1, {}, {0: SIGMA_Z / 2}, {})
        plus = np.array([[1, 1], [1, 1]], dtype=complex) / 2
        out = pauli_coeffs_to_rho(expm(t * dense) @ pauli_rho_to_coeffs(plus, 1).real, 1)
        u = expm(-1j * t * SIGMA_Z / 2)
        np.testing.assert_allclose(out, u @ plus @ u.conj().T, atol=1e-12)

    def test_amplitude_damping_populations_decay_exponentially(self):
        gamma, t = 0.3, 1.7
        dense = pauli_dense_lindbladian(1, {}, {}, {0: [(SIGMA_MINUS, gamma)]})
        excited = np.diag([0.0, 1.0]).astype(complex)
        out = pauli_coeffs_to_rho(expm(t * dense) @ pauli_rho_to_coeffs(excited, 1).real, 1)
        self.assertAlmostEqual(np.exp(-gamma * t), out[1, 1].real, places=12)
        self.assertAlmostEqual(1.0, np.trace(out).real, places=12)

    def test_dephasing_kills_coherence_at_the_lindblad_rate(self):
        gamma, t = 0.25, 1.3
        dense = pauli_dense_lindbladian(1, {}, {}, {0: [(SIGMA_Z, gamma)]})
        plus = np.array([[1, 1], [1, 1]], dtype=complex) / 2
        out = pauli_coeffs_to_rho(expm(t * dense) @ pauli_rho_to_coeffs(plus, 1).real, 1)
        self.assertAlmostEqual(0.5 * np.exp(-2 * gamma * t), out[0, 1].real, places=12)

    def test_non_hermiticity_preserving_term_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not real"):
            pauli_dense_lindbladian(1, {}, {0: SIGMA_MINUS}, {})


if __name__ == "__main__":
    main()
