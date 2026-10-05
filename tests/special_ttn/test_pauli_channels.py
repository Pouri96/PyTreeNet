import warnings
from unittest import TestCase, main

import numpy as np
from scipy.linalg import expm

from pytreenet.special_ttn.pauli import (
    dissipator_generator, pauli_channel_ptm, pauli_coeffs_to_rho, pauli_dense_lindbladian,
    pauli_dissipator_channel_ptm, pauli_gate_channels, pauli_gate_ptm, pauli_rho_to_coeffs)
from pytreenet.special_ttn.pauli.channels import _ptm_multi

SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)
SIGMA_Z = np.diag([1.0, -1.0]).astype(complex)
T_GATE = np.diag([1, np.exp(0.25j * np.pi)])
S_GATE = np.diag([1, 1j])
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def haar_unitary(rng, dim):
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(mat)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_hermitian(rng, dim):
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (mat + mat.conj().T) / 2


class PauliTestCase(TestCase):
    def setUp(self):
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        self.rng = np.random.default_rng(99)


class TestGatePTM(PauliTestCase):
    def check_gate(self, gate):
        k = int(np.log2(gate.shape[0]))
        ptm = pauli_gate_ptm(gate)
        self.assertEqual((4 ** k, 4 ** k), ptm.shape)
        self.assertEqual(np.float64, ptm.dtype)
        rho = random_hermitian(self.rng, 2 ** k)
        expected = pauli_rho_to_coeffs(gate @ rho @ gate.conj().T, k)
        self.assertTrue(np.allclose(ptm @ pauli_rho_to_coeffs(rho, k).real, expected.real,
                                    atol=1e-12))

    def test_haar_gates_on_one_two_and_three_qubits(self):
        for k in (1, 2, 3):
            with self.subTest(qubits=k):
                self.check_gate(haar_unitary(self.rng, 2 ** k))

    def test_complex_clifford_and_t_gates_have_real_ptms(self):
        for gate in (T_GATE, S_GATE, HADAMARD, CNOT):
            with self.subTest(gate=gate.shape):
                self.check_gate(gate)

    def test_gate_ptm_is_orthogonal(self):
        ptm = pauli_gate_ptm(haar_unitary(self.rng, 4))
        self.assertTrue(np.allclose(ptm @ ptm.T, np.eye(16), atol=1e-12))

    def test_gate_ptm_fixes_the_identity_row(self):
        ptm = pauli_gate_ptm(haar_unitary(self.rng, 4))
        self.assertTrue(np.allclose(ptm[0], np.eye(16)[0], atol=1e-12))

    def test_kron_order_first_qubit_is_leftmost(self):
        gate = np.kron(HADAMARD, np.eye(2))
        rho = np.kron(np.diag([1.0, 0.0]), np.diag([1.0, 0.0])).astype(complex)
        out = pauli_coeffs_to_rho(pauli_gate_ptm(gate) @ pauli_rho_to_coeffs(rho, 2).real, 2)
        self.assertTrue(np.allclose(out, gate @ rho @ gate.conj().T, atol=1e-12))

    def test_bad_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            pauli_gate_ptm(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            pauli_gate_ptm(np.eye(4), num_qubits=3)

    def test_one_sided_map_has_no_real_ptm(self):
        u = haar_unitary(self.rng, 2)
        with self.assertRaisesRegex(ValueError, "not Hermiticity-preserving"):
            _ptm_multi(np.kron(u, np.eye(2)), 1, "one-sided")


class TestChannelPTM(PauliTestCase):
    @staticmethod
    def amplitude_damping(p):
        return [np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex),
                np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)]

    def test_single_unitary_kraus_equals_gate_ptm(self):
        u = haar_unitary(self.rng, 4)
        self.assertTrue(np.allclose(pauli_channel_ptm([u]), pauli_gate_ptm(u), atol=1e-12))

    def test_amplitude_damping_matches_dense(self):
        ops = self.amplitude_damping(0.3)
        rho = random_hermitian(self.rng, 2)
        expected = sum(k @ rho @ k.conj().T for k in ops)
        got = pauli_channel_ptm(ops) @ pauli_rho_to_coeffs(rho, 1).real
        self.assertTrue(np.allclose(got, pauli_rho_to_coeffs(expected, 1).real, atol=1e-12))

    def test_two_qubit_correlated_channel_matches_dense(self):
        ops = [np.sqrt(0.6) * np.eye(4, dtype=complex), np.sqrt(0.4) * CNOT]
        rho = random_hermitian(self.rng, 4)
        expected = sum(k @ rho @ k.conj().T for k in ops)
        got = pauli_channel_ptm(ops) @ pauli_rho_to_coeffs(rho, 2).real
        self.assertTrue(np.allclose(got, pauli_rho_to_coeffs(expected, 2).real, atol=1e-12))

    def test_complete_set_is_trace_preserving_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ptm = pauli_channel_ptm(self.amplitude_damping(0.2))
        self.assertTrue(np.allclose(ptm[0], np.eye(4)[0], atol=1e-12))

    def test_incomplete_kraus_set_warns(self):
        with self.assertWarnsRegex(RuntimeWarning, "not trace-preserving"):
            pauli_channel_ptm(self.amplitude_damping(0.2)[:1])

    def test_bad_input_is_rejected(self):
        with self.assertRaises(ValueError):
            pauli_channel_ptm([])
        with self.assertRaises(ValueError):
            pauli_channel_ptm([np.eye(2), np.eye(4)])
        with self.assertRaises(ValueError):
            pauli_channel_ptm([np.eye(3)])


class TestDissipator(PauliTestCase):
    JUMPS = [(SIGMA_MINUS, 0.13), (SIGMA_Z, 0.05)]

    def test_generator_is_trace_preserving(self):
        gen = dissipator_generator(self.JUMPS)
        trace_functional = np.array([1, 0, 0, 1])
        self.assertTrue(np.allclose(trace_functional @ gen, 0, atol=1e-14))

    def test_generator_rejects_non_two_by_two(self):
        with self.assertRaises(ValueError):
            dissipator_generator([(np.eye(3), 1.0)])

    def test_channel_ptm_is_the_matrix_exponential_of_the_dense_generator(self):
        dt = 0.37
        dense = pauli_dense_lindbladian(1, {}, {}, {0: self.JUMPS})
        ptm = pauli_dissipator_channel_ptm(self.JUMPS, dt)
        self.assertEqual((4, 4), ptm.shape)
        self.assertTrue(np.allclose(ptm, expm(dt * dense), atol=1e-12))

    def test_channel_ptm_is_trace_preserving(self):
        ptm = pauli_dissipator_channel_ptm(self.JUMPS, 0.5)
        self.assertTrue(np.allclose(ptm[0], np.eye(4)[0], atol=1e-12))

    def test_amplitude_damping_decay_law(self):
        gamma, dt = 0.4, 0.9
        excited = np.diag([0.0, 1.0]).astype(complex)
        ptm = pauli_dissipator_channel_ptm([(SIGMA_MINUS, gamma)], dt)
        out = pauli_coeffs_to_rho(ptm @ pauli_rho_to_coeffs(excited, 1).real, 1)
        self.assertAlmostEqual(np.exp(-gamma * dt), out[1, 1].real, places=12)


class TestGateChannels(PauliTestCase):
    @staticmethod
    def reconstruct(channels):
        return sum(np.kron(a, b) for a, b in channels)

    def test_reconstruction_and_channel_counts(self):
        cases = {"generic": (haar_unitary(self.rng, 4), 16), "cnot": (CNOT, 4),
                 "product": (np.kron(HADAMARD, T_GATE), 1)}
        for name, (gate, count) in cases.items():
            with self.subTest(gate=name):
                ptm = pauli_gate_ptm(gate)
                channels = pauli_gate_channels(ptm)
                self.assertEqual(count, len(channels))
                self.assertTrue(np.allclose(self.reconstruct(channels), ptm, atol=1e-12))
                for a, b in channels:
                    self.assertEqual(np.float64, a.dtype)
                    self.assertEqual((4, 4), a.shape)
                    self.assertEqual((4, 4), b.shape)

    def test_tolerance_drops_weak_channels(self):
        channels = pauli_gate_channels(pauli_gate_ptm(CNOT), tol=1e-12)
        self.assertEqual(4, len(channels))
        self.assertEqual(16, len(pauli_gate_channels(pauli_gate_ptm(haar_unitary(self.rng, 4)))))
        self.assertLess(len(pauli_gate_channels(pauli_gate_ptm(haar_unitary(self.rng, 4)),
                                                tol=0.5)), 16)

    def test_bad_ptm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "16x16"):
            pauli_gate_channels(np.eye(4))
        with self.assertRaisesRegex(ValueError, "not real"):
            pauli_gate_channels(np.eye(16) * (1 + 1j))


if __name__ == "__main__":
    main()
