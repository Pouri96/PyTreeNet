import warnings
from copy import deepcopy
from unittest import TestCase, main

import numpy as np

from pytreenet.core.node import Node
from pytreenet.special_ttn.pauli import (
    PAULIS,
    network_is_real,
    pauli_channel_ptm,
    pauli_coeffs_to_rho,
    pauli_dense_lindbladian,
    pauli_gate_channels,
    pauli_gate_ptm,
    pauli_gauge,
    pauli_hermiticity_defect,
    pauli_lindbladian_ttno,
    pauli_local_expectation,
    pauli_physical_state,
    pauli_product_mps,
    pauli_product_tree_allphys,
    pauli_product_tree_virtnode,
    pauli_purity,
    pauli_reweight_channels,
    pauli_reweight_matrix,
    pauli_reweight_mpo,
    pauli_reweight_network,
    pauli_reweight_ptm,
    pauli_rho_to_coeffs,
    pauli_site_ids,
    pauli_to_coeffs,
    pauli_to_dense,
    pauli_trace,
    pauli_trace_cap,
)
from pytreenet.special_ttn.pauli.frame import GAUGE_ATTR
from pytreenet.ttns.ttns import TreeTensorNetworkState

SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)
SIGMA_Z = np.diag([1.0, -1.0]).astype(complex)
GAMMA = 2.2


def random_ket(rng):
    vec = rng.normal(size=2) + 1j * rng.normal(size=2)
    return vec / np.linalg.norm(vec)


def random_hermitian(rng, dim):
    mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    return (mat + mat.conj().T) / 2


def random_real_net(template, chi, rng):
    """A real network on the topology of ``template`` with every virtual bond set to ``chi``."""
    net = deepcopy(template)
    for node_id, node in net.nodes.items():
        open_dim = int(np.prod([net.tensors[node_id].shape[leg] for leg in node.open_legs]))
        shape = [chi] * node.nneighbours() + [open_dim]
        net.replace_tensor(node_id, rng.normal(size=shape), new_shape=True)
    return net


def topologies(kets):
    return {"chain": pauli_product_mps(kets),
            "allphys": pauli_product_tree_allphys(kets),
            "heap": pauli_product_tree_allphys(kets, order="heap"),
            "virtnode": pauli_product_tree_virtnode(kets)}


def weights(num_qubits, gamma):
    """The diagonal of ``G^(x)N``, ``gamma**-weight`` for every Pauli string."""
    diag = np.array([1.0, 1.0 / gamma, 1.0 / gamma, 1.0 / gamma])
    out = np.ones(1)
    for _ in range(num_qubits):
        out = np.kron(out, diag)
    return out


def ttno_dense(ttno, site_ids):
    """The TTNO as a ``4^N x 4^N`` matrix in qubit order."""
    mat, order = ttno.as_matrix()
    dims = [int(round(np.sqrt(ttno.nodes[n].open_dimension()))) for n in order]
    tensor = mat.reshape(dims + dims)
    n = len(order)
    keep = [order.index(s) for s in site_ids]
    drop = [i for i in range(n) if i not in keep]
    tensor = tensor.transpose(keep + drop + [n + i for i in keep] + [n + i for i in drop])
    return tensor.reshape(4 ** len(keep), 4 ** len(keep))


def raw_coeffs(net, site_ids):
    """The stored coefficients, bypassing the refusal, by reading a copy without the tag."""
    bare = deepcopy(net)
    if hasattr(bare, GAUGE_ATTR):
        delattr(bare, GAUGE_ATTR)
    return pauli_to_coeffs(bare, site_ids).real


def random_mpo(rng, num_sites, bonds):
    """Real random MPO site tensors with legs (w_l, w_r, out, in) and the given inner bonds."""
    dims = [1, *bonds, 1]
    return [rng.normal(size=(dims[k], dims[k + 1], 4, 4)) for k in range(num_sites)]


def mpo_to_matrix(tensors):
    """The ``4**k x 4**k`` matrix of an MPO, site 0 the most significant leg."""
    num = len(tensors)
    acc = tensors[0][0]  # (w_r, o_0, i_0)
    for site in tensors[1:]:
        acc = np.tensordot(acc, site, axes=([0], [0]))  # (o.., i.., w_r, o_new, i_new)
        acc = np.moveaxis(acc, acc.ndim - 3, 0)
    acc = acc[0]  # drop the closing bond
    outs = list(range(0, 2 * num, 2))
    ins = list(range(1, 2 * num, 2))
    return acc.transpose(outs + ins).reshape(4 ** num, 4 ** num)


class ReweightTestCase(TestCase):
    def setUp(self):
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        self.rng = np.random.default_rng(99)


class TestGaugeMatrix(ReweightTestCase):
    def test_matrix_and_inverse(self):
        mat = pauli_reweight_matrix(GAMMA)
        np.testing.assert_allclose(np.diag(mat), [1, 1 / GAMMA, 1 / GAMMA, 1 / GAMMA])
        np.testing.assert_allclose(mat @ pauli_reweight_matrix(GAMMA, inverse=True), np.eye(4))

    def test_identity_entry_is_one_so_the_trace_direction_is_untouched(self):
        for gamma in (1.0, 1.5, 3.0):
            self.assertEqual(1.0, pauli_reweight_matrix(gamma)[0, 0])

    def test_non_positive_gamma_is_refused(self):
        for gamma in (0.0, -1.0, float("nan")):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    pauli_reweight_matrix(gamma)


class TestStateGauge(ReweightTestCase):
    def test_coefficients_go_to_g_c_on_every_topology(self):
        for n in (2, 3, 4):
            kets = [random_ket(self.rng) for _ in range(n)]
            for name, template in topologies(kets).items():
                net = random_real_net(template, 3, self.rng)
                site_ids = pauli_site_ids(net)
                before = pauli_to_coeffs(net, site_ids).real
                pauli_reweight_network(net, GAMMA)
                with self.subTest(n=n, topology=name):
                    np.testing.assert_allclose(raw_coeffs(net, site_ids),
                                               weights(n, GAMMA) * before, atol=1e-12)

    def test_round_trip_restores_the_state_and_clears_the_tag(self):
        kets = [random_ket(self.rng) for _ in range(4)]
        for name, template in topologies(kets).items():
            net = random_real_net(template, 3, self.rng)
            ids = pauli_site_ids(net)
            before = pauli_to_coeffs(net, ids).real
            pauli_reweight_network(net, GAMMA)
            pauli_reweight_network(net, GAMMA, inverse=True)
            with self.subTest(topology=name):
                self.assertFalse(hasattr(net, GAUGE_ATTR))
                np.testing.assert_allclose(pauli_to_coeffs(net, ids).real, before, atol=1e-13)

    def test_gamma_one_changes_nothing_and_sets_no_tag(self):
        net = random_real_net(pauli_product_mps([random_ket(self.rng) for _ in range(3)]), 3,
                              self.rng)
        before = {k: v.copy() for k, v in net.tensors.items()}
        pauli_reweight_network(net, 1.0)
        self.assertFalse(hasattr(net, GAUGE_ATTR))
        for node_id, tensor in before.items():
            self.assertTrue(np.array_equal(tensor, net.tensors[node_id]))

    def test_gauges_compose_and_the_tag_records_the_product(self):
        net = random_real_net(pauli_product_mps([random_ket(self.rng) for _ in range(3)]), 3,
                              self.rng)
        ids = pauli_site_ids(net)
        before = pauli_to_coeffs(net, ids).real
        pauli_reweight_network(net, 1.5)
        pauli_reweight_network(net, 2.0)
        self.assertAlmostEqual(3.0, pauli_gauge(net), places=12)
        np.testing.assert_allclose(raw_coeffs(net, ids), weights(3, 3.0) * before, atol=1e-12)

    def test_physical_state_is_a_copy_in_physical_coordinates(self):
        net = random_real_net(pauli_product_mps([random_ket(self.rng) for _ in range(4)]), 3,
                              self.rng)
        ids = pauli_site_ids(net)
        before = pauli_to_coeffs(net, ids).real
        pauli_reweight_network(net, GAMMA)
        physical = pauli_physical_state(net)
        self.assertEqual(1.0, pauli_gauge(physical))
        self.assertEqual(GAMMA, pauli_gauge(net))
        np.testing.assert_allclose(pauli_to_coeffs(physical, ids).real, before, atol=1e-12)

    def test_physical_state_of_a_physical_state_is_an_equal_copy(self):
        net = random_real_net(pauli_product_mps([random_ket(self.rng) for _ in range(3)]), 3,
                              self.rng)
        copy = pauli_physical_state(net)
        self.assertIsNot(copy, net)
        np.testing.assert_allclose(pauli_to_coeffs(copy).real, pauli_to_coeffs(net).real)

    def test_the_gauge_keeps_a_real_network_real(self):
        net = random_real_net(pauli_product_mps([random_ket(self.rng) for _ in range(4)]), 3,
                              self.rng)
        pauli_reweight_network(net, GAMMA)
        self.assertTrue(network_is_real(net, tol=0.0))

    def test_deepcopy_keeps_the_tag(self):
        net = pauli_product_mps([random_ket(self.rng) for _ in range(3)])
        pauli_reweight_network(net, GAMMA)
        self.assertEqual(GAMMA, pauli_gauge(deepcopy(net)))

    def test_nodes_with_more_than_two_open_legs_are_refused(self):
        net = TreeTensorNetworkState()
        net.add_root(Node(identifier="a"), np.ones((4, 4, 4)))
        self.assertEqual(3, len(net.nodes["a"].open_legs))
        with self.assertRaisesRegex(ValueError, "open legs"):
            pauli_reweight_network(net, GAMMA)


class TestReadoutsInTheGauge(ReweightTestCase):
    def setUp(self):
        super().setUp()
        kets = [random_ket(self.rng) for _ in range(4)]
        self.physical = random_real_net(pauli_product_mps(kets), 3, self.rng)
        self.gauged = deepcopy(self.physical)
        pauli_reweight_network(self.gauged, GAMMA)

    def test_trace_is_gauge_invariant_including_through_the_cap(self):
        self.assertAlmostEqual(pauli_trace(self.physical).real, pauli_trace(self.gauged).real,
                               places=11)
        cap = pauli_trace_cap(self.gauged)
        self.assertEqual(1.0, pauli_gauge(cap))
        self.assertAlmostEqual(pauli_trace(self.physical).real,
                               pauli_trace(self.gauged, cap).real, places=11)

    def test_hermiticity_defect_and_bond_dimensions_are_gauge_invariant(self):
        self.assertEqual(0.0, pauli_hermiticity_defect(self.gauged))
        for node_id, node in self.physical.nodes.items():
            for neighbour in node.neighbouring_nodes():
                self.assertEqual(node.neighbour_dim(neighbour),
                                 self.gauged.nodes[node_id].neighbour_dim(neighbour))

    def test_gauge_sensitive_readouts_refuse_and_point_at_the_remedy(self):
        readouts = {"purity": lambda net: pauli_purity(net),
                    "coefficients": lambda net: pauli_to_coeffs(net),
                    "dense": lambda net: pauli_to_dense(net),
                    "expectation": lambda net: pauli_local_expectation(net, PAULIS[3], 1)}
        for name, readout in readouts.items():
            with self.subTest(readout=name):
                with self.assertRaisesRegex(ValueError, "physical_state"):
                    readout(self.gauged)
                readout(self.physical)

    def test_readouts_of_the_physical_copy_agree_with_the_ungauged_original(self):
        physical = pauli_physical_state(self.gauged)
        self.assertAlmostEqual(pauli_purity(self.physical), pauli_purity(physical), places=9)
        np.testing.assert_allclose(pauli_to_dense(self.physical), pauli_to_dense(physical),
                                   atol=1e-11)


class TestGaugedGenerator(ReweightTestCase):
    def model(self, n):
        bonds = {(q, q + 1): random_hermitian(self.rng, 4) for q in range(n - 1)}
        if n >= 3:
            bonds[(0, 2)] = random_hermitian(self.rng, 4)
        sites = {q: random_hermitian(self.rng, 2) for q in range(n)}
        jumps = {q: [(SIGMA_MINUS, 0.1 + 0.05 * q)] for q in range(n)}
        jumps[0].append((SIGMA_Z, 0.07))
        return bonds, sites, jumps

    def test_ttno_goes_to_g_l_g_inverse_on_every_topology(self):
        """Crossing the (out, in) legs would give G^-1 L G, which differs at O(1)."""
        for n in (2, 3, 4):
            model = self.model(n)
            dense = pauli_dense_lindbladian(n, *model)
            gauge = np.diag(weights(n, GAMMA))
            right = gauge @ dense @ np.linalg.inv(gauge)
            crossed = np.linalg.inv(gauge) @ dense @ gauge
            self.assertGreater(np.abs(right - crossed).max(), 0.1)
            kets = [random_ket(self.rng) for _ in range(n)]
            for name, state in topologies(kets).items():
                site_ids = pauli_site_ids(state)
                ttno = pauli_lindbladian_ttno(state, *model)
                pauli_reweight_network(ttno, GAMMA)
                with self.subTest(n=n, topology=name):
                    np.testing.assert_allclose(ttno_dense(ttno, site_ids), right, atol=1e-11)

    def test_gauged_pair_reproduces_the_physical_dynamics(self):
        n = 3
        model = self.model(n)
        dense = pauli_dense_lindbladian(n, *model)
        state = random_real_net(pauli_product_mps([random_ket(self.rng) for _ in range(n)]), 2,
                                self.rng)
        ids = pauli_site_ids(state)
        coeffs = pauli_to_coeffs(state, ids).real
        gauged_state = deepcopy(state)
        pauli_reweight_network(gauged_state, GAMMA)
        ttno = pauli_lindbladian_ttno(state, *model)
        pauli_reweight_network(ttno, GAMMA)
        lhs = ttno_dense(ttno, ids) @ raw_coeffs(gauged_state, ids)
        np.testing.assert_allclose(lhs, weights(n, GAMMA) * (dense @ coeffs), atol=1e-10)

    def test_gauging_the_generator_does_not_touch_its_tag_or_realness(self):
        n = 3
        state = pauli_product_mps([random_ket(self.rng) for _ in range(n)])
        ttno = pauli_lindbladian_ttno(state, *self.model(n))
        pauli_reweight_network(ttno, GAMMA)
        self.assertFalse(hasattr(ttno, GAUGE_ATTR))
        self.assertTrue(network_is_real(ttno, tol=0.0))

    def test_virtual_nodes_of_the_ttno_are_left_alone(self):
        n = 3
        state = pauli_product_tree_virtnode([random_ket(self.rng) for _ in range(n)])
        ttno = pauli_lindbladian_ttno(state, *self.model(n))
        virtual = set(ttno.nodes) - set(pauli_site_ids(state))
        self.assertTrue(virtual)
        before = {nid: ttno.tensors[nid].copy() for nid in virtual}
        pauli_reweight_network(ttno, GAMMA)
        for nid in virtual:
            self.assertTrue(np.array_equal(before[nid], ttno.tensors[nid]))


class TestChannelGauge(ReweightTestCase):
    def test_conjugated_ptm_acts_on_gauged_coefficients(self):
        for n in (1, 2):
            rho = random_hermitian(self.rng, 2 ** n)
            rho = rho @ rho.conj().T
            rho /= np.trace(rho).real
            coeffs = pauli_rho_to_coeffs(rho, n).real
            gate = np.linalg.qr(self.rng.normal(size=(2 ** n, 2 ** n))
                                + 1j * self.rng.normal(size=(2 ** n, 2 ** n)))[0]
            ptm = pauli_gate_ptm(gate, n)
            hat = pauli_reweight_ptm(ptm, n, GAMMA)
            expected = weights(n, GAMMA) * (ptm @ coeffs)
            np.testing.assert_allclose(hat @ (weights(n, GAMMA) * coeffs), expected, atol=1e-12)

    def test_conjugated_ptm_is_real_and_not_orthogonal(self):
        gate = np.linalg.qr(self.rng.normal(size=(4, 4)) + 1j * self.rng.normal(size=(4, 4)))[0]
        hat = pauli_reweight_ptm(pauli_gate_ptm(gate, 2), 2, GAMMA)
        self.assertTrue(np.isrealobj(hat))
        self.assertGreater(np.abs(hat @ hat.T - np.eye(16)).max(), 1e-3)

    def test_conjugated_channels_sum_to_the_conjugated_ptm_with_the_same_count(self):
        gate = np.linalg.qr(self.rng.normal(size=(4, 4)) + 1j * self.rng.normal(size=(4, 4)))[0]
        ptm = pauli_gate_ptm(gate, 2)
        channels = pauli_gate_channels(ptm)
        hat = pauli_reweight_channels(channels, GAMMA)
        self.assertEqual(len(channels), len(hat))
        total = sum(np.kron(a, b) for a, b in hat)
        np.testing.assert_allclose(total, pauli_reweight_ptm(ptm, 2, GAMMA), atol=1e-11)

    def test_a_kraus_channel_conjugates_the_same_way(self):
        gamma_damp = 0.3
        k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma_damp)]], dtype=complex)
        k1 = np.array([[0, np.sqrt(gamma_damp)], [0, 0]], dtype=complex)
        ptm = pauli_channel_ptm([k0, k1], 1)
        rho = np.array([[0.4, 0.2 - 0.1j], [0.2 + 0.1j, 0.6]])
        coeffs = pauli_rho_to_coeffs(rho, 1).real
        hat = pauli_reweight_ptm(ptm, 1, GAMMA)
        np.testing.assert_allclose(hat @ (weights(1, GAMMA) * coeffs),
                                   weights(1, GAMMA) * (ptm @ coeffs), atol=1e-12)
        back = pauli_coeffs_to_rho(ptm @ coeffs, 1)
        self.assertAlmostEqual(1.0, np.trace(back).real, places=12)


class TestMPOGauge(ReweightTestCase):
    def test_the_contracted_mpo_is_the_conjugated_ptm(self):
        for num in (1, 2, 3):
            tensors = random_mpo(self.rng, num, [3, 5][: num - 1])
            hat = pauli_reweight_mpo(tensors, GAMMA)
            with self.subTest(sites=num):
                np.testing.assert_allclose(
                    mpo_to_matrix(hat),
                    pauli_reweight_ptm(mpo_to_matrix(tensors), num, GAMMA), atol=1e-12)

    def test_crossing_out_and_in_would_be_caught(self):
        tensors = random_mpo(self.rng, 2, [4])
        crossed = [np.swapaxes(w, 2, 3) for w in pauli_reweight_mpo(
            [np.swapaxes(w, 2, 3) for w in tensors], GAMMA)]
        wrong = mpo_to_matrix(crossed)
        right = pauli_reweight_ptm(mpo_to_matrix(tensors), 2, GAMMA)
        self.assertGreater(np.abs(wrong - right).max(), 1e-2)

    def test_gamma_one_is_bit_identical_and_the_inputs_are_not_modified(self):
        tensors = random_mpo(self.rng, 3, [3, 4])
        before = [w.copy() for w in tensors]
        hat = pauli_reweight_mpo(tensors, 1.0)
        for got, want, orig in zip(hat, before, tensors):
            np.testing.assert_array_equal(got, want)
            np.testing.assert_array_equal(orig, want)

    def test_bonds_and_dtype_are_unchanged(self):
        tensors = random_mpo(self.rng, 3, [3, 4])
        hat = pauli_reweight_mpo(tensors, GAMMA)
        self.assertEqual([w.shape for w in hat], [w.shape for w in tensors])
        self.assertTrue(all(w.dtype == np.float64 for w in hat))

    def test_the_gauge_round_trips(self):
        tensors = random_mpo(self.rng, 2, [4])
        back = pauli_reweight_mpo(pauli_reweight_mpo(tensors, GAMMA), 1.0 / GAMMA)
        for got, want in zip(back, tensors):
            np.testing.assert_allclose(got, want, atol=1e-13)

    def test_a_tensor_of_the_wrong_shape_is_refused(self):
        with self.assertRaisesRegex(ValueError, "legs"):
            pauli_reweight_mpo([np.zeros((1, 1, 2, 2))], GAMMA)
        with self.assertRaisesRegex(ValueError, "legs"):
            pauli_reweight_mpo([np.zeros((4, 4))], GAMMA)

    def test_a_non_positive_gamma_is_refused(self):
        with self.assertRaises(ValueError):
            pauli_reweight_mpo(random_mpo(self.rng, 1, []), 0.0)


if __name__ == "__main__":
    main()
