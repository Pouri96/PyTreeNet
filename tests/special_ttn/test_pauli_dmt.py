import warnings
from copy import deepcopy
from unittest import TestCase, main

import numpy as np

from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.special_ttn.pauli import (
    network_is_real, pauli_hermiticity_defect, pauli_product_mps, pauli_product_tree_virtnode,
    pauli_reweight_network, pauli_rho_to_coeffs, pauli_site_ids, pauli_to_coeffs, pauli_trace)
from pytreenet.special_ttn.pauli.dmt import (
    _ordered_orthonormal_basis, _radius_levels, _respect_degenerate, _tol_rank,
    dmt_min_chi_for_radius, orient_observable, pauli_dmt_truncate, pauli_operator_tensors)

SIGMA_Z = np.diag([1.0, -1.0]).astype(complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
N = 8


def random_real_chain(n, chi, rng):
    kets = [np.array([1, 0], dtype=complex)] * n
    net = pauli_product_mps(kets)
    for node_id, node in net.nodes.items():
        net.replace_tensor(node_id, rng.normal(size=[chi] * node.nneighbours() + [4]),
                           new_shape=True)
    return net


def bonds(net, ids):
    return [net.nodes[ids[k]].neighbour_dim(ids[k + 1]) for k in range(len(ids) - 1)]


def marginal_change(before, after, n, length):
    """Largest relative change of a contiguous ``length``-site marginal. The marginal on
    ``[a, a + length)`` is the coefficient slice with the identity on every other qubit."""
    worst = 0.0
    for a in range(n - length + 1):
        idx = tuple([0] * a + [slice(None)] * length + [0] * (n - a - length))
        worst = max(worst, np.linalg.norm(after[idx] - before[idx]) / np.linalg.norm(before[idx]))
    return worst


class DMTTestCase(TestCase):
    def setUp(self):
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        self.rng = np.random.default_rng(7)

    def coeffs(self, net, ids):
        return pauli_to_coeffs(net, ids).real.reshape((4,) * len(ids))


class TestGuarantee(DMTTestCase):
    """Every operator on at most 2R+1 contiguous qubits survives, and the next shell does not.
    The second half matters: a fixture that truncated nothing would pass the first vacuously."""

    def run_radius(self, radius, chi=12):
        net = random_real_chain(N, 16, self.rng)
        ids = pauli_site_ids(net)
        out, report = pauli_dmt_truncate(net, chi, radius=radius)
        return ids, self.coeffs(net, ids), self.coeffs(out, ids), out, report

    def test_radius_zero_protects_single_site_marginals_only(self):
        ids, before, after, out, _ = self.run_radius(0)
        self.assertLess(marginal_change(before, after, N, 1), 1e-12)
        self.assertGreater(marginal_change(before, after, N, 2), 1e-2)
        self.assertEqual(12, max(bonds(out, ids)))

    def test_radius_one_protects_up_to_three_sites(self):
        ids, before, after, out, _ = self.run_radius(1)
        for length in (1, 2, 3):
            self.assertLess(marginal_change(before, after, N, length), 1e-12, msg=length)
        self.assertGreater(marginal_change(before, after, N, 4), 1e-2)
        self.assertEqual(12, max(bonds(out, ids)))

    def test_radius_two_protects_up_to_five_sites(self):
        net = random_real_chain(9, 60, self.rng)
        ids = pauli_site_ids(net)
        before = self.coeffs(net, ids)
        out, _ = pauli_dmt_truncate(net, 40, radius=2)
        after = self.coeffs(out, ids)
        for length in (1, 2, 3, 4, 5):
            self.assertLess(marginal_change(before, after, 9, length), 1e-11, msg=length)
        self.assertGreater(marginal_change(before, after, 9, 6), 1e-2)

    def test_the_trace_is_exact_without_any_pin(self):
        net = random_real_chain(N, 16, self.rng)
        out, _ = pauli_dmt_truncate(net, 12)
        np.testing.assert_allclose(pauli_trace(out).real, pauli_trace(net).real, rtol=1e-12)

    def test_plain_svd_breaks_what_dmt_keeps(self):
        """The contrast that shows the truncation is real: a plain SVD at the same budget changes
        single-site marginals, and DMT does not."""
        net = random_real_chain(N, 16, self.rng)
        ids = pauli_site_ids(net)
        before = self.coeffs(net, ids)
        plain = deepcopy(net)
        plain.canonical_form(ids[-1], preserve_legs_order=True)
        from pytreenet.core.canonical_form import split_svd_contract_sv_to_neighbour
        from pytreenet.util.tensor_splitting import SVDParameters
        params = SVDParameters(max_bond_dim=12, rel_tol=1e-12, total_tol=1e-12)
        for k in range(N - 1, 0, -1):
            split_svd_contract_sv_to_neighbour(plain, ids[k], ids[k - 1], params,
                                               preserve_legs_order=True)
            plain.orthogonality_center_id = ids[k - 1]
        self.assertGreater(marginal_change(before, self.coeffs(plain, ids), N, 1), 1e-3)

    def test_the_result_is_real_and_hermitian_exactly(self):
        net = random_real_chain(N, 16, self.rng)
        out, _ = pauli_dmt_truncate(net, 12)
        self.assertTrue(network_is_real(out, tol=0.0))
        self.assertEqual(0.0, pauli_hermiticity_defect(out))


class TestRankLimitedBonds(DMTTestCase):
    def padded_chain(self, n, true_bond, padded_bond):
        """A chain whose true bond is small, zero-padded to a larger one: the rank-limited
        scaffolding that an augmented sweep produces."""
        small = random_real_chain(n, true_bond, self.rng)
        ids = pauli_site_ids(small)
        net = random_real_chain(n, padded_bond, self.rng)
        for k, sid in enumerate(ids):
            src = small.tensors[sid]
            dst = np.zeros(net.tensors[sid].shape)
            slices = tuple(slice(0, s) for s in src.shape)
            dst[slices] = src
            net.replace_tensor(sid, dst, new_shape=False)
        return net, ids

    def test_a_lossless_pass_over_a_rank_limited_chain_stays_lossless(self):
        """Without re-projecting the complement off the reserved span, the arbitrary completions
        that the SVD returns for the null space corrupt the projector. A random full-rank chain
        never shows it."""
        net, ids = self.padded_chain(N, true_bond=6, padded_bond=20)
        before = self.coeffs(net, ids)
        out, report = pauli_dmt_truncate(net, 12)
        self.assertLess(np.linalg.norm(self.coeffs(out, ids) - before) / np.linalg.norm(before),
                        1e-12)
        self.assertLess(report.discarded, 1e-12)
        self.assertLessEqual(max(bonds(out, ids)), 12)


class TestZeroToleranceComplement(TestRankLimitedBonds):
    """With no tolerance the complement is cut at the residual's numerical noise, so it contains
    arbitrary null-space completions. They must be projected off the reserved span.

    Only that projection is observable here. Dropping collapsed columns and re-orthonormalising
    keep the projector an isometry by construction, and no fixture tried makes them matter beyond
    rounding noise, because the null-space directions carry no weight."""

    def test_noise_directions_do_not_corrupt_a_lossless_pass(self):
        net, ids = self.padded_chain(N, true_bond=6, padded_bond=20)
        before = self.coeffs(net, ids)
        out, _ = pauli_dmt_truncate(net, 14, rel_tol=0.0, total_tol=0.0)
        self.assertLess(np.linalg.norm(self.coeffs(out, ids) - before) / np.linalg.norm(before),
                        1e-11)
        self.assertLessEqual(max(bonds(out, ids)), 14)

    def test_the_marginals_survive_the_zero_tolerance_cut(self):
        net, ids = self.padded_chain(N, true_bond=6, padded_bond=20)
        before = self.coeffs(net, ids)
        out, _ = pauli_dmt_truncate(net, 12, rel_tol=0.0, total_tol=0.0)
        for length in (1, 2, 3):
            self.assertLess(marginal_change(before, self.coeffs(out, ids), N, length), 1e-11)


class TestDegenerateWiring(DMTTestCase):
    def test_the_degeneracy_rule_is_applied_only_when_asked(self):
        from unittest import mock

        import pytreenet.special_ttn.pauli.dmt as dmt_module
        net = random_real_chain(N, 16, self.rng)
        with mock.patch.object(dmt_module, "_respect_degenerate",
                               wraps=dmt_module._respect_degenerate) as spy:
            pauli_dmt_truncate(net, 12, respect_degenerate=True)
            self.assertGreater(spy.call_count, 0)
            spy.reset_mock()
            pauli_dmt_truncate(net, 12, respect_degenerate=False)
            self.assertEqual(0, spy.call_count)


class TestLedgerAndReport(DMTTestCase):
    def test_a_lossless_pass_reports_exactly_zero(self):
        net = random_real_chain(N, 8, self.rng)
        _, report = pauli_dmt_truncate(net, 64)
        self.assertEqual(0.0, report.discarded)
        self.assertEqual({}, report.reserved_ranks)

    def test_the_ledger_bounds_the_relative_error(self):
        net = random_real_chain(N, 16, self.rng)
        ids = pauli_site_ids(net)
        before = self.coeffs(net, ids)
        out, report = pauli_dmt_truncate(net, 12)
        error = np.linalg.norm(self.coeffs(out, ids) - before) / np.linalg.norm(before)
        self.assertGreater(report.discarded, 0.0)
        self.assertLessEqual(error, report.discarded * (1 + 1e-9))

    def test_the_per_bond_weights_are_consistent_with_the_ledger_and_the_error(self):
        """``discarded_weights`` feeds the fidelity ledger of an integrator, so each weight must be
        the relative squared weight of one cut and the ledger their square roots summed."""
        net = random_real_chain(N, 16, self.rng)
        ids = pauli_site_ids(net)
        before = self.coeffs(net, ids)
        out, report = pauli_dmt_truncate(net, 12)
        weights = report.discarded_weights
        self.assertGreater(len(weights), 0)
        self.assertTrue(all(0.0 < w < 1.0 for w in weights))
        self.assertAlmostEqual(report.discarded, float(np.sum(np.sqrt(weights))), places=12)
        error = np.linalg.norm(self.coeffs(out, ids) - before) / np.linalg.norm(before)
        self.assertLessEqual(error, np.sqrt(np.sum(weights)) * (1 + 1e-9) + 1e-12)
        self.assertLessEqual(len(weights), len(report.reserved_ranks))

    def test_no_weights_are_reported_for_a_lossless_pass(self):
        _, report = pauli_dmt_truncate(random_real_chain(N, 8, self.rng), 64)
        self.assertEqual([], report.discarded_weights)

    def test_reserved_ranks_never_exceed_the_budget(self):
        net = random_real_chain(N, 16, self.rng)
        _, report = pauli_dmt_truncate(net, 12)
        self.assertTrue(report.reserved_ranks)
        self.assertLessEqual(max(report.reserved_ranks.values()), 12)

    def test_the_floor_helper(self):
        self.assertEqual([4, 10, 34], [dmt_min_chi_for_radius(r) for r in (0, 1, 2)])


class TestBudgetClipping(DMTTestCase):
    def test_a_budget_below_the_reservation_warns_and_still_respects_the_cap(self):
        net = random_real_chain(N, 16, self.rng)
        ids = pauli_site_ids(net)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, report = pauli_dmt_truncate(net, 6, radius=1)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning)
                            and "does not hold" in str(w.message) for w in caught))
        self.assertLessEqual(max(bonds(out, ids)), 6)
        self.assertTrue(report.clipped_bonds)

    def test_a_clipped_reservation_keeps_the_trace_first(self):
        """The trace is the highest-priority direction and must survive any clipping."""
        net = random_real_chain(N, 16, self.rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, _ = pauli_dmt_truncate(net, 3, radius=1)
        self.assertAlmostEqual(pauli_trace(net).real, pauli_trace(out).real, places=10)

    def test_no_warning_when_the_reservation_fits(self):
        net = random_real_chain(N, 16, self.rng)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pauli_dmt_truncate(net, dmt_min_chi_for_radius(1))


class TestWindows(DMTTestCase):
    def centred(self, net, ids, site):
        net = deepcopy(net)
        net.canonical_form(ids[site], preserve_legs_order=True)
        return net

    def test_only_the_bonds_of_the_window_are_truncated(self):
        net = random_real_chain(N, 16, self.rng)
        ids = pauli_site_ids(net)
        net = self.centred(net, ids, 5)
        before = bonds(net, ids)
        out, _ = pauli_dmt_truncate(net, 12, window=(2, 5))
        after = bonds(out, ids)
        for k in range(N - 1):
            if 2 <= k < 5:
                self.assertLessEqual(after[k], 12, msg=k)
            else:
                self.assertEqual(before[k], after[k], msg=k)
        self.assertEqual(ids[2], out.orthogonality_center_id)

    def test_a_window_preserves_the_global_marginals(self):
        net = random_real_chain(N, 16, self.rng)
        ids = pauli_site_ids(net)
        net = self.centred(net, ids, 5)
        before = self.coeffs(net, ids)
        out, _ = pauli_dmt_truncate(net, 12, window=(2, 5))
        for length in (1, 2, 3):
            self.assertLess(marginal_change(before, self.coeffs(out, ids), N, length), 1e-12)

    def test_the_full_window_of_a_canonical_chain_equals_the_gauge_robust_call(self):
        net = random_real_chain(N, 16, self.rng)
        ids = pauli_site_ids(net)
        net = self.centred(net, ids, N - 1)
        windowed, _ = pauli_dmt_truncate(net, 12, window=(0, N - 1))
        whole, _ = pauli_dmt_truncate(net, 12)
        np.testing.assert_allclose(self.coeffs(windowed, ids), self.coeffs(whole, ids),
                                   atol=1e-10)

    def test_in_place_returns_the_same_object(self):
        net = random_real_chain(N, 16, self.rng)
        out, _ = pauli_dmt_truncate(net, 12, in_place=True)
        self.assertIs(net, out)
        copy_net = random_real_chain(N, 16, self.rng)
        ids = pauli_site_ids(copy_net)
        before = bonds(copy_net, ids)
        out, _ = pauli_dmt_truncate(copy_net, 12)
        self.assertIsNot(copy_net, out)
        self.assertEqual(before, bonds(copy_net, ids))


class TestChainLayouts(DMTTestCase):
    def test_a_chain_rooted_in_the_middle_gives_the_same_result(self):
        n = 6
        tensors = [self.rng.normal(size=s) for s in
                   [(14, 4), (14, 14, 4), (14, 14, 4), (14, 14, 4), (14, 14, 4), (14, 4)]]
        end = MatrixProductState.from_tensor_list(tensors, node_prefix="qubit", root_site=0)
        middle = MatrixProductState.from_tensor_list(tensors, node_prefix="qubit", root_site=3)
        ids = [f"qubit{q}" for q in range(n)]
        out_end, _ = pauli_dmt_truncate(end, 10, site_ids=ids)
        out_mid, _ = pauli_dmt_truncate(middle, 10, site_ids=ids)
        np.testing.assert_allclose(pauli_to_coeffs(out_end, ids), pauli_to_coeffs(out_mid, ids),
                                   atol=1e-10)
        self.assertEqual(10, max(bonds(out_mid, ids)))

    def test_sites_that_are_not_a_chain_are_refused(self):
        net = random_real_chain(5, 4, self.rng)
        ids = pauli_site_ids(net)
        with self.assertRaisesRegex(ValueError, "not a chain"):
            pauli_dmt_truncate(net, 3, site_ids=[ids[0], ids[2], ids[1], ids[3], ids[4]])

    def test_virtual_nodes_are_refused(self):
        kets = [np.array([1, 0], dtype=complex)] * 4
        with self.assertRaisesRegex(ValueError, "virtual nodes"):
            pauli_dmt_truncate(pauli_product_tree_virtnode(kets), 4)

    def test_a_negative_radius_is_refused(self):
        with self.assertRaisesRegex(ValueError, "radius"):
            pauli_dmt_truncate(random_real_chain(4, 4, self.rng), 4, radius=-1)

    def test_a_reweighted_state_is_refused(self):
        net = random_real_chain(4, 4, self.rng)
        pauli_reweight_network(net, 2.0)
        with self.assertRaisesRegex(ValueError, "reweighted"):
            pauli_dmt_truncate(net, 3)


class TestObservableReservation(DMTTestCase):
    def dense_observable(self, n, bond_terms, site_terms):
        dim = 2 ** n
        out = np.zeros((dim, dim), dtype=complex)

        def embed(op, sites):
            rest = [q for q in range(n) if q not in sites]
            order = list(sites) + rest
            full = np.kron(op, np.eye(2 ** (n - len(sites)))).reshape((2,) * (2 * n))
            inv = np.argsort(order)
            return full.transpose(list(inv) + [n + p for p in inv]).reshape(dim, dim)

        for q, h in site_terms.items():
            out += embed(h, [q])
        for (i, j), h in bond_terms.items():
            out += embed(h, [i, j])
        return out

    def model(self):
        zz = np.kron(SIGMA_Z, SIGMA_Z)
        xx = np.kron(SIGMA_X, SIGMA_X)
        bonds_ = {(q, q + 1): zz + 0.4 * xx for q in range(N - 1)}
        bonds_[(1, 4)] = 0.7 * zz
        sites = {q: 0.5 * SIGMA_X + 0.2 * SIGMA_Z for q in range(N)}
        return bonds_, sites

    @staticmethod
    def flat(tensors):
        """The flat ``(4^N,)`` coefficient vector of a chain of ``(left, right, 4)`` tensors,
        qubit 0 most significant."""
        vec = tensors[0][0].T                                  # (4, r)
        for t in tensors[1:]:
            vec = np.einsum("xa,arp->xpr", vec, t).reshape(-1, t.shape[1])
        return vec[:, 0]

    def test_operator_tensors_match_the_dense_observable(self):
        n = 5
        bonds_ = {(0, 1): np.kron(SIGMA_Z, SIGMA_Z), (1, 3): 0.6 * np.kron(SIGMA_X, SIGMA_X)}
        sites = {2: 0.3 * SIGMA_Z, 4: SIGMA_X}
        expected = pauli_rho_to_coeffs(self.dense_observable(n, bonds_, sites), n)
        self.assertLess(np.abs(expected.imag).max(), 1e-12)
        got = self.flat(pauli_operator_tensors(n, bonds_, sites))
        np.testing.assert_allclose(got, expected.real, atol=1e-12)

    def test_the_trace_cap_is_the_observable_of_the_identity(self):
        tensors = pauli_operator_tensors(3)
        for t in tensors:
            np.testing.assert_allclose(t.reshape(4), np.sqrt(2) * np.array([1, 0, 0, 0]))

    def test_a_term_on_a_qubit_outside_the_chain_is_refused(self):
        """It would otherwise be dropped silently and the observable would become the identity."""
        for kwargs in ({"site_terms": {5: SIGMA_Z}}, {"site_terms": {-1: SIGMA_Z}},
                       {"bond_terms": {(1, 4): np.kron(SIGMA_Z, SIGMA_Z)}}):
            with self.subTest(**{k: list(v) for k, v in kwargs.items()}):
                with self.assertRaisesRegex(ValueError, "outside"):
                    pauli_operator_tensors(3, **kwargs)

    def test_non_hermitian_and_unordered_terms_are_refused(self):
        with self.assertRaises(ValueError):
            pauli_operator_tensors(4, bond_terms={(2, 1): np.eye(4)})
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            pauli_operator_tensors(4, bond_terms={(0, 1): 1j * np.kron(SIGMA_Z, SIGMA_Z)})
        with self.assertRaisesRegex(ValueError, "site term on qubit 1 is not Hermitian"):
            pauli_operator_tensors(4, site_terms={1: np.array([[0, 1], [0, 0]])})

    def test_the_observable_bond_is_its_operator_schmidt_rank(self):
        bonds_, sites = self.model()
        tensors = pauli_operator_tensors(N, bonds_, sites)
        self.assertLessEqual(max(t.shape[1] for t in tensors[:-1]), 8)

    def expectation(self, net, ids, bond_terms, site_terms):
        rho_h = pauli_rho_to_coeffs(self.dense_observable(len(ids), bond_terms, site_terms),
                                    len(ids)).real
        return float(rho_h @ pauli_to_coeffs(net, ids).real)

    def test_reserving_the_observable_makes_it_exact_beyond_the_radius(self):
        """H has a range-3 coupling, which radius 1 cannot reach."""
        bonds_, sites = self.model()
        net = random_real_chain(N, 40, self.rng)
        ids = pauli_site_ids(net)
        exact = self.expectation(net, ids, bonds_, sites)
        obs = [pauli_operator_tensors(N, bonds_, sites)]
        reserved, report = pauli_dmt_truncate(net, 24, observables=obs)
        radius_only, _ = pauli_dmt_truncate(net, 24)
        scale = abs(exact) + 1.0
        self.assertLess(abs(self.expectation(reserved, ids, bonds_, sites) - exact) / scale, 1e-11)
        self.assertGreater(abs(self.expectation(radius_only, ids, bonds_, sites) - exact) / scale,
                           1e-4)
        self.assertLessEqual(max(report.reserved_ranks.values()), 24)

    def test_an_observable_of_the_wrong_length_is_refused(self):
        net = random_real_chain(5, 4, self.rng)
        with self.assertRaisesRegex(ValueError, "sites"):
            pauli_dmt_truncate(net, 3, observables=[pauli_operator_tensors(4)])

    def test_a_complex_observable_on_a_real_chain_is_refused(self):
        net = random_real_chain(4, 4, self.rng)
        tensors = [t.astype(complex) for t in pauli_operator_tensors(4)]
        tensors[1] = tensors[1] * 1j
        with self.assertRaisesRegex(ValueError, "imaginary"):
            pauli_dmt_truncate(net, 3, observables=[tensors])


class TestOrientation(DMTTestCase):
    def terms(self):
        bonds = {(0, 1): np.kron(SIGMA_Z, SIGMA_Z), (1, 3): 0.6 * np.kron(SIGMA_X, SIGMA_X)}
        return bonds, {2: 0.3 * SIGMA_Z, 3: SIGMA_X}

    def test_a_reversed_path_gives_the_observable_of_the_reversed_chain(self):
        """Reversing the sweep must give the same operator with its qubits renumbered, so the
        flat coefficient vector of the oriented chain is that of the reversed qubit order."""
        n = 4
        qubits = [f"q{k}" for k in range(n)]
        bonds, sites = self.terms()
        tensors = pauli_operator_tensors(n, bonds, sites)
        flat = TestObservableReservation.flat
        forward = flat(orient_observable(tensors, qubits, qubits)).reshape((4,) * n)
        backward = flat(orient_observable(tensors, qubits, qubits[::-1])).reshape((4,) * n)
        np.testing.assert_allclose(backward, np.transpose(forward, (3, 2, 1, 0)), atol=1e-12)

    def test_the_forward_orientation_is_a_copy_of_the_same_tensors(self):
        tensors = pauli_operator_tensors(3, site_terms={1: SIGMA_Z})
        qubits = ["a", "b", "c"]
        oriented = orient_observable(tensors, qubits, qubits)
        for got, want in zip(oriented, tensors):
            self.assertTrue(np.array_equal(got, want))

    def test_any_other_order_is_refused(self):
        tensors = pauli_operator_tensors(3)
        with self.assertRaisesRegex(ValueError, "qubit order"):
            orient_observable(tensors, ["a", "b", "c"], ["b", "a", "c"])


class TestHelpers(DMTTestCase):
    def test_tol_rank(self):
        sing = np.array([1.0, 0.5, 1e-3, 1e-9])
        self.assertEqual(3, _tol_rank(sing, 1e-6, 0.0))
        self.assertEqual(2, _tol_rank(sing, 1e-6, 0.01))
        self.assertEqual(0, _tol_rank(np.array([]), 1e-6, 0.0))

    def test_respect_degenerate_pulls_back_to_the_multiplet_start(self):
        sing = np.array([2.0, 1.0, 1.0, 1.0, 0.1])
        self.assertEqual(1, _respect_degenerate(sing, 3, 1e-10))
        self.assertEqual(4, _respect_degenerate(sing, 4, 1e-10))
        self.assertEqual(2, _respect_degenerate(np.array([1.0, 1.0, 0.5]), 2, 1e-10))

    def test_respect_degenerate_keeps_a_block_that_is_one_multiplet(self):
        self.assertEqual(2, _respect_degenerate(np.array([1.0, 1.0, 1.0]), 2, 1e-10))

    def test_radius_levels(self):
        self.assertEqual([0], list(_radius_levels(0)))
        levels = _radius_levels(2)
        self.assertEqual(16, levels.size)
        counts = {level: int((levels == level).sum()) for level in (0, 1, 2)}
        self.assertEqual({0: 1, 1: 3, 2: 12}, counts)

    def test_the_ordered_basis_keeps_priority_order_and_drops_dependent_columns(self):
        cols = np.array([[1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
        basis = _ordered_orthonormal_basis(cols, 1e-12)
        self.assertEqual(2, basis.shape[1])
        np.testing.assert_allclose(basis[:, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-14)


if __name__ == "__main__":
    main()
