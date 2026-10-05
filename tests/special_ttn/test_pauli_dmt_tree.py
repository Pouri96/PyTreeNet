import itertools
import warnings
from copy import deepcopy
from unittest import TestCase, main

import numpy as np

from pytreenet.special_ttn.pauli import (
    dmt_min_chi_for_tree_radius, dmt_tree_reserved_count, network_is_real,
    pauli_dmt_truncate, pauli_dmt_truncate_tree, pauli_hermiticity_defect, pauli_product_mps,
    pauli_product_tree_allphys, pauli_product_tree_virtnode, pauli_reweight_network,
    pauli_site_ids, pauli_to_coeffs, pauli_trace)

N = 7
KETS = [np.array([1, 0], dtype=complex)] * N


def random_real_tree(template, chi, rng):
    net = deepcopy(template)
    for node_id, node in net.nodes.items():
        open_dim = int(np.prod([net.tensors[node_id].shape[leg] for leg in node.open_legs]))
        net.replace_tensor(node_id, rng.normal(size=[chi] * node.nneighbours() + [open_dim]),
                           new_shape=True)
    return net


def max_bond(net):
    return max(node.neighbour_dim(nb) for node in net.nodes.values()
               for nb in node.neighbouring_nodes())


def tree_distance(net, a, b):
    return len(net.path_from_to(a, b)) - 1


def is_connected(net, nodes):
    nodes = list(nodes)
    seen, frontier = {nodes[0]}, [nodes[0]]
    while frontier:
        cur = frontier.pop()
        for nb in net.nodes[cur].neighbouring_nodes():
            if nb in nodes and nb not in seen:
                seen.add(nb)
                frontier.append(nb)
    return len(seen) == len(nodes)


def changes_by_diameter(net, before, after, ids):
    """``{tree-diameter: [relative change of the marginal]}`` over every connected node set.
    The marginal on a set is the coefficient slice with the identity on every other qubit."""
    n = len(ids)
    out = {}
    for size in range(1, n + 1):
        for subset in itertools.combinations(range(n), size):
            nodes = [ids[q] for q in subset]
            if not is_connected(net, nodes):
                continue
            diameter = max(tree_distance(net, a, b) for a in nodes for b in nodes)
            idx = tuple(slice(None) if q in subset else 0 for q in range(n))
            change = np.linalg.norm(after[idx] - before[idx]) / np.linalg.norm(before[idx])
            out.setdefault(diameter, []).append(change)
    return out


class TreeDMTTestCase(TestCase):
    def setUp(self):
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        self.rng = np.random.default_rng(11)
        self.layouts = {"inorder": pauli_product_tree_allphys(KETS),
                        "heap": pauli_product_tree_allphys(KETS, order="heap")}

    def coeffs(self, net, ids):
        return pauli_to_coeffs(net, ids).real.reshape((4,) * len(ids))


class TestGuaranteeInTreeDistance(TreeDMTTestCase):
    """Every operator on a connected set of tree-diameter at most 2R survives, and the sets beyond
    it do not. The second half guards against a fixture that truncated nothing."""

    def check(self, layout, radius, chi):
        net = random_real_tree(self.layouts[layout], 16, self.rng)
        ids = pauli_site_ids(net)
        before = self.coeffs(net, ids)
        out, _ = pauli_dmt_truncate_tree(net, chi, radius=radius)
        self.assertLessEqual(max_bond(out), chi)
        return changes_by_diameter(net, before, self.coeffs(out, ids), ids)

    def test_radius_one_protects_diameter_two_on_both_layouts(self):
        for layout in self.layouts:
            changes = self.check(layout, 1, 10)
            with self.subTest(layout=layout):
                for diameter in (0, 1, 2):
                    self.assertLess(max(changes[diameter]), 1e-11, msg=diameter)
                for diameter in (3, 4):
                    if diameter in changes:
                        self.assertGreater(max(changes[diameter]), 1e-2, msg=diameter)

    def test_radius_zero_protects_single_nodes_only(self):
        for layout in self.layouts:
            changes = self.check(layout, 0, 6)
            with self.subTest(layout=layout):
                self.assertLess(max(changes[0]), 1e-11)
                self.assertGreater(max(changes[1]), 1e-2)

    def test_the_trace_is_exact_without_any_pin(self):
        net = random_real_tree(self.layouts["inorder"], 16, self.rng)
        out, _ = pauli_dmt_truncate_tree(net, 10)
        np.testing.assert_allclose(pauli_trace(out).real, pauli_trace(net).real, rtol=1e-12)

    def test_a_chain_is_a_tree_and_gets_the_chain_guarantee(self):
        net = random_real_tree(pauli_product_mps(KETS), 16, self.rng)
        ids = pauli_site_ids(net)
        before = self.coeffs(net, ids)
        out, _ = pauli_dmt_truncate_tree(net, 10, radius=1)
        changes = changes_by_diameter(net, before, self.coeffs(out, ids), ids)
        self.assertLess(max(max(changes[d]) for d in (0, 1, 2)), 1e-11)
        self.assertGreater(max(changes[3]), 1e-2)


class TestVirtualInterior(TreeDMTTestCase):
    def setUp(self):
        super().setUp()
        self.kets = [np.array([1, 0], dtype=complex)] * 5
        self.template = pauli_product_tree_virtnode(self.kets)

    def test_single_qubit_marginals_survive(self):
        net = random_real_tree(self.template, 14, self.rng)
        ids = pauli_site_ids(net)
        n = len(ids)
        before = self.coeffs(net, ids)
        out, _ = pauli_dmt_truncate_tree(net, 10, radius=1)
        self.assertLessEqual(max_bond(out), 10)
        after = self.coeffs(out, ids)
        for q in range(n):
            idx = tuple(slice(None) if k == q else 0 for k in range(n))
            self.assertLess(np.linalg.norm(after[idx] - before[idx])
                            / np.linalg.norm(before[idx]), 1e-11, msg=q)

    def test_the_trace_is_exact(self):
        net = random_real_tree(self.template, 14, self.rng)
        out, _ = pauli_dmt_truncate_tree(net, 10)
        np.testing.assert_allclose(pauli_trace(out).real, pauli_trace(net).real, rtol=1e-12)

    def test_a_virtual_node_is_not_a_variable_so_the_floor_is_lower(self):
        self.assertEqual(5, dmt_tree_reserved_count(self.template, 1))
        self.assertEqual(7, dmt_min_chi_for_tree_radius(self.template, 1))
        self.assertEqual(4, dmt_min_chi_for_tree_radius(self.template, 0))


class TestFloors(TreeDMTTestCase):
    def test_all_physical_tree_floors_grow_with_radius(self):
        net = self.layouts["inorder"]
        self.assertEqual(2, dmt_tree_reserved_count(net, 0))
        self.assertEqual(8, dmt_tree_reserved_count(net, 1))
        self.assertEqual([4, 10, 82], [dmt_min_chi_for_tree_radius(net, r) for r in (0, 1, 2)])

    def test_the_floor_grows_with_the_node_degree(self):
        """Radius 2 reserves the endpoint and its other neighbours, so a degree-3 endpoint
        contributes three variables, against two on a chain."""
        chain = pauli_product_mps(KETS)
        tree = self.layouts["inorder"]
        self.assertEqual(34, dmt_min_chi_for_tree_radius(chain, 2))
        self.assertGreater(dmt_min_chi_for_tree_radius(tree, 2),
                           dmt_min_chi_for_tree_radius(chain, 2))

    def test_the_chain_floor_matches_the_chain_helper(self):
        from pytreenet.special_ttn.pauli import dmt_min_chi_for_radius
        chain = pauli_product_mps(KETS)
        for radius in (0, 1, 2):
            self.assertEqual(dmt_min_chi_for_radius(radius),
                             dmt_min_chi_for_tree_radius(chain, radius))


class TestRankLimitedBonds(TreeDMTTestCase):
    def padded_tree(self, layout, true_bond, padded_bond=16):
        """A tree whose true bond is small, zero-padded to a larger one: the rank-limited
        scaffolding that an augmented sweep produces."""
        small = random_real_tree(self.layouts[layout], true_bond, self.rng)
        net = random_real_tree(self.layouts[layout], padded_bond, self.rng)
        ids = pauli_site_ids(net)
        for sid in ids:
            src = small.tensors[sid]
            dst = np.zeros(net.tensors[sid].shape)
            dst[tuple(slice(0, s) for s in src.shape)] = src
            net.replace_tensor(sid, dst, new_shape=False)
        return net, ids

    def test_a_lossless_pass_over_rank_limited_trees_stays_lossless(self):
        """The arbitrary completions that the SVD returns for the null space must be projected
        off the reserved span. Whether they corrupt a given tree depends on its data (the error
        reaches O(1) on some draws and is rounding on others), so this sweeps a grid."""
        for layout in self.layouts:
            for true_bond in (3, 4):
                for chi in (8, 10, 12):
                    net, ids = self.padded_tree(layout, true_bond)
                    before = self.coeffs(net, ids)
                    out, _ = pauli_dmt_truncate_tree(net, chi)
                    error = np.linalg.norm(self.coeffs(out, ids) - before) / np.linalg.norm(before)
                    with self.subTest(layout=layout, true_bond=true_bond, chi=chi):
                        self.assertLess(error, 1e-11)


class TestReportAndOptions(TreeDMTTestCase):
    def test_a_lossless_pass_reports_exactly_zero_and_changes_nothing(self):
        net = random_real_tree(self.layouts["inorder"], 3, self.rng)
        ids = pauli_site_ids(net)
        before = self.coeffs(net, ids)
        out, report = pauli_dmt_truncate_tree(net, 64)
        self.assertEqual(0.0, report.discarded)
        self.assertEqual({}, report.reserved_ranks)
        np.testing.assert_allclose(self.coeffs(out, ids), before, atol=1e-11)

    def test_the_ledger_bounds_the_relative_error(self):
        net = random_real_tree(self.layouts["inorder"], 16, self.rng)
        ids = pauli_site_ids(net)
        before = self.coeffs(net, ids)
        out, report = pauli_dmt_truncate_tree(net, 10)
        error = np.linalg.norm(self.coeffs(out, ids) - before) / np.linalg.norm(before)
        self.assertGreater(report.discarded, 0.0)
        self.assertLessEqual(error, report.discarded * (1 + 1e-9))

    def test_the_per_bond_weights_match_the_ledger(self):
        net = random_real_tree(self.layouts["inorder"], 16, self.rng)
        _, report = pauli_dmt_truncate_tree(net, 10)
        self.assertGreater(len(report.discarded_weights), 0)
        self.assertAlmostEqual(report.discarded, float(np.sum(np.sqrt(report.discarded_weights))),
                               places=12)

    def test_the_report_is_keyed_by_the_child_node_of_the_edge(self):
        net = random_real_tree(self.layouts["inorder"], 16, self.rng)
        _, report = pauli_dmt_truncate_tree(net, 10)
        non_root = {nid for nid, node in net.nodes.items() if node.parent is not None}
        self.assertTrue(report.reserved_ranks)
        self.assertLessEqual(set(report.reserved_ranks), non_root)
        self.assertLessEqual(max(report.reserved_ranks.values()), 10)

    def test_the_result_is_real_hermitian_and_centred_at_the_root(self):
        net = random_real_tree(self.layouts["inorder"], 16, self.rng)
        out, _ = pauli_dmt_truncate_tree(net, 10)
        self.assertTrue(network_is_real(out, tol=0.0))
        self.assertEqual(0.0, pauli_hermiticity_defect(out))
        self.assertEqual(out.root_id, out.orthogonality_center_id)

    def test_in_place_returns_the_same_object_and_the_default_does_not(self):
        net = random_real_tree(self.layouts["inorder"], 16, self.rng)
        before = max_bond(net)
        out, _ = pauli_dmt_truncate_tree(net, 10)
        self.assertIsNot(out, net)
        self.assertEqual(before, max_bond(net))
        same, _ = pauli_dmt_truncate_tree(net, 10, in_place=True)
        self.assertIs(same, net)
        self.assertLessEqual(max_bond(net), 10)


class TestBudgetAndRefusals(TreeDMTTestCase):
    def test_a_budget_below_the_reservation_warns_and_respects_the_cap(self):
        net = random_real_tree(self.layouts["inorder"], 16, self.rng)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, report = pauli_dmt_truncate_tree(net, 5, radius=1)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning)
                            and "degree" in str(w.message) for w in caught))
        self.assertLessEqual(max_bond(out), 5)
        self.assertTrue(report.clipped_bonds)

    def test_a_clipped_reservation_keeps_the_trace_first(self):
        """The cap on the reserved set is what protects the trace. Without it the lossless
        re-factorisation would truncate by singular value, which does not know the priority."""
        net = random_real_tree(self.layouts["inorder"], 16, self.rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, _ = pauli_dmt_truncate_tree(net, 3, radius=1)
        np.testing.assert_allclose(pauli_trace(out).real, pauli_trace(net).real, rtol=1e-11)

    def test_the_refactorisation_after_the_projection_is_lossless(self):
        """DMT has already chosen the rank, so the split that restores canonical form must not
        apply a tolerance of its own."""
        from unittest import mock

        import pytreenet.special_ttn.pauli.dmt_tree as tree_module
        net = random_real_tree(self.layouts["inorder"], 16, self.rng)
        with mock.patch.object(tree_module, "split_svd_contract_sv_to_neighbour",
                               wraps=tree_module.split_svd_contract_sv_to_neighbour) as spy:
            pauli_dmt_truncate_tree(net, 10, rel_tol=1e-6, total_tol=1e-6)
        self.assertGreater(spy.call_count, 0)
        for call in spy.call_args_list:
            params = call.args[3]
            self.assertEqual(10, params.max_bond_dim)
            self.assertLessEqual(params.rel_tol, 1e-14)
            self.assertEqual(0.0, params.total_tol)

    def test_no_warning_when_the_reservation_fits(self):
        net = random_real_tree(self.layouts["inorder"], 16, self.rng)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pauli_dmt_truncate_tree(net, dmt_min_chi_for_tree_radius(net, 1))

    def test_a_negative_radius_is_refused(self):
        with self.assertRaisesRegex(ValueError, "radius"):
            pauli_dmt_truncate_tree(self.layouts["inorder"], 4, radius=-1)

    def test_a_reweighted_state_is_refused(self):
        net = deepcopy(self.layouts["inorder"])
        pauli_reweight_network(net, 2.0)
        with self.assertRaisesRegex(ValueError, "reweighted"):
            pauli_dmt_truncate_tree(net, 4)

    def test_the_chain_function_refuses_a_tree_with_virtual_nodes_and_points_here(self):
        with self.assertRaisesRegex(ValueError, "pauli_dmt_truncate_tree"):
            pauli_dmt_truncate(pauli_product_tree_virtnode(KETS[:4]), 4)


if __name__ == "__main__":
    main()
