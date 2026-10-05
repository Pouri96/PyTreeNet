"""The DMT truncations against the original implementation they were ported from.

``data/bug1_dmt_reference.npz`` holds fixed random inputs and the results of the original chain
and tree DMT code (``pauli_dmt_truncate`` and ``pauli_dmt_truncate_tree`` of the BUG_1
branch, with ``rel_tol=total_tol=1e-12`` and ``respect_degenerate=True``), recorded on
2026-10-04 before that code was retired. The port agreed with it to 6e-15 when recorded.
"""
import os
import warnings
from unittest import TestCase, main

import numpy as np

from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.special_ttn.pauli import (
    pauli_dmt_truncate, pauli_dmt_truncate_tree, pauli_operator_tensors,
    pauli_product_tree_allphys, pauli_product_tree_virtnode, pauli_site_ids, pauli_to_coeffs)

REFERENCE = os.path.join(os.path.dirname(__file__), "data", "bug1_dmt_reference.npz")
N = 7
KETS = [np.array([1, 0], dtype=complex)] * N
TOLERANCE = 1e-10


class GoldenTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.load(REFERENCE)

    def setUp(self):
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        warnings.simplefilter("ignore", RuntimeWarning)

    def relative(self, got, want):
        return np.abs(got - want).max() / np.abs(want).max()


class TestChainAgainstTheOriginal(GoldenTestCase):
    def chain(self):
        tensors = [self.data[f"chain_tensor_{k}"] for k in range(N)]
        core = ([tensors[0].reshape(tensors[0].shape[1], 4)] + tensors[1:-1]
                + [tensors[-1].reshape(tensors[-1].shape[0], 4)])
        return MatrixProductState.from_tensor_list(core, node_prefix="qubit")

    def test_the_input_is_the_recorded_one(self):
        chain = self.chain()
        got = pauli_to_coeffs(chain, pauli_site_ids(chain)).real
        self.assertLess(self.relative(got, self.data["chain_c0"]), 1e-12)

    def test_radius_zero_and_one(self):
        for radius in (0, 1):
            chain = self.chain()
            ids = pauli_site_ids(chain)
            out, report = pauli_dmt_truncate(chain, 12, radius=radius)
            with self.subTest(radius=radius):
                got = pauli_to_coeffs(out, ids).real
                self.assertLess(self.relative(got, self.data[f"chain_radius_{radius}_coeffs"]),
                                TOLERANCE)
                self.assertAlmostEqual(float(self.data[f"chain_radius_{radius}_eta"]),
                                       report.discarded, places=10)

    def test_a_reserved_observable(self):
        bonds = dict(zip(map(tuple, self.data["obs_bond_keys"]), self.data["obs_bond_ops"]))
        sites = dict(zip(map(int, self.data["obs_site_keys"]), self.data["obs_site_ops"]))
        chain = self.chain()
        ids = pauli_site_ids(chain)
        out, report = pauli_dmt_truncate(
            chain, 20, radius=1, observables=[pauli_operator_tensors(N, bonds, sites)])
        got = pauli_to_coeffs(out, ids).real
        self.assertLess(self.relative(got, self.data["chain_observable_1_coeffs"]), TOLERANCE)
        self.assertAlmostEqual(float(self.data["chain_observable_1_eta"]), report.discarded,
                               places=10)


class TestTreeAgainstTheOriginal(GoldenTestCase):
    LAYOUTS = {"allphys": lambda: pauli_product_tree_allphys(KETS),
               "virtnode": lambda: pauli_product_tree_virtnode(KETS[:5])}

    def tree(self, name):
        tree = self.LAYOUTS[name]()
        prefix = f"tree_{name}_tensor_"
        for key in self.data.files:
            if key.startswith(prefix):
                tree.replace_tensor(key[len(prefix):], self.data[key], new_shape=True)
        return tree

    def test_the_input_is_the_recorded_one(self):
        for name in self.LAYOUTS:
            tree = self.tree(name)
            got = pauli_to_coeffs(tree, pauli_site_ids(tree)).real
            with self.subTest(layout=name):
                self.assertLess(self.relative(got, self.data[f"tree_{name}_c0"]), 1e-12)

    def test_both_layouts_at_radius_one_and_zero(self):
        for name in self.LAYOUTS:
            for radius, chi in ((1, 10), (0, 6)):
                tree = self.tree(name)
                ids = pauli_site_ids(tree)
                out, report = pauli_dmt_truncate_tree(tree, chi, radius=radius)
                key = f"tree_{name}_r{radius}_chi{chi}"
                with self.subTest(layout=name, radius=radius):
                    got = pauli_to_coeffs(out, ids).real
                    self.assertLess(self.relative(got, self.data[key + "_coeffs"]), TOLERANCE)
                    self.assertAlmostEqual(float(self.data[key + "_eta"]), report.discarded,
                                           places=10)


if __name__ == "__main__":
    main()
