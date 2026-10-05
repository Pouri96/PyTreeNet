from dataclasses import FrozenInstanceError
from unittest import TestCase, main

from pytreenet.special_ttn.pauli import DMT, PLAIN, RTEBD, PlainSVD, TruncationPolicy


class TestPolicies(TestCase):
    def test_all_policies_share_a_base(self):
        for policy in (PLAIN, DMT(), RTEBD(gamma=2.0)):
            self.assertIsInstance(policy, TruncationPolicy)

    def test_plain_is_the_default_singleton_and_compares_equal(self):
        self.assertEqual(PLAIN, PlainSVD())
        self.assertNotEqual(PLAIN, DMT())

    def test_dmt_defaults(self):
        policy = DMT()
        self.assertEqual(1, policy.radius)
        self.assertIsNone(policy.min_bond_dim)
        self.assertTrue(policy.respect_degenerate)
        self.assertIsNone(policy.conserved_terms)

    def test_policies_are_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            DMT().radius = 2
        with self.assertRaises(FrozenInstanceError):
            RTEBD(gamma=2.0).gamma = 3.0

    def test_policies_with_equal_fields_are_equal(self):
        self.assertEqual(DMT(radius=0), DMT(radius=0))
        self.assertNotEqual(DMT(radius=0), DMT(radius=1))
        self.assertEqual(RTEBD(gamma=2.2), RTEBD(gamma=2.2))

    def test_dmt_radius_validation(self):
        for bad in (-1, 1.5, True, "1", None):
            with self.subTest(radius=bad):
                with self.assertRaisesRegex(ValueError, "radius"):
                    DMT(radius=bad)
        DMT(radius=0)

    def test_dmt_min_bond_dim_validation(self):
        for bad in (0, -3, 2.5, True):
            with self.subTest(min_bond_dim=bad):
                with self.assertRaisesRegex(ValueError, "min_bond_dim"):
                    DMT(min_bond_dim=bad)
        DMT(min_bond_dim=12)

    def test_dmt_conserved_terms_validation(self):
        DMT(conserved_terms={"bond_terms": {}, "site_terms": {}})
        DMT(conserved_terms={"site_terms": {0: None}})
        for bad in ({}, [], "H", {"terms": {}}, {"bond_terms": {}, "extra": 1}):
            with self.subTest(conserved_terms=bad):
                with self.assertRaisesRegex(ValueError, "conserved_terms"):
                    DMT(conserved_terms=bad)

    def test_rtebd_requires_gamma(self):
        with self.assertRaises(TypeError):
            RTEBD()

    def test_rtebd_gamma_validation(self):
        for bad in (0.5, 0.0, -2.0, float("nan"), float("inf")):
            with self.subTest(gamma=bad):
                with self.assertRaises(ValueError):
                    RTEBD(gamma=bad)
        for bad in (True, "2", None):
            with self.subTest(gamma=bad):
                with self.assertRaisesRegex(ValueError, "number"):
                    RTEBD(gamma=bad)

    def test_rtebd_accepts_gamma_one_and_integers(self):
        self.assertEqual(1, RTEBD(gamma=1).gamma)
        self.assertEqual(3, RTEBD(gamma=3).gamma)


if __name__ == "__main__":
    main()
