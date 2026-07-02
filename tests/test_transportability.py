"""Unit tests for geometric transportability measures.

Includes synthetic known-truth cases: planted rotation should be
recoverable by principal-angle distance and H^1.
"""
import numpy as np
import pytest
from scipy import linalg

from src.transportability import (
    principal_angles,
    geodesic_distance,
    transport_matrix,
    cocycle_holonomy,
    top_k_subspace,
    sheaf_h1_two_cohort,
    sheaf_h1_multi_cohort,
    sheaf_q_test,
)


def _random_orthonormal(d, k, rng):
    M = rng.standard_normal((d, k))
    Q, _ = linalg.qr(M, mode="economic")
    return Q


def _rotate_subspace(U, angle, plane=(0, 1)):
    d = U.shape[0]
    R = np.eye(d)
    i, j = plane
    c, s = np.cos(angle), np.sin(angle)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    rotated = R @ U
    Q, _ = linalg.qr(rotated, mode="economic")
    return Q


class TestPrincipalAngles:
    def test_identical_subspaces_have_zero_angles(self):
        rng = np.random.default_rng(123)
        U = _random_orthonormal(20, 3, rng)
        angles = principal_angles(U, U)
        assert np.allclose(angles, 0, atol=1e-6)

    def test_planted_rotation_produces_nonzero_angles(self):
        rng = np.random.default_rng(456)
        U1 = _random_orthonormal(20, 3, rng)
        theta = 0.4
        U2 = _rotate_subspace(U1, theta, plane=(0, 1))
        dist = geodesic_distance(U1, U2)
        assert dist > 0.01

    def test_orthogonal_subspaces_have_pi_half(self):
        U1 = np.eye(4, 2)
        U2 = np.eye(4, 2)[:, ::-1]
        U2 = np.roll(np.eye(4), 2, axis=1)[:, :2]
        angles = principal_angles(U1, U2)
        assert all(a == pytest.approx(np.pi / 2, abs=0.01) for a in angles)


class TestGeodesicDistance:
    def test_same_subspace_zero_distance(self):
        rng = np.random.default_rng(789)
        U = _random_orthonormal(15, 4, rng)
        assert geodesic_distance(U, U) == pytest.approx(0, abs=1e-6)

    def test_planted_rotation_nonzero(self):
        rng = np.random.default_rng(101)
        U1 = _random_orthonormal(15, 4, rng)
        U2 = _rotate_subspace(U1, 0.5)
        assert geodesic_distance(U1, U2) > 0.3


class TestCocycleHolonomy:
    def test_trivial_cycle_identity(self):
        rng = np.random.default_rng(202)
        U = _random_orthonormal(10, 3, rng)
        _, dev = cocycle_holonomy([U, U, U])
        assert dev == pytest.approx(0, abs=1e-6)

    def test_planted_curvature_detected(self):
        rng = np.random.default_rng(303)
        d, k = 20, 3
        U0 = _random_orthonormal(d, k, rng)
        subspaces = [U0]
        planes = [(0, 1), (2, 3), (0, 3), (1, 2)]
        for i, plane in enumerate(planes):
            subspaces.append(_rotate_subspace(subspaces[-1], 0.5, plane=plane))
        _, dev = cocycle_holonomy(subspaces)
        assert dev > 0.01


class TestSheafH1TwoCohort:
    def test_identical_data_low_h1(self):
        rng = np.random.default_rng(404)
        X = rng.standard_normal((200, 50))
        result = sheaf_h1_two_cohort(X, X, k=5)
        assert result["geodesic_dist"] < 0.5

    def test_shifted_data_high_h1(self):
        rng = np.random.default_rng(505)
        X1 = rng.standard_normal((200, 50))
        from scipy.stats import ortho_group
        rotation = ortho_group.rvs(50, random_state=42)
        X2 = X1 @ rotation
        result = sheaf_h1_two_cohort(X1, X2, k=5)
        assert result["geodesic_dist"] > 0.5


class TestSheafQTest:
    def test_homogeneous_estimates_nonsignificant(self):
        estimates = {
            f"site_{i}": {"beta": 0.3 + 0.01 * i, "se": 0.05}
            for i in range(5)
        }
        p, Q, df = sheaf_q_test(estimates)
        assert p > 0.05

    def test_heterogeneous_estimates_significant(self):
        estimates = {
            "site_0": {"beta": 0.3, "se": 0.02},
            "site_1": {"beta": 0.3, "se": 0.02},
            "site_2": {"beta": 0.3, "se": 0.02},
            "site_3": {"beta": 0.8, "se": 0.02},
        }
        p, Q, df = sheaf_q_test(estimates)
        assert p < 0.001


class TestTopKSubspace:
    def test_returns_orthonormal_basis(self):
        rng = np.random.default_rng(606)
        X = rng.standard_normal((100, 30))
        U, ev = top_k_subspace(X, k=5)
        assert U.shape == (30, 5)
        assert np.allclose(U.T @ U, np.eye(5), atol=1e-10)

    def test_explained_variance_sums_to_less_than_one(self):
        rng = np.random.default_rng(707)
        X = rng.standard_normal((100, 30))
        _, ev = top_k_subspace(X, k=5)
        assert ev.sum() < 1.0
        assert all(e > 0 for e in ev)
