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
    pca_regularized_wcov_distance,
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


# ----------------------------------------------------------------------
# pca_regularized_wcov_distance
# ----------------------------------------------------------------------

def _two_class_cohort(n_per_class, n_features, scale=1.0):
    """Two balanced classes with unit-ish within-class scatter, mean-separated."""
    a = np.random.randn(n_per_class, n_features) * scale
    b = np.random.randn(n_per_class, n_features) * scale + 3.0
    X = np.vstack([a, b])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def test_pca_regularized_wcov_distance_is_zero_for_identical_cohorts():
    X, y = _two_class_cohort(200, 12)
    d = pca_regularized_wcov_distance(X, y, X.copy(), y.copy(), k=5)
    assert d == pytest.approx(0.0, abs=1e-8)


def test_pca_regularized_wcov_distance_is_invariant_to_shared_rotation():
    # Rotating the feature space of BOTH cohorts must not change the distance:
    # the joint PCA basis rotates with the data and the Frobenius norm of a
    # log-covariance difference is orthogonally invariant.
    X1, y1 = _two_class_cohort(150, 10)
    X2, y2 = _two_class_cohort(150, 10, scale=1.8)
    Q = np.linalg.qr(np.random.randn(10, 10))[0]

    plain = pca_regularized_wcov_distance(X1, y1, X2, y2, k=4)
    rotated = pca_regularized_wcov_distance(X1 @ Q, y1, X2 @ Q, y2, k=4)
    assert rotated == pytest.approx(plain, rel=1e-6)


def test_pca_regularized_wcov_distance_grows_with_within_class_variance_ratio():
    # A cohort whose within-class scatter is inflated by a larger factor must
    # sit further away. Run many draws and compare medians so no seed is needed.
    def median_distance(scale, trials=40):
        out = []
        for _ in range(trials):
            X1, y1 = _two_class_cohort(120, 10)
            X2, y2 = _two_class_cohort(120, 10, scale=scale)
            out.append(pca_regularized_wcov_distance(X1, y1, X2, y2, k=4))
        return float(np.median(out))

    near = median_distance(1.2)
    far = median_distance(3.0)
    assert far > near


def test_pca_regularized_wcov_distance_recovers_known_isotropic_scaling():
    # Scaling every feature of cohort 2 by s scales its within-class covariance
    # by s^2 in any shared orthonormal subspace, so logm differs by 2*log(s)*I
    # and the Frobenius norm is sqrt(k) * 2 * |log s|.
    s, k = 2.5, 4
    errs = []
    for _ in range(25):
        X1, y1 = _two_class_cohort(400, 8)
        d = pca_regularized_wcov_distance(X1, y1, X1 * s, y1.copy(), k=k)
        errs.append(abs(d - np.sqrt(k) * 2 * np.log(s)))
    assert float(np.median(errs)) < 0.05
