"""Tests for Gauss-Hermite quadrature utilities."""

import numpy as np
import pytest
from insurance_jlm.models.quadrature import (
    gauss_hermite_points,
    product_rule_2d,
    ghq_integral_1d,
    ghq_integral_2d,
    posterior_covariance_approx,
)


class TestGaussHermitePoints:
    def test_returns_correct_number_of_points(self):
        pts, wts = gauss_hermite_points(7)
        assert len(pts) == 7
        assert len(wts) == 7

    def test_single_point(self):
        pts, wts = gauss_hermite_points(1)
        assert len(pts) == 1
        # Single GHQ point should be at 0
        assert abs(pts[0]) < 1e-10

    def test_points_are_symmetric(self):
        pts, wts = gauss_hermite_points(7)
        # Points should be symmetric around 0
        assert abs(pts[0] + pts[-1]) < 1e-10

    def test_weights_are_positive(self):
        _, wts = gauss_hermite_points(7)
        assert (wts > 0).all()

    def test_weights_sum_to_sqrt_pi(self):
        # For standard hermgauss, weights sum to sqrt(pi)
        _, wts = gauss_hermite_points(7)
        assert abs(np.sum(wts) - np.sqrt(np.pi)) < 1e-10

    def test_larger_n_points(self):
        pts, wts = gauss_hermite_points(15)
        assert len(pts) == 15
        assert (wts > 0).all()

    def test_points_ascending(self):
        pts, _ = gauss_hermite_points(7)
        assert np.all(np.diff(pts) > 0)


class TestProductRule2D:
    def test_returns_correct_shape(self):
        nodes, weights = product_rule_2d(7)
        assert nodes.shape == (49, 2)
        assert weights.shape == (49,)

    def test_weights_positive(self):
        _, weights = product_rule_2d(7)
        assert (weights > 0).all()

    def test_three_points(self):
        nodes, weights = product_rule_2d(3)
        assert nodes.shape == (9, 2)

    def test_nodes_span_correct_range(self):
        nodes, _ = product_rule_2d(7)
        pts_1d, _ = gauss_hermite_points(7)
        assert abs(nodes[:, 0].min() - pts_1d.min()) < 1e-10


class TestGHQIntegral1D:
    def test_integrates_constant(self):
        """∫ 5 N(b; 0, 1) db = 5"""
        result = ghq_integral_1d(lambda b: np.full_like(b, 5.0), 0.0, 1.0, n_points=7)
        assert abs(result - 5.0) < 1e-6

    def test_integrates_linear(self):
        """∫ b N(b; μ, σ²) db = μ"""
        mu = 3.0
        result = ghq_integral_1d(lambda b: b, mu, 1.0, n_points=7)
        assert abs(result - mu) < 1e-5

    def test_integrates_quadratic_variance(self):
        """∫ b² N(b; 0, σ²) db = σ²"""
        sigma2 = 4.0
        result = ghq_integral_1d(lambda b: b ** 2, 0.0, sigma2, n_points=7)
        assert abs(result - sigma2) < 1e-4

    def test_integrates_gaussian_pdf(self):
        """∫ N(b; 0, 1) db = 1"""
        from scipy.stats import norm
        result = ghq_integral_1d(
            lambda b: norm.pdf(b), 0.0, 1.0, n_points=7
        )
        assert abs(result - 1.0) < 1e-5

    def test_handles_nonzero_mean(self):
        """∫ (b - 2) N(b; 2, 1) db = 0"""
        result = ghq_integral_1d(lambda b: b - 2.0, 2.0, 1.0, n_points=7)
        assert abs(result) < 1e-5


class TestGHQIntegral2D:
    def test_integrates_constant_2d(self):
        """∫ 3 N(b; 0, I) db = 3"""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        result = ghq_integral_2d(lambda b: 3.0, mean, cov, n_points=7)
        assert abs(result - 3.0) < 1e-4

    def test_integrates_sum_of_components(self):
        """∫ (b1 + b2) N(b; μ, I) db = μ1 + μ2"""
        mean = np.array([2.0, -1.0])
        cov = np.eye(2)
        result = ghq_integral_2d(lambda b: b[0] + b[1], mean, cov, n_points=7)
        assert abs(result - (2.0 + (-1.0))) < 1e-4

    def test_integrates_quadratic_form(self):
        """∫ b'b N(b; 0, I) db = dim = 2"""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        result = ghq_integral_2d(lambda b: b @ b, mean, cov, n_points=7)
        assert abs(result - 2.0) < 1e-3

    def test_handles_correlated_cov(self):
        """Integral should still work with off-diagonal covariance."""
        mean = np.array([1.0, 1.0])
        cov = np.array([[4.0, 1.0], [1.0, 2.0]])
        # ∫ 1 N(b; mean, cov) db = 1
        result = ghq_integral_2d(lambda b: 1.0, mean, cov, n_points=7)
        assert abs(result - 1.0) < 1e-3


class TestPosteriorCovarianceApprox:
    def test_returns_positive_definite(self):
        """Result should be positive definite."""
        H = np.array([[-5.0, -0.5], [-0.5, -3.0]])  # negative definite
        cov = posterior_covariance_approx(H)
        eigvals = np.linalg.eigvalsh(cov)
        assert (eigvals > 0).all()

    def test_inverse_relationship(self):
        """Cov = -H⁻¹ approximately."""
        H = np.array([[-4.0, -0.2], [-0.2, -2.0]])
        cov = posterior_covariance_approx(H)
        expected = np.linalg.inv(-H)
        np.testing.assert_allclose(cov, expected, atol=1e-10)

    def test_1d_case(self):
        H = np.array([[-5.0]])
        cov = posterior_covariance_approx(H)
        assert abs(cov[0, 0] - 0.2) < 1e-10

    def test_symmetric_output(self):
        H = np.array([[-6.0, -1.0], [-1.0, -4.0]])
        cov = posterior_covariance_approx(H)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
