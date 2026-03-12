"""Gauss-Hermite quadrature utilities for the EM E-step.

The EM algorithm requires integrating over the random effects distribution.
We use Gauss-Hermite quadrature (GHQ), which is the standard approach for
this class of problem — see Rizopoulos (2012) Chapter 3.

For a dim=2 random effects vector (random intercept + slope), a product rule
with 7 points per dimension gives 49 evaluation points. This is accurate
enough for typical problems. Higher dimensions need adaptive GHQ or MCMC.

The key identity exploited is:

  ∫ f(b) exp(-b'b) db ≈ Σ_k w_k f(x_k)

where (x_k, w_k) are GHQ abscissae and weights from numpy.polynomial.hermite.
For a general Gaussian integral N(μ, Σ), we transform via the Cholesky factor.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss


def gauss_hermite_points(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return standard Gauss-Hermite abscissae and weights.

    The returned weights are the probabilist's convention (include the
    exp(-x²) factor). Abscissae are in ascending order.

    Parameters
    ----------
    n_points:
        Number of quadrature points. 7 is the standard choice for dim(b)=2.
        Higher values increase accuracy at the cost of computation.

    Returns
    -------
    abscissae:
        Shape (n_points,). Quadrature nodes.
    weights:
        Shape (n_points,). Corresponding weights (sum to sqrt(π)).
    """
    abscissae, weights = hermgauss(n_points)
    return abscissae, weights


def product_rule_2d(
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 2-D product Gauss-Hermite rule.

    For integrating f(b1, b2) exp(-(b1² + b2²)) db1 db2, we take the
    Cartesian product of the 1-D rule.

    Parameters
    ----------
    n_points:
        Points per dimension. Total evaluation points = n_points².

    Returns
    -------
    nodes:
        Shape (n_points², 2). Each row is a (b1, b2) quadrature node.
    weights:
        Shape (n_points²,). Product weights.
    """
    pts, wts = gauss_hermite_points(n_points)
    # Cartesian product
    b1, b2 = np.meshgrid(pts, pts, indexing="ij")
    w1, w2 = np.meshgrid(wts, wts, indexing="ij")
    nodes = np.column_stack([b1.ravel(), b2.ravel()])
    weights = (w1 * w2).ravel()
    return nodes, weights


def ghq_integral_1d(
    func: "callable",
    mean: float,
    var: float,
    n_points: int,
) -> float:
    """Approximate ∫ func(b) * N(b; mean, var) db via GHQ.

    Parameters
    ----------
    func:
        Function of a scalar b.
    mean:
        Mean of the Gaussian.
    var:
        Variance (not std) of the Gaussian.
    n_points:
        GHQ points.

    Returns
    -------
    Approximation of the integral.
    """
    pts, wts = gauss_hermite_points(n_points)
    sd = np.sqrt(var)
    # Transform: b = mean + sqrt(2) * sd * x
    b_vals = mean + np.sqrt(2.0) * sd * pts
    # Include the Jacobian sqrt(2)*sd and the normalisation 1/sqrt(π)
    return np.sum(wts * func(b_vals)) * np.sqrt(2.0) * sd / np.sqrt(np.pi)


def ghq_integral_2d(
    func: "callable",
    mean: np.ndarray,
    cov: np.ndarray,
    n_points: int,
) -> float:
    """Approximate ∫ func(b) * N(b; mean, cov) db via 2-D product GHQ.

    Uses the Cholesky factor of cov to handle correlated random effects.

    Parameters
    ----------
    func:
        Function of a length-2 array b. Should return a scalar.
    mean:
        Shape (2,). Mean vector.
    cov:
        Shape (2, 2). Covariance matrix. Must be positive definite.
    n_points:
        GHQ points per dimension.

    Returns
    -------
    Approximation of the 2-D integral.
    """
    nodes, weights = product_rule_2d(n_points)
    L = np.linalg.cholesky(cov)
    # Transform: b = mean + sqrt(2) * L @ x
    b_vals = mean + np.sqrt(2.0) * (nodes @ L.T)
    # Evaluate integrand at all nodes
    f_vals = np.array([func(b) for b in b_vals])
    # Include Jacobian det(sqrt(2)*L) and normalisation 1/pi
    jac = (np.sqrt(2.0) ** 2) * np.linalg.det(L)
    return np.sum(weights * f_vals) * jac / np.pi


def posterior_covariance_approx(
    log_posterior_hessian: np.ndarray,
) -> np.ndarray:
    """Compute posterior covariance from the Hessian of the log-posterior.

    At the posterior mode, the Laplace approximation gives:
      Var(b | data) ≈ -H⁻¹  where H = ∇² log p(b | data)

    Parameters
    ----------
    log_posterior_hessian:
        Shape (d, d). Hessian of the log-posterior at the mode. Should be
        negative definite.

    Returns
    -------
    Shape (d, d). Approximate posterior covariance matrix.
    """
    H = np.asarray(log_posterior_hessian)
    cov = np.linalg.solve(-H, np.eye(H.shape[0]))
    # Symmetrise to guard against numerical drift
    return 0.5 * (cov + cov.T)
