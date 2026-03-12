"""Longitudinal sub-model: linear mixed-effects via statsmodels MixedLM.

The longitudinal sub-model is:
  y_ij = β'z_ij + b_i'w_ij + ε_ij

where:
  y_ij  = observed marker (e.g. telematics score) for subject i at time t_ij
  z_ij  = fixed-effect design vector (intercept, time, covariates)
  w_ij  = random-effects design vector (intercept, time by default)
  b_i   ~ N(0, D)  — subject-specific random effects
  ε_ij  ~ N(0, σ²) — residual error

This class wraps statsmodels MixedLM but adds the EM-specific operations:
extracting posterior random effects, computing the trajectory mi(t), and
updating parameters in the M-step.

Design choice: we use statsmodels rather than a bespoke LME implementation
because it handles edge cases (singular D, constraints) that arise in practice
with real insurance data. The cost is a slower first-fit; the benefit is
correctness on messy data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults


@dataclass
class LongitudinalParams:
    """Parameters of the longitudinal sub-model."""

    beta: np.ndarray
    """Fixed-effects coefficients. Shape (p,)."""

    D: np.ndarray
    """Random-effects covariance matrix. Shape (q, q)."""

    sigma2: float
    """Residual variance σ²."""

    fixed_names: list[str] = field(default_factory=list)
    """Names matching columns of the fixed-effects design matrix."""

    random_names: list[str] = field(default_factory=list)
    """Names matching columns of the random-effects design matrix."""


class LongitudinalSubmodel:
    """Linear mixed-effects sub-model for the longitudinal marker.

    Parameters
    ----------
    long_model:
        Functional form for time in the longitudinal model.
        'linear'    — random intercept + slope
        'quadratic' — adds time² fixed effect (random intercept + slope only)
        'intercept' — random intercept only
    """

    def __init__(self, long_model: str = "linear") -> None:
        if long_model not in ("linear", "quadratic", "intercept"):
            raise ValueError(
                f"long_model must be 'linear', 'quadratic', or 'intercept', "
                f"got '{long_model}'"
            )
        self.long_model = long_model
        self.params_: Optional[LongitudinalParams] = None
        self._statsmodels_result_: Optional[MixedLMResults] = None
        self._fixed_cols_: Optional[list[str]] = None
        self._random_cols_: Optional[list[str]] = None

    def fit(
        self,
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        y_col: str,
        covariates: list[str],
    ) -> "LongitudinalSubmodel":
        """Fit the linear mixed-effects model.

        Parameters
        ----------
        data:
            Long-format DataFrame. One row per measurement per subject.
        id_col:
            Column name for subject identifier.
        time_col:
            Column name for measurement time.
        y_col:
            Column name for longitudinal outcome.
        covariates:
            Additional fixed-effect covariate names.

        Returns
        -------
        self
        """
        df = data.copy()

        # Build design matrices
        fixed_cols = ["intercept", time_col] + (
            [time_col + "_sq"] if self.long_model == "quadratic" else []
        ) + covariates

        if "intercept" not in df.columns:
            df["intercept"] = 1.0
        if self.long_model == "quadratic":
            df[time_col + "_sq"] = df[time_col] ** 2

        if self.long_model == "intercept":
            random_cols = ["intercept"]
        else:
            random_cols = ["intercept", time_col]

        self._fixed_cols_ = fixed_cols
        self._random_cols_ = random_cols

        # Build formula for statsmodels
        fixed_formula_parts = [time_col] + (
            [time_col + "_sq"] if self.long_model == "quadratic" else []
        ) + covariates
        formula = f"{y_col} ~ " + " + ".join(fixed_formula_parts)

        # Random effects: intercept + slope (or intercept only)
        if self.long_model == "intercept":
            re_formula = "~1"
        else:
            re_formula = f"~{time_col}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                formula,
                data=df,
                groups=df[id_col],
                re_formula=re_formula,
            )
            result = model.fit(method="lbfgs", disp=False)

        self._statsmodels_result_ = result

        # Extract parameters
        beta = result.fe_params.values
        beta_names = list(result.fe_params.index)

        # statsmodels stores RE covariance in result.cov_re
        D = result.cov_re.values
        sigma2 = result.scale

        self.params_ = LongitudinalParams(
            beta=beta,
            D=D,
            sigma2=sigma2,
            fixed_names=beta_names,
            random_names=list(result.cov_re.index),
        )
        return self

    def predict_trajectory(
        self,
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        covariates: list[str],
        times: np.ndarray,
    ) -> pd.DataFrame:
        """Predict marginal trajectory (fixed effects only) at given times.

        This is the population-average trajectory μ(t) = β'z(t). Individual
        trajectories require adding the posterior random effects b_i.

        Parameters
        ----------
        data:
            Must contain a single subject's data (filtered by caller).
        id_col:
            Subject ID column.
        time_col:
            Measurement time column.
        covariates:
            Fixed-effect covariate names.
        times:
            Times at which to evaluate the trajectory.

        Returns
        -------
        DataFrame with columns: id, time, trajectory.
        """
        if self.params_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        # Take covariate values from first row of this subject
        first_row = data.iloc[0]
        subject_id = first_row[id_col]

        results = []
        for t in times:
            z = self._build_fixed_vector(t, first_row, time_col, covariates)
            mu = float(z @ self.params_.beta)
            results.append({"id": subject_id, "time": t, "trajectory": mu})
        return pd.DataFrame(results)

    def get_random_effects(
        self,
        data: pd.DataFrame,
        id_col: str,
    ) -> pd.DataFrame:
        """Return posterior mode of random effects for each subject.

        Uses the BLUP (Best Linear Unbiased Predictor) from statsmodels.

        Parameters
        ----------
        data:
            Long-format data that was used to fit the model.
        id_col:
            Subject ID column.

        Returns
        -------
        DataFrame indexed by subject ID with columns for each random effect.
        """
        if self._statsmodels_result_ is None:
            raise RuntimeError("Model has not been fitted.")
        re = self._statsmodels_result_.random_effects
        rows = []
        for subj_id, effects in re.items():
            row = {"id": subj_id}
            row.update(effects.to_dict())
            rows.append(row)
        return pd.DataFrame(rows).set_index("id")

    def marker_value(
        self,
        t: float,
        b_i: np.ndarray,
        covariate_row: pd.Series,
        time_col: str,
        covariates: list[str],
    ) -> float:
        """Compute m_i(t) = β'z_i(t) + b_i'w_i(t) for a single subject.

        This is the predicted longitudinal value at time t given random effects
        b_i. Used in the survival sub-model as the time-varying covariate.

        Parameters
        ----------
        t:
            Evaluation time.
        b_i:
            Subject random effects. Shape matches random_names.
        covariate_row:
            A row (or representative row) from the subject's data providing
            baseline covariate values.
        time_col:
            Measurement time column name.
        covariates:
            Fixed-effect covariate names.

        Returns
        -------
        Scalar m_i(t).
        """
        if self.params_ is None:
            raise RuntimeError("Model has not been fitted.")
        z = self._build_fixed_vector(t, covariate_row, time_col, covariates)
        w = self._build_random_vector(t, time_col)
        return float(z @ self.params_.beta + w @ b_i)

    def _build_fixed_vector(
        self,
        t: float,
        row: pd.Series,
        time_col: str,
        covariates: list[str],
    ) -> np.ndarray:
        """Build fixed-effect design vector at time t."""
        parts = [1.0, t]  # intercept, time
        if self.long_model == "quadratic":
            parts.append(t ** 2)
        for cov in covariates:
            parts.append(float(row[cov]))
        return np.array(parts)

    def _build_random_vector(self, t: float, time_col: str) -> np.ndarray:
        """Build random-effect design vector at time t."""
        if self.long_model == "intercept":
            return np.array([1.0])
        return np.array([1.0, t])

    def summary(self) -> pd.DataFrame:
        """Return fixed-effects coefficients with standard errors.

        Returns
        -------
        DataFrame with columns: parameter, estimate, std_err, z_stat, p_value.
        """
        if self._statsmodels_result_ is None:
            raise RuntimeError("Model has not been fitted.")
        res = self._statsmodels_result_
        df = pd.DataFrame(
            {
                "parameter": res.fe_params.index,
                "estimate": res.fe_params.values,
                "std_err": np.sqrt(np.diag(res.cov_params()[:len(res.fe_params), :len(res.fe_params)])),
            }
        )
        df["z_stat"] = df["estimate"] / df["std_err"]
        df["p_value"] = 2.0 * (1.0 - _norm_cdf(np.abs(df["z_stat"])))
        return df.reset_index(drop=True)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via scipy."""
    from scipy.special import ndtr
    return ndtr(x)
