"""Survival sub-model: Cox proportional hazards with time-varying covariate.

The hazard for subject i at time t is:

  h_i(t) = h₀(t) · exp(γ'x_i + α · m_i(t))

where:
  h₀(t)  = baseline hazard (estimated via Breslow estimator)
  x_i    = baseline (time-fixed) covariates for subject i
  γ       = regression coefficients for baseline covariates
  m_i(t) = predicted longitudinal trajectory at time t (from LME sub-model)
  α      = association parameter — the key parameter linking the two sub-models

The α parameter is what makes this a joint model. A positive α means that
subjects with higher telematics scores at any time t have higher claim hazard
at that time. This is the 'current value' association.

Estimation: during the EM M-step, we maximise the partial log-likelihood with
respect to (γ, α) for fixed m_i(t). The baseline hazard h₀(t) is updated via
the Breslow estimator.

We do NOT use lifelines' built-in time-varying covariates interface because it
assumes the covariate is observed at every event time, whereas here m_i(t) is a
model prediction. Instead we compute the partial likelihood directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SurvivalParams:
    """Parameters of the survival sub-model."""

    gamma: np.ndarray
    """Baseline covariate coefficients. Shape (p_surv,)."""

    alpha: float
    """Association parameter linking longitudinal to survival."""

    baseline_times: np.ndarray
    """Event times at which baseline hazard is defined. Shape (n_events,)."""

    baseline_hazard: np.ndarray
    """Cumulative baseline hazard increments at baseline_times. Shape (n_events,)."""

    surv_covariate_names: list[str] = field(default_factory=list)


class SurvivalSubmodel:
    """Cox PH survival sub-model with a predicted time-varying covariate m_i(t).

    Parameters
    ----------
    association:
        How to link longitudinal trajectory to hazard.
        'current_value' — uses m_i(t) at the event/censoring time
        'slope'         — uses dm_i/dt at time t
        'area'          — uses cumulative ∫₀ᵗ m_i(s) ds
    """

    def __init__(self, association: str = "current_value") -> None:
        if association not in ("current_value", "slope", "area"):
            raise ValueError(
                f"association must be 'current_value', 'slope', or 'area', "
                f"got '{association}'"
            )
        self.association = association
        self.params_: Optional[SurvivalParams] = None

    def fit(
        self,
        survival_data: pd.DataFrame,
        event_time_col: str,
        event_col: str,
        surv_covariates: list[str],
        marker_func: Callable[[str, float], float],
        init_gamma: Optional[np.ndarray] = None,
        init_alpha: float = 0.0,
    ) -> "SurvivalSubmodel":
        """Fit the Cox PH sub-model.

        This is called during the EM M-step with fixed m_i(t) from the
        longitudinal sub-model.

        Parameters
        ----------
        survival_data:
            One row per subject with columns for event_time, event indicator,
            and baseline covariates. Must have an 'id' column.
        event_time_col:
            Time-to-event column.
        event_col:
            Event indicator (1=event, 0=censored).
        surv_covariates:
            Baseline covariate column names.
        marker_func:
            Function (subject_id, time) → m_i(t). Called at event times.
        init_gamma:
            Initial values for γ. Zeros if None.
        init_alpha:
            Initial value for α.

        Returns
        -------
        self
        """
        n_gamma = len(surv_covariates)
        if init_gamma is None:
            init_gamma = np.zeros(n_gamma)

        theta0 = np.concatenate([init_gamma, [init_alpha]])

        def neg_partial_loglik(theta: np.ndarray) -> float:
            gamma = theta[:n_gamma]
            alpha = theta[n_gamma]
            return -self._partial_loglik(
                survival_data, event_time_col, event_col,
                surv_covariates, marker_func, gamma, alpha,
            )

        result = minimize(
            neg_partial_loglik,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-9},
        )

        gamma_hat = result.x[:n_gamma]
        alpha_hat = result.x[n_gamma]

        # Breslow baseline hazard estimate
        bt, bh = self._breslow_estimator(
            survival_data, event_time_col, event_col,
            surv_covariates, marker_func, gamma_hat, alpha_hat,
        )

        self.params_ = SurvivalParams(
            gamma=gamma_hat,
            alpha=alpha_hat,
            baseline_times=bt,
            baseline_hazard=bh,
            surv_covariate_names=surv_covariates,
        )
        return self

    def cumulative_hazard(
        self,
        t: float,
        x_i: np.ndarray,
        marker_func_i: Callable[[float], float],
    ) -> float:
        """Compute cumulative hazard H_i(t) = ∫₀ᵗ h_i(s) ds.

        Uses the Breslow baseline hazard summed up to time t.

        Parameters
        ----------
        t:
            Time up to which to integrate.
        x_i:
            Baseline covariates for subject i. Shape (p_surv,).
        marker_func_i:
            Function time → m_i(time) for this subject.

        Returns
        -------
        Scalar cumulative hazard.
        """
        if self.params_ is None:
            raise RuntimeError("Model has not been fitted.")
        p = self.params_

        # Indices of event times <= t
        mask = p.baseline_times <= t
        if not mask.any():
            return 0.0

        times_up = p.baseline_times[mask]
        h0_increments = p.baseline_hazard[mask]

        # For each event time, compute exp(linear predictor) at that time
        lp = float(p.gamma @ x_i)
        # Association contribution at each baseline event time
        marker_vals = np.array([marker_func_i(s) for s in times_up])
        cum_h = np.sum(h0_increments * np.exp(lp + p.alpha * marker_vals))
        return float(cum_h)

    def survival(
        self,
        t: float,
        x_i: np.ndarray,
        marker_func_i: Callable[[float], float],
    ) -> float:
        """Compute survival probability S_i(t) = exp(-H_i(t)).

        Parameters
        ----------
        t:
            Time of interest.
        x_i:
            Baseline covariates. Shape (p_surv,).
        marker_func_i:
            Function time → m_i(time).

        Returns
        -------
        Survival probability in [0, 1].
        """
        return float(np.exp(-self.cumulative_hazard(t, x_i, marker_func_i)))

    def _partial_loglik(
        self,
        data: pd.DataFrame,
        event_time_col: str,
        event_col: str,
        surv_covariates: list[str],
        marker_func: Callable[[str, float], float],
        gamma: np.ndarray,
        alpha: float,
    ) -> float:
        """Cox partial log-likelihood with time-varying m_i(t).

        The partial log-likelihood is:
          l(γ, α) = Σ_i δ_i [ γ'x_i + α·m_i(T_i) - log Σ_{j: T_j >= T_i} exp(γ'x_j + α·m_j(T_i)) ]

        where δ_i is the event indicator.
        """
        ids = data["id"].values
        event_times = data[event_time_col].values
        events = data[event_col].values
        X = data[surv_covariates].values if surv_covariates else np.zeros((len(data), 0))

        n = len(data)
        loglik = 0.0

        # Pre-compute linear predictors (baseline part)
        lp_base = X @ gamma if X.shape[1] > 0 else np.zeros(n)

        for i in range(n):
            if events[i] == 0:
                continue
            t_i = event_times[i]
            m_i_ti = marker_func(ids[i], t_i)
            lp_i = lp_base[i] + alpha * m_i_ti

            # Risk set: all j with T_j >= T_i
            in_risk = event_times >= t_i
            # Compute m_j(t_i) for all j in risk set
            risk_ids = ids[in_risk]
            risk_lp_base = lp_base[in_risk]
            m_risk = np.array([marker_func(jid, t_i) for jid in risk_ids])
            log_denom = np.log(np.sum(np.exp(risk_lp_base + alpha * m_risk)))

            loglik += lp_i - log_denom

        return loglik

    def _breslow_estimator(
        self,
        data: pd.DataFrame,
        event_time_col: str,
        event_col: str,
        surv_covariates: list[str],
        marker_func: Callable[[str, float], float],
        gamma: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Breslow non-parametric baseline hazard estimator.

        Returns
        -------
        event_times:
            Unique event times in ascending order.
        h0_increments:
            Baseline hazard increment at each event time.
        """
        ids = data["id"].values
        event_times = data[event_time_col].values
        events = data[event_col].values
        X = data[surv_covariates].values if surv_covariates else np.zeros((len(data), 0))
        n = len(data)
        lp_base = X @ gamma if X.shape[1] > 0 else np.zeros(n)

        unique_times = np.sort(np.unique(event_times[events == 1]))
        h0_increments = np.zeros(len(unique_times))

        for k, t_k in enumerate(unique_times):
            # Number of events at t_k
            n_events_k = float(np.sum((event_times == t_k) & (events == 1)))
            # Risk set denominator
            in_risk = event_times >= t_k
            risk_ids = ids[in_risk]
            risk_lp_base = lp_base[in_risk]
            m_risk = np.array([marker_func(jid, t_k) for jid in risk_ids])
            denom = np.sum(np.exp(risk_lp_base + alpha * m_risk))
            h0_increments[k] = n_events_k / denom if denom > 0 else 0.0

        return unique_times, h0_increments

    def summary(self) -> pd.DataFrame:
        """Return association and covariate coefficients.

        Returns
        -------
        DataFrame with columns: parameter, estimate.
        """
        if self.params_ is None:
            raise RuntimeError("Model has not been fitted.")
        p = self.params_
        rows = []
        for name, val in zip(p.surv_covariate_names, p.gamma):
            rows.append({"parameter": name, "estimate": val, "component": "gamma"})
        rows.append({"parameter": "alpha (association)", "estimate": p.alpha, "component": "alpha"})
        return pd.DataFrame(rows)
