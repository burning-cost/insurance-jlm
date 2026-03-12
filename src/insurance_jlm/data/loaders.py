"""Convenience loaders for common insurance data shapes.

These functions handle the data wrangling to get from typical UK insurance
data formats into the long format that JointModel expects.

UK telematics data typically arrives in two tables:
- telematics_df: one row per monthly score reading per policy
- claims_df: one row per policy with claim indicator and claim date

UK NCD data typically arrives as:
- ncd_df: NCD level at each renewal, indexed by policy reference
- lapse_df: lapse indicator at each renewal
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..models.joint_model import JointModel


def jlm_from_telematics(
    telematics_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    policy_col: str = "policy_id",
    month_col: str = "month",
    score_col: str = "telematics_score",
    claim_month_col: str = "claim_month",
    claim_indicator_col: str = "had_claim",
    long_covariates: Optional[list[str]] = None,
    surv_covariates: Optional[list[str]] = None,
    model_kwargs: Optional[dict] = None,
) -> JointModel:
    """Fit a joint model from standard UK telematics + claims tables.

    Parameters
    ----------
    telematics_df:
        Monthly telematics scores. Must contain policy_col, month_col,
        score_col. May contain additional covariates.
    claims_df:
        One row per policy with claim_month_col (months to first claim or
        censoring time) and claim_indicator_col (1=claim, 0=censored).
        Must contain policy_col and any covariates.
    policy_col:
        Policy identifier column name (must match in both DataFrames).
    month_col:
        Month number column in telematics_df (typically 1-12).
    score_col:
        Telematics score column name.
    claim_month_col:
        Time-to-first-claim column in claims_df.
    claim_indicator_col:
        Claim indicator column in claims_df (1=claimed, 0=censored).
    long_covariates:
        Covariates to include in the longitudinal sub-model.
    surv_covariates:
        Covariates to include in the survival sub-model.
    model_kwargs:
        Additional keyword arguments passed to JointModel().

    Returns
    -------
    Fitted JointModel.

    Examples
    --------
    >>> model = jlm_from_telematics(
    ...     telematics_df=telem,
    ...     claims_df=claims,
    ...     long_covariates=['age'],
    ...     surv_covariates=['age', 'vehicle_age'],
    ... )
    >>> model.association_summary()
    """
    if long_covariates is None:
        long_covariates = []
    if surv_covariates is None:
        surv_covariates = []
    if model_kwargs is None:
        model_kwargs = {}

    # Merge survival info onto telematics
    surv_cols = [policy_col, claim_month_col, claim_indicator_col] + surv_covariates
    surv_cols = list(dict.fromkeys(surv_cols))  # deduplicate
    available_surv = [c for c in surv_cols if c in claims_df.columns]

    data = telematics_df.merge(
        claims_df[available_surv],
        on=policy_col,
        how="inner",
    )

    model = JointModel(**model_kwargs)
    model.fit(
        data=data,
        id_col=policy_col,
        time_col=month_col,
        y_col=score_col,
        event_time_col=claim_month_col,
        event_col=claim_indicator_col,
        long_covariates=long_covariates,
        surv_covariates=surv_covariates,
    )
    return model


def jlm_from_ncd(
    ncd_df: pd.DataFrame,
    lapse_df: pd.DataFrame,
    policy_col: str = "policy_id",
    renewal_col: str = "renewal_number",
    ncd_col: str = "ncd_level",
    lapse_renewal_col: str = "lapse_renewal",
    lapse_indicator_col: str = "lapsed",
    long_covariates: Optional[list[str]] = None,
    surv_covariates: Optional[list[str]] = None,
    model_kwargs: Optional[dict] = None,
) -> JointModel:
    """Fit a joint model from NCD trajectory + lapse data.

    NCD (No Claims Discount) level is the longitudinal marker. Lapse is the
    event. This models how the trajectory of NCD level affects the probability
    of a policyholder lapsing.

    The NCD trajectory captures earned loyalty: a policyholder on a rising NCD
    trajectory is less likely to lapse (they are building their discount).
    A policyholder whose NCD is capped (typically at 60-70%) may be more price-
    sensitive and therefore more likely to lapse.

    Parameters
    ----------
    ncd_df:
        NCD level at each renewal. Must contain policy_col, renewal_col, ncd_col.
    lapse_df:
        One row per policy with lapse_renewal_col (renewal number at which lapse
        or censoring occurred) and lapse_indicator_col (1=lapsed, 0=retained).
    policy_col:
        Policy identifier column.
    renewal_col:
        Renewal number column (integer, 1-based).
    ncd_col:
        NCD level column (typically 0-60 or 0-9 depending on insurer scale).
    lapse_renewal_col:
        Renewal number at which lapse or censoring occurred.
    lapse_indicator_col:
        Lapse indicator (1=lapsed, 0=retained/censored).
    long_covariates:
        Covariates for longitudinal sub-model.
    surv_covariates:
        Covariates for survival sub-model.
    model_kwargs:
        Additional keyword arguments for JointModel.

    Returns
    -------
    Fitted JointModel.
    """
    if long_covariates is None:
        long_covariates = []
    if surv_covariates is None:
        surv_covariates = []
    if model_kwargs is None:
        model_kwargs = {}

    surv_cols = [policy_col, lapse_renewal_col, lapse_indicator_col] + surv_covariates
    surv_cols = list(dict.fromkeys(surv_cols))
    available_surv = [c for c in surv_cols if c in lapse_df.columns]

    data = ncd_df.merge(
        lapse_df[available_surv],
        on=policy_col,
        how="inner",
    )

    model = JointModel(**model_kwargs)
    model.fit(
        data=data,
        id_col=policy_col,
        time_col=renewal_col,
        y_col=ncd_col,
        event_time_col=lapse_renewal_col,
        event_col=lapse_indicator_col,
        long_covariates=long_covariates,
        surv_covariates=surv_covariates,
    )
    return model


def make_synthetic_telematics(
    n_subjects: int = 500,
    max_months: int = 12,
    base_claim_rate: float = 0.08,
    alpha_true: float = 0.03,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic UK telematics + claims data for testing.

    Generates data from the true joint model with known parameters. Useful
    for validating that the EM algorithm recovers the true α.

    Parameters
    ----------
    n_subjects:
        Number of simulated policyholders.
    max_months:
        Maximum policy duration in months.
    base_claim_rate:
        Annual claim rate for an average driver.
    alpha_true:
        True association parameter. Positive = higher score → higher hazard.
        For telematics, we typically expect α > 0 (riskier drivers score lower
        but have higher score variance — the score is constructed inversely in
        practice, but this demo uses raw numeric values).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    telematics_df:
        Long-format telematics scores. Columns: policy_id, month,
        telematics_score, age, vehicle_age.
    claims_df:
        One row per policy. Columns: policy_id, claim_month, had_claim,
        age, vehicle_age.
    """
    rng = np.random.default_rng(random_state)

    # Subject-level characteristics
    ages = rng.integers(17, 75, size=n_subjects).astype(float)
    vehicle_ages = rng.integers(0, 15, size=n_subjects).astype(float)

    # Random effects: (b0, b1) ~ N(0, D) where D = [[4, 0.5], [0.5, 0.25]]
    D_true = np.array([[4.0, 0.5], [0.5, 0.25]])
    b_effects = rng.multivariate_normal([0.0, 0.0], D_true, size=n_subjects)

    # True fixed effects: score = 70 + 0.1*age - 0.5*month + b0 + b1*month
    beta_true = np.array([70.0, 0.1, -0.5])  # intercept, age, month slope
    sigma_true = 3.0

    # Survival: Cox PH with Weibull baseline
    # h0(t) = lambda * p * t^(p-1) with lambda=0.005, p=1.2
    lambda_h0 = base_claim_rate / max_months
    gamma_age = 0.015  # Older drivers have higher hazard (raw, not adjusted)
    gamma_veh = 0.02   # Older vehicles have higher hazard

    telematics_rows = []
    claims_rows = []

    for i in range(n_subjects):
        b0i, b1i = b_effects[i]
        age_i = ages[i]
        veh_i = vehicle_ages[i]

        # Generate survival time via inversion method
        # H(t) = ∫₀ᵗ h0(s) exp(γ'xi + α*m_i(s)) ds
        # We integrate numerically on a fine grid
        dt = 0.1
        t_grid = np.arange(dt, max_months + dt, dt)
        cum_hazard = 0.0
        T_i = max_months  # default: censored
        had_claim = 0

        lp_base = gamma_age * age_i + gamma_veh * veh_i

        for t in t_grid:
            m_t = (beta_true[0] + beta_true[1] * age_i + beta_true[2] * t +
                   b0i + b1i * t)
            h0_t = lambda_h0
            h_t = h0_t * np.exp(lp_base + alpha_true * m_t) * dt
            cum_hazard += h_t
            U = rng.uniform()
            if U < h_t and had_claim == 0:
                T_i = t
                had_claim = 1
                break

        # Generate monthly telematics readings
        observed_months = range(1, int(T_i) + 1)
        for month in observed_months:
            score = (beta_true[0] + beta_true[1] * age_i + beta_true[2] * month +
                     b0i + b1i * month + rng.normal(0, sigma_true))
            telematics_rows.append({
                "policy_id": f"P{i:06d}",
                "month": month,
                "telematics_score": round(score, 1),
                "age": age_i,
                "vehicle_age": veh_i,
            })

        claims_rows.append({
            "policy_id": f"P{i:06d}",
            "claim_month": round(T_i, 1),
            "had_claim": had_claim,
            "age": age_i,
            "vehicle_age": veh_i,
        })

    telematics_df = pd.DataFrame(telematics_rows)
    claims_df = pd.DataFrame(claims_rows)
    return telematics_df, claims_df
