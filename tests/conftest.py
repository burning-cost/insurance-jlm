"""Shared fixtures for insurance-jlm tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def small_dataset():
    """Minimal synthetic dataset: 80 subjects, 1-6 monthly readings each."""
    rng = np.random.default_rng(0)
    n = 80
    rows = []
    surv_rows = []

    D = np.array([[4.0, 0.5], [0.5, 0.25]])
    b_effects = rng.multivariate_normal([0.0, 0.0], D, size=n)
    ages = rng.uniform(20, 60, size=n)
    veh_ages = rng.uniform(0, 10, size=n)

    for i in range(n):
        b0, b1 = b_effects[i]
        age = ages[i]
        veh = veh_ages[i]
        n_obs = rng.integers(2, 7)
        # Claim time
        T_i = rng.exponential(10.0)
        T_i = min(T_i, 12.0)
        had_claim = 1 if T_i < 12.0 else 0

        for m in range(1, int(min(T_i + 1, n_obs + 1))):
            score = 70 + 0.1 * age - 0.5 * m + b0 + b1 * m + rng.normal(0, 3)
            rows.append({
                "policy_id": f"P{i:04d}",
                "month": float(m),
                "telematics_score": score,
                "age": age,
                "vehicle_age": veh,
                "claim_month": T_i,
                "had_claim": had_claim,
            })

    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def medium_dataset():
    """300-subject dataset for EM fitting tests."""
    from insurance_jlm.data import make_synthetic_telematics
    telem, claims = make_synthetic_telematics(n_subjects=300, random_state=1)
    data = telem.merge(
        claims[["policy_id", "claim_month", "had_claim"]],
        on="policy_id",
    )
    return data, telem, claims


@pytest.fixture(scope="session")
def fitted_model(medium_dataset):
    """Pre-fitted JointModel (no SEs for speed)."""
    from insurance_jlm import JointModel
    data, _, _ = medium_dataset
    model = JointModel(
        n_quad_points=3,
        max_iter=5,
        se_method="none",
        random_state=42,
    )
    model.fit(
        data=data,
        id_col="policy_id",
        time_col="month",
        y_col="telematics_score",
        event_time_col="claim_month",
        event_col="had_claim",
        long_covariates=["age"],
        surv_covariates=["age", "vehicle_age"],
    )
    return model, data
