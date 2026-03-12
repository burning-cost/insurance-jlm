"""Residual diagnostics for the joint model.

Standard residuals for joint models (Rizopoulos 2012, Chapter 5):

1. Marginal residuals: y_ij - β'z_ij  (ignores random effects)
2. Subject-specific residuals: y_ij - β'z_ij - b_i'w_ij  (uses BLUPs)
3. Martingale residuals: δ_i - H_i(T_i)  (survival sub-model)
4. Cox-Snell residuals: H_i(T_i) ~ Exp(1) if model is correct

For telematics pricing, the most useful check is the martingale residuals
plotted against telematics score — any remaining pattern suggests the
functional form of the association (current_value vs slope vs area) needs
revision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..models.joint_model import JointModel


def longitudinal_residuals(
    model: "JointModel",
    data: pd.DataFrame,
    type: str = "subject_specific",
) -> pd.DataFrame:
    """Compute longitudinal sub-model residuals.

    Parameters
    ----------
    model:
        A fitted JointModel.
    data:
        The training data (same format as used in fit()).
    type:
        'marginal'        — y_ij - β'z_ij (population average)
        'subject_specific' — y_ij - β'z_ij - b_i'w_ij (using BLUPs)

    Returns
    -------
    DataFrame with columns: id, time, observed, fitted, residual.
    """
    if model.long_submodel_ is None:
        raise RuntimeError("Model has not been fitted.")

    params = model.long_submodel_.params_
    re_df = model.long_submodel_.get_random_effects(data, model._id_col)

    rows = []
    for _, row in data.iterrows():
        subj_id = row[model._id_col]
        t = row[model._time_col]
        y = row[model._y_col]

        covariate_row = model._get_covariate_row(subj_id)
        z = model.long_submodel_._build_fixed_vector(
            t, covariate_row, model._time_col, model._long_covariates
        )
        fitted_fixed = float(z @ params.beta)

        if type == "subject_specific" and subj_id in re_df.index:
            w = model.long_submodel_._build_random_vector(t, model._time_col)
            b_i = re_df.loc[subj_id].values
            fitted = fitted_fixed + float(w @ b_i)
        else:
            fitted = fitted_fixed

        rows.append({
            "id": subj_id,
            "time": t,
            "observed": y,
            "fitted": fitted,
            "residual": y - fitted,
        })

    return pd.DataFrame(rows)


def martingale_residuals(
    model: "JointModel",
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Cox-Snell / martingale residuals for the survival sub-model.

    Martingale residuals: M_i = δ_i - Ĥ_i(T_i)

    These should have mean approximately zero. Plotting them against covariates
    reveals functional form misspecification in the survival sub-model.

    Parameters
    ----------
    model:
        A fitted JointModel.
    data:
        Training data.

    Returns
    -------
    DataFrame with columns: id, event_time, event, cumulative_hazard,
    martingale_residual, cox_snell_residual.
    """
    if model.surv_submodel_ is None:
        raise RuntimeError("Model has not been fitted.")

    surv_data = model._build_survival_data(data)
    re_df = model.long_submodel_.get_random_effects(data, model._id_col)

    rows = []
    for _, srow in surv_data.iterrows():
        subj_id = srow["id"]
        T_i = srow[model._event_time_col]
        delta_i = srow[model._event_col]
        covariate_row = model._get_covariate_row(subj_id)
        x_i = np.array([covariate_row[c] for c in model._surv_covariates])

        b_i = re_df.loc[subj_id].values if subj_id in re_df.index else np.zeros(
            len(model.long_submodel_.params_.random_names)
        )

        def marker_func(t: float) -> float:
            return model.long_submodel_.marker_value(
                t, b_i, covariate_row, model._time_col, model._long_covariates
            )

        H_i = model.surv_submodel_.cumulative_hazard(T_i, x_i, marker_func)
        martingale = float(delta_i) - H_i
        cox_snell = H_i  # Should be Exp(1) if model is correct

        rows.append({
            "id": subj_id,
            "event_time": T_i,
            "event": delta_i,
            "cumulative_hazard": H_i,
            "martingale_residual": martingale,
            "cox_snell_residual": cox_snell,
        })

    return pd.DataFrame(rows)


def deviance_residuals(
    model: "JointModel",
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute deviance residuals for the survival sub-model.

    Deviance residuals are a signed version of the martingale residuals that
    are more symmetrically distributed. Useful for identifying outlying subjects.

    d_i = sign(M_i) * sqrt(-2 * [M_i + δ_i * log(δ_i - M_i)])

    Parameters
    ----------
    model:
        A fitted JointModel.
    data:
        Training data.

    Returns
    -------
    DataFrame with columns: id, martingale_residual, deviance_residual.
    """
    mart_df = martingale_residuals(model, data)
    results = []
    for _, row in mart_df.iterrows():
        M_i = row["martingale_residual"]
        delta_i = row["event"]
        # Guard against log(0)
        inner = M_i + float(delta_i) * np.log(max(float(delta_i) - M_i, 1e-10))
        d_i = float(np.sign(M_i) * np.sqrt(-2.0 * inner))
        results.append({
            "id": row["id"],
            "martingale_residual": M_i,
            "deviance_residual": d_i,
        })
    return pd.DataFrame(results)
