"""Calibration and discrimination metrics for dynamic predictions.

Standard metrics from Rizopoulos (2012) and Blanche et al. (2015):

1. Time-dependent AUC: discriminates claimants from non-claimants using the
   predicted survival probability at a given landmark time.

2. Brier score: measures calibration of survival predictions.
   BS(t) = (1/n) Σ_i [Ŝ(t|Ĥ_i)² * I(T_i ≤ t, δ_i=1) / KM_weights_i
                      + (1-Ŝ(t|Ĥ_i))² * I(T_i > t) / KM_weights_i]

3. Expected-to-Actual ratio: actuary-friendly check. Compare the number of
   predicted events to observed events over a time window.

The Brier score and time-dependent AUC require inverse probability of
censoring weighting (IPCW) to handle censored subjects correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..models.joint_model import JointModel


def brier_score(
    model: "JointModel",
    data: pd.DataFrame,
    landmark_time: float,
    horizon: float,
    n_mc: int = 100,
) -> float:
    """Compute the integrated Brier score at a single horizon.

    Uses IPCW weighting to handle censoring. The censoring distribution is
    estimated via the Kaplan-Meier estimator on the event indicator reversed
    (i.e., censoring = event in the KM).

    Parameters
    ----------
    model:
        A fitted JointModel.
    data:
        Test data (can be same as training for apparent performance).
    landmark_time:
        Landmark time t.
    horizon:
        Prediction horizon Δ. We predict P(T > t + Δ | T > t).
    n_mc:
        Monte Carlo samples for dynamic prediction.

    Returns
    -------
    Scalar Brier score. Smaller is better. A naive model (always predict 0.5)
    gives BS = 0.25.
    """
    pred_df = model.predict_survival(
        data, model._id_col, landmark_time, horizon, n_mc
    )
    subject_data = data.groupby(model._id_col).first().reset_index()
    merged = pred_df.merge(
        subject_data[[model._id_col, model._event_time_col, model._event_col]].rename(
            columns={model._id_col: "id"}
        ),
        on="id",
    )

    t_eval = landmark_time + horizon
    ipcw_weights = _ipcw_weights(
        subject_data[model._event_time_col].values,
        subject_data[model._event_col].values,
        t_eval,
    )
    id_to_weight = dict(zip(
        subject_data[model._id_col].values,
        ipcw_weights,
    ))

    bs = 0.0
    n = 0
    for _, row in merged.iterrows():
        T_i = row[model._event_time_col]
        delta_i = row[model._event_col]
        S_hat = row["survival_prob"]
        w_i = id_to_weight.get(row["id"], 1.0)

        if T_i <= t_eval and delta_i == 1:
            # Event occurred in window
            bs += (S_hat ** 2) * w_i
        elif T_i > t_eval:
            # Censored or survived past horizon
            bs += ((1.0 - S_hat) ** 2) * w_i
        # Censored before t_eval: excluded (weight effectively 0 via IPCW)
        n += 1

    return bs / n if n > 0 else np.nan


def time_dependent_auc(
    model: "JointModel",
    data: pd.DataFrame,
    landmark_time: float,
    horizon: float,
    n_mc: int = 100,
) -> float:
    """Compute time-dependent AUC for dynamic predictions.

    Uses the incident/dynamic definition (Heagerty & Zheng, 2005):
    AUC(t) = P(Ŝ(t+Δ|j) < Ŝ(t+Δ|k) | T_j ≤ t+Δ, T_k > t+Δ)

    A score of 0.5 indicates no discrimination. Values above 0.7 are
    generally considered acceptable for insurance pricing models.

    Parameters
    ----------
    model:
        A fitted JointModel.
    data:
        Test data.
    landmark_time:
        Landmark time.
    horizon:
        Prediction horizon.
    n_mc:
        Monte Carlo samples.

    Returns
    -------
    Scalar AUC in [0, 1].
    """
    pred_df = model.predict_survival(
        data, model._id_col, landmark_time, horizon, n_mc
    )
    subject_data = data.groupby(model._id_col).first().reset_index()
    merged = pred_df.merge(
        subject_data[[model._id_col, model._event_time_col, model._event_col]].rename(
            columns={model._id_col: "id"}
        ),
        on="id",
    )

    t_eval = landmark_time + horizon
    events = merged[(merged[model._event_time_col] <= t_eval) & (merged[model._event_col] == 1)]
    non_events = merged[merged[model._event_time_col] > t_eval]

    if len(events) == 0 or len(non_events) == 0:
        return np.nan

    # Lower survival prob for cases (events), higher for controls (non-events)
    # AUC = P(Ŝ_case < Ŝ_control)
    concordant = 0
    total = 0
    for _, e_row in events.iterrows():
        for _, c_row in non_events.iterrows():
            total += 1
            if e_row["survival_prob"] < c_row["survival_prob"]:
                concordant += 1
            elif e_row["survival_prob"] == c_row["survival_prob"]:
                concordant += 0.5

    return concordant / total if total > 0 else np.nan


def expected_actual_ratio(
    model: "JointModel",
    data: pd.DataFrame,
    landmark_time: float,
    horizon: float,
    n_mc: int = 100,
) -> pd.DataFrame:
    """Compute Expected/Actual event ratio — the actuarial calibration check.

    Groups subjects by predicted quintile of survival probability and computes
    the E/A ratio within each quintile. Ratios near 1.0 indicate good
    calibration. Systematic patterns indicate mis-calibration.

    Parameters
    ----------
    model:
        A fitted JointModel.
    data:
        Test data.
    landmark_time, horizon:
        Prediction window.
    n_mc:
        Monte Carlo samples.

    Returns
    -------
    DataFrame with columns: quintile, n_subjects, n_events_actual,
    n_events_expected, ea_ratio.
    """
    pred_df = model.predict_survival(
        data, model._id_col, landmark_time, horizon, n_mc
    )
    subject_data = data.groupby(model._id_col).first().reset_index()
    merged = pred_df.merge(
        subject_data[[model._id_col, model._event_time_col, model._event_col]].rename(
            columns={model._id_col: "id"}
        ),
        on="id",
    )

    t_eval = landmark_time + horizon
    # Predicted claim probability = 1 - survival_prob
    merged["claim_prob"] = 1.0 - merged["survival_prob"]
    merged["actual_claim"] = (
        (merged[model._event_time_col] <= t_eval) &
        (merged[model._event_col] == 1)
    ).astype(int)

    merged["quintile"] = pd.qcut(merged["claim_prob"], q=5, labels=False)

    results = []
    for q in range(5):
        subset = merged[merged["quintile"] == q]
        n = len(subset)
        actual = int(subset["actual_claim"].sum())
        expected = float(subset["claim_prob"].sum())
        ea = expected / actual if actual > 0 else np.nan
        results.append({
            "quintile": q + 1,
            "n_subjects": n,
            "n_events_actual": actual,
            "n_events_expected": round(expected, 2),
            "ea_ratio": round(ea, 3) if not np.isnan(ea) else np.nan,
        })

    return pd.DataFrame(results)


def _ipcw_weights(
    times: np.ndarray,
    events: np.ndarray,
    t_eval: float,
) -> np.ndarray:
    """Inverse Probability of Censoring Weights (IPCW) at time t_eval.

    Estimates the censoring distribution via Kaplan-Meier with the event
    indicator flipped, then returns 1/G(T_i) for each subject.
    """
    # Simple KM estimate of censoring distribution
    censoring = 1 - events  # censor indicator = 1 where event didn't occur
    # Sort by time
    order = np.argsort(times)
    sorted_times = times[order]
    sorted_censoring = censoring[order]

    n = len(times)
    G = np.ones(n)  # survival function for censoring
    G_t = 1.0

    unique_times = np.unique(sorted_times)
    G_at_times = {}

    at_risk = n
    for t in unique_times:
        at_t = np.sum((sorted_times == t) & (sorted_censoring == 1))
        G_t *= (1.0 - at_t / at_risk) if at_risk > 0 else G_t
        G_at_times[t] = max(G_t, 1e-6)
        at_risk -= np.sum(sorted_times == t)

    weights = np.ones(n)
    for i, t_i in enumerate(times):
        if t_i <= t_eval and events[i] == 1:
            # Use G(T_i-)
            prev_times = [t for t in G_at_times if t < t_i]
            g_val = G_at_times[max(prev_times)] if prev_times else 1.0
            weights[i] = 1.0 / max(g_val, 1e-6)
        elif t_i > t_eval:
            g_val = G_at_times.get(t_eval, G_at_times.get(
                max((t for t in G_at_times if t <= t_eval), default=0.0), 1.0
            ))
            weights[i] = 1.0 / max(g_val, 1e-6)
        else:
            weights[i] = 0.0  # Censored before t_eval, excluded

    return weights
