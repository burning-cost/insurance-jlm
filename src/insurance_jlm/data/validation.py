"""Data validation utilities.

Real insurance data is messy. These validators give clear, actionable error
messages before the EM algorithm starts — not cryptic numpy errors halfway
through an hour-long fitting run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when input data fails validation checks."""
    pass


def validate_long_format(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    y_col: str,
    event_time_col: str,
    event_col: str,
    long_covariates: list[str],
    surv_covariates: list[str],
) -> list[str]:
    """Validate long-format data and return list of warnings.

    Raises DataValidationError for hard failures (will prevent fitting).
    Returns a list of warning strings for soft issues (fitting can proceed
    but results may be unreliable).

    Parameters
    ----------
    data:
        Long-format DataFrame.
    id_col:
        Subject identifier column.
    time_col:
        Measurement time column.
    y_col:
        Longitudinal outcome column.
    event_time_col:
        Time-to-event column.
    event_col:
        Event indicator column.
    long_covariates:
        Longitudinal fixed-effect covariate names.
    surv_covariates:
        Survival covariate names.

    Returns
    -------
    List of warning messages.
    """
    warnings_list = []

    # Check required columns
    required = [id_col, time_col, y_col, event_time_col, event_col]
    required += long_covariates + surv_covariates
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise DataValidationError(
            f"Missing columns: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    # Check event indicator
    unique_events = set(data[event_col].dropna().unique())
    if not unique_events.issubset({0, 1, 0.0, 1.0}):
        raise DataValidationError(
            f"event_col '{event_col}' must contain only 0 and 1. "
            f"Found: {unique_events}"
        )

    # Check event time is constant within subject
    et_var = data.groupby(id_col)[event_time_col].nunique()
    bad_subjects = list(et_var[et_var > 1].index[:5])
    if bad_subjects:
        raise DataValidationError(
            f"event_time_col '{event_time_col}' varies within subjects: {bad_subjects}. "
            f"Each subject must have a single event or censoring time."
        )

    # Check measurement times are <= event times
    max_meas_time = data.groupby(id_col)[time_col].max()
    event_times = data.groupby(id_col)[event_time_col].first()
    late_measurements = max_meas_time > event_times
    if late_measurements.any():
        n_late = int(late_measurements.sum())
        warnings_list.append(
            f"{n_late} subjects have measurements after their event time. "
            f"These measurements will be used in fitting but may indicate a "
            f"data alignment issue."
        )

    # Check minimum subjects per event
    n_subjects = data[id_col].nunique()
    n_events = data.groupby(id_col)[event_col].first().sum()
    event_rate = n_events / n_subjects
    if n_subjects < 50:
        raise DataValidationError(
            f"Only {n_subjects} subjects. Joint models need at least 50 "
            f"subjects (500+ recommended) for stable EM estimation."
        )
    if n_events < 20:
        raise DataValidationError(
            f"Only {int(n_events)} events. Need at least 20 events for "
            f"stable survival sub-model estimation."
        )
    if event_rate < 0.01:
        warnings_list.append(
            f"Event rate is very low ({event_rate:.3%}). The survival "
            f"sub-model may be unstable. Consider longer follow-up or pooling."
        )

    # Check longitudinal observations per subject
    obs_per_subject = data.groupby(id_col).size()
    if obs_per_subject.median() < 2:
        warnings_list.append(
            f"Median observations per subject is {obs_per_subject.median():.1f}. "
            f"Random slopes require at least 3-4 observations per subject on average."
        )

    # Check for missing values in key columns
    key_cols = [time_col, y_col, event_time_col, event_col]
    for col in key_cols:
        n_missing = data[col].isna().sum()
        if n_missing > 0:
            raise DataValidationError(
                f"Column '{col}' has {n_missing} missing values. "
                f"Impute or remove rows with missing values before fitting."
            )

    # Check for negative times
    if (data[time_col] < 0).any():
        raise DataValidationError(
            f"time_col '{time_col}' contains negative values. "
            f"Times must be non-negative."
        )
    if (data[event_time_col] < 0).any():
        raise DataValidationError(
            f"event_time_col '{event_time_col}' contains negative values."
        )

    # Check numeric types
    for col in [time_col, y_col, event_time_col]:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise DataValidationError(
                f"Column '{col}' must be numeric, got dtype {data[col].dtype}."
            )

    # Warn about outliers in y_col
    y_std = data[y_col].std()
    y_mean = data[y_col].mean()
    n_outliers = ((data[y_col] - y_mean).abs() > 5 * y_std).sum()
    if n_outliers > 0:
        warnings_list.append(
            f"{n_outliers} observations in '{y_col}' are more than 5 standard "
            f"deviations from the mean. Check for data entry errors."
        )

    return warnings_list


def summarise_data(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    y_col: str,
    event_time_col: str,
    event_col: str,
) -> pd.DataFrame:
    """Return a summary table of the longitudinal dataset.

    Parameters
    ----------
    data, id_col, time_col, y_col, event_time_col, event_col:
        As in validate_long_format().

    Returns
    -------
    DataFrame with descriptive statistics.
    """
    n_subjects = data[id_col].nunique()
    n_obs = len(data)
    n_events = int(data.groupby(id_col)[event_col].first().sum())
    obs_per_subj = data.groupby(id_col).size()

    summary = pd.DataFrame([
        {"statistic": "N subjects", "value": n_subjects},
        {"statistic": "N observations", "value": n_obs},
        {"statistic": "N events", "value": n_events},
        {"statistic": "Event rate", "value": f"{n_events / n_subjects:.1%}"},
        {"statistic": "Obs per subject (median)", "value": obs_per_subj.median()},
        {"statistic": "Obs per subject (min)", "value": obs_per_subj.min()},
        {"statistic": "Obs per subject (max)", "value": obs_per_subj.max()},
        {"statistic": f"{y_col} mean", "value": round(data[y_col].mean(), 2)},
        {"statistic": f"{y_col} std", "value": round(data[y_col].std(), 2)},
        {"statistic": f"{event_time_col} mean", "value": round(
            data.groupby(id_col)[event_time_col].first().mean(), 2
        )},
    ])
    return summary
