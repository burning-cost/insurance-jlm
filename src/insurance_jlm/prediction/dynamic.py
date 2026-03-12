"""Dynamic predictor: update claim hazard as new telematics data arrives.

The key insight of the joint model is that we can update our estimate of a
driver's claim probability in real time as new telematics readings arrive,
without refitting the model. This is the 'dynamic prediction' capability.

For a subject currently observed at time t with history ỹ_i(t):
  P(T_i > t + Δ | T_i > t, ỹ_i(t))

As new readings arrive, this probability is re-evaluated using the updated
posterior of the random effects given the extended history.

In practice for UK telematics:
- Telematics scoring typically runs monthly
- So Δ = 1 month is the natural prediction horizon
- An improving driver (b₁ > 0, positive slope) gets lower hazard
- A deteriorating driver (b₁ < 0) gets higher hazard

This class wraps the JointModel to provide a stateful interface for
real-time updates, suitable for use in a pricing engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..models.joint_model import JointModel


class DynamicPredictor:
    """Real-time dynamic predictor for a single subject.

    Given a fitted JointModel, this class maintains a subject's longitudinal
    history and provides updated hazard/survival estimates as new measurements
    arrive.

    Parameters
    ----------
    model:
        A fitted JointModel.
    subject_id:
        The subject's identifier.
    initial_data:
        Long-format data for this subject up to the current time.
    n_mc:
        Monte Carlo samples for posterior integration.

    Examples
    --------
    >>> predictor = DynamicPredictor(model, "P001", initial_data, n_mc=100)
    >>> predictor.predict_survival(horizon=3.0)
    0.87
    >>> predictor.update({"month": 4, "telematics_score": 65})
    >>> predictor.predict_survival(horizon=3.0)
    0.82  # Worsening score reduces survival probability
    """

    def __init__(
        self,
        model: "JointModel",
        subject_id: str,
        initial_data: pd.DataFrame,
        n_mc: int = 200,
    ) -> None:
        self.model = model
        self.subject_id = subject_id
        self.history = initial_data.copy()
        self.n_mc = n_mc
        self._hazard_history_: list[dict] = []

    def update(self, new_measurement: dict) -> float:
        """Add a new measurement and return the updated survival probability.

        Parameters
        ----------
        new_measurement:
            Dict with keys matching the data columns (time_col, y_col, and
            any longitudinal covariates). Event columns are not needed here —
            the subject is still in the risk set by construction.

        Returns
        -------
        Updated survival probability for the default 1-unit horizon.
        """
        # Append new measurement to history
        new_row = {self.model._id_col: self.subject_id}
        new_row.update(new_measurement)
        new_df = pd.DataFrame([new_row])
        self.history = pd.concat([self.history, new_df], ignore_index=True)

        current_time = float(new_measurement[self.model._time_col])
        surv_prob = self._compute_survival(current_time, current_time + 1.0)

        self._hazard_history_.append({
            "time": current_time,
            "survival_prob_1_unit": surv_prob,
        })
        return surv_prob

    def predict_survival(
        self,
        horizon: float,
        landmark_time: Optional[float] = None,
    ) -> float:
        """Predict survival probability from the landmark time to landmark + horizon.

        Parameters
        ----------
        horizon:
            Prediction horizon (time units beyond the landmark).
        landmark_time:
            Current time. Defaults to the latest measurement time in history.

        Returns
        -------
        Survival probability P(T > landmark + horizon | T > landmark, history).
        """
        if landmark_time is None:
            landmark_time = float(self.history[self.model._time_col].max())
        return self._compute_survival(landmark_time, landmark_time + horizon)

    def risk_trajectory(self) -> pd.DataFrame:
        """Return the time-varying survival probability history.

        Returns a DataFrame tracking how the 1-unit-ahead survival probability
        has changed as new measurements arrived.

        Returns
        -------
        DataFrame with columns: time, survival_prob_1_unit.
        """
        if not self._hazard_history_:
            return pd.DataFrame(columns=["time", "survival_prob_1_unit"])
        return pd.DataFrame(self._hazard_history_)

    def _compute_survival(
        self, landmark_time: float, horizon_time: float
    ) -> float:
        """Internal: compute dynamic survival probability."""
        obs_data = self.history[
            self.history[self.model._time_col] <= landmark_time
        ].copy()

        # Add required event columns (censored at landmark)
        if self.model._event_time_col not in obs_data.columns:
            obs_data[self.model._event_time_col] = landmark_time
        if self.model._event_col not in obs_data.columns:
            obs_data[self.model._event_col] = 0

        return self.model._dynamic_survival(
            obs_data,
            self.subject_id,
            landmark_time,
            horizon_time,
            self.n_mc,
        )
