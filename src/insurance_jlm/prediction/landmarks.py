"""Landmark analysis: simpler alternative to dynamic prediction.

Landmarking (van Houwelingen & Putter, 2011) is a pragmatic alternative to the
full dynamic prediction from a joint model. Instead of computing the conditional
survival probability via integration over random effects, we:

1. Fix a set of landmark times t_L
2. For each t_L, take the subset of subjects still at risk at t_L
3. Fit a standard Cox model using covariate values measured at t_L
4. Predict from t_L to t_L + w (the window)

Landmarking is faster and more interpretable than the joint model's dynamic
prediction, but it:
- Discards historical trajectory information (only uses current value)
- Requires re-fitting a model at each landmark time
- Does not naturally extrapolate to unseen landmark times

We include it here as a diagnostic benchmark. If the joint model's dynamic
predictions substantially differ from landmark predictions, it suggests the
trajectory shape (not just current value) matters — which is the whole point
of using a joint model in the first place.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class LandmarkPredictor:
    """Landmark-based dynamic prediction for comparison with JointModel.

    Parameters
    ----------
    landmark_times:
        List of times at which landmark models are fitted.
    window:
        Prediction window beyond each landmark time.
    """

    def __init__(
        self,
        landmark_times: list[float],
        window: float,
    ) -> None:
        self.landmark_times = sorted(landmark_times)
        self.window = window
        self._landmark_models_: dict = {}

    def fit(
        self,
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        y_col: str,
        event_time_col: str,
        event_col: str,
        covariates: list[str],
    ) -> "LandmarkPredictor":
        """Fit a landmark Cox model at each landmark time.

        Parameters
        ----------
        data:
            Long-format data. The marker at the landmark time is extracted
            as the last observation at or before the landmark.
        id_col, time_col, y_col, event_time_col, event_col:
            Column names as in JointModel.fit().
        covariates:
            Baseline covariate names to include alongside the marker.

        Returns
        -------
        self
        """
        try:
            from lifelines import CoxPHFitter
        except ImportError as exc:
            raise ImportError(
                "lifelines is required for landmark analysis. "
                "Install with: pip install lifelines"
            ) from exc

        for t_L in self.landmark_times:
            # Risk set: subjects with event_time > t_L
            subject_data = data.groupby(id_col).first().reset_index()
            at_risk = subject_data[subject_data[event_time_col] > t_L]

            # Marker at landmark: last observation <= t_L
            landmark_markers = (
                data[data[time_col] <= t_L]
                .groupby(id_col)[y_col]
                .last()
                .reset_index()
                .rename(columns={y_col: "landmark_marker"})
            )

            # Build landmark dataset
            landmark_data = at_risk.merge(landmark_markers, on=id_col, how="left")
            landmark_data["duration"] = np.minimum(
                landmark_data[event_time_col] - t_L, self.window
            )
            landmark_data["event_in_window"] = (
                (landmark_data[event_time_col] <= t_L + self.window) &
                (landmark_data[event_col] == 1)
            ).astype(int)

            fit_cols = ["duration", "event_in_window", "landmark_marker"] + covariates
            fit_data = landmark_data[fit_cols].dropna()

            if len(fit_data) < 10 or fit_data["event_in_window"].sum() < 3:
                continue

            cph = CoxPHFitter()
            try:
                cph.fit(fit_data, duration_col="duration", event_col="event_in_window")
                self._landmark_models_[t_L] = cph
            except Exception:
                continue

        return self

    def predict(
        self,
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        y_col: str,
        event_time_col: str,
        event_col: str,
        covariates: list[str],
        landmark_time: float,
    ) -> pd.DataFrame:
        """Predict survival probability from the nearest landmark.

        Parameters
        ----------
        landmark_time:
            The landmark time for which to predict.

        Returns
        -------
        DataFrame with columns: id, landmark_time, survival_prob.
        """
        # Find nearest fitted landmark
        available = sorted(self._landmark_models_.keys())
        if not available:
            raise RuntimeError("No landmark models fitted. Call fit() first.")

        nearest = min(available, key=lambda t: abs(t - landmark_time))
        cph = self._landmark_models_[nearest]

        # Prepare prediction data
        subject_data = data.groupby(id_col).first().reset_index()
        landmark_markers = (
            data[data[time_col] <= landmark_time]
            .groupby(id_col)[y_col]
            .last()
            .reset_index()
            .rename(columns={y_col: "landmark_marker"})
        )
        pred_data = subject_data.merge(landmark_markers, on=id_col, how="left")
        pred_cols = ["landmark_marker"] + covariates
        pred_input = pred_data[pred_cols].fillna(0)

        surv_probs = cph.predict_survival_function(pred_input, times=[self.window])
        results = []
        for i, subj_id in enumerate(pred_data[id_col]):
            prob = float(surv_probs.iloc[0, i])
            results.append({
                "id": subj_id,
                "landmark_time": nearest,
                "survival_prob": prob,
            })
        return pd.DataFrame(results)
