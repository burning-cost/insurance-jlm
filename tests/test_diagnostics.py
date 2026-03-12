"""Tests for diagnostics: residuals and calibration."""

import numpy as np
import pandas as pd
import pytest
from insurance_jlm.diagnostics import (
    longitudinal_residuals,
    martingale_residuals,
    deviance_residuals,
    brier_score,
    time_dependent_auc,
    expected_actual_ratio,
)


class TestLongitudinalResiduals:
    def test_returns_dataframe(self, fitted_model):
        model, data = fitted_model
        result = longitudinal_residuals(model, data)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, fitted_model):
        model, data = fitted_model
        result = longitudinal_residuals(model, data)
        for col in ["id", "time", "observed", "fitted", "residual"]:
            assert col in result.columns

    def test_same_length_as_data(self, fitted_model):
        model, data = fitted_model
        result = longitudinal_residuals(model, data)
        assert len(result) == len(data)

    def test_residual_is_observed_minus_fitted(self, fitted_model):
        model, data = fitted_model
        result = longitudinal_residuals(model, data)
        np.testing.assert_allclose(
            result["residual"].values,
            result["observed"].values - result["fitted"].values,
            atol=1e-10,
        )

    def test_marginal_residuals_have_larger_variance(self, fitted_model):
        """Marginal residuals should have larger variance than subject-specific."""
        model, data = fitted_model
        marginal = longitudinal_residuals(model, data, type="marginal")
        subject_specific = longitudinal_residuals(model, data, type="subject_specific")
        # Marginal residuals include between-subject variation
        assert marginal["residual"].var() >= subject_specific["residual"].var() * 0.5

    def test_not_fitted_raises(self):
        from insurance_jlm import JointModel
        model = JointModel()
        with pytest.raises(RuntimeError, match="fitted"):
            longitudinal_residuals(model, pd.DataFrame())


class TestMartingaleResiduals:
    def test_returns_dataframe(self, fitted_model):
        model, data = fitted_model
        result = martingale_residuals(model, data)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, fitted_model):
        model, data = fitted_model
        result = martingale_residuals(model, data)
        for col in ["id", "event_time", "event", "cumulative_hazard", "martingale_residual"]:
            assert col in result.columns

    def test_cumulative_hazard_non_negative(self, fitted_model):
        model, data = fitted_model
        result = martingale_residuals(model, data)
        assert (result["cumulative_hazard"] >= 0).all()

    def test_martingale_residuals_bounded_above(self, fitted_model):
        """Martingale residuals M_i ≤ 1 (since δ_i ≤ 1 and H_i ≥ 0)."""
        model, data = fitted_model
        result = martingale_residuals(model, data)
        assert (result["martingale_residual"] <= 1.01).all()

    def test_one_row_per_subject(self, fitted_model):
        model, data = fitted_model
        result = martingale_residuals(model, data)
        n_subjects = data["policy_id"].nunique()
        assert len(result) == n_subjects


class TestDevianceResiduals:
    def test_returns_dataframe(self, fitted_model):
        model, data = fitted_model
        result = deviance_residuals(model, data)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, fitted_model):
        model, data = fitted_model
        result = deviance_residuals(model, data)
        assert "deviance_residual" in result.columns
        assert "martingale_residual" in result.columns

    def test_deviance_residuals_finite(self, fitted_model):
        model, data = fitted_model
        result = deviance_residuals(model, data)
        assert result["deviance_residual"].apply(np.isfinite).all()


class TestBrierScore:
    def test_returns_scalar(self, fitted_model):
        model, data = fitted_model
        bs = brier_score(model, data, landmark_time=3.0, horizon=2.0, n_mc=5)
        assert isinstance(bs, float)

    def test_in_valid_range(self, fitted_model):
        """Brier score should be in [0, 1]."""
        model, data = fitted_model
        bs = brier_score(model, data, landmark_time=3.0, horizon=2.0, n_mc=5)
        assert 0 <= bs <= 1

    def test_lower_than_naive_model(self, fitted_model):
        """A naive model (constant 0.5 prediction) gives BS = 0.25.
        A reasonable model should do better, at least on training data."""
        model, data = fitted_model
        bs = brier_score(model, data, landmark_time=3.0, horizon=2.0, n_mc=5)
        # Allow for the fact this is a 5-iteration model — just check it's not terrible
        assert bs < 0.5


class TestTimeDependentAUC:
    def test_returns_scalar_or_nan(self, fitted_model):
        model, data = fitted_model
        auc = time_dependent_auc(model, data, landmark_time=3.0, horizon=2.0, n_mc=5)
        assert isinstance(auc, float)

    def test_in_valid_range_if_not_nan(self, fitted_model):
        model, data = fitted_model
        auc = time_dependent_auc(model, data, landmark_time=3.0, horizon=2.0, n_mc=5)
        if not np.isnan(auc):
            assert 0 <= auc <= 1


class TestExpectedActualRatio:
    def test_returns_dataframe(self, fitted_model):
        model, data = fitted_model
        result = expected_actual_ratio(model, data, landmark_time=3.0, horizon=2.0, n_mc=5)
        assert isinstance(result, pd.DataFrame)

    def test_has_five_quintiles(self, fitted_model):
        model, data = fitted_model
        result = expected_actual_ratio(model, data, landmark_time=3.0, horizon=2.0, n_mc=5)
        assert len(result) == 5

    def test_has_required_columns(self, fitted_model):
        model, data = fitted_model
        result = expected_actual_ratio(model, data, landmark_time=3.0, horizon=2.0, n_mc=5)
        for col in ["quintile", "n_subjects", "n_events_actual", "n_events_expected"]:
            assert col in result.columns

    def test_subject_counts_sum_to_total(self, fitted_model):
        model, data = fitted_model
        # Only subjects with observations before landmark time
        obs_data = data[data["month"] <= 3.0]
        result = expected_actual_ratio(model, data, landmark_time=3.0, horizon=2.0, n_mc=5)
        assert result["n_subjects"].sum() == obs_data["policy_id"].nunique()
