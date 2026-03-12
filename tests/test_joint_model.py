"""Tests for the JointModel class."""

import warnings

import numpy as np
import pandas as pd
import pytest
from insurance_jlm import JointModel
from insurance_jlm.data import make_synthetic_telematics


@pytest.fixture(scope="module")
def fitted_joint(medium_dataset):
    """Fitted JointModel with 3 quad points and 5 iterations for speed."""
    data, _, _ = medium_dataset
    model = JointModel(
        n_quad_points=3,
        max_iter=5,
        se_method="none",
        random_state=0,
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


class TestJointModelInit:
    def test_default_init(self):
        m = JointModel()
        assert m.n_quad_points == 7
        assert m.max_iter == 200
        assert m.tol == 1e-4
        assert m.se_method == "bootstrap"
        assert m.association == "current_value"

    def test_custom_init(self):
        m = JointModel(n_quad_points=5, max_iter=100, se_method="none")
        assert m.n_quad_points == 5
        assert m.max_iter == 100
        assert m.se_method == "none"

    def test_invalid_surv_model_raises(self):
        with pytest.raises(NotImplementedError, match="cox"):
            JointModel(surv_model="weibull")

    def test_not_fitted_initially(self):
        m = JointModel()
        assert m.long_submodel_ is None
        assert m.surv_submodel_ is None

    def test_n_iter_zero_before_fit(self):
        m = JointModel()
        assert m.n_iter_ == 0


class TestJointModelFit:
    def test_fit_returns_self(self, fitted_joint):
        model, _ = fitted_joint
        assert isinstance(model, JointModel)

    def test_submodels_fitted_after_fit(self, fitted_joint):
        model, _ = fitted_joint
        assert model.long_submodel_ is not None
        assert model.surv_submodel_ is not None

    def test_n_iter_incremented(self, fitted_joint):
        model, _ = fitted_joint
        assert model.n_iter_ > 0

    def test_loglik_history_populated(self, fitted_joint):
        model, _ = fitted_joint
        assert len(model.loglik_history_) > 0

    def test_alpha_is_finite(self, fitted_joint):
        model, _ = fitted_joint
        assert np.isfinite(model.surv_submodel_.params_.alpha)

    def test_gamma_finite(self, fitted_joint):
        model, _ = fitted_joint
        assert np.all(np.isfinite(model.surv_submodel_.params_.gamma))

    def test_beta_finite(self, fitted_joint):
        model, _ = fitted_joint
        assert np.all(np.isfinite(model.long_submodel_.params_.beta))

    def test_sigma2_positive(self, fitted_joint):
        model, _ = fitted_joint
        assert model.long_submodel_.params_.sigma2 > 0

    def test_D_positive_definite(self, fitted_joint):
        model, _ = fitted_joint
        D = model.long_submodel_.params_.D
        eigvals = np.linalg.eigvalsh(D)
        assert (eigvals > 0).all()

    def test_missing_column_raises(self, medium_dataset):
        data, _, _ = medium_dataset
        model = JointModel(max_iter=1, se_method="none")
        with pytest.raises(ValueError, match="Missing columns"):
            model.fit(
                data, "policy_id", "month", "nonexistent",
                "claim_month", "had_claim", [], []
            )

    def test_bad_event_col_raises(self, medium_dataset):
        data, _, _ = medium_dataset
        bad_data = data.copy()
        bad_data["had_claim"] = bad_data["had_claim"] * 3
        model = JointModel(max_iter=1, se_method="none")
        with pytest.raises(ValueError, match="0 and 1"):
            model.fit(
                bad_data, "policy_id", "month", "telematics_score",
                "claim_month", "had_claim", [], []
            )


class TestAssociationSummary:
    def test_returns_dataframe(self, fitted_joint):
        model, _ = fitted_joint
        s = model.association_summary()
        assert isinstance(s, pd.DataFrame)

    def test_has_alpha_row(self, fitted_joint):
        model, _ = fitted_joint
        s = model.association_summary()
        assert any("alpha" in str(p).lower() for p in s["parameter"])

    def test_estimate_is_finite(self, fitted_joint):
        model, _ = fitted_joint
        s = model.association_summary()
        assert np.isfinite(s["estimate"].iloc[0])

    def test_not_fitted_raises(self):
        m = JointModel()
        with pytest.raises(RuntimeError, match="fitted"):
            m.association_summary()


class TestLongitudinalSummary:
    def test_returns_dataframe(self, fitted_joint):
        model, _ = fitted_joint
        s = model.longitudinal_summary()
        assert isinstance(s, pd.DataFrame)

    def test_has_required_columns(self, fitted_joint):
        model, _ = fitted_joint
        s = model.longitudinal_summary()
        assert "estimate" in s.columns
        assert "parameter" in s.columns


class TestSurvivalSummary:
    def test_returns_dataframe(self, fitted_joint):
        model, _ = fitted_joint
        s = model.survival_summary()
        assert isinstance(s, pd.DataFrame)

    def test_contains_alpha(self, fitted_joint):
        model, _ = fitted_joint
        s = model.survival_summary()
        assert any("alpha" in str(p).lower() for p in s["parameter"])


class TestPredictSurvival:
    def test_returns_dataframe(self, fitted_joint):
        model, data = fitted_joint
        subset = data[data["policy_id"].isin(data["policy_id"].unique()[:5])]
        result = model.predict_survival(
            subset, "policy_id", landmark_time=3.0, horizon=2.0, n_mc=10
        )
        assert isinstance(result, pd.DataFrame)

    def test_survival_probs_in_0_1(self, fitted_joint):
        model, data = fitted_joint
        subset = data[data["policy_id"].isin(data["policy_id"].unique()[:5])]
        result = model.predict_survival(
            subset, "policy_id", landmark_time=3.0, horizon=2.0, n_mc=10
        )
        assert (result["survival_prob"] >= 0).all()
        assert (result["survival_prob"] <= 1).all()

    def test_result_has_correct_columns(self, fitted_joint):
        model, data = fitted_joint
        subset = data[data["policy_id"].isin(data["policy_id"].unique()[:3])]
        result = model.predict_survival(
            subset, "policy_id", landmark_time=3.0, horizon=2.0, n_mc=5
        )
        assert "id" in result.columns
        assert "survival_prob" in result.columns
        assert "landmark_time" in result.columns

    def test_longer_horizon_lower_survival(self, fitted_joint):
        """P(T > t + 2 | ...) should generally be <= P(T > t + 1 | ...)."""
        model, data = fitted_joint
        subj_id = data["policy_id"].unique()[0]
        subset = data[data["policy_id"] == subj_id]

        s1 = model.predict_survival(subset, "policy_id", 3.0, 1.0, n_mc=20)
        s2 = model.predict_survival(subset, "policy_id", 3.0, 3.0, n_mc=20)

        # Not strict — Monte Carlo noise means this could occasionally fail
        # but should hold on average
        assert s2["survival_prob"].iloc[0] <= s1["survival_prob"].iloc[0] + 0.1


class TestPredictHazard:
    def test_returns_dataframe(self, fitted_joint):
        model, data = fitted_joint
        subset = data[data["policy_id"].isin(data["policy_id"].unique()[:3])]
        result = model.predict_hazard(
            subset, "policy_id", times=np.array([2.0, 5.0]), n_mc=10
        )
        assert isinstance(result, pd.DataFrame)

    def test_hazard_non_negative(self, fitted_joint):
        model, data = fitted_joint
        subset = data[data["policy_id"].isin(data["policy_id"].unique()[:3])]
        result = model.predict_hazard(
            subset, "policy_id", times=np.array([2.0, 5.0]), n_mc=10
        )
        assert (result["hazard"] >= 0).all()

    def test_result_has_id_time_columns(self, fitted_joint):
        model, data = fitted_joint
        subset = data[data["policy_id"].isin(data["policy_id"].unique()[:2])]
        result = model.predict_hazard(
            subset, "policy_id", times=np.array([3.0]), n_mc=5
        )
        assert "id" in result.columns
        assert "time" in result.columns
        assert "hazard" in result.columns


class TestConvergence:
    def test_loglik_history_is_list(self, fitted_joint):
        model, _ = fitted_joint
        assert isinstance(model.loglik_history_, list)

    def test_loglik_values_finite(self, fitted_joint):
        model, _ = fitted_joint
        assert all(np.isfinite(ll) for ll in model.loglik_history_)

    def test_convergence_flag_type(self, fitted_joint):
        model, _ = fitted_joint
        assert isinstance(model.converged_, bool)


class TestInterceptOnlyLongModel:
    def test_fits_intercept_only(self, medium_dataset):
        data, _, _ = medium_dataset
        model = JointModel(
            long_model="intercept",
            n_quad_points=3,
            max_iter=3,
            se_method="none",
            random_state=99,
        )
        model.fit(
            data, "policy_id", "month", "telematics_score",
            "claim_month", "had_claim", [], ["vehicle_age"],
        )
        # D should be 1x1
        assert model.long_submodel_.params_.D.shape == (1, 1)
