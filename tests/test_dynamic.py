"""Tests for DynamicPredictor and LandmarkPredictor."""

import numpy as np
import pandas as pd
import pytest
from insurance_jlm import JointModel, DynamicPredictor, LandmarkPredictor
from insurance_jlm.data import make_synthetic_telematics


class TestDynamicPredictor:
    @pytest.fixture
    def predictor_and_model(self, fitted_model):
        model, data = fitted_model
        subj_id = data["policy_id"].unique()[0]
        initial_data = data[data["policy_id"] == subj_id].copy()
        predictor = DynamicPredictor(model, subj_id, initial_data, n_mc=10)
        return predictor, model, subj_id, data

    def test_init(self, predictor_and_model):
        predictor, model, subj_id, _ = predictor_and_model
        assert predictor.subject_id == subj_id
        assert predictor.model is model

    def test_update_returns_float(self, predictor_and_model):
        predictor, _, _, data = predictor_and_model
        result = predictor.update({"month": 9.0, "telematics_score": 68.0,
                                   "age": 30.0, "vehicle_age": 5.0})
        assert isinstance(result, float)

    def test_update_survival_in_0_1(self, predictor_and_model):
        predictor, _, _, _ = predictor_and_model
        result = predictor.update({"month": 9.0, "telematics_score": 68.0,
                                   "age": 30.0, "vehicle_age": 5.0})
        assert 0 <= result <= 1

    def test_risk_trajectory_empty_before_update(self, predictor_and_model):
        predictor, _, _, _ = predictor_and_model
        # Create fresh predictor with no updates
        model, data = predictor_and_model[1], predictor_and_model[3]
        subj_id = data["policy_id"].unique()[1]
        initial_data = data[data["policy_id"] == subj_id].copy()
        fresh = DynamicPredictor(model, subj_id, initial_data, n_mc=10)
        traj = fresh.risk_trajectory()
        assert len(traj) == 0

    def test_risk_trajectory_after_updates(self, predictor_and_model):
        model, data = predictor_and_model[1], predictor_and_model[3]
        subj_id = data["policy_id"].unique()[2]
        initial_data = data[data["policy_id"] == subj_id].copy()
        pred = DynamicPredictor(model, subj_id, initial_data, n_mc=5)
        pred.update({"month": 8.0, "telematics_score": 72.0,
                     "age": 25.0, "vehicle_age": 3.0})
        pred.update({"month": 9.0, "telematics_score": 69.0,
                     "age": 25.0, "vehicle_age": 3.0})
        traj = pred.risk_trajectory()
        assert len(traj) == 2

    def test_risk_trajectory_columns(self, predictor_and_model):
        predictor, _, _, _ = predictor_and_model
        predictor.update({"month": 10.0, "telematics_score": 65.0,
                          "age": 30.0, "vehicle_age": 5.0})
        traj = predictor.risk_trajectory()
        assert "time" in traj.columns
        assert "survival_prob_1_unit" in traj.columns

    def test_predict_survival_returns_float(self, predictor_and_model):
        predictor, _, _, _ = predictor_and_model
        result = predictor.predict_survival(horizon=2.0)
        assert isinstance(result, float)

    def test_predict_survival_in_0_1(self, predictor_and_model):
        predictor, _, _, _ = predictor_and_model
        result = predictor.predict_survival(horizon=2.0)
        assert 0 <= result <= 1

    def test_history_grows_after_update(self, predictor_and_model):
        model, data = predictor_and_model[1], predictor_and_model[3]
        subj_id = data["policy_id"].unique()[3]
        initial_data = data[data["policy_id"] == subj_id].copy()
        initial_len = len(initial_data)
        pred = DynamicPredictor(model, subj_id, initial_data, n_mc=5)
        pred.update({"month": 99.0, "telematics_score": 70.0,
                     "age": 30.0, "vehicle_age": 5.0})
        assert len(pred.history) == initial_len + 1


class TestLandmarkPredictor:
    @pytest.fixture
    def landmark_data(self):
        telem, claims = make_synthetic_telematics(n_subjects=200, random_state=5)
        data = telem.merge(
            claims[["policy_id", "claim_month", "had_claim"]], on="policy_id"
        )
        return data

    def test_fit_returns_self(self, landmark_data):
        lm = LandmarkPredictor(landmark_times=[3.0, 6.0], window=3.0)
        result = lm.fit(
            landmark_data, "policy_id", "month", "telematics_score",
            "claim_month", "had_claim", ["age"],
        )
        assert result is lm

    def test_models_fitted_at_landmarks(self, landmark_data):
        lm = LandmarkPredictor(landmark_times=[3.0, 6.0], window=3.0)
        lm.fit(
            landmark_data, "policy_id", "month", "telematics_score",
            "claim_month", "had_claim", ["age"],
        )
        assert len(lm._landmark_models_) > 0

    def test_predict_returns_dataframe(self, landmark_data):
        lm = LandmarkPredictor(landmark_times=[3.0, 6.0], window=3.0)
        lm.fit(
            landmark_data, "policy_id", "month", "telematics_score",
            "claim_month", "had_claim", ["age"],
        )
        result = lm.predict(
            landmark_data, "policy_id", "month", "telematics_score",
            "claim_month", "had_claim", ["age"],
            landmark_time=3.0,
        )
        assert isinstance(result, pd.DataFrame)

    def test_predict_survival_in_0_1(self, landmark_data):
        lm = LandmarkPredictor(landmark_times=[3.0], window=3.0)
        lm.fit(
            landmark_data, "policy_id", "month", "telematics_score",
            "claim_month", "had_claim", ["age"],
        )
        result = lm.predict(
            landmark_data, "policy_id", "month", "telematics_score",
            "claim_month", "had_claim", ["age"],
            landmark_time=3.0,
        )
        if len(result) > 0:
            assert (result["survival_prob"] >= 0).all()
            assert (result["survival_prob"] <= 1).all()

    def test_predict_without_fit_raises(self):
        lm = LandmarkPredictor(landmark_times=[3.0], window=3.0)
        with pytest.raises(RuntimeError, match="fitted"):
            lm.predict(
                pd.DataFrame(), "id", "time", "y", "T", "event", [],
                landmark_time=3.0,
            )
