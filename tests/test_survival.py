"""Tests for the survival sub-model."""

import numpy as np
import pandas as pd
import pytest
from insurance_jlm.models.survival import SurvivalSubmodel, SurvivalParams


@pytest.fixture
def survival_data():
    """Simple survival dataset: 100 subjects."""
    rng = np.random.default_rng(20)
    n = 100
    ages = rng.uniform(20, 60, n)
    event_times = rng.exponential(8.0, n).clip(0.5, 12.0)
    events = (event_times < 12.0).astype(int)
    marker_vals = 70 - 0.5 * event_times + rng.normal(0, 2, n)

    data = pd.DataFrame({
        "id": [f"P{i:04d}" for i in range(n)],
        "event_time": event_times,
        "event": events,
        "age": ages,
    })

    # Store true marker values at event times
    marker_map = dict(zip(data["id"], marker_vals))

    def marker_func(subj_id, t):
        return marker_map.get(subj_id, 70.0)

    return data, marker_func


class TestSurvivalSubmodelInit:
    def test_default_init(self):
        m = SurvivalSubmodel()
        assert m.association == "current_value"

    def test_slope_association(self):
        m = SurvivalSubmodel("slope")
        assert m.association == "slope"

    def test_invalid_association_raises(self):
        with pytest.raises(ValueError, match="association"):
            SurvivalSubmodel("invalid")

    def test_params_none_before_fit(self):
        m = SurvivalSubmodel()
        assert m.params_ is None


class TestSurvivalSubmodelFit:
    def test_fit_returns_self(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        result = m.fit(data, "event_time", "event", ["age"], marker_func)
        assert result is m

    def test_params_not_none_after_fit(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        assert m.params_ is not None

    def test_gamma_shape(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        assert len(m.params_.gamma) == 1  # one covariate: age

    def test_alpha_is_scalar(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        assert isinstance(m.params_.alpha, float)

    def test_baseline_hazard_positive(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        assert (m.params_.baseline_hazard >= 0).all()

    def test_baseline_times_ascending(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        assert np.all(np.diff(m.params_.baseline_times) >= 0)

    def test_no_covariates(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", [], marker_func)
        assert len(m.params_.gamma) == 0


class TestCumulativeHazard:
    def test_returns_non_negative(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        row = data.iloc[0]
        x_i = np.array([row["age"]])

        def mf(t):
            return marker_func(row["id"], t)

        H = m.cumulative_hazard(5.0, x_i, mf)
        assert H >= 0

    def test_increases_with_time(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        row = data.iloc[0]
        x_i = np.array([row["age"]])

        def mf(t):
            return marker_func(row["id"], t)

        H1 = m.cumulative_hazard(3.0, x_i, mf)
        H5 = m.cumulative_hazard(5.0, x_i, mf)
        assert H5 >= H1

    def test_zero_at_t_zero(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        row = data.iloc[0]
        x_i = np.array([row["age"]])

        def mf(t):
            return marker_func(row["id"], t)

        H = m.cumulative_hazard(0.0, x_i, mf)
        assert H == 0.0


class TestSurvival:
    def test_returns_in_0_1(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        row = data.iloc[0]
        x_i = np.array([row["age"]])

        def mf(t):
            return marker_func(row["id"], t)

        S = m.survival(5.0, x_i, mf)
        assert 0 <= S <= 1

    def test_decreases_with_time(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        row = data.iloc[0]
        x_i = np.array([row["age"]])

        def mf(t):
            return marker_func(row["id"], t)

        S3 = m.survival(3.0, x_i, mf)
        S8 = m.survival(8.0, x_i, mf)
        assert S8 <= S3

    def test_survival_at_zero_is_one(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        row = data.iloc[0]
        x_i = np.array([row["age"]])

        def mf(t):
            return marker_func(row["id"], t)

        S = m.survival(0.0, x_i, mf)
        assert abs(S - 1.0) < 1e-10


class TestSurvivalSummary:
    def test_summary_returns_dataframe(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        s = m.summary()
        assert isinstance(s, pd.DataFrame)

    def test_summary_contains_alpha(self, survival_data):
        data, marker_func = survival_data
        m = SurvivalSubmodel()
        m.fit(data, "event_time", "event", ["age"], marker_func)
        s = m.summary()
        assert any("alpha" in str(p).lower() for p in s["parameter"])
