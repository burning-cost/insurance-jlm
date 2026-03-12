"""Tests for the longitudinal sub-model."""

import numpy as np
import pandas as pd
import pytest
from insurance_jlm.models.longitudinal import LongitudinalSubmodel, LongitudinalParams


@pytest.fixture
def simple_long_data():
    """Simple longitudinal dataset: 100 subjects, 4 obs each."""
    rng = np.random.default_rng(10)
    n = 100
    rows = []
    for i in range(n):
        b0 = rng.normal(0, 2)
        b1 = rng.normal(0, 0.5)
        age = rng.uniform(20, 60)
        for m in range(1, 5):
            score = 70 + 0.1 * age - 0.5 * m + b0 + b1 * m + rng.normal(0, 2)
            rows.append({
                "id": f"P{i:04d}", "month": float(m),
                "score": score, "age": age,
            })
    return pd.DataFrame(rows)


class TestLongitudinalSubmodelInit:
    def test_default_init(self):
        m = LongitudinalSubmodel()
        assert m.long_model == "linear"

    def test_quadratic_init(self):
        m = LongitudinalSubmodel("quadratic")
        assert m.long_model == "quadratic"

    def test_intercept_only_init(self):
        m = LongitudinalSubmodel("intercept")
        assert m.long_model == "intercept"

    def test_invalid_long_model_raises(self):
        with pytest.raises(ValueError, match="long_model"):
            LongitudinalSubmodel("cubic")

    def test_params_none_before_fit(self):
        m = LongitudinalSubmodel()
        assert m.params_ is None


class TestLongitudinalSubmodelFit:
    def test_fit_returns_self(self, simple_long_data):
        m = LongitudinalSubmodel()
        result = m.fit(simple_long_data, "id", "month", "score", [])
        assert result is m

    def test_params_not_none_after_fit(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        assert m.params_ is not None

    def test_beta_shape(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", ["age"])
        # intercept + month + age = 3
        assert len(m.params_.beta) == 3

    def test_D_shape_linear(self, simple_long_data):
        m = LongitudinalSubmodel("linear")
        m.fit(simple_long_data, "id", "month", "score", [])
        assert m.params_.D.shape == (2, 2)

    def test_D_shape_intercept(self, simple_long_data):
        m = LongitudinalSubmodel("intercept")
        m.fit(simple_long_data, "id", "month", "score", [])
        assert m.params_.D.shape == (1, 1)

    def test_sigma2_positive(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        assert m.params_.sigma2 > 0

    def test_D_positive_definite(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        eigvals = np.linalg.eigvalsh(m.params_.D)
        assert (eigvals > 0).all()

    def test_intercept_roughly_correct(self, simple_long_data):
        """True intercept is approximately 70."""
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        # statsmodels uses 'Intercept' as name
        intercept = m.params_.beta[0]
        assert abs(intercept - 70) < 15  # Wide tolerance — small n

    def test_quadratic_fit_has_more_params(self, simple_long_data):
        m_linear = LongitudinalSubmodel("linear")
        m_quad = LongitudinalSubmodel("quadratic")
        m_linear.fit(simple_long_data, "id", "month", "score", [])
        m_quad.fit(simple_long_data, "id", "month", "score", [])
        assert len(m_quad.params_.beta) == len(m_linear.params_.beta) + 1


class TestMarkerValue:
    def test_marker_value_returns_scalar(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        row = simple_long_data.iloc[0]
        b_i = np.array([0.0, 0.0])
        result = m.marker_value(1.0, b_i, row, "month", [])
        assert isinstance(result, float)

    def test_marker_value_changes_with_b(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        row = simple_long_data.iloc[0]
        v1 = m.marker_value(1.0, np.array([0.0, 0.0]), row, "month", [])
        v2 = m.marker_value(1.0, np.array([5.0, 0.0]), row, "month", [])
        assert abs(v2 - v1 - 5.0) < 1e-8

    def test_marker_value_changes_with_time(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        row = simple_long_data.iloc[0]
        b_i = np.zeros(2)
        v1 = m.marker_value(1.0, b_i, row, "month", [])
        v5 = m.marker_value(5.0, b_i, row, "month", [])
        # Time coefficient should be negative (true β_time = -0.5)
        assert v5 < v1

    def test_random_effects_shift_trajectory(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        row = simple_long_data.iloc[0]
        b_zero = np.array([0.0, 0.0])
        b_pos = np.array([10.0, 0.0])
        v0 = m.marker_value(3.0, b_zero, row, "month", [])
        vp = m.marker_value(3.0, b_pos, row, "month", [])
        assert abs(vp - v0 - 10.0) < 1e-8


class TestGetRandomEffects:
    def test_returns_dataframe(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        re = m.get_random_effects(simple_long_data, "id")
        assert isinstance(re, pd.DataFrame)

    def test_index_is_subject_id(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        re = m.get_random_effects(simple_long_data, "id")
        expected_ids = set(simple_long_data["id"].unique())
        assert set(re.index) == expected_ids

    def test_blups_have_zero_mean(self, simple_long_data):
        """BLUPs should have approximately zero mean (they are deviations)."""
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        re = m.get_random_effects(simple_long_data, "id")
        # Allow wide tolerance — small sample
        assert abs(re.iloc[:, 0].mean()) < 2.0


class TestLongitudinalSummary:
    def test_summary_returns_dataframe(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        s = m.summary()
        assert isinstance(s, pd.DataFrame)

    def test_summary_has_required_columns(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        s = m.summary()
        for col in ["parameter", "estimate", "std_err", "z_stat", "p_value"]:
            assert col in s.columns

    def test_std_err_positive(self, simple_long_data):
        m = LongitudinalSubmodel()
        m.fit(simple_long_data, "id", "month", "score", [])
        s = m.summary()
        assert (s["std_err"] > 0).all()

    def test_not_fitted_raises(self):
        m = LongitudinalSubmodel()
        with pytest.raises(RuntimeError, match="fitted"):
            m.summary()
