"""Tests for data loading and validation."""

import numpy as np
import pandas as pd
import pytest
from insurance_jlm.data import (
    make_synthetic_telematics,
    validate_long_format,
    summarise_data,
    DataValidationError,
)


class TestMakeSyntheticTelematics:
    def test_returns_two_dataframes(self):
        telem, claims = make_synthetic_telematics(n_subjects=50, random_state=0)
        assert isinstance(telem, pd.DataFrame)
        assert isinstance(claims, pd.DataFrame)

    def test_telematics_columns(self):
        telem, _ = make_synthetic_telematics(n_subjects=50, random_state=0)
        expected = {"policy_id", "month", "telematics_score", "age", "vehicle_age"}
        assert expected.issubset(set(telem.columns))

    def test_claims_columns(self):
        _, claims = make_synthetic_telematics(n_subjects=50, random_state=0)
        expected = {"policy_id", "claim_month", "had_claim", "age", "vehicle_age"}
        assert expected.issubset(set(claims.columns))

    def test_n_subjects_in_claims(self):
        telem, claims = make_synthetic_telematics(n_subjects=100, random_state=0)
        assert len(claims) == 100

    def test_event_indicator_binary(self):
        _, claims = make_synthetic_telematics(n_subjects=100, random_state=0)
        assert set(claims["had_claim"].unique()).issubset({0, 1})

    def test_claim_month_positive(self):
        _, claims = make_synthetic_telematics(n_subjects=100, random_state=0)
        assert (claims["claim_month"] > 0).all()

    def test_telematics_scores_in_reasonable_range(self):
        telem, _ = make_synthetic_telematics(n_subjects=200, random_state=0)
        assert telem["telematics_score"].between(30, 120).mean() > 0.95

    def test_month_col_positive_integer(self):
        telem, _ = make_synthetic_telematics(n_subjects=50, random_state=0)
        assert (telem["month"] >= 1).all()

    def test_reproducibility(self):
        t1, c1 = make_synthetic_telematics(n_subjects=50, random_state=7)
        t2, c2 = make_synthetic_telematics(n_subjects=50, random_state=7)
        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(c1, c2)

    def test_different_seeds_differ(self):
        t1, _ = make_synthetic_telematics(n_subjects=50, random_state=1)
        t2, _ = make_synthetic_telematics(n_subjects=50, random_state=2)
        assert not t1["telematics_score"].equals(t2["telematics_score"])

    def test_max_months_respected(self):
        _, claims = make_synthetic_telematics(n_subjects=200, max_months=6, random_state=0)
        assert (claims["claim_month"] <= 6.1).all()

    def test_no_missing_values(self):
        telem, claims = make_synthetic_telematics(n_subjects=100, random_state=0)
        assert telem.isna().sum().sum() == 0
        assert claims.isna().sum().sum() == 0


class TestValidateLongFormat:
    @pytest.fixture
    def valid_data(self):
        telem, claims = make_synthetic_telematics(n_subjects=100, random_state=0)
        data = telem.merge(
            claims[["policy_id", "claim_month", "had_claim"]], on="policy_id"
        )
        return data

    def test_valid_data_returns_no_errors(self, valid_data):
        warnings = validate_long_format(
            valid_data, "policy_id", "month", "telematics_score",
            "claim_month", "had_claim", ["age"], ["age", "vehicle_age"]
        )
        assert isinstance(warnings, list)

    def test_missing_column_raises(self, valid_data):
        with pytest.raises(DataValidationError, match="Missing columns"):
            validate_long_format(
                valid_data, "policy_id", "month", "nonexistent_col",
                "claim_month", "had_claim", [], []
            )

    def test_non_binary_event_raises(self, valid_data):
        data = valid_data.copy()
        data["had_claim"] = data["had_claim"] * 2  # Now has value 2
        with pytest.raises(DataValidationError, match="0 and 1"):
            validate_long_format(
                data, "policy_id", "month", "telematics_score",
                "claim_month", "had_claim", [], []
            )

    def test_varying_event_time_raises(self, valid_data):
        data = valid_data.copy()
        # Manually corrupt one subject's event time
        first_id = data["policy_id"].iloc[0]
        data.loc[data["policy_id"] == first_id, "claim_month"] = [1.0, 2.0, 3.0][
            :len(data[data["policy_id"] == first_id])
        ]
        with pytest.raises(DataValidationError, match="varies within subjects"):
            validate_long_format(
                data, "policy_id", "month", "telematics_score",
                "claim_month", "had_claim", [], []
            )

    def test_too_few_subjects_raises(self, valid_data):
        tiny = valid_data[valid_data["policy_id"].isin(
            valid_data["policy_id"].unique()[:20]
        )]
        with pytest.raises(DataValidationError, match="subjects"):
            validate_long_format(
                tiny, "policy_id", "month", "telematics_score",
                "claim_month", "had_claim", [], []
            )

    def test_negative_time_raises(self, valid_data):
        data = valid_data.copy()
        data.loc[data.index[0], "month"] = -1.0
        with pytest.raises(DataValidationError, match="negative"):
            validate_long_format(
                data, "policy_id", "month", "telematics_score",
                "claim_month", "had_claim", [], []
            )

    def test_missing_y_values_raises(self, valid_data):
        data = valid_data.copy()
        data.loc[data.index[0], "telematics_score"] = np.nan
        with pytest.raises(DataValidationError, match="missing values"):
            validate_long_format(
                data, "policy_id", "month", "telematics_score",
                "claim_month", "had_claim", [], []
            )


class TestSummariseData:
    def test_returns_dataframe(self):
        telem, claims = make_synthetic_telematics(n_subjects=100, random_state=0)
        data = telem.merge(claims[["policy_id", "claim_month", "had_claim"]], on="policy_id")
        result = summarise_data(data, "policy_id", "month", "telematics_score",
                                "claim_month", "had_claim")
        assert isinstance(result, pd.DataFrame)

    def test_contains_n_subjects(self):
        telem, claims = make_synthetic_telematics(n_subjects=100, random_state=0)
        data = telem.merge(claims[["policy_id", "claim_month", "had_claim"]], on="policy_id")
        result = summarise_data(data, "policy_id", "month", "telematics_score",
                                "claim_month", "had_claim")
        n_subjects_row = result[result["statistic"] == "N subjects"]
        assert int(n_subjects_row["value"].iloc[0]) == 100
