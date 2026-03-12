"""Tests for convenience loader functions."""

import numpy as np
import pandas as pd
import pytest
from insurance_jlm import JointModel
from insurance_jlm.data import jlm_from_telematics, jlm_from_ncd, make_synthetic_telematics


class TestJlmFromTelematics:
    @pytest.fixture
    def telematics_data(self):
        telem, claims = make_synthetic_telematics(n_subjects=150, random_state=3)
        return telem, claims

    def test_returns_joint_model(self, telematics_data):
        telem, claims = telematics_data
        model = jlm_from_telematics(
            telem, claims,
            model_kwargs={"max_iter": 2, "se_method": "none",
                          "n_quad_points": 3, "random_state": 0},
        )
        assert isinstance(model, JointModel)

    def test_model_has_fitted_submodels(self, telematics_data):
        telem, claims = telematics_data
        model = jlm_from_telematics(
            telem, claims,
            model_kwargs={"max_iter": 2, "se_method": "none",
                          "n_quad_points": 3, "random_state": 0},
        )
        assert model.long_submodel_ is not None
        assert model.surv_submodel_ is not None

    def test_with_covariates(self, telematics_data):
        telem, claims = telematics_data
        model = jlm_from_telematics(
            telem, claims,
            long_covariates=["age"],
            surv_covariates=["vehicle_age"],
            model_kwargs={"max_iter": 2, "se_method": "none",
                          "n_quad_points": 3, "random_state": 0},
        )
        assert isinstance(model, JointModel)

    def test_association_summary_works(self, telematics_data):
        telem, claims = telematics_data
        model = jlm_from_telematics(
            telem, claims,
            model_kwargs={"max_iter": 2, "se_method": "none",
                          "n_quad_points": 3, "random_state": 0},
        )
        s = model.association_summary()
        assert len(s) == 1


class TestJlmFromNCD:
    @pytest.fixture
    def ncd_data(self):
        """Synthetic NCD + lapse data."""
        rng = np.random.default_rng(4)
        n = 150
        ncd_rows = []
        lapse_rows = []

        for i in range(n):
            ncd_start = rng.integers(0, 5)
            n_renewals = rng.integers(2, 8)
            lapse_renewal = rng.integers(2, n_renewals + 1)
            lapsed = 1 if rng.uniform() < 0.3 else 0
            age = rng.uniform(20, 65)

            for r in range(1, n_renewals + 1):
                ncd = min(9, ncd_start + r)  # NCD accrues
                ncd_rows.append({
                    "policy_id": f"NCD{i:04d}",
                    "renewal_number": float(r),
                    "ncd_level": float(ncd),
                    "age": age,
                })

            lapse_rows.append({
                "policy_id": f"NCD{i:04d}",
                "lapse_renewal": float(lapse_renewal),
                "lapsed": lapsed,
                "age": age,
            })

        return pd.DataFrame(ncd_rows), pd.DataFrame(lapse_rows)

    def test_returns_joint_model(self, ncd_data):
        ncd_df, lapse_df = ncd_data
        model = jlm_from_ncd(
            ncd_df, lapse_df,
            model_kwargs={"max_iter": 2, "se_method": "none",
                          "n_quad_points": 3, "random_state": 0},
        )
        assert isinstance(model, JointModel)

    def test_column_names_respected(self, ncd_data):
        ncd_df, lapse_df = ncd_data
        # Rename columns to test custom naming
        ncd_renamed = ncd_df.rename(columns={
            "policy_id": "pol_ref",
            "renewal_number": "renewal",
            "ncd_level": "ncd",
        })
        lapse_renamed = lapse_df.rename(columns={
            "policy_id": "pol_ref",
            "lapse_renewal": "lapse_time",
            "lapsed": "is_lapsed",
        })
        model = jlm_from_ncd(
            ncd_renamed, lapse_renamed,
            policy_col="pol_ref",
            renewal_col="renewal",
            ncd_col="ncd",
            lapse_renewal_col="lapse_time",
            lapse_indicator_col="is_lapsed",
            model_kwargs={"max_iter": 2, "se_method": "none",
                          "n_quad_points": 3, "random_state": 0},
        )
        assert isinstance(model, JointModel)
