"""Joint model: EM algorithm linking longitudinal and survival sub-models.

The Wulfsohn-Tsiatis shared random effects model (Biometrics, 1997) alternates:

E-step: compute E[b_i | data, θ] and E[b_i b_i' | data, θ] for each subject
        using Gauss-Hermite quadrature to approximate the posterior.

M-step: update all model parameters using the expected sufficient statistics
        from the E-step.

The key convergence diagnostic is the observed-data log-likelihood. We monitor
its change across iterations; convergence when |ΔLL| / (1 + |LL|) < tol.

Bootstrap standard errors: we resample subjects (not observations) and refit
the full joint model, then take the empirical standard deviation of estimates
across bootstrap resamples. This correctly handles the within-subject
correlation in the longitudinal data.

Scalability note: the EM E-step loops over all subjects × quadrature points.
At n=10,000 subjects, this takes 2-5 minutes. At n=100,000, expect 20-50
minutes. For production UK telematics (1M+ policies), you need distributed
computation (Spark) or a stochastic EM approximation — neither is in scope
for v0.1. The Databricks demo notebook shows the parallelisation pattern.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .longitudinal import LongitudinalSubmodel, LongitudinalParams
from .survival import SurvivalSubmodel, SurvivalParams
from .quadrature import gauss_hermite_points, product_rule_2d, posterior_covariance_approx


class JointModel:
    """Wulfsohn-Tsiatis Shared Random Effects Joint Model.

    Fits a linear mixed-effects longitudinal sub-model and a Cox PH survival
    sub-model simultaneously via the EM algorithm with Gauss-Hermite quadrature.

    Parameters
    ----------
    long_model:
        Functional form for time in the longitudinal model.
        'linear' (default), 'quadratic', or 'intercept'.
    surv_model:
        Survival sub-model type. Only 'cox' is implemented in v0.1.
    association:
        How the longitudinal trajectory enters the hazard.
        'current_value' (default): h_i(t) = h₀(t) exp(γ'x_i + α·m_i(t))
        'slope': uses dm_i/dt as the time-varying covariate
        'area': uses ∫₀ᵗ m_i(s) ds
    n_quad_points:
        Gauss-Hermite quadrature points per random-effect dimension.
        7 is accurate for most problems; increase to 11 if you suspect
        numerical issues with highly non-Gaussian posteriors.
    max_iter:
        Maximum EM iterations.
    tol:
        Convergence tolerance on normalised log-likelihood change.
    se_method:
        Standard error method. 'bootstrap' (default, 50 resamples) or
        'none' (faster, no SEs).
    n_bootstrap:
        Bootstrap resamples for SE estimation. Ignored if se_method='none'.
    random_state:
        Seed for reproducibility.

    Examples
    --------
    >>> model = JointModel(n_quad_points=7, random_state=42)
    >>> model.fit(
    ...     data=df,
    ...     id_col='policy_id',
    ...     time_col='month',
    ...     y_col='telematics_score',
    ...     event_time_col='claim_month',
    ...     event_col='had_claim',
    ...     long_covariates=['age', 'vehicle_age'],
    ...     surv_covariates=['age', 'vehicle_class_hatchback'],
    ... )
    >>> model.association_summary()
    """

    def __init__(
        self,
        long_model: str = "linear",
        surv_model: str = "cox",
        association: str = "current_value",
        n_quad_points: int = 7,
        max_iter: int = 200,
        tol: float = 1e-4,
        se_method: str = "bootstrap",
        n_bootstrap: int = 50,
        random_state: Optional[int] = None,
    ) -> None:
        if surv_model != "cox":
            raise NotImplementedError(
                "Only 'cox' is implemented in v0.1. Weibull and Gompertz "
                "parametric options are planned for v0.2."
            )
        self.long_model = long_model
        self.surv_model = surv_model
        self.association = association
        self.n_quad_points = n_quad_points
        self.max_iter = max_iter
        self.tol = tol
        self.se_method = se_method
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)

        # Set after fit()
        self.long_submodel_: Optional[LongitudinalSubmodel] = None
        self.surv_submodel_: Optional[SurvivalSubmodel] = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.loglik_history_: list[float] = []

        # Column names stored at fit time
        self._id_col: Optional[str] = None
        self._time_col: Optional[str] = None
        self._y_col: Optional[str] = None
        self._event_time_col: Optional[str] = None
        self._event_col: Optional[str] = None
        self._long_covariates: Optional[list[str]] = None
        self._surv_covariates: Optional[list[str]] = None

        # Bootstrap SE storage
        self._bootstrap_params_: list[dict] = []
        self._se_alpha_: Optional[float] = None
        self._se_gamma_: Optional[np.ndarray] = None
        self._se_beta_: Optional[np.ndarray] = None

        # Subject data cache for prediction
        self._subject_covariate_rows_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        y_col: str,
        event_time_col: str,
        event_col: str,
        long_covariates: list[str],
        surv_covariates: list[str],
        exposure_col: Optional[str] = None,
    ) -> "JointModel":
        """Fit the joint model via EM.

        Parameters
        ----------
        data:
            Long-format DataFrame with one row per measurement per subject.
            Survival information (event_time, event indicator) must be the
            same on every row for a given subject.
        id_col:
            Subject/policy identifier column.
        time_col:
            Longitudinal measurement time column (numeric).
        y_col:
            Longitudinal outcome column (e.g. telematics score).
        event_time_col:
            Time-to-event column. For censored subjects, the censoring time.
        event_col:
            Event indicator. 1 = claim/event occurred, 0 = censored.
        long_covariates:
            Fixed-effect covariate names for the longitudinal sub-model.
        surv_covariates:
            Baseline covariate names for the survival sub-model.
        exposure_col:
            Not used in v0.1. Reserved for exposure-weighted models.

        Returns
        -------
        self
        """
        # Store column names
        self._id_col = id_col
        self._time_col = time_col
        self._y_col = y_col
        self._event_time_col = event_time_col
        self._event_col = event_col
        self._long_covariates = long_covariates
        self._surv_covariates = surv_covariates

        # Validate data
        _validate_data(data, id_col, time_col, y_col, event_time_col, event_col,
                       long_covariates, surv_covariates)

        # Cache one row per subject for covariate lookups
        self._subject_covariate_rows_ = (
            data.groupby(id_col).first().reset_index()
        )

        # Run EM
        self._em_fit(data)

        # Bootstrap standard errors
        if self.se_method == "bootstrap":
            self._bootstrap_se(data)

        return self

    def predict_survival(
        self,
        data: pd.DataFrame,
        id_col: str,
        landmark_time: float,
        horizon: float,
        n_mc: int = 200,
    ) -> pd.DataFrame:
        """Predict dynamic survival probability beyond the landmark time.

        For subjects with longitudinal history up to ``landmark_time``, predict
        P(T > landmark_time + horizon | T > landmark_time, ỹ_i(landmark_time)).

        This is the Rizopoulos landmark prediction — conditional on still being
        in the risk set at landmark_time, what is the probability of surviving
        a further ``horizon`` time units?

        Parameters
        ----------
        data:
            Long-format data for the subjects of interest. Only measurements
            at or before ``landmark_time`` are used.
        id_col:
            Subject identifier column.
        landmark_time:
            Current time t (must have T_i > t for included subjects).
        horizon:
            Predict survival to t + horizon.
        n_mc:
            Monte Carlo samples from the posterior of random effects.

        Returns
        -------
        DataFrame with columns: id, landmark_time, horizon, survival_prob.
        """
        self._check_fitted()
        obs_data = data[data[self._time_col] <= landmark_time].copy()
        results = []
        for subj_id in obs_data[id_col].unique():
            subj_data = obs_data[obs_data[id_col] == subj_id]
            prob = self._dynamic_survival(
                subj_data, subj_id, landmark_time,
                landmark_time + horizon, n_mc,
            )
            results.append({
                "id": subj_id,
                "landmark_time": landmark_time,
                "horizon": horizon,
                "survival_prob": prob,
            })
        return pd.DataFrame(results)

    def predict_hazard(
        self,
        data: pd.DataFrame,
        id_col: str,
        times: np.ndarray,
        n_mc: int = 200,
    ) -> pd.DataFrame:
        """Predict instantaneous hazard at specified times.

        Parameters
        ----------
        data:
            Longitudinal history up to the current time for each subject.
        id_col:
            Subject identifier column.
        times:
            Array of times at which to evaluate the hazard.
        n_mc:
            Monte Carlo samples for posterior integration.

        Returns
        -------
        DataFrame with columns: id, time, hazard.
        """
        self._check_fitted()
        results = []
        for subj_id in data[id_col].unique():
            subj_data = data[data[id_col] == subj_id]
            covariate_row = self._get_covariate_row(subj_id)
            x_i = np.array([covariate_row[c] for c in self._surv_covariates])

            for t in times:
                obs_up_to_t = subj_data[subj_data[self._time_col] <= t]
                b_samples = self._sample_random_effects_posterior(
                    obs_up_to_t, subj_id, t, n_mc
                )
                hazards = []
                for b in b_samples:
                    m_t = self.long_submodel_.marker_value(
                        t, b, covariate_row, self._time_col, self._long_covariates
                    )
                    lp = float(self.surv_submodel_.params_.gamma @ x_i)
                    lp += self.surv_submodel_.params_.alpha * m_t
                    h0_t = self._baseline_hazard_at(t)
                    hazards.append(h0_t * np.exp(lp))
                results.append({
                    "id": subj_id,
                    "time": t,
                    "hazard": float(np.mean(hazards)),
                })
        return pd.DataFrame(results)

    def association_summary(self) -> pd.DataFrame:
        """Summary of the association parameter α.

        Returns
        -------
        DataFrame with columns: parameter, estimate, std_err, z_stat, p_value.
        """
        self._check_fitted()
        alpha = self.surv_submodel_.params_.alpha
        se = self._se_alpha_ if self._se_alpha_ is not None else np.nan
        z = alpha / se if se > 0 else np.nan
        from scipy.special import ndtr
        p = 2.0 * (1.0 - ndtr(abs(z))) if not np.isnan(z) else np.nan
        return pd.DataFrame([{
            "parameter": "alpha (association)",
            "estimate": alpha,
            "std_err": se,
            "z_stat": z,
            "p_value": p,
        }])

    def longitudinal_summary(self) -> pd.DataFrame:
        """Summary of longitudinal fixed-effects coefficients β.

        Returns
        -------
        DataFrame with columns: parameter, estimate, std_err, z_stat, p_value.
        """
        self._check_fitted()
        df = self.long_submodel_.summary()
        if self._se_beta_ is not None:
            df["std_err"] = self._se_beta_
            df["z_stat"] = df["estimate"] / df["std_err"]
            from scipy.special import ndtr
            df["p_value"] = 2.0 * (1.0 - ndtr(df["z_stat"].abs()))
        return df

    def survival_summary(self) -> pd.DataFrame:
        """Summary of survival baseline covariate coefficients γ.

        Returns
        -------
        DataFrame with columns: parameter, estimate, std_err.
        """
        self._check_fitted()
        df = self.surv_submodel_.summary()
        if self._se_gamma_ is not None:
            gamma_mask = df["component"] == "gamma"
            se_vals = np.concatenate([self._se_gamma_, [self._se_alpha_ or np.nan]])
            df["std_err"] = se_vals
        return df

    # ------------------------------------------------------------------
    # EM algorithm
    # ------------------------------------------------------------------

    def _em_fit(self, data: pd.DataFrame) -> None:
        """Run the EM algorithm."""
        # Step 1: Initialise via independent fits
        self.long_submodel_ = LongitudinalSubmodel(self.long_model)
        self.long_submodel_.fit(
            data, self._id_col, self._time_col, self._y_col, self._long_covariates
        )

        # Build survival data (one row per subject)
        surv_data = self._build_survival_data(data)
        self.surv_submodel_ = SurvivalSubmodel(self.association)

        # Initial marker function using statsmodels BLUPs
        re_df = self.long_submodel_.get_random_effects(data, self._id_col)

        def marker_func_init(subj_id: str, t: float) -> float:
            row = self._get_covariate_row(subj_id)
            if subj_id in re_df.index:
                re_vals = re_df.loc[subj_id].values
            else:
                re_vals = np.zeros(len(self.long_submodel_.params_.random_names))
            return self.long_submodel_.marker_value(
                t, re_vals, row, self._time_col, self._long_covariates
            )

        self.surv_submodel_.fit(
            surv_data, self._event_time_col, self._event_col,
            self._surv_covariates, marker_func_init,
        )

        # EM iterations
        prev_loglik = -np.inf
        for iteration in range(self.max_iter):
            # E-step: compute posterior expectations for each subject
            e_step_results = self._e_step(data, surv_data)

            # M-step: update parameters
            self._m_step(data, surv_data, e_step_results)

            # Compute observed-data log-likelihood
            loglik = self._observed_loglik(data, surv_data, e_step_results)
            self.loglik_history_.append(loglik)
            self.n_iter_ = iteration + 1

            # Convergence check
            delta = abs(loglik - prev_loglik) / (1.0 + abs(loglik))
            if delta < self.tol and iteration > 2:
                self.converged_ = True
                break
            prev_loglik = loglik

        if not self.converged_:
            warnings.warn(
                f"EM did not converge after {self.max_iter} iterations. "
                f"Final log-likelihood change: {delta:.6f}. "
                f"Consider increasing max_iter or checking your data.",
                RuntimeWarning,
                stacklevel=4,
            )

    def _e_step(
        self,
        long_data: pd.DataFrame,
        surv_data: pd.DataFrame,
    ) -> dict:
        """E-step: compute posterior expectations via Gauss-Hermite quadrature.

        For each subject i, we approximate:
          E[b_i | data, θ]      — posterior mean of random effects
          E[b_i b_i' | data, θ] — second moment

        Returns
        -------
        dict with keys:
          'Eb':   list of E[b_i], one per subject
          'Ebb':  list of E[b_i b_i'], one per subject
          'posterior_modes': list of b_i mode estimates
          'posterior_covs':  list of posterior covariance approximations
        """
        subject_ids = surv_data["id"].values
        Eb_list = []
        Ebb_list = []
        modes_list = []
        covs_list = []

        q_dim = len(self.long_submodel_.params_.random_names)

        for subj_id in subject_ids:
            subj_long = long_data[long_data[self._id_col] == subj_id]
            subj_surv = surv_data[surv_data["id"] == subj_id].iloc[0]
            covariate_row = self._get_covariate_row(subj_id)
            x_i = np.array([covariate_row[c] for c in self._surv_covariates])
            T_i = subj_surv[self._event_time_col]
            delta_i = subj_surv[self._event_col]

            # Find posterior mode via optimisation
            b_mode = self._posterior_mode(
                subj_long, subj_id, T_i, delta_i, x_i, covariate_row, q_dim
            )
            # Approximate posterior covariance via Laplace (Hessian at mode)
            H = self._log_posterior_hessian(
                b_mode, subj_long, subj_id, T_i, delta_i, x_i, covariate_row
            )
            try:
                post_cov = posterior_covariance_approx(H)
                # Ensure positive definiteness
                eigvals = np.linalg.eigvalsh(post_cov)
                if eigvals.min() < 1e-10:
                    post_cov += (1e-6 - eigvals.min()) * np.eye(q_dim)
            except np.linalg.LinAlgError:
                post_cov = self.long_submodel_.params_.D.copy()

            modes_list.append(b_mode)
            covs_list.append(post_cov)

            # GHQ to compute posterior expectations
            if q_dim == 1:
                Eb, Ebb = self._ghq_expectations_1d(
                    b_mode, post_cov,
                    subj_long, subj_id, T_i, delta_i, x_i, covariate_row,
                )
            else:
                Eb, Ebb = self._ghq_expectations_2d(
                    b_mode, post_cov,
                    subj_long, subj_id, T_i, delta_i, x_i, covariate_row,
                )
            Eb_list.append(Eb)
            Ebb_list.append(Ebb)

        return {
            "Eb": Eb_list,
            "Ebb": Ebb_list,
            "posterior_modes": modes_list,
            "posterior_covs": covs_list,
            "subject_ids": list(subject_ids),
        }

    def _m_step(
        self,
        long_data: pd.DataFrame,
        surv_data: pd.DataFrame,
        e_step_results: dict,
    ) -> None:
        """M-step: update all parameters.

        Updates:
          D    — random effects covariance (average of E[b_i b_i'])
          β    — fixed effects (WLS given E[b_i])
          σ²   — residual variance
          γ, α — survival coefficients (partial likelihood maximisation)
          h₀   — baseline hazard (Breslow estimator)
        """
        subject_ids = e_step_results["subject_ids"]
        Eb_list = e_step_results["Eb"]
        Ebb_list = e_step_results["Ebb"]

        n = len(subject_ids)

        # Update D: average of E[b_i b_i']
        D_new = np.mean(np.array(Ebb_list), axis=0)
        # Ensure symmetry and positive definiteness
        D_new = 0.5 * (D_new + D_new.T)
        eigvals = np.linalg.eigvalsh(D_new)
        if eigvals.min() < 1e-8:
            D_new += (1e-6 - eigvals.min()) * np.eye(D_new.shape[0])
        self.long_submodel_.params_.D = D_new

        # Update β: WLS using expected b_i
        # For each subject, compute expected residuals and update β
        # We use a simple update: regress (y_ij - Z_ij' E[b_i]) on X_ij
        self._update_beta(long_data, subject_ids, Eb_list)

        # Update σ²
        self._update_sigma2(long_data, subject_ids, Eb_list, Ebb_list)

        # Update γ, α, h₀ via Cox partial likelihood
        id_to_Eb = dict(zip(subject_ids, Eb_list))

        def marker_func(subj_id: str, t: float) -> float:
            row = self._get_covariate_row(subj_id)
            b_i = id_to_Eb.get(subj_id, np.zeros(len(Eb_list[0])))
            return self.long_submodel_.marker_value(
                t, b_i, row, self._time_col, self._long_covariates
            )

        prev_gamma = self.surv_submodel_.params_.gamma.copy()
        prev_alpha = self.surv_submodel_.params_.alpha

        self.surv_submodel_.fit(
            surv_data,
            self._event_time_col,
            self._event_col,
            self._surv_covariates,
            marker_func,
            init_gamma=prev_gamma,
            init_alpha=prev_alpha,
        )

    def _update_beta(
        self,
        long_data: pd.DataFrame,
        subject_ids: list,
        Eb_list: list,
    ) -> None:
        """Update β by regressing corrected outcomes on fixed-effect design matrix."""
        params = self.long_submodel_.params_
        rows_X = []
        rows_y_adj = []

        id_to_Eb = dict(zip(subject_ids, Eb_list))

        for _, row in long_data.iterrows():
            subj_id = row[self._id_col]
            t = row[self._time_col]
            y = row[self._y_col]
            b_i = id_to_Eb.get(subj_id, np.zeros(len(params.random_names)))

            z = self.long_submodel_._build_fixed_vector(
                t, row, self._time_col, self._long_covariates
            )
            w = self.long_submodel_._build_random_vector(t, self._time_col)
            y_adj = y - float(w @ b_i)
            rows_X.append(z)
            rows_y_adj.append(y_adj)

        X = np.array(rows_X)
        y_adj = np.array(rows_y_adj)

        # OLS: β = (X'X)⁻¹ X' y_adj
        try:
            beta_new = np.linalg.lstsq(X, y_adj, rcond=None)[0]
            self.long_submodel_.params_.beta = beta_new
        except np.linalg.LinAlgError:
            pass  # Keep current β on failure

    def _update_sigma2(
        self,
        long_data: pd.DataFrame,
        subject_ids: list,
        Eb_list: list,
        Ebb_list: list,
    ) -> None:
        """Update σ² = E[||y - Xβ - Zb||² | data] / N_total."""
        params = self.long_submodel_.params_
        id_to_Eb = dict(zip(subject_ids, Eb_list))
        id_to_Ebb = dict(zip(subject_ids, Ebb_list))

        total_sq = 0.0
        N_total = 0

        for subj_id in subject_ids:
            subj_data = long_data[long_data[self._id_col] == subj_id]
            b_i = id_to_Eb[subj_id]
            Ebb_i = id_to_Ebb[subj_id]

            for _, row in subj_data.iterrows():
                t = row[self._time_col]
                y = row[self._y_col]
                z = self.long_submodel_._build_fixed_vector(
                    t, row, self._time_col, self._long_covariates
                )
                w = self.long_submodel_._build_random_vector(t, self._time_col)
                mu = float(z @ params.beta)
                r = y - mu - float(w @ b_i)
                total_sq += r ** 2 + float(w @ Ebb_i @ w) - float(w @ b_i) ** 2
                N_total += 1

        if N_total > 0:
            self.long_submodel_.params_.sigma2 = max(total_sq / N_total, 1e-6)

    def _observed_loglik(
        self,
        long_data: pd.DataFrame,
        surv_data: pd.DataFrame,
        e_step_results: dict,
    ) -> float:
        """Approximate observed-data log-likelihood via GHQ.

        Uses the posterior mode approximation for efficiency.
        """
        subject_ids = e_step_results["subject_ids"]
        modes = e_step_results["posterior_modes"]
        covs = e_step_results["posterior_covs"]
        loglik = 0.0
        id_to_mode = dict(zip(subject_ids, modes))

        for subj_id in subject_ids:
            subj_long = long_data[long_data[self._id_col] == subj_id]
            subj_surv = surv_data[surv_data["id"] == subj_id].iloc[0]
            covariate_row = self._get_covariate_row(subj_id)
            x_i = np.array([covariate_row[c] for c in self._surv_covariates])
            T_i = subj_surv[self._event_time_col]
            delta_i = subj_surv[self._event_col]
            b_i = id_to_mode[subj_id]

            ll_i = self._joint_log_density(
                b_i, subj_long, subj_id, T_i, delta_i, x_i, covariate_row
            )
            loglik += ll_i

        return loglik

    # ------------------------------------------------------------------
    # Quadrature helpers
    # ------------------------------------------------------------------

    def _ghq_expectations_1d(
        self,
        b_mode: np.ndarray,
        post_cov: np.ndarray,
        subj_long: pd.DataFrame,
        subj_id: str,
        T_i: float,
        delta_i: int,
        x_i: np.ndarray,
        covariate_row: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """1-D GHQ for random intercept only model."""
        pts, wts = gauss_hermite_points(self.n_quad_points)
        mean = float(b_mode[0])
        var = float(post_cov[0, 0])
        sd = np.sqrt(var)

        nodes = mean + np.sqrt(2.0) * sd * pts
        log_weights_unnorm = []
        for node in nodes:
            b = np.array([node])
            ll = self._joint_log_density(
                b, subj_long, subj_id, T_i, delta_i, x_i, covariate_row
            )
            log_weights_unnorm.append(ll)

        log_w_arr = np.array(log_weights_unnorm)
        log_w_arr -= log_w_arr.max()
        unnorm_weights = wts * np.exp(log_w_arr)
        norm_const = np.sum(unnorm_weights)
        if norm_const < 1e-300:
            norm_weights = wts / np.sum(wts)
        else:
            norm_weights = unnorm_weights / norm_const

        Eb = np.array([np.sum(norm_weights * nodes)])
        Ebb = np.array([[np.sum(norm_weights * nodes ** 2)]])
        return Eb, Ebb

    def _ghq_expectations_2d(
        self,
        b_mode: np.ndarray,
        post_cov: np.ndarray,
        subj_long: pd.DataFrame,
        subj_id: str,
        T_i: float,
        delta_i: int,
        x_i: np.ndarray,
        covariate_row: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """2-D product GHQ for random intercept + slope model."""
        nodes_std, weights_std = product_rule_2d(self.n_quad_points)
        L = np.linalg.cholesky(post_cov)
        nodes = b_mode + np.sqrt(2.0) * (nodes_std @ L.T)

        log_weights_unnorm = []
        for node in nodes:
            ll = self._joint_log_density(
                node, subj_long, subj_id, T_i, delta_i, x_i, covariate_row
            )
            log_weights_unnorm.append(ll)

        log_w_arr = np.array(log_weights_unnorm)
        log_w_arr -= log_w_arr.max()
        unnorm_weights = weights_std * np.exp(log_w_arr)
        norm_const = np.sum(unnorm_weights)
        if norm_const < 1e-300:
            norm_weights = weights_std / np.sum(weights_std)
        else:
            norm_weights = unnorm_weights / norm_const

        Eb = np.sum(norm_weights[:, np.newaxis] * nodes, axis=0)
        Ebb = np.sum(
            norm_weights[:, np.newaxis, np.newaxis] *
            nodes[:, :, np.newaxis] * nodes[:, np.newaxis, :],
            axis=0,
        )
        return Eb, Ebb

    # ------------------------------------------------------------------
    # Joint density and posterior
    # ------------------------------------------------------------------

    def _joint_log_density(
        self,
        b_i: np.ndarray,
        subj_long: pd.DataFrame,
        subj_id: str,
        T_i: float,
        delta_i: int,
        x_i: np.ndarray,
        covariate_row: pd.Series,
    ) -> float:
        """Log joint density log p(y_i, T_i, δ_i | b_i, θ).

        = log p(y_i | b_i, β, σ²) + log p(T_i, δ_i | b_i, γ, α) + log p(b_i | D)
        """
        params_long = self.long_submodel_.params_
        params_surv = self.surv_submodel_.params_

        # Longitudinal log-likelihood: Σ_j log N(y_ij; m_ij, σ²)
        ll_long = 0.0
        for _, row in subj_long.iterrows():
            t = row[self._time_col]
            y = row[self._y_col]
            m = self.long_submodel_.marker_value(
                t, b_i, covariate_row, self._time_col, self._long_covariates
            )
            ll_long += -0.5 * np.log(2 * np.pi * params_long.sigma2)
            ll_long -= (y - m) ** 2 / (2.0 * params_long.sigma2)

        # Survival log-likelihood: δ_i·log h_i(T_i) - H_i(T_i)
        def marker_func_i(t: float) -> float:
            return self.long_submodel_.marker_value(
                t, b_i, covariate_row, self._time_col, self._long_covariates
            )

        m_Ti = marker_func_i(T_i)
        lp = float(params_surv.gamma @ x_i) + params_surv.alpha * m_Ti
        h0_Ti = self._baseline_hazard_at(T_i)
        H_i = self.surv_submodel_.cumulative_hazard(T_i, x_i, marker_func_i)
        ll_surv = delta_i * (np.log(max(h0_Ti, 1e-300)) + lp) - H_i

        # Random effects prior: log N(b_i; 0, D)
        D = params_long.D
        q = len(b_i)
        sign, logdet = np.linalg.slogdet(D)
        if sign <= 0:
            logdet = -30.0
        try:
            D_inv = np.linalg.solve(D, np.eye(q))
            ll_re = -0.5 * (q * np.log(2 * np.pi) + logdet + b_i @ D_inv @ b_i)
        except np.linalg.LinAlgError:
            ll_re = -0.5 * q * np.log(2 * np.pi)

        return ll_long + ll_surv + ll_re

    def _posterior_mode(
        self,
        subj_long: pd.DataFrame,
        subj_id: str,
        T_i: float,
        delta_i: int,
        x_i: np.ndarray,
        covariate_row: pd.Series,
        q_dim: int,
    ) -> np.ndarray:
        """Find posterior mode of b_i via gradient-based optimisation."""
        def neg_log_post(b: np.ndarray) -> float:
            return -self._joint_log_density(
                b, subj_long, subj_id, T_i, delta_i, x_i, covariate_row
            )

        b0 = np.zeros(q_dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(neg_log_post, b0, method="L-BFGS-B",
                              options={"maxiter": 100, "ftol": 1e-8})
        return result.x

    def _log_posterior_hessian(
        self,
        b_mode: np.ndarray,
        subj_long: pd.DataFrame,
        subj_id: str,
        T_i: float,
        delta_i: int,
        x_i: np.ndarray,
        covariate_row: pd.Series,
    ) -> np.ndarray:
        """Numerical Hessian of log-posterior at the mode."""
        q = len(b_mode)
        eps = 1e-5
        H = np.zeros((q, q))

        def f(b: np.ndarray) -> float:
            return self._joint_log_density(
                b, subj_long, subj_id, T_i, delta_i, x_i, covariate_row
            )

        f0 = f(b_mode)
        for i in range(q):
            for j in range(i, q):
                b_pp = b_mode.copy(); b_pp[i] += eps; b_pp[j] += eps
                b_pm = b_mode.copy(); b_pm[i] += eps; b_pm[j] -= eps
                b_mp = b_mode.copy(); b_mp[i] -= eps; b_mp[j] += eps
                b_mm = b_mode.copy(); b_mm[i] -= eps; b_mm[j] -= eps
                h_ij = (f(b_pp) - f(b_pm) - f(b_mp) + f(b_mm)) / (4 * eps ** 2)
                H[i, j] = h_ij
                H[j, i] = h_ij

        return H

    # ------------------------------------------------------------------
    # Dynamic prediction
    # ------------------------------------------------------------------

    def _dynamic_survival(
        self,
        subj_data: pd.DataFrame,
        subj_id: str,
        t_landmark: float,
        t_horizon: float,
        n_mc: int,
    ) -> float:
        """P(T > t_horizon | T > t_landmark, ỹ_i(t_landmark))."""
        covariate_row = self._get_covariate_row(subj_id)
        x_i = np.array([covariate_row[c] for c in self._surv_covariates])
        b_samples = self._sample_random_effects_posterior(
            subj_data, subj_id, t_landmark, n_mc
        )

        ratios = []
        for b in b_samples:
            def mf(t: float) -> float:
                return self.long_submodel_.marker_value(
                    t, b, covariate_row, self._time_col, self._long_covariates
                )
            S_landmark = self.surv_submodel_.survival(t_landmark, x_i, mf)
            S_horizon = self.surv_submodel_.survival(t_horizon, x_i, mf)
            if S_landmark > 1e-10:
                ratios.append(S_horizon / S_landmark)
            else:
                ratios.append(0.0)

        return float(np.mean(ratios))

    def _sample_random_effects_posterior(
        self,
        subj_data: pd.DataFrame,
        subj_id: str,
        current_time: float,
        n_mc: int,
    ) -> list[np.ndarray]:
        """Draw samples from the posterior of b_i using Metropolis-Hastings.

        Proposal: multivariate normal centred at the posterior mode with
        covariance from the Laplace approximation. This gives a reasonable
        acceptance rate for typical posteriors.
        """
        covariate_row = self._get_covariate_row(subj_id)
        x_i = np.array([covariate_row[c] for c in self._surv_covariates])
        q_dim = len(self.long_submodel_.params_.random_names)

        # Find posterior mode
        surv_row = pd.Series({
            self._event_time_col: current_time,
            self._event_col: 0,
        })
        b_mode = self._posterior_mode(
            subj_data, subj_id, current_time, 0, x_i, covariate_row, q_dim
        )
        H = self._log_posterior_hessian(
            b_mode, subj_data, subj_id, current_time, 0, x_i, covariate_row
        )
        try:
            proposal_cov = posterior_covariance_approx(H)
            eigvals = np.linalg.eigvalsh(proposal_cov)
            if eigvals.min() < 1e-10:
                proposal_cov += (1e-6 - eigvals.min()) * np.eye(q_dim)
        except np.linalg.LinAlgError:
            proposal_cov = self.long_submodel_.params_.D.copy()

        def log_post(b: np.ndarray) -> float:
            return self._joint_log_density(
                b, subj_data, subj_id, current_time, 0, x_i, covariate_row
            )

        # MH with Laplace proposal
        samples = []
        b_current = b_mode.copy()
        lp_current = log_post(b_current)

        # Burn-in
        n_burnin = min(200, n_mc)
        for _ in range(n_burnin + n_mc):
            b_prop = self.rng_.multivariate_normal(b_current, proposal_cov)
            lp_prop = log_post(b_prop)
            log_accept = lp_prop - lp_current
            if np.log(self.rng_.uniform()) < log_accept:
                b_current = b_prop
                lp_current = lp_prop
            if len(samples) < n_mc - n_burnin:
                pass  # burn-in phase, discard
            else:
                samples.append(b_current.copy())

        # Simpler approach: just collect n_mc samples after burn-in
        samples = []
        b_current = b_mode.copy()
        lp_current = log_post(b_current)
        for _ in range(n_burnin):
            b_prop = self.rng_.multivariate_normal(b_current, proposal_cov)
            lp_prop = log_post(b_prop)
            if np.log(self.rng_.uniform()) < lp_prop - lp_current:
                b_current = b_prop
                lp_current = lp_prop
        for _ in range(n_mc):
            b_prop = self.rng_.multivariate_normal(b_current, proposal_cov)
            lp_prop = log_post(b_prop)
            if np.log(self.rng_.uniform()) < lp_prop - lp_current:
                b_current = b_prop
                lp_current = lp_prop
            samples.append(b_current.copy())

        return samples

    # ------------------------------------------------------------------
    # Bootstrap SEs
    # ------------------------------------------------------------------

    def _bootstrap_se(self, data: pd.DataFrame) -> None:
        """Compute bootstrap standard errors by resampling subjects."""
        all_ids = data[self._id_col].unique()
        n = len(all_ids)

        alpha_samples = []
        gamma_samples = []
        beta_samples = []

        rng = np.random.default_rng(self.random_state)

        for b_idx in range(self.n_bootstrap):
            # Resample subjects with replacement
            boot_ids = rng.choice(all_ids, size=n, replace=True)
            # Build bootstrap dataset
            chunks = []
            for new_id, orig_id in enumerate(boot_ids):
                chunk = data[data[self._id_col] == orig_id].copy()
                chunk[self._id_col] = f"_bs_{new_id}"
                chunks.append(chunk)
            boot_data = pd.concat(chunks, ignore_index=True)

            try:
                boot_model = JointModel(
                    long_model=self.long_model,
                    surv_model=self.surv_model,
                    association=self.association,
                    n_quad_points=self.n_quad_points,
                    max_iter=self.max_iter,
                    tol=self.tol * 10,  # Looser tolerance for bootstrap
                    se_method="none",
                    random_state=int(rng.integers(1e6)),
                )
                boot_model.fit(
                    boot_data,
                    self._id_col,
                    self._time_col,
                    self._y_col,
                    self._event_time_col,
                    self._event_col,
                    self._long_covariates,
                    self._surv_covariates,
                )
                alpha_samples.append(boot_model.surv_submodel_.params_.alpha)
                gamma_samples.append(boot_model.surv_submodel_.params_.gamma.copy())
                beta_samples.append(boot_model.long_submodel_.params_.beta.copy())
            except Exception:
                # Skip failed bootstrap samples
                continue

        if len(alpha_samples) >= 5:
            self._se_alpha_ = float(np.std(alpha_samples, ddof=1))
            self._se_gamma_ = np.std(np.array(gamma_samples), axis=0, ddof=1)
            self._se_beta_ = np.std(np.array(beta_samples), axis=0, ddof=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_survival_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build one-row-per-subject survival dataset."""
        cols = ["id", self._event_time_col, self._event_col] + self._surv_covariates
        available = ["id" if c == self._id_col else c for c in cols]
        surv_data = data.rename(columns={self._id_col: "id"})
        keep = [c for c in ["id", self._event_time_col, self._event_col] + self._surv_covariates
                if c in surv_data.columns]
        return surv_data[keep].groupby("id").first().reset_index()

    def _get_covariate_row(self, subj_id: str) -> pd.Series:
        """Return representative covariate row for a subject."""
        mask = self._subject_covariate_rows_[self._id_col] == subj_id
        rows = self._subject_covariate_rows_[mask]
        if len(rows) == 0:
            return pd.Series(dtype=float)
        return rows.iloc[0]

    def _baseline_hazard_at(self, t: float) -> float:
        """Instantaneous baseline hazard at time t (using the Breslow jumps)."""
        p = self.surv_submodel_.params_
        mask = p.baseline_times == t
        if mask.any():
            return float(p.baseline_hazard[mask][0])
        # Between event times, return 0 (Breslow is a step function)
        return 1e-10

    def _check_fitted(self) -> None:
        if self.long_submodel_ is None or self.surv_submodel_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")


def _validate_data(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    y_col: str,
    event_time_col: str,
    event_col: str,
    long_covariates: list[str],
    surv_covariates: list[str],
) -> None:
    """Validate input data format and column existence."""
    required_cols = [id_col, time_col, y_col, event_time_col, event_col]
    required_cols += long_covariates + surv_covariates

    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    if data[event_col].nunique() > 2 or not set(data[event_col].unique()).issubset({0, 1}):
        raise ValueError(
            f"event_col '{event_col}' must contain only 0 and 1 values."
        )

    # Check that event time is constant within subject
    et_var = data.groupby(id_col)[event_time_col].nunique()
    if (et_var > 1).any():
        bad = list(et_var[et_var > 1].index[:3])
        raise ValueError(
            f"event_time_col '{event_time_col}' varies within subjects: {bad}. "
            f"Each subject must have a single event/censoring time."
        )

    if data[y_col].isna().any():
        raise ValueError(f"y_col '{y_col}' contains missing values. Impute before fitting.")
