# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-jlm: Joint Longitudinal-Survival Models for Telematics Pricing
# MAGIC
# MAGIC This notebook demonstrates the full workflow for fitting a joint model on
# MAGIC synthetic UK telematics data, evaluating its performance, and using it for
# MAGIC dynamic claim probability updates.
# MAGIC
# MAGIC **Use case**: 1,000 UK motor policies with monthly telematics scores.
# MAGIC We model whether the telematics *trajectory* (not just the current score)
# MAGIC predicts time-to-first-claim, and use this to update per-driver claim
# MAGIC probabilities in real time.

# COMMAND ----------

# MAGIC %pip install insurance-jlm lifelines statsmodels matplotlib

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from insurance_jlm import JointModel, DynamicPredictor
from insurance_jlm.data import make_synthetic_telematics, summarise_data, validate_long_format
from insurance_jlm.diagnostics import (
    longitudinal_residuals,
    martingale_residuals,
    brier_score,
    time_dependent_auc,
    expected_actual_ratio,
)
from insurance_jlm.viz import (
    plot_trajectories,
    plot_loglik_convergence,
    plot_dynamic_risk,
    plot_baseline_hazard,
)

print("insurance-jlm loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic UK Telematics Data
# MAGIC
# MAGIC We generate data from the true joint model with known parameters:
# MAGIC - True association α = 0.03 (higher score → slightly higher claim hazard)
# MAGIC - Random intercept variance: 4.0 (drivers differ substantially in baseline risk)
# MAGIC - Random slope variance: 0.25 (some drivers improve, others deteriorate)

# COMMAND ----------

RANDOM_STATE = 42
N_SUBJECTS = 1000

telem, claims = make_synthetic_telematics(
    n_subjects=N_SUBJECTS,
    max_months=12,
    base_claim_rate=0.08,
    alpha_true=0.03,
    random_state=RANDOM_STATE,
)

print(f"Telematics data: {len(telem):,} rows, {telem['policy_id'].nunique():,} policies")
print(f"Claims data:     {len(claims):,} rows")
print(f"\nClaim rate: {claims['had_claim'].mean():.1%}")
print(f"\nTelematics score summary:")
print(telem["telematics_score"].describe().round(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare Long-Format Data

# COMMAND ----------

data = telem.merge(
    claims[["policy_id", "claim_month", "had_claim"]],
    on="policy_id",
)

print("Long-format data shape:", data.shape)
print("\nSample rows:")
display(data.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data validation and summary

# COMMAND ----------

validation_warnings = validate_long_format(
    data,
    id_col="policy_id",
    time_col="month",
    y_col="telematics_score",
    event_time_col="claim_month",
    event_col="had_claim",
    long_covariates=["age"],
    surv_covariates=["age", "vehicle_age"],
)
if validation_warnings:
    for w in validation_warnings:
        print(f"WARNING: {w}")
else:
    print("Data validation passed — no issues found.")

summary = summarise_data(data, "policy_id", "month", "telematics_score", "claim_month", "had_claim")
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit the Joint Model
# MAGIC
# MAGIC We use 7 Gauss-Hermite quadrature points (the standard for dim(b)=2).
# MAGIC
# MAGIC With 1,000 subjects on Databricks serverless compute, expect 5-15 minutes.
# MAGIC The EM should converge in 20-50 iterations.

# COMMAND ----------

# Train/test split — 80% train, 20% test
all_ids = data["policy_id"].unique()
rng = np.random.default_rng(RANDOM_STATE)
train_ids = set(rng.choice(all_ids, size=int(0.8 * len(all_ids)), replace=False))
test_ids = set(all_ids) - train_ids

train_data = data[data["policy_id"].isin(train_ids)].copy()
test_data = data[data["policy_id"].isin(test_ids)].copy()

print(f"Training: {len(train_ids)} policies, {len(train_data):,} observations")
print(f"Test:     {len(test_ids)} policies, {len(test_data):,} observations")

# COMMAND ----------

model = JointModel(
    long_model="linear",        # Random intercept + slope
    surv_model="cox",
    association="current_value",
    n_quad_points=7,
    max_iter=100,
    tol=1e-4,
    se_method="none",           # Set to 'bootstrap' for production — adds ~10 min
    random_state=RANDOM_STATE,
)

print("Fitting joint model...")
model.fit(
    data=train_data,
    id_col="policy_id",
    time_col="month",
    y_col="telematics_score",
    event_time_col="claim_month",
    event_col="had_claim",
    long_covariates=["age"],
    surv_covariates=["age", "vehicle_age"],
)

print(f"\nConverged: {model.converged_}")
print(f"Iterations: {model.n_iter_}")
print(f"Final log-likelihood: {model.loglik_history_[-1]:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model Results
# MAGIC
# MAGIC ### Association parameter α
# MAGIC
# MAGIC This is the key result. A positive α means that higher telematics scores
# MAGIC at time t are associated with higher claim hazard at time t — i.e., the
# MAGIC trajectory shape matters beyond the raw score level.

# COMMAND ----------

print("=== ASSOCIATION PARAMETER ===")
display(model.association_summary())

print("\n=== LONGITUDINAL FIXED EFFECTS ===")
display(model.longitudinal_summary())

print("\n=== SURVIVAL COEFFICIENTS ===")
display(model.survival_summary())

# COMMAND ----------

print(f"\nRandom effects covariance (D):")
D = model.long_submodel_.params_.D
print(f"  Intercept variance:  {D[0, 0]:.3f}  (true: 4.0)")
print(f"  Slope variance:      {D[1, 1]:.3f}  (true: 0.25)")
print(f"  Covariance:          {D[0, 1]:.3f}  (true: 0.5)")
print(f"\nResidual variance σ²: {model.long_submodel_.params_.sigma2:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Convergence Diagnostics

# COMMAND ----------

fig = plot_loglik_convergence(model)
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Longitudinal Trajectory Plots
# MAGIC
# MAGIC Visual check: are the individual BLUP trajectories capturing the shape of
# MAGIC the observed scores? We want smooth fitted lines through the noisy observations.

# COMMAND ----------

fig = plot_trajectories(model, train_data, n_ids=12, figsize=(14, 6))
fig.suptitle("UK Telematics: Observed Scores and Fitted Trajectories", fontsize=12)
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Residual Diagnostics

# COMMAND ----------

# Longitudinal residuals
long_resid = longitudinal_residuals(model, train_data, type="subject_specific")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(long_resid["residual"], bins=40, edgecolor="white")
axes[0].set_title("Subject-specific residuals (should be N(0, σ²))")
axes[0].set_xlabel("Residual")

axes[1].scatter(long_resid["fitted"], long_resid["residual"], alpha=0.3, s=5)
axes[1].axhline(0, color="red", linewidth=1)
axes[1].set_title("Residuals vs Fitted")
axes[1].set_xlabel("Fitted value")
axes[1].set_ylabel("Residual")

plt.tight_layout()
display(fig)
plt.close()

print(f"\nResidual mean: {long_resid['residual'].mean():.3f}  (should be ~0)")
print(f"Residual std:  {long_resid['residual'].std():.3f}  (should be ~{model.long_submodel_.params_.sigma2**0.5:.3f})")

# COMMAND ----------

# Martingale residuals
mart_resid = martingale_residuals(model, train_data)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(mart_resid["martingale_residual"], bins=30, edgecolor="white", color="steelblue")
axes[0].set_title("Martingale residuals (survival sub-model)")
axes[0].set_xlabel("Martingale residual")

# Plot against cumulative hazard (Cox-Snell check)
axes[1].plot(np.sort(mart_resid["cox_snell_residual"]),
             -np.log(1 - np.linspace(0.01, 0.99, len(mart_resid))),
             ".", alpha=0.5)
axes[1].plot([0, 3], [0, 3], "r--", label="45° line (ideal)")
axes[1].set_xlabel("Cox-Snell residual")
axes[1].set_ylabel("-log(1 - KM estimate)")
axes[1].set_title("Cox-Snell residual check (should follow 45° line)")
axes[1].legend()

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Performance Metrics on Test Set

# COMMAND ----------

landmark_time = 4.0  # Predict from month 4 onwards
horizon = 3.0        # Predict claim probability over next 3 months

print(f"Landmark time: month {landmark_time}")
print(f"Prediction horizon: {horizon} months (i.e., predict through month {landmark_time + horizon})")
print()

bs = brier_score(model, test_data, landmark_time=landmark_time, horizon=horizon, n_mc=50)
auc = time_dependent_auc(model, test_data, landmark_time=landmark_time, horizon=horizon, n_mc=50)

print(f"Brier score:          {bs:.4f}  (naive model: 0.25)")
print(f"Time-dependent AUC:   {auc:.4f}  (random: 0.50)")

# COMMAND ----------

print("\nExpected/Actual ratio by predicted risk quintile:")
ea = expected_actual_ratio(model, test_data, landmark_time=landmark_time, horizon=horizon, n_mc=50)
display(ea)
print("\nE/A ratios near 1.0 indicate good calibration.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Baseline Hazard

# COMMAND ----------

fig = plot_baseline_hazard(model, figsize=(10, 4))
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Dynamic Prediction: Real-Time Risk Updates
# MAGIC
# MAGIC The most valuable feature of the joint model for telematics pricing:
# MAGIC as new monthly scores arrive, we update the predicted claim probability
# MAGIC without refitting the model.
# MAGIC
# MAGIC **Scenario**: Two young drivers both have a telematics score of 68 at month 4.
# MAGIC Driver A has been improving (72 → 70 → 68). Driver B has been deteriorating
# MAGIC (62 → 65 → 68). The joint model treats them differently.

# COMMAND ----------

# Create initial data for two hypothetical drivers
improving_driver_data = pd.DataFrame({
    "policy_id": ["DEMO_A"] * 3,
    "month": [1.0, 2.0, 3.0],
    "telematics_score": [72.0, 70.0, 68.0],
    "age": [24.0, 24.0, 24.0],
    "vehicle_age": [3.0, 3.0, 3.0],
    "claim_month": [12.0, 12.0, 12.0],
    "had_claim": [0, 0, 0],
})

deteriorating_driver_data = pd.DataFrame({
    "policy_id": ["DEMO_B"] * 3,
    "month": [1.0, 2.0, 3.0],
    "telematics_score": [62.0, 65.0, 68.0],
    "age": [24.0, 24.0, 24.0],
    "vehicle_age": [3.0, 3.0, 3.0],
    "claim_month": [12.0, 12.0, 12.0],
    "had_claim": [0, 0, 0],
})

# COMMAND ----------

pred_A = DynamicPredictor(model, "DEMO_A", improving_driver_data, n_mc=100)
pred_B = DynamicPredictor(model, "DEMO_B", deteriorating_driver_data, n_mc=100)

# Month 4 arrives: both score 68
new_reading = {"month": 4.0, "telematics_score": 68.0, "age": 24.0, "vehicle_age": 3.0}

surv_A = pred_A.update(new_reading)
surv_B = pred_B.update(new_reading)

print("After receiving month 4 score of 68 for both drivers:")
print(f"Driver A (improving 72→70→68→68): P(no claim, next 3 months) = {surv_A:.3f}")
print(f"Driver B (deteriorating 62→65→68→68): P(no claim, next 3 months) = {surv_B:.3f}")

# Continue updating monthly
months_5_to_8 = [
    (5.0, 67.0, 63.0),
    (6.0, 65.0, 67.0),
    (7.0, 64.0, 70.0),
    (8.0, 63.0, 72.0),
]
for month, score_A, score_B in months_5_to_8:
    pred_A.update({"month": month, "telematics_score": score_A, "age": 24.0, "vehicle_age": 3.0})
    pred_B.update({"month": month, "telematics_score": score_B, "age": 24.0, "vehicle_age": 3.0})

print("\nAfter 8 months of monitoring:")
traj_A = pred_A.risk_trajectory()
traj_B = pred_B.risk_trajectory()
print(f"Driver A latest claim probability: {1 - traj_A['survival_prob_1_unit'].iloc[-1]:.3f}")
print(f"Driver B latest claim probability: {1 - traj_B['survival_prob_1_unit'].iloc[-1]:.3f}")

# COMMAND ----------

# Plot risk trajectories
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, pred, driver_id, label in [
    (axes[0], pred_A, "DEMO_A", "Driver A (initially improving)"),
    (axes[1], pred_B, "DEMO_B", "Driver B (initially deteriorating)"),
]:
    hist = pred.history.sort_values("month")
    traj = pred.risk_trajectory()

    ax.plot(hist["month"], hist["telematics_score"], "o-", color="steelblue",
            label="Telematics score", linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(traj["time"], 1 - traj["survival_prob_1_unit"], "s--",
             color="crimson", label="Claim prob (next month)", linewidth=1.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("Telematics score", color="steelblue")
    ax2.set_ylabel("Monthly claim probability", color="crimson")
    ax.set_title(label)

plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Scalability Pattern for Large Fleets
# MAGIC
# MAGIC For production fleets with 100k+ policies, the EM loop cannot run on a
# MAGIC single node in reasonable time. The practical approach on Databricks is:
# MAGIC
# MAGIC 1. **Segment the fleet** into homogeneous groups (e.g., by vehicle class,
# MAGIC    age band, or telematics quartile at month 1)
# MAGIC 2. **Fit one joint model per segment** (typically 5,000-20,000 policies each)
# MAGIC 3. **Distribute across workers** using Spark UDFs or Databricks Jobs
# MAGIC
# MAGIC The code pattern below shows how to fit segment models in parallel.

# COMMAND ----------

# Example: fit separate models by age band
data_with_band = train_data.copy()
data_with_band["age_band"] = pd.cut(data_with_band["age"], bins=[0, 25, 35, 50, 100],
                                     labels=["17-25", "26-35", "36-50", "51+"])

segment_models = {}
for band in ["17-25", "26-35", "36-50", "51+"]:
    segment = data_with_band[data_with_band["age_band"] == band]
    if segment["policy_id"].nunique() < 50:
        print(f"Skipping {band}: too few policies ({segment['policy_id'].nunique()})")
        continue

    n_events = segment.groupby("policy_id")["had_claim"].first().sum()
    if n_events < 10:
        print(f"Skipping {band}: too few events ({int(n_events)})")
        continue

    print(f"Fitting {band}: {segment['policy_id'].nunique()} policies, {int(n_events)} events")
    seg_model = JointModel(n_quad_points=3, max_iter=10, se_method="none", random_state=0)
    seg_model.fit(
        data=segment,
        id_col="policy_id",
        time_col="month",
        y_col="telematics_score",
        event_time_col="claim_month",
        event_col="had_claim",
        long_covariates=[],
        surv_covariates=["vehicle_age"],
    )
    segment_models[band] = seg_model
    alpha = seg_model.surv_submodel_.params_.alpha
    print(f"  α = {alpha:.4f}")

print("\nSegment-level association parameters:")
for band, m in segment_models.items():
    print(f"  {band}: α = {m.surv_submodel_.params_.alpha:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook has demonstrated:
# MAGIC
# MAGIC 1. **Data preparation**: long-format telematics + claims data with validation
# MAGIC 2. **Model fitting**: EM algorithm with Gauss-Hermite quadrature, converging
# MAGIC    in ~30-50 iterations for 1,000 policies
# MAGIC 3. **Inference**: the association parameter α quantifies how much the
# MAGIC    telematics trajectory (not just current score) predicts claim hazard
# MAGIC 4. **Diagnostics**: residual checks and calibration metrics
# MAGIC 5. **Dynamic prediction**: real-time claim probability updates as new
# MAGIC    telematics readings arrive — the core value for live pricing engines
# MAGIC 6. **Scalability**: segment-level modelling pattern for large fleets
# MAGIC
# MAGIC **Key finding from synthetic data**: the true α = 0.03. The joint model
# MAGIC recovers this (check `model.association_summary()` above). A naive
# MAGIC cross-sectional Cox model that ignores measurement error would be biased
# MAGIC towards zero (attenuation bias).
# MAGIC
# MAGIC **Next steps for production**:
# MAGIC - Bootstrap standard errors on α for statistical significance
# MAGIC - Sensitivity analysis: does `association='slope'` improve AUC over `'current_value'`?
# MAGIC - Integration with the pricing GLM: use predicted individual hazard as an
# MAGIC   additional rating factor alongside traditional variables
