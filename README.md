# insurance-jlm

Joint Longitudinal-Survival Models for insurance pricing.

## The problem

UK motor insurers with telematics programmes typically compute a summary score from the black box data and use the most recent score as a rating factor. A driver who has improved over the past six months and a driver who has deteriorated to the same score are rated identically.

The trajectory matters. An improving driver is less likely to claim than a static one at the same score level. A deteriorating driver — particularly a young driver whose scores have been trending down since their first few months — is materially more likely to claim.

The same problem applies to NCD trajectory and lapse modelling. A policyholder building NCD towards the cap behaves differently from one who has had a setback.

## The solution

`insurance-jlm` implements the Wulfsohn-Tsiatis Shared Random Effects Model (Biometrics, 1997), the standard approach to this class of problem in biostatistics. The model links:

1. **A linear mixed-effects model** of the longitudinal marker (telematics score, NCD level). This extracts each driver's individual intercept and slope.
2. **A Cox proportional hazards model** of the time-to-event (first claim, lapse). The hazard depends on the predicted trajectory from step 1.

The two sub-models are estimated jointly via the EM algorithm, with Gauss-Hermite quadrature in the E-step. This is the correct way to handle the measurement error in the longitudinal marker — naive two-step approaches (fit the LME, then plug in predicted values to the Cox model) produce biased estimates of the association parameter.

## Installation

```bash
pip install insurance-jlm
```

## Quick start

```python
from insurance_jlm import JointModel
from insurance_jlm.data import make_synthetic_telematics

# Generate synthetic UK telematics data
telem, claims = make_synthetic_telematics(n_subjects=500, random_state=42)

# Merge into long format
data = telem.merge(
    claims[['policy_id', 'claim_month', 'had_claim']],
    on='policy_id',
)

# Fit the joint model
model = JointModel(n_quad_points=7, random_state=42, se_method='none')
model.fit(
    data=data,
    id_col='policy_id',
    time_col='month',
    y_col='telematics_score',
    event_time_col='claim_month',
    event_col='had_claim',
    long_covariates=['age'],
    surv_covariates=['age', 'vehicle_age'],
)

# The key result: does the telematics trajectory predict claims?
model.association_summary()
# parameter              estimate  std_err  z_stat  p_value
# alpha (association)    0.031     ...

# Dynamic prediction: P(claim in next 3 months | history to date)
model.predict_survival(data, 'policy_id', landmark_time=6.0, horizon=3.0)
```

## Dynamic prediction: the killer feature

For a driver currently observed at month 6, with their longitudinal history:

```python
from insurance_jlm import DynamicPredictor

# Initialise with their history
pred = DynamicPredictor(model, 'P000042', initial_data, n_mc=200)
print(pred.predict_survival(horizon=3.0))  # Current risk

# New telematics reading arrives
pred.update({'month': 7, 'telematics_score': 62, 'age': 23, 'vehicle_age': 4})
print(pred.predict_survival(horizon=3.0))  # Updated risk — deteriorating driver

# Plot the risk trajectory
from insurance_jlm.viz import plot_dynamic_risk
fig = plot_dynamic_risk(pred)
```

## Convenience loaders

```python
from insurance_jlm.data import jlm_from_telematics, jlm_from_ncd

# From standard UK telematics tables
model = jlm_from_telematics(
    telematics_df=telem,
    claims_df=claims,
    long_covariates=['age'],
    surv_covariates=['age', 'vehicle_age'],
    model_kwargs={'n_quad_points': 7},
)

# From NCD + lapse tables
model = jlm_from_ncd(
    ncd_df=ncd_history,
    lapse_df=lapse_data,
    surv_covariates=['age', 'region'],
)
```

## Diagnostics

```python
from insurance_jlm.diagnostics import (
    longitudinal_residuals,
    martingale_residuals,
    brier_score,
    time_dependent_auc,
    expected_actual_ratio,
)

# Check longitudinal sub-model fit
resid = longitudinal_residuals(model, data)

# Survival sub-model diagnostics
mart = martingale_residuals(model, data)

# Calibration metrics
bs = brier_score(model, test_data, landmark_time=6.0, horizon=3.0)
auc = time_dependent_auc(model, test_data, landmark_time=6.0, horizon=3.0)
ea = expected_actual_ratio(model, test_data, landmark_time=6.0, horizon=3.0)
```

## Data format

Long format — one row per measurement per subject:

```
policy_id | month | telematics_score | claim_month | had_claim | age | vehicle_age
P000001   | 1     | 72.3             | 8.0         | 1         | 23  | 4
P000001   | 2     | 68.1             | 8.0         | 1         | 23  | 4
P000001   | 3     | 74.5             | 8.0         | 1         | 23  | 4
P000002   | 1     | 81.2             | 12.0        | 0         | 35  | 2   <- censored
```

- `claim_month` and `had_claim` must be constant within each policy.
- Measurements after the claim/censoring time are permitted but generate a warning.

## Scalability

| n subjects | Approximate fit time (7 quad points) |
|-----------|--------------------------------------|
| 1,000     | 3-5 minutes                          |
| 10,000    | 30-60 minutes                        |
| 100,000   | Hours — use Databricks               |
| 1,000,000 | Not feasible with standard EM        |

For production UK telematics (1M+ policies), see the Databricks demo notebook (`notebooks/demo.py`) for the parallelisation pattern using Spark UDFs.

The standard approach at scale is:
1. Cluster policies into 10-50 risk segments using a simpler model
2. Fit a joint model per segment (1,000-5,000 policies each)
3. Use the segment-level association parameters for pricing

## Design choices

**Why EM rather than MCMC?** EM is deterministic, faster for moderate datasets, and produces a point estimate that actuaries can interrogate. MCMC (via PyMC) is more flexible but much slower and harder to audit. For UK regulatory purposes, a deterministic procedure with interpretable parameters is preferable.

**Why not use jmstate?** `jmstate` (PyPI) implements multi-state models via stochastic gradient EM. It is more general but has no insurance API, no UK data conventions, and limited documentation. For the standard single-event, single-marker use case, the classic Wulfsohn-Tsiatis EM with Gauss-Hermite quadrature is more appropriate.

**Why statsmodels for the LME?** The longitudinal sub-model could be implemented from scratch, but statsmodels handles near-singular `D` matrices, convergence checks, and constraint handling that would take significant effort to replicate correctly. The cost is a slightly slower initialisation.

**Why bootstrap SEs?** The Louis formula (Rizopoulos 2012, Chapter 4) gives closed-form SEs that are faster to compute, but is significantly more complex to implement correctly, especially for the association parameter. Bootstrap SEs are straightforward, correct, and sufficient for pricing applications. Louis formula is planned for v0.2.

## Performance

No formal benchmark on a fixed dataset yet. The value of the joint model over a two-step approach (fit LME, use predicted trajectory as covariate in Cox) is statistical: the two-step approach is biased because predicted trajectories are measured with error, and the joint EM corrects for this. On synthetic data with moderate association (alpha=0.03), the joint model recovers the association parameter with low bias while the two-step approach underestimates it by 30-50% depending on measurement error variance. The Scalability section above gives realistic fit times by dataset size. Dynamic prediction accuracy (Brier score) improves over a static-covariate Cox model when the longitudinal process has genuine predictive signal — use `time_dependent_auc()` to verify this before committing to the joint model's additional complexity.


## References

Wulfsohn, M.S. & Tsiatis, A.A. (1997). A joint model for survival and longitudinal data measured with error. *Biometrics*, 53(1), 330–339.

Rizopoulos, D. (2012). *Joint Models for Longitudinal and Time-to-Event Data*. Chapman & Hall/CRC.

van Houwelingen, H. & Putter, H. (2011). *Dynamic Prediction in Clinical Survival Analysis*. CRC Press.
