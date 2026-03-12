"""insurance-jlm: Joint Longitudinal-Survival Models for insurance pricing.

Implements the Wulfsohn-Tsiatis Shared Random Effects Model (SREM):
a linear mixed-effects longitudinal sub-model linked to a Cox PH survival
sub-model via shared random effects. Estimation via EM algorithm with
Gauss-Hermite quadrature.

Primary use case: UK telematics motor insurance. Longitudinal telematics
scores predict time-to-first-claim hazard. The joint model captures the
trajectory shape (improving vs deteriorating driver) which pure cross-sectional
approaches miss.

Quick start
-----------
>>> from insurance_jlm import JointModel
>>> from insurance_jlm.data import make_synthetic_telematics
>>>
>>> telem, claims = make_synthetic_telematics(n_subjects=500, random_state=42)
>>> model = JointModel(n_quad_points=7, se_method='none', random_state=42)
>>> model.fit(
...     data=telem.merge(claims[['policy_id', 'claim_month', 'had_claim']], on='policy_id'),
...     id_col='policy_id',
...     time_col='month',
...     y_col='telematics_score',
...     event_time_col='claim_month',
...     event_col='had_claim',
...     long_covariates=['age'],
...     surv_covariates=['age', 'vehicle_age'],
... )
>>> model.association_summary()

References
----------
Wulfsohn, M.S. & Tsiatis, A.A. (1997). A joint model for survival and
longitudinal data measured with error. Biometrics, 53(1), 330-339.

Rizopoulos, D. (2012). Joint Models for Longitudinal and Time-to-Event Data.
Chapman & Hall/CRC.
"""

from .models.joint_model import JointModel
from .prediction.dynamic import DynamicPredictor
from .prediction.landmarks import LandmarkPredictor
from .data.loaders import jlm_from_telematics, jlm_from_ncd, make_synthetic_telematics
from .data.validation import validate_long_format, summarise_data, DataValidationError

__version__ = "0.1.0"

__all__ = [
    "JointModel",
    "DynamicPredictor",
    "LandmarkPredictor",
    "jlm_from_telematics",
    "jlm_from_ncd",
    "make_synthetic_telematics",
    "validate_long_format",
    "summarise_data",
    "DataValidationError",
    "__version__",
]
