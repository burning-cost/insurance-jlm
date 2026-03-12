"""Visualisation utilities for joint models.

These plots are designed to be used in pricing team presentations and model
review packs. They follow the conventions of UK actuarial reports: clear labels,
no jargon, readable at A4 size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure
    from ..models.joint_model import JointModel
    from ..prediction.dynamic import DynamicPredictor


def plot_trajectories(
    model: "JointModel",
    data: pd.DataFrame,
    ids: Optional[list] = None,
    n_ids: int = 10,
    figsize: tuple[float, float] = (12, 6),
) -> "matplotlib.figure.Figure":
    """Plot observed and fitted longitudinal trajectories.

    Shows the raw telematics scores alongside the fitted trajectory from the
    joint model (both population average and individual BLUP trajectories).
    Useful for checking whether the longitudinal sub-model is capturing the
    shape of the trajectories.

    Parameters
    ----------
    model:
        A fitted JointModel.
    data:
        Long-format data containing the subjects to plot.
    ids:
        Specific subject IDs to plot. If None, a random sample of n_ids is used.
    n_ids:
        Number of subjects to plot when ids is None.
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if model.long_submodel_ is None:
        raise RuntimeError("Model has not been fitted.")

    all_ids = data[model._id_col].unique()
    if ids is None:
        rng = np.random.default_rng(42)
        ids = list(rng.choice(all_ids, size=min(n_ids, len(all_ids)), replace=False))

    re_df = model.long_submodel_.get_random_effects(data, model._id_col)

    fig, ax = plt.subplots(figsize=figsize)
    colours = cm.tab10(np.linspace(0, 1, len(ids)))

    for colour, subj_id in zip(colours, ids):
        subj_data = data[data[model._id_col] == subj_id].sort_values(model._time_col)
        if len(subj_data) == 0:
            continue

        times = subj_data[model._time_col].values
        scores = subj_data[model._y_col].values
        covariate_row = model._get_covariate_row(subj_id)

        # Observed scores
        ax.scatter(times, scores, color=colour, alpha=0.6, s=20, zorder=3)

        # Fitted trajectory (individual BLUP)
        t_fine = np.linspace(times.min(), times.max(), 50)
        if subj_id in re_df.index:
            b_i = re_df.loc[subj_id].values
        else:
            b_i = np.zeros(len(model.long_submodel_.params_.random_names))

        fitted = [
            model.long_submodel_.marker_value(
                t, b_i, covariate_row, model._time_col, model._long_covariates
            )
            for t in t_fine
        ]
        ax.plot(t_fine, fitted, color=colour, linewidth=1.5, label=str(subj_id))

    # Population average trajectory
    first_row = data.groupby(model._id_col).first().iloc[0]
    t_range = np.linspace(data[model._time_col].min(), data[model._time_col].max(), 100)
    pop_avg = [
        model.long_submodel_.marker_value(
            t, np.zeros(len(model.long_submodel_.params_.random_names)),
            first_row, model._time_col, model._long_covariates
        )
        for t in t_range
    ]
    ax.plot(t_range, pop_avg, color="black", linewidth=2.5, linestyle="--",
            label="Population average", zorder=4)

    ax.set_xlabel(model._time_col.replace("_", " ").title())
    ax.set_ylabel(model._y_col.replace("_", " ").title())
    ax.set_title("Individual Longitudinal Trajectories (BLUP fitted)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_dynamic_risk(
    predictor: "DynamicPredictor",
    figsize: tuple[float, float] = (10, 5),
) -> "matplotlib.figure.Figure":
    """Plot how a subject's 1-month claim risk evolves over time.

    As new telematics readings arrive, the dynamic predictor updates its
    estimate of the driver's claim probability. This plot shows that trajectory.

    Parameters
    ----------
    predictor:
        A DynamicPredictor that has been updated with new measurements.
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    risk_df = predictor.risk_trajectory()
    if len(risk_df) == 0:
        raise ValueError(
            "No risk history available. Call predictor.update() at least once."
        )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Top panel: telematics score history
    time_col = predictor.model._time_col
    y_col = predictor.model._y_col
    hist = predictor.history.sort_values(time_col)
    ax1.plot(hist[time_col], hist[y_col], "o-", color="steelblue", linewidth=1.5)
    ax1.set_ylabel(y_col.replace("_", " ").title())
    ax1.set_title(f"Dynamic Risk Profile — Policy {predictor.subject_id}")
    ax1.axhline(hist[y_col].mean(), color="grey", linestyle=":", alpha=0.7,
                label="Mean score")
    ax1.legend()

    # Bottom panel: 1-month-ahead claim risk
    claim_risk = 1.0 - risk_df["survival_prob_1_unit"]
    ax2.plot(risk_df["time"], claim_risk, "o-", color="crimson", linewidth=1.5)
    ax2.set_ylabel("Claim probability (next month)")
    ax2.set_xlabel(time_col.replace("_", " ").title())
    ax2.set_ylim(0, min(1.0, claim_risk.max() * 1.3))

    fig.tight_layout()
    return fig


def plot_baseline_hazard(
    model: "JointModel",
    figsize: tuple[float, float] = (8, 4),
) -> "matplotlib.figure.Figure":
    """Plot the Breslow cumulative baseline hazard estimate.

    Parameters
    ----------
    model:
        A fitted JointModel.
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if model.surv_submodel_ is None:
        raise RuntimeError("Model has not been fitted.")

    p = model.surv_submodel_.params_
    cum_hazard = np.cumsum(p.baseline_hazard)

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(p.baseline_times, cum_hazard, where="post", color="navy", linewidth=1.5)
    ax.set_xlabel(f"Time ({model._event_time_col.replace('_', ' ')})")
    ax.set_ylabel("Cumulative baseline hazard H₀(t)")
    ax.set_title("Breslow Baseline Hazard Estimate")
    fig.tight_layout()
    return fig


def plot_loglik_convergence(
    model: "JointModel",
    figsize: tuple[float, float] = (8, 4),
) -> "matplotlib.figure.Figure":
    """Plot the log-likelihood convergence across EM iterations.

    A well-behaved EM should show monotone increase in the observed-data
    log-likelihood. Non-monotone behaviour suggests numerical issues in the
    GHQ approximation or the optimisation routines.

    Parameters
    ----------
    model:
        A fitted JointModel (must have been fitted first).
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if not model.loglik_history_:
        raise RuntimeError("No log-likelihood history. Fit the model first.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(model.loglik_history_) + 1), model.loglik_history_,
            "o-", color="darkgreen", linewidth=1.5, markersize=4)
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title(f"EM Convergence ({'Converged' if model.converged_ else 'Did not converge'})")
    ax.axvline(model.n_iter_, color="red", linestyle="--", alpha=0.5, label=f"Final iteration: {model.n_iter_}")
    ax.legend()
    fig.tight_layout()
    return fig
