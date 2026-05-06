from __future__ import annotations

import argparse
import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

import config
from src.fit_model import _build_model, _load_train_arrays
from src.priors import get_prior

logger = logging.getLogger(__name__)

_FIG_DIR = config.FIGURES_DIR / "posterior"
_DPI     = 150

# Scalar fixed-effect variables written by fit_model
_FIXED_VARS   = ["gamma_00", "gamma_10", "beta_def", "beta_sc", "beta_clutch"]
# Variables for the full summary (adds random-effect SDs from LKJ decomposition)
_SUMMARY_VARS = _FIXED_VARS + ["chol_cov_stds"]

# Human-readable x-axis labels aligned with _FIXED_VARS order
_FIXED_LABELS = {
    "gamma_00":   "Population intercept (gamma_00)",
    "gamma_10":   "Population distance slope (gamma_10)",
    "beta_def":   "Defender distance (β_def)",
    "beta_sc":    "Shot clock (β_sc)",
    "beta_clutch":"Clutch (β_clutch)",
}


# Helpers
def _nc_path(prior_name: str) -> Path:
    return config.MODELS_DIR / f"fit_prior_{prior_name}.nc"


def _fig_path(prior_name: str, stem: str) -> Path:
    return _FIG_DIR / f"{stem}_prior{prior_name}.png"


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def _load_player_index() -> pd.DataFrame:
    path = config.DATA_PROCESSED_DIR / config.PLAYER_INDEX_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Player index not found: {path}\n"
            "Run `python -m src.prepare_features` first."
        )
    return pd.read_csv(path)


def _ensure_ppc(
    idata: az.InferenceData,
    prior_name: str,
    nc_path: Path,
) -> az.InferenceData:
    if hasattr(idata, "posterior_predictive"):
        logger.info("Posterior predictive group already present.")
        return idata

    logger.info(
        "Posterior predictive missing — sampling now "
        "(rebuilds model, may take several minutes) ..."
    )
    prior      = get_prior(prior_name)
    train_path = config.DATA_PROCESSED_DIR / config.TRAIN_CSV
    y, dist_z, def_z, sc_z, clutch, player_idx, n_players = _load_train_arrays(train_path)
    model = _build_model(y, dist_z, def_z, sc_z, clutch, player_idx, n_players, prior)

    with model:
        pm.sample_posterior_predictive(
            idata,
            extend_inferencedata=True,
            random_seed=config.RANDOM_SEED,
        )

    out_path = nc_path.with_name(nc_path.stem + "_with_ppc.nc")
    az.to_netcdf(idata, str(out_path))
    print(f"Saved updated InferenceData with PPC to {out_path}")
    logger.info("Updated NetCDF with posterior predictive: %s", nc_path)
    return idata


# Plot functions
def _plot_forest(idata: az.InferenceData, prior_name: str) -> None:
    """Forest plot of fixed-effect posteriors."""
    az.plot_forest(
        idata,
        var_names=_FIXED_VARS,
        combined=True,
        hdi_prob=0.94,
        figsize=(8, 5),
    )
    fig = plt.gcf()
    fig.suptitle(f"Fixed-Effect Posteriors — Prior {prior_name}", y=1.01)
    fig.tight_layout()
    _save_fig(fig, _fig_path(prior_name, "forest"))


def _plot_caterpillar(
    idata: az.InferenceData,
    prior_name: str,
    player_index: pd.DataFrame,
) -> None:
    beta0_post = idata.posterior["beta0_j"]          # (chain, draw, n_players)
    n_chains, n_draws, n_players = beta0_post.shape
    flat = beta0_post.values.reshape(n_chains * n_draws, n_players)

    means    = flat.mean(axis=0)                      # (n_players,)
    hdi_vals = az.hdi(flat, hdi_prob=0.94)            # (n_players, 2)

    df = pd.DataFrame({
        config.COL_PLAYER_IDX: range(n_players),
        "mean":     means,
        "hdi_low":  hdi_vals[:, 0],
        "hdi_high": hdi_vals[:, 1],
    }).merge(player_index, on=config.COL_PLAYER_IDX)

    top20    = df.nlargest(20, "mean").sort_values("mean", ascending=True)
    bottom20 = df.nsmallest(20, "mean").sort_values("mean", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, subset, title in [
        (axes[0], top20,    f"Top 20 Player Intercepts — Prior {prior_name}"),
        (axes[1], bottom20, f"Bottom 20 Player Intercepts — Prior {prior_name}"),
    ]:
        y_pos = range(len(subset))
        xerr_low  = (subset["mean"] - subset["hdi_low"]).values
        xerr_high = (subset["hdi_high"] - subset["mean"]).values
        ax.errorbar(
            x=subset["mean"].values,
            y=list(y_pos),
            xerr=[xerr_low, xerr_high],
            fmt="o",
            capsize=3,
            markersize=4,
            linewidth=1,
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(subset[config.COL_PLAYER_NAME].values, fontsize=8)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("log-odds intercept")
        ax.set_title(title)

    fig.tight_layout()
    _save_fig(fig, _fig_path(prior_name, "caterpillar"))


def _plot_ppc(idata: az.InferenceData, prior_name: str) -> None:
    ax = az.plot_ppc(
        idata,
        var_names=["y_obs"],
        num_pp_samples=200,
        alpha=0.3,
        figsize=(8, 4),
    )
    fig = ax.get_figure() if hasattr(ax, "get_figure") else plt.gcf()
    fig.suptitle(f"Posterior Predictive Check — Prior {prior_name}", y=1.02)
    fig.tight_layout()
    _save_fig(fig, _fig_path(prior_name, "ppc"))


def _plot_trace(idata: az.InferenceData, prior_name: str) -> None:
    axes = az.plot_trace(
        idata,
        var_names=_FIXED_VARS,
        combined=False,
        figsize=(12, 2.5 * len(_FIXED_VARS)),
    )
    fig = plt.gcf()
    fig.suptitle(f"Trace — Prior {prior_name}", y=1.005)
    fig.tight_layout()
    _save_fig(fig, _fig_path(prior_name, "trace"))


# Public API
def analyze(prior_name: str) -> None:
    nc_path = _nc_path(prior_name)
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Fit file not found: {nc_path}\n"
            "Run `python -m src.fit_model --prior {prior_name}` first."
        )

    logger.info("Loading %s ...", nc_path)
    idata = az.from_netcdf(str(nc_path))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    player_index = _load_player_index()

    # 1. Summary table
    logger.info("Computing posterior summary ...")
    summary_df = az.summary(
        idata,
        var_names=_SUMMARY_VARS,
        round_to=4,
    )
    report_path = config.REPORTS_DIR / f"posterior_summary_prior{prior_name}.csv"
    summary_df.to_csv(report_path)
    logger.info("Saved: %s", report_path)
    logger.info("\n%s", summary_df.to_string())

    # 2. Forest plot
    logger.info("Generating forest plot ...")
    _plot_forest(idata, prior_name)

    # 3. Caterpillar plot
    logger.info("Generating caterpillar plot ...")
    _plot_caterpillar(idata, prior_name, player_index)

    # 4. Posterior predictive check
    logger.info("Generating PPC plot ...")
    idata = _ensure_ppc(idata, prior_name, nc_path)
    _plot_ppc(idata, prior_name)

    # 5. Trace plot
    logger.info("Generating trace plot ...")
    _plot_trace(idata, prior_name)

    logger.info(
        "Posterior analysis complete for Prior %s. "
        "Outputs in %s and %s",
        prior_name, _FIG_DIR, config.REPORTS_DIR,
    )


if __name__ == "__main__":
    config.setup_logging()
    parser = argparse.ArgumentParser(
        description="Posterior summaries and plots for a fitted prior model."
    )
    parser.add_argument(
        "--prior",
        choices=["A", "B"],
        required=True,
        help='"A" or "B".',
    )
    args = parser.parse_args()
    analyze(prior_name=args.prior)
