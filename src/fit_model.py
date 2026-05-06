from __future__ import annotations

import argparse
import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

import config
from src.priors import get_prior

logger = logging.getLogger(__name__)


# Helpers
def _output_path(prior_name: str) -> Path:
    return config.MODELS_DIR / f"fit_prior_{prior_name}.nc"


def _load_train_arrays(
    train_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    df = pd.read_csv(train_path)

    y          = df[config.COL_SHOT_RESULT].to_numpy(dtype=int)
    dist_z     = df[f"{config.COL_SHOT_DIST}_z"].to_numpy(dtype=float)
    def_z      = df[f"{config.COL_CLOSE_DEF}_z"].to_numpy(dtype=float)
    sc_z       = df[f"{config.COL_SHOT_CLOCK}_z"].to_numpy(dtype=float)
    clutch     = df["clutch"].to_numpy(dtype=float)
    player_idx = df[config.COL_PLAYER_IDX].to_numpy(dtype=int)
    n_players  = int(player_idx.max()) + 1

    logger.info(
        "Train data: %d shots, %d players, FG%% = %.3f",
        len(y), n_players, y.mean(),
    )
    return y, dist_z, def_z, sc_z, clutch, player_idx, n_players


def _build_model(
    y: np.ndarray,
    dist_z: np.ndarray,
    def_z: np.ndarray,
    sc_z: np.ndarray,
    clutch: np.ndarray,
    player_idx: np.ndarray,
    n_players: int,
    prior: dict,
) -> pm.Model:
    with pm.Model() as model:

        # Population-level means (hyperpriors)
        gamma_00 = pm.Normal(
            "gamma_00",
            mu=prior["intercept"]["mu"],
            sigma=prior["intercept"]["sigma"],
        )
        gamma_10 = pm.Normal(
            "gamma_10",
            mu=prior["slopes"]["distance"]["mu"],
            sigma=prior["slopes"]["distance"]["sigma"],
        )

        # Fixed effects
        beta_def = pm.Normal(
            "beta_def",
            mu=prior["slopes"]["def_dist"]["mu"],
            sigma=prior["slopes"]["def_dist"]["sigma"],
        )
        beta_sc = pm.Normal(
            "beta_sc",
            mu=prior["slopes"]["shot_clock"]["mu"],
            sigma=prior["slopes"]["shot_clock"]["sigma"],
        )
        beta_clutch = pm.Normal(
            "beta_clutch",
            mu=prior["slopes"]["clutch"]["mu"],
            sigma=prior["slopes"]["clutch"]["sigma"],
        )

        # LKJ Cholesky covariance for (intercept, distance slope)
        sd_dist = pm.HalfNormal.dist(
            sigma=prior["random_effect_sd"]["sigma"], shape=2
        )
        chol_cov, corr, stds = pm.LKJCholeskyCov(
            "chol_cov",
            n=2,
            eta=prior["lkj_eta"],
            sd_dist=sd_dist,
            compute_corr=True,
        )

        # Non-centered player offsets
        z_player = pm.Normal(
            "z_player", mu=0.0, sigma=1.0, shape=(n_players, 2)
        )
        player_offsets = pt.dot(z_player, chol_cov.T)

        # Player-specific parameters
        beta0_j = pm.Deterministic(
            "beta0_j", gamma_00 + player_offsets[:, 0]
        )
        beta_dist_j = pm.Deterministic(
            "beta_dist_j", gamma_10 + player_offsets[:, 1]
        )

        # Linear predictor
        eta = (
            beta0_j[player_idx]
            + beta_dist_j[player_idx] * dist_z
            + beta_def * def_z
            + beta_sc * sc_z
            + beta_clutch * clutch
        )

        # Likelihood
        p = pm.math.invlogit(eta)
        pm.Bernoulli("y_obs", p=p, observed=y)

    return model


def _log_diagnostics(idata: az.InferenceData) -> None:
    scalar_vars = ["gamma_00", "gamma_10", "beta_def", "beta_sc", "beta_clutch"]
    summary = az.summary(idata, var_names=scalar_vars, round_to=4)

    rhat_max      = float(summary["r_hat"].max())
    ess_bulk_min  = float(summary["ess_bulk"].min())
    ess_tail_min  = float(summary["ess_tail"].min())
    n_divergences = int(idata.sample_stats["diverging"].values.sum())

    logger.info("─── MCMC Diagnostics (scalar parameters) ───")
    logger.info("  Max R-hat       : %.4f  (target ≤ 1.01)", rhat_max)
    logger.info("  Min ESS bulk    : %.0f", ess_bulk_min)
    logger.info("  Min ESS tail    : %.0f", ess_tail_min)
    logger.info("  Divergences     : %d", n_divergences)

    if rhat_max > 1.01:
        logger.warning("R-hat > 1.01 — chains may not have converged.")
    if n_divergences > 0:
        logger.warning(
            "%d divergences detected — consider increasing target_accept "
            "or reviewing the parameterization.",
            n_divergences,
        )


# Public API
def fit(prior_name: str, force: bool = False) -> az.InferenceData:
    output_path = _output_path(prior_name)

    if output_path.exists() and not force:
        logger.info("Skipping; fit exists: %s", output_path)
        return az.from_netcdf(str(output_path))

    prior = get_prior(prior_name)          # raises ValueError for unknown name
    logger.info("Prior %s: %s", prior_name, prior["description"])

    train_path = config.DATA_PROCESSED_DIR / config.TRAIN_CSV
    if not train_path.exists():
        raise FileNotFoundError(
            f"Train CSV not found: {train_path}\n"
            "Run `python -m src.prepare_features` first."
        )

    y, dist_z, def_z, sc_z, clutch, player_idx, n_players = _load_train_arrays(train_path)

    logger.info("Building PyMC model (prior=%s, n_players=%d) ...", prior_name, n_players)
    model = _build_model(y, dist_z, def_z, sc_z, clutch, player_idx, n_players, prior)

    logger.info(
        "Sampling: chains=%d  draws=%d  tune=%d  target_accept=%.2f  seed=%d",
        config.N_CHAINS, config.N_DRAWS, config.N_TUNE,
        config.TARGET_ACCEPT, config.RANDOM_SEED,
    )

    with model:
        try:
            idata = pm.sample(
                draws=config.N_DRAWS,
                tune=config.N_TUNE,
                chains=config.N_CHAINS,
                target_accept=config.TARGET_ACCEPT,
                random_seed=config.RANDOM_SEED,
                progressbar=True,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
            )
        except Exception as exc:
            logger.error("Sampling failed: %s", exc)
            raise

    n_divergences = int(idata.sample_stats["diverging"].values.sum())
    total_samples = config.N_CHAINS * config.N_DRAWS
    if n_divergences > 0.05 * total_samples:
        msg = (
            f"{n_divergences} divergences ({100 * n_divergences / total_samples:.1f}% "
            f"of {total_samples} samples). "
            "Increase target_accept or review parameterization."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    _log_diagnostics(idata)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, str(output_path))
    logger.info("Saved: %s", output_path)

    return idata


if __name__ == "__main__":
    config.setup_logging()
    parser = argparse.ArgumentParser(
        description="Fit Bayesian multilevel logistic regression on NBA shot data."
    )
    parser.add_argument(
        "--prior",
        choices=["A", "B"],
        required=True,
        help='"A" = weakly informative, "B" = strongly informative.',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fit even if outputs/models/fit_prior_{A|B}.nc already exists.",
    )
    args = parser.parse_args()
    fit(prior_name=args.prior, force=args.force)
