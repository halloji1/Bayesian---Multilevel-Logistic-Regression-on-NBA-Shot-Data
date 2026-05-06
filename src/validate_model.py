"""Evaluate fitted models on the held-out test set.

Inputs:  outputs/models/fit_prior_{A|B}.nc
         data/processed/test.csv
Outputs: outputs/reports/validation_metrics.csv
         outputs/figures/validation/calibration_prior{A|B}.png
         outputs/figures/validation/roc_prior{A|B}.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

_FIG_DIR     = config.FIGURES_DIR / "validation"
_METRICS_CSV = config.REPORTS_DIR / "validation_metrics.csv"
_DPI         = 150
# Posterior draws processed per iteration to cap peak memory.
# 250 draws × ~24 K test shots × 8 bytes ≈ 48 MB per chunk.
_CHUNK_SIZE  = 250


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nc_path(prior_name: str) -> Path:
    return config.MODELS_DIR / f"fit_prior_{prior_name}.nc"


def _load_test_arrays(
    test_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read test CSV and return typed numpy arrays.

    Returns:
        (y, dist_z, def_z, sc_z, clutch, player_idx)
    """
    df = pd.read_csv(test_path)
    return (
        df[config.COL_SHOT_RESULT].to_numpy(dtype=int),
        df[f"{config.COL_SHOT_DIST}_z"].to_numpy(dtype=float),
        df[f"{config.COL_CLOSE_DEF}_z"].to_numpy(dtype=float),
        df[f"{config.COL_SHOT_CLOCK}_z"].to_numpy(dtype=float),
        df["clutch"].to_numpy(dtype=float),
        df[config.COL_PLAYER_IDX].to_numpy(dtype=int),
    )


def _compute_p_hat(
    idata: az.InferenceData,
    dist_z: np.ndarray,
    def_z: np.ndarray,
    sc_z: np.ndarray,
    clutch: np.ndarray,
    player_idx: np.ndarray,
) -> np.ndarray:
    """Compute per-shot posterior-averaged predicted probability.

    Reconstructs the linear predictor for every posterior draw using the
    sampled fixed effects and player-level parameters, applies the logistic
    sigmoid, then averages across all draws.  Draw processing is chunked so
    peak memory stays bounded regardless of chain/draw count.

    Args:
        idata:      InferenceData containing the posterior group.
        dist_z:     Standardized shot-distance values, shape (n_test,).
        def_z:      Standardized defender-distance values.
        sc_z:       Standardized shot-clock values.
        clutch:     Binary clutch-situation flag.
        player_idx: 0-indexed player factor, shape (n_test,).

    Returns:
        Array of shape (n_test,) with values in (0, 1).
    """
    n_chains, n_draws, n_players = idata.posterior["beta0_j"].shape
    n_samples = n_chains * n_draws
    n_test    = len(dist_z)

    # Sanity check: all test player indices must exist in the posterior
    if player_idx.max() >= n_players:
        raise ValueError(
            f"Test set contains player_idx={player_idx.max()} but posterior "
            f"has only {n_players} players. Check that train/test use the same "
            "player_index.csv."
        )

    # Extract all posterior arrays upfront (small total: ~36 MB for default settings)
    b0_flat    = idata.posterior["beta0_j"].values.reshape(n_samples, n_players)
    bd_flat    = idata.posterior["beta_dist_j"].values.reshape(n_samples, n_players)
    bdef_flat  = idata.posterior["beta_def"].values.flatten()       # (n_samples,)
    bsc_flat   = idata.posterior["beta_sc"].values.flatten()
    bcl_flat   = idata.posterior["beta_clutch"].values.flatten()

    p_sum   = np.zeros(n_test, dtype=np.float64)
    n_chunks = int(np.ceil(n_samples / _CHUNK_SIZE))

    for i in tqdm(range(n_chunks), desc=f"Posterior predictions", unit="chunk", leave=False):
        lo, hi = i * _CHUNK_SIZE, min((i + 1) * _CHUNK_SIZE, n_samples)

        # eta shape: (chunk, n_test)
        eta = (
            b0_flat[lo:hi][:, player_idx]
            + bd_flat[lo:hi][:, player_idx] * dist_z
            + bdef_flat[lo:hi, np.newaxis]  * def_z
            + bsc_flat[lo:hi, np.newaxis]   * sc_z
            + bcl_flat[lo:hi, np.newaxis]   * clutch
        )
        p_sum += (1.0 / (1.0 + np.exp(-eta))).sum(axis=0)

    return p_sum / n_samples


def _plot_calibration(y: np.ndarray, p_hat: np.ndarray, prior_name: str) -> None:
    prob_true, prob_pred = calibration_curve(y, p_hat, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, "o-", linewidth=1.5, label=f"Prior {prior_name}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.15)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration — Prior {prior_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    out = _FIG_DIR / f"calibration_prior{prior_name}.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def _plot_roc(y: np.ndarray, p_hat: np.ndarray, prior_name: str, auc: float) -> None:
    fpr, tpr, _ = roc_curve(y, p_hat)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=1.5, label=f"Prior {prior_name}  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC Curve — Prior {prior_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = _FIG_DIR / f"roc_prior{prior_name}.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def _upsert_metrics(prior_name: str, metrics: dict[str, float]) -> pd.DataFrame:
    """Insert or replace the row for prior_name in validation_metrics.csv."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if _METRICS_CSV.exists():
        df = pd.read_csv(_METRICS_CSV, index_col=0)
    else:
        df = pd.DataFrame(columns=list(metrics.keys()))
    df.loc[f"prior_{prior_name}"] = metrics
    df.index.name = "model"
    df.to_csv(_METRICS_CSV)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(prior_name: str) -> dict[str, float]:
    """Evaluate one prior fit on the held-out test set.

    Generates posterior-averaged predicted probabilities for each test shot
    using all posterior draws of fixed effects (beta_def, beta_sc,
    beta_clutch) and player-level effects (beta0_j, beta_dist_j).  No model
    rebuild is required — everything is computed from the saved InferenceData.

    Args:
        prior_name: "A" or "B".

    Returns:
        Dict with keys: auc, brier_score, log_loss.

    Raises:
        FileNotFoundError: If the NetCDF fit file or test.csv is missing.

    Side effects:
        Writes / updates outputs/reports/validation_metrics.csv.
        Writes two PNG figures to outputs/figures/validation/.
    """
    nc_path   = _nc_path(prior_name)
    test_path = config.DATA_PROCESSED_DIR / config.TEST_CSV

    for label, path in [("Fit file", nc_path), ("Test CSV", test_path)]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                "Run the pipeline steps before validate."
            )

    logger.info("Loading %s ...", nc_path)
    idata = az.from_netcdf(str(nc_path))

    logger.info("Loading %s ...", test_path)
    y, dist_z, def_z, sc_z, clutch, player_idx = _load_test_arrays(test_path)
    logger.info("Test set: %d shots, observed FG%% = %.3f", len(y), y.mean())

    # ── Posterior-averaged predictions ───────────────────────────────────
    logger.info("Computing posterior predictive probabilities ...")
    p_hat = _compute_p_hat(idata, dist_z, def_z, sc_z, clutch, player_idx)
    logger.info("Mean predicted FG%% = %.3f", p_hat.mean())

    # ── Metrics ──────────────────────────────────────────────────────────
    auc     = float(roc_auc_score(y, p_hat))
    brier   = float(brier_score_loss(y, p_hat))
    logloss = float(log_loss(y, p_hat))
    metrics = {"auc": auc, "brier_score": brier, "log_loss": logloss}

    logger.info(
        "Prior %s — AUC=%.4f  Brier=%.4f  LogLoss=%.4f",
        prior_name, auc, brier, logloss,
    )

    # ── Figures ──────────────────────────────────────────────────────────
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_calibration(y, p_hat, prior_name)
    _plot_roc(y, p_hat, prior_name, auc)

    # ── Persist metrics ───────────────────────────────────────────────────
    full_df = _upsert_metrics(prior_name, metrics)
    logger.info("Metrics table:\n%s", full_df.to_string())

    return metrics


def validate_all() -> pd.DataFrame:
    """Run validate() for Prior A then Prior B; return combined metrics DataFrame."""
    for prior_name in ("A", "B"):
        validate(prior_name)
    return pd.read_csv(_METRICS_CSV, index_col=0)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config.setup_logging()
    parser = argparse.ArgumentParser(
        description="Validate fitted models on the held-out test set."
    )
    parser.add_argument(
        "--prior",
        choices=["A", "B"],
        default=None,
        help='Run for a single prior. Omit to run both A and B.',
    )
    args = parser.parse_args()
    if args.prior:
        result = validate(args.prior)
        print(pd.Series(result).to_string())
    else:
        df = validate_all()
        print(df.to_string())
