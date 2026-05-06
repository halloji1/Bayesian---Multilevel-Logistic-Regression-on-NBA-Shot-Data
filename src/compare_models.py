"""Compare Prior A and Prior B fits using LOO cross-validation.

Inputs:  outputs/models/fit_prior_A.nc
         outputs/models/fit_prior_B.nc
Outputs: outputs/reports/model_comparison.csv
         outputs/figures/comparison/loo_compare.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

import config

logger = logging.getLogger(__name__)

_MODEL_A_PATH   = config.MODELS_DIR / "fit_prior_A.nc"
_MODEL_B_PATH   = config.MODELS_DIR / "fit_prior_B.nc"
_REPORT_PATH    = config.REPORTS_DIR / "model_comparison.csv"
_FIGURE_DIR     = config.FIGURES_DIR / "comparison"
_FIGURE_PATH    = _FIGURE_DIR / "loo_compare.png"
_DPI            = 150


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare() -> pd.DataFrame:
    """Compare Prior A and Prior B fits using leave-one-out cross-validation.

    Loads both InferenceData files, computes individual LOO scores, runs
    az.compare() for a side-by-side ranking, saves the result to CSV, and
    produces a comparison plot.

    Returns:
        DataFrame produced by az.compare(), indexed by model name, with
        columns: rank, elpd_loo, p_loo, elpd_diff, weight, se, dse,
        warning, scale.

    Raises:
        FileNotFoundError: If either NetCDF fit file is missing.

    Side effects:
        Writes outputs/reports/model_comparison.csv.
        Writes outputs/figures/comparison/loo_compare.png.
    """
    for path in (_MODEL_A_PATH, _MODEL_B_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Fit file not found: {path}\n"
                "Run `python main.py --fit` (or `python -m src.fit_model --prior A` "
                "and `--prior B`) before comparing."
            )

    # ------------------------------------------------------------------
    # 1. Load InferenceData
    # ------------------------------------------------------------------
    logger.info("Loading %s ...", _MODEL_A_PATH)
    idata_A = az.from_netcdf(str(_MODEL_A_PATH))

    logger.info("Loading %s ...", _MODEL_B_PATH)
    idata_B = az.from_netcdf(str(_MODEL_B_PATH))

    # ------------------------------------------------------------------
    # 2. Individual LOO scores
    # ------------------------------------------------------------------
    logger.info("Computing LOO for Prior A ...")
    loo_A = az.loo(idata_A, pointwise=True)
    logger.info(
        "Prior A — ELPD_LOO: %.2f  SE: %.2f  p_LOO: %.2f",
        loo_A.elpd_loo, loo_A.se, loo_A.p_loo,
    )

    logger.info("Computing LOO for Prior B ...")
    loo_B = az.loo(idata_B, pointwise=True)
    logger.info(
        "Prior B — ELPD_LOO: %.2f  SE: %.2f  p_LOO: %.2f",
        loo_B.elpd_loo, loo_B.se, loo_B.p_loo,
    )

    # ------------------------------------------------------------------
    # 3. Side-by-side comparison
    # ------------------------------------------------------------------
    logger.info("Running az.compare() ...")
    comparison_df = az.compare(
        {"prior_A": idata_A, "prior_B": idata_B},
        ic="loo",
    )
    logger.info("Comparison result:\n%s", comparison_df.to_string())

    # ------------------------------------------------------------------
    # 4. Save CSV
    # ------------------------------------------------------------------
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(_REPORT_PATH)
    logger.info("Saved: %s", _REPORT_PATH)

    # ------------------------------------------------------------------
    # 5. Comparison plot
    # ------------------------------------------------------------------
    _FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 3))
    az.plot_compare(comparison_df, ax=ax)
    ax.set_title("LOO Comparison — Prior A vs Prior B")
    fig.tight_layout()
    fig.savefig(_FIGURE_PATH, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", _FIGURE_PATH)

    return comparison_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config.setup_logging()
    df = compare()
    print(df.to_string())
