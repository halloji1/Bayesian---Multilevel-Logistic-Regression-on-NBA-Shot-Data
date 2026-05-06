"""Side-by-side sensitivity analysis of Prior A vs Prior B posteriors.

Inputs:  outputs/models/fit_prior_A.nc
         outputs/models/fit_prior_B.nc
Outputs: outputs/reports/sensitivity_analysis.csv
         outputs/figures/sensitivity/overlay_{param}.png  (one per fixed effect)
"""

from __future__ import annotations

import logging
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config
from src.priors import get_prior

logger = logging.getLogger(__name__)

_MODEL_A_PATH = config.MODELS_DIR / "fit_prior_A.nc"
_MODEL_B_PATH = config.MODELS_DIR / "fit_prior_B.nc"
_REPORT_PATH  = config.REPORTS_DIR / "sensitivity_analysis.csv"
_FIG_DIR      = config.FIGURES_DIR / "sensitivity"
_DPI          = 150
_HDI_PROB     = 0.95

# ---------------------------------------------------------------------------
# Parameter registry
# ---------------------------------------------------------------------------

# Scalar fixed-effect variable names as they appear in idata.posterior.
_FIXED_PARAMS = ["gamma_00", "gamma_10", "beta_def", "beta_sc", "beta_clutch"]

# Vector posterior variables: (pymc_name, slice_idx, row_label)
_VECTOR_PARAMS = [
    ("chol_cov_stds", 0, "chol_cov_stds[0]"),   # random intercept SD
    ("chol_cov_stds", 1, "chol_cov_stds[1]"),   # random distance slope SD
]

_PARAM_LABELS = {
    "gamma_00":         "Population intercept (gamma_00)",
    "gamma_10":         "Population distance slope (gamma_10)",
    "beta_def":         "Defender distance (β_def)",
    "beta_sc":          "Shot clock (β_sc)",
    "beta_clutch":      "Clutch (β_clutch)",
    "chol_cov_stds[0]": "RE SD — player intercept",
    "chol_cov_stds[1]": "RE SD — player distance slope",
}

# Maps scalar fixed-effect names to the location in the prior dict.
_PRIOR_KEY_FN: dict[str, callable] = {
    "gamma_00":    lambda p: p["intercept"],
    "gamma_10":    lambda p: p["slopes"]["distance"],
    "beta_def":    lambda p: p["slopes"]["def_dist"],
    "beta_sc":     lambda p: p["slopes"]["shot_clock"],
    "beta_clutch": lambda p: p["slopes"]["clutch"],
}

# Research hypotheses: param label (must match _PARAM_LABELS key), expected
# direction ("negative": 95% HDI < 0, "positive": 95% HDI > 0), description.
_HYPOTHESES: dict[str, tuple[str, str, str]] = {
    "H1": ("gamma_10",         "negative", "Shot distance reduces FG% (slope < 0)"),
    "H2": ("beta_def",         "positive", "Open shots improve FG% (def_dist slope > 0)"),
    "H3": ("beta_sc",          "positive", "More shot-clock time improves FG% (slope > 0)"),
    "H4": ("beta_clutch",      "negative", "Clutch situations reduce FG% (slope < 0)"),
    "H5": ("chol_cov_stds[0]", "positive", "Meaningful player intercept heterogeneity (RE SD > 0)"),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_samples(idata: az.InferenceData, var: str, idx: int | None) -> np.ndarray:
    """Return flat posterior sample array for a scalar or indexed vector param."""
    raw = idata.posterior[var].values          # (chain, draw) or (chain, draw, k)
    if idx is not None:
        raw = raw[:, :, idx]
    return raw.flatten()


def _hdi(samples: np.ndarray) -> tuple[float, float]:
    """Return (low, high) of the HDI_PROB highest density interval."""
    interval = az.hdi(samples, hdi_prob=_HDI_PROB)
    return float(interval[0]), float(interval[1])


def _overlap_pct(hdi_a: tuple[float, float], hdi_b: tuple[float, float]) -> float:
    """Percentage of the combined span that the two HDIs share.

    Uses Jaccard-style: overlap / union × 100.
    """
    lo = max(hdi_a[0], hdi_b[0])
    hi = min(hdi_a[1], hdi_b[1])
    overlap = max(0.0, hi - lo)
    union   = max(hdi_a[1], hdi_b[1]) - min(hdi_a[0], hdi_b[0])
    return 100.0 * overlap / union if union > 0 else 0.0


def _sign_agreement(hdi_a: tuple[float, float], hdi_b: tuple[float, float]) -> bool:
    """True if both HDIs exclude zero on the same side."""
    a_pos = hdi_a[0] > 0
    a_neg = hdi_a[1] < 0
    b_pos = hdi_b[0] > 0
    b_neg = hdi_b[1] < 0
    return (a_pos and b_pos) or (a_neg and b_neg)


def _supports_hypothesis(hdi: tuple[float, float], direction: str) -> bool:
    """True if the HDI excludes zero in the predicted direction."""
    if direction == "positive":
        return hdi[0] > 0
    if direction == "negative":
        return hdi[1] < 0
    return False


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def _build_param_rows(
    idata_A: az.InferenceData,
    idata_B: az.InferenceData,
) -> pd.DataFrame:
    """Compute posterior statistics for every parameter; return summary DataFrame."""
    rows: list[dict] = []

    # Scalar fixed effects
    for var in _FIXED_PARAMS:
        samples_A = _extract_samples(idata_A, var, None)
        samples_B = _extract_samples(idata_B, var, None)
        hdi_A = _hdi(samples_A)
        hdi_B = _hdi(samples_B)
        rows.append({
            "parameter":       var,
            "label":           _PARAM_LABELS[var],
            "mean_A":          float(samples_A.mean()),
            "mean_B":          float(samples_B.mean()),
            "hdi_low_A":       hdi_A[0],
            "hdi_high_A":      hdi_A[1],
            "hdi_low_B":       hdi_B[0],
            "hdi_high_B":      hdi_B[1],
            "mean_shift":      float(samples_B.mean()) - float(samples_A.mean()),
            "hdi_overlap_pct": _overlap_pct(hdi_A, hdi_B),
            "sign_agreement":  _sign_agreement(hdi_A, hdi_B),
        })

    # Random-effect SD components
    for pymc_var, idx, label in _VECTOR_PARAMS:
        samples_A = _extract_samples(idata_A, pymc_var, idx)
        samples_B = _extract_samples(idata_B, pymc_var, idx)
        hdi_A = _hdi(samples_A)
        hdi_B = _hdi(samples_B)
        rows.append({
            "parameter":       label,
            "label":           _PARAM_LABELS[label],
            "mean_A":          float(samples_A.mean()),
            "mean_B":          float(samples_B.mean()),
            "hdi_low_A":       hdi_A[0],
            "hdi_high_A":      hdi_A[1],
            "hdi_low_B":       hdi_B[0],
            "hdi_high_B":      hdi_B[1],
            "mean_shift":      float(samples_B.mean()) - float(samples_A.mean()),
            "hdi_overlap_pct": _overlap_pct(hdi_A, hdi_B),
            "sign_agreement":  _sign_agreement(hdi_A, hdi_B),
        })

    return pd.DataFrame(rows).set_index("parameter")


def _build_hypothesis_rows(param_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate each hypothesis against the computed HDIs; return summary DataFrame."""
    rows: list[dict] = []
    for hyp_id, (param, direction, description) in _HYPOTHESES.items():
        hdi_A = (param_df.loc[param, "hdi_low_A"], param_df.loc[param, "hdi_high_A"])
        hdi_B = (param_df.loc[param, "hdi_low_B"], param_df.loc[param, "hdi_high_B"])
        sup_A  = _supports_hypothesis(hdi_A, direction)
        sup_B  = _supports_hypothesis(hdi_B, direction)
        rows.append({
            "hypothesis":  hyp_id,
            "parameter":   param,
            "direction":   direction,
            "description": description,
            "supported_A": sup_A,
            "supported_B": sup_B,
            "agreement":   sup_A == sup_B,
        })
        logger.info(
            "%-3s %-22s  A=%s  B=%s  agree=%s  | %s",
            hyp_id, param,
            "YES" if sup_A else "NO ",
            "YES" if sup_B else "NO ",
            "YES" if sup_A == sup_B else "NO ",
            description,
        )
    return pd.DataFrame(rows).set_index("hypothesis")


# ---------------------------------------------------------------------------
# Overlay density plots
# ---------------------------------------------------------------------------

def _overlay_plot(
    idata_A: az.InferenceData,
    idata_B: az.InferenceData,
    var: str,
    prior_A: dict,
    prior_B: dict,
) -> None:
    """Posterior + prior overlay for a single fixed-effect parameter."""
    samples_A = _extract_samples(idata_A, var, None)
    samples_B = _extract_samples(idata_B, var, None)

    # x range that covers posteriors and priors
    all_vals = np.concatenate([samples_A, samples_B])
    mu_A, sig_A = _PRIOR_KEY_FN[var](prior_A)["mu"], _PRIOR_KEY_FN[var](prior_A)["sigma"]
    mu_B, sig_B = _PRIOR_KEY_FN[var](prior_B)["mu"], _PRIOR_KEY_FN[var](prior_B)["sigma"]
    x_lo = min(all_vals.min(), mu_A - 3 * sig_A, mu_B - 3 * sig_B)
    x_hi = max(all_vals.max(), mu_A + 3 * sig_A, mu_B + 3 * sig_B)
    x    = np.linspace(x_lo, x_hi, 500)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Posterior KDEs
    sns.kdeplot(samples_A, ax=ax, color="steelblue", linewidth=2,
                label="Posterior A")
    sns.kdeplot(samples_B, ax=ax, color="tomato",    linewidth=2,
                label="Posterior B")

    # Analytical prior densities (dashed)
    ax.plot(x, stats.norm.pdf(x, mu_A, sig_A),
            color="steelblue", linestyle="--", linewidth=1.2, alpha=0.7,
            label="Prior A (Normal)")
    ax.plot(x, stats.norm.pdf(x, mu_B, sig_B),
            color="tomato",    linestyle="--", linewidth=1.2, alpha=0.7,
            label="Prior B (Normal)")

    ax.axvline(0, color="grey", linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Density")
    ax.set_title(f"Prior Sensitivity — {_PARAM_LABELS.get(var, var)}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    safe_name = var.replace("[", "").replace("]", "")
    out_path  = _FIG_DIR / f"overlay_{safe_name}.png"
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_sensitivity() -> pd.DataFrame:
    """Compare Prior A and Prior B posteriors across all fixed-effect parameters.

    Steps:
        1. Load both InferenceData objects.
        2. Compute posterior means, 95% HDIs, mean shift, HDI overlap %, and
           sign agreement for fixed effects and random-effect SDs.
        3. Evaluate five directional hypotheses (H1–H5) against both posteriors.
        4. Save the parameter table and hypothesis table to
           outputs/reports/sensitivity_analysis.csv (two sections, blank row
           between them).
        5. Save one overlay density PNG per scalar fixed effect to
           outputs/figures/sensitivity/.

    Returns:
        Parameter-level summary DataFrame (rows = parameters, includes all
        computed statistics).

    Raises:
        FileNotFoundError: If either fit NetCDF is missing.
    """
    for path in (_MODEL_A_PATH, _MODEL_B_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Fit file not found: {path}\n"
                "Run `python -m src.fit_model --prior A` and `--prior B` first."
            )

    logger.info("Loading fit_prior_A.nc ...")
    idata_A = az.from_netcdf(str(_MODEL_A_PATH))

    logger.info("Loading fit_prior_B.nc ...")
    idata_B = az.from_netcdf(str(_MODEL_B_PATH))

    prior_A = get_prior("A")
    prior_B = get_prior("B")

    # ------------------------------------------------------------------
    # 1. Parameter-level statistics
    # ------------------------------------------------------------------
    logger.info("Computing parameter statistics ...")
    param_df = _build_param_rows(idata_A, idata_B)

    for row in param_df.itertuples():
        logger.info(
            "  %-22s  mean_A=%7.4f  mean_B=%7.4f  shift=%+7.4f  "
            "overlap=%.1f%%  sign_agree=%s",
            row.Index, row.mean_A, row.mean_B, row.mean_shift,
            row.hdi_overlap_pct, row.sign_agreement,
        )

    # ------------------------------------------------------------------
    # 2. Hypothesis-level summary
    # ------------------------------------------------------------------
    logger.info("Evaluating hypotheses H1–H5 ...")
    hyp_df = _build_hypothesis_rows(param_df)

    # ------------------------------------------------------------------
    # 3. Save CSV (two sections)
    # ------------------------------------------------------------------
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "w") as fh:
        fh.write("# Parameter sensitivity summary\n")
        param_df.to_csv(fh)
        fh.write("\n# Hypothesis evaluation\n")
        hyp_df.to_csv(fh)
    logger.info("Saved: %s", _REPORT_PATH)

    # ------------------------------------------------------------------
    # 4. Overlay density plots
    # ------------------------------------------------------------------
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme()
    for var in _FIXED_PARAMS:
        logger.info("Generating overlay plot for %s ...", var)
        _overlay_plot(idata_A, idata_B, var, prior_A, prior_B)

    logger.info("Sensitivity analysis complete.")
    return param_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config.setup_logging()
    df = run_sensitivity()
    print(df.to_string())
