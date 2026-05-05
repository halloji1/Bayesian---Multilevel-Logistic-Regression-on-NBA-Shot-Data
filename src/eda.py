"""Generate exploratory plots and summary statistics from shots_clean.csv.

Input:   data/processed/shots_clean.csv
Outputs: outputs/figures/eda/*.png
         outputs/reports/eda_summary.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

logger = logging.getLogger(__name__)

_EDA_DIR     = config.FIGURES_DIR / "eda"
_REPORT_PATH = config.REPORTS_DIR / "eda_summary.txt"
_DPI         = 150

_DIST_BINS   = [0, 3, 10, 16, 23, float("inf")]
_DIST_LABELS = ["0-3 ft", "3-10 ft", "10-16 ft", "16-23 ft", "23+ ft"]

_CORR_COLS = [
    config.COL_SHOT_DIST,
    config.COL_CLOSE_DEF,
    config.COL_SHOT_CLOCK,
    config.COL_TOUCH_TIME,
]
_CORR_LABELS = ["Shot Dist", "Def Dist", "Shot Clock", "Touch Time"]

_PCT_FMT = plt.FuncFormatter(lambda x, _: f"{x:.0%}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> None:
    path = _EDA_DIR / filename
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Plot 1: shot distance distribution
# ---------------------------------------------------------------------------

def _plot_shot_dist_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        df[config.COL_SHOT_DIST],
        bins=50,
        kde=True,
        ax=ax,
    )
    ax.set_xlabel("Shot Distance (ft)")
    ax.set_ylabel("Count")
    ax.set_title("Shot Distance Distribution")
    _save(fig, "01_shot_distance_distribution.png")


# ---------------------------------------------------------------------------
# Plot 2: FG% by distance bucket
# ---------------------------------------------------------------------------

def _plot_fg_pct_by_distance(df: pd.DataFrame) -> None:
    tmp = df.copy()
    tmp["dist_bucket"] = pd.cut(
        tmp[config.COL_SHOT_DIST],
        bins=_DIST_BINS,
        labels=_DIST_LABELS,
        right=False,
    )
    bucket_fg = (
        tmp.groupby("dist_bucket", observed=True)[config.COL_SHOT_RESULT]
        .mean()
        .reset_index(name="fg_pct")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=bucket_fg, x="dist_bucket", y="fg_pct", ax=ax)
    ax.set_ylim(0, 0.75)
    ax.yaxis.set_major_formatter(_PCT_FMT)
    ax.set_xlabel("Distance Bucket")
    ax.set_ylabel("Field Goal %")
    ax.set_title("FG% by Shot Distance Bucket")
    _save(fig, "02_fg_pct_by_distance_bucket.png")


# ---------------------------------------------------------------------------
# Plot 3: FG% by player — top 20 and bottom 20
# ---------------------------------------------------------------------------

def _plot_fg_pct_by_player(df: pd.DataFrame, min_shots: int) -> None:
    player_fg = (
        df.groupby(config.COL_PLAYER_NAME)[config.COL_SHOT_RESULT]
        .agg(fg_pct="mean", shots="count")
        .query(f"shots >= {min_shots}")
        .sort_values("fg_pct")
    )

    top20    = player_fg.tail(20).iloc[::-1]
    bottom20 = player_fg.head(20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, subset, title in [
        (axes[0], top20,    f"Top 20 Players by FG% (≥{min_shots} shots)"),
        (axes[1], bottom20, f"Bottom 20 Players by FG% (≥{min_shots} shots)"),
    ]:
        ax.barh(subset.index, subset["fg_pct"])
        ax.xaxis.set_major_formatter(_PCT_FMT)
        ax.set_xlabel("Field Goal %")
        ax.set_title(title)
        ax.invert_yaxis()

    fig.tight_layout()
    _save(fig, "03_fg_pct_by_player_top_bottom.png")


# ---------------------------------------------------------------------------
# Plot 4: defender distance distribution split by shot result
# ---------------------------------------------------------------------------

def _plot_defender_dist_hist(df: pd.DataFrame) -> None:
    tmp = df[[config.COL_CLOSE_DEF, config.COL_SHOT_RESULT]].copy()
    tmp["Result"] = tmp[config.COL_SHOT_RESULT].map({1: "Made", 0: "Missed"})

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=tmp,
        x=config.COL_CLOSE_DEF,
        hue="Result",
        bins=40,
        kde=True,
        stat="density",
        common_norm=False,
        alpha=0.45,
        ax=ax,
    )
    ax.set_xlabel("Closest Defender Distance (ft)")
    ax.set_ylabel("Density")
    ax.set_title("Defender Distance Distribution by Shot Result")
    _save(fig, "04_defender_distance_by_result.png")


# ---------------------------------------------------------------------------
# Plot 5: correlation heatmap
# ---------------------------------------------------------------------------

def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[_CORR_COLS].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=_CORR_LABELS,
        yticklabels=_CORR_LABELS,
        ax=ax,
    )
    ax.set_title("Predictor Correlation Heatmap")
    fig.tight_layout()
    _save(fig, "05_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Summary text report
# ---------------------------------------------------------------------------

def _write_summary(df: pd.DataFrame) -> None:
    stats = df[_CORR_COLS].agg(["mean", "median", "std"]).T
    stats.columns = ["Mean", "Median", "SD"]

    lines = [
        "=" * 60,
        "EDA SUMMARY — NBA Shot Logs",
        "=" * 60,
        f"Total shots        : {len(df):>10,}",
        f"Unique players     : {df[config.COL_PLAYER_ID].nunique():>10,}",
        f"League FG%         : {df[config.COL_SHOT_RESULT].mean():>10.3f}",
        f"Clutch shots       : {int(df['clutch'].sum()):>10,}  "
        f"({df['clutch'].mean():.1%} of all shots)",
        "",
        "Numeric Predictor Statistics",
        "-" * 60,
        f"  {'Column':<22}  {'Mean':>7}  {'Median':>7}  {'SD':>7}",
        f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}",
    ]
    for col in _CORR_COLS:
        r = stats.loc[col]
        lines.append(
            f"  {col:<22}  {r['Mean']:7.2f}  {r['Median']:7.2f}  {r['SD']:7.2f}"
        )
    lines.append("=" * 60)

    text = "\n".join(lines)
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(text + "\n")
    logger.info("Saved: %s", _REPORT_PATH)
    logger.info("\n%s", text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eda(data_path: Path | None = None) -> None:
    """Generate all EDA figures and the summary report.

    Args:
        data_path: Path to the cleaned CSV. Defaults to
                   config.DATA_PROCESSED_DIR / config.CLEAN_CSV.

    Raises:
        FileNotFoundError: If data_path does not exist.

    Side effects:
        Creates outputs/figures/eda/ and outputs/reports/ if missing.
        Writes five 150-DPI PNG files and one plain-text summary.
    """
    if data_path is None:
        data_path = config.DATA_PROCESSED_DIR / config.CLEAN_CSV

    if not data_path.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found: {data_path}\n"
            "Run `python -m src.clean_data` first."
        )

    logger.info("Loading %s ...", data_path)
    df = pd.read_csv(data_path)
    logger.info(
        "Loaded %d rows, %d players",
        len(df),
        df[config.COL_PLAYER_ID].nunique(),
    )

    _EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme()

    logger.info("Plot 1/5: shot distance distribution ...")
    _plot_shot_dist_hist(df)

    logger.info("Plot 2/5: FG%% by distance bucket ...")
    _plot_fg_pct_by_distance(df)

    logger.info("Plot 3/5: FG%% by player (top/bottom 20) ...")
    _plot_fg_pct_by_player(df, min_shots=config.MIN_SHOTS_PER_PLAYER)

    logger.info("Plot 4/5: defender distance by shot result ...")
    _plot_defender_dist_hist(df)

    logger.info("Plot 5/5: correlation heatmap ...")
    _plot_correlation_heatmap(df)

    logger.info("Writing summary report ...")
    _write_summary(df)

    logger.info("EDA complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config.setup_logging()
    run_eda()
