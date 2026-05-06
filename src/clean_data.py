from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import config

logger = logging.getLogger(__name__)

# Raw-CSV column names to keep before any renaming.
_KEEP_COLS = [
    "SHOT_DIST",
    "CLOSE_DEF_DIST",
    "SHOT_CLOCK",
    "TOUCH_TIME",
    "PERIOD",
    "GAME_CLOCK",
    "FGM",
    "player_name",
    "player_id",
    "LOCATION",
    "SHOT_RESULT",
    "FINAL_MARGIN",
]

# Maps raw-CSV names → clean output names aligned with config constants.
# GAME_CLOCK is parsed into game_clock_sec and then dropped.
_RENAME_MAP = {
    "SHOT_DIST":      config.COL_SHOT_DIST,    # shot_dist
    "CLOSE_DEF_DIST": config.COL_CLOSE_DEF,    # close_def_dist
    "SHOT_CLOCK":     config.COL_SHOT_CLOCK,   # shot_clock
    "TOUCH_TIME":     config.COL_TOUCH_TIME,   # touch_time
    "PERIOD":         config.COL_PERIOD,       # period
    "FGM":            config.COL_SHOT_RESULT,  # shot_result  (binary target)
    "player_name":    config.COL_PLAYER_NAME,  # player_name
    "player_id":      config.COL_PLAYER_ID,    # player_id
    "LOCATION":       "location",
    "SHOT_RESULT":    "shot_result_label",      # "made" / "missed" string, kept for reference
    "FINAL_MARGIN":   "final_margin",
}


# Helpers
def _parse_game_clock(series: pd.Series) -> pd.Series:
    # Convert "MM:SS" strings to total seconds as float.
    split = series.str.split(":", expand=True).astype(float)
    return split[0] * 60 + split[1]


def _log_shape(df: pd.DataFrame, label: str) -> None:
    logger.info("  %-35s  rows=%d", label, len(df))


# Public API
def clean(
    input_path: Path | None = None,
    output_path: Path | None = None,
    min_shots: int | None = None,
) -> pd.DataFrame:
    if input_path is None:
        input_path = config.DATA_RAW_DIR / config.RAW_CSV
    if output_path is None:
        output_path = config.DATA_PROCESSED_DIR / config.CLEAN_CSV
    if min_shots is None:
        min_shots = config.MIN_SHOTS_PER_PLAYER

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found: {input_path}\n"
            "Run `python -m src.download_data` first."
        )

    # 1. Load
    logger.info("Loading %s ...", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded  %d rows, %d columns", len(df), len(df.columns))

    # 2. Keep relevant columns
    missing = [c for c in _KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found in raw CSV: {missing}")
    df = df[_KEEP_COLS].copy()
    _log_shape(df, "after column selection")

    # 3. Drop rows with missing SHOT_CLOCK
    before = len(df)
    df = df.dropna(subset=["SHOT_CLOCK"])
    logger.info("  %-35s  dropped=%d  rows=%d", "drop missing SHOT_CLOCK", before - len(df), len(df))

    # 4. Drop implausible values
    before = len(df)
    mask_bad = (df["TOUCH_TIME"] < 0) | (df["SHOT_DIST"] < 0) | (df["CLOSE_DEF_DIST"] < 0)
    df = df[~mask_bad]
    logger.info("  %-35s  dropped=%d  rows=%d", "drop implausible negatives", before - len(df), len(df))

    # 5. Parse GAME_CLOCK ("MM:SS") → game_clock_sec (float seconds)
    df["game_clock_sec"] = _parse_game_clock(df["GAME_CLOCK"])
    df = df.drop(columns=["GAME_CLOCK"])

    # 6. CLUTCH flag: period >= 4, <= 5 min left, game within 5 points.
    df["clutch"] = (
        (df["PERIOD"] >= 4)
        & (df["game_clock_sec"] <= 300)
        & (df["FINAL_MARGIN"].abs() <= 5)
    ).astype(int)
    logger.info(
        "  %-35s  clutch=%d (%.1f%%)",
        "clutch flag",
        df["clutch"].sum(),
        100 * df["clutch"].mean(),
    )

    # 7. Filter to players with >= min_shots attempts
    shot_counts = df.groupby("player_id")["FGM"].transform("count")
    before = len(df)
    n_players_before = df["player_id"].nunique()
    df = df[shot_counts >= min_shots]
    logger.info(
        "  %-35s  dropped=%d rows, %d players removed  rows=%d",
        f"min {min_shots} shots per player",
        before - len(df),
        n_players_before - df["player_id"].nunique(),
        len(df),
    )

    # 8. Rename to config-aligned names
    df = df.rename(columns=_RENAME_MAP)

    # 9. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        "Saved %s  —  shape=%s  players=%d",
        output_path,
        df.shape,
        df[config.COL_PLAYER_ID].nunique(),
    )
    return df


if __name__ == "__main__":
    config.setup_logging()
    clean()
